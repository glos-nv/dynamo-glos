# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load the BFL NVFP4 single-file checkpoint into a diffusers Flux2Transformer2DModel.

diffusers 0.39's from_single_file converter is written for the BF16 checkpoint and
crashes on the quantized tensors (it lets `weight_scale` through its `.weight`
substring filter, then tries to torch.chunk the 0-dim `input_scale`).

We reuse the *same* converter but feed it three separate views of the state dict,
so every quantized tensor gets the identical key remapping and QKV splitting:

  view A  .weight            packed uint8 fp4x2  (or bf16 for unquantized layers)
  view B  .weight_scale      e4m3 per-16 block scales, [N, K/16]
  view C  .weight_scale_2 /  fp32 scalars, broadcast to [3] so the QKV chunk
          .input_scale       works uniformly, then read back element 0

Splitting QKV happens along dim 0 (N), which is safe: FP4 packing is along K, and
the block scales are [N, K/16], so both chunk consistently.

The reassembled weights become torchao NVFP4Tensors, which dispatch F.linear to
the Blackwell FP4 tensor-core path (torch._scaled_mm) with correctly swizzled
scales.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file

BLOCK = 16


def _converter():
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )

    return convert_flux2_transformer_checkpoint_to_diffusers


def load_nvfp4_state(ckpt_path, verbose=True):
    """Return (base_sd, quant) where quant maps diffusers module path -> dict of parts."""
    raw = load_file(ckpt_path)

    with safe_open(ckpt_path, framework="pt") as f:
        meta = f.metadata() or {}
    qmeta = json.loads(meta.get("_quantization_metadata", '{"layers":{}}'))["layers"]
    qlayers = {k for k, v in qmeta.items() if v.get("format") == "nvfp4"}
    if not qlayers or len(qlayers) != len(qmeta):
        raise ValueError(
            "Klein checkpoint must declare nonempty, exclusively NVFP4 quantization metadata"
        )
    for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
        sources = {
            key.removesuffix(f".{suffix}") for key in raw if key.endswith(f".{suffix}")
        }
        if sources != qlayers:
            raise ValueError(
                f"NVFP4 metadata/{suffix} mismatch: missing={sorted(qlayers - sources)}, "
                f"unexpected={sorted(sources - qlayers)}"
            )
    if verbose:
        print(
            f"[nvfp4] {len(qlayers)} quantized layers declared in checkpoint metadata"
        )

    view_a, view_b, view_c_w2, view_c_in = {}, {}, {}, {}
    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            view_b[k[: -len(".weight_scale")] + ".weight"] = v
        elif k.endswith(".weight_scale_2"):
            view_c_w2[k[: -len(".weight_scale_2")] + ".weight"] = v.reshape(1).repeat(3)
        elif k.endswith(".input_scale"):
            view_c_in[k[: -len(".input_scale")] + ".weight"] = v.reshape(1).repeat(3)
        else:
            view_a[k] = v

    conv = _converter()
    a = conv(dict(view_a))
    b = conv(dict(view_b))
    c2 = conv(dict(view_c_w2))
    ci = conv(dict(view_c_in))
    if not b or b.keys() != c2.keys() or b.keys() != ci.keys():
        raise ValueError(
            "Converted NVFP4 block, weight and activation scale targets do not match"
        )

    if verbose:
        print(
            f"[nvfp4] converted: base={len(a)} wscale={len(b)} wscale2={len(c2)} inscale={len(ci)}"
        )

    quant = {}
    for wk, scale in b.items():
        mod = wk[: -len(".weight")]
        qdata = a.pop(wk, None)
        if qdata is None:
            raise KeyError(f"no packed weight for quantized module {mod}")
        if qdata.dtype != torch.uint8:
            raise TypeError(f"{mod}: expected uint8 packed fp4, got {qdata.dtype}")
        if scale.dtype != torch.float8_e4m3fn or scale.shape != (
            qdata.shape[0],
            qdata.shape[1] // 8,
        ):
            raise ValueError(
                f"{mod}: expected E4M3 scales for packed NVFP4 blocks of 16"
            )
        quant[mod] = {
            "qdata": qdata,
            "scale": scale,
            "weight_scale_2": c2[wk].reshape(-1)[0].clone(),
            "input_scale": ci[wk].reshape(-1)[0].clone(),
        }
    return a, quant


def _restore_nvfp4_weight(qdata, scale, per_tensor_scale, act_per_tensor_scale):
    """Restore calibrated NVFP4 weights with Triton activation quantization."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
    )

    return NVFP4Tensor(
        qdata,
        scale,
        BLOCK,
        torch.bfloat16,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=True,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            block_size=BLOCK,
            is_swizzled_scales=True,
            # Fuse activation quantization and scale swizzling for each linear.
            use_triton_kernel=True,
            use_dynamic_per_tensor_scale=False,
        ),
    )


# Dynamo sees plain tensors and an opaque operator, so strict export never
# traces the TorchAO constructor or its activation configuration object. Native
# quantization and GEMM stay in TorchAO, at the cost of fusion across this call.
# Import this module before loading a PT2 package to register its runtime kernel.
@torch.library.custom_op(
    "dynamo_flux::nvfp4_linear", mutates_args=(), device_types="cuda"
)
def _nvfp4_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    act_per_tensor_scale: torch.Tensor,
) -> torch.Tensor:
    weight = _restore_nvfp4_weight(qdata, scale, per_tensor_scale, act_per_tensor_scale)
    return torch.nn.functional.linear(x, weight)


@_nvfp4_linear.register_fake
def _nvfp4_linear_fake(x, qdata, scale, per_tensor_scale, act_per_tensor_scale):
    return x.new_empty((x.shape[0], qdata.shape[0]))


class NVFP4Linear(torch.nn.Module):
    """Bias-free linear backed by a torchao NVFP4Tensor.

    torchao's F.linear dispatch has an edge case on some 3-D inputs (it falls
    through to aten and then hits an unimplemented aten.expand on the subclass).
    Flattening to 2-D first is what torchao's own nvfp4_linear does internally,
    so this is free and always takes the FP4 tensor-core path.
    """

    def __init__(self, weight, in_features, out_features):
        super().__init__()
        if weight.per_tensor_scale is None or weight.act_per_tensor_scale is None:
            raise ValueError(
                "NVFP4Linear requires the saved weight and activation global scales"
            )
        # Export may retain references while AITune offloads the source module.
        # Store plain parameters so offload never swaps a tensor subclass.
        self.qdata = torch.nn.Parameter(weight.qdata, requires_grad=False)
        self.scale = torch.nn.Parameter(weight.scale, requires_grad=False)
        self.per_tensor_scale = torch.nn.Parameter(
            weight.per_tensor_scale, requires_grad=False
        )
        self.act_per_tensor_scale = torch.nn.Parameter(
            weight.act_per_tensor_scale, requires_grad=False
        )
        restored_weight = self.weight
        if (
            type(weight) is not type(restored_weight)
            or weight.dtype != torch.bfloat16
            or weight.__tensor_flatten__() != restored_weight.__tensor_flatten__()
            or weight.shape != (out_features, in_features)
        ):
            raise ValueError(
                "NVFP4Linear requires the calibrated Klein NVFP4 tensor configuration"
            )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        shp = x.shape
        y = _nvfp4_linear(
            x.reshape(-1, shp[-1]),
            self.qdata,
            self.scale,
            self.per_tensor_scale,
            self.act_per_tensor_scale,
        )
        return y.view(*shp[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4=True"

    @property
    def weight(self):
        """Reconstruct the native weight for inspection, outside compiled forward."""
        return _restore_nvfp4_weight(
            self.qdata, self.scale, self.per_tensor_scale, self.act_per_tensor_scale
        )


def _set_module(root, path, new):
    parent = root
    parts = path.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def build_nvfp4_transformer(
    ckpt_path, config_repo, subfolder="transformer", device="cuda", verbose=True
):
    """Load packed NVFP4 weights without allocating dense transformer weights.

    TorchAO 0.16 uses ``is_swizzled_scales``. Plain tensor parameters and the
    custom operator keep its tensor subclass outside export and CPU offload.
    Native execution requires Blackwell; loading/export alone do not prove it.
    """
    from diffusers import Flux2Transformer2DModel

    base_sd, quant = load_nvfp4_state(ckpt_path, verbose=verbose)

    config = Flux2Transformer2DModel.load_config(config_repo, subfolder=subfolder)
    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(config)
    # Set dense placeholder dtype before installing quantized layers. Assign the
    # checkpoint directly; never allocate a dense copy of the packed transformer.
    model = model.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(base_sd, strict=False, assign=True)
    missing = sorted(set(missing) - {f"{path}.weight" for path in quant})
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint/config mismatch: missing={missing}, unexpected={unexpected}"
        )

    mods = dict(model.named_modules())
    installed = 0
    for path, parts in quant.items():
        m = mods.get(path)
        if m is None:
            raise KeyError(f"module {path} not found in diffusers model")
        out_f, in_f_packed = parts["qdata"].shape
        if not isinstance(m, torch.nn.Linear):
            raise TypeError(f"{path}: expected a linear layer")
        if m.bias is not None:
            raise RuntimeError(f"{path} unexpectedly has a bias")
        if m.in_features != in_f_packed * 2:
            raise ValueError(
                f"{path}: in_features {m.in_features} != {in_f_packed * 2}"
            )
        if m.out_features != out_f:
            raise ValueError(f"{path}: out_features {m.out_features} != {out_f}")
        # ModelOpt packs the two FP4 values in each byte in the opposite order
        # from torchao's unpack, and stores the e4m3 block scales already in the
        # swizzled (128x4 tiled) layout. Preserve both checkpoint conventions.
        q = parts["qdata"].to(device)
        q = ((q & 0x0F) << 4) | ((q >> 4) & 0x0F)
        # The activation global scale is carried on the tensor itself
        # (act_per_tensor_scale); the kwargs object only holds layout flags.
        # ModelOpt's `input_scale` is amax/(FP4_MAX*E4M3_MAX), the same
        # convention torchao's per_tensor_amax_to_scale produces.
        t = _restore_nvfp4_weight(
            q.contiguous(),
            parts["scale"].to(device).contiguous(),
            parts["weight_scale_2"].to(device),
            parts["input_scale"].to(device),
        )
        _set_module(model, path, NVFP4Linear(t, m.in_features, m.out_features))
        installed += 1

    if verbose:
        print(f"[nvfp4] installed {installed} NVFP4 linears")
    model.eval()
    return model
