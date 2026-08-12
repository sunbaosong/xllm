# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-torch conv1d helpers for KDA (Kimi Delta Attention) layers.

The delta-rule runs on fla_npu fused ops (``chunk_kda_fwd`` / ``recurrent_kda``);
only conv1d stays pure-torch here, in dim-last ``[B, S, C]`` layout so the conv
state cache keeps its native ``[N, state_len, C]`` shape (no transpose on
read/write). No model coupling and no fla_npu runtime dependency — import-safe
everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: sqrt(sum(x^2)+eps) then divide (NOT F.normalize)."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def _causal_conv1d_fn(
    x: torch.Tensor, weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Depthwise causal conv1d (left-pad K-1) + activation on dim-last layout.

    ``x`` is ``[B, S, C]``; ``weight`` is ``[C, K]``. Avoids F.conv1d so the
    conv cache stays in its native dim-last ``[N, state_len, C]`` layout.
    """
    kernel_size = weight.shape[-1]
    pad = kernel_size - 1
    orig_dtype = x.dtype
    x = x.to(weight.dtype)
    xpad = F.pad(x, (0, 0, pad, 0))          # left-pad the S dimension
    win = xpad.unfold(1, kernel_size, 1)      # [B, S, C, K]
    out = (win * weight).sum(-1)             # [B, S, C]
    if activation == "silu":
        out = F.silu(out)
    return out.to(orig_dtype)


def _causal_conv1d_update(
    seg: torch.Tensor, conv_state: torch.Tensor,
    weight: torch.Tensor, activation: str = "silu",
) -> torch.Tensor:
    """Incremental depthwise causal conv1d + activation (single-token decode).

    Dim-last: ``seg`` is ``[B, 1, C]``; ``conv_state`` is ``[B, K-1, C]``;
    ``weight`` is ``[C, K]``. ``conv_state`` is updated in place.
    """
    kernel_size = weight.shape[-1]
    state_len = conv_state.shape[1]
    orig_dtype = seg.dtype
    cin = torch.cat([conv_state, seg.to(weight.dtype)], dim=1)   # [B, K, C]
    win = cin.unfold(1, kernel_size, 1)                           # [B, 1, C, K]
    out = (win * weight).sum(-1)                                 # [B, 1, C]
    conv_state.copy_(cin[:, -state_len:, :])
    if activation == "silu":
        out = F.silu(out)
    return out.to(orig_dtype)
