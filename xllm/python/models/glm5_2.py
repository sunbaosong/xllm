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

"""GLM-5.2 (model_type=glm_moe_dsa) causal LM, adapted from DeepSeek-V3.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # type: ignore[assignment]

from xllm.python import ops
from xllm.python.attention.backend import MlaIndexContext
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
    RowParallelLinear,
)
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekV3MLP as Glm52MLP,
    DeepseekV3MoE as Glm52MoE,
    DeepseekYarnRotaryEmbedding as Glm52YarnRotaryEmbedding,
    W8A8DynamicLinear,
    W8A8StaticLinear,
    _tp_rank_from_device,
    _yarn_get_mscale,
)


def _apply_interleave_rope(
    rotary: nn.Module, x: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE via ``npu_interleave_rope`` for ``[T, H, D]`` tensors."""
    cos_sin = rotary.cos_sin_cache[positions]
    half = cos_sin.size(-1) // 2
    cos32, sin32 = cos_sin[..., :half], cos_sin[..., half:]
    cos = torch.cat([cos32, cos32], dim=-1).unsqueeze(1).unsqueeze(1)
    sin = torch.cat([sin32, sin32], dim=-1).unsqueeze(1).unsqueeze(1)
    t, h, d = x.shape
    return torch_npu.npu_interleave_rope(
        x.view(t, h, 1, d), cos, sin
    ).view(t, h, d)


@dataclass
class Glm52Config:
    """GLM-5.2 (glm_moe_dsa) architecture parameters."""

    model_type: str = "glm_moe_dsa"
    hidden_size: int = 6144
    n_layers: int = 78
    n_heads: int = 64
    head_dim: int = 0
    intermediate_size: int = 12288
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1.0e6
    max_position_embeddings: int = 202752
    original_max_position_embeddings: int = 202752
    rope_scaling_factor: float = 1.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    tie_word_embeddings: bool = False
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 2048
    tp_size: int = 1
    tp_rank: int = 0
    indexer_types: Optional[list] = None
    mlp_layer_types: Optional[list] = None
    index_skip_topk_offset: int = 2
    index_topk_freq: int = 1
    index_topk_pattern: Optional[list] = None
    indexer_rope_interleave: bool = True
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Glm52Config":
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        rs_raw = d.get("rope_scaling")
        rs = rs_raw if isinstance(rs_raw, dict) else {}
        if not rs:
            rp = d.get("rope_parameters")
            if isinstance(rp, dict):
                rs = rp

        def rpick(*keys, default=None):
            for k in keys:
                if isinstance(rs, dict) and k in rs and rs[k] is not None:
                    return rs[k]
                fk = f"rope_scaling_{k}"
                if fk in d and d[fk] is not None:
                    return d[fk]
                if k in d and d[k] is not None:
                    return d[k]
            return default

        def rpick_nz(*keys, default):
            v = rpick(*keys, default=None)
            if v is None or v == 0 or v == -1 or v == "":
                return default
            return v

        hidden = int(pick("hidden_size", default=6144))
        n_heads = int(pick("n_heads", "num_attention_heads", default=64))
        max_pe = int(pick("max_position_embeddings", default=202752))
        rope_scaling_factor = float(rpick_nz("factor", "rope_scaling_factor", default=1.0))
        original_max = int(rpick_nz("original_max_position_embeddings", default=max_pe))

        cfg = cls(
            model_type=str(pick("model_type", default="glm_moe_dsa")),
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=78)),
            n_heads=n_heads,
            head_dim=int(pick("head_dim", default=hidden // n_heads if n_heads else 0)),
            intermediate_size=int(pick("intermediate_size", default=12288)),
            vocab_size=int(pick("vocab_size", default=154880)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=1.0e6)),
            max_position_embeddings=max_pe,
            original_max_position_embeddings=original_max,
            rope_scaling_factor=rope_scaling_factor,
            rope_beta_fast=int(rpick_nz("beta_fast", default=32)),
            rope_beta_slow=int(rpick_nz("beta_slow", default=1)),
            rope_mscale=float(rpick_nz("mscale", default=1.0)),
            rope_mscale_all_dim=float(rpick_nz("mscale_all_dim", default=1.0)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=2048)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            index_n_heads=int(pick("index_n_heads", default=32)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=192)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            qk_head_dim=int(pick("qk_head_dim", default=256)),
            v_head_dim=int(pick("v_head_dim", default=256)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=3)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            n_routed_experts=int(
                pick("n_routed_experts", "num_local_experts", "num_experts", default=256)
            ),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(
                pick("moe_intermediate_size", default=2048)
            ),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            indexer_types=pick("indexer_types", default=None),
            mlp_layer_types=pick("mlp_layer_types", default=None),
            index_skip_topk_offset=int(pick("index_skip_topk_offset", default=2)),
            index_topk_freq=int(pick("index_topk_freq", default=1)),
            index_topk_pattern=pick("index_topk_pattern", default=None),
            indexer_rope_interleave=bool(pick("indexer_rope_interleave", default=True)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            index_share_for_mtp_iteration=bool(
                pick("index_share_for_mtp_iteration", default=False)
            ),
        )
        cfg._resolve_indexer_types()
        cfg._resolve_mlp_layer_types()
        return cfg

    def _resolve_indexer_types(self) -> None:
        """Derive per-layer indexer mode (full/shared)."""
        if self.indexer_types is not None:
            return
        pattern = self.index_topk_pattern
        if pattern:
            if isinstance(pattern, str):
                self.indexer_types = [
                    {"F": "full", "S": "shared"}[c] for c in pattern
                ]
            else:
                self.indexer_types = list(pattern)
            return
        freq = max(self.index_topk_freq, 1)
        offset = self.index_skip_topk_offset
        self.indexer_types = [
            "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
            for i in range(self.n_layers)
        ]

    def _resolve_mlp_layer_types(self) -> None:
        """Derive per-layer MLP mode (dense/sparse)."""
        if self.mlp_layer_types is not None:
            return
        n_dense = min(self.first_k_dense_replace, self.n_layers)
        self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (
            self.n_layers - n_dense
        )

    def head_split(self) -> Tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1)."""
        num_heads_local = self.n_heads // self.tp_size
        return num_heads_local, 1


class Glm52MLAAttention(Attention):
    """Absorbed-MLA attention with per-layer full/shared DSA indexer."""

    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        assert cfg.n_heads % tp == 0
        num_heads = cfg.n_heads // tp
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v_head = cfg.v_head_dim
        scale = (qk_nope + qk_rope) ** -0.5
        attn_mscale = _yarn_get_mscale(
            cfg.rope_scaling_factor, cfg.rope_mscale_all_dim
        )
        scale = scale * attn_mscale * attn_mscale
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=kv_lora,
            scale=scale,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.qk_nope_head_dim = qk_nope
        self.qk_rope_head_dim = qk_rope
        self.v_head_dim = v_head
        self.kv_lora_rank = kv_lora
        self.num_heads_local = num_heads

        self.q_a_proj = W8A8StaticLinear(cfg.hidden_size, cfg.q_lora_rank, device)
        self.kv_a_proj_with_mqa = W8A8StaticLinear(cfg.hidden_size, kv_lora + qk_rope, device)
        self.q_a_layernorm = RMSNorm(
            cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.q_b_proj = W8A8StaticLinear(
            cfg.q_lora_rank, num_heads * (qk_nope + qk_rope), device
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora,
            num_heads * (qk_nope + v_head),
            tp,
            dtype=dtype,
            device=device,
        )
        self.o_proj = W8A8StaticLinear(num_heads * v_head, cfg.hidden_size, device,
                                       row_parallel=True)
        self.rotary = Glm52YarnRotaryEmbedding(
            qk_rope,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "W_UK",
            torch.empty(num_heads, qk_nope, kv_lora, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(num_heads, kv_lora, v_head, dtype=dtype, device=device),
            persistent=False,
        )
        self.is_shared = (
            cfg.indexer_types is not None
            and layer_id < len(cfg.indexer_types)
            and cfg.indexer_types[layer_id] == "shared"
        )
        self.indexer = None
        if not self.is_shared:
            self.indexer = Glm52Indexer(cfg, dtype, device)
            self.indexer.rotary = self.rotary

    def process_weights_after_loading(self) -> None:
        self.q_a_proj.process_weights_after_loading()
        self.kv_a_proj_with_mqa.process_weights_after_loading()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        if self.indexer is not None:
            self.indexer.process_weights_after_loading()

    def _interleaved_rope(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return _apply_interleave_rope(self.rotary, x, positions)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden.shape[0]
        q_a = self.q_a_proj(hidden)
        q_c = self.q_a_layernorm(q_a)
        backend = get_forward_context().attention_backend
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            topk = self.indexer.select_qli(hidden, q_c, positions, ctx)
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer (prev_topk_indices is None)."
                )
            topk = prev_topk_indices
        q = self.q_b_proj(q_c)
        q = q.view(
            num_tokens,
            self.num_heads_local,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_latent = torch.bmm(
            q_nope.transpose(0, 1), self.W_UK
        ).transpose(0, 1)
        q_pe = self._interleaved_rope(q_rope, positions)
        kv = self.kv_a_proj_with_mqa(hidden)
        k_latent_raw, k_rope_raw = kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_pe = self._interleaved_rope(
            k_rope_raw.unsqueeze(1), positions
        )
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe_3d = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)

        attn_out = backend.execute_mla(
            q_latent, q_pe, k_latent_3d, k_pe_3d, self, topk=topk
        )
        v_full = torch.bmm(
            attn_out.transpose(0, 1), self.W_UV
        ).transpose(0, 1)
        v_full = v_full.reshape(
            num_tokens, self.num_heads_local * self.v_head_dim
        )
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            ops.all_reduce_(o)
        return o, topk


class Glm52Indexer(nn.Module):
    """GLM-5.2 DSA lightning indexer (wq_b W8A8, weights_proj fp32, interleave RoPE)."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.wq_b = W8A8StaticLinear(cfg.q_lora_rank, self.n_head * self.head_dim,
                                     device)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim,
                            bias=False, dtype=dtype, device=device)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head,
                                      bias=False, dtype=torch.float32,
                                      device=device)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6,
                                   dtype=dtype, device=device)
        self.rotary = None

    def process_weights_after_loading(self) -> None:
        self.wq_b.process_weights_after_loading()

    def select_qli(
        self, hidden, qr, positions, ctx: MlaIndexContext
    ) -> torch.Tensor:
        index_cache = ctx.index_cache
        slot_mapping = ctx.slot_mapping
        actual_seq_q = ctx.actual_seq_q
        actual_seq_kv = ctx.actual_seq_kv
        block_table = ctx.block_table
        q = self.wq_b(qr).view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k = self.wk(hidden)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        q_pe = self._interleave_rope(q_pe, positions)
        k_pe = self._interleave_rope(k_pe.unsqueeze(1), positions).squeeze(1)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        if index_cache is not None and slot_mapping is not None:
            k_view = index_cache.view(-1, index_cache.size(-1))
            ops.scatter_nd_update(
                k_view, slot_mapping.reshape(-1, 1).clamp_min(0), k
            )
        weights = self.weights_proj(hidden.to(torch.float32)).to(torch.bfloat16)
        topk = ops.lightning_indexer(
            q, index_cache, weights,
            actual_seq_q, actual_seq_kv, block_table,
            "TND", "PA_BSND", self.topk, 3,
            9223372036854775807, 9223372036854775807,
            False,
        )
        return topk

    def _interleave_rope(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return _apply_interleave_rope(self.rotary, x, positions)


class Glm52DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.self_attn = Glm52MLAAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        mlp_type = (
            cfg.mlp_layer_types[layer_id]
            if cfg.mlp_layer_types is not None
            and layer_id < len(cfg.mlp_layer_types)
            else ("dense" if layer_id < cfg.first_k_dense_replace else "sparse")
        )
        if mlp_type == "dense":
            self.mlp = Glm52MLP(
                cfg, cfg.intermediate_size, dtype, device
            )
        else:
            self.mlp = Glm52MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, topk_indices = self.self_attn(hidden, positions, prev_topk_indices)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual, topk_indices


class Glm52Model(nn.Module):
    def __init__(
        self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                Glm52DecoderLayer(cfg, i, dtype, device)
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        residual: Optional[torch.Tensor] = None
        prev_topk: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden, residual, prev_topk = layer(
                hidden, residual, positions, prev_topk
            )
        hidden, last_hidden = self.norm(hidden, residual)
        return hidden


class Glm52ForCausalLM(PyModelBase):
    """GLM-5.2 causal LM. Registered under ``model_type='glm_moe_dsa'``."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Glm52Config.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get(
            "tp_rank", _tp_rank_from_device(config.get("device", "npu:0"))))
        dtype = self.resolve_dtype(
            config.get("dtype") or config.get("torch_dtype")
        )
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model = Glm52Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg
        tp_size = cfg.tp_size
        tp_rank = cfg.tp_rank

        params_by_name = dict(self.named_parameters())
        buffers_by_name = dict(self.named_buffers())

        def find(name: str):
            for sd in state_dicts:
                if sd.has(name):
                    return sd
            return None

        def load_tensor(name: str) -> torch.Tensor:
            sd = find(name)
            assert sd is not None, f"checkpoint tensor not found: {name}"
            return sd.get_tensor(name)

        def shard(t: torch.Tensor, dim: int, world: int = tp_size, rank: int = tp_rank) -> torch.Tensor:
            if world <= 1:
                return t
            cs = t.size(dim) // world
            return t.narrow(dim, rank * cs, cs).contiguous()

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            p = params_by_name.get(param_name)
            if p is None:
                p = buffers_by_name.get(param_name)
            assert p is not None, f"no parameter/buffer named {param_name}"
            p.data.copy_(tensor.to(dtype=p.dtype, device=p.device))

        def load_w8a8_a(prefix: str, proj: str, shard_dims=None) -> None:
            for suffix in ("weight", "deq_scale", "quant_bias",
                           "input_scale", "input_offset"):
                t = load_tensor(prefix + proj + "." + suffix)
                dim = (shard_dims or {}).get(suffix)
                if dim is not None:
                    t = shard(t, dim=dim)
                copy_in(prefix + proj + "." + suffix, t)

        def load_w8a8_b(mlp_pfx: str) -> None:
            gw = load_tensor(mlp_pfx + "gate_proj.weight")
            gs = load_tensor(mlp_pfx + "gate_proj.weight_scale")
            go = load_tensor(mlp_pfx + "gate_proj.weight_offset")
            uw = load_tensor(mlp_pfx + "up_proj.weight")
            us = load_tensor(mlp_pfx + "up_proj.weight_scale")
            uo = load_tensor(mlp_pfx + "up_proj.weight_offset")
            copy_in(mlp_pfx + "gate_up_proj.weight",
                    torch.cat([shard(gw, 0), shard(uw, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "gate_up_proj.weight_scale",
                    torch.cat([shard(gs, 0), shard(us, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "gate_up_proj.weight_offset",
                    torch.cat([shard(go, 0), shard(uo, 0)], dim=0).contiguous())
            copy_in(mlp_pfx + "down_proj.weight",
                    shard(load_tensor(mlp_pfx + "down_proj.weight"), dim=1))
            copy_in(mlp_pfx + "down_proj.weight_scale",
                    load_tensor(mlp_pfx + "down_proj.weight_scale"))
            copy_in(mlp_pfx + "down_proj.weight_offset",
                    load_tensor(mlp_pfx + "down_proj.weight_offset"))

        def layer_is_shared(i: int) -> bool:
            return (
                cfg.indexer_types is not None
                and i < len(cfg.indexer_types)
                and cfg.indexer_types[i] == "shared"
            )

        def layer_mlp_type(i: int) -> str:
            if cfg.mlp_layer_types is not None and i < len(cfg.mlp_layer_types):
                return cfg.mlp_layer_types[i]
            return "dense" if i < cfg.first_k_dense_replace else "sparse"

        copy_in("model.embed_tokens.weight",
                shard(load_tensor("model.embed_tokens.weight"), dim=1))

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            copy_in(p + "input_layernorm.weight",
                    load_tensor(p + "input_layernorm.weight"))
            copy_in(p + "post_attention_layernorm.weight",
                    load_tensor(p + "post_attention_layernorm.weight"))
            attn = p + "self_attn."
            load_w8a8_a(attn, "q_a_proj")
            copy_in(attn + "q_a_layernorm.weight",
                    load_tensor(attn + "q_a_layernorm.weight"))
            load_w8a8_a(attn, "q_b_proj",
                        {"weight": 0, "deq_scale": 0, "quant_bias": 0})
            load_w8a8_a(attn, "kv_a_proj_with_mqa")
            copy_in(attn + "kv_a_layernorm.weight",
                    load_tensor(attn + "kv_a_layernorm.weight"))
            copy_in(attn + "kv_b_proj.weight",
                    shard(load_tensor(attn + "kv_b_proj.weight"), dim=0))
            load_w8a8_a(attn, "o_proj", {"weight": 1})
            if not layer_is_shared(i):
                idx = attn + "indexer."
                load_w8a8_a(idx, "wq_b")
                copy_in(idx + "wk.weight", load_tensor(idx + "wk.weight"))
                copy_in(idx + "k_norm.weight",
                        load_tensor(idx + "k_norm.weight"))
                copy_in(idx + "k_norm.bias", load_tensor(idx + "k_norm.bias"))
                copy_in(idx + "weights_proj.weight",
                        load_tensor(idx + "weights_proj.weight"))
            self.model.layers[i].self_attn.process_weights_after_loading()

            mlp_type = layer_mlp_type(i)
            if mlp_type == "dense":
                load_w8a8_b(p + "mlp.")
                self.model.layers[i].mlp.process_weights_after_loading()
            else:
                se = p + "mlp.experts."
                w13_param = self.get_parameter(p + "mlp.experts_w13")
                w2_param = self.get_parameter(p + "mlp.experts_w2")
                w13_scale = self.get_buffer(p + "mlp.experts_w13_scale")
                w13_offset = self.get_buffer(p + "mlp.experts_w13_offset")
                w2_scale = self.get_buffer(p + "mlp.experts_w2_scale")
                w2_offset = self.get_buffer(p + "mlp.experts_w2_offset")
                for j in range(cfg.n_routed_experts):
                    gw = load_tensor(se + f"{j}.gate_proj.weight")
                    gs = load_tensor(se + f"{j}.gate_proj.weight_scale")
                    go = load_tensor(se + f"{j}.gate_proj.weight_offset")
                    uw = load_tensor(se + f"{j}.up_proj.weight")
                    us = load_tensor(se + f"{j}.up_proj.weight_scale")
                    uo = load_tensor(se + f"{j}.up_proj.weight_offset")
                    dw = load_tensor(se + f"{j}.down_proj.weight")
                    ds = load_tensor(se + f"{j}.down_proj.weight_scale")
                    do = load_tensor(se + f"{j}.down_proj.weight_offset")
                    w13_param.data[j].copy_(
                        torch.cat([shard(gw, 0), shard(uw, 0)], dim=0).contiguous())
                    w13_scale.data[j].copy_(
                        torch.cat([shard(gs, 0), shard(us, 0)], dim=0).contiguous())
                    w13_offset.data[j].copy_(
                        torch.cat([shard(go, 0), shard(uo, 0)], dim=0).contiguous())
                    w2_param.data[j].copy_(shard(dw, 1).contiguous())
                    w2_scale.data[j].copy_(ds.contiguous())
                    w2_offset.data[j].copy_(do.contiguous())
                copy_in(p + "mlp.gate.weight", load_tensor(p + "mlp.gate.weight"))
                copy_in(p + "mlp.e_score_correction_bias",
                        load_tensor(p + "mlp.gate.e_score_correction_bias"))
                load_w8a8_b(p + "mlp.shared_experts.")
                self.model.layers[i].mlp.process_weights_after_loading()

        copy_in("model.norm.weight", load_tensor("model.norm.weight"))
        copy_in("lm_head.weight", shard(load_tensor("lm_head.weight"), dim=0))
