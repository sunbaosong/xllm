# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_npu

from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
)
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

# Ascend FIA sparse_mode values (see CANN aclnnFusedInferAttentionScore docs).
# 0: no compressed mask; used for single-query decode where no causal mask is
#    needed.
# 3: rightDownCausal; the causal mask is right-aligned to the KV tail, for the
#    prefix-cache / chunked-prefill case where q_len < kv_len so the new queries
#    attend the full cached prefix plus their own tokens (mode 2, leftUpCausal,
#    only aligns when q_len == kv_len and would misalign on a cache hit).
_SPARSE_MODE_NONE = 0
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3


class NpuPagedAttentionBackend(AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device

        self._kv_caches: list[LayerCache] = []
        self._paged_caches: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._linear_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1)
            .to(torch.int8)
            .contiguous()
            .to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        if not self._kv_caches:
            return 0
        key_cache = self._kv_caches[0].key
        return key_cache.shape[0] if key_cache is not None else 0

    @property
    def page_size(self) -> int:
        if not self._kv_caches:
            return 1
        key_cache = self._kv_caches[0].key
        return key_cache.shape[1] if key_cache is not None else 1

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches
        self._paged_caches = []
        self._linear_caches = []
        for c in kv_caches:
            self._paged_caches.append((c.key, c.value, c.index))
            self._linear_caches.append((c.conv, c.ssm))

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        """KDA delta-rule over framework conv/ssm slots.

        Returns ``[B, S, num_heads_local, head_dim]``. conv1d is pure-torch;
        the delta-rule runs on fla_npu fused ops (``_kda_fla_npu``). The engine
        flattens a multi-sequence batch to ``[1, conv_dim, T]`` with one
        linear-state slot per sequence — split by ``q_cu_seq_lens`` and run
        per sequence. Single-sequence keeps the simple path.
        """
        from xllm.python.kernels_npu.kda import (
            _causal_conv1d_fn,
            _causal_conv1d_update,
        )

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        conv_cache, ssm_cache = self._linear_caches[layer.layer_id]

        batch_size, _, seq_len = mixed_qkv.shape
        conv_kernel_size = layer.conv_kernel_size
        conv_state_len = conv_kernel_size - 1
        head_dim = layer.head_dim
        num_heads_local = layer.num_heads_local
        qkv_dim = layer.qkv_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        idx = metadata.linear_state_indices
        num_seqs = idx.shape[0] if idx is not None else batch_size
        if idx is None:
            device = mixed_qkv.device
            conv_state = torch.zeros(
                batch_size, layer.conv_dim, conv_state_len,
                dtype=layer.conv1d.weight.dtype, device=device,
            )
            ssm_state = torch.zeros(
                batch_size, num_heads_local, head_dim, head_dim,
                dtype=torch.float32, device=device,
            )
        else:
            if idx.dtype != torch.int64:
                idx = idx.to(torch.int64)
            conv_i = conv_cache.index_select(0, idx)
            conv_i = conv_i.transpose(1, 2).contiguous()
            ssm_i = ssm_cache.index_select(0, idx)
            his = metadata.has_initial_state
            if his is not None and len(his) == num_seqs:
                if not isinstance(his, torch.Tensor):
                    his = torch.tensor(
                        his, dtype=torch.int64, device=conv_i.device
                    )
                warm = his.to(torch.bool).view(num_seqs, 1, 1)
                conv_i = torch.where(warm, conv_i, torch.zeros_like(conv_i))
                ssm_i = torch.where(
                    warm.view(num_seqs, 1, 1, 1),
                    ssm_i,
                    torch.zeros_like(ssm_i),
                )
            conv_state, ssm_state = conv_i, ssm_i

        conv_weight = layer.conv1d.weight.squeeze(1)
        activation = layer.activation
        if num_seqs == batch_size:
            if seq_len == 1:
                mixed_qkv = _causal_conv1d_update(
                    mixed_qkv, conv_state, conv_weight, activation
                )
            else:
                conv_in = torch.cat([conv_state, mixed_qkv], dim=-1)
                mixed_qkv = _causal_conv1d_fn(
                    conv_in, conv_weight, activation
                )[:, :, -seq_len:]
                conv_state = conv_in[..., -conv_state_len:]
                if conv_state.shape[-1] < conv_state_len:
                    conv_state = F.pad(
                        conv_state,
                        (conv_state_len - conv_state.shape[-1], 0), value=0,
                    )
        else:
            q_cu = metadata.q_cu_seq_lens
            assert q_cu is not None, (
                "multi-sequence linear attention needs q_cu_seq_lens")
            q_cu = q_cu.tolist()
            outs = []
            for s in range(num_seqs):
                t0, t1 = q_cu[s], q_cu[s + 1]
                seg = mixed_qkv[:, :, t0:t1]  # [1, conv_dim, seg_len]
                cs = conv_state[s:s + 1]      # [1, conv_dim, state_len]
                seg_len = t1 - t0
                if seg_len == 1:
                    outs.append(_causal_conv1d_update(
                        seg, cs, conv_weight, activation
                    ))
                else:
                    cin = torch.cat([cs, seg], dim=-1)
                    outs.append(_causal_conv1d_fn(
                        cin, conv_weight, activation
                    )[:, :, -seg_len:])
                    cs_new = cin[..., -conv_state_len:]
                    if cs_new.shape[-1] < conv_state_len:
                        cs_new = F.pad(
                            cs_new,
                            (conv_state_len - cs_new.shape[-1], 0), value=0,
                        )
                    conv_state[s] = cs_new[0]
            mixed_qkv = torch.cat(outs, dim=-1)
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1
        )
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        if num_seqs == batch_size:
            core_attn_out, final_state = self._kda_fla_npu(
                query, key, value, gate, beta, ssm_state, head_dim,
            )
        else:
            g = gate.view(hidden_shape)
            b = beta.view(batch_size, seq_len, num_heads_local)
            out_parts = []
            final_state = ssm_state.clone()
            q_cu = metadata.q_cu_seq_lens.tolist()
            for s in range(num_seqs):
                t0, t1 = q_cu[s], q_cu[s + 1]
                q_s = query[:, t0:t1]
                k_s = key[:, t0:t1]
                v_s = value[:, t0:t1]
                st = ssm_state[s:s + 1]
                g_s = g[:, t0:t1]
                b_s = b[:, t0:t1]
                co, fs = self._kda_fla_npu(
                    q_s, k_s, v_s, g_s, b_s, st, head_dim,
                )
                out_parts.append(co)
                if fs is not None:
                    final_state[s] = fs
            core_attn_out = torch.cat(out_parts, dim=1)

        if idx is not None:
            conv_cache.index_copy_(
                0, idx, conv_state.transpose(1, 2).contiguous()
            )
            ssm_cache.index_copy_(
                0, idx, final_state.float().contiguous()
            )
        return core_attn_out

    def _kda_fla_npu(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        gate: torch.Tensor, beta: torch.Tensor, ssm_state: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AscendC fused KDA delta-rule for ONE sequence (fla_npu).

        ``query/key/value/gate`` are BSND ``[1, S, heads, dim]``; ``beta`` is
        ``[1, S, heads]``; ``gate`` is the step-wise forget gate (the kernel does
        the cumsum); ``ssm_state`` is ``[1, nh, k_hd, v_hd]`` fp32. Decode (S==1)
        lets the kernel do l2norm; prefill does it here in fp32 -> bf16
        (``chunk_kda_fwd`` has no l2norm switch).
        """
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
        from xllm.python.kernels_npu.kda import _l2norm

        seq_len = query.shape[1]
        scale = 1.0 / (head_dim ** 0.5)

        if seq_len == 1:
            core_attn_out, final_state = recurrent_kda(
                query, key, value, gate, beta,
                initial_state=ssm_state,
                layout="BSND", scale=scale,
                output_final_state=True,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
            )
            return core_attn_out, final_state if final_state is not None else ssm_state

        q_in = _l2norm(query.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
        k_in = _l2norm(key.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
        v_in = value.to(torch.bfloat16).contiguous()
        result = chunk_kda_fwd(
            q_in, k_in, v_in, gate, beta, scale,
            chunk_size=64, layout="BSND",
            initial_state=ssm_state,
            output_final_state=True,
            use_gate_in_kernel=False,
            A_log=None, dt_bias=None,
            return_intermediate_states=False,
        )
        core_attn_out = result[0].to(query.dtype)
        final_state = result[1]
        return core_attn_out, final_state

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        if metadata.q_cu_seq_lens is not None:
            self._actual_seq_lens: list[int] | None = (
                metadata.q_cu_seq_lens[1:].cpu().tolist()
            )
        else:
            self._actual_seq_lens = None

        if metadata.block_table is not None:
            self._block_table_i32 = metadata.block_table.to(torch.int32)

            real_batch = metadata.block_table.shape[0]

            kv_host = metadata.kv_seq_lens_host
            if kv_host is not None:
                kv_host = kv_host.cpu()
                if kv_host.numel() == real_batch + 1:
                    per_seq_kv = kv_host[1:] - kv_host[:-1]
                else:
                    per_seq_kv = kv_host
            else:
                per_seq_kv = torch.ones(real_batch, dtype=torch.int32)

            kv_list = per_seq_kv[:real_batch].tolist()

            self._actual_seq_q: list[int] = list(range(1, real_batch + 1))
            self._actual_seq_kv: list[int] = kv_list
        else:
            self._block_table_i32 = None

        if graph_mode and self._block_table_i32 is not None:
            graph_batch_size = self._block_table_i32.shape[0]
            if self._graph_workspace is None:
                block_size = self.page_size
                dummy_q = torch.empty(
                    graph_batch_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                dummy_kv = torch.empty(
                    self.num_kv_blocks, block_size,
                    self.num_kv_heads * self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                self._graph_workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=self._actual_seq_q,
                        actual_seq_lengths_kv=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=_SPARSE_MODE_NONE,
                        scale=self.scale,
                        softmax_lse_flag=False,
                    )
                )
            if graph_batch_size not in self._graph_outputs:
                self._graph_outputs[graph_batch_size] = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._graph_lses[graph_batch_size] = torch.empty(
                    0, dtype=self.dtype, device=self.device
                )
            self._current_graph_output = self._graph_outputs[graph_batch_size]
            self._current_graph_lse = self._graph_lses[graph_batch_size]

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        if metadata.kv_seq_lens is not None:
            kv_seq_lens = metadata.kv_seq_lens
            mla_device = kv_seq_lens.device
            self._mla_actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if metadata.q_cu_seq_lens is not None:
                self._mla_actual_seq_q = metadata.q_cu_seq_lens[1:].to(
                    torch.int32
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                self._mla_actual_seq_q = torch.arange(
                    1, batch + 1, dtype=torch.int32, device=mla_device
                )
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        k_cache, v_cache = layer_cache.key, layer_cache.value
        if k_cache is None or v_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer_id}")
        num_tokens = q.shape[0]

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        kernels.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        if metadata.is_prefill or metadata.is_chunked_prefill:
            return self._prefill(
                q_3d, k_3d, v_3d, k_cache, v_cache, metadata, num_tokens
            )
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        layer: "Attention",
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        if topk is None:
            raise NotImplementedError(
                "dense MLA (topk=None) is not yet supported on "
                "NpuPagedAttentionBackend"
            )
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        # MLA reuses the K/V slots for the latent (nope) and rope caches.
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer_id}")

        torch.ops.xllm_ops.reshape_paged_cache(
            metadata.slot_mapping, k_latent_3d, k_pe_3d, nope_cache, rope_cache
        )
        return self._mla_sparse(
            q_latent, q_pe, nope_cache, rope_cache, topk, metadata.block_table
        )

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        index_cache = self._kv_caches[layer.layer_id].index
        if index_cache is None:
            raise RuntimeError(
                f"MLA index cache is missing for layer {layer.layer_id}"
            )
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table,
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
        )

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.ops.xllm_ops.sparse_flash_attention(
            q_latent, nope_cache, nope_cache, topk,
            block_table,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
            q_pe, rope_cache, self.scale, 1,
            "TND", "PA_BSND", 3,
        )
        return out  # [T, H, kv_lora]

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self, q_3d: torch.Tensor, k_3d: torch.Tensor, v_3d: torch.Tensor,
        k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        # Prefix-cache hit (or chunked prefill with prior context): part of the
        # KV already lives in the paged cache, so this forward only carries the
        # new tokens (q_len < kv_len). Attend over the full paged KV via
        # block_table, mirroring _decode. Without this, the new query tokens
        # would only see their own KV (actual_seq_lengths_kv == q_len) and never
        # the cached prefix, diverging from a full recompute.
        if metadata.block_table is not None:
            block_size = k_cache.size(1)
            k_flat = k_cache.view(k_cache.size(0), block_size, -1)
            v_flat = v_cache.view(v_cache.size(0), block_size, -1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d, k_flat, v_flat,
                pse_shift=None,
                atten_mask=self._causal_mask,
                block_table=self._block_table_i32,
                actual_seq_lengths=actual_seq,
                actual_seq_lengths_kv=self._actual_seq_kv,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                block_size=block_size,
                sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
                softmax_lse_flag=False,
            )
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_3d, v_3d,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_size: int,
    ) -> None:
        torch.ops.npu.npu_fused_infer_attention_score.out(
            q, k, v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=self._graph_workspace,
            out=[self._current_graph_output, self._current_graph_lse],
        )

    def _decode(
        self, q_3d: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(q_3d, k_flat, v_flat, block_size)
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(q_3d, k_flat, v_flat, block_size)

            graph_context.tasks.append(
                AclGraphTask(event, handle, _update_fia_args)
            )
            return self._current_graph_output.reshape(
                num_tokens, self.num_heads * self.head_dim
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_flat, v_flat,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q[:num_tokens],
            actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cumulative_seq_lens(
        self, metadata: AttentionMetadata, num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
