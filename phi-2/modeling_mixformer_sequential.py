# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# BSD 3-Clause License
# 
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mixformer_sequential import MixFormerSequentialConfig


try:
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    from flash_attn.ops.fused_dense import FusedDense
except:
    FlashRotaryEmbedding = None
    FusedDense = None


@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.

    Args:
        max_seqlen: Maximum sequence length.
        max_batch_size: Maximum batch size.
        seqlen_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        lengths_per_sample: Lengths per sample.

    """

    max_seqlen: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    seqlen_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})


class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        return hidden_states


def _apply_rotary_emb(
    x: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
) -> torch.FloatTensor:
    _, seqlen, _, head_dim = x.shape
    rotary_seqlen, rotary_dim = cos.shape
    rotary_dim *= 2

    assert rotary_dim <= head_dim
    assert seqlen <= rotary_seqlen
    assert cos.shape == sin.shape == (rotary_seqlen, rotary_dim // 2)

    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    x1, x2 = x_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    x1, x2, c, s = [t.to(dtype=torch.float32) for t in [x1, x2, c, s]]

    x_rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).to(x.dtype)

    return torch.cat([x_rot, x_pass], axis=-1)


def _apply_rotary_emb_kv(
    kv: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    cos_k: Optional[torch.FloatTensor] = None,
    sin_k: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    _, seqlen, two, _, head_dim = kv.shape
    assert two == 2

    rotary_seqlen, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= head_dim
    assert seqlen <= rotary_seqlen
    assert cos.shape == sin.shape == (rotary_seqlen, rotary_dim // 2)

    k_rot = kv[:, :, 0, :, :rotary_dim]
    k_pass = kv[:, :, 0, :, rotary_dim:]

    k1, k2 = k_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    k1, k2, c, s = [t.to(dtype=torch.float32) for t in [k1, k2, c, s]]

    k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(kv.dtype)

    return torch.cat(
        [
            torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
            kv[:, :, 1:2, :, :],
        ],
        axis=2,
    )


def _apply_rotary_emb_qkv(
    qkv: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    cos_k: Optional[torch.FloatTensor] = None,
    sin_k: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    _, seqlen, three, _, head_dim = qkv.shape
    assert three == 3

    rotary_seqlen, rotary_dim = cos.shape
    rotary_dim *= 2
    assert rotary_dim <= head_dim
    assert seqlen <= rotary_seqlen
    assert cos.shape == sin.shape == (rotary_seqlen, rotary_dim // 2)

    q_rot = qkv[:, :, 0, :, :rotary_dim]
    q_pass = qkv[:, :, 0, :, rotary_dim:]

    k_rot = qkv[:, :, 1, :, :rotary_dim]
    k_pass = qkv[:, :, 1, :, rotary_dim:]

    q1, q2 = q_rot.chunk(2, dim=-1)
    k1, k2 = k_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    q1, q2, k1, k2, c, s = [t.to(dtype=torch.float32) for t in [q1, q2, k1, k2, c, s]]

    q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).to(qkv.dtype)
    k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(qkv.dtype)

    return torch.cat(
        [
            torch.cat([q_rot, q_pass], axis=-1).unsqueeze(2),
            torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
            qkv[:, :, 2:3, :, :],
        ],
        axis=2,
    )


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://arxiv.org/pdf/2104.09864.pdf.

    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if scale_base is not None:
            raise NotImplementedError

        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.device = device

        # Generate and save the inverse frequency buffer (non-trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Generate and save the scale buffer (non-trainable)
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device: Optional[str] = None) -> torch.FloatTensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(
        self, seqlen: int, device: Optional[str] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        # Reset the tables if sequence length has been chaned, if we are on a
        # new device or if we are switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            # fp32 is preferred since the output of `torch.arange` can be quite large
            # and bf16 would lose a lot of precision
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            # `torch.outer` is preferred since `torch.einsum` converts from fp32 to fp16 if used with AMP
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

                # Force the scale multiplication to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: int = 0,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqlen = qkv.shape[1]

        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        else:
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)

        if kv is None:
            return _apply_rotary_emb_qkv(qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
        else:
            q = _apply_rotary_emb(qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
            kv = _apply_rotary_emb_kv(kv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])

            return q, kv


class MLP(nn.Module):
    """Multi-Layer Perceptron.

    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.

    """

    def __init__(self, config: PretrainedConfig, n_inner: Optional[int] = None, act_fn: Optional[str] = None) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        assert act_fn in ACT2FN.keys(), f"`act_fn` must be one of: {ACT2FN.keys()}."

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[act_fn]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SelfAttention(nn.Module):
    """Self-attention layer (compatible with PyTorch).
    
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.

    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        qkv: torch.FloatTensor,
        causal: bool = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if attention_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(attention_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


class CrossAttention(nn.Module):
    """Cross-attention layer (compatible with PyTorch).
    
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.
    
    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        causal: bool = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if attention_mask is not None:
            padding_mask = torch.full((batch_size, seqlen_k), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(attention_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            rows = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
            cols = torch.arange(seqlen_k, device=k.device, dtype=torch.long)
            causal_mask = cols > rows + seqlen_k - seqlen_q

            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


def _find_mha_dims(
    config: PretrainedConfig,
    n_head: Optional[int] = None,
    n_head_kv: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Tuple[int, int]:
    assert all(
        hasattr(config, attr) for attr in ["n_embd", "n_head"]
    ), "`config` must have `n_embd` and `n_head` attributes."

    if head_dim is None:
        assert (
            config.n_embd % config.n_head == 0
        ), f"Hidden size ({config.n_embd}) must be divisible by the number of heads ({config.n_head})."

    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")

    if n_head_kv is None:
        n_head_kv = getattr(config, "n_head_kv", None) or n_head
    assert n_head % n_head_kv == 0, "`n_head` must be divisible by `n_head_kv`."

    return n_head, n_head_kv, head_dim


def _update_kv_cache(kv: torch.FloatTensor, inference_params: InferenceParams, layer_idx: int) -> torch.FloatTensor:
    num_heads, head_dim = kv.shape[-2:]

    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]

    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    assert batch_end <= kv_cache.shape[0]

    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert sequence_end <= kv_cache.shape[1]

    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    kv = kv_cache[batch_start:batch_end, :sequence_end, ...]

    return kv


class MHA(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        rotary_dim: Optional[int] = None,
        rotary_emb_scale_base: Optional[float] = None,
        n_head: Optional[int] = None,
        n_head_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        layer_idx: Optional[int] = None,
        return_residual: bool = False,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Rotary embedding
        self.rotary_emb_dim = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", 0)
        if self.rotary_emb_dim > 0:
            rotary_kwargs = {"device": device}
            if rotary_emb_scale_base is not None and rotary_emb_scale_base > 0.0:
                rotary_kwargs["scale_base"] = rotary_emb_scale_base
            
            rotary_cls = FlashRotaryEmbedding if config.flash_rotary else RotaryEmbedding
            if rotary_cls is None:
                rotary_cls = RotaryEmbedding
            self.rotary_emb = rotary_cls(self.rotary_emb_dim, **rotary_kwargs)
        
        # MLP
        self.n_head, self.n_head_kv, self.head_dim = _find_mha_dims(config, n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim)
        op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        hidden_size = config.n_embd

        linear_cls = FusedDense if config.fused_dense else nn.Linear
        if linear_cls is None:
            linear_cls = nn.Linear

        self.Wqkv = linear_cls(hidden_size, op_size, bias=bias, device=device, dtype=dtype)
        self.out_proj = linear_cls(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Attention
        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=config.attn_pdrop)
        self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=config.attn_pdrop)

        self.layer_idx = layer_idx
        self.return_residual = return_residual
        self.checkpointing = checkpointing

    def _forward_self_attn(
        self, x: torch.FloatTensor, attention_mask: Optional[torch.BoolTensor]
    ) -> torch.FloatTensor:
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(qkv)

        if self.checkpointing:
            return torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, attention_mask=attention_mask)

        return self.inner_attn(qkv, attention_mask=attention_mask)

    def _forward_cross_attn(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams],
        attention_mask: Optional[torch.BoolTensor],
    ) -> torch.FloatTensor:
        qkv = self.Wqkv(x)

        q = qkv[..., : self.n_head * self.head_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        kv = qkv[..., self.n_head * self.head_dim :]
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        seqlen_offset = past_key_values.seqlen_offset if past_key_values is not None else 0
        causal = None if seqlen_offset == 0 else False
        if self.rotary_emb_dim > 0:
            q, kv = self.rotary_emb(q, kv=kv, seqlen_offset=seqlen_offset)

        if past_key_values is not None:
            kv = _update_kv_cache(kv, past_key_values, self.layer_idx)

        if self.checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.inner_cross_attn, q, kv, attention_mask=attention_mask, causal=causal
            )

        return self.inner_cross_attn(q, kv, attention_mask=attention_mask, causal=causal)

    def forward(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if attention_mask is not None and torch.any(~attention_mask.bool()):
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        # MHA
        if self.n_head == self.n_head_kv:
            if past_key_values is None:
                # If `past_key_values` are not supplied, we run self-attention
                attn_output = self._forward_self_attn(x, attention_mask)
            else:
                # If `past_key_values` are supplied, it means that we might have cached values and
                # could take advantage of cross-attention
                attn_output = self._forward_cross_attn(x, past_key_values, attention_mask)
        # MQA / GQA
        else:
            # Regardless of `past_key_values` being supplied or not, it always use cross-attention
            # because `q` and `kv` lengths might be different
            attn_output = self._forward_cross_attn(x, past_key_values, attention_mask)

        output = rearrange(attn_output, "... h d -> ... (h d)")
        output = self.out_proj(output)

        return output if not self.return_residual else (output, x)


class ParallelBlock(nn.Module):
    """Parallel block.

    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).

    """

    def __init__(
        self,
        config: PretrainedConfig,
        block_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = MHA(config, layer_idx=block_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(hidden_states, past_key_values=past_key_values, attention_mask=attention_mask)
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class CausalLMHead(nn.Module):
    """Causal Language Modeling head.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        logits = self.linear(hidden_states).to(torch.float32)

        return logits


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss


class MixFormerSequentialPreTrainedModel(PreTrainedModel):
    """MixFormer (sequential for DeepSpeed) pre-trained model."""

    config_class = MixFormerSequentialConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[Union[torch.LongTensor, torch.BoolTensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
            past_key_values = InferenceParams(
                max_seqlen=self.config.n_positions,
                max_batch_size=input_ids.shape[0],
                seqlen_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,
            )
        else:
            # Assume that `past_key_values` has cached all tokens up to the last token in `input_ids`
            past_key_values.seqlen_offset = len(input_ids[0]) - 1
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
    
    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MixFormerSequentialPreTrainedModel):
            module.gradient_checkpointing = value


class MixFormerSequentialForCausalLM(MixFormerSequentialPreTrainedModel):
    """MixFormer (sequential for DeepSpeed) for Causal Language Modeling."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"layers\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]
    _no_split_modules = ["ParallelBlock"]

    def __init__(self, config: MixFormerSequentialConfig) -> None:
        super().__init__(config)

        modules = [Embedding(config)]
        modules += [ParallelBlock(config, block_idx=i) for i in range(config.n_layer)]
        modules.append(CausalLMHead(config))

        self.layers = nn.Sequential(*modules)
        self.loss = CausalLMLoss()

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.layers[0].wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.layers[-1].linear

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.layers[-1].linear = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_layer = self.layers[0](input_ids)
        for module in self.layers[1:-1]:
            hidden_layer = module(hidden_layer, past_key_values=past_key_values, attention_mask=attention_mask)
        lm_logits = self.layers[-1](hidden_layer)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)