"""Dual-Stream Axial Attention (DSAA).

Implements §3.2 of the paper. Standard attention is factorized along the
self / other perspective axes: each stream has its own Q/K/V projections,
learned attention bias, and the two stream outputs are combined via a
token-level fusion weight that satisfies the simplex constraint α + β = 1.

References (paper):
    Eq. 1–2 : per-stream projections.
    Eq. 3–4 : biased attention.
    Eq. 5–7 : adaptive fusion (sigmoid + complement).
    Eq. 7   : H_out = α ⊙ (A_self V_self) + β ⊙ (A_other V_other).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSKTConfig
from .mupe import MutualUnderstandingPositionEncoding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q,k of shape [B, H, T, D]."""
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv * n_rep, T, D]; GQA-style head expansion."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class _SingleStreamAttention(nn.Module):
    """One stream of DSAA: standard GQA + a learnable additive attention bias.

    The bias term `B_self` / `B_other` of Eq. 3–4 is implemented as a learned
    scalar-per-(query-position, key-position-type) added in logit space. We
    parameterize it as a *content-conditioned* bias (a function of the role /
    user mask), not as a full n×n matrix, so it scales with the actual
    sequence length seen at runtime.
    """

    def __init__(self, config: KSKTConfig, *, stream: str):
        super().__init__()
        assert stream in {"self", "other"}
        self.config = config
        self.stream = stream

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Separate Q,K,V matrices per stream (Eq. 1–2). GQA: Q expands to
        # `num_heads * head_dim`, K/V to `num_kv_heads * head_dim`.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-Norm (Qwen3 default; see Appendix A.2 "Base Model Architecture Compatibility").
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Learned scalar bias that emphasizes role tokens (self-stream) or
        # user tokens (other-stream). Init from N(0, 0.02) per Appendix A.2.
        self.bias_scale = nn.Parameter(torch.zeros(1).normal_(mean=0.0, std=config.attention_bias_init_std))

        self.attention_dropout = config.attention_dropout
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(proj.weight, gain=std)

    def _build_bias(self, role_mask: torch.Tensor, user_mask: torch.Tensor) -> torch.Tensor:
        """Bias of shape [B, 1, T_q, T_k] added to logits."""
        target_mask = role_mask if self.stream == "self" else user_mask
        # Encourage attention to flow toward the stream's "own" token type.
        # bias[b, q, k] = bias_scale if k ∈ target tokens else 0.
        b, t = target_mask.shape
        bias = target_mask.to(role_mask.dtype).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]
        return bias * self.bias_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        role_mask: torch.Tensor,
        user_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # [B, H, T, D]
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        k = _repeat_kv(k, self.num_heads // self.num_kv_heads)
        v = _repeat_kv(v, self.num_heads // self.num_kv_heads)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, T, T]
        logits = logits + self._build_bias(role_mask, user_mask)
        if attention_mask is not None:
            logits = logits + attention_mask  # additive causal/padding mask

        attn = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = F.dropout(attn, p=self.attention_dropout, training=self.training)

        out = torch.matmul(attn, v)                               # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(b, t, -1)     # [B, T, H*D]
        return self.o_proj(out)


class DualStreamAxialAttention(nn.Module):
    """Two-stream factorized attention with token-level adaptive fusion."""

    def __init__(self, config: KSKTConfig, mupe: Optional[MutualUnderstandingPositionEncoding] = None):
        super().__init__()
        self.config = config
        self.self_stream = _SingleStreamAttention(config, stream="self")
        self.other_stream = _SingleStreamAttention(config, stream="other")

        # Fusion (Eq. 5–6): scalar gate per token from a single linear layer.
        self.fusion_proj = nn.Linear(config.hidden_size, 1, bias=True)
        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

        self.mupe = mupe  # optional MUPE injection; if None, RoPE-only.

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        role_mask: torch.Tensor,
        user_mask: torch.Tensor,
        role_repr: Optional[torch.Tensor] = None,
        user_repr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (output, alpha, beta, h_self, h_other).

        alpha, beta are token-level mixing weights with α + β = 1 and α,β ≥ 0.
        h_self and h_other are the per-stream attention outputs *before*
        fusion -- exposed so the linear-probing analysis can read them
        directly without re-running attention.
        """
        if self.mupe is not None and role_repr is not None and user_repr is not None:
            hidden_states = self.mupe(hidden_states, role_repr, user_repr)

        h_self = self.self_stream(hidden_states, cos, sin, attention_mask, role_mask, user_mask)
        h_other = self.other_stream(hidden_states, cos, sin, attention_mask, role_mask, user_mask)

        # Token-level fusion (Eq. 5–7). Sigmoid ensures α ∈ (0,1); β=1-α gives
        # the simplex constraint exactly. Per the paper, this also yields an
        # interpretable "self-perspective trust" signal.
        logits = self.fusion_proj(hidden_states)                       # [B,T,1]
        alpha = torch.sigmoid(logits)
        beta = 1.0 - alpha

        out = alpha * h_self + beta * h_other
        return out, alpha, beta, h_self, h_other
