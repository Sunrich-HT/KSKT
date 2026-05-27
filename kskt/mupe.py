"""Mutual-Understanding Position Encoding (MUPE).

Implements §3.2 ("Mutual-Understanding Position Encoding") and Appendix A.2.
Augments standard RoPE with two learned relevance signals: one that
encodes a token's relevance to the character context R_proc, and one for
the user context U_proc.

    PE_mutual(i) = PE_base(i) + W_self · f_self(i, R_proc) + W_other · f_other(i, U_proc)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import KSKTConfig


class _RelevanceMLP(nn.Module):
    """Two-layer MLP that fuses an absolute positional embedding with a
    perspective-specific context vector (mean-pooled R_proc or U_proc)."""

    def __init__(self, hidden_size: int, hidden_factor: float):
        super().__init__()
        h = max(int(hidden_size * hidden_factor), 1)
        self.fc1 = nn.Linear(hidden_size * 2, h)
        self.fc2 = nn.Linear(h, hidden_size)
        self.act = nn.ReLU()

    def forward(self, pe_abs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # pe_abs: [B, T, D], context: [B, D] -> broadcast.
        ctx = context.unsqueeze(1).expand_as(pe_abs)
        x = torch.cat([pe_abs, ctx], dim=-1)
        return self.fc2(self.act(self.fc1(x)))


class MutualUnderstandingPositionEncoding(nn.Module):
    """Learned additive correction to the input hidden states.

    Note on injection timing (Appendix A.2): the rotary RoPE itself is applied
    inside attention; MUPE is added to hidden states *before* attention, so
    the rotary mechanism is preserved.
    """

    def __init__(self, config: KSKTConfig):
        super().__init__()
        d = config.hidden_size
        self.abs_pe = nn.Embedding(config.max_position_embeddings, d)
        self.f_self = _RelevanceMLP(d, config.mupe_hidden_factor)
        self.f_other = _RelevanceMLP(d, config.mupe_hidden_factor)
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_other = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.abs_pe.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.w_self.weight)
        nn.init.zeros_(self.w_other.weight)
        self.temperature = config.mupe_temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        role_repr: torch.Tensor,
        user_repr: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        positions = torch.arange(t, device=hidden_states.device).unsqueeze(0).expand(b, t)
        pe = self.abs_pe(positions)

        # Mean-pool role / user reprs to a single context vector per batch.
        role_ctx = role_repr.mean(dim=1) if role_repr.dim() == 3 else role_repr
        user_ctx = user_repr.mean(dim=1) if user_repr.dim() == 3 else user_repr

        delta = self.w_self(self.f_self(pe, role_ctx)) + self.w_other(self.f_other(pe, user_ctx))
        return hidden_states + self.temperature * delta
