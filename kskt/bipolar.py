"""Bipolar Reasoning Module (BRM).

Implements §3.3 + Appendix A.3. Two pathways:

  * System-1 (fast): a single SwiGLU FFN over H_DSAA.
  * System-2 (slow): a ThinkingChain of T ∈ {2,4,6,8,10} cross-attention
    steps that integrate role and user context.

A two-stage gating mechanism avoids the circular dependency between gate
and slow path:

  * Pre-gate uses only (H_DSAA, H_fast) to decide whether System-2 fires.
  * Post-gate fuses (H_DSAA, H_fast, H_slow) into H_reason via a scalar
    token-level gate g.

The budget T is predicted from H_DSAA via a 5-way classifier and is
supervised, at training time, by a conflict score computed from fusion
weight divergence |α-β| and expert-routing entropy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSKTConfig


class _SwiGLUFFN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: Optional[int] = None):
        super().__init__()
        d_out = d_out or d_in
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _CrossAttention(nn.Module):
    """Single-head cross-attention from queries onto a [R_proc; U_proc]
    concatenation. Used inside the ThinkingChain (Appendix A.3, Eq.
    "c_i = CrossAttention(h_{i-1}, [R_proc; U_proc])")."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.q(queries)
        k = self.k(context)
        v = self.v(context)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        return torch.matmul(attn, v)


class ThinkingChain(nn.Module):
    """Sequential reasoning chain of length T_max, with early-stop driven by
    the predicted budget T. Implements:

        c_i = CrossAttention(h_{i-1}, [R_proc; U_proc])
        t_i = MLP(concat(h_{i-1}, c_i))
        h_i = LayerNorm(h_{i-1} + t_i)
    """

    def __init__(self, config: KSKTConfig):
        super().__init__()
        d = config.hidden_size
        self.cross_attn = _CrossAttention(d)
        self.step_mlp = _SwiGLUFFN(d_in=2 * d, d_hidden=config.intermediate_size, d_out=d)
        self.norm = nn.LayerNorm(d, eps=config.rms_norm_eps)
        self.budgets = config.bipolar_budgets

    def forward(
        self,
        h: torch.Tensor,
        role_repr: torch.Tensor,
        user_repr: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        context = torch.cat([role_repr, user_repr], dim=1)
        h_i = h
        for _ in range(budget):
            c_i = self.cross_attn(h_i, context)
            t_i = self.step_mlp(torch.cat([h_i, c_i], dim=-1))
            h_i = self.norm(h_i + t_i)
        return h_i


class BipolarReasoningModule(nn.Module):
    def __init__(self, config: KSKTConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size

        # Fast path (System-1): single SwiGLU FFN.
        self.fast_ffn = _SwiGLUFFN(d, config.intermediate_size)
        self.fast_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)

        # Slow path (System-2).
        self.thinking_chain = ThinkingChain(config)

        # Pre-gate (Eq. for p_sys2): two-layer MLP over [H_DSAA; H_fast].
        self.pre_gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Budget predictor: 5-way classifier over H_DSAA (pooled).
        self.budget_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, len(config.bipolar_budgets)),
        )

        # Post-gate: 3-layer MLP, dropout p=0.1.
        self.post_gate = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(config.bipolar_post_gate_dropout),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(config.bipolar_post_gate_dropout),
            nn.Linear(d, 1),
        )

        self.pre_gate_threshold = config.bipolar_pre_gate_threshold

    def predict_budget(self, h_dsaa: torch.Tensor) -> Tuple[int, torch.Tensor]:
        pooled = h_dsaa.mean(dim=1)                          # [B, D]
        logits = self.budget_head(pooled)                    # [B, K]
        idx = int(logits.argmax(dim=-1).float().mean().round().item())
        idx = max(0, min(idx, len(self.config.bipolar_budgets) - 1))
        return self.config.bipolar_budgets[idx], logits

    def forward(
        self,
        h_dsaa: torch.Tensor,
        role_repr: torch.Tensor,
        user_repr: torch.Tensor,
        force_system2: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # System-1.
        h_fast = self.fast_ffn(self.fast_norm(h_dsaa))       # Eq. 8

        # Pre-gate decides if System-2 should fire (sequence-pooled).
        pre_inp = torch.cat([h_dsaa.mean(dim=1), h_fast.mean(dim=1)], dim=-1)
        p_sys2 = torch.sigmoid(self.pre_gate(pre_inp)).squeeze(-1)  # [B]

        # Budget logits are always emitted (used by aux loss); only consumed
        # when we actually run System-2.
        _, budget_logits = self.predict_budget(h_dsaa)

        trigger = force_system2 if force_system2 is not None else bool((p_sys2 > self.pre_gate_threshold).any())

        if trigger:
            budget, _ = self.predict_budget(h_dsaa)
            h_slow = self.thinking_chain(h_dsaa, role_repr, user_repr, budget)
            gate_inp = torch.cat([h_dsaa, h_fast, h_slow], dim=-1)
            g = torch.sigmoid(self.post_gate(gate_inp))      # [B, T, 1]
            h_reason = g * h_fast + (1.0 - g) * h_slow       # Eq. 9
        else:
            budget = self.config.bipolar_budgets[0]
            h_slow = h_fast
            g = torch.ones(h_fast.shape[0], h_fast.shape[1], 1, device=h_fast.device, dtype=h_fast.dtype)
            h_reason = h_fast

        aux = {
            "p_sys2": p_sys2,
            "budget": budget,
            "budget_logits": budget_logits,
            "gate": g,
            "h_fast": h_fast,
            "h_slow": h_slow,
            "system2_triggered": trigger,
        }
        return h_reason, aux
