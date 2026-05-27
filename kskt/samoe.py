"""Self-Awareness Mixture of Experts (SAMOE).

Implements §3.4 + Appendix A.4. Four SwiGLU experts specialize over
character dimensions: Personality, Knowledge, Emotion, Capability (P/K/E/C).
Routing is *character-conditioned*: a query vector q_self is computed by
attending from the post-arbitration hidden states onto the role context,
then concatenated with a pooled role context and passed through a SwiGLU.

Routing is top-k (k=2 by default, Appendix A.4 "Calibration and Conflict
Handling"), with renormalized residual combination. Load balancing follows
the L_balance = Σ_j (f_j - 1/N)^2 penalty (Eq. 10).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSKTConfig


class _SwiGLUExpert(nn.Module):
    def __init__(self, d: int, d_hidden: int, init_std: float):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_hidden, bias=False)
        self.up_proj = nn.Linear(d, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d, bias=False)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SelfAwarenessMoE(nn.Module):
    def __init__(self, config: KSKTConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.samoe_top_k
        assert self.top_k <= self.num_experts

        self.experts = nn.ModuleList(
            [_SwiGLUExpert(d, config.intermediate_size, config.expert_init_std)
             for _ in range(self.num_experts)]
        )

        # Character-specific routing query (Appendix A.4, Eq. for q_self):
        #   M_role  = softmax(R_proc W H^T / sqrt(d))     attention map
        #   h_role  = pooled H weighted by M_role
        #   h_ctx   = mean(R_proc)
        #   q_self  = SwiGLU(W_concat [h_role; h_ctx])
        self.w_proj = nn.Linear(d, d, bias=False)
        self.gate_in = nn.Linear(2 * d, d, bias=False)
        self.gate_up = nn.Linear(2 * d, d, bias=False)
        self.gate_down = nn.Linear(d, self.num_experts, bias=True)

        # Learnable temperature (initialized at 1.0; the paper observes it
        # converges to [0.7, 1.3]).
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def _route(self, h_reason: torch.Tensor, role_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = h_reason.shape
        # Attention: each role token over each hidden state.
        m = torch.matmul(role_repr, self.w_proj(h_reason).transpose(-1, -2)) / (d ** 0.5)
        m = torch.softmax(m, dim=-1)                              # [B, n_r, T]
        # h_role: aggregate H using the column-marginal of the role attention.
        weights = m.sum(dim=1, keepdim=True)                      # [B, 1, T]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        h_role = (weights.transpose(-1, -2) * h_reason).sum(dim=1)  # [B, D]
        h_ctx = role_repr.mean(dim=1)                              # [B, D]
        q_in = torch.cat([h_role, h_ctx], dim=-1)
        q_self = F.silu(self.gate_in(q_in)) * self.gate_up(q_in)   # SwiGLU-style
        tau = torch.exp(self.log_temperature).clamp(min=1e-3)
        logits = self.gate_down(q_self) / tau                      # [B, N]
        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def forward(
        self,
        h_reason: torch.Tensor,
        role_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        b, t, d = h_reason.shape
        probs, logits = self._route(h_reason, role_repr)           # [B, N]

        # Top-k sparse routing with renormalization.
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)         # [B, k]
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(h_reason)
        for k in range(self.top_k):
            idx_k = top_idx[:, k]                                  # [B]
            w_k = top_vals[:, k].view(b, 1, 1)                     # [B, 1, 1]
            # Per-sample expert dispatch.
            for e_id in range(self.num_experts):
                mask = (idx_k == e_id)
                if not mask.any():
                    continue
                inp = h_reason[mask]
                e_out = self.experts[e_id](inp)
                out[mask] = out[mask] + w_k[mask] * e_out

        # Stats for load balancing & analysis.
        with torch.no_grad():
            f_j = probs.mean(dim=0)                                # [N]
        aux = {
            "routing_probs": probs,                                # [B, N]
            "routing_logits": logits,
            "expert_load": f_j,                                    # [N]
            "top_k_idx": top_idx,
            "top_k_weights": top_vals,
            "routing_entropy": -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1),
        }
        return out, aux
