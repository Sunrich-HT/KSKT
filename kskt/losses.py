"""Auxiliary losses (Eq. 11).

  L = L_CLM + λ1 · L_consistency + λ2 · L_understanding + λ3 · L_balance

* L_CLM            : standard causal LM cross-entropy on response tokens.
* L_consistency    : encourages the self-stream to stay aligned with the role
                     description -- we compute it as the cosine distance
                     between pooled self-stream activations and pooled role
                     representations (lower is better).
* L_understanding  : the symmetric criterion against the user representation.
* L_balance        : standard MoE load-balancing penalty
                     Σ_j (f_j - 1/N)^2 (Eq. 10), encouraging uniform usage.

A phase-aware combiner is provided that follows the three-phase curriculum
from §3.5 / Appendix C.3 (phase 1: CLM+cons; phase 2: + und; phase 3: + bal).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def consistency_loss(self_stream: torch.Tensor, role_repr: torch.Tensor) -> torch.Tensor:
    """1 - cosine(mean(self_stream), mean(role_repr)) averaged over batch."""
    a = self_stream.mean(dim=1)
    b = role_repr.mean(dim=1)
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def understanding_loss(other_stream: torch.Tensor, user_repr: torch.Tensor) -> torch.Tensor:
    a = other_stream.mean(dim=1)
    b = user_repr.mean(dim=1)
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def balance_loss(expert_loads: List[torch.Tensor]) -> torch.Tensor:
    """Σ_layers Σ_j (f_j - 1/N)^2 averaged over layers."""
    if not expert_loads:
        return torch.tensor(0.0)
    losses = []
    for f in expert_loads:
        target = 1.0 / f.numel()
        losses.append(((f - target) ** 2).sum())
    return torch.stack(losses).mean()


def budget_supervision(
    budget_logits: torch.Tensor,
    fusion_alphas: torch.Tensor,
    routing_entropies: torch.Tensor,
    budgets: List[int],
    lam_f: float = 0.6,
    lam_e: float = 0.4,
) -> torch.Tensor:
    """Cross-entropy supervision for the budget classifier (Appendix A.3).

    Ground-truth label is derived from a discretized conflict score:
        ConflictScore = lam_f · E[|α - β|] + lam_e · S(p_expert)
    binned into len(budgets) classes by quantile.
    """
    with torch.no_grad():
        alpha = fusion_alphas.squeeze(-1)
        divergence = (alpha - (1.0 - alpha)).abs().mean(dim=-1)         # [B]
        entropy = routing_entropies.mean(dim=-1) if routing_entropies.dim() > 1 else routing_entropies
        score = lam_f * divergence + lam_e * entropy
        # Discretize via quantile bins of `len(budgets)` classes.
        n = len(budgets)
        # rank-based quantile mapping (stable, batch-local)
        ranks = score.argsort().argsort().float()
        labels = (ranks * n / score.numel()).clamp(max=n - 1).long()
    return F.cross_entropy(budget_logits, labels)


def kskt_total_loss(
    lm_loss: torch.Tensor,
    *,
    aux: Dict[str, torch.Tensor],
    phase: int,
    lambdas: Dict[str, float],
) -> torch.Tensor:
    """Combine LM + auxiliary losses according to the three-phase schedule.

    aux is expected to contain keys:
        'consistency', 'understanding', 'balance', 'budget' (optional)
    Each is a scalar tensor.
    """
    total = lm_loss
    if phase >= 1 and "consistency" in aux:
        total = total + lambdas["consistency"] * aux["consistency"]
    if phase >= 2 and "understanding" in aux:
        total = total + lambdas["understanding"] * aux["understanding"]
    if phase >= 3 and "balance" in aux:
        total = total + lambdas["balance"] * aux["balance"]
    # Budget supervision is light and active whenever computed.
    if "budget" in aux and aux["budget"] is not None:
        total = total + 0.05 * aux["budget"]
    return total
