"""Three-phase KSKT trainer (§3.5 / Appendix C.3).

Phase 1 (2 epochs): L_CLM + λ1 · L_consistency.        lr = 2e-5
Phase 2 (1 epoch ): + λ2 · L_understanding.             lr = 1e-5
Phase 3 (1 epoch ): + λ3 · L_balance (full Eq. 11).     lr = 5e-6 cosine

All phases share the same 130K mixture. We use AdamW + linear warmup
(500 steps in phase 1, none in subsequent phases by default) and grad
clipping at 1.0.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import KSKTConfig
from ..losses import (
    balance_loss,
    budget_supervision,
    consistency_loss,
    kskt_total_loss,
    understanding_loss,
)


@dataclass
class PhaseSpec:
    name: str
    phase_index: int            # 1, 2, or 3
    epochs: int
    learning_rate: float
    warmup_steps: int = 0
    schedule: str = "linear"    # "linear" | "cosine" | "constant"


def build_three_phase_schedule(config: KSKTConfig) -> List[PhaseSpec]:
    return [
        PhaseSpec("phase1_self", 1, config.phase1_epochs, config.phase1_lr, warmup_steps=config.warmup_steps, schedule="linear"),
        PhaseSpec("phase2_other", 2, config.phase2_epochs, config.phase2_lr, schedule="linear"),
        PhaseSpec("phase3_mutual", 3, config.phase3_epochs, config.phase3_lr, schedule="cosine"),
    ]


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n.lower() or "bias" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def _make_scheduler(optimizer, total_steps: int, warmup: int, kind: str):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class KSKTTrainer:
    """Minimal, framework-agnostic trainer that owns the phase curriculum."""

    def __init__(
        self,
        model,
        config: KSKTConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        log_fn: Callable[[Dict], None] = print,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.log_fn = log_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lambdas = {
            "consistency": config.lambda_consistency,
            "understanding": config.lambda_understanding,
            "balance": config.lambda_balance,
        }

    def _compute_aux_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        aux_layers = outputs["aux"]
        if not aux_layers:
            return {}
        # Use the *last* KSKT layer's α/β, h_self/h_other, and routing for
        # auxiliary supervision. Balance loss aggregates loads across layers.
        last = aux_layers[-1]
        h_self = last["h_self"]
        h_other = last["h_other"]

        # Pseudo R_proc / U_proc from token-level masks (mean pool over role /
        # user tokens). These are the same vectors the model consumed.
        role_mask = batch["role_mask"].unsqueeze(-1)
        user_mask = batch["user_mask"].unsqueeze(-1)
        # Pool over time to a single [B, D] vector each.
        role_repr = (h_self * role_mask).sum(1) / role_mask.sum(1).clamp_min(1e-6)
        user_repr = (h_other * user_mask).sum(1) / user_mask.sum(1).clamp_min(1e-6)

        c = consistency_loss(h_self, role_repr.unsqueeze(1))
        u = understanding_loss(h_other, user_repr.unsqueeze(1))
        b = balance_loss([a["expert_load"] for a in aux_layers])
        bud = budget_supervision(
            last["budget_logits"],
            last["alpha"],
            last["routing_entropy"],
            budgets=list(self.config.bipolar_budgets),
            lam_f=self.config.conflict_lambda_fusion,
            lam_e=self.config.conflict_lambda_entropy,
        )
        return {"consistency": c, "understanding": u, "balance": b, "budget": bud}

    def _train_phase(self, spec: PhaseSpec) -> None:
        optimizer = _make_optimizer(self.model, spec.learning_rate, self.config.weight_decay)
        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, self.gradient_accumulation_steps))
        total_steps = steps_per_epoch * spec.epochs
        scheduler = _make_scheduler(optimizer, total_steps, spec.warmup_steps, spec.schedule)

        self.model.train()
        global_step = 0
        for epoch in range(spec.epochs):
            optimizer.zero_grad()
            for it, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    role_mask=batch["role_mask"],
                    user_mask=batch["user_mask"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                lm_loss = outputs["loss"]
                aux = self._compute_aux_losses(outputs, batch)
                loss = kskt_total_loss(lm_loss, aux=aux, phase=spec.phase_index, lambdas=self.lambdas)
                (loss / self.gradient_accumulation_steps).backward()

                if (it + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 20 == 0:
                        log = {
                            "phase": spec.name,
                            "epoch": epoch,
                            "step": global_step,
                            "lr": scheduler.get_last_lr()[0],
                            "loss": float(loss.detach()),
                            "lm_loss": float(lm_loss.detach()),
                        }
                        for k, v in aux.items():
                            log[f"aux/{k}"] = float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                        self.log_fn(log)

    def train(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for spec in build_three_phase_schedule(self.config):
            self.log_fn({"event": "phase_start", "phase": spec.name, "lr": spec.learning_rate, "epochs": spec.epochs})
            self._train_phase(spec)
            ckpt_path = os.path.join(output_dir, f"{spec.name}.pt")
            torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, ckpt_path)
            self.log_fn({"event": "phase_end", "phase": spec.name, "checkpoint": ckpt_path})
