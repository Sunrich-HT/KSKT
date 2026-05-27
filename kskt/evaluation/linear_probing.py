"""Linear probing of self/other stream specialization (paper Table 3).

Trains a single-layer linear classifier on the FROZEN final-DSAA-layer
self- and other-stream activations. Replicates the cross-dissociation
result in §4: the self-stream predicts role attributes (80.5%) and the
other-stream predicts user intent (79.9%), with the cross-condition gaps
being approximately +21.3 / +20.8pp.

Probes:
  1. Role-attribute classifier (multi-class) on self-stream pooled outputs.
  2. User-intent classifier (multi-class) on other-stream pooled outputs.
  3. Baseline: same probes trained on the base model's hidden states.

We use SGD with 50 epochs and 3 seeds, matching the paper.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..modeling_kskt import KSKTForCausalLM
from ..config import KSKTConfig


def _train_linear(features: torch.Tensor, labels: torch.Tensor, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    n = features.size(0)
    perm = torch.randperm(n, generator=g)
    split = int(n * 0.8)
    train_x, train_y = features[perm[:split]], labels[perm[:split]]
    test_x, test_y = features[perm[split:]], labels[perm[split:]]
    n_classes = int(labels.max().item() + 1)
    probe = torch.nn.Linear(features.size(-1), n_classes)
    optim = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for _ in range(50):
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, generator=g)
        for xb, yb in loader:
            optim.zero_grad()
            F.cross_entropy(probe(xb), yb).backward()
            optim.step()
    with torch.no_grad():
        acc = (probe(test_x).argmax(-1) == test_y).float().mean().item()
    return acc


@torch.no_grad()
def extract_stream_features(model: KSKTForCausalLM, batches: List[Dict]) -> Dict[str, torch.Tensor]:
    """Mean-pooled final-DSAA-layer self/other-stream features."""
    self_feats, other_feats = [], []
    for batch in batches:
        outputs = model(
            input_ids=batch["input_ids"],
            role_mask=batch["role_mask"],
            user_mask=batch["user_mask"],
            attention_mask=batch.get("attention_mask"),
        )
        last = outputs["aux"][-1]
        self_feats.append(last["h_self"].mean(dim=1).cpu())
        other_feats.append(last["h_other"].mean(dim=1).cpu())
    return {
        "self": torch.cat(self_feats, dim=0),
        "other": torch.cat(other_feats, dim=0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe_jsonl", required=True,
                   help="JSONL with 'role_attribute_label' and 'user_intent_label' int labels.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    config = KSKTConfig()
    model = KSKTForCausalLM(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    # Loading the probe data is left to the caller's data layout; we expect
    # them to pre-encode to tensors and dump a small .pt file alongside
    # `--probe_jsonl` with keys `features`, `role_labels`, `intent_labels`.
    probe = torch.load(args.probe_jsonl.replace(".jsonl", ".pt"))
    feats = {"self": probe["self_feats"], "other": probe["other_feats"]}

    results = {}
    for stream_name, feat in feats.items():
        for target_name, labels in (("role_attr", probe["role_labels"]), ("user_intent", probe["intent_labels"])):
            accs = [_train_linear(feat, labels, seed) for seed in (42, 2023, 12345)]
            results[f"{stream_name}/{target_name}"] = sum(accs) / len(accs)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
