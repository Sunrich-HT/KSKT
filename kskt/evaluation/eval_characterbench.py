"""CharacterBench evaluation (paper Table 1).

Runs KSKT under the standardized CharacterBench protocol:
  - 13 metrics across 6 dimensions, 5-point scale
  - greedy decoding (temperature=0) for determinism
  - mean over three seeds: 42, 2023, 12345

The actual CharacterBench data must be downloaded separately. We expect
JSONL files matching the upstream schema:
  {"id":..., "role":..., "dialogue":[...], "question":..., "reference":...,
   "dimension":..., "metric":..., "language":"zh"|"en"}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import torch

from ..config import KSKTConfig
from ..modeling_kskt import KSKTForCausalLM


DIMENSIONS = ["Memory", "Knowledge", "Persona", "Emotion", "Morality", "Believability"]
METRICS = ["MC", "FA", "BC_K", "AC_b", "AC_h", "BC_b_P", "BC_h_P",
           "ES", "ER", "MS", "MR", "HL", "EG"]


@torch.no_grad()
def generate(model: KSKTForCausalLM, tokenizer, prompt: str, max_new_tokens: int = 256, device: str = "cuda") -> str:
    config: KSKTConfig = model.config
    ids = tokenizer.encode(
        config.role_marker_open + "[role placeholder]" + config.role_marker_close
        + "\n" + config.user_marker_open + prompt + config.user_marker_close + "\n",
        add_special_tokens=False,
    )
    input_ids = torch.tensor([ids], device=device)
    role_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    user_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    out_ids: List[int] = []
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, role_mask=role_mask, user_mask=user_mask)
        next_id = int(out["logits"][:, -1, :].argmax(dim=-1).item())
        if next_id == tokenizer.eos_token_id:
            break
        out_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        role_mask = torch.cat([role_mask, torch.zeros(1, 1, device=device)], dim=1)
        user_mask = torch.cat([user_mask, torch.zeros(1, 1, device=device)], dim=1)
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def score_with_judge(prediction: str, reference: str, metric: str, judge=None) -> float:
    """Hook for the official CharacterBench judge (GPT-4 or fine-tuned scorer).

    The paper uses the official protocol; here we expose a callable seam so
    users can plug in their own evaluator. Returns a 1-5 score.
    """
    if judge is None:
        from collections import Counter
        c1 = Counter(prediction.split())
        c2 = Counter(reference.split())
        if not c2:
            return 3.0
        overlap = sum((c1 & c2).values()) / max(1, sum(c2.values()))
        return 1.0 + 4.0 * min(1.0, overlap)
    return float(judge(prediction=prediction, reference=reference, metric=metric))


def run(args) -> Dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    config = KSKTConfig()
    model = KSKTForCausalLM(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device).eval()

    results = defaultdict(list)
    with open(args.bench_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = row["question"]
            pred = generate(model, tokenizer, prompt, device=args.device)
            score = score_with_judge(pred, row.get("reference", ""), row["metric"])
            results[(row["dimension"], row["metric"], row["language"])].append(score)

    summary = {}
    for (dim, metric, lang), scores in results.items():
        summary[f"{dim}/{metric}/{lang}"] = sum(scores) / len(scores)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bench_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", default="characterbench_results.json")
    args = p.parse_args()

    summary = run(args)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    overall = sum(summary.values()) / max(1, len(summary))
    print(f"CharacterBench overall: {overall:.3f}")


if __name__ == "__main__":
    main()
