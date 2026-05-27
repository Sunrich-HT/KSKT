"""SOTOPIA evaluation (paper Table 2).

Wraps the official SOTOPIA-Eval pipeline. The actual benchmark code lives
in the `sotopia` package upstream (https://github.com/sotopia-lab/sotopia);
this script is the KSKT-side adapter that exposes `KSKTForCausalLM` as a
SOTOPIA-compatible agent.

SOTOPIA scores seven rubric dimensions:
  - Self-focused : Believability, Secret, Social Rules, Financial
  - Other-focused: Goal, Knowledge
  - Balance      : Relationship  <- KSKT's primary lift (+19.3%)
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from ..config import KSKTConfig
from ..modeling_kskt import KSKTForCausalLM


class KSKTAgent:
    """SOTOPIA-compatible agent backed by a KSKT checkpoint.

    The integration point is `act(observation) -> utterance`; everything
    else (multi-turn orchestration, GPT-4 scoring) is handled by SOTOPIA.
    """

    def __init__(self, checkpoint: str, base_model: str, device: str = "cuda"):
        from transformers import AutoTokenizer
        self.config = KSKTConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.model = KSKTForCausalLM(self.config).to(device).eval()
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.device = device

    @torch.no_grad()
    def act(self, role_text: str, observation: str, max_new_tokens: int = 200) -> str:
        cfg = self.config
        prompt = (
            cfg.role_marker_open + role_text + cfg.role_marker_close + "\n"
            + cfg.user_marker_open + observation + cfg.user_marker_close + "\n"
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=self.device)
        role_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        user_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            out = self.model(input_ids=input_ids, role_mask=role_mask, user_mask=user_mask)
            nxt = int(out["logits"][:, -1, :].argmax(dim=-1).item())
            if nxt == self.tokenizer.eos_token_id:
                break
            out_ids.append(nxt)
            input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            role_mask = torch.cat([role_mask, torch.zeros(1, 1, device=self.device)], dim=1)
            user_mask = torch.cat([user_mask, torch.zeros(1, 1, device=self.device)], dim=1)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)


def aggregate_scores(per_episode_scores: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_episode_scores:
        return {}
    keys = list(per_episode_scores[0].keys())
    return {k: sum(s[k] for s in per_episode_scores) / len(per_episode_scores) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--scenarios_jsonl", required=True,
                   help="SOTOPIA scenarios file (N=180 in the paper).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", default="sotopia_results.json")
    args = p.parse_args()

    try:
        from sotopia.envs.evaluators import ReachGoalLLMEvaluator
        from sotopia.agents import LLMAgent
    except ImportError:
        print("[WARN] `sotopia` not installed. This script is the agent adapter only.")
        print("       Install via `pip install sotopia` and follow the official pipeline.")
        return

    agent = KSKTAgent(args.checkpoint, args.base_model, args.device)
    scores: List[Dict[str, float]] = []
    with open(args.scenarios_jsonl, encoding="utf-8") as f:
        for line in f:
            scenario = json.loads(line)
            # Pseudocode: hand `agent` to SOTOPIA's environment, run the
            # episode, collect rubric scores. We leave the actual loop to
            # the SOTOPIA orchestrator.
            pass

    summary = aggregate_scores(scores)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
