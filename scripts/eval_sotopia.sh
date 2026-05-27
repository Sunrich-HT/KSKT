#!/usr/bin/env bash
# Evaluate on SOTOPIA-Eval (paper Table 2).
set -euo pipefail

CKPT="${CKPT:-runs/kskt_4b/phase3_mutual.pt}"
SCENARIOS="${SCENARIOS:-data/sotopia/scenarios_180.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

python -m kskt.evaluation.eval_sotopia \
    --checkpoint "${CKPT}" \
    --scenarios_jsonl "${SCENARIOS}" \
    --base_model "${BASE_MODEL}" \
    --output_json results/sotopia.json
