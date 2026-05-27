#!/usr/bin/env bash
# Evaluate a trained KSKT checkpoint on CharacterBench (paper Table 1).
set -euo pipefail

CKPT="${CKPT:-runs/kskt_4b/phase3_mutual.pt}"
BENCH="${BENCH:-data/characterbench/test.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

python -m kskt.evaluation.eval_characterbench \
    --checkpoint "${CKPT}" \
    --bench_jsonl "${BENCH}" \
    --base_model "${BASE_MODEL}" \
    --output_json results/characterbench.json
