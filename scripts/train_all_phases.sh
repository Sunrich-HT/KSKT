#!/usr/bin/env bash
# Train the full KSKT 4B model end-to-end (all 3 phases).
#
# Reproduces the paper recipe:
#   Phase 1 (2 epochs, lr=2e-5) : self-understanding foundation
#   Phase 2 (1 epoch , lr=1e-5) : other-understanding addition
#   Phase 3 (1 epoch , lr=5e-6) : mutual integration (full L)
set -euo pipefail

TRAIN_JSONL="${TRAIN_JSONL:-data/role_dialogues.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/kskt_4b}"

python -m kskt.training.train \
    --train_jsonl "${TRAIN_JSONL}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 4
