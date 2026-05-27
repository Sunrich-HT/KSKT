"""Top-level training entry point for KSKT.

Usage:
    python -m kskt.training.train \\
        --train_jsonl data/role_dialogues.jsonl \\
        --base_model Qwen/Qwen3-4B-Thinking-2507 \\
        --output_dir runs/kskt_4b
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from ..config import KSKTConfig
from ..data.dataset import KSKTDialogueDataset, collate_kskt
from ..modeling_kskt import KSKTForCausalLM
from .trainer import KSKTTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--output_dir", default="runs/kskt_4b")
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--config_overrides", default=None,
                   help="Optional JSON dict of KSKTConfig overrides.")
    args = p.parse_args()

    config = KSKTConfig()
    if args.config_overrides:
        for k, v in json.loads(args.config_overrides).items():
            if hasattr(config, k):
                setattr(config, k, v)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # Register the role/user marker tokens so they survive tokenization.
    tokenizer.add_special_tokens({"additional_special_tokens": [
        config.role_marker_open, config.role_marker_close,
        config.user_marker_open, config.user_marker_close,
    ]})

    dataset = KSKTDialogueDataset(args.train_jsonl, tokenizer, config)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_kskt(b, pad_id=pad_id),
        num_workers=2,
        drop_last=True,
    )

    model = KSKTForCausalLM(config)
    if tokenizer.vocab_size != config.vocab_size:
        # Resize embedding to match the (possibly expanded) tokenizer.
        new_vocab = len(tokenizer)
        new_emb = torch.nn.Embedding(new_vocab, config.hidden_size)
        new_emb.weight.data[: config.vocab_size] = model.model.embed_tokens.weight.data[: min(new_vocab, config.vocab_size)]
        model.model.embed_tokens = new_emb
        new_head = torch.nn.Linear(config.hidden_size, new_vocab, bias=False)
        new_head.weight.data[: config.vocab_size] = model.lm_head.weight.data[: min(new_vocab, config.vocab_size)]
        model.lm_head = new_head
        config.vocab_size = new_vocab
    try:
        model.load_qwen3_weights(args.base_model)
    except Exception as e:
        print(f"[WARN] could not load base weights: {e}; continuing with fresh init.")

    trainer = KSKTTrainer(
        model=model,
        config=config,
        train_loader=loader,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.train(args.output_dir)


if __name__ == "__main__":
    main()
