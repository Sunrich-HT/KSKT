"""Minimal smoke test: build a KSKT model and run a forward pass.

Designed to be runnable on CPU (uses a small toy config) so contributors
can verify the architecture wiring without GPUs or downloaded weights.
"""

from __future__ import annotations

import torch

from kskt import KSKTConfig, KSKTForCausalLM


def main() -> None:
    # A tiny config so this runs on a laptop. The released model uses the
    # full Qwen3-4B-Thinking-2507 configuration (see configs/kskt_4b.yaml).
    config = KSKTConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        dsaa_layer_indices=(1, 3),
        max_position_embeddings=64,
    )

    model = KSKTForCausalLM(config)
    model.eval()

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    role_mask = torch.zeros(B, T)
    role_mask[:, 4:10] = 1
    user_mask = torch.zeros(B, T)
    user_mask[:, 12:18] = 1
    attn_mask = torch.ones(B, T)

    out = model(
        input_ids=input_ids,
        role_mask=role_mask,
        user_mask=user_mask,
        attention_mask=attn_mask,
        labels=input_ids,
    )
    print("logits shape :", tuple(out["logits"].shape))
    print("loss         :", float(out["loss"]))
    print("kskt layers  :", len(out["aux"]))
    if out["aux"]:
        last = out["aux"][-1]
        print("alpha range  :", float(last["alpha"].min()), float(last["alpha"].max()))
        print("expert load  :", last["expert_load"].tolist())


if __name__ == "__main__":
    main()
