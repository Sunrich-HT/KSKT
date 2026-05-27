"""KSKT configuration.

Mirrors the hyperparameters reported in the paper (Table 4 in the appendix,
"Complete Hyperparameter Settings"). Defaults match the Qwen3-4B-Thinking-2507
backbone we build on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class KSKTConfig:
    # -------- Backbone (Qwen3-4B-Thinking-2507) --------
    base_model: str = "Qwen/Qwen3-4B-Thinking-2507"
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8           # GQA
    head_dim: int = 128
    max_position_embeddings: int = 262_144
    rope_theta: float = 5_000_000.0
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151_936
    tie_word_embeddings: bool = False

    # -------- KSKT layer placement --------
    # DSAA replaces standard attention every 4th layer (paper, Appendix A.3
    # "Component Integration Strategy"): layers 4, 8, 12, 16, 20, 24, 28, 32, 36.
    dsaa_layer_indices: Tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31, 35)

    # -------- DSAA: Dual-Stream Axial Attention --------
    fusion_eps: float = 1e-8
    attention_bias_init_std: float = 0.02

    # -------- MUPE: Mutual-Understanding Position Encoding --------
    mupe_hidden_factor: float = 0.5         # MLP hidden = d * factor
    mupe_temperature: float = 0.5

    # -------- Bipolar Reasoning Module --------
    bipolar_budgets: Tuple[int, ...] = (2, 4, 6, 8, 10)
    bipolar_pre_gate_threshold: float = 0.5
    bipolar_gate_temperature: float = 1.0
    bipolar_post_gate_dropout: float = 0.1
    conflict_lambda_fusion: float = 0.6
    conflict_lambda_entropy: float = 0.4

    # -------- SAMOE: Self-Awareness MoE --------
    num_experts: int = 4                    # P, K, E, C
    expert_names: Tuple[str, ...] = ("personality", "knowledge", "emotion", "capability")
    samoe_temperature: float = 1.0          # learnable scalar
    samoe_top_k: int = 2
    expert_init_std: float = 0.01

    # -------- Loss weights (paper, Eq. 11; values in Appendix C.2) --------
    lambda_consistency: float = 0.1
    lambda_understanding: float = 0.2
    lambda_balance: float = 0.01

    # -------- Optimization (Appendix C.2, Table 4) --------
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 500
    global_batch_size: int = 128
    sequence_length: int = 2048
    # Phase-specific LR overrides (Appendix C.3)
    phase1_lr: float = 2e-5
    phase2_lr: float = 1e-5
    phase3_lr: float = 5e-6
    phase1_epochs: int = 2
    phase2_epochs: int = 1
    phase3_epochs: int = 1

    # -------- I/O --------
    role_marker_open: str = "<|role|>"
    role_marker_close: str = "<|/role|>"
    user_marker_open: str = "<|user|>"
    user_marker_close: str = "<|/user|>"

    # -------- Misc --------
    use_cache: bool = True
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"

    def head_total_q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    def head_total_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim
