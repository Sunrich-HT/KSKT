"""KSKT: Know Thyself, Know Thy User — Intrinsic Dual-Perspective Reasoning for Role-Playing LLMs (ICML 2026)."""

from .config import KSKTConfig
from .dsaa import DualStreamAxialAttention
from .mupe import MutualUnderstandingPositionEncoding
from .bipolar import BipolarReasoningModule
from .samoe import SelfAwarenessMoE
from .modeling_kskt import KSKTModel, KSKTForCausalLM, KSKTLayer
from .losses import (
    consistency_loss,
    understanding_loss,
    balance_loss,
    kskt_total_loss,
)

__version__ = "0.1.0"

__all__ = [
    "KSKTConfig",
    "DualStreamAxialAttention",
    "MutualUnderstandingPositionEncoding",
    "BipolarReasoningModule",
    "SelfAwarenessMoE",
    "KSKTModel",
    "KSKTForCausalLM",
    "KSKTLayer",
    "consistency_loss",
    "understanding_loss",
    "balance_loss",
    "kskt_total_loss",
]
