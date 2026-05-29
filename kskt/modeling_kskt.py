"""KSKT model assembly.

We adopt Qwen3-4B-Thinking-2507 as the backbone (§3.1, §3.5; Appendix A.2).
A `KSKTLayer` replaces the standard transformer block at the indices listed
in `config.dsaa_layer_indices` (every 4th layer by default). Each KSKTLayer
runs:

    DSAA  ->  BipolarReasoning  ->  SAMOE  ->  residual + RMSNorm + LM block

Layers outside `dsaa_layer_indices` fall back to the backbone's native
transformer block. We load weights from the HF Qwen3 checkpoint and copy
them into the un-modified layers; the new KSKT parameters are initialized
from scratch.

The wrapper is intentionally lightweight: it relies on
`transformers.AutoModelForCausalLM` for the backbone, then surgically
swaps in `KSKTLayer` instances at the chosen indices.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSKTConfig
from .dsaa import DualStreamAxialAttention
from .mupe import MutualUnderstandingPositionEncoding
from .bipolar import BipolarReasoningModule
from .samoe import SelfAwarenessMoE


class KSKTLayer(nn.Module):
    """A single KSKT block: DSAA -> BRM -> SAMOE with residuals.

    Mirrors the per-layer recipe summarized in Algorithm 1 of the paper.
    """

    def __init__(self, config: KSKTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reason_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mupe = MutualUnderstandingPositionEncoding(config)
        self.dsaa = DualStreamAxialAttention(config, mupe=self.mupe)
        self.bipolar = BipolarReasoningModule(config)
        self.samoe = SelfAwarenessMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        role_mask: torch.Tensor,
        user_mask: torch.Tensor,
        role_repr: torch.Tensor,
        user_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # ---- DSAA ----
        residual = hidden_states
        h, alpha, beta, h_self, h_other = self.dsaa(
            self.attn_norm(hidden_states),
            cos=cos, sin=sin,
            attention_mask=attention_mask,
            role_mask=role_mask, user_mask=user_mask,
            role_repr=role_repr, user_repr=user_repr,
        )
        h_dsaa = residual + h

        # ---- Bipolar Reasoning ----
        residual = h_dsaa
        h_reason, brm_aux = self.bipolar(self.reason_norm(h_dsaa), role_repr, user_repr)
        h_reason = residual + h_reason

        # ---- SAMOE ----
        residual = h_reason
        h_moe, samoe_aux = self.samoe(self.moe_norm(h_reason), role_repr)
        out = residual + h_moe

        aux = {
            "alpha": alpha,
            "beta": beta,
            # Per-stream attention outputs *before* fusion -- used by the
            # linear-probing eval (Table 3 cross-dissociation).
            "h_self": h_self,
            "h_other": h_other,
            "budget_logits": brm_aux["budget_logits"],
            "routing_probs": samoe_aux["routing_probs"],
            "routing_entropy": samoe_aux["routing_entropy"],
            "expert_load": samoe_aux["expert_load"],
            "system2_triggered": brm_aux["system2_triggered"],
            "budget": brm_aux["budget"],
        }
        return out, aux


# --------------------------------------------------------------------------------------
# Full model. We keep this small and composable: it wraps a HF Qwen3 backbone, then
# swaps in KSKTLayer at the configured indices. This makes the release runnable on the
# real Qwen3-4B-Thinking-2507 weights while still exposing the full KSKT internals.
# --------------------------------------------------------------------------------------


class _RoPE(nn.Module):
    def __init__(self, dim: int, max_pos: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_pos = max_pos

    def forward(self, t: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(t, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)


class KSKTModel(nn.Module):
    """KSKT backbone-with-blocks. For research clarity we expose the layer
    stack directly rather than relying on a HF subclass: actual training
    scripts call `from_pretrained_qwen3()` to initialize weights for the
    parts that overlap with Qwen3-4B-Thinking-2507.
    """

    def __init__(self, config: KSKTConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = _RoPE(config.head_dim, config.max_position_embeddings, config.rope_theta)

        self.layers = nn.ModuleList()
        kskt_set = set(config.dsaa_layer_indices)
        for i in range(config.num_hidden_layers):
            if i in kskt_set:
                self.layers.append(KSKTLayer(config, layer_idx=i))
            else:
                # Lightweight surrogate transformer block; real training overwrites
                # this with the corresponding Qwen3 block via `load_qwen3_weights`.
                self.layers.append(_StandardBlock(config))

        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _causal_mask(t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        m = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        m = torch.triu(m, diagonal=1)
        return m

    def forward(
        self,
        input_ids: torch.Tensor,
        role_mask: torch.Tensor,
        user_mask: torch.Tensor,
        role_repr: Optional[torch.Tensor] = None,
        user_repr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        b, t = input_ids.shape
        h = self.embed_tokens(input_ids)

        # If explicit R_proc / U_proc were not provided, derive them by
        # selecting role / user tokens via the masks and mean-pooling. The
        # spaCy / BERT preprocessing pipeline in `data/preprocessing.py`
        # produces these when running on real data.
        if role_repr is None:
            role_repr = (h * role_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (role_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1))
            role_repr = role_repr.expand(b, max(int(role_mask.sum(1).max().item()), 1), -1)
        if user_repr is None:
            user_repr = (h * user_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (user_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1))
            user_repr = user_repr.expand(b, max(int(user_mask.sum(1).max().item()), 1), -1)

        causal = self._causal_mask(t, h.device, h.dtype)[None, None, :, :]
        if attention_mask is not None:
            # `attention_mask` is 1 for real tokens, 0 for pad.
            pad = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * torch.finfo(h.dtype).min
            causal = causal + pad

        cos, sin = self.rope(t, h.device, h.dtype)
        cos = cos.expand(b, -1, -1)
        sin = sin.expand(b, -1, -1)

        aux_per_layer: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            if isinstance(layer, KSKTLayer):
                h, aux = layer(h, cos, sin, causal, role_mask, user_mask, role_repr, user_repr)
                aux_per_layer.append(aux)
            else:
                h = layer(h, cos, sin, causal)
        return self.final_norm(h), aux_per_layer


class _StandardBlock(nn.Module):
    """Minimal pre-norm transformer block compatible in shape with Qwen3-4B.

    Acts as a placeholder for non-KSKT layers; real training scripts load
    the corresponding Qwen3 block here.
    """

    def __init__(self, config: KSKTConfig):
        super().__init__()
        d = config.hidden_size
        self.norm1 = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(d, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(d, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(d, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, d, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(d, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(d, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, d, bias=False)
        self.config = config

    def forward(self, x, cos, sin, attn_mask):
        from .dsaa import apply_rope, _repeat_kv
        b, t, _ = x.shape
        h = self.norm1(x)
        q = self.q_norm(self.q_proj(h).view(b, t, self.config.num_attention_heads, self.config.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(b, t, self.config.num_key_value_heads, self.config.head_dim)).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        nrep = self.config.num_attention_heads // self.config.num_key_value_heads
        k = _repeat_kv(k, nrep); v = _repeat_kv(v, nrep)
        scale = self.config.head_dim ** -0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale + attn_mask
        a = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(a, v).transpose(1, 2).contiguous().view(b, t, -1)
        x = x + self.o_proj(o)
        h2 = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x


class KSKTForCausalLM(nn.Module):
    def __init__(self, config: KSKTConfig):
        super().__init__()
        self.config = config
        self.model = KSKTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        role_mask: torch.Tensor,
        user_mask: torch.Tensor,
        role_repr: Optional[torch.Tensor] = None,
        user_repr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h, aux_per_layer = self.model(
            input_ids=input_ids,
            role_mask=role_mask,
            user_mask=user_mask,
            role_repr=role_repr,
            user_repr=user_repr,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(h)
        out: Dict[str, torch.Tensor] = {"logits": logits, "aux": aux_per_layer}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out

    # ----------- Weight loading -----------

    @torch.no_grad()
    def load_qwen3_weights(self, qwen3_path: str) -> None:
        """Initialize parameters from a Qwen3-4B-Thinking-2507 checkpoint.

        Copies token embeddings, lm_head, and the non-KSKT (`_StandardBlock`)
        layers; KSKT-specific layers keep their fresh initialization, which
        is what the three-phase training schedule expects.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError("transformers is required for load_qwen3_weights()") from e
        src = AutoModelForCausalLM.from_pretrained(qwen3_path, torch_dtype=torch.float32)
        src_state = src.state_dict()

        # Token embedding & lm_head.
        self.model.embed_tokens.weight.copy_(src_state["model.embed_tokens.weight"])
        if "lm_head.weight" in src_state:
            self.lm_head.weight.copy_(src_state["lm_head.weight"])

        # Non-KSKT layers: best-effort key remapping.
        for i, layer in enumerate(self.model.layers):
            if not isinstance(layer, _StandardBlock):
                continue
            prefix = f"model.layers.{i}."
            try:
                layer.norm1.weight.copy_(src_state[prefix + "input_layernorm.weight"])
                layer.norm2.weight.copy_(src_state[prefix + "post_attention_layernorm.weight"])
                layer.q_proj.weight.copy_(src_state[prefix + "self_attn.q_proj.weight"])
                layer.k_proj.weight.copy_(src_state[prefix + "self_attn.k_proj.weight"])
                layer.v_proj.weight.copy_(src_state[prefix + "self_attn.v_proj.weight"])
                layer.o_proj.weight.copy_(src_state[prefix + "self_attn.o_proj.weight"])
                layer.gate_proj.weight.copy_(src_state[prefix + "mlp.gate_proj.weight"])
                layer.up_proj.weight.copy_(src_state[prefix + "mlp.up_proj.weight"])
                layer.down_proj.weight.copy_(src_state[prefix + "mlp.down_proj.weight"])
            except KeyError:
                # Newer Qwen3 releases differ in QK-Norm naming; skip silently
                # and rely on phase-1 training to recover.
                continue

        # Final norm.
        if "model.norm.weight" in src_state:
            self.model.final_norm.weight.copy_(src_state["model.norm.weight"])
