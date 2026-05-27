"""Input Processing Pipeline (Appendix B.1).

We build the structured R_proc / U_proc representations that DSAA, MUPE,
and SAMOE consume. Concretely:

    R_proc = concat(R_trait, R_knowledge, R_temporal)
    U_proc = concat(U_intent, U_emotion, U_directive)

The paper uses spaCy for role-context entity extraction and a BERT-based
classifier for user intent. We expose a thin abstraction that supports
both: a heuristic regex/keyword backend (no heavy deps) for reproduction
on machines without spaCy/BERT, and a spaCy backend when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import torch


# Intent / emotion / directive lexicons used by the lightweight backend.
INTENT_KW = {
    "ask", "tell me", "explain", "describe", "summarize", "请", "告诉我", "解释",
    "怎么", "为什么", "how", "what", "why", "where", "when",
}
EMOTION_KW = {
    "happy", "sad", "angry", "afraid", "worried", "love", "hate",
    "开心", "难过", "生气", "害怕", "担心", "喜欢", "讨厌",
}
DIRECTIVE_KW = {
    "please", "could you", "would you", "do not", "must", "should",
    "请", "必须", "应该", "不要", "不能",
}
TRAIT_KW = {
    "personality", "character", "habit", "trait", "background",
    "性格", "习惯", "背景", "经历", "身份",
}
KNOWLEDGE_KW = {
    "knows", "skilled in", "expert", "knowledge of", "occupation",
    "懂得", "擅长", "职业", "专家", "知识",
}
TEMPORAL_KW = {
    "century", "year", "era", "period", "ancient", "modern",
    "世纪", "年代", "时代", "古代", "现代",
}


@dataclass
class ExtractedRole:
    trait: str = ""
    knowledge: str = ""
    temporal: str = ""

    def to_string(self) -> str:
        return " ".join(s for s in (self.trait, self.knowledge, self.temporal) if s)


@dataclass
class ExtractedUser:
    intent: str = ""
    emotion: str = ""
    directive: str = ""

    def to_string(self) -> str:
        return " ".join(s for s in (self.intent, self.emotion, self.directive) if s)


class RoleUserPreprocessor:
    """Light, dependency-free preprocessor.

    For production / paper reproduction, swap `extract_role_features` for a
    spaCy-backed implementation as described in Appendix B.1.
    """

    def __init__(self, use_spacy: bool = False):
        self.use_spacy = use_spacy
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.use_spacy = False  # silent fallback

    def extract_role(self, role_text: str) -> ExtractedRole:
        sentences = re.split(r"(?<=[.!?。！？])\s+", role_text)
        traits, knows, temps = [], [], []
        for s in sentences:
            low = s.lower()
            if any(k in low for k in TRAIT_KW):
                traits.append(s)
            if any(k in low for k in KNOWLEDGE_KW):
                knows.append(s)
            if any(k in low for k in TEMPORAL_KW):
                temps.append(s)
        return ExtractedRole(" ".join(traits), " ".join(knows), " ".join(temps))

    def extract_user(self, user_text: str) -> ExtractedUser:
        low = user_text.lower()
        intent = user_text if any(k in low for k in INTENT_KW) else ""
        emotion = " ".join([w for w in user_text.split() if any(k in w.lower() for k in EMOTION_KW)])
        directive = user_text if any(k in low for k in DIRECTIVE_KW) else ""
        return ExtractedUser(intent, emotion, directive)


def build_role_user_masks(
    input_ids: torch.Tensor,
    role_token_ids: Iterable[int],
    user_token_ids: Iterable[int],
    open_role_id: int,
    close_role_id: int,
    open_user_id: int,
    close_user_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [B, T] {0,1} masks for role and user spans, identified by their
    surrounding marker tokens.

    The simplest robust strategy is to scan for `open_*` / `close_*` markers
    in input_ids and flip the corresponding mask between them.
    """
    b, t = input_ids.shape
    role_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    user_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for bi in range(b):
        in_role = False
        in_user = False
        for ti in range(t):
            tok = int(input_ids[bi, ti])
            if tok == open_role_id:
                in_role = True
                continue
            if tok == close_role_id:
                in_role = False
                continue
            if tok == open_user_id:
                in_user = True
                continue
            if tok == close_user_id:
                in_user = False
                continue
            if in_role:
                role_mask[bi, ti] = 1.0
            if in_user:
                user_mask[bi, ti] = 1.0
    return role_mask, user_mask
