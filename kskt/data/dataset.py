"""KSKT dialogue dataset.

Expects JSONL files with the structure produced by our data construction
pipeline (Appendix C.1):

    {"role": "<character description>", "history": [{"speaker":"user","text":"..."},
     {"speaker":"assistant","text":"..."}, ...], "response": "<gold reply>"}

The dataset emits tensors of input_ids, role_mask, user_mask, and labels
(with -100 on non-response tokens) ready for KSKTForCausalLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..config import KSKTConfig


@dataclass
class _Example:
    input_ids: List[int]
    role_mask: List[int]
    user_mask: List[int]
    labels: List[int]


class KSKTDialogueDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        config: KSKTConfig,
        max_length: Optional[int] = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length or config.sequence_length

        self._open_role = self._marker_id(config.role_marker_open)
        self._close_role = self._marker_id(config.role_marker_close)
        self._open_user = self._marker_id(config.user_marker_open)
        self._close_user = self._marker_id(config.user_marker_close)

        self.examples: List[_Example] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ex = self._encode(row)
                if ex is not None:
                    self.examples.append(ex)

    def _marker_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer cannot encode marker '{token}'.")
        return ids[0]

    def _encode(self, row: Dict) -> Optional[_Example]:
        role_text = row.get("role", "")
        history = row.get("history", [])
        response = row.get("response", "")
        if not response:
            return None

        # Build the dialogue string with explicit role/user markers so masks
        # are unambiguous downstream.
        parts: List[str] = [
            self.config.role_marker_open + role_text + self.config.role_marker_close
        ]
        for turn in history:
            if turn["speaker"] == "user":
                parts.append(self.config.user_marker_open + turn["text"] + self.config.user_marker_close)
            else:
                parts.append(turn["text"])

        prompt = "\n".join(parts) + "\n"
        full = prompt + response
        ids = self.tokenizer.encode(full, add_special_tokens=False)
        ids = ids[: self.max_length]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        n_prompt = min(len(prompt_ids), len(ids))

        labels = [-100] * n_prompt + ids[n_prompt:]
        labels = labels[: len(ids)]

        role_mask, user_mask = self._compute_masks(ids)
        return _Example(ids, role_mask, user_mask, labels)

    def _compute_masks(self, ids: List[int]):
        role_mask, user_mask = [], []
        in_role = False
        in_user = False
        for tok in ids:
            if tok == self._open_role:
                in_role = True
                role_mask.append(0); user_mask.append(0); continue
            if tok == self._close_role:
                in_role = False
                role_mask.append(0); user_mask.append(0); continue
            if tok == self._open_user:
                in_user = True
                role_mask.append(0); user_mask.append(0); continue
            if tok == self._close_user:
                in_user = False
                role_mask.append(0); user_mask.append(0); continue
            role_mask.append(1 if in_role else 0)
            user_mask.append(1 if in_user else 0)
        return role_mask, user_mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "role_mask": torch.tensor(ex.role_mask, dtype=torch.float32),
            "user_mask": torch.tensor(ex.user_mask, dtype=torch.float32),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
        }


def collate_kskt(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length examples."""
    max_len = max(int(x["input_ids"].size(0)) for x in batch)
    out: Dict[str, torch.Tensor] = {}
    for key in ("input_ids", "role_mask", "user_mask", "labels"):
        pad_val = pad_id if key == "input_ids" else (-100 if key == "labels" else 0)
        dtype = batch[0][key].dtype
        padded = torch.full((len(batch), max_len), pad_val, dtype=dtype)
        for i, x in enumerate(batch):
            n = x[key].size(0)
            padded[i, :n] = x[key]
        out[key] = padded
    out["attention_mask"] = (out["input_ids"] != pad_id).to(torch.float32)
    return out
