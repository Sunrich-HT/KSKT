"""Data side of KSKT: role/user extraction and dialogue formatting."""

from .preprocessing import RoleUserPreprocessor, build_role_user_masks
from .dataset import KSKTDialogueDataset, collate_kskt

__all__ = [
    "RoleUserPreprocessor",
    "build_role_user_masks",
    "KSKTDialogueDataset",
    "collate_kskt",
]
