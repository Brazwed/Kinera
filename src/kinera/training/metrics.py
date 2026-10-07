"""Shared target validation and safe metric arithmetic."""

from __future__ import annotations

import math
from typing import Any

from kinera.monitor.metrics.core import safe_ratio


def _multitask_targets(example: dict[str, Any]) -> dict[str, int | float]:
    target_char = example.get("target_char_id")
    target_syllable = example.get("target_syllable_id")
    target_boundary = example.get("target_is_syllable_end")
    if isinstance(target_char, bool) or not isinstance(target_char, int):
        raise ValueError("target_char_id deve ser inteiro")
    if isinstance(target_syllable, bool) or not isinstance(target_syllable, int):
        raise ValueError("target_syllable_id deve ser inteiro")
    if isinstance(target_boundary, bool):
        target_boundary = float(target_boundary)
    if not isinstance(target_boundary, (int, float)) or not math.isfinite(target_boundary):
        raise ValueError("target_is_syllable_end deve ser numérico e finito")
    return {
        "char": target_char,
        "syllable": target_syllable,
        "boundary": float(target_boundary),
    }


_safe_ratio = safe_ratio
