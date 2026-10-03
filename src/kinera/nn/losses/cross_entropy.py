"""Validated public APIs delegating probability math to a backend."""

from __future__ import annotations

import math
from typing import Sequence

from kinera.backends import Backend, resolve_backend


def _validate_logits(logits: Sequence[object]) -> None:
    if not isinstance(logits, list):
        raise TypeError("logits deve ser uma lista")
    if not logits:
        raise ValueError("logits não pode ser vazio")
    for index, value in enumerate(logits):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"logits[{index}] deve ser numérico")
        if not math.isfinite(value):
            raise ValueError(f"logits[{index}] deve ser finito")


def _validate_target(logits: list[float], target_id: object) -> None:
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        raise TypeError("target_id deve ser inteiro")
    if not 0 <= target_id < len(logits):
        raise ValueError("target_id fora do vocabulário")


def _softmax_validated(
    logits: list[float], *, backend: Backend | None = None
) -> list[float]:
    """Backend operation for a list already validated by the caller."""
    return resolve_backend(backend).softmax(logits)


def _cross_entropy_validated(
    logits: list[float], target_id: int, *, backend: Backend | None = None
) -> float:
    """Backend operation for inputs already validated by the caller."""
    return resolve_backend(backend).cross_entropy(logits, target_id)


def _cross_entropy_gradient_validated(
    logits: list[float],
    target_id: int,
    *,
    backend: Backend | None = None,
) -> list[float]:
    """Backend operation for inputs already validated by the caller."""
    return resolve_backend(backend).cross_entropy_gradient(logits, target_id)


def softmax(
    logits: list[float], *, backend: Backend | None = None
) -> list[float]:
    """Converte logits em probabilidades usando a forma numericamente estável."""
    _validate_logits(logits)
    return resolve_backend(backend).softmax(logits)


def cross_entropy(
    logits: list[float],
    target_id: int,
    *,
    backend: Backend | None = None,
) -> float:
    """Retorna a cross-entropy para um alvo inteiro."""
    _validate_logits(logits)
    _validate_target(logits, target_id)
    return resolve_backend(backend).cross_entropy(logits, target_id)


def cross_entropy_gradient(
    logits: list[float],
    target_id: int,
    *,
    backend: Backend | None = None,
) -> list[float]:
    """Retorna o gradiente de cross-entropy em relação aos logits."""
    _validate_logits(logits)
    _validate_target(logits, target_id)
    return resolve_backend(backend).cross_entropy_gradient(logits, target_id)
