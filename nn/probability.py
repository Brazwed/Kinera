"""Probabilidades e perda de classificação em Python puro."""

from __future__ import annotations

import math
from typing import Sequence


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


def _softmax_validated(logits: list[float]) -> list[float]:
    """Aplica softmax a logits já validados; não verifica entradas."""
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _cross_entropy_validated(logits: list[float], target_id: int) -> float:
    """Calcula cross-entropy após validação de logits e target."""
    maximum = max(logits)
    log_sum_exp = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return log_sum_exp - logits[target_id]


def _cross_entropy_gradient_validated(
    logits: list[float],
    target_id: int,
) -> list[float]:
    """Calcula o gradiente após validação de logits e target."""
    probabilities = _softmax_validated(logits)
    probabilities[target_id] -= 1.0
    return probabilities


def softmax(logits: list[float]) -> list[float]:
    """Converte logits em probabilidades usando a forma numericamente estável."""
    _validate_logits(logits)
    return _softmax_validated(logits)


def cross_entropy(logits: list[float], target_id: int) -> float:
    """Retorna a cross-entropy para um alvo inteiro."""
    _validate_logits(logits)
    _validate_target(logits, target_id)
    return _cross_entropy_validated(logits, target_id)


def cross_entropy_gradient(logits: list[float], target_id: int) -> list[float]:
    """Retorna o gradiente de cross-entropy em relação aos logits."""
    _validate_logits(logits)
    _validate_target(logits, target_id)
    return _cross_entropy_gradient_validated(logits, target_id)
