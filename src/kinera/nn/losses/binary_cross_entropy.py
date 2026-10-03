"""Validated binary cross-entropy API routed through the backend contract."""

from __future__ import annotations

import math

from kinera.backends import Backend, resolve_backend


def binary_cross_entropy_with_logits(
    logit: float, target: float, *, backend: Backend | None = None
) -> float:
    """Calcula BCE diretamente dos logits com estabilidade numérica."""
    target_value = _validate_binary_target(target)
    _validate_number(logit, "logit")
    return resolve_backend(backend).binary_cross_entropy_with_logits(logit, target_value)


def binary_cross_entropy_with_logits_gradient(
    logit: float, target: float, *, backend: Backend | None = None
) -> float:
    """Calcula dBCE/dlogit."""
    target_value = _validate_binary_target(target)
    _validate_number(logit, "logit")
    return resolve_backend(backend).binary_cross_entropy_with_logits_gradient(logit, target_value)


def _validate_binary_target(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("target de fronteira deve ser numérico")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("target de fronteira deve estar entre 0 e 1")
    return float(value)


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito")
