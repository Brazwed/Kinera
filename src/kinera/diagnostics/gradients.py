"""Numerical gradient checking utilities for backend equivalence tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class GradientCheckResult:
    """Elementwise and aggregate comparison of analytic/numeric gradients."""

    max_absolute_error: float
    max_relative_error: float
    passed: bool


def finite_difference_gradient(
    objective: Callable[[Sequence[float]], float],
    parameters: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    """Compute a central-difference gradient without mutating caller data."""
    if not callable(objective):
        raise TypeError("objective deve ser chamável")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon deve ser numérico")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon deve ser finito e positivo")
    values = list(parameters)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise TypeError("parameters deve conter apenas números")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("parameters deve conter apenas valores finitos")
    gradient: list[float] = []
    for index, value in enumerate(values):
        plus = values.copy()
        minus = values.copy()
        plus[index] = value + epsilon
        minus[index] = value - epsilon
        upper = objective(plus)
        lower = objective(minus)
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (upper, lower)):
            raise TypeError("objective deve retornar um número")
        if not math.isfinite(upper) or not math.isfinite(lower):
            raise ValueError("objective deve retornar valores finitos")
        gradient.append((upper - lower) / (2.0 * epsilon))
    return gradient


def compare_gradients(
    analytic: Sequence[float],
    numeric: Sequence[float],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
) -> GradientCheckResult:
    """Compare gradients elementwise using absolute OR relative tolerance."""
    if len(analytic) != len(numeric):
        raise ValueError("gradients devem ter o mesmo comprimento")
    tolerances = (absolute_tolerance, relative_tolerance)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in tolerances):
        raise TypeError("tolerâncias devem ser numéricas")
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("tolerâncias devem ser finitas e não negativas")
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    passed = True
    for expected, actual in zip(analytic, numeric, strict=True):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (expected, actual)):
            raise TypeError("gradients devem conter apenas números")
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError("gradients devem conter apenas valores finitos")
        absolute = abs(expected - actual)
        relative = absolute / max(abs(expected), abs(actual), 1e-15)
        abs_errors.append(absolute)
        rel_errors.append(relative)
        passed = passed and (
            absolute <= absolute_tolerance or relative <= relative_tolerance
        )
    return GradientCheckResult(
        max_absolute_error=max(abs_errors, default=0.0),
        max_relative_error=max(rel_errors, default=0.0),
        passed=passed,
    )


__all__ = ["GradientCheckResult", "compare_gradients", "finite_difference_gradient"]
