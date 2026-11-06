"""Numerical helpers shared by legacy metrics consumers."""

from collections.abc import Sequence
from typing import Literal, TypeVar


Number = TypeVar("Number", int, float)
DeltaOrientation = Literal["base - candidate", "candidate - base"]


def mean_from_total(total: float, count: int) -> float:
    """Divide an accumulated total by its known-positive sample count.

    This deliberately uses direct division rather than ``safe_ratio``: callers
    must preserve the existing ``ZeroDivisionError`` if an empty set reaches
    this operation.
    """
    return total / count


def oriented_delta(
    base: Number,
    candidate: Number,
    *,
    orientation: DeltaOrientation,
) -> Number:
    """Subtract paired scalar observations in the explicitly requested direction."""
    if orientation == "base - candidate":
        return base - candidate
    if orientation == "candidate - base":
        return candidate - base
    raise ValueError(f"orientação de delta inválida: {orientation!r}")


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return the support and arithmetic mean of an ordered observation set.

    Empty input has no defined mean. For nonempty input this intentionally
    keeps the legacy ``sum(values) / count`` order and arithmetic.
    """
    count = len(values)
    return {
        "count": count,
        "mean": None if count == 0 else sum(values) / count,
    }


def summarize_deltas(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize example-level deltas and their strict improvement/regression rates.

    Positive and negative counts use strict comparisons, so exact ties belong
    to neither category. The population here is the supplied observations;
    callers must not use this for a different unit such as seeds.
    """
    summary = summarize_values(values)
    count = len(values)
    positive_count = sum(value > 0 for value in values)
    negative_count = sum(value < 0 for value in values)
    return {
        **summary,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": (
            None if count == 0 else safe_ratio(positive_count, count)
        ),
        "negative_fraction": (
            None if count == 0 else safe_ratio(negative_count, count)
        ),
    }


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    zero_denominator: float = 0.0,
) -> float:
    """Return a ratio, preserving the caller's legacy zero-denominator value."""
    return zero_denominator if denominator == 0.0 else numerator / denominator
