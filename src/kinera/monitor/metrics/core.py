"""Numerical helpers shared by legacy metrics consumers."""


def mean_from_total(total: float, count: int) -> float:
    """Divide an accumulated total by its known-positive sample count.

    This deliberately uses direct division rather than ``safe_ratio``: callers
    must preserve the existing ``ZeroDivisionError`` if an empty set reaches
    this operation.
    """
    return total / count


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    zero_denominator: float = 0.0,
) -> float:
    """Return a ratio, preserving the caller's legacy zero-denominator value."""
    return zero_denominator if denominator == 0.0 else numerator / denominator
