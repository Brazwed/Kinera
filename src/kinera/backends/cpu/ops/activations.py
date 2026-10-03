"""Scalar CPU activations used by the neural backend."""

import math


_SUPPORTED = {"relu", "leaky_relu", "linear"}


def forward(name: str, value: float, negative_slope: float = 0.01) -> float:
    if name == "relu":
        return max(0.0, value)
    if name == "leaky_relu":
        _validate_negative_slope(negative_slope)
        return value if value > 0.0 else negative_slope * value
    if name == "linear":
        return value
    raise ValueError(f"activation desconhecida: {name}")


def derivative(name: str, value: float, negative_slope: float = 0.01) -> float:
    if name == "relu":
        return 1.0 if value > 0.0 else 0.0
    if name == "leaky_relu":
        _validate_negative_slope(negative_slope)
        return 1.0 if value > 0.0 else negative_slope
    if name == "linear":
        return 1.0
    raise ValueError(f"activation desconhecida: {name}")


def _validate_negative_slope(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("negative_slope deve ser numérico")
    if not math.isfinite(value):
        raise ValueError("negative_slope deve ser finito")
    if not 0.0 < value <= 1.0:
        raise ValueError("negative_slope deve estar entre 0 e 1")
