"""Funções de ativação do núcleo neural."""

import math


def relu(value: float) -> float:
    """Retorna o máximo entre o valor e zero."""
    return max(0.0, value)


def relu_derivative(value: float) -> float:
    """Retorna a derivada da ReLU, usando zero no ponto não diferenciável."""
    return 1.0 if value > 0.0 else 0.0


def linear(value: float) -> float:
    """Ativação identidade."""
    return value


def linear_derivative(value: float) -> float:
    """Derivada constante da ativação identidade."""
    return 1.0


def leaky_relu(value: float, negative_slope: float = 0.01) -> float:
    """Mantém gradiente pequeno e não nulo no lado negativo."""
    _validate_negative_slope(negative_slope)
    return value if value > 0.0 else negative_slope * value


def leaky_relu_derivative(
    value: float,
    negative_slope: float = 0.01,
) -> float:
    """Retorna a derivada da LeakyReLU."""
    _validate_negative_slope(negative_slope)
    return 1.0 if value > 0.0 else negative_slope


def _validate_negative_slope(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("negative_slope deve ser numérico")
    if not math.isfinite(value):
        raise ValueError("negative_slope deve ser finito")
    if not 0.0 < value <= 1.0:
        raise ValueError("negative_slope deve estar entre 0 e 1")
