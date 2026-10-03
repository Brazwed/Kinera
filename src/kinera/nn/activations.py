"""Public activation functions routed through the selected backend."""

from __future__ import annotations

import math

from kinera.backends import Backend, resolve_backend


def relu(value: float, *, backend: Backend | None = None) -> float:
    """Retorna o máximo entre o valor e zero."""
    return resolve_backend(backend).activation("relu", value)


def relu_derivative(value: float, *, backend: Backend | None = None) -> float:
    """Retorna a derivada da ReLU, usando zero no ponto não diferenciável."""
    return resolve_backend(backend).activation_derivative("relu", value)


def linear(value: float, *, backend: Backend | None = None) -> float:
    """Ativação identidade."""
    return resolve_backend(backend).activation("linear", value)


def linear_derivative(value: float, *, backend: Backend | None = None) -> float:
    """Derivada constante da ativação identidade."""
    return resolve_backend(backend).activation_derivative("linear", value)


def leaky_relu(
    value: float,
    negative_slope: float = 0.01,
    *,
    backend: Backend | None = None,
) -> float:
    """Mantém gradiente pequeno e não nulo no lado negativo."""
    _validate_negative_slope(negative_slope)
    return resolve_backend(backend).activation("leaky_relu", value, negative_slope)


def leaky_relu_derivative(
    value: float,
    negative_slope: float = 0.01,
    *,
    backend: Backend | None = None,
) -> float:
    """Retorna a derivada da LeakyReLU."""
    _validate_negative_slope(negative_slope)
    return resolve_backend(backend).activation_derivative(
        "leaky_relu", value, negative_slope
    )


def _validate_negative_slope(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("negative_slope deve ser numérico")
    if not math.isfinite(value):
        raise ValueError("negative_slope deve ser finito")
    if not 0.0 < value <= 1.0:
        raise ValueError("negative_slope deve estar entre 0 e 1")
