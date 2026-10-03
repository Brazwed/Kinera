"""Minimal operations consumed by Kinera neural components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Base error for backend execution failures."""


class BackendUnavailableError(BackendError):
    """Raised when a backend scaffold cannot execute an operation yet."""


@runtime_checkable
class Backend(Protocol):
    """Hardware-independent contract for the operations used by Kinera."""

    name: str

    def linear_forward(
        self, weights: list[list[float]], bias: list[float], inputs: list[float]
    ) -> list[float]: ...

    def linear_backward(
        self, weights: list[list[float]], inputs: list[float], grad_output: list[float]
    ) -> tuple[list[list[float]], list[float], list[float]]: ...

    def linear_step(
        self,
        weights: list[list[float]],
        bias: list[float],
        grad_weights: list[list[float]],
        grad_bias: list[float],
        learning_rate: float,
    ) -> tuple[list[list[float]], list[float]]: ...

    def embedding_forward(
        self, weights: list[list[float]], ids: list[int]
    ) -> list[list[float]]: ...

    def embedding_backward(
        self,
        grad_weights: list[list[float]],
        ids: list[int],
        grad_output: list[list[float]],
    ) -> list[list[float]]: ...

    def embedding_step(
        self,
        weights: list[list[float]],
        grad_weights: list[list[float]],
        learning_rate: float,
    ) -> list[list[float]]: ...

    def neuron_forward(
        self,
        weights: list[float],
        bias: float,
        inputs: list[float],
        activation: str,
        negative_slope: float,
    ) -> tuple[float, float]: ...

    def neuron_backward(
        self,
        weights: list[float],
        inputs: list[float],
        z: float,
        grad_output: float,
        activation: str,
        negative_slope: float,
    ) -> tuple[list[float], float, list[float]]: ...

    def neuron_step(
        self,
        weights: list[float],
        bias: float,
        grad_weights: list[float],
        grad_bias: float,
        learning_rate: float,
    ) -> tuple[list[float], float]: ...

    def activation(self, name: str, value: float, negative_slope: float = 0.01) -> float: ...

    def activation_derivative(
        self, name: str, value: float, negative_slope: float = 0.01
    ) -> float: ...

    def softmax(self, logits: list[float]) -> list[float]: ...

    def cross_entropy(self, logits: list[float], target_id: int) -> float: ...

    def cross_entropy_gradient(
        self, logits: list[float], target_id: int
    ) -> list[float]: ...

    def binary_cross_entropy_with_logits(self, logit: float, target: float) -> float: ...

    def binary_cross_entropy_with_logits_gradient(
        self, logit: float, target: float
    ) -> float: ...
