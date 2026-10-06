"""Minimal operations consumed by Kinera neural components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BackendError(RuntimeError):
    """Base error for backend execution failures."""


class BackendUnavailableError(BackendError):
    """Raised when a backend scaffold cannot execute an operation yet."""


@runtime_checkable
class BackendBuffer(Protocol):
    """Hardware-neutral read-only metadata for a backend-owned array."""

    @property
    def device(self) -> str: ...

    @property
    def device_id(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> str: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def layout(self) -> str: ...

    @property
    def owns_memory(self) -> bool: ...

    @property
    def closed(self) -> bool: ...


FloatArray = list[float] | list[list[float]]
IntArray = list[int] | list[list[int]]
HostArray = FloatArray | IntArray
BackendArray = HostArray | BackendBuffer


def release_backend_buffers(*buffers: object) -> None:
    """Explicitly close distinct owned backend buffers when they support it."""
    seen: set[int] = set()
    for buffer in buffers:
        identity = id(buffer)
        if identity in seen:
            continue
        seen.add(identity)
        if not getattr(buffer, "owns_memory", False) or getattr(buffer, "closed", False):
            continue
        close = getattr(buffer, "close", None)
        if callable(close):
            close()


@runtime_checkable
class Backend(Protocol):
    """Core hardware-independent contract shared by compatible backends."""

    name: str

    def buffer_zeros(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> BackendArray: ...

    def buffer_from_host(
        self,
        values: HostArray,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> BackendArray: ...

    def buffer_to_host(self, buffer: BackendArray) -> HostArray: ...

    def linear_forward(
        self, weights: BackendArray, bias: BackendArray, inputs: BackendArray
    ) -> BackendArray: ...

    def linear_backward(
        self, weights: BackendArray, inputs: BackendArray, grad_output: BackendArray
    ) -> tuple[BackendArray, BackendArray, BackendArray]: ...

    def linear_step(
        self,
        weights: BackendArray,
        bias: BackendArray,
        grad_weights: BackendArray,
        grad_bias: BackendArray,
        learning_rate: float,
    ) -> tuple[BackendArray, BackendArray]: ...

    def embedding_forward(
        self, weights: BackendArray, ids: list[int] | BackendBuffer
    ) -> BackendArray: ...

    def embedding_backward(
        self,
        grad_weights: BackendArray,
        ids: list[int] | BackendBuffer,
        grad_output: BackendArray,
    ) -> BackendArray: ...

    def embedding_step(
        self,
        weights: BackendArray,
        grad_weights: BackendArray,
        learning_rate: float,
    ) -> BackendArray: ...

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
