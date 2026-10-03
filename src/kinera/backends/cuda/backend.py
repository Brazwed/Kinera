"""Non-executing CUDA backend scaffold for the next implementation phase."""

from __future__ import annotations

from typing import NoReturn

from kinera.backends.base import BackendUnavailableError


class CUDABackend:
    """Backend contract placeholder that fails closed without production kernels."""

    name = "cuda"
    is_available = False

    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise BackendUnavailableError(
            f"CUDA backend scaffold only; {operation} is not implemented"
        )

    def linear_forward(self, *args, **kwargs):
        self._unavailable("linear_forward")

    def linear_backward(self, *args, **kwargs):
        self._unavailable("linear_backward")

    def linear_step(self, *args, **kwargs):
        self._unavailable("linear_step")

    def embedding_forward(self, *args, **kwargs):
        self._unavailable("embedding_forward")

    def embedding_backward(self, *args, **kwargs):
        self._unavailable("embedding_backward")

    def embedding_step(self, *args, **kwargs):
        self._unavailable("embedding_step")

    def neuron_forward(self, *args, **kwargs):
        self._unavailable("neuron_forward")

    def neuron_backward(self, *args, **kwargs):
        self._unavailable("neuron_backward")

    def neuron_step(self, *args, **kwargs):
        self._unavailable("neuron_step")

    def activation(self, *args, **kwargs):
        self._unavailable("activation")

    def activation_derivative(self, *args, **kwargs):
        self._unavailable("activation_derivative")

    def softmax(self, *args, **kwargs):
        self._unavailable("softmax")

    def cross_entropy(self, *args, **kwargs):
        self._unavailable("cross_entropy")

    def cross_entropy_gradient(self, *args, **kwargs):
        self._unavailable("cross_entropy_gradient")

    def binary_cross_entropy_with_logits(self, *args, **kwargs):
        self._unavailable("binary_cross_entropy_with_logits")

    def binary_cross_entropy_with_logits_gradient(self, *args, **kwargs):
        self._unavailable("binary_cross_entropy_with_logits_gradient")
