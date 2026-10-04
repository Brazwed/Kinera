"""Stateless CPU backend preserving the original Python arithmetic."""

from __future__ import annotations

import math
from typing import cast

from kinera.backends.base import BackendArray, HostArray
from kinera.backends.cpu.ops import activations, embedding, linear, probability


class CPUBackend:
    """Python-standard-library oracle for the model's numerical operations."""

    name = "cpu"

    def buffer_from_host(
        self,
        values: HostArray,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> HostArray:
        """Copy a validated host array while preserving the CPU list oracle."""
        if dtype != "float64":
            raise ValueError("CPU backend supports buffer dtype='float64' in Phase 9")
        normalized_shape = self._normalize_shape(shape)
        copied, observed_shape = self._copy_host_tree(values)
        if observed_shape != normalized_shape:
            raise ValueError(
                f"shape {normalized_shape} does not match host data shape {observed_shape}"
            )
        return cast(HostArray, copied)

    def buffer_to_host(self, buffer: BackendArray) -> HostArray:
        """Return an independent Python-list copy of CPU-owned buffer data."""
        if not isinstance(buffer, list):
            raise TypeError("CPUBackend can export only CPU-owned list buffers")
        copied, shape = self._copy_host_tree(buffer)
        self._normalize_shape(shape)
        return cast(HostArray, copied)

    @staticmethod
    def _normalize_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if not isinstance(shape, tuple) or not shape:
            raise ValueError("shape must be a non-empty tuple")
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"shape[{axis}] must be an integer")
            if dimension <= 0:
                raise ValueError(f"shape[{axis}] must be positive")
        return shape

    @classmethod
    def _copy_host_tree(cls, value: object) -> tuple[object, tuple[int, ...]]:
        if isinstance(value, list):
            if not value:
                raise ValueError("host data cannot contain an empty dimension")
            children = [cls._copy_host_tree(child) for child in value]
            child_shape = children[0][1]
            if any(shape != child_shape for _, shape in children[1:]):
                raise ValueError("host data must be rectangular")
            return [child for child, _ in children], (len(value), *child_shape)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("host data must contain only numeric values")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("host data must contain only finite values")
        return number, ()

    def linear_forward(self, weights, bias, inputs):
        return linear.forward(weights, bias, inputs)

    def linear_backward(self, weights, inputs, grad_output):
        return linear.backward(weights, inputs, grad_output)

    def linear_step(self, weights, bias, grad_weights, grad_bias, learning_rate):
        return linear.step(weights, bias, grad_weights, grad_bias, learning_rate)

    def embedding_forward(self, weights, ids):
        return embedding.forward(weights, ids)

    def embedding_backward(self, grad_weights, ids, grad_output):
        return embedding.backward(grad_weights, ids, grad_output)

    def embedding_step(self, weights, grad_weights, learning_rate):
        return embedding.step(weights, grad_weights, learning_rate)

    def neuron_forward(self, weights, bias, inputs, activation, negative_slope):
        z = sum(input_value * weight for input_value, weight in zip(inputs, weights)) + bias
        return z, activations.forward(activation, z, negative_slope)

    def neuron_backward(
        self, weights, inputs, z, grad_output, activation, negative_slope
    ):
        grad_z = float(grad_output) * activations.derivative(
            activation, z, negative_slope
        )
        grad_weights = [grad_z * input_value for input_value in inputs]
        grad_bias = grad_z
        grad_inputs = [grad_z * weight for weight in weights]
        return grad_weights, grad_bias, grad_inputs

    def neuron_step(self, weights, bias, grad_weights, grad_bias, learning_rate):
        weights = [
            weight - float(learning_rate) * gradient
            for weight, gradient in zip(weights, grad_weights)
        ]
        bias -= float(learning_rate) * grad_bias
        return weights, bias

    def activation(self, name, value, negative_slope=0.01):
        return activations.forward(name, value, negative_slope)

    def activation_derivative(self, name, value, negative_slope=0.01):
        return activations.derivative(name, value, negative_slope)

    def softmax(self, logits):
        return probability.softmax(logits)

    def cross_entropy(self, logits, target_id):
        return probability.cross_entropy(logits, target_id)

    def cross_entropy_gradient(self, logits, target_id):
        return probability.cross_entropy_gradient(logits, target_id)

    def binary_cross_entropy_with_logits(self, logit, target):
        return probability.binary_cross_entropy_with_logits(logit, target)

    def binary_cross_entropy_with_logits_gradient(self, logit, target):
        return probability.binary_cross_entropy_with_logits_gradient(logit, target)
