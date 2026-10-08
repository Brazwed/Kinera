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
        normalized_shape = self._normalize_shape(shape)
        if dtype == "float64":
            copied, observed_shape = self._copy_host_tree(values)
        elif dtype == "int32":
            copied, observed_shape = self._copy_int32_tree(values)
        else:
            raise ValueError("CPU backend supports buffer dtypes 'float64' and 'int32'")
        if observed_shape != normalized_shape:
            raise ValueError(
                f"shape {normalized_shape} does not match host data shape {observed_shape}"
            )
        return cast(HostArray, copied)

    def buffer_zeros(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> HostArray:
        """Create a fresh zero-filled CPU list buffer for a supported shape."""
        normalized_shape = self._normalize_shape(shape)
        if dtype not in {"float64", "int32"}:
            raise ValueError("CPU backend supports buffer dtypes 'float64' and 'int32'")
        if len(normalized_shape) not in {1, 2}:
            raise ValueError("CPU buffer_zeros supports rank 1 or rank 2")

        zero: float | int = 0.0 if dtype == "float64" else 0
        if len(normalized_shape) == 1:
            return cast(HostArray, [zero for _ in range(normalized_shape[0])])
        rows, columns = normalized_shape
        return cast(
            HostArray,
            [[zero for _ in range(columns)] for _ in range(rows)],
        )

    def buffer_to_host(self, buffer: BackendArray) -> HostArray:
        """Return an independent Python-list copy of CPU-owned buffer data."""
        if not isinstance(buffer, list):
            raise TypeError("CPUBackend can export only CPU-owned list buffers")
        if self._is_int_tree(buffer):
            copied, shape = self._copy_int32_tree(buffer)
        else:
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

    @staticmethod
    def _is_int_tree(value: object) -> bool:
        if isinstance(value, list):
            return bool(value) and all(CPUBackend._is_int_tree(child) for child in value)
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    def _copy_int32_tree(cls, value: object) -> tuple[object, tuple[int, ...]]:
        if isinstance(value, list):
            if not value:
                raise ValueError("host data cannot contain an empty dimension")
            children = [cls._copy_int32_tree(child) for child in value]
            child_shape = children[0][1]
            if any(shape != child_shape for _, shape in children[1:]):
                raise ValueError("host data must be rectangular")
            return [child for child, _ in children], (len(value), *child_shape)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("int32 host data must contain only integer values")
        if value < -(2**31) or value > (2**31) - 1:
            raise ValueError("int32 host data is outside the signed 32-bit range")
        return value, ()

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

    def hybrid_representation_forward(
        self, char_vectors, feature_vectors, syllable_vectors
    ):
        """Concatenate aligned character, feature, and syllable channels."""
        matrices = (char_vectors, feature_vectors, syllable_vectors)
        if any(not isinstance(matrix, list) for matrix in matrices):
            raise TypeError("CPU hybrid representation inputs must be Python lists")
        row_counts = tuple(len(matrix) for matrix in matrices)
        if len(set(row_counts)) != 1:
            raise ValueError("hybrid representation matrices must have equal row counts")
        if row_counts[0] == 0:
            return []

        copied_matrices = [self._copy_host_tree(matrix) for matrix in matrices]
        shapes = [shape for _, shape in copied_matrices]
        if any(len(shape) != 2 for shape in shapes):
            raise ValueError("hybrid representation inputs must have rank 2")
        if any(shape[0] != row_counts[0] for shape in shapes):
            raise ValueError("hybrid representation matrices must be rectangular")
        char_values = cast(list[list[float]], copied_matrices[0][0])
        feature_values = cast(list[list[float]], copied_matrices[1][0])
        syllable_values = cast(list[list[float]], copied_matrices[2][0])
        return [
            char_row + feature_row + syllable_row
            for char_row, feature_row, syllable_row in zip(
                char_values, feature_values, syllable_values
            )
        ]

    def mean_pool_forward(self, vectors):
        """Average each feature by sequentially visiting context positions."""
        if not isinstance(vectors, list):
            raise TypeError("CPU mean pooling input must be a Python list")
        copied, shape = self._copy_host_tree(vectors)
        if len(shape) != 2:
            raise ValueError("mean pooling input must have rank 2")
        context_size, feature_dim = shape
        matrix = cast(list[list[float]], copied)
        return [
            sum(vector[dimension] for vector in matrix) / context_size
            for dimension in range(feature_dim)
        ]

    def argmax(self, logits):
        """Return the first index holding the maximum finite host logit."""
        if not isinstance(logits, list):
            raise TypeError("CPU argmax input must be a Python list")
        if not logits:
            raise ValueError("CPU argmax input cannot be empty")
        for index, value in enumerate(logits):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"logits[{index}] must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"logits[{index}] must be finite")
        return max(range(len(logits)), key=logits.__getitem__)

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
