"""Camada afim com múltiplas saídas, implementada em Python puro."""

import math
import operator
from random import Random
from typing import cast

from kinera.backends import Backend, resolve_backend
from kinera.backends.base import (
    BackendArray,
    BackendBuffer,
    BackendError,
    release_backend_buffers,
)


class Linear:
    """Calcula y = W x + b sem aplicar função de ativação."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seed: int | None = None,
        *,
        backend: Backend | None = None,
    ) -> None:
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size deve ser um inteiro")
        if isinstance(output_size, bool) or not isinstance(output_size, int):
            raise TypeError("output_size deve ser um inteiro")
        if input_size <= 0:
            raise ValueError("input_size deve ser positivo")
        if output_size <= 0:
            raise ValueError("output_size deve ser positivo")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed deve ser um inteiro ou None")

        self.input_size = input_size
        self.output_size = output_size
        self.backend = resolve_backend(backend)
        generator = Random(seed)
        initial_weights = [
            [float(generator.uniform(-0.1, 0.1)) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        initial_bias = [
            float(generator.uniform(-0.1, 0.1)) for _ in range(output_size)
        ]
        self.weights: BackendArray = self.backend.buffer_from_host(
            initial_weights, shape=(output_size, input_size), dtype="float64"
        )
        self.bias: BackendArray = self.backend.buffer_from_host(
            initial_bias, shape=(output_size,), dtype="float64"
        )

        self.last_input: BackendArray | None = None
        self.output: BackendArray | None = None
        if isinstance(self.weights, BackendBuffer):
            self.grad_weights: BackendArray | None = None
            self.grad_bias: BackendArray | None = None
        else:
            self.grad_weights = [
                [0.0 for _ in range(input_size)] for _ in range(output_size)
            ]
            self.grad_bias = [0.0 for _ in range(output_size)]
        self._has_backward = False

    @staticmethod
    def _validate_number(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} deve ser numérico")
        if not math.isfinite(value):
            raise ValueError(f"{name} deve ser finito")

    @classmethod
    def _validate_backend_array(
        cls, value: object, shape: tuple[int, ...], name: str
    ) -> BackendArray:
        if isinstance(value, BackendBuffer):
            if value.closed:
                raise BackendError(f"{name} buffer has been released")
            if value.shape != shape:
                raise ValueError(f"{name} buffer must have shape {shape}")
            if value.dtype != "float64":
                raise ValueError(f"{name} buffer must use dtype float64")
            return value
        if not isinstance(value, list):
            raise TypeError(f"{name} deve ser uma lista ou buffer do backend")
        if len(shape) == 1:
            if len(value) != shape[0]:
                raise ValueError(f"{name} deve ter tamanho {shape[0]}")
            for index, item in enumerate(value):
                cls._validate_number(item, f"{name}[{index}]")
            return value
        if len(shape) == 2:
            if len(value) != shape[0]:
                raise ValueError(f"{name} deve ter {shape[0]} linhas")
            for row_index, row in enumerate(value):
                if not isinstance(row, list) or len(row) != shape[1]:
                    raise ValueError(f"{name}[{row_index}] deve ter {shape[1]} colunas")
                for column_index, item in enumerate(row):
                    cls._validate_number(item, f"{name}[{row_index}][{column_index}]")
            return value
        raise ValueError("Linear backend arrays support rank 1 or rank 2")

    def _prepare_backend_array(
        self, value: object, shape: tuple[int, ...], name: str
    ) -> BackendArray:
        validated = self._validate_backend_array(value, shape, name)
        if isinstance(validated, BackendBuffer):
            return validated
        return self.backend.buffer_from_host(
            validated, shape=shape, dtype="float64"
        )

    def forward(
        self, inputs: list[float] | BackendBuffer
    ) -> list[float] | BackendBuffer:
        """Calcula y = W x + b, mantendo resultados de device no backend."""
        prepared_weights = self._prepare_backend_array(
            self.weights, (self.output_size, self.input_size), "weights"
        )
        prepared_bias = self._prepare_backend_array(
            self.bias, (self.output_size,), "bias"
        )
        prepared_inputs = self._prepare_backend_array(
            inputs, (self.input_size,), "inputs"
        )
        output = self.backend.linear_forward(
            prepared_weights, prepared_bias, prepared_inputs
        )
        validated_output = self._validate_backend_array(
            output, (self.output_size,), "output"
        )

        self.release_backward_state()
        self.weights = prepared_weights
        self.bias = prepared_bias
        self.last_input = prepared_inputs
        self.output = validated_output
        if isinstance(self.grad_weights, list):
            self.grad_weights = [
                [0.0 for _ in range(self.input_size)]
                for _ in range(self.output_size)
            ]
            self.grad_bias = [0.0 for _ in range(self.output_size)]
        else:
            self.grad_weights = None
            self.grad_bias = None
        self._has_backward = False
        if isinstance(validated_output, list):
            return cast(list[float], validated_output.copy())
        return validated_output

    def backward(
        self, grad_output: list[float] | BackendBuffer
    ) -> list[float] | BackendBuffer:
        """Calcula gradientes dos parâmetros e da entrada."""
        if self.last_input is None or self.output is None:
            raise RuntimeError("backward exige um forward anterior")
        prepared_gradient = self._prepare_backend_array(
            grad_output, (self.output_size,), "grad_output"
        )
        grad_weights, grad_bias, grad_inputs = self.backend.linear_backward(
            self.weights, self.last_input, prepared_gradient
        )
        previous_gradients = (self.grad_weights, self.grad_bias)
        validated_grad_weights = self._validate_backend_array(
            grad_weights, (self.output_size, self.input_size), "grad_weights"
        )
        validated_grad_bias = self._validate_backend_array(
            grad_bias, (self.output_size,), "grad_bias"
        )
        validated_grad_inputs = self._validate_backend_array(
            grad_inputs, (self.input_size,), "grad_inputs"
        )
        release_backend_buffers(
            *(
                previous
                for previous, replacement in zip(
                    previous_gradients,
                    (validated_grad_weights, validated_grad_bias),
                )
                if previous is not replacement
            )
        )
        self.grad_weights = validated_grad_weights
        self.grad_bias = validated_grad_bias
        self._has_backward = True
        if isinstance(validated_grad_inputs, list):
            return cast(list[float], validated_grad_inputs.copy())
        return validated_grad_inputs

    def release_backward_state(self) -> None:
        """Discard temporary backend-owned parameter gradients explicitly."""
        if isinstance(self.grad_weights, BackendBuffer) or isinstance(
            self.grad_bias, BackendBuffer
        ):
            release_backend_buffers(self.grad_weights, self.grad_bias)
            self.grad_weights = None
            self.grad_bias = None
        self._has_backward = False

    def step(self, learning_rate: float) -> None:
        """Atualiza pesos e biases com gradient descent."""
        self._validate_number(learning_rate, "learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate não pode ser negativo")
        if not self._has_backward:
            raise RuntimeError("step exige um backward anterior")
        if self.grad_weights is None or self.grad_bias is None:
            raise RuntimeError("backward não produziu gradientes válidos")

        rate = float(learning_rate)
        self.weights, self.bias = self.backend.linear_step(
            self.weights,
            self.bias,
            self.grad_weights,
            self.grad_bias,
            rate,
        )
        self._has_backward = False
