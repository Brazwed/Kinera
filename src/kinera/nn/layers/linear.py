"""Camada afim com múltiplas saídas, implementada em Python puro."""

import math
import operator
from random import Random

from kinera.backends import Backend, resolve_backend


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

        self.backend = resolve_backend(backend)
        generator = Random(seed)
        self.weights = [
            [float(generator.uniform(-0.1, 0.1)) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.bias = [float(generator.uniform(-0.1, 0.1)) for _ in range(output_size)]

        self.last_input: list[float] | None = None
        self.output: list[float] | None = None
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

    def forward(self, inputs: list[float]) -> list[float]:
        """Calcula o vetor de saídas afins."""
        if not isinstance(inputs, list):
            raise TypeError("inputs deve ser uma lista")
        if len(inputs) != len(self.weights[0]):
            raise ValueError("inputs deve ter o mesmo tamanho das colunas")
        for index, value in enumerate(inputs):
            self._validate_number(value, f"inputs[{index}]")

        self.last_input = [float(value) for value in inputs]
        self.output = self.backend.linear_forward(
            self.weights, self.bias, self.last_input
        )
        self.grad_weights = [
            [0.0 for _ in row] for row in self.weights
        ]
        self.grad_bias = [0.0 for _ in self.bias]
        self._has_backward = False
        return self.output.copy()

    def backward(self, grad_output: list[float]) -> list[float]:
        """Calcula gradientes dos parâmetros e da entrada."""
        if self.last_input is None or self.output is None:
            raise RuntimeError("backward exige um forward anterior")
        if not isinstance(grad_output, list):
            raise TypeError("grad_output deve ser uma lista")
        if len(grad_output) != len(self.weights):
            raise ValueError("grad_output deve ter o mesmo tamanho das saídas")
        for index, value in enumerate(grad_output):
            self._validate_number(value, f"grad_output[{index}]")

        grad_output_float = [float(value) for value in grad_output]
        self.grad_weights, self.grad_bias, grad_inputs = self.backend.linear_backward(
            self.weights, self.last_input, grad_output_float
        )
        self._has_backward = True
        return grad_inputs

    def step(self, learning_rate: float) -> None:
        """Atualiza pesos e biases com gradient descent."""
        self._validate_number(learning_rate, "learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate não pode ser negativo")
        if not self._has_backward:
            raise RuntimeError("step exige um backward anterior")

        rate = float(learning_rate)
        self.weights, self.bias = self.backend.linear_step(
            self.weights,
            self.bias,
            self.grad_weights,
            self.grad_bias,
            rate,
        )
        self._has_backward = False
