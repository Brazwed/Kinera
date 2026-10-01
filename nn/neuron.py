"""Neurônio artificial mínimo implementado sem bibliotecas numéricas."""

import math
from random import Random

from nn.activations import (
    leaky_relu,
    leaky_relu_derivative,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
)

_SUPPORTED_ACTIVATIONS = {"relu", "leaky_relu", "linear"}


class Neuron:
    """Unidade treinável que receberá um vetor e produzirá uma saída."""

    def __init__(
        self,
        input_size: int,
        seed: int | None = None,
        *,
        activation: str = "relu",
        negative_slope: float = 0.01,
    ) -> None:
        if isinstance(input_size, bool) or not isinstance(input_size, int):
            raise TypeError("input_size deve ser um inteiro")
        if input_size <= 0:
            raise ValueError("input_size deve ser positivo")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed deve ser um inteiro ou None")
        if not isinstance(activation, str):
            raise TypeError("activation deve ser uma string")
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f"activation desconhecida: {activation}")
        if activation == "leaky_relu":
            leaky_relu(0.0, negative_slope=negative_slope)
        elif isinstance(negative_slope, bool) or not isinstance(
            negative_slope, (int, float)
        ):
            raise TypeError("negative_slope deve ser numérico")

        generator = Random(seed)
        self.weights = [float(generator.uniform(-0.1, 0.1)) for _ in range(input_size)]
        self.bias = float(generator.uniform(-0.1, 0.1))
        self.activation = activation
        self.negative_slope = float(negative_slope)

        self.last_input: list[float] | None = None
        self.last_z: float | None = None
        self.output: float | None = None
        self.grad_weights = [0.0 for _ in range(input_size)]
        self.grad_bias = 0.0
        self._has_backward = False

    @staticmethod
    def _validate_number(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} deve ser numérico")
        if not math.isfinite(value):
            raise ValueError(f"{name} deve ser finito")

    def forward(self, inputs: list[float]) -> float:
        """Calcula e armazena a saída ativada para uma entrada."""
        if not isinstance(inputs, list):
            raise TypeError("inputs deve ser uma lista")
        if len(inputs) != len(self.weights):
            raise ValueError("inputs deve ter o mesmo tamanho dos pesos")

        for index, value in enumerate(inputs):
            self._validate_number(value, f"inputs[{index}]")

        self.last_input = [float(value) for value in inputs]
        self.last_z = sum(
            input_value * weight
            for input_value, weight in zip(self.last_input, self.weights)
        ) + self.bias
        self.output = self._activate(self.last_z)
        self.grad_weights = [0.0 for _ in self.weights]
        self.grad_bias = 0.0
        self._has_backward = False
        return self.output

    def backward(self, grad_output: float) -> list[float]:
        """Calcula os gradientes dos parâmetros e da entrada."""
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("backward exige um forward anterior")
        self._validate_number(grad_output, "grad_output")

        grad_z = float(grad_output) * self._activation_derivative(self.last_z)
        self.grad_weights = [
            grad_z * input_value for input_value in self.last_input
        ]
        self.grad_bias = grad_z
        self._has_backward = True
        return [grad_z * weight for weight in self.weights]

    def step(self, learning_rate: float) -> None:
        """Atualiza os parâmetros usando os gradientes calculados."""
        self._validate_number(learning_rate, "learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate não pode ser negativo")
        if not self._has_backward:
            raise RuntimeError("step exige um backward anterior")

        self.weights = [
            weight - float(learning_rate) * gradient
            for weight, gradient in zip(self.weights, self.grad_weights)
        ]
        self.bias -= float(learning_rate) * self.grad_bias
        self._has_backward = False

    def _activate(self, value: float) -> float:
        if self.activation == "relu":
            return relu(value)
        if self.activation == "leaky_relu":
            return leaky_relu(value, self.negative_slope)
        return linear(value)

    def _activation_derivative(self, value: float) -> float:
        if self.activation == "relu":
            return relu_derivative(value)
        if self.activation == "leaky_relu":
            return leaky_relu_derivative(value, self.negative_slope)
        return linear_derivative(value)
