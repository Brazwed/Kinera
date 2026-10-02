"""Embedding lookup treinável implementado com listas Python."""

from __future__ import annotations

import math
from random import Random


class Embedding:
    """Tabela de vetores treináveis indexada por IDs inteiros."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seed: int | None = None,
    ) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size deve ser um inteiro")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim deve ser um inteiro")
        if vocab_size <= 0:
            raise ValueError("vocab_size deve ser positivo")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser positivo")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed deve ser um inteiro ou None")

        generator = Random(seed)
        self.weights = [
            [float(generator.uniform(-0.1, 0.1)) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.grad_weights = [
            [0.0 for _ in range(embedding_dim)] for _ in range(vocab_size)
        ]
        self.last_ids: list[int] | None = None
        self.output: list[list[float]] | None = None
        self._has_backward = False

    @property
    def vocab_size(self) -> int:
        return len(self.weights)

    @property
    def embedding_dim(self) -> int:
        return len(self.weights[0])

    def forward(self, ids: list[int]) -> list[list[float]]:
        """Busca vetores na ordem dos IDs fornecidos."""
        if not isinstance(ids, list):
            raise TypeError("ids deve ser uma lista")
        vocab_size = self.vocab_size
        embedding_dim = self.embedding_dim
        for position, index in enumerate(ids):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"ids[{position}] deve ser um inteiro")
            if not 0 <= index < vocab_size:
                raise ValueError(f"ids[{position}] fora do vocabulário")

        self.last_ids = ids.copy()
        weights = self.weights
        self.output = [weights[index].copy() for index in ids]
        self.grad_weights = [
            [0.0] * embedding_dim for _ in range(vocab_size)
        ]
        self._has_backward = False
        return [vector.copy() for vector in self.output]

    def backward(self, grad_output: list[list[float]]) -> None:
        """Acumula gradientes por ID; IDs repetidos somam suas contribuições."""
        if self.last_ids is None or self.output is None:
            raise RuntimeError("backward exige um forward anterior")
        if not isinstance(grad_output, list):
            raise TypeError("grad_output deve ser uma lista")
        if len(grad_output) != len(self.last_ids):
            raise ValueError("grad_output deve ter o mesmo comprimento dos IDs")

        for position, vector in enumerate(grad_output):
            if not isinstance(vector, list):
                raise TypeError(f"grad_output[{position}] deve ser uma lista")
            if len(vector) != self.embedding_dim:
                raise ValueError(
                    f"grad_output[{position}] deve ter dimensão {self.embedding_dim}"
                )
            for dimension, value in enumerate(vector):
                self._validate_number(
                    value,
                    f"grad_output[{position}][{dimension}]",
                )
            index = self.last_ids[position]
            self.grad_weights[index] = [
                current + float(gradient)
                for current, gradient in zip(self.grad_weights[index], vector)
            ]
        self._has_backward = True

    def step(self, learning_rate: float) -> None:
        """Atualiza a tabela inteira usando os gradientes acumulados."""
        self._validate_number(learning_rate, "learning_rate")
        if learning_rate < 0.0:
            raise ValueError("learning_rate não pode ser negativo")
        if not self._has_backward:
            raise RuntimeError("step exige um backward anterior")

        self.weights = [
            [
                weight - float(learning_rate) * gradient
                for weight, gradient in zip(row, gradient_row)
            ]
            for row, gradient_row in zip(self.weights, self.grad_weights)
        ]
        self._has_backward = False

    def _validate_id(self, value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} deve ser um inteiro")
        if not 0 <= value < self.vocab_size:
            raise ValueError(f"{name} fora do vocabulário")

    @staticmethod
    def _validate_number(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} deve ser numérico")
        if not math.isfinite(value):
            raise ValueError(f"{name} deve ser finito")
