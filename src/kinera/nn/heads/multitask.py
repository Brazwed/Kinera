"""Cabeças de saída compartilhando a mesma representação contextual."""

from __future__ import annotations

import math
from typing import Any

from kinera.backends import Backend, resolve_backend

from .boundary import BoundaryHead
from .character import CharacterHead
from .syllable import SyllableHead


class MultiTaskHeads:
    """Cabeças independentes para caractere, sílaba e fronteira."""

    def __init__(
        self,
        input_dim: int,
        char_vocab_size: int,
        syllable_vocab_size: int,
        *,
        seed: int | None = None,
        backend: Backend | None = None,
    ) -> None:
        self._validate_dimension(input_dim, "input_dim")
        self._validate_dimension(char_vocab_size, "char_vocab_size")
        self._validate_dimension(syllable_vocab_size, "syllable_vocab_size")
        self.backend = resolve_backend(backend)
        self.char = CharacterHead(
            input_dim, char_vocab_size, seed=seed, backend=self.backend
        )
        self.syllable = SyllableHead(
            input_dim,
            syllable_vocab_size,
            seed=None if seed is None else seed + 1,
            backend=self.backend,
        )
        self.boundary = BoundaryHead(
            input_dim,
            seed=None if seed is None else seed + 2,
            backend=self.backend,
        )

    def forward(self, representation: list[float]) -> dict[str, list[float]]:
        """Calcula logits das três tarefas sobre uma entrada compartilhada."""
        self._validate_representation(representation)
        return {
            "char": self.char.forward(representation),
            "syllable": self.syllable.forward(representation),
            "boundary": self.boundary.forward(representation),
        }

    def backward(
        self,
        gradients: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        """Propaga gradientes de cada head e devolve gradientes da entrada."""
        if not isinstance(gradients, dict) or set(gradients) != {
            "char",
            "syllable",
            "boundary",
        }:
            raise ValueError("gradients deve conter char, syllable e boundary")
        return {
            "char": self.char.backward(gradients["char"]),
            "syllable": self.syllable.backward(gradients["syllable"]),
            "boundary": self.boundary.backward(gradients["boundary"]),
        }

    def step(self, learning_rate: float) -> None:
        """Atualiza os parâmetros das três cabeças."""
        self.char.step(learning_rate)
        self.syllable.step(learning_rate)
        self.boundary.step(learning_rate)

    @staticmethod
    def _validate_dimension(value: object, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} deve ser inteiro")
        if value <= 0:
            raise ValueError(f"{name} deve ser positivo")

    @staticmethod
    def _validate_representation(representation: list[float]) -> None:
        if not isinstance(representation, list):
            raise TypeError("representation deve ser uma lista")
        for index, value in enumerate(representation):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"representation[{index}] deve ser numérico")
            if not math.isfinite(value):
                raise ValueError(f"representation[{index}] deve ser finito")
