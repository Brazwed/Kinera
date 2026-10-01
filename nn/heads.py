"""Cabeças de saída compartilhando a mesma representação contextual."""

from __future__ import annotations

import math
from typing import Any

from nn.linear import Linear


class MultiTaskHeads:
    """Cabeças independentes para caractere, sílaba e fronteira."""

    def __init__(
        self,
        input_dim: int,
        char_vocab_size: int,
        syllable_vocab_size: int,
        *,
        seed: int | None = None,
    ) -> None:
        self._validate_dimension(input_dim, "input_dim")
        self._validate_dimension(char_vocab_size, "char_vocab_size")
        self._validate_dimension(syllable_vocab_size, "syllable_vocab_size")
        self.char = Linear(input_dim, char_vocab_size, seed=seed)
        self.syllable = Linear(
            input_dim,
            syllable_vocab_size,
            seed=None if seed is None else seed + 1,
        )
        self.boundary = Linear(
            input_dim,
            1,
            seed=None if seed is None else seed + 2,
        )

    def forward(self, representation: list[float]) -> dict[str, list[float]]:
        """Calcula logits das três tarefas sobre uma entrada compartilhada."""
        self._validate_representation(representation)
        return {
            "char": self.char.forward(representation),
            "syllable": self.syllable.forward(representation),
            "boundary": self.boundary.forward(representation),
        }

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
