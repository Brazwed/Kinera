"""Losses manuais para o treinamento multiobjetivo."""

from __future__ import annotations

from typing import Mapping

from kinera.backends import Backend, resolve_backend
from .binary_cross_entropy import _validate_binary_target, _validate_number


_REQUIRED_HEADS = ("char", "syllable", "boundary")


class MultiTaskLoss:
    """Combina losses de caractere, sílaba e fronteira."""

    def __init__(
        self,
        *,
        char_weight: float = 1.0,
        syllable_weight: float = 0.7,
        boundary_weight: float = 0.3,
        backend: Backend | None = None,
    ) -> None:
        for value, name in (
            (char_weight, "char_weight"),
            (syllable_weight, "syllable_weight"),
            (boundary_weight, "boundary_weight"),
        ):
            _validate_number(value, name)
            if value < 0.0:
                raise ValueError(f"{name} não pode ser negativo")
        if char_weight == syllable_weight == boundary_weight == 0.0:
            raise ValueError("ao menos um peso deve ser positivo")
        self.backend = resolve_backend(backend)
        self.weights = {
            "char": float(char_weight),
            "syllable": float(syllable_weight),
            "boundary": float(boundary_weight),
        }

    def forward(
        self,
        outputs: Mapping[str, list[float]],
        targets: Mapping[str, object],
    ) -> dict[str, float]:
        """Retorna losses individuais e a loss total ponderada."""
        self._validate_outputs(outputs)
        char_target = _validate_class_target(targets.get("char"), len(outputs["char"]), "char")
        syllable_target = _validate_class_target(
            targets.get("syllable"), len(outputs["syllable"]), "syllable"
        )
        boundary_target = _validate_binary_target(targets.get("boundary"))
        char_loss = self.backend.cross_entropy(outputs["char"], char_target)
        syllable_loss = self.backend.cross_entropy(
            outputs["syllable"], syllable_target
        )
        boundary_loss = self.backend.binary_cross_entropy_with_logits(
            outputs["boundary"][0], boundary_target
        )
        return {
            "char_loss": char_loss,
            "syllable_loss": syllable_loss,
            "boundary_loss": boundary_loss,
            "total_loss": (
                self.weights["char"] * char_loss
                + self.weights["syllable"] * syllable_loss
                + self.weights["boundary"] * boundary_loss
            ),
        }

    def backward(
        self,
        outputs: Mapping[str, list[float]],
        targets: Mapping[str, object],
    ) -> dict[str, list[float]]:
        """Retorna gradientes ponderados em relação aos logits de cada head."""
        self._validate_outputs(outputs)
        char_target = _validate_class_target(targets.get("char"), len(outputs["char"]), "char")
        syllable_target = _validate_class_target(
            targets.get("syllable"), len(outputs["syllable"]), "syllable"
        )
        boundary_target = _validate_binary_target(targets.get("boundary"))
        return {
            "char": [
                self.weights["char"] * value
                for value in self.backend.cross_entropy_gradient(
                    outputs["char"], char_target
                )
            ],
            "syllable": [
                self.weights["syllable"] * value
                for value in self.backend.cross_entropy_gradient(
                    outputs["syllable"], syllable_target
                )
            ],
            "boundary": [
                self.weights["boundary"]
                * self.backend.binary_cross_entropy_with_logits_gradient(
                    outputs["boundary"][0], boundary_target
                )
            ],
        }

    @staticmethod
    def _validate_outputs(outputs: Mapping[str, list[float]]) -> None:
        if not isinstance(outputs, Mapping) or set(outputs) != set(_REQUIRED_HEADS):
            raise ValueError("outputs deve conter char, syllable e boundary")
        for name in _REQUIRED_HEADS:
            values = outputs[name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"outputs[{name}] deve ser uma lista não vazia")
            for index, value in enumerate(values):
                _validate_number(value, f"outputs[{name}][{index}]")
        if len(outputs["boundary"]) != 1:
            raise ValueError("outputs[boundary] deve conter um logit")


def _validate_class_target(value: object, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"targets[{name}] deve ser inteiro")
    if not 0 <= value < size:
        raise ValueError(f"targets[{name}] fora do vocabulário")
    return value
