"""Losses manuais para o treinamento multiobjetivo."""

from __future__ import annotations

import math
from typing import Mapping

from nn.probability import cross_entropy, cross_entropy_gradient


_REQUIRED_HEADS = ("char", "syllable", "boundary")


def binary_cross_entropy_with_logits(logit: float, target: float) -> float:
    """Calcula BCE diretamente dos logits com estabilidade numérica."""
    _validate_binary_target(target)
    _validate_number(logit, "logit")
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def binary_cross_entropy_with_logits_gradient(logit: float, target: float) -> float:
    """Calcula dBCE/dlogit."""
    _validate_binary_target(target)
    _validate_number(logit, "logit")
    if logit >= 0.0:
        exponential = math.exp(-logit)
        sigmoid = 1.0 / (1.0 + exponential)
    else:
        exponential = math.exp(logit)
        sigmoid = exponential / (1.0 + exponential)
    return sigmoid - target


class MultiTaskLoss:
    """Combina losses de caractere, sílaba e fronteira."""

    def __init__(
        self,
        *,
        char_weight: float = 1.0,
        syllable_weight: float = 0.7,
        boundary_weight: float = 0.3,
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
        char_loss = cross_entropy(outputs["char"], char_target)
        syllable_loss = cross_entropy(outputs["syllable"], syllable_target)
        boundary_loss = binary_cross_entropy_with_logits(
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
                for value in cross_entropy_gradient(outputs["char"], char_target)
            ],
            "syllable": [
                self.weights["syllable"] * value
                for value in cross_entropy_gradient(outputs["syllable"], syllable_target)
            ],
            "boundary": [
                self.weights["boundary"]
                * binary_cross_entropy_with_logits_gradient(
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


def _validate_binary_target(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("target de fronteira deve ser numérico")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("target de fronteira deve estar entre 0 e 1")
    return float(value)


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito")
