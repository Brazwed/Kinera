"""Losses manuais para o treinamento multiobjetivo."""

from __future__ import annotations

from typing import Callable, Mapping, cast

from kinera.backends import Backend, resolve_backend
from kinera.backends.base import (
    BackendBuffer,
    BackendError,
    BackendUnavailableError,
    release_backend_buffers,
)
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
        outputs: Mapping[str, list[float] | BackendBuffer],
        targets: Mapping[str, object],
    ) -> dict[str, float]:
        """Retorna losses individuais e a loss total ponderada."""
        resident = self._validate_outputs(outputs)
        char_target = _validate_class_target(
            targets.get("char"), self._output_size(outputs["char"]), "char"
        )
        syllable_target = _validate_class_target(
            targets.get("syllable"), self._output_size(outputs["syllable"]), "syllable"
        )
        boundary_target = _validate_binary_target(targets.get("boundary"))
        if resident:
            char_loss = self._resident_loss(
                "cross_entropy_forward_buffer", outputs["char"], char_target
            )
            syllable_loss = self._resident_loss(
                "cross_entropy_forward_buffer", outputs["syllable"], syllable_target
            )
            boundary_loss = self._resident_loss(
                "binary_cross_entropy_with_logits_forward_buffer",
                outputs["boundary"],
                boundary_target,
            )
        else:
            char_logits = outputs["char"]
            syllable_logits = outputs["syllable"]
            boundary_logits = outputs["boundary"]
            assert isinstance(char_logits, list)
            assert isinstance(syllable_logits, list)
            assert isinstance(boundary_logits, list)
            char_loss = self.backend.cross_entropy(char_logits, char_target)
            syllable_loss = self.backend.cross_entropy(
                syllable_logits, syllable_target
            )
            boundary_loss = self.backend.binary_cross_entropy_with_logits(
                boundary_logits[0], boundary_target
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
        outputs: Mapping[str, list[float] | BackendBuffer],
        targets: Mapping[str, object],
    ) -> dict[str, list[float] | BackendBuffer]:
        """Retorna gradientes ponderados em relação aos logits de cada head."""
        resident = self._validate_outputs(outputs)
        char_target = _validate_class_target(
            targets.get("char"), self._output_size(outputs["char"]), "char"
        )
        syllable_target = _validate_class_target(
            targets.get("syllable"), self._output_size(outputs["syllable"]), "syllable"
        )
        boundary_target = _validate_binary_target(targets.get("boundary"))
        if resident:
            gradients: dict[str, list[float] | BackendBuffer] = {}
            try:
                gradients["char"] = self._resident_gradient(
                    "cross_entropy_gradient_buffer",
                    outputs["char"],
                    char_target,
                    self.weights["char"],
                )
                gradients["syllable"] = self._resident_gradient(
                    "cross_entropy_gradient_buffer",
                    outputs["syllable"],
                    syllable_target,
                    self.weights["syllable"],
                )
                gradients["boundary"] = self._resident_gradient(
                    "binary_cross_entropy_with_logits_gradient_buffer",
                    outputs["boundary"],
                    boundary_target,
                    self.weights["boundary"],
                )
                return gradients
            except Exception:
                release_backend_buffers(*gradients.values())
                raise

        char_logits = outputs["char"]
        syllable_logits = outputs["syllable"]
        boundary_logits = outputs["boundary"]
        assert isinstance(char_logits, list)
        assert isinstance(syllable_logits, list)
        assert isinstance(boundary_logits, list)
        return {
            "char": [
                self.weights["char"] * value
                for value in self.backend.cross_entropy_gradient(
                    char_logits, char_target
                )
            ],
            "syllable": [
                self.weights["syllable"] * value
                for value in self.backend.cross_entropy_gradient(
                    syllable_logits, syllable_target
                )
            ],
            "boundary": [
                self.weights["boundary"]
                * self.backend.binary_cross_entropy_with_logits_gradient(
                    boundary_logits[0], boundary_target
                )
            ],
        }

    def _resident_loss(
        self,
        capability_name: str,
        logits: list[float] | BackendBuffer,
        target: int | float,
    ) -> float:
        assert isinstance(logits, BackendBuffer)
        capability = self._require_capability(capability_name)
        value = capability(logits, target)
        _validate_number(value, capability_name)
        return float(cast(int | float, value))

    def _resident_gradient(
        self,
        capability_name: str,
        logits: list[float] | BackendBuffer,
        target: int | float,
        scale: float,
    ) -> BackendBuffer:
        assert isinstance(logits, BackendBuffer)
        capability = self._require_capability(capability_name)
        gradient = capability(logits, target, scale)
        if not isinstance(gradient, BackendBuffer):
            release_backend_buffers(gradient)
            raise TypeError(f"{capability_name} must return a BackendBuffer")
        try:
            if gradient.closed:
                raise BackendError(f"{capability_name} returned a released buffer")
            if gradient.dtype != "float64" or gradient.shape != logits.shape:
                raise BackendError(f"{capability_name} returned an invalid gradient buffer")
            if (gradient.device, gradient.device_id) != (logits.device, logits.device_id):
                raise BackendError(f"{capability_name} returned a gradient on the wrong device")
        except Exception:
            release_backend_buffers(gradient)
            raise
        return gradient

    def _require_capability(self, name: str) -> Callable[..., object]:
        capability = getattr(self.backend, name, None)
        if not callable(capability):
            raise BackendUnavailableError(
                f"backend must provide {name} for device-resident loss operations"
            )
        return capability

    @staticmethod
    def _output_size(values: list[float] | BackendBuffer) -> int:
        return values.shape[0] if isinstance(values, BackendBuffer) else len(values)

    @staticmethod
    def _validate_outputs(
        outputs: Mapping[str, list[float] | BackendBuffer],
    ) -> bool:
        if not isinstance(outputs, Mapping) or set(outputs) != set(_REQUIRED_HEADS):
            raise ValueError("outputs deve conter char, syllable e boundary")
        resident_flags = [
            isinstance(outputs[name], BackendBuffer) for name in _REQUIRED_HEADS
        ]
        if any(resident_flags):
            if not all(resident_flags):
                raise TypeError(
                    "outputs deve usar listas ou buffers do backend, sem misturá-los"
                )
            buffers = [outputs[name] for name in _REQUIRED_HEADS]
            first = buffers[0]
            assert isinstance(first, BackendBuffer)
            for name, value in zip(_REQUIRED_HEADS, buffers):
                assert isinstance(value, BackendBuffer)
                if value.closed:
                    raise BackendError(f"outputs[{name}] buffer has been released")
                if value.dtype != "float64":
                    raise ValueError(
                        f"outputs[{name}] buffer must use dtype float64"
                    )
                if len(value.shape) != 1 or value.shape[0] <= 0:
                    raise ValueError(
                        f"outputs[{name}] buffer must be a non-empty vector"
                    )
                if (value.device, value.device_id) != (
                    first.device,
                    first.device_id,
                ):
                    raise BackendError(
                        "all multitask logits must reside on the same device"
                    )
            boundary = outputs["boundary"]
            assert isinstance(boundary, BackendBuffer)
            if boundary.shape != (1,):
                raise ValueError("outputs[boundary] deve conter um logit")
            return True
        for name in _REQUIRED_HEADS:
            values = outputs[name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"outputs[{name}] deve ser uma lista não vazia")
            for index, value in enumerate(values):
                _validate_number(value, f"outputs[{name}][{index}]")
        if len(outputs["boundary"]) != 1:
            raise ValueError("outputs[boundary] deve conter um logit")
        return False


def _validate_class_target(value: object, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"targets[{name}] deve ser inteiro")
    if not 0 <= value < size:
        raise ValueError(f"targets[{name}] fora do vocabulário")
    return value
