"""Shared training-state validation and snapshot helpers."""

from __future__ import annotations

import math
from typing import TypeAlias

from kinera.nn.layers.neuron import Neuron

Sample: TypeAlias = tuple[list[float], float]


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito")

def _validate_samples(neuron: Neuron, samples: list[Sample], name: str) -> None:
    if not isinstance(samples, list):
        raise TypeError(f"{name} deve ser uma lista")
    if not samples:
        raise ValueError(f"{name} não pode ser vazio")

    expected_size = len(neuron.weights)
    for index, sample in enumerate(samples):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise TypeError(f"{name}[{index}] deve ser (inputs, target)")
        inputs, target = sample
        if not isinstance(inputs, list):
            raise TypeError(f"{name}[{index}].inputs deve ser uma lista")
        if len(inputs) != expected_size:
            raise ValueError(
                f"{name}[{index}].inputs deve ter {expected_size} valores"
            )
        for input_index, value in enumerate(inputs):
            _validate_number(value, f"{name}[{index}].inputs[{input_index}]")
        _validate_number(target, f"{name}[{index}].target")

def _snapshot(neuron: Neuron) -> dict[str, object]:
    return {
        "weights": neuron.weights.copy(),
        "bias": neuron.bias,
        "last_input": None
        if neuron.last_input is None
        else neuron.last_input.copy(),
        "last_z": neuron.last_z,
        "output": neuron.output,
        "grad_weights": neuron.grad_weights.copy(),
        "grad_bias": neuron.grad_bias,
        "has_backward": neuron._has_backward,
    }

def _restore(neuron: Neuron, snapshot: dict[str, object]) -> None:
    neuron.weights = list(snapshot["weights"])
    neuron.bias = float(snapshot["bias"])
    last_input = snapshot["last_input"]
    neuron.last_input = None if last_input is None else list(last_input)
    neuron.last_z = snapshot["last_z"]
    neuron.output = snapshot["output"]
    neuron.grad_weights = list(snapshot["grad_weights"])
    neuron.grad_bias = float(snapshot["grad_bias"])
    neuron._has_backward = bool(snapshot["has_backward"])
