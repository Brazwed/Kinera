"""Geração e divisão determinística dos dados do primeiro experimento."""

from __future__ import annotations

import math
import random

Sample = tuple[list[float], float]


def target_function(x1: float, x2: float) -> float:
    """Função linear usada para validar o aprendizado do neurônio."""
    return 2.0 * x1 + 3.0 * x2 + 1.0


def generate_samples(
    count: int,
    seed: int | None = None,
    *,
    low: float = 0.25,
    high: float = 4.0,
) -> list[Sample]:
    """Gera amostras positivas e reproduzíveis para a função-alvo."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count deve ser um inteiro")
    if count <= 0:
        raise ValueError("count deve ser positivo")
    _validate_seed(seed)
    _validate_number(low, "low")
    _validate_number(high, "high")
    if low < 0.0:
        raise ValueError("low não pode ser negativo nesta etapa")
    if low >= high:
        raise ValueError("low deve ser menor que high")

    generator = random.Random(seed)
    samples: list[Sample] = []
    for _ in range(count):
        x1 = generator.uniform(float(low), float(high))
        x2 = generator.uniform(float(low), float(high))
        samples.append(([x1, x2], target_function(x1, x2)))
    return samples


def split_samples(
    samples: list[Sample],
    validation_ratio: float,
    seed: int | None = None,
) -> tuple[list[Sample], list[Sample]]:
    """Divide amostras em treino e validação sem compartilhar listas internas."""
    _validate_seed(seed)
    _validate_number(validation_ratio, "validation_ratio")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio deve estar entre 0 e 1")
    if not isinstance(samples, list):
        raise TypeError("samples deve ser uma lista")
    if len(samples) < 2:
        raise ValueError("samples deve conter pelo menos duas amostras")

    copied_samples = _copy_and_validate_samples(samples)
    indices = list(range(len(copied_samples)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(len(samples) - 1, int(len(samples) * validation_ratio)))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]

    train_samples = [copied_samples[index] for index in train_indices]
    validation_samples = [copied_samples[index] for index in validation_indices]
    return train_samples, validation_samples


def _copy_and_validate_samples(samples: list[Sample]) -> list[Sample]:
    copied: list[Sample] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise TypeError(f"samples[{index}] deve ser (inputs, target)")
        inputs, target = sample
        if not isinstance(inputs, list) or not inputs:
            raise TypeError(f"samples[{index}].inputs deve ser uma lista não vazia")
        copied_inputs: list[float] = []
        for input_index, value in enumerate(inputs):
            _validate_number(value, f"samples[{index}].inputs[{input_index}]")
            copied_inputs.append(float(value))
        _validate_number(target, f"samples[{index}].target")
        copied.append((copied_inputs, float(target)))
    return copied


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito")


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed deve ser um inteiro ou None")
