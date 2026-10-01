"""Treinamento por épocas para os componentes neurais da Kinera."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias

from nn.neuron import Neuron

Sample: TypeAlias = tuple[list[float], float]


@dataclass(frozen=True)
class TrainingConfig:
    """Parâmetros explícitos de uma execução de treinamento."""

    learning_rate: float
    max_epochs: int
    patience: int
    min_delta: float = 0.0
    shuffle: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_number(self.learning_rate, "learning_rate")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate deve ser positivo")
        if isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int):
            raise TypeError("max_epochs deve ser um inteiro")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs deve ser positivo")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise TypeError("patience deve ser um inteiro")
        if self.patience < 0:
            raise ValueError("patience não pode ser negativo")
        _validate_number(self.min_delta, "min_delta")
        if self.min_delta < 0.0:
            raise ValueError("min_delta não pode ser negativo")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle deve ser booleano")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed deve ser um inteiro ou None")


@dataclass(frozen=True)
class TrainingHistory:
    """Métricas e motivo de parada de uma execução."""

    train_losses: list[float]
    validation_losses: list[float]
    steps: int
    best_epoch: int
    best_validation_loss: float
    stopped_reason: str

    @property
    def epochs_completed(self) -> int:
        return len(self.train_losses)


@dataclass(frozen=True)
class TrainingResult:
    """Resultado completo, incluindo os parâmetros restaurados do melhor ponto."""

    history: TrainingHistory
    final_weights: list[float]
    final_bias: float


def evaluate(neuron: Neuron, samples: list[Sample]) -> float:
    """Calcula a loss média sem alterar o estado observável do neurônio."""
    _validate_samples(neuron, samples, "samples")
    snapshot = _snapshot(neuron)
    try:
        total_loss = 0.0
        for inputs, target in samples:
            prediction = neuron.forward(inputs)
            total_loss += 0.5 * (prediction - target) ** 2
        return total_loss / len(samples)
    finally:
        _restore(neuron, snapshot)


def train_epoch(
    neuron: Neuron,
    samples: list[Sample],
    learning_rate: float,
    *,
    shuffle: bool = False,
    rng: random.Random | None = None,
) -> float:
    """Executa uma época de SGD e retorna a loss média da época."""
    _validate_samples(neuron, samples, "samples")
    _validate_number(learning_rate, "learning_rate")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate deve ser positivo")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle deve ser booleano")

    order = list(range(len(samples)))
    if shuffle:
        (rng or random.Random()).shuffle(order)

    total_loss = 0.0
    for index in order:
        inputs, target = samples[index]
        prediction = neuron.forward(inputs)
        error = prediction - target
        total_loss += 0.5 * error**2
        neuron.backward(error)
        neuron.step(learning_rate)
    return total_loss / len(samples)


def fit(
    neuron: Neuron,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    config: TrainingConfig,
) -> TrainingResult:
    """Treina por épocas, aplica early stopping e restaura o melhor estado."""
    if not isinstance(config, TrainingConfig):
        raise TypeError("config deve ser uma instância de TrainingConfig")
    _validate_samples(neuron, train_samples, "train_samples")
    _validate_samples(neuron, validation_samples, "validation_samples")

    rng = random.Random(config.seed)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, object] | None = None
    stopped_reason = "max_epochs"

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_epoch(
            neuron,
            train_samples,
            config.learning_rate,
            shuffle=config.shuffle,
            rng=rng,
        )
        validation_loss = evaluate(neuron, validation_samples)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = _snapshot(neuron)
        else:
            stale_epochs += 1

        if stale_epochs > 0 and stale_epochs >= config.patience:
            stopped_reason = "early_stopping"
            break

    if best_state is None:
        raise RuntimeError("nenhum estado válido foi produzido durante o treino")
    _restore(neuron, best_state)

    history = TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        steps=len(train_samples) * len(train_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_reason=stopped_reason,
    )
    return TrainingResult(
        history=history,
        final_weights=neuron.weights.copy(),
        final_bias=neuron.bias,
    )


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
