"""Training procedures for neuron and autoregressive models."""

from __future__ import annotations

import math
import random
from typing import Any

from kinera.monitor.metrics.core import mean_from_total, safe_ratio
from kinera.language.datasets.adapter import example_to_inputs
from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.layers.neuron import Neuron
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.common import (
    Sample,
    _restore,
    _snapshot,
    _validate_number,
    _validate_samples,
)
from kinera.training.config import TrainingConfig, TrainingHistory, TrainingResult
from kinera.training.evaluator import evaluate, evaluate_multitask_examples
from kinera.training.metrics import _multitask_targets


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
    return mean_from_total(total_loss, len(samples))

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

def train_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    epochs: int,
    learning_rate: float,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Treina exemplos em ordem determinística e retorna métricas por época."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs deve ser inteiro positivo")
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")

    history: list[dict[str, float]] = []
    for _ in range(epochs):
        total_loss = 0.0
        correct = 0
        for example in selected:
            inputs = example_to_inputs(
                example,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
            )
            target_id = example.get("target_char_id")
            if isinstance(target_id, bool) or not isinstance(target_id, int):
                raise ValueError("target_char_id deve ser inteiro")
            prediction = model.predict(*inputs)
            total_loss += model.train_step(
                *inputs,
                target_id=target_id,
                learning_rate=learning_rate,
            )
            correct += prediction == target_id
        history.append(
            {
                "loss": safe_ratio(total_loss, len(selected)),
                "accuracy": safe_ratio(correct, len(selected)),
            }
        )
    return {"epochs": epochs, "examples": len(selected), "history": history}

def train_multitask_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    epochs: int,
    learning_rate: float,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
    loss_function: MultiTaskLoss | None = None,
) -> dict[str, Any]:
    """Treina as três heads e registra métricas por época."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs deve ser inteiro positivo")
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")

    history: list[dict[str, float]] = []
    for _ in range(epochs):
        for example in selected:
            inputs = example_to_inputs(
                example,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
            )
            model.multitask_train_step(
                *inputs,
                targets=_multitask_targets(example),
                learning_rate=learning_rate,
                loss_function=loss_function,
            )
        metrics = evaluate_multitask_examples(
            model,
            selected,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
            loss_function=loss_function,
        )
        history.append({
            key: value for key, value in metrics.items() if key != "examples"
        })
    return {"epochs": epochs, "examples": len(selected), "history": history}
