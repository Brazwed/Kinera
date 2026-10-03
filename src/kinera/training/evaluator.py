"""Evaluation and greedy generation for Kinera models."""

from __future__ import annotations

import math
from typing import Any

from kinera.language.datasets.adapter import example_to_inputs
from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.layers.neuron import Neuron
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.losses.cross_entropy import cross_entropy
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.common import Sample, _restore, _snapshot, _validate_samples
from kinera.training.metrics import _multitask_targets, _safe_ratio


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

def evaluate_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
) -> dict[str, float | int]:
    """Mede loss e acurácia sem executar backward ou step."""
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")

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
        logits = model.forward(*inputs)
        total_loss += cross_entropy(logits, target_id, backend=model.backend)
        correct += max(range(len(logits)), key=logits.__getitem__) == target_id
    return {
        "examples": len(selected),
        "loss": total_loss / len(selected),
        "accuracy": correct / len(selected),
    }

def evaluate_multitask_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
    loss_function: MultiTaskLoss | None = None,
) -> dict[str, Any]:
    """Avalia as três heads sem executar backward ou step."""
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")
    loss_function = MultiTaskLoss(backend=model.backend) if loss_function is None else loss_function
    totals = {
        "char_loss": 0.0,
        "syllable_loss": 0.0,
        "boundary_loss": 0.0,
        "total_loss": 0.0,
        "char_accuracy": 0.0,
        "syllable_accuracy": 0.0,
        "boundary_accuracy": 0.0,
    }
    syllable_known_total = 0
    syllable_known_correct = 0
    boundary_true_positive = 0
    boundary_false_positive = 0
    boundary_false_negative = 0
    boundary_target_distribution = {"0": 0, "1": 0}
    for example in selected:
        inputs = example_to_inputs(
            example,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        targets = _multitask_targets(example)
        outputs = model.forward_heads(*inputs)
        losses = loss_function.forward(outputs, targets)
        totals["char_loss"] += losses["char_loss"]
        totals["syllable_loss"] += losses["syllable_loss"]
        totals["boundary_loss"] += losses["boundary_loss"]
        totals["total_loss"] += losses["total_loss"]
        totals["char_accuracy"] += max(
            range(len(outputs["char"])), key=outputs["char"].__getitem__
        ) == targets["char"]
        syllable_prediction = max(
            range(len(outputs["syllable"])), key=outputs["syllable"].__getitem__
        )
        syllable_correct = syllable_prediction == targets["syllable"]
        totals["syllable_accuracy"] += syllable_correct
        if targets["syllable"] != syllable_unknown_id:
            syllable_known_total += 1
            syllable_known_correct += syllable_correct
        boundary_target = int(targets["boundary"] >= 0.5)
        boundary_prediction = int(outputs["boundary"][0] >= 0.0)
        boundary_target_distribution[str(boundary_target)] += 1
        totals["boundary_accuracy"] += boundary_prediction == boundary_target
        if boundary_prediction == 1 and boundary_target == 1:
            boundary_true_positive += 1
        elif boundary_prediction == 1:
            boundary_false_positive += 1
        elif boundary_target == 1:
            boundary_false_negative += 1
    count = len(selected)
    boundary_precision = _safe_ratio(
        boundary_true_positive,
        boundary_true_positive + boundary_false_positive,
    )
    boundary_recall = _safe_ratio(
        boundary_true_positive,
        boundary_true_positive + boundary_false_negative,
    )
    boundary_f1 = _safe_ratio(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )
    return {
        "examples": count,
        **{name: value / count for name, value in totals.items()},
        "syllable_accuracy_all": totals["syllable_accuracy"] / count,
        "syllable_accuracy_known": _safe_ratio(
            syllable_known_correct, syllable_known_total
        ),
        "boundary_target_distribution": boundary_target_distribution,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
    }

def generate_text(
    model: AutoregressiveModel,
    vocabulary: CharacterVocabulary,
    *,
    context_size: int,
    max_tokens: int,
    syllable_unknown_id: int = 0,
) -> str:
    """Gera texto de forma gulosa até ``<END>`` ou o limite definido."""
    if context_size <= 0 or max_tokens < 0:
        raise ValueError("context_size deve ser positivo e max_tokens não negativo")
    context = [vocabulary.start_id] * context_size
    generated: list[int] = []
    for _ in range(max_tokens):
        example = {"context_ids": context}
        inputs = example_to_inputs(
            example,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        next_id = model.predict(*inputs)
        if next_id == vocabulary.end_id:
            break
        generated.append(next_id)
        context = [*context[1:], next_id]
    return vocabulary.decode(generated, skip_special=True)
