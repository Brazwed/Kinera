"""Evaluation and greedy generation for Kinera models."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, cast

from kinera.language.datasets.adapter import example_to_inputs
from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.layers.neuron import Neuron
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.losses.cross_entropy import cross_entropy
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.common import Sample, _restore, _snapshot, _validate_samples
from kinera.monitor.metrics import (
    BinaryConfusion,
    balanced_accuracy,
    f1_score,
    metric_result,
    majority_class,
    mean_from_total,
    precision as confusion_precision,
    recall as confusion_recall,
    safe_ratio,
    specificity,
    METRICS_SCHEMA_VERSION,
)
from kinera.training.metrics import _multitask_targets


def evaluate(neuron: Neuron, samples: list[Sample]) -> float:
    """Calcula a loss média sem alterar o estado observável do neurônio."""
    _validate_samples(neuron, samples, "samples")
    snapshot = _snapshot(neuron)
    try:
        total_loss = 0.0
        for inputs, target in samples:
            prediction = neuron.forward(inputs)
            total_loss += 0.5 * (prediction - target) ** 2
        return mean_from_total(total_loss, len(samples))
    finally:
        _restore(neuron, snapshot)

def evaluate_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
) -> dict[str, Any]:
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
    loss = safe_ratio(total_loss, len(selected))
    accuracy = safe_ratio(correct, len(selected))
    return {
        "examples": len(selected),
        "loss": loss,
        "accuracy": accuracy,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "metric_results": {
            "loss": metric_result(
                loss,
                availability="available",
                reason=None,
                support={"examples": len(selected)},
            ),
            "accuracy": metric_result(
                accuracy,
                availability="available",
                reason=None,
                support={"examples": len(selected)},
            ),
        },
    }

def evaluate_multitask_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
    loss_function: MultiTaskLoss | None = None,
    train_boundary_distribution: Mapping[int | str, int] | None = None,
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
    boundary_true_negative = 0
    boundary_target_distribution = {"0": 0, "1": 0}
    char_target_distribution: Counter[int] = Counter()
    char_prediction_distribution: Counter[int] = Counter()
    syllable_target_distribution: Counter[int] = Counter()
    syllable_prediction_distribution: Counter[int] = Counter()
    syllable_target_status_distribution = {"known": 0, "unknown": 0, "end": 0}
    for example in selected:
        inputs = example_to_inputs(
            example,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        targets = _multitask_targets(example)
        result = model.multitask_eval_step(
            *inputs,
            targets=targets,
            loss_function=loss_function,
        )
        losses = cast(dict[str, float], result["losses"])
        predictions = cast(dict[str, int], result["predictions"])
        char_target_distribution[targets["char"]] += 1
        char_prediction_distribution[predictions["char"]] += 1
        syllable_target_distribution[targets["syllable"]] += 1
        syllable_prediction_distribution[predictions["syllable"]] += 1
        if targets["char"] == vocabulary.end_id:
            syllable_target_status_distribution["end"] += 1
        elif targets["syllable"] == syllable_unknown_id:
            syllable_target_status_distribution["unknown"] += 1
        else:
            syllable_target_status_distribution["known"] += 1
        totals["char_loss"] += losses["char_loss"]
        totals["syllable_loss"] += losses["syllable_loss"]
        totals["boundary_loss"] += losses["boundary_loss"]
        totals["total_loss"] += losses["total_loss"]
        totals["char_accuracy"] += predictions["char"] == targets["char"]
        syllable_prediction = predictions["syllable"]
        syllable_correct = syllable_prediction == targets["syllable"]
        totals["syllable_accuracy"] += syllable_correct
        if targets["syllable"] != syllable_unknown_id:
            syllable_known_total += 1
            syllable_known_correct += syllable_correct
        boundary_target = int(targets["boundary"] >= 0.5)
        boundary_prediction = predictions["boundary"]
        boundary_target_distribution[str(boundary_target)] += 1
        totals["boundary_accuracy"] += boundary_prediction == boundary_target
        if boundary_prediction == 1 and boundary_target == 1:
            boundary_true_positive += 1
        elif boundary_prediction == 1 and boundary_target == 0:
            boundary_false_positive += 1
        elif boundary_prediction == 0 and boundary_target == 1:
            boundary_false_negative += 1
        elif boundary_prediction == 0 and boundary_target == 0:
            boundary_true_negative += 1

    count = len(selected)
    boundary_precision = safe_ratio(
        boundary_true_positive,
        boundary_true_positive + boundary_false_positive,
    )
    boundary_recall = safe_ratio(
        boundary_true_positive,
        boundary_true_positive + boundary_false_negative,
    )
    boundary_f1 = safe_ratio(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )

    boundary_confusion = BinaryConfusion(
        tp=boundary_true_positive,
        tn=boundary_true_negative,
        fp=boundary_false_positive,
        fn=boundary_false_negative,
    )
    boundary_spec = specificity(boundary_confusion)
    boundary_bal_acc = balanced_accuracy(boundary_confusion)
    boundary_prediction_distribution = {
        "0": boundary_confusion.predicted_negative,
        "1": boundary_confusion.predicted_positive,
    }
    total_boundary_count = boundary_confusion.total
    if total_boundary_count > 0:
        boundary_baseline_c0 = boundary_confusion.support_negative / total_boundary_count
        boundary_baseline_c1 = boundary_confusion.support_positive / total_boundary_count
    else:
        boundary_baseline_c0 = None
        boundary_baseline_c1 = None

    if train_boundary_distribution is not None:
        maj_class = majority_class(train_boundary_distribution)
        if total_boundary_count > 0:
            boundary_baseline_maj = (
                boundary_confusion.support_negative / total_boundary_count
                if maj_class == 0
                else boundary_confusion.support_positive / total_boundary_count
            )
        else:
            boundary_baseline_maj = None
    else:
        boundary_baseline_maj = None

    def format_distribution(counts: Counter[int]) -> dict[str, int]:
        return {str(label): counts[label] for label in sorted(counts)}

    syllable_known_value = (
        safe_ratio(syllable_known_correct, syllable_known_total)
        if syllable_known_total
        else None
    )
    boundary_precision_value = confusion_precision(boundary_confusion)
    boundary_recall_value = confusion_recall(boundary_confusion)
    boundary_f1_value = f1_score(boundary_confusion)
    boundary_specificity_value = specificity(boundary_confusion)
    boundary_balanced_accuracy_value = balanced_accuracy(boundary_confusion)

    metric_results: dict[str, dict[str, Any]] = {}
    for name in ("char_loss", "syllable_loss", "boundary_loss", "total_loss"):
        metric_results[name] = metric_result(
            safe_ratio(totals[name], count),
            availability="available",
            reason=None,
            support={"examples": count},
        )
    char_accuracy_value = safe_ratio(totals["char_accuracy"], count)
    syllable_accuracy_all_value = safe_ratio(totals["syllable_accuracy"], count)
    boundary_accuracy_value = safe_ratio(totals["boundary_accuracy"], count)
    metric_results["char_accuracy"] = metric_result(
        char_accuracy_value,
        availability="available",
        reason=None,
        support={"examples": count},
    )
    metric_results["syllable_accuracy"] = metric_result(
        syllable_accuracy_all_value,
        availability="available",
        reason=None,
        support={"examples": count},
    )
    metric_results["syllable_accuracy_all"] = metric_result(
        syllable_accuracy_all_value,
        availability="available",
        reason=None,
        support={"examples": count},
    )
    if syllable_known_total:
        metric_results["syllable_accuracy_known"] = metric_result(
            syllable_known_value,
            availability="available",
            reason=None,
            support={"known_targets": syllable_known_total},
        )
    else:
        metric_results["syllable_accuracy_known"] = metric_result(
            None,
            availability="not_applicable",
            reason="no_known_targets",
            support={"known_targets": 0},
        )
    metric_results["boundary_accuracy"] = metric_result(
        boundary_accuracy_value,
        availability="available",
        reason=None,
        support={"examples": count},
    )
    metric_results["boundary_true_negative"] = metric_result(
        boundary_true_negative,
        availability="available",
        reason=None,
        support={"examples": count},
    )
    binary_metric_specs = (
        (
            "boundary_precision",
            boundary_precision_value,
            boundary_confusion.predicted_positive,
            "predicted_positive",
            "no_predicted_positive",
        ),
        (
            "boundary_recall",
            boundary_recall_value,
            boundary_confusion.support_positive,
            "gold_positive",
            "no_gold_positive",
        ),
        (
            "boundary_specificity",
            boundary_specificity_value,
            boundary_confusion.support_negative,
            "gold_negative",
            "no_gold_negative",
        ),
    )
    for name, value, support_count, support_name, reason in binary_metric_specs:
        if value is None:
            metric_results[name] = metric_result(
                None,
                availability="not_applicable",
                reason=reason,
                support={support_name: support_count},
            )
        else:
            metric_results[name] = metric_result(
                value,
                availability="available",
                reason=None,
                support={support_name: support_count},
            )
    if boundary_f1_value is None:
        metric_results["boundary_f1"] = metric_result(
            None,
            availability="not_applicable",
            reason="no_gold_or_predicted_positive",
            support={
                "gold_positive": boundary_confusion.support_positive,
                "predicted_positive": boundary_confusion.predicted_positive,
            },
        )
    else:
        metric_results["boundary_f1"] = metric_result(
            boundary_f1_value,
            availability="available",
            reason=None,
            support={
                "gold_positive": boundary_confusion.support_positive,
                "predicted_positive": boundary_confusion.predicted_positive,
            },
        )
    if boundary_balanced_accuracy_value is None:
        metric_results["boundary_balanced_accuracy"] = metric_result(
            None,
            availability="not_applicable",
            reason="missing_positive_or_negative_class",
            support={
                "gold_positive": boundary_confusion.support_positive,
                "gold_negative": boundary_confusion.support_negative,
            },
        )
    else:
        metric_results["boundary_balanced_accuracy"] = metric_result(
            boundary_balanced_accuracy_value,
            availability="available",
            reason=None,
            support={
                "gold_positive": boundary_confusion.support_positive,
                "gold_negative": boundary_confusion.support_negative,
            },
        )
    for name, value in (
        ("boundary_baseline_constant_0", boundary_baseline_c0),
        ("boundary_baseline_constant_1", boundary_baseline_c1),
    ):
        metric_results[name] = metric_result(
            value,
            availability="available",
            reason=None,
            support={"examples": count},
        )
    if train_boundary_distribution is None:
        metric_results["boundary_baseline_majority"] = metric_result(
            None,
            availability="unavailable",
            reason="training_distribution_missing",
            support={"examples": count},
        )
    else:
        metric_results["boundary_baseline_majority"] = metric_result(
            boundary_baseline_maj,
            availability="available",
            reason=None,
            support={"examples": count},
        )

    return {
        "examples": count,
        **{name: safe_ratio(value, count) for name, value in totals.items()},
        "syllable_accuracy_all": safe_ratio(totals["syllable_accuracy"], count),
        "syllable_accuracy_known": safe_ratio(
            syllable_known_correct, syllable_known_total
        ),
        "boundary_target_distribution": boundary_target_distribution,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "boundary_confusion": boundary_confusion.to_dict(),
        "boundary_true_negative": boundary_true_negative,
        "boundary_specificity": boundary_spec,
        "boundary_balanced_accuracy": boundary_bal_acc,
        "boundary_prediction_distribution": boundary_prediction_distribution,
        "boundary_baseline_constant_0": boundary_baseline_c0,
        "boundary_baseline_constant_1": boundary_baseline_c1,
        "boundary_baseline_majority": boundary_baseline_maj,
        "char_target_distribution": format_distribution(char_target_distribution),
        "char_prediction_distribution": format_distribution(char_prediction_distribution),
        "syllable_target_distribution": format_distribution(syllable_target_distribution),
        "syllable_prediction_distribution": format_distribution(syllable_prediction_distribution),
        "syllable_target_status_distribution": syllable_target_status_distribution,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "metric_results": metric_results,
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
