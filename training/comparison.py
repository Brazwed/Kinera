"""Experimentos A/B/C/D com controles de entrada equivalentes."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

from language.vocabulary import CharacterVocabulary
from training.autoregressive import (
    AutoregressiveModel,
    evaluate_multitask_examples,
    train_multitask_examples,
)

VARIANTS = ("A", "B", "C", "D")


def prepare_variant_examples(
    examples: list[dict[str, Any]],
    variant: str,
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int,
) -> list[dict[str, Any]]:
    """Mascara canais de entrada sem alterar targets ou shapes."""
    if variant not in VARIANTS:
        raise ValueError(f"variante desconhecida: {variant}")
    prepared = []
    for original in examples:
        example = copy.deepcopy(original)
        context_ids = example.get("context_ids")
        if not isinstance(context_ids, list) or not context_ids:
            raise ValueError("example deve conter context_ids não vazio")
        context_size = len(context_ids)
        feature_vectors = example.get("context_feature_vectors")
        if not isinstance(feature_vectors, list) or len(feature_vectors) != context_size:
            feature_vectors = [[0.0] * 10 for _ in range(context_size)]
        if variant in ("A", "C"):
            feature_vectors = [[0.0] * len(vector) for vector in feature_vectors]
        syllable_ids = example.get("context_syllable_ids")
        if not isinstance(syllable_ids, list) or len(syllable_ids) != context_size:
            syllable_ids = [syllable_unknown_id] * context_size
        if variant in ("A", "B"):
            syllable_ids = [syllable_unknown_id] * context_size
        example["context_feature_vectors"] = feature_vectors
        example["context_syllable_ids"] = syllable_ids
        prepared.append(example)
    return prepared


def count_trainable_parameters(model: AutoregressiveModel) -> int:
    """Conta escalares treináveis, mantendo a comparação explícita."""
    total = 0
    for table in (
        model.representation.char_embedding.weights,
        model.representation.syllable_embedding.weights,
        model.heads.char.weights,
        model.heads.char.bias,
        model.heads.syllable.weights,
        model.heads.syllable.bias,
        model.heads.boundary.weights,
        model.heads.boundary.bias,
    ):
        total += sum(len(row) if isinstance(row, list) else 1 for row in table)
    return total


def run_controlled_experiments(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_vocab_size: int,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    max_examples: int | None = None,
    syllable_unknown_id: int = 0,
) -> dict[str, Any]:
    """Executa A/B/C/D com os mesmos controles e agrega média/desvio."""
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds deve conter inteiros")
    reports: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    parameter_counts: dict[str, int] = {}
    input_usage: dict[str, dict[str, int]] = {}
    metric_names = (
        "char_loss",
        "syllable_loss",
        "boundary_loss",
        "total_loss",
        "char_accuracy",
        "syllable_accuracy_all",
        "syllable_accuracy_known",
        "boundary_accuracy",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
    )

    for variant in VARIANTS:
        prepared_train = prepare_variant_examples(
            train_examples,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        prepared_validation = prepare_variant_examples(
            validation_examples,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        if max_examples is not None:
            prepared_train = prepared_train[:max_examples]
            prepared_validation = prepared_validation[:max_examples]
        input_usage[variant] = _input_usage(
            prepared_train,
            prepared_validation,
            syllable_unknown_id,
        )
        reports[variant] = []
        final_metrics: list[dict[str, float]] = []
        for seed in seeds:
            started = time.perf_counter()
            model = AutoregressiveModel(
                len(vocabulary),
                syllable_vocab_size,
                seed=seed,
            )
            parameter_counts[variant] = count_trainable_parameters(model)
            initial = evaluate_multitask_examples(
                model,
                prepared_validation,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
            )
            training = train_multitask_examples(
                model,
                prepared_train,
                vocabulary,
                epochs=epochs,
                learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id,
            )
            final = evaluate_multitask_examples(
                model,
                prepared_validation,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
            )
            final_metrics.append({name: float(final[name]) for name in metric_names})
            reports[variant].append({
                "seed": seed,
                "initial_validation": initial,
                "training": training,
                "final_validation": final,
                "parameter_count": parameter_counts[variant],
                "duration_seconds": time.perf_counter() - started,
            })
        aggregates[variant] = {"mean": {}, "std": {}}
        for name in metric_names:
            values = [metrics[name] for metrics in final_metrics]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            aggregates[variant]["mean"][name] = mean
            aggregates[variant]["std"][name] = math.sqrt(variance)

    return {
        "variants": list(VARIANTS),
        "seeds": seeds.copy(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_examples": max_examples,
        "loss_weights": {"char": 1.0, "syllable": 0.7, "boundary": 0.3},
        "runs": reports,
        "aggregates": aggregates,
        "parameter_counts": parameter_counts,
        "input_usage": input_usage,
    }


def _input_usage(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    syllable_unknown_id: int,
) -> dict[str, int]:
    def measure(examples: list[dict[str, Any]]) -> tuple[int, int, int]:
        positions = 0
        non_unknown = 0
        unique_ids: set[int] = set()
        for example in examples:
            ids = example.get("context_syllable_ids", [])
            positions += len(ids)
            for value in ids:
                if value != syllable_unknown_id:
                    non_unknown += 1
                    unique_ids.add(value)
        return positions, non_unknown, len(unique_ids)

    train_positions, train_non_unknown, train_unique = measure(train_examples)
    validation_positions, validation_non_unknown, validation_unique = measure(
        validation_examples
    )
    return {
        "train_context_positions": train_positions,
        "train_non_unk_context_positions": train_non_unknown,
        "train_unique_syllable_ids": train_unique,
        "validation_context_positions": validation_positions,
        "validation_non_unk_context_positions": validation_non_unknown,
        "validation_unique_syllable_ids": validation_unique,
    }
