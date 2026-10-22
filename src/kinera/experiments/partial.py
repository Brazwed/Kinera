"""Comparação da representação silábica concluída contra a parcial causal."""

from __future__ import annotations

import time
from typing import Any

from kinera.monitor.metrics.aggregation import aggregate_metric_values, mean_std

from kinera.language.syllables.causal import prepare_partial_variant_examples
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples
from kinera.experiments.comparison import (
    CONTROLLED_VARIANTS as BASE_VARIANTS,
    count_trainable_parameters,
    prepare_variant_examples,
)

PARTIAL_VARIANTS = (*BASE_VARIANTS, "E", "F")


def prepare_partial_experiment_examples(
    examples: list[dict[str, Any]],
    variant: str,
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
) -> list[dict[str, Any]]:
    if variant in BASE_VARIANTS:
        return prepare_variant_examples(
            examples,
            variant,
            character_vocabulary,
            syllable_unknown_id=syllable_vocabulary.unk_id,
        )
    return prepare_partial_variant_examples(
        examples,
        variant,
        character_vocabulary,
        syllable_vocabulary,
    )


def run_partial_comparison(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    *,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Executa A/B/C/D/E/F com heads e loss idênticas."""
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    train = train_examples if max_examples is None else train_examples[:max_examples]
    validation = validation_examples if max_examples is None else validation_examples[:max_examples]
    if not train or not validation:
        raise ValueError("treino e validação devem conter exemplos")
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        metrics = {}
        parameter_counts = {}
        for variant in PARTIAL_VARIANTS:
            prepared_train = prepare_partial_experiment_examples(
                train, variant, character_vocabulary, syllable_vocabulary
            )
            prepared_validation = prepare_partial_experiment_examples(
                validation, variant, character_vocabulary, syllable_vocabulary
            )
            model = AutoregressiveModel(
                len(character_vocabulary),
                len(syllable_vocabulary),
                seed=seed,
            )
            parameter_counts[variant] = count_trainable_parameters(model)
            training = train_multitask_examples(
                model,
                prepared_train,
                character_vocabulary,
                epochs=epochs,
                learning_rate=learning_rate,
                syllable_unknown_id=syllable_vocabulary.unk_id,
            )
            validation_metrics = evaluate_multitask_examples(
                model,
                prepared_validation,
                character_vocabulary,
                syllable_unknown_id=syllable_vocabulary.unk_id,
            )
            metrics[variant] = {
                "training": training,
                "validation": validation_metrics,
                "parameter_count": parameter_counts[variant],
            }
        runs.append({
            "seed": seed,
            "status": "completed",
            "metrics": metrics,
            "duration_seconds": time.perf_counter() - started,
        })
    aggregates = {}
    metric_names = ("char_loss", "char_accuracy", "total_loss")
    for variant in PARTIAL_VARIANTS:
        aggregates[variant] = {"mean": {}, "std": {}}
        for metric in metric_names:
            observations = [
                (run["seed"], run["metrics"][variant]["validation"].get(metric))
                for run in runs
            ]
            summary = aggregate_metric_values(observations)
            aggregates[variant]["mean"][metric] = summary.mean
            aggregates[variant]["std"][metric] = summary.std
    parameter_counts = {
        variant: runs[0]["metrics"][variant]["parameter_count"]
        for variant in PARTIAL_VARIANTS
    }
    return {
        "phase": "5.4_partial_syllable_comparison",
        "variants": list(PARTIAL_VARIANTS),
        "partial_variants": {"E": "characters + causal partial syllable", "F": "characters + features + causal partial syllable"},
        "seeds": seeds.copy(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_examples": max_examples,
        "parameter_counts": parameter_counts,
        "runs": runs,
        "aggregates": aggregates,
    }
