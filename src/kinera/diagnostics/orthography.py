"""Análise do efeito silábico por classe ortográfica."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from kinera.monitor.metrics.core import oriented_delta, summarize_deltas, summarize_values
from kinera.monitor.metrics.schema import METRICS_SCHEMA_VERSION, metric_result

from kinera.language.orthography.features import character_features
from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.losses.cross_entropy import cross_entropy, softmax
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.trainer import train_multitask_examples
from kinera.experiments.comparison import CONTROLLED_VARIANTS, prepare_variant_examples

PAIR_VARIANTS = {"A_vs_C": ("A", "C"), "B_vs_D": ("B", "D")}
CLASS_NAMES = ("vogal", "consoante", "vogal_acentuada", "<END>", "separador_pontuacao", "outros")


def classify_target(target: str | None) -> str:
    if target == "<END>":
        return "<END>"
    if not isinstance(target, str) or len(target) != 1:
        return "outros"
    features = character_features(target)
    if features["is_separator"] or features["is_punctuation"]:
        return "separador_pontuacao"
    if features["is_vowel"] and features["is_accented"]:
        return "vogal_acentuada"
    if features["is_vowel"]:
        return "vogal"
    if features["is_consonant"]:
        return "consoante"
    return "outros"


def run_orthographic_class_analysis(
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
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    train = train_examples if max_examples is None else train_examples[:max_examples]
    validation = validation_examples if max_examples is None else validation_examples[:max_examples]
    if not train or not validation:
        raise ValueError("treino e validação devem conter exemplos")
    prepared_train = {
        variant: prepare_variant_examples(train, variant, vocabulary, syllable_unknown_id=syllable_unknown_id)
        for variant in CONTROLLED_VARIANTS
    }
    prepared_validation = {
        variant: prepare_variant_examples(validation, variant, vocabulary, syllable_unknown_id=syllable_unknown_id)
        for variant in CONTROLLED_VARIANTS
    }
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        models = {}
        for variant in CONTROLLED_VARIANTS:
            model = AutoregressiveModel(len(vocabulary), syllable_vocab_size, seed=seed)
            train_multitask_examples(
                model,
                prepared_train[variant],
                vocabulary,
                epochs=epochs,
                learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id,
            )
            models[variant] = model
        rows = {variant: [] for variant in CONTROLLED_VARIANTS}
        for index, original in enumerate(validation):
            target = int(original["target_char_id"])
            target_class = classify_target(original.get("target_char"))
            for variant in CONTROLLED_VARIANTS:
                inputs = (
                    prepared_validation[variant][index]["context_ids"],
                    prepared_validation[variant][index]["context_syllable_ids"],
                    prepared_validation[variant][index]["context_feature_vectors"],
                )
                logits = models[variant].forward(*inputs)
                probabilities = softmax(logits)
                prediction = max(range(len(logits)), key=logits.__getitem__)
                rows[variant].append({
                    "index": index,
                    "target_class": target_class,
                    "loss": cross_entropy(logits, target),
                    "correct": prediction == target,
                    "prob_correct": probabilities[target],
                })
        comparisons = {}
        for pair_name, (base, augmented) in PAIR_VARIANTS.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for index in range(len(validation)):
                left, right = rows[base][index], rows[augmented][index]
                grouped[left["target_class"]].append({
                    "base_loss": left["loss"],
                    "augmented_loss": right["loss"],
                    "delta_loss": oriented_delta(
                        left["loss"], right["loss"], orientation="base - candidate"
                    ),
                    "delta_accuracy": oriented_delta(
                        int(left["correct"]),
                        int(right["correct"]),
                        orientation="candidate - base",
                    ),
                    "base_prob_correct": left["prob_correct"],
                    "augmented_prob_correct": right["prob_correct"],
                })
            comparisons[pair_name] = {
                class_name: _summarize(rows_for_class)
                for class_name, rows_for_class in sorted(grouped.items())
            }
        runs.append({
            "seed": seed,
            "status": "completed",
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "duration_seconds": time.perf_counter() - started,
            "comparisons": comparisons,
        })
    return {
        "phase": "5.5_orthographic_class_analysis",
        "classes": list(CLASS_NAMES),
        "seeds": seeds.copy(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_examples": max_examples,
        "runs": runs,
        "completed_runs": len(runs),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    loss_deltas = summarize_deltas([row["delta_loss"] for row in rows])
    accuracy_deltas = summarize_values([row["delta_accuracy"] for row in rows])
    base_probabilities = summarize_values([row["base_prob_correct"] for row in rows])
    augmented_probabilities = summarize_values(
        [row["augmented_prob_correct"] for row in rows]
    )
    legacy: dict[str, Any] = {"count": count}
    if count:
        legacy.update({
            "mean_delta_loss": loss_deltas["mean"],
            "delta_accuracy": accuracy_deltas["mean"],
            "mean_base_prob_correct": base_probabilities["mean"],
            "mean_augmented_prob_correct": augmented_probabilities["mean"],
            "improved_loss_fraction": loss_deltas["positive_fraction"],
            "regressed_loss_fraction": loss_deltas["negative_fraction"],
        })
    legacy["metric_results"] = _metric_results(
        {
            "mean_delta_loss": loss_deltas["mean"],
            "delta_accuracy": accuracy_deltas["mean"],
            "mean_base_prob_correct": base_probabilities["mean"],
            "mean_augmented_prob_correct": augmented_probabilities["mean"],
            "improved_loss_fraction": loss_deltas["positive_fraction"],
            "regressed_loss_fraction": loss_deltas["negative_fraction"],
        },
        {"examples": count},
    )
    return legacy


def _metric_results(
    values: dict[str, float | int | None], support: dict[str, int]
) -> dict[str, dict[str, Any]]:
    available = support["examples"] > 0
    return {
        name: metric_result(
            value=value if available else None,
            availability="available" if available else "not_applicable",
            reason=None if available else "empty_support",
            support=support,
        )
        for name, value in values.items()
    }
