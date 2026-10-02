"""Análise do efeito silábico por classe ortográfica."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from language.features import character_features
from language.vocabulary import CharacterVocabulary
from nn.probability import cross_entropy, softmax
from training.autoregressive import AutoregressiveModel, train_multitask_examples
from training.comparison import CONTROLLED_VARIANTS, prepare_variant_examples

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
                    "delta_loss": left["loss"] - right["loss"],
                    "delta_accuracy": int(right["correct"]) - int(left["correct"]),
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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "mean_delta_loss": sum(row["delta_loss"] for row in rows) / count,
        "delta_accuracy": sum(row["delta_accuracy"] for row in rows) / count,
        "mean_base_prob_correct": sum(row["base_prob_correct"] for row in rows) / count,
        "mean_augmented_prob_correct": sum(row["augmented_prob_correct"] for row in rows) / count,
        "improved_loss_fraction": sum(row["delta_loss"] > 0 for row in rows) / count,
        "regressed_loss_fraction": sum(row["delta_loss"] < 0 for row in rows) / count,
    }
