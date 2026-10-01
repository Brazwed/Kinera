"""Diagnóstico causal do ganho atribuído ao canal silábico."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from typing import Any

from language.vocabulary import CharacterVocabulary
from nn.probability import cross_entropy, softmax
from training.autoregressive import (
    AutoregressiveModel,
    example_to_inputs,
    train_multitask_examples,
)
from training.comparison import VARIANTS, prepare_variant_examples


PAIR_VARIANTS = {"A_vs_C": ("A", "C"), "B_vs_D": ("B", "D")}


def run_syllable_gain_analysis(
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
    top_examples: int = 20,
) -> dict[str, Any]:
    """Treina A/B/C/D e explica o ganho por grupos causais e linguísticos."""
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds deve conter inteiros")
    if isinstance(top_examples, bool) or not isinstance(top_examples, int) or top_examples <= 0:
        raise ValueError("top_examples deve ser inteiro positivo")
    train_selected = train_examples if max_examples is None else train_examples[:max_examples]
    validation_selected = (
        validation_examples if max_examples is None else validation_examples[:max_examples]
    )
    if not train_selected or not validation_selected:
        raise ValueError("treino e validação devem conter exemplos")

    prepared_train = {
        variant: prepare_variant_examples(
            train_selected,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        for variant in VARIANTS
    }
    prepared_validation = {
        variant: prepare_variant_examples(
            validation_selected,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        for variant in VARIANTS
    }
    syllable_frequencies: Counter[int] = Counter()
    for example in train_selected:
        target_syllable = example.get("target_syllable_id")
        if isinstance(target_syllable, int) and target_syllable != syllable_unknown_id:
            syllable_frequencies[target_syllable] += 1
    positions = _position_metadata(validation_selected)
    runs = []
    for seed in seeds:
        started = time.perf_counter()
        models = {}
        training = {}
        for variant in VARIANTS:
            model = AutoregressiveModel(len(vocabulary), syllable_vocab_size, seed=seed)
            training[variant] = train_multitask_examples(
                model,
                prepared_train[variant],
                vocabulary,
                epochs=epochs,
                learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id,
            )
            models[variant] = model
        rows = {variant: [] for variant in VARIANTS}
        for index, original in enumerate(validation_selected):
            for variant in VARIANTS:
                rows[variant].append(
                    _evaluate_row(
                        index,
                        original,
                        prepared_validation[variant][index],
                        models[variant],
                        vocabulary,
                        syllable_unknown_id,
                        positions[index],
                        syllable_frequencies,
                    )
                )
        pair_deltas = {}
        for pair_name, (base, augmented) in PAIR_VARIANTS.items():
            pair_rows = [_pair_row(rows[base][i], rows[augmented][i]) for i in range(len(validation_selected))]
            pair_deltas[pair_name] = {
                "overall": _summarize_pair(pair_rows),
                "by_syllable_info": _grouped_pair_summary(pair_rows, "syllable_info"),
            }
        runs.append({
            "seed": seed,
            "status": "completed",
            "duration_seconds": time.perf_counter() - started,
            "training": training,
            "coverage": _coverage(rows["C"]),
            "buckets": {
                "word_position": _bucket_pair_summaries(rows, "word_position"),
                "boundary_proximity": _bucket_pair_summaries(rows, "boundary_proximity"),
                "syllable_frequency": _bucket_pair_summaries(rows, "syllable_frequency"),
            },
            "pair_deltas": pair_deltas,
            "target_char_effects": _target_char_effects(rows),
            "examples": _representative_examples(rows, top_examples),
        })
    return {
        "phase": "5.3_syllable_gain_analysis",
        "variants": list(VARIANTS),
        "seeds": seeds.copy(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_examples": max_examples,
        "syllable_unknown_id": syllable_unknown_id,
        "completed_runs": len(runs),
        "runs": runs,
    }


def _evaluate_row(
    index: int,
    original: dict[str, Any],
    prepared: dict[str, Any],
    model: AutoregressiveModel,
    vocabulary: CharacterVocabulary,
    syllable_unknown_id: int,
    position: dict[str, Any],
    syllable_frequencies: Counter[int],
) -> dict[str, Any]:
    inputs = example_to_inputs(
        prepared,
        vocabulary,
        syllable_unknown_id=syllable_unknown_id,
    )
    outputs = model.forward_heads(*inputs)
    target = int(original["target_char_id"])
    probabilities = softmax(outputs["char"])
    prediction = max(range(len(outputs["char"])), key=outputs["char"].__getitem__)
    syllable_ids = prepared.get("context_syllable_ids", [])
    active_count = sum(value != syllable_unknown_id for value in syllable_ids)
    target_syllable = original.get("target_syllable_id")
    frequency = (
        syllable_frequencies.get(target_syllable, 0)
        if isinstance(target_syllable, int)
        else 0
    )
    if target_syllable == syllable_unknown_id:
        frequency_bucket = "unknown_target"
    elif frequency == 0:
        frequency_bucket = "unseen_in_train"
    elif frequency <= 2:
        frequency_bucket = "rare"
    else:
        frequency_bucket = "frequent"
    if active_count == 0:
        syllable_info = "unavailable"
    else:
        syllable_info = "active"
    if syllable_ids and syllable_ids[-1] != syllable_unknown_id:
        boundary_proximity = "after_completed_or_completed_context"
    elif active_count:
        boundary_proximity = "inside_next_syllable"
    else:
        boundary_proximity = "unavailable"
    return {
        "index": index,
        "word_key": original.get("word_key"),
        "target_char": original.get("target_char"),
        "target_char_id": target,
        "loss": cross_entropy(outputs["char"], target),
        "correct": prediction == target,
        "prob_correct": probabilities[target],
        "syllable_info": syllable_info,
        "active_syllable_positions": active_count,
        "word_position": position["word_position"],
        "boundary_proximity": boundary_proximity,
        "syllable_frequency": frequency_bucket,
    }


def _pair_row(base: dict[str, Any], augmented: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": base["index"],
        "word_key": base["word_key"],
        "target_char": base["target_char"],
        # A/B mascaram o canal silábico; buckets devem refletir C/D.
        "syllable_info": augmented["syllable_info"],
        "word_position": base["word_position"],
        "boundary_proximity": augmented["boundary_proximity"],
        "syllable_frequency": base["syllable_frequency"],
        "base_loss": base["loss"],
        "augmented_loss": augmented["loss"],
        "delta_loss": base["loss"] - augmented["loss"],
        "base_correct": base["correct"],
        "augmented_correct": augmented["correct"],
        "delta_accuracy": int(augmented["correct"]) - int(base["correct"]),
        "base_prob_correct": base["prob_correct"],
        "augmented_prob_correct": augmented["prob_correct"],
    }


def _summarize_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0, "mean_delta_loss": 0.0, "accuracy_delta": 0.0}
    return {
        "examples": len(rows),
        "mean_delta_loss": sum(row["delta_loss"] for row in rows) / len(rows),
        "accuracy_delta": sum(row["delta_accuracy"] for row in rows) / len(rows),
        "improved_loss_fraction": sum(row["delta_loss"] > 0 for row in rows) / len(rows),
        "regressed_loss_fraction": sum(row["delta_loss"] < 0 for row in rows) / len(rows),
    }


def _grouped_pair_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _summarize_pair(group) for name, group in sorted(groups.items())}


def _bucket_pair_summaries(rows: dict[str, list[dict[str, Any]]], key: str) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for pair_name, (base, augmented) in PAIR_VARIANTS.items():
        pair_rows = [_pair_row(rows[base][i], rows[augmented][i]) for i in range(len(rows[base]))]
        result[pair_name] = _grouped_pair_summary(pair_rows, key)
    return result


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(row["syllable_info"] for row in rows)
    total = len(rows)
    return {
        "examples": total,
        "active": groups["active"],
        "unavailable": groups["unavailable"],
        "active_fraction": groups["active"] / total if total else 0.0,
    }


def _representative_examples(rows: dict[str, list[dict[str, Any]]], limit: int) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for pair_name, (base, augmented) in PAIR_VARIANTS.items():
        pair_rows = [_pair_row(rows[base][i], rows[augmented][i]) for i in range(len(rows[base]))]
        result[pair_name] = {
            "gain": sorted(pair_rows, key=lambda row: row["delta_loss"], reverse=True)[:limit],
            "regression": sorted(pair_rows, key=lambda row: row["delta_loss"])[:limit],
        }
    return result


def _target_char_effects(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {}
    for pair_name, (base, augmented) in PAIR_VARIANTS.items():
        pair_rows = [_pair_row(rows[base][i], rows[augmented][i]) for i in range(len(rows[base]))]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            grouped[str(row["target_char"])].append(row)
        effects = []
        for target, group in grouped.items():
            effects.append({
                "target_char": target,
                "examples": len(group),
                "mean_delta_loss": sum(row["delta_loss"] for row in group) / len(group),
            })
        result[pair_name] = {
            "most_helped": sorted(effects, key=lambda row: row["mean_delta_loss"], reverse=True)[:10],
            "most_hurt": sorted(effects, key=lambda row: row["mean_delta_loss"])[:10],
        }
    return result


def _position_metadata(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[str(example.get("word_key", ""))].append(index)
    metadata = [{} for _ in examples]
    for indices in grouped.values():
        non_end = [index for index in indices if examples[index].get("target_char") != "<END>"]
        for position, index in enumerate(non_end):
            if position == 0:
                bucket = "start"
            elif position == len(non_end) - 1:
                bucket = "end"
            else:
                bucket = "middle"
            metadata[index] = {"word_position": bucket}
        for index in indices:
            if not metadata[index]:
                metadata[index] = {"word_position": "end"}
    return metadata
