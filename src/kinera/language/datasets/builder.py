"""Deterministic language dataset grouping and example builders."""

from __future__ import annotations

import copy
import random
from typing import Any

from kinera.language.orthography.features import features_to_vector
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


def group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrupa análises da mesma palavra sem perder fontes ou variantes."""
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("cada registro deve ser um dicionário")
        word_key = record.get("word_key")
        if not isinstance(word_key, str) or not word_key:
            raise ValueError("registro sem word_key")
        if word_key not in groups:
            groups[word_key] = {
                "word_key": word_key,
                "word_raw": record.get("word_raw", word_key),
                "analyses": [],
            }
        groups[word_key]["analyses"].append(copy.deepcopy(record))

    result = []
    for group in groups.values():
        signatures = {
            tuple(analysis.get("syllables", []))
            for analysis in group["analyses"]
        }
        group["distinct_syllabifications"] = len(signatures)
        group["conflict"] = len(signatures) > 1
        result.append(group)
    return result


def split_by_word(
    groups: list[dict[str, Any]],
    validation_ratio: float,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Divide grupos mantendo todas as análises de uma palavra no mesmo split."""
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("groups deve conter pelo menos dois grupos")
    if isinstance(validation_ratio, bool) or not isinstance(validation_ratio, (int, float)):
        raise TypeError("validation_ratio deve ser numérico")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio deve estar entre 0 e 1")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed deve ser um inteiro ou None")

    indices = list(range(len(groups)))
    random.Random(seed).shuffle(indices)
    validation_count = max(
        1,
        min(len(groups) - 1, int(len(groups) * validation_ratio)),
    )
    validation_indices = set(indices[:validation_count])
    train = [copy.deepcopy(group) for index, group in enumerate(groups) if index not in validation_indices]
    validation = [copy.deepcopy(group) for index, group in enumerate(groups) if index in validation_indices]
    return train, validation


def build_autoregressive_examples(
    records: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    context_size: int,
    syllable_vocabulary: SyllableVocabulary | None = None,
) -> list[dict[str, Any]]:
    """Converte anotações em exemplos de previsão do próximo caractere."""
    if isinstance(context_size, bool) or not isinstance(context_size, int):
        raise TypeError("context_size deve ser inteiro")
    if context_size <= 0:
        raise ValueError("context_size deve ser positivo")

    examples: list[dict[str, Any]] = []
    for record in records:
        word_key = record.get("word_key", record.get("word_normalized", ""))
        characters = record.get("characters")
        if not isinstance(word_key, str) or not word_key:
            raise ValueError("registro sem word_key")
        if not isinstance(characters, list) or not characters:
            raise ValueError("registro sem characters anotados")

        char_ids = [vocabulary.encode(item["char"])[0] for item in characters]
        context_syllable_ids: list[int] | None = None
        context_feature_vectors: list[list[float]] | None = None
        if syllable_vocabulary is not None:
            if not isinstance(syllable_vocabulary, SyllableVocabulary):
                raise TypeError("syllable_vocabulary inválido")
            char_feature_vectors = []
            for item in characters:
                char_feature_vectors.append(features_to_vector(item))
            feature_sequence = (
                [[0.0] * 10 for _ in range(context_size)]
                + char_feature_vectors
                + [[0.0] * 10]
            )
        sequence = [vocabulary.start_id] * context_size + char_ids + [vocabulary.end_id]
        for target_position in range(context_size, len(sequence)):
            target_id = sequence[target_position]
            is_end = target_id == vocabulary.end_id
            char_index = target_position - context_size
            target_item = None if is_end else characters[char_index]
            example = {
                    "word_key": word_key,
                    "context_ids": sequence[
                        target_position - context_size : target_position
                    ],
                    "target_char_id": target_id,
                    "target_char": vocabulary.id_to_token[target_id],
                    "target_syllable_index": (
                        None
                        if target_item is None
                        else target_item.get("syllable_index")
                    ),
                    "target_is_syllable_end": (
                        True
                        if target_item is None
                        else bool(target_item.get("is_syllable_end", False))
                    ),
                }
            if syllable_vocabulary is not None:
                observed_count = len(characters) if is_end else char_index
                completed_syllables = {
                    item.get("syllable_index")
                    for item in characters[:observed_count]
                    if (
                        isinstance(item, dict)
                        and bool(item.get("is_syllable_end", False))
                        and isinstance(item.get("syllable_index"), int)
                        and not isinstance(item.get("syllable_index"), bool)
                    )
                }
                causal_syllable_ids = []
                syllables = record.get("syllables", [])
                for item in characters:
                    syllable_index = item.get("syllable_index")
                    if (
                        isinstance(syllable_index, bool)
                        or not isinstance(syllable_index, int)
                        or syllable_index not in completed_syllables
                        or not 0 <= syllable_index < len(syllables)
                    ):
                        causal_syllable_ids.append(syllable_vocabulary.unk_id)
                    else:
                        causal_syllable_ids.append(
                            syllable_vocabulary.encode(syllables[syllable_index])
                        )
                syllable_sequence = (
                    [syllable_vocabulary.unk_id] * context_size
                    + causal_syllable_ids
                    + [syllable_vocabulary.unk_id]
                )
                context_syllable_ids = syllable_sequence[
                    target_position - context_size : target_position
                ]
                context_feature_vectors = feature_sequence[
                    target_position - context_size : target_position
                ]
                example["context_syllable_ids"] = context_syllable_ids
                example["context_feature_vectors"] = context_feature_vectors
                if target_item is None:
                    example["target_syllable_id"] = syllable_vocabulary.unk_id
                else:
                    target_syllable_index = target_item.get("syllable_index")
                    syllables = record.get("syllables", [])
                    if (
                        isinstance(target_syllable_index, bool)
                        or not isinstance(target_syllable_index, int)
                        or not 0 <= target_syllable_index < len(syllables)
                    ):
                        example["target_syllable_id"] = syllable_vocabulary.unk_id
                    else:
                        example["target_syllable_id"] = syllable_vocabulary.encode(
                            syllables[target_syllable_index]
                        )
            examples.append(example)
    return examples
