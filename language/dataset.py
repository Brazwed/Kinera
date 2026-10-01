"""Conciliação, agrupamento e divisão de registros linguísticos."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

from language.vocabulary import CharacterVocabulary


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"linha JSONL {line_number} não é objeto")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return destination


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
        sequence = [vocabulary.start_id] * context_size + char_ids + [vocabulary.end_id]
        for target_position in range(context_size, len(sequence)):
            target_id = sequence[target_position]
            is_end = target_id == vocabulary.end_id
            char_index = target_position - context_size
            target_item = None if is_end else characters[char_index]
            examples.append(
                {
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
            )
    return examples
