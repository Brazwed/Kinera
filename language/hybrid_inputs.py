"""Converte registros anotados em entradas alinhadas da representação híbrida."""

from __future__ import annotations

from typing import Any

from language.features import features_to_vector
from language.vocabulary import CharacterVocabulary, SyllableVocabulary


def record_to_hybrid_inputs(
    record: dict[str, Any],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Gera IDs e features na mesma ordem dos caracteres anotados."""
    if not isinstance(record, dict):
        raise TypeError("record deve ser um dicionário")
    if not isinstance(character_vocabulary, CharacterVocabulary):
        raise TypeError("character_vocabulary inválido")
    if not isinstance(syllable_vocabulary, SyllableVocabulary):
        raise TypeError("syllable_vocabulary inválido")

    characters = record.get("characters")
    syllables = record.get("syllables", [])
    if not isinstance(characters, list):
        raise ValueError("record deve conter characters como lista")
    if not isinstance(syllables, list):
        raise ValueError("record deve conter syllables como lista")

    char_ids: list[int] = []
    syllable_ids: list[int] = []
    feature_vectors: list[list[float]] = []
    for position, item in enumerate(characters):
        if not isinstance(item, dict) or not isinstance(item.get("char"), str):
            raise ValueError(f"characters[{position}] inválido")
        char_ids.append(character_vocabulary.encode(item["char"])[0])
        syllable_index = item.get("syllable_index")
        if (
            isinstance(syllable_index, bool)
            or not isinstance(syllable_index, int)
            or not 0 <= syllable_index < len(syllables)
        ):
            syllable_ids.append(syllable_vocabulary.unk_id)
        else:
            syllable_ids.append(syllable_vocabulary.encode(syllables[syllable_index]))
        feature_vectors.append(features_to_vector(item))
    return char_ids, syllable_ids, feature_vectors
