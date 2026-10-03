"""Features ortográficas e anotação caractere-sílaba."""

from __future__ import annotations

import copy
import unicodedata
from typing import Any

from kinera.language.orthography.alignment import align_characters, alignment_summary
from kinera.language.syllables.tokenizer import VOWELS

ORTHOGRAPHIC_FEATURE_NAMES = (
    "is_vowel",
    "is_consonant",
    "is_accented",
    "is_alpha",
    "is_digit",
    "is_space",
    "is_punctuation",
    "is_upper",
    "is_lower",
    "is_separator",
)


def character_features(char: str) -> dict[str, Any]:
    """Extrai flags estruturais de um caractere Unicode."""
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("char deve conter exatamente um caractere")
    normalized = unicodedata.normalize("NFC", char)
    folded = normalized.casefold()
    category = unicodedata.category(normalized)
    is_alpha = normalized.isalpha()
    is_vowel = folded in VOWELS
    return {
        "char": normalized,
        "normalized_char": folded,
        "unicode_category": category,
        "is_vowel": is_vowel,
        "is_consonant": is_alpha and not is_vowel,
        "is_accented": _has_combining_mark(normalized),
        "is_alpha": is_alpha,
        "is_digit": normalized.isdigit(),
        "is_space": normalized.isspace(),
        "is_punctuation": category.startswith("P"),
        "is_upper": normalized.isupper(),
        "is_lower": normalized.islower(),
        "is_separator": normalized in {"-", "|"},
    }


def features_to_vector(features: dict[str, Any]) -> list[float]:
    """Converte features ortográficas em vetor binário de ordem estável."""
    if not isinstance(features, dict):
        raise TypeError("features deve ser um dicionário")
    return [1.0 if bool(features.get(name, False)) else 0.0 for name in ORTHOGRAPHIC_FEATURE_NAMES]


def annotate_word(word: str, syllables: list[str]) -> list[dict[str, Any]]:
    """Combina features ortográficas com o alinhamento de sílabas."""
    aligned = align_characters(word, syllables)
    annotated = []
    for item in aligned:
        enriched = dict(item)
        enriched.update(character_features(item["char"]))
        annotated.append(enriched)
    return annotated


def annotate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Adiciona alinhamento e features sem alterar o registro original."""
    if not isinstance(record, dict):
        raise TypeError("record deve ser um dicionário")
    word = record.get("word_normalized", record.get("word_raw", ""))
    syllables = record.get("syllables", [])
    result = copy.deepcopy(record)
    result["characters"] = annotate_word(word, syllables)
    result["alignment"] = alignment_summary(word, syllables)
    return result


def _has_combining_mark(char: str) -> bool:
    decomposed = unicodedata.normalize("NFD", char)
    return any(unicodedata.category(part) == "Mn" for part in decomposed[1:])
