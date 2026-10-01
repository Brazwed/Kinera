"""Normalização de registros lexicais provenientes de fontes externas."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


_SYLLABLE_SEPARATOR = re.compile(r"[-|]")
_SOURCE_MARKERS = ("__", "**")


def normalize_word(value: str) -> tuple[str, str]:
    """Retorna texto NFC preservado e chave casefold estável."""
    if not isinstance(value, str):
        raise TypeError("word deve ser uma string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError("word não pode ser vazio")
    return normalized, normalized.casefold()


def parse_syllable_markup(value: str) -> tuple[list[str], int | None]:
    """Remove separadores/marcadores do campo syl e identifica tonicidade."""
    if not isinstance(value, str):
        raise TypeError("syl deve ser uma string")
    raw = value.strip()
    if not raw:
        return [], None

    syllables: list[str] = []
    stressed_index: int | None = None
    for index, token in enumerate(_SYLLABLE_SEPARATOR.split(raw)):
        cleaned = token.strip()
        is_stressed = "**" in cleaned
        for marker in _SOURCE_MARKERS:
            cleaned = cleaned.replace(marker, "")
        cleaned = unicodedata.normalize("NFC", cleaned.strip())
        if not cleaned:
            continue
        syllables.append(cleaned)
        if is_stressed and stressed_index is None:
            stressed_index = len(syllables) - 1
    return syllables, stressed_index


def normalize_silabas_row(
    row: dict[str, Any],
    *,
    source: str = "silabas-pt",
) -> dict[str, Any]:
    """Converte uma linha do CSV para o schema normalizado da Kinera."""
    if not isinstance(row, dict):
        raise TypeError("row deve ser um dicionário")
    word_raw = row.get("word", "")
    word_normalized, word_key = normalize_word(word_raw)
    syllable_text_raw = row.get("syl", "")
    syllables, markup_stressed_index = parse_syllable_markup(syllable_text_raw)
    quality_flags: list[str] = []

    stressed_raw = row.get("syl_strong", "")
    stressed_index = _parse_optional_int(stressed_raw)
    if stressed_index is None:
        stressed_index = markup_stressed_index
    elif markup_stressed_index is not None and stressed_index != markup_stressed_index:
        quality_flags.append("stress_sources_disagree")

    positions_raw = row.get("syl_positions", "")
    positions = _parse_positions(positions_raw, quality_flags)
    if not syllables:
        quality_flags.append("missing_syllables")
    if stressed_index is not None and not 0 <= stressed_index < len(syllables):
        quality_flags.append("stress_index_out_of_range")
        stressed_index = None

    comparable_word = re.sub(r"[-|]", "", word_normalized)
    reconstructed = "".join(syllables)
    alignment_status = "valid"
    if syllables and comparable_word != reconstructed:
        alignment_status = "mismatch"
        quality_flags.append("syllable_reconstruction_mismatch")

    return {
        "word_raw": word_raw,
        "word_normalized": word_normalized,
        "word_key": word_key,
        "syllables": syllables,
        "syllable_text_raw": syllable_text_raw,
        "stressed_syllable_index": stressed_index,
        "stressed_syllable_raw": stressed_raw,
        "category": row.get("category") or None,
        "ipa": row.get("ipa") or None,
        "variant": row.get("variant") or None,
        "source": source,
        "source_record_id": row.get("id") or None,
        "source_fields": dict(row),
        "syl_positions": positions,
        "alignment_status": alignment_status,
        "quality_flags": quality_flags,
    }


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_positions(value: Any, quality_flags: list[str]) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            quality_flags.append("invalid_syllable_positions")
            return []
    else:
        quality_flags.append("invalid_syllable_positions")
        return []
    if not isinstance(parsed, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        quality_flags.append("invalid_syllable_positions")
        return []
    return parsed
