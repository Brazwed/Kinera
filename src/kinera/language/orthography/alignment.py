"""Alinhamento explícito entre caracteres e sílabas de uma palavra."""

from __future__ import annotations

import unicodedata
from typing import Any


SEPARATOR_CHARS = {"-", "|"}


def align_characters(word: str, syllables: list[str]) -> list[dict[str, Any]]:
    """Mapeia cada caractere para uma sílaba, preservando separadores."""
    normalized_word = unicodedata.normalize("NFC", word)
    if not normalized_word:
        raise ValueError("word não pode ser vazio")
    if not isinstance(syllables, list) or not syllables:
        raise ValueError("syllables não pode ser vazio")
    if any(not isinstance(syllable, str) or not syllable for syllable in syllables):
        raise ValueError("syllables deve conter strings não vazias")

    normalized_syllables = [unicodedata.normalize("NFC", s) for s in syllables]
    flattened = "".join(normalized_syllables).casefold()
    offsets = []
    offset = 0
    for syllable in normalized_syllables:
        offsets.append((offset, offset + len(syllable)))
        offset += len(syllable)

    result: list[dict[str, Any]] = []
    syllable_cursor = 0
    for char_index, char in enumerate(normalized_word):
        is_separator = char in SEPARATOR_CHARS
        if is_separator:
            result.append(
                {
                    "char": char,
                    "char_index": char_index,
                    "syllable_index": None,
                    "position_in_syllable": None,
                    "is_syllable_start": False,
                    "is_syllable_end": False,
                    "is_separator": True,
                    "is_unmatched": False,
                }
            )
            continue

        char_folded = char.casefold()
        matches = (
            syllable_cursor < len(flattened)
            and char_folded == flattened[syllable_cursor]
        )
        if not matches:
            result.append(
                {
                    "char": char,
                    "char_index": char_index,
                    "syllable_index": None,
                    "position_in_syllable": None,
                    "is_syllable_start": False,
                    "is_syllable_end": False,
                    "is_separator": False,
                    "is_unmatched": True,
                }
            )
            continue

        syllable_index = _syllable_index_for_offset(offsets, syllable_cursor)
        start, end = offsets[syllable_index]
        position = syllable_cursor - start
        result.append(
            {
                "char": char,
                "char_index": char_index,
                "syllable_index": syllable_index,
                "position_in_syllable": position,
                "is_syllable_start": syllable_cursor == start,
                "is_syllable_end": syllable_cursor == end - 1,
                "is_separator": False,
                "is_unmatched": False,
            }
        )
        syllable_cursor += 1
    return result


def reconstruct_from_alignment(aligned: list[dict[str, Any]]) -> str:
    """Reconstrói apenas os caracteres associados a sílabas."""
    return "".join(
        item["char"]
        for item in aligned
        if item.get("syllable_index") is not None
    )


def alignment_summary(word: str, syllables: list[str]) -> dict[str, Any]:
    """Resume alinhamento e sinaliza diferenças de reconstrução."""
    aligned = align_characters(word, syllables)
    word_without_separators = "".join(
        item["char"] for item in aligned if not item["is_separator"]
    )
    reconstruction = "".join(syllables)
    flags: list[str] = []
    if word_without_separators.casefold() != reconstruction.casefold():
        flags.append("reconstruction_mismatch")
    if any(item["is_unmatched"] for item in aligned):
        flags.append("unmatched_character")
    return {
        "valid": not flags,
        "word_without_separators": word_without_separators,
        "syllable_reconstruction": reconstruction,
        "flags": flags,
        "aligned": aligned,
    }


def _syllable_index_for_offset(
    offsets: list[tuple[int, int]],
    offset: int,
) -> int:
    for index, (start, end) in enumerate(offsets):
        if start <= offset < end:
            return index
    raise RuntimeError("offset de sílaba fora dos limites")
