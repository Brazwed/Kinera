"""Segmentador silábico inicial para português e métricas contra dados gold."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any

from kinera.monitor.metrics.core import safe_ratio

VOWELS = set("aeiouáéíóúâêôãõàü")
DIPHTHONGS = {
    "ai",
    "au",
    "ei",
    "eu",
    "iu",
    "oi",
    "ou",
    "ui",
    "ia",
    "ie",
    "io",
    "ua",
    "ue",
    "uo",
    "ãe",
    "ãi",
    "ão",
    "õe",
}
TRIPHTHONGS = {"uai", "uei", "uão"}
LEGAL_ONSETS = {
    "bl",
    "br",
    "ch",
    "cl",
    "cr",
    "dr",
    "fl",
    "fr",
    "gl",
    "gr",
    "lh",
    "nh",
    "pl",
    "pr",
    "qu",
    "tr",
    "vr",
}


def segment_word(word: str) -> list[str]:
    """Segmenta uma palavra usando regras ortográficas simplificadas."""
    if not isinstance(word, str):
        raise TypeError("word deve ser uma string")
    normalized = unicodedata.normalize("NFC", word.strip().casefold())
    normalized = normalized.replace("-", "")
    if not normalized:
        return []

    nuclei = _find_nuclei(normalized)
    if not nuclei:
        return [normalized]
    if len(nuclei) == 1:
        return [normalized]

    boundaries = []
    for current, following in zip(nuclei, nuclei[1:]):
        cluster = normalized[current[1] : following[0]]
        onset_length = _onset_length(cluster)
        boundaries.append(following[0] - onset_length)

    starts = [0] + boundaries
    ends = boundaries + [len(normalized)]
    return [normalized[start:end] for start, end in zip(starts, ends)]


def syllable_boundaries(word: str, syllables: list[str]) -> list[int]:
    """Converte sílabas em offsets cumulativos sem o último limite."""
    normalized = unicodedata.normalize("NFC", word.strip().casefold()).replace("-", "")
    if "" in syllables:
        raise ValueError("syllables não pode conter strings vazias")
    if "".join(syllables) != normalized:
        raise ValueError("sílabas não reconstroem a palavra")
    offsets = []
    offset = 0
    for syllable in syllables[:-1]:
        offset += len(syllable)
        offsets.append(offset)
    return offsets


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compara o segmentador com registros normalizados de referência."""
    exact_matches = 0
    boundary_correct = 0
    boundary_total = 0
    predicted_boundary_total = 0
    scored = 0
    errors = Counter()

    for record in records:
        word = record.get("word_normalized", record.get("word_raw", ""))
        gold = record.get("syllables", [])
        try:
            predicted = segment_word(word)
            gold_boundaries = set(syllable_boundaries(word, gold))
            predicted_boundaries = set(syllable_boundaries(word, predicted))
        except (TypeError, ValueError) as error:
            errors[type(error).__name__] += 1
            continue
        scored += 1
        exact_matches += predicted == gold
        boundary_correct += len(gold_boundaries & predicted_boundaries)
        boundary_total += len(gold_boundaries)
        predicted_boundary_total += len(predicted_boundaries)

    offset_prec = (
        boundary_correct / predicted_boundary_total
        if predicted_boundary_total > 0
        else None
    )
    offset_rec = (
        boundary_correct / boundary_total
        if boundary_total > 0
        else None
    )
    if offset_prec is not None and offset_rec is not None and (offset_prec + offset_rec) > 0:
        offset_f1 = (2.0 * offset_prec * offset_rec) / (offset_prec + offset_rec)
    else:
        offset_f1 = None

    return {
        "total": len(records),
        "scored": scored,
        "exact_matches": exact_matches,
        "exact_accuracy": safe_ratio(exact_matches, scored),
        "boundary_correct": boundary_correct,
        "boundary_total": boundary_total,
        "boundary_accuracy": safe_ratio(
            boundary_correct, boundary_total, zero_denominator=1.0
        ),
        "boundary_offset_precision": offset_prec,
        "boundary_offset_recall": offset_rec,
        "boundary_offset_f1": offset_f1,
        "errors": dict(errors),
    }


def _find_nuclei(word: str) -> list[tuple[int, int]]:
    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(word):
        if not _is_pronounced_vowel(word, index):
            index += 1
            continue
        start = index
        remaining = word[index : index + 3]
        if len(remaining) == 3 and remaining in TRIPHTHONGS:
            index += 3
        elif len(remaining) >= 2 and remaining[:2] in DIPHTHONGS:
            index += 2
        else:
            index += 1
        nuclei.append((start, index))
    return nuclei


def _is_pronounced_vowel(word: str, index: int) -> bool:
    char = word[index]
    if char not in VOWELS:
        return False
    if char == "u" and index > 0 and word[index - 1] in {"q", "g"}:
        next_char = word[index + 1] if index + 1 < len(word) else ""
        if next_char in {"e", "i"}:
            return False
    return True


def _onset_length(cluster: str) -> int:
    if not cluster:
        return 0
    for length in (3, 2):
        if len(cluster) >= length and cluster[-length:] in LEGAL_ONSETS:
            return length
    return 1
