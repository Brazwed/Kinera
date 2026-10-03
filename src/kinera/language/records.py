"""Compatibility exports for normalized language records."""

from kinera.language.datasets.records import (
    normalize_silabas_row,
    normalize_word,
    parse_syllable_markup,
)

__all__ = ["normalize_silabas_row", "normalize_word", "parse_syllable_markup"]
