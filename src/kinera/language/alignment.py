"""Compatibility exports for the reorganized orthography alignment module."""

from kinera.language.orthography.alignment import (
    SEPARATOR_CHARS,
    align_characters,
    alignment_summary,
    reconstruct_from_alignment,
)

__all__ = [
    "SEPARATOR_CHARS",
    "align_characters",
    "alignment_summary",
    "reconstruct_from_alignment",
]
