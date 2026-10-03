"""Compatibility exports for reorganized orthographic feature helpers."""

from kinera.language.orthography.features import (
    ORTHOGRAPHIC_FEATURE_NAMES,
    annotate_record,
    annotate_word,
    character_features,
    features_to_vector,
)

__all__ = [
    "ORTHOGRAPHIC_FEATURE_NAMES",
    "annotate_record",
    "annotate_word",
    "character_features",
    "features_to_vector",
]
