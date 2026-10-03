"""Compatibility exports for the reorganized language dataset modules."""

from kinera.language.datasets.builder import (
    build_autoregressive_examples,
    group_records,
    split_by_word,
)
from kinera.language.datasets.loader import iter_jsonl, load_jsonl, write_jsonl

__all__ = [
    "build_autoregressive_examples",
    "group_records",
    "iter_jsonl",
    "load_jsonl",
    "split_by_word",
    "write_jsonl",
]
