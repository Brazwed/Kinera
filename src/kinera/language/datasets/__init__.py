"""Language dataset records, loaders, builders, and input adapters."""

from .builder import build_autoregressive_examples, group_records, split_by_word
from .inputs import record_to_hybrid_inputs
from .loader import iter_jsonl, load_jsonl, write_jsonl
from .records import normalize_silabas_row, normalize_word, parse_syllable_markup

__all__ = [
    "build_autoregressive_examples",
    "group_records",
    "iter_jsonl",
    "load_jsonl",
    "normalize_silabas_row",
    "normalize_word",
    "parse_syllable_markup",
    "record_to_hybrid_inputs",
    "split_by_word",
    "write_jsonl",
]
