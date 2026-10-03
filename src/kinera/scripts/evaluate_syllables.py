"""Avalia o segmentador contra um JSONL normalizado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinera.language.datasets.loader import load_jsonl
from kinera.language.syllables.tokenizer import evaluate_records


def evaluate_jsonl(path: str | Path, *, limit: int | None = None) -> dict:
    records = load_jsonl(path)
    if limit is not None:
        records = records[:limit]
    return evaluate_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia segmentação silábica.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_jsonl(args.input, limit=args.limit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
