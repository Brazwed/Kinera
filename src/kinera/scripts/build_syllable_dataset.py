"""Constrói dataset agrupado e splits lexicais por palavra."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from kinera.language.datasets.builder import group_records, split_by_word
from kinera.language.datasets.loader import load_jsonl, write_jsonl


def build_dataset(
    input_paths: list[str | Path],
    grouped_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    report_path: str | Path | None = None,
    *,
    validation_ratio: float = 0.2,
    seed: int | None = 42,
) -> dict[str, Any]:
    records = []
    for path in input_paths:
        records.extend(load_jsonl(path))
    groups = group_records(records)
    train, validation = split_by_word(groups, validation_ratio, seed)
    write_jsonl(grouped_path, groups)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    source_counts = Counter(
        analysis.get("source")
        for group in groups
        for analysis in group["analyses"]
    )
    report = {
        "inputs": [str(path) for path in input_paths],
        "raw_records": len(records),
        "groups": len(groups),
        "train_groups": len(train),
        "validation_groups": len(validation),
        "conflicting_groups": sum(group["conflict"] for group in groups),
        "sources": dict(source_counts),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "outputs": {
            "grouped": str(grouped_path),
            "train": str(train_path),
            "validation": str(validation_path),
        },
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Constrói dataset silábico.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--grouped", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_dataset(
        args.inputs,
        args.grouped,
        args.train,
        args.validation,
        args.report,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
