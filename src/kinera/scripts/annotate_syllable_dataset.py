"""Anota registros normalizados com alinhamento e features ortográficas."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from kinera.language.datasets.loader import load_jsonl, write_jsonl
from kinera.language.orthography.features import annotate_record


def annotate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_jsonl(input_path)
    if limit is not None:
        records = records[:limit]
    annotated = [annotate_record(record) for record in records]
    write_jsonl(output_path, annotated)

    flags = Counter(
        flag
        for record in annotated
        for flag in record["alignment"]["flags"]
    )
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(annotated),
        "valid_alignments": sum(
            record["alignment"]["valid"] for record in annotated
        ),
        "alignment_flags": dict(flags),
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

    parser = argparse.ArgumentParser(description="Anota dataset com features.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = annotate_jsonl(args.input, args.output, args.report, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
