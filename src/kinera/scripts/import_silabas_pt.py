"""Importador do CSV silabas-pt para JSONL normalizado."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kinera.language.datasets.records import normalize_silabas_row

REQUIRED_FIELDS = {"word", "category", "syl", "syl_positions", "syl_strong"}


def import_csv(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    source: str = "silabas-pt",
) -> dict[str, Any]:
    """Converte CSV lexical em JSONL e retorna o relatório de qualidade."""
    source_path = Path(input_path)
    destination = Path(output_path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"colunas obrigatórias ausentes: {sorted(missing)}")

        records: list[dict[str, Any]] = []
        alignment_counts = Counter()
        quality_counts = Counter()
        word_counts = Counter()
        category_counts = Counter()
        rows = 0
        for row in reader:
            rows += 1
            record = normalize_silabas_row(row, source=source)
            records.append(record)
            alignment_counts[record["alignment_status"]] += 1
            word_counts[record["word_key"]] += 1
            if record["category"] is not None:
                category_counts[record["category"]] += 1
            for flag in record["quality_flags"]:
                quality_counts[flag] += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    report: dict[str, Any] = {
        "source": source,
        "input": str(source_path),
        "output": str(destination),
        "fields": sorted(fields),
        "rows": rows,
        "records_written": len(records),
        "unique_words": len(word_counts),
        "duplicate_word_rows": sum(count - 1 for count in word_counts.values() if count > 1),
        "alignment_status": dict(alignment_counts),
        "quality_flags": dict(quality_counts),
        "categories": dict(category_counts),
    }
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Importa o CSV silabas-pt.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = import_csv(args.input, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
