"""Inspeção estrutural do CSV de silabas-pt."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def inspect_csv(path: str | Path, *, sample_limit: int = 5) -> dict[str, Any]:
    """Retorna schema, contagens de linhas e amostras sem transformar dados."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        empty_fields = Counter()
        samples = []
        rows = 0
        separator_counts = Counter()
        for row in reader:
            rows += 1
            if len(samples) < sample_limit:
                samples.append(dict(row))
            for field in fields:
                if not (row.get(field) or "").strip():
                    empty_fields[field] += 1
            syl = row.get("syl") or ""
            separator_counts.update(
                separator for separator in ("-", "|") if separator in syl
            )
    return {
        "path": str(source),
        "fields": fields,
        "rows": rows,
        "empty_fields": {field: empty_fields[field] for field in fields},
        "syllable_separators": dict(separator_counts),
        "samples": samples,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspeciona um CSV lexical.")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_csv(args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
