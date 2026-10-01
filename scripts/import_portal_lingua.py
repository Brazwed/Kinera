"""Adapter para exportações locais do Portal da Língua Portuguesa."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from language.records import normalize_silabas_row


def import_export(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    *,
    source: str = "portal_lingua",
) -> dict[str, Any]:
    """Normaliza JSONL, JSON ou CSV exportado por uma fonte externa."""
    source_path = Path(input_path)
    rows = list(_read_rows(source_path))
    records = [
        normalize_silabas_row(_adapt_row(row), source=source)
        for row in rows
    ]
    for record, original in zip(records, rows):
        record["source_fields"] = dict(original)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    alignment = Counter(record["alignment_status"] for record in records)
    quality = Counter(
        flag
        for record in records
        for flag in record["quality_flags"]
    )
    report = {
        "source": source,
        "input": str(source_path),
        "output": str(destination),
        "rows": len(rows),
        "records_written": len(records),
        "alignment_status": dict(alignment),
        "quality_flags": dict(quality),
    }
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        report_destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"linha JSONL {line_number} não é objeto")
                yield value
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("records", value.get("results", [value]))
        if not isinstance(value, list):
            raise ValueError("JSON deve conter uma lista de registros")
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("registro JSON não é objeto")
            yield row
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
        return
    raise ValueError("formato de exportação não suportado")


def _adapt_row(row: dict[str, Any]) -> dict[str, Any]:
    word = row.get("word", row.get("lemma", row.get("text", "")))
    raw_syllables = row.get("syllables", row.get("syl", ""))
    stress = row.get(
        "stress",
        row.get("stressed_syllable_index", row.get("syl_strong", "")),
    )
    syl = _syllables_to_markup(raw_syllables, stress)
    return {
        "word": word,
        "category": row.get("category", row.get("pos", "")),
        "syl": syl,
        "syl_positions": row.get("syl_positions", ""),
        "syl_strong": stress,
        "ipa": row.get("ipa", row.get("phonetic", row.get("phonetics", ""))),
        "variant": row.get("variant", row.get("region", "")),
    }


def _syllables_to_markup(value: Any, stress: Any) -> str:
    if isinstance(value, list):
        syllables = [str(item) for item in value]
    elif isinstance(value, str):
        return value
    else:
        return ""
    try:
        stress_index = int(stress)
    except (TypeError, ValueError):
        stress_index = None
    result = []
    for index, syllable in enumerate(syllables):
        result.append(
            f"**{syllable}**" if stress_index == index else syllable
        )
    return "|".join(result)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Importa uma exportação do Portal.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = import_export(args.input, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
