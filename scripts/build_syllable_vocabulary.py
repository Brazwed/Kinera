"""Constrói um vocabulário silábico sem carregar o JSONL inteiro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from language.vocabulary import SYLLABLE_UNKNOWN_TOKEN, SyllableVocabulary


def build_syllable_vocabulary(
    source_path: str | Path,
    vocabulary_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    syllables: set[str] = set()
    records_read = 0
    invalid_records = 0
    source = Path(source_path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                values = record.get("syllables", [])
                if not isinstance(values, list):
                    raise ValueError("syllables não é lista")
                for syllable in values:
                    if not isinstance(syllable, str) or not syllable:
                        raise ValueError("sílaba inválida")
                    syllables.add(syllable)
                records_read += 1
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                invalid_records += 1

    vocabulary = SyllableVocabulary([SYLLABLE_UNKNOWN_TOKEN, *sorted(syllables)])
    destination = Path(vocabulary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "source": str(source_path),
        "records_read": records_read,
        "invalid_records": invalid_records,
        "syllable_count": len(syllables),
        "vocabulary_size": len(vocabulary),
        "unknown_token": SYLLABLE_UNKNOWN_TOKEN,
        "output": str(vocabulary_path),
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

    parser = argparse.ArgumentParser(description="Constrói o vocabulário silábico.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_syllable_vocabulary(args.source, args.vocabulary, args.report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
