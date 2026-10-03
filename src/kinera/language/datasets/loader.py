"""Streaming JSONL loading and writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Itera objetos JSONL sem manter o arquivo inteiro na memória."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"linha JSONL {line_number} não é objeto")
            yield value


def load_jsonl(
    path: str | Path,
    *,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Carrega JSONL; use ``max_records`` para limitar a leitura."""
    if max_records is not None and (
        isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0
    ):
        raise ValueError("max_records deve ser inteiro positivo ou None")
    records = []
    for value in iter_jsonl(path):
        records.append(value)
        if max_records is not None and len(records) >= max_records:
            break
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return destination
