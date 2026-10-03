"""Utilitários compartilhados por runners experimentais."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import RunRecord, aggregate_run_status


def write_json_checkpoint(path: str | Path, report: dict[str, Any]) -> Path:
    """Escreve JSON de checkpoint com substituição atômica."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_seed_checkpoint(
    path: str | Path,
    configuration: dict[str, Any],
    seeds: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Carrega checkpoint e permite somente ampliar o conjunto de seeds."""
    source = Path(path)
    if not source.exists():
        raise ValueError("checkpoint inexistente")
    report = json.loads(source.read_text(encoding="utf-8"))
    for key, value in configuration.items():
        if key == "seeds":
            continue
        if report.get(key) != value:
            raise ValueError("checkpoint incompatível com a configuração")
    if not set(report.get("seeds", [])) <= set(seeds):
        raise ValueError("checkpoint tem seeds incompatíveis")
    if not isinstance(report.get("runs", []), list):
        raise ValueError("checkpoint contém runs inválidos")
    return report


def pending_seeds(results: list[dict[str, Any]], seeds: list[int] | tuple[int, ...]) -> list[int]:
    """Retorna seeds ainda não concluídas, preservando a ordem solicitada."""
    completed = {item.get("seed") for item in results if item.get("status") == "completed"}
    return [seed for seed in seeds if seed not in completed]


def aggregate_run_dicts(records: list[dict[str, Any]]) -> dict[str, int | str]:
    """Agrega status de registros JSON sem duplicar lógica nos runners."""
    normalized = []
    for record in records:
        normalized.append({**record, "payload": record.get("payload", record.get("run", {}))})
    return aggregate_run_status([RunRecord.from_dict(record) for record in normalized])
