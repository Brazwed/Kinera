"""Contratos comuns para configuração, jobs e agregação de experimentos."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int
    learning_rate: float
    max_examples: int | None
    context_size: int = 8
    top_examples: int | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("phase não pode ser vazia")
        if not self.variants:
            raise ValueError("variants não pode ser vazio")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds deve ser não vazio e sem duplicatas")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds deve conter inteiros")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs deve ser inteiro positivo")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate deve ser numérico")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate deve ser finito e positivo")
        if isinstance(self.max_examples, bool) or (
            self.max_examples is not None
            and (not isinstance(self.max_examples, int) or self.max_examples <= 0)
        ):
            raise ValueError("max_examples deve ser inteiro positivo ou None")
        if isinstance(self.context_size, bool) or not isinstance(self.context_size, int) or self.context_size <= 0:
            raise ValueError("context_size deve ser inteiro positivo")
        if self.top_examples is not None and (
            isinstance(self.top_examples, bool)
            or not isinstance(self.top_examples, int)
            or self.top_examples <= 0
        ):
            raise ValueError("top_examples deve ser inteiro positivo ou None")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "phase": self.phase,
            "variants": list(self.variants),
            "seeds": list(self.seeds),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "max_examples": self.max_examples,
            "context_size": self.context_size,
        }
        if self.top_examples is not None:
            result["top_examples"] = self.top_examples
        return result


@dataclass(frozen=True)
class ExecutionConfig:
    """Controles operacionais que não alteram a matemática do experimento."""

    workers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers <= 0:
            raise ValueError("workers deve ser inteiro positivo")

    def to_dict(self) -> dict[str, int]:
        return {"workers": self.workers}


@dataclass
class RunRecord:
    seed: int
    status: JobStatus
    worker_pid: int | None = None
    wall_time_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"status inválido: {self.status}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed deve ser inteiro")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "worker_pid": self.worker_pid,
            "wall_time_seconds": self.wall_time_seconds,
            "payload": self.payload.copy(),
            "error": self.error,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunRecord":
        if not isinstance(value, dict):
            raise TypeError("run record deve ser um dicionário")
        return cls(
            seed=value["seed"],
            status=value["status"],
            worker_pid=value.get("worker_pid"),
            wall_time_seconds=value.get("wall_time_seconds"),
            payload=dict(value.get("payload", {})),
            error=value.get("error"),
            traceback=value.get("traceback"),
        )


def aggregate_run_status(records: list[RunRecord]) -> dict[str, int | str]:
    total = len(records)
    completed = sum(record.status == "completed" for record in records)
    failed = sum(record.status == "failed" for record in records)
    pending = sum(record.status == "pending" for record in records)
    running = sum(record.status == "running" for record in records)
    if failed:
        status = "failed"
    elif total and completed == total:
        status = "completed"
    else:
        status = "running"
    return {
        "total_runs": total,
        "completed_runs": completed,
        "failed_runs": failed,
        "pending_runs": pending,
        "running_runs": running,
        "status": status,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values não pode ser vazio")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def aggregate_metric_paths(
    records: list[dict[str, Any]],
    paths: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Agrega métricas aninhadas somente de registros concluídos."""
    completed = [record for record in records if record.get("status") == "completed"]
    metrics: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        values: list[float] = []
        for record in completed:
            value: Any = record
            for key in path:
                value = value[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"métrica {name} não é numérica")
            values.append(float(value))
        if values:
            metrics[name] = mean_std(values)
    return {"completed_runs": len(completed), "metrics": metrics}
