"""Runner multi-seed genérico com checkpoint, resume e workers."""

from __future__ import annotations

import concurrent.futures
import copy
import traceback
from pathlib import Path
from typing import Any, Callable, cast

from kinera.monitor.telemetry.records import (
    TelemetryEnvelope,
    _merge_run_payloads,
    _prepare_envelope_payload,
)

from .runner_common import (
    aggregate_run_dicts,
    load_seed_checkpoint,
    pending_seeds,
    write_json_checkpoint,
)

Worker = Callable[[Any], dict[str, Any]]
JobFactory = Callable[[int], Any]
Aggregator = Callable[[list[dict[str, Any]]], dict[str, Any]]


def run_seed_jobs(
    *,
    seeds: list[int] | tuple[int, ...],
    experiment_config: dict[str, Any],
    execution_config: dict[str, Any],
    checkpoint_path: str | Path,
    worker: Worker,
    job_factory: JobFactory | None = None,
    aggregate: Aggregator | None = None,
    resume: bool = False,
    telemetry: TelemetryEnvelope | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa jobs por seed e persiste cada transição de estado."""
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds não pode conter duplicatas")
    if not callable(worker):
        raise TypeError("worker deve ser chamável")
    if job_factory is not None and not callable(job_factory):
        raise TypeError("job_factory deve ser chamável ou None")
    if not isinstance(execution_config, dict):
        raise TypeError("execution_config deve ser dicionário")
    backend = execution_config.get("backend", "cpu")
    if not isinstance(backend, str) or backend not in {"cpu", "cuda"}:
        raise ValueError("backend deve ser 'cpu' ou 'cuda'")
    if backend == "cuda":
        raise ValueError(
            "run_seed_jobs não suporta seleção de backend CUDA; use o runner controlado"
        )
    workers = execution_config.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers deve ser inteiro positivo")

    telemetry_section = _serialize_envelope_telemetry(telemetry)
    if resume:
        report = load_seed_checkpoint(checkpoint_path, experiment_config, seeds)
        persisted_telemetry = report.get("telemetry")
        if persisted_telemetry is not None:
            restored_telemetry = _serialize_envelope_telemetry(persisted_telemetry)
            if restored_telemetry is not None:
                telemetry_section = restored_telemetry
        results = report.get("runs", [])
    else:
        results = []
        report = {
            **experiment_config,
            **execution_config,
            "seeds": list(seeds),
            "runs": results,
            "status": "running",
        }
    results = prepare_seed_records(results, seeds)
    write_json_checkpoint(checkpoint_path, {
        **experiment_config,
        **execution_config,
        "seeds": list(seeds),
        "runs": results,
        "status": "running",
        **({"telemetry": copy.deepcopy(telemetry_section)} if telemetry_section is not None else {}),
    })

    pending = pending_seeds(results, seeds)
    _persist(checkpoint_path, experiment_config, execution_config, seeds, results, aggregate, telemetry_section)

    jobs = [(seed, seed if job_factory is None else job_factory(seed)) for seed in pending]
    if workers == 1:
        for seed, job in jobs:
            _mark_running(results, seed)
            result = _execute_worker(worker, seed, job)
            _replace_result(results, seed, result)
            _persist(checkpoint_path, experiment_config, execution_config, seeds, results, aggregate, telemetry_section)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for seed, job in jobs:
                _mark_running(results, seed)
                futures[executor.submit(worker, job)] = seed
            _persist(checkpoint_path, experiment_config, execution_config, seeds, results, aggregate, telemetry_section)
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    if not isinstance(result, dict):
                        raise TypeError("worker deve retornar dicionário")
                except Exception:
                    result = {
                        "seed": seed,
                        "status": "failed",
                        "worker_pid": None,
                        "traceback": traceback.format_exc(),
                    }
                _replace_result(results, seed, result)
                _persist(checkpoint_path, experiment_config, execution_config, seeds, results, aggregate, telemetry_section)

    status = aggregate_run_dicts(results)
    report = {
        **experiment_config,
        **execution_config,
        "seeds": list(seeds),
        "runs": results,
        **status,
        "status": status["status"],
    }
    if aggregate is not None:
        report["aggregates"] = aggregate(results)
    if telemetry_section is not None:
        report["telemetry"] = copy.deepcopy(telemetry_section)
    write_json_checkpoint(checkpoint_path, report)
    return report


def prepare_seed_records(
    results: list[dict[str, Any]],
    seeds: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Normaliza registros para exatamente um registro por seed solicitada."""
    by_seed: dict[int, dict[str, Any]] = {}
    for record in results:
        seed = cast(int, record.get("seed"))
        if seed not in seeds:
            continue
        previous = by_seed.get(seed)
        if previous is None:
            by_seed[seed] = dict(record)
            continue
        selected = (
            dict(record)
            if record.get("status") == "completed" and previous.get("status") != "completed"
            else previous
        )
        telemetry = _merge_record_telemetry(
            previous.get("telemetry"), record.get("telemetry")
        )
        if telemetry is None:
            selected.pop("telemetry", None)
        else:
            selected["telemetry"] = telemetry
        by_seed[seed] = selected

    prepared: list[dict[str, Any]] = []
    for seed in seeds:
        record = by_seed.get(seed)
        if record is not None and record.get("status") == "completed":
            prepared.append(record)
        else:
            pending = {"seed": seed, "status": "pending", "worker_pid": None}
            if record is not None and record.get("telemetry") is not None:
                pending["telemetry"] = copy.deepcopy(record["telemetry"])
            prepared.append(pending)
    return prepared


def _execute_worker(worker: Worker, seed: int, job: Any) -> dict[str, Any]:
    try:
        result = worker(job)
        if not isinstance(result, dict):
            raise TypeError("worker deve retornar dicionário")
        return result
    except Exception:
        return {
            "seed": seed,
            "status": "failed",
            "worker_pid": None,
            "traceback": traceback.format_exc(),
        }


def _mark_running(results: list[dict[str, Any]], seed: int) -> None:
    entry = next(item for item in results if item.get("seed") == seed)
    entry["status"] = "running"


def _replace_result(results: list[dict[str, Any]], seed: int, result: dict[str, Any]) -> None:
    entry = next(item for item in results if item.get("seed") == seed)
    replacement = dict(result)
    telemetry = _merge_record_telemetry(
        entry.get("telemetry"), replacement.get("telemetry")
    )
    if telemetry is None:
        replacement.pop("telemetry", None)
    else:
        replacement["telemetry"] = telemetry
    entry.clear()
    entry.update(replacement)
    entry.setdefault("seed", seed)


def _merge_record_telemetry(previous: Any, incoming: Any) -> dict[str, Any] | None:
    return _merge_run_payloads(
        previous,
        incoming,
        component="experiments.multi_seed",
    )


def _serialize_envelope_telemetry(
    value: TelemetryEnvelope | dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _prepare_envelope_payload(
        value,
        component="experiments.multi_seed.envelope",
    )


def _sanitize_run_telemetry(results: list[dict[str, Any]], *, component: str) -> None:
    for record in results:
        if not isinstance(record, dict) or "telemetry" not in record:
            continue
        telemetry = record.get("telemetry")
        if telemetry is None:
            record.pop("telemetry", None)
            continue
        prepared = _merge_run_payloads(
            None,
            telemetry,
            component=f"{component}.seed={record.get('seed', 'unknown')}",
        )
        if prepared is None:
            record.pop("telemetry", None)
        else:
            record["telemetry"] = prepared


def _persist(
    checkpoint_path: str | Path,
    experiment_config: dict[str, Any],
    execution_config: dict[str, Any],
    seeds: list[int] | tuple[int, ...],
    results: list[dict[str, Any]],
    aggregate: Aggregator | None,
    telemetry: dict[str, Any] | None,
) -> None:
    _sanitize_run_telemetry(results, component="experiments.multi_seed.persist")
    report: dict[str, Any] = {
        **experiment_config,
        **execution_config,
        "seeds": list(seeds),
        "runs": results,
        "status": "running",
        **aggregate_run_dicts(results),
    }
    if aggregate is not None:
        report["aggregates"] = aggregate(results)
    if telemetry is not None:
        report["telemetry"] = copy.deepcopy(telemetry)
    write_json_checkpoint(checkpoint_path, report)
