"""Experimentos A/B/C/D com controles de entrada equivalentes."""

from __future__ import annotations

import copy
import concurrent.futures
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any, cast

from kinera.backends import get_backend
from kinera.experiments.config import ExperimentConfig, ExecutionConfig
from kinera.language.vocabulary import CharacterVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples
from kinera.experiments.variants import (
    BASELINE_VARIANTS,
    CONTROLLED_VARIANTS,
    DEFAULT_VARIANTS,
    OPTIONAL_ABLATION_VARIANTS,
    PRIMARY_VARIANTS,
    SUPPORTED_VARIANTS,
    VARIANTS,
    _validate_variant_selection,
)


def _max_rss_kb() -> int:
    """Retorna o pico de RSS do processo em KiB no Linux/macOS."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(value / 1024)
    return int(value)


def prepare_variant_examples(
    examples: list[dict[str, Any]],
    variant: str,
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int,
) -> list[dict[str, Any]]:
    """Mascara canais de entrada sem alterar targets ou shapes."""
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variante desconhecida: {variant}")
    prepared = []
    for original in examples:
        example = copy.deepcopy(original)
        context_ids = example.get("context_ids")
        if not isinstance(context_ids, list) or not context_ids:
            raise ValueError("example deve conter context_ids não vazio")
        context_size = len(context_ids)
        feature_vectors = example.get("context_feature_vectors")
        if not isinstance(feature_vectors, list) or len(feature_vectors) != context_size:
            feature_vectors = [[0.0] * 10 for _ in range(context_size)]
        if variant in ("A", "C"):
            feature_vectors = [[0.0] * len(vector) for vector in feature_vectors]
        syllable_ids = example.get("context_syllable_ids")
        if not isinstance(syllable_ids, list) or len(syllable_ids) != context_size:
            syllable_ids = [syllable_unknown_id] * context_size
        if variant in ("A", "B"):
            syllable_ids = [syllable_unknown_id] * context_size
        example["context_feature_vectors"] = feature_vectors
        example["context_syllable_ids"] = syllable_ids
        prepared.append(example)
    return prepared


def count_trainable_parameters(model: AutoregressiveModel) -> int:
    """Conta parâmetros pelas dimensões, sem ler ou materializar seus valores."""
    embeddings = (
        model.representation.char_embedding,
        model.representation.syllable_embedding,
    )
    total = sum(
        embedding.vocab_size * embedding.embedding_dim for embedding in embeddings
    )
    for layer in (model.heads.char, model.heads.syllable, model.heads.boundary):
        total += layer.output_size * layer.input_size + layer.output_size
    return total


def _create_execution_backend(execution_config: ExecutionConfig) -> Any | None:
    """Create an explicitly selected backend without importing CUDA on CPU runs."""
    if execution_config.backend == "cpu":
        return None
    return get_backend(execution_config.backend)


def run_controlled_experiments(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_vocab_size: int,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    max_examples: int | None = None,
    syllable_unknown_id: int = 0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    workers: int | None = None,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    execution_config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Executa as variantes selecionadas, com checkpoint por variante/seed."""
    variants = _validate_variant_selection(variants)
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds deve conter inteiros")
    if execution_config is not None and not isinstance(execution_config, ExecutionConfig):
        raise TypeError("execution_config deve ser ExecutionConfig")
    if execution_config is None:
        execution_config = ExecutionConfig(workers=1 if workers is None else workers)
    elif workers is not None and workers != execution_config.workers:
        raise ValueError("workers deve corresponder a execution_config.workers")
    workers = execution_config.workers
    metric_names = (
        "char_loss",
        "syllable_loss",
        "boundary_loss",
        "total_loss",
        "char_accuracy",
        "syllable_accuracy_all",
        "syllable_accuracy_known",
        "boundary_accuracy",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    experiment_config = ExperimentConfig(
        phase="controlled_comparison",
        variants=variants,
        seeds=tuple(seeds),
        epochs=epochs,
        learning_rate=learning_rate,
        max_examples=max_examples,
        context_size=None,
    )
    experiment_configuration = experiment_config.to_dict()
    experiment_configuration.update(
        {
            "syllable_vocab_size": syllable_vocab_size,
            "syllable_unknown_id": syllable_unknown_id,
            "loss_weights": {"char": 1.0, "syllable": 0.7, "boundary": 0.3},
        }
    )
    configuration = {
        **experiment_configuration,
        "experiment_config": experiment_configuration.copy(),
        "execution_config": execution_config.to_dict(),
        "workers": workers,
    }
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("resume exige checkpoint_path existente")
        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        persisted_execution = persisted.get("execution_config")
        if not isinstance(persisted_execution, dict):
            raise ValueError("checkpoint backend ausente ou incompatível")
        try:
            persisted_execution_config = ExecutionConfig(
                workers=cast(Any, persisted_execution.get("workers")),
                backend=cast(Any, persisted_execution.get("backend")),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint execution_config inválida") from exc
        if persisted_execution_config.backend != execution_config.backend:
            raise ValueError("checkpoint backend incompatível com a configuração atual")
        persisted_configuration = {
            key: persisted.get(key)
            for key in configuration
            if key not in {"seeds", "workers", "experiment_config", "execution_config"}
        }
        persisted_seeds = set(persisted.get("seeds", []))
        current_configuration = {
            key: configuration[key]
            for key in configuration
            if key not in {"seeds", "workers", "experiment_config", "execution_config"}
        }
        current_seeds = set(configuration["seeds"])
        if persisted_configuration != current_configuration or not persisted_seeds <= current_seeds:
            raise ValueError("checkpoint incompatível com a configuração atual")
        persisted_runs = persisted["runs"]
        reports = {variant: persisted_runs.get(variant, []) for variant in variants}
        persisted_aggregates = persisted.get("aggregates", {})
        aggregates = {
            variant: persisted_aggregates[variant]
            for variant in variants
            if variant in persisted_aggregates
        }
        persisted_parameter_counts = persisted.get("parameter_counts", {})
        parameter_counts = {
            variant: persisted_parameter_counts[variant]
            for variant in variants
            if variant in persisted_parameter_counts
        }
        persisted_input_usage = persisted.get("input_usage", {})
        input_usage = {
            variant: persisted_input_usage[variant]
            for variant in variants
            if variant in persisted_input_usage
        }
    else:
        reports = {variant: [] for variant in variants}
        aggregates = {}
        parameter_counts = {}
        input_usage = {}
    if checkpoint is not None and not resume:
        _persist_checkpoint(
            checkpoint,
            _checkpoint_report(
                configuration,
                reports,
                aggregates,
                parameter_counts,
                input_usage,
                metric_names,
            ),
        )

    prepared_variants: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for variant in variants:
        prepared_train = prepare_variant_examples(
            train_examples,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        prepared_validation = prepare_variant_examples(
            validation_examples,
            variant,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        if max_examples is not None:
            prepared_train = prepared_train[:max_examples]
            prepared_validation = prepared_validation[:max_examples]
        input_usage[variant] = _input_usage(
            prepared_train,
            prepared_validation,
            syllable_unknown_id,
        )
        prepared_variants[variant] = (prepared_train, prepared_validation)

    if workers > 1:
        return _run_parallel_experiments(
            prepared_variants,
            vocabulary,
            syllable_vocab_size,
            seeds,
            epochs,
            learning_rate,
            syllable_unknown_id,
            checkpoint,
            resume,
            configuration,
            reports,
            aggregates,
            parameter_counts,
            input_usage,
            metric_names,
            workers,
            variants,
        )

    backend = _create_execution_backend(execution_config)
    for variant in variants:
        prepared_train, prepared_validation = prepared_variants[variant]
        reports.setdefault(variant, [])
        for seed in seeds:
            previous = next(
                (run for run in reports[variant] if run.get("seed") == seed),
                None,
            )
            if previous is not None and previous.get("status") == "completed":
                continue
            if previous is not None:
                reports[variant].remove(previous)
            started = time.perf_counter()
            model: AutoregressiveModel | None = None
            try:
                model = AutoregressiveModel(
                    len(vocabulary),
                    syllable_vocab_size,
                    seed=seed,
                    backend=backend,
                )
                parameter_counts[variant] = count_trainable_parameters(model)
                initial = evaluate_multitask_examples(
                    model,
                    prepared_validation,
                    vocabulary,
                    syllable_unknown_id=syllable_unknown_id,
                )
                training = train_multitask_examples(
                    model,
                    prepared_train,
                    vocabulary,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    syllable_unknown_id=syllable_unknown_id,
                )
                final = evaluate_multitask_examples(
                    model,
                    prepared_validation,
                    vocabulary,
                    syllable_unknown_id=syllable_unknown_id,
                )
                model.close()
                model = None
                reports[variant].append({
                    "seed": seed,
                    "status": "completed",
                    "pid": os.getpid(),
                    "max_rss_kb": _max_rss_kb(),
                    "initial_validation": initial,
                    "training": training,
                    "final_validation": final,
                    "parameter_count": parameter_counts[variant],
                    "duration_seconds": time.perf_counter() - started,
                })
            except Exception:
                error_traceback = traceback.format_exc()
                if model is not None:
                    try:
                        model.close()
                    except Exception:
                        error_traceback += "\nCleanup failure:\n" + traceback.format_exc()
                if checkpoint is None:
                    raise
                reports[variant].append({
                    "seed": seed,
                    "status": "failed",
                    "pid": os.getpid(),
                    "max_rss_kb": _max_rss_kb(),
                    "error": "experiment failed",
                    "traceback": error_traceback,
                    "duration_seconds": time.perf_counter() - started,
                })
            aggregates = _build_aggregates(reports, metric_names, variants)
            if checkpoint is not None:
                _persist_checkpoint(
                    checkpoint,
                    _checkpoint_report(
                        configuration,
                        reports,
                        aggregates,
                        parameter_counts,
                        input_usage,
                        metric_names,
                    ),
                )

    aggregates = _build_aggregates(reports, metric_names, variants)
    result = _checkpoint_report(
        configuration,
        reports,
        aggregates,
        parameter_counts,
        input_usage,
        metric_names,
    )
    if checkpoint is not None:
        _persist_checkpoint(checkpoint, result)
    return result


def _execute_single_experiment(
    variant: str,
    seed: int,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    syllable_vocab_size: int,
    epochs: int,
    learning_rate: float,
    syllable_unknown_id: int,
) -> tuple[str, int, dict[str, Any]]:
    started = time.perf_counter()
    model: AutoregressiveModel | None = None
    try:
        model = AutoregressiveModel(
            len(vocabulary),
            syllable_vocab_size,
            seed=seed,
        )
        parameter_count = count_trainable_parameters(model)
        initial = evaluate_multitask_examples(
            model,
            validation_examples,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        training = train_multitask_examples(
            model,
            train_examples,
            vocabulary,
            epochs=epochs,
            learning_rate=learning_rate,
            syllable_unknown_id=syllable_unknown_id,
        )
        final = evaluate_multitask_examples(
            model,
            validation_examples,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        model.close()
        model = None
        record = {
            "seed": seed,
            "status": "completed",
            "pid": os.getpid(),
            "max_rss_kb": _max_rss_kb(),
            "initial_validation": initial,
            "training": training,
            "final_validation": final,
            "parameter_count": parameter_count,
            "duration_seconds": time.perf_counter() - started,
        }
    except Exception:
        error_traceback = traceback.format_exc()
        if model is not None:
            try:
                model.close()
            except Exception:
                error_traceback += "\nCleanup failure:\n" + traceback.format_exc()
        record = {
            "seed": seed,
            "status": "failed",
            "pid": os.getpid(),
            "max_rss_kb": _max_rss_kb(),
            "error": "experiment failed",
            "traceback": error_traceback,
            "duration_seconds": time.perf_counter() - started,
        }
    return variant, seed, record


def _run_parallel_experiments(
    prepared_variants: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    vocabulary: CharacterVocabulary,
    syllable_vocab_size: int,
    seeds: list[int],
    epochs: int,
    learning_rate: float,
    syllable_unknown_id: int,
    checkpoint: Path | None,
    resume: bool,
    configuration: dict[str, Any],
    reports: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, dict[str, float]]],
    parameter_counts: dict[str, int],
    input_usage: dict[str, dict[str, int]],
    metric_names: tuple[str, ...],
    workers: int,
    variants: tuple[str, ...],
) -> dict[str, Any]:
    del resume
    for variant in variants:
        reports.setdefault(variant, [])

    futures: dict[concurrent.futures.Future[tuple[str, int, dict[str, Any]]], tuple[str, int]] = {}
    pending_jobs: list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for variant in variants:
            train_examples, validation_examples = prepared_variants[variant]
            for seed in seeds:
                previous = next(
                    (run for run in reports[variant] if run.get("seed") == seed),
                    None,
                )
                if previous is not None and previous.get("status") == "completed":
                    continue
                if previous is not None:
                    reports[variant].remove(previous)
                reports[variant].append({
                    "seed": seed,
                    "status": "pending",
                    "pid": None,
                })
                pending_jobs.append((variant, seed, train_examples, validation_examples))
        if checkpoint is not None:
            _persist_checkpoint(
                checkpoint,
                _checkpoint_report(
                    configuration,
                    reports,
                    aggregates,
                    parameter_counts,
                    input_usage,
                    metric_names,
                ),
            )
        for variant, seed, train_examples, validation_examples in pending_jobs:
            pending = next(run for run in reports[variant] if run.get("seed") == seed)
            pending["status"] = "running"
            future = executor.submit(
                _execute_single_experiment,
                variant,
                seed,
                train_examples,
                validation_examples,
                vocabulary,
                syllable_vocab_size,
                epochs,
                learning_rate,
                syllable_unknown_id,
            )
            futures[future] = (variant, seed)
        if checkpoint is not None:
            _persist_checkpoint(
                checkpoint,
                _checkpoint_report(
                    configuration,
                    reports,
                    aggregates,
                    parameter_counts,
                    input_usage,
                    metric_names,
                ),
            )
        for future in concurrent.futures.as_completed(futures):
            variant, seed = futures[future]
            try:
                _, _, record = future.result()
            except Exception:
                record = {
                    "seed": seed,
                    "status": "failed",
                    "pid": None,
                    "max_rss_kb": None,
                    "error": "worker process failed",
                    "traceback": traceback.format_exc(),
                }
            existing = next(run for run in reports[variant] if run.get("seed") == seed)
            reports[variant][reports[variant].index(existing)] = record
            if record.get("status") == "completed":
                parameter_counts[variant] = record["parameter_count"]
            aggregates = _build_aggregates(reports, metric_names, variants)
            if checkpoint is not None:
                _persist_checkpoint(
                    checkpoint,
                    _checkpoint_report(
                        configuration,
                        reports,
                        aggregates,
                        parameter_counts,
                        input_usage,
                        metric_names,
                    ),
                )

    aggregates = _build_aggregates(reports, metric_names, variants)
    result = _checkpoint_report(
        configuration,
        reports,
        aggregates,
        parameter_counts,
        input_usage,
        metric_names,
    )
    if checkpoint is not None:
        _persist_checkpoint(checkpoint, result)
    return result


def _build_aggregates(
    reports: dict[str, list[dict[str, Any]]],
    metric_names: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variants:
        completed = [run for run in reports.get(variant, []) if run.get("status") == "completed"]
        if not completed:
            continue
        aggregates[variant] = {"mean": {}, "std": {}}
        for name in metric_names:
            values = [float(run["final_validation"][name]) for run in completed]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            aggregates[variant]["mean"][name] = mean
            aggregates[variant]["std"][name] = math.sqrt(variance)
    return aggregates


def _checkpoint_report(
    configuration: dict[str, Any],
    reports: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, dict[str, float]]],
    parameter_counts: dict[str, int],
    input_usage: dict[str, dict[str, int]],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    del metric_names
    completed_runs = sum(
        run.get("status") == "completed"
        for variant in configuration["variants"]
        for run in reports.get(variant, [])
    )
    failed_runs = sum(
        run.get("status") == "failed"
        for variant in configuration["variants"]
        for run in reports.get(variant, [])
    )
    total_runs = len(configuration["variants"]) * len(configuration["seeds"])
    return {
        **configuration,
        "status": "failed" if failed_runs else ("completed" if completed_runs == total_runs else "running"),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "runs": reports,
        "aggregates": aggregates,
        "parameter_counts": parameter_counts,
        "input_usage": input_usage,
    }


def _persist_checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _input_usage(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    syllable_unknown_id: int,
) -> dict[str, int]:
    def measure(examples: list[dict[str, Any]]) -> tuple[int, int, int]:
        positions = 0
        non_unknown = 0
        unique_ids: set[int] = set()
        for example in examples:
            ids = example.get("context_syllable_ids", [])
            positions += len(ids)
            for value in ids:
                if value != syllable_unknown_id:
                    non_unknown += 1
                    unique_ids.add(value)
        return positions, non_unknown, len(unique_ids)

    train_positions, train_non_unknown, train_unique = measure(train_examples)
    validation_positions, validation_non_unknown, validation_unique = measure(
        validation_examples
    )
    return {
        "train_context_positions": train_positions,
        "train_non_unk_context_positions": train_non_unknown,
        "train_unique_syllable_ids": train_unique,
        "validation_context_positions": validation_positions,
        "validation_non_unk_context_positions": validation_non_unknown,
        "validation_unique_syllable_ids": validation_unique,
    }
