"""Experimentos A/B/C/D com controles de entrada equivalentes."""

from __future__ import annotations

import copy
import concurrent.futures
import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any, cast

from kinera.backends import get_backend
from kinera.experiments.config import ExperimentConfig, ExecutionConfig
from kinera.monitor.metrics.aggregation import (
    aggregate_metric_values,
    mean_std,
    summarize_paired_deltas,
)
from kinera.monitor.telemetry.records import (
    TELEMETRY_SCHEMA_VERSION,
    TelemetryAttemptRecord,
    TelemetryEnvelope,
    TelemetryRunRecord,
    TelemetrySegmentRecord,
    _merge_run_payloads,
    _prepare_envelope_payload,
    _report_telemetry_diagnostic,
)
from kinera.monitor.telemetry.environment import collect_environment_snapshot
from kinera.monitor.telemetry.resources import collect_resource_snapshot
from kinera.monitor.telemetry.timing import TimingSession, _seconds_from_ns, utc_now
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


def _safe_monotonic_delta_seconds(start_ns: int | None, finish_ns: int | None) -> float | None:
    if start_ns is None or finish_ns is None:
        return None
    delta = finish_ns - start_ns
    if delta < 0:
        return None
    try:
        return _seconds_from_ns(delta)
    except Exception:
        return None


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
    *,
    backend: Any | None = None,
    collect_telemetry: bool = False,
    prior_telemetry: dict[str, Any] | None = None,
    process_id: str | None = None,
    submitted_at: str | None = None,
    submitted_monotonic_ns: int | None = None,
    reraise: bool = False,
) -> tuple[str, int, dict[str, Any]]:
    started = time.perf_counter()
    model: AutoregressiveModel | None = None
    cleanup_stage: Any | None = None
    cleanup_error: str | None = None
    timing: TimingSession | None = None
    attempt_id: str | None = None
    run_id: str | None = None
    segment_id: str | None = None
    attempt_started_civil: str | None = None
    attempt_started_mono: int | None = None
    worker_started_civil: str | None = None
    queue_duration_seconds: float | None = None

    if collect_telemetry:
        try:
            attempt_started_civil = utc_now().isoformat()
            attempt_started_mono = time.monotonic_ns()
            timing = TimingSession()
            if (
                prior_telemetry is not None
                and isinstance(prior_telemetry, dict)
                and "run_id" in prior_telemetry
            ):
                run_id = prior_telemetry["run_id"]
            else:
                run_id = f"{variant}-seed-{seed}"

            existing_attempts = (
                prior_telemetry.get("attempts", [])
                if isinstance(prior_telemetry, dict)
                else []
            )
            existing_ids = {
                a.get("attempt_id")
                for a in existing_attempts
                if isinstance(a, dict)
            }
            attempt_num = len(existing_attempts) + 1
            while f"attempt-{attempt_num}" in existing_ids:
                attempt_num += 1
            attempt_id = f"attempt-{attempt_num}"

            if process_id is None:
                process_id = f"controlled-{variant}-seed-{seed}-{attempt_id}"
            elif not process_id.endswith(f"-{attempt_id}"):
                process_id = f"{process_id}-{attempt_id}"
            segment_id = f"segment-{attempt_id}"

            if submitted_at is not None:
                worker_started_civil = attempt_started_civil
            if submitted_monotonic_ns is not None:
                queue_duration_seconds = _safe_monotonic_delta_seconds(
                    submitted_monotonic_ns, attempt_started_mono
                )
        except Exception as exc:
            _report_telemetry_diagnostic("controlled_comparison.init_telemetry", exc)
            collect_telemetry = False
            timing = None

    try:
        if collect_telemetry and timing is not None:
            with timing.stage(stage_id="setup", name="setup"):
                model = AutoregressiveModel(
                    len(vocabulary),
                    syllable_vocab_size,
                    seed=seed,
                    backend=backend,
                )
                parameter_count = count_trainable_parameters(model)
        else:
            model = AutoregressiveModel(
                len(vocabulary),
                syllable_vocab_size,
                seed=seed,
                backend=backend,
            )
            parameter_count = count_trainable_parameters(model)

        train_boundary_dist: dict[str, int] = {"0": 0, "1": 0}
        for ex in train_examples:
            b_val = ex.get("target_is_syllable_end")
            if b_val is not None:
                train_boundary_dist[str(int(b_val >= 0.5))] += 1

        if collect_telemetry and timing is not None:
            with timing.stage(stage_id="initial_validation", name="initial_validation"):
                initial = evaluate_multitask_examples(
                    model,
                    validation_examples,
                    vocabulary,
                    syllable_unknown_id=syllable_unknown_id,
                    train_boundary_distribution=train_boundary_dist,
                )
        else:
            initial = evaluate_multitask_examples(
                model,
                validation_examples,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
                train_boundary_distribution=train_boundary_dist,
            )

        if collect_telemetry and timing is not None:
            with timing.stage(stage_id="training", name="training"):
                training = train_multitask_examples(
                    model,
                    train_examples,
                    vocabulary,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    syllable_unknown_id=syllable_unknown_id,
                    timing_session=timing,
                    stage_id_prefix="training",
                )
        else:
            training = train_multitask_examples(
                model,
                train_examples,
                vocabulary,
                epochs=epochs,
                learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id,
            )

        if collect_telemetry and timing is not None:
            with timing.stage(stage_id="final_validation", name="final_validation"):
                final = evaluate_multitask_examples(
                    model,
                    validation_examples,
                    vocabulary,
                    syllable_unknown_id=syllable_unknown_id,
                    train_boundary_distribution=train_boundary_dist,
                )
        else:
            final = evaluate_multitask_examples(
                model,
                validation_examples,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
                train_boundary_distribution=train_boundary_dist,
            )

        if model is not None:
            if collect_telemetry and timing is not None:
                with timing.stage(stage_id="cleanup", name="cleanup"):
                    model.close()
                    model = None
                cleanup_stage = next(
                    (s for s in timing.stages if s.stage_id == "cleanup"), None
                )
            else:
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
        if (
            collect_telemetry
            and timing is not None
            and attempt_id
            and run_id
            and segment_id
        ):
            try:
                attempt_finished_civil = utc_now().isoformat()
                attempt_finished_mono = time.monotonic_ns()
                attempt_duration = _safe_monotonic_delta_seconds(
                    attempt_started_mono, attempt_finished_mono
                )
                segment = TelemetrySegmentRecord(
                    segment_id=segment_id,
                    status="completed",
                    started_at=attempt_started_civil,
                    finished_at=attempt_finished_civil,
                    duration_seconds=attempt_duration,
                    stages=list(timing.stages),
                )
                attempt_rec = TelemetryAttemptRecord(
                    attempt_id=attempt_id,
                    process_id=cast(str, process_id),
                    pid=os.getpid(),
                    status="completed",
                    submitted_at=submitted_at,
                    worker_started_at=worker_started_civil,
                    queue_duration_seconds=queue_duration_seconds,
                    started_at=attempt_started_civil,
                    finished_at=attempt_finished_civil,
                    duration_seconds=attempt_duration,
                    cleanup_status="completed" if cleanup_stage else "not_run",
                    segments=[segment],
                )
                if cleanup_stage is not None:
                    attempt_rec.record_cleanup_stage(cleanup_stage)
                run_rec = TelemetryRunRecord(
                    telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
                    run_id=run_id,
                    attempts=[attempt_rec],
                )
                if prior_telemetry is not None:
                    merged_rec = _merge_run_payloads(
                        prior_telemetry,
                        run_rec,
                        component="controlled_comparison.record_telemetry",
                    )
                    record["telemetry"] = merged_rec if merged_rec is not None else run_rec.to_dict()
                else:
                    record["telemetry"] = run_rec.to_dict()
            except Exception as t_exc:
                _report_telemetry_diagnostic(
                    "controlled_comparison.record_telemetry", t_exc
                )
    except Exception as exc:
        error_traceback = traceback.format_exc()
        if model is not None:
            try:
                if collect_telemetry and timing is not None:
                    cleanup_stage_id = (
                        "cleanup_error"
                        if "cleanup" in timing._stage_ids
                        else "cleanup"
                    )
                    with timing.stage(stage_id=cleanup_stage_id, name="cleanup"):
                        model.close()
                        model = None
                    cleanup_stage = next(
                        (s for s in timing.stages if s.stage_id == cleanup_stage_id), None
                    )
                else:
                    model.close()
                    model = None
            except Exception as c_exc:
                cleanup_error = str(c_exc)
                error_traceback += "\nCleanup failure:\n" + traceback.format_exc()

        if reraise:
            raise

        record = {
            "seed": seed,
            "status": "failed",
            "pid": os.getpid(),
            "max_rss_kb": _max_rss_kb(),
            "error": "experiment failed",
            "traceback": error_traceback,
            "duration_seconds": time.perf_counter() - started,
        }
        if (
            collect_telemetry
            and timing is not None
            and attempt_id
            and run_id
            and segment_id
        ):
            try:
                attempt_finished_civil = utc_now().isoformat()
                attempt_finished_mono = time.monotonic_ns()
                attempt_duration = _safe_monotonic_delta_seconds(
                    attempt_started_mono, attempt_finished_mono
                )
                segment = TelemetrySegmentRecord(
                    segment_id=segment_id,
                    status="failed",
                    started_at=attempt_started_civil,
                    finished_at=attempt_finished_civil,
                    duration_seconds=attempt_duration,
                    error=str(exc),
                    stages=list(timing.stages),
                )
                attempt_rec = TelemetryAttemptRecord(
                    attempt_id=attempt_id,
                    process_id=cast(str, process_id),
                    pid=os.getpid(),
                    status="failed",
                    submitted_at=submitted_at,
                    worker_started_at=worker_started_civil,
                    queue_duration_seconds=queue_duration_seconds,
                    started_at=attempt_started_civil,
                    finished_at=attempt_finished_civil,
                    duration_seconds=attempt_duration,
                    error=str(exc),
                    cleanup_status=(
                        "failed"
                        if cleanup_error
                        else ("completed" if cleanup_stage else "not_run")
                    ),
                    cleanup_error=cleanup_error,
                    segments=[segment],
                )
                if cleanup_stage is not None and cleanup_error is None:
                    attempt_rec.record_cleanup_stage(cleanup_stage)
                run_rec = TelemetryRunRecord(
                    telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
                    run_id=run_id,
                    attempts=[attempt_rec],
                )
                if prior_telemetry is not None:
                    merged_rec = _merge_run_payloads(
                        prior_telemetry,
                        run_rec,
                        component="controlled_comparison.record_telemetry",
                    )
                    record["telemetry"] = merged_rec if merged_rec is not None else run_rec.to_dict()
                else:
                    record["telemetry"] = run_rec.to_dict()
            except Exception as t_exc:
                _report_telemetry_diagnostic(
                    "controlled_comparison.failed_telemetry", t_exc
                )

    return variant, seed, record


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
    telemetry: TelemetryEnvelope | dict[str, Any] | None = None,
    collect_telemetry: bool | None = None,
) -> dict[str, Any]:
    """Executa as variantes selecionadas, com checkpoint por variante/seed."""
    variants = _validate_variant_selection(variants)
    if not seeds:
        raise ValueError("seeds não pode ser vazio")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds deve conter inteiros")
    if execution_config is not None and not isinstance(execution_config, ExecutionConfig):
        raise TypeError("execution_config deve ser ExecutionConfig")
    if collect_telemetry is not None and not isinstance(collect_telemetry, bool):
        raise TypeError("collect_telemetry deve ser booleano ou None")
    if execution_config is None:
        execution_config = ExecutionConfig(
            workers=1 if workers is None else workers,
            collect_telemetry=False if collect_telemetry is None else collect_telemetry,
        )
    elif workers is not None and workers != execution_config.workers:
        raise ValueError("workers deve corresponder a execution_config.workers")

    if collect_telemetry is None:
        collect_telemetry = execution_config.collect_telemetry
    elif collect_telemetry != execution_config.collect_telemetry:
        execution_config = ExecutionConfig(
            workers=execution_config.workers,
            backend=execution_config.backend,
            collect_telemetry=collect_telemetry,
        )
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
        "boundary_true_negative",
        "boundary_specificity",
        "boundary_balanced_accuracy",
        "boundary_baseline_constant_0",
        "boundary_baseline_constant_1",
        "boundary_baseline_majority",
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    telemetry_section = _serialize_telemetry_envelope(telemetry)
    if collect_telemetry and telemetry_section is None:
        now_dt = utc_now()
        session_id = f"controlled-comparison-{now_dt.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{time.monotonic_ns()}"
        env_snapshot = None
        res_snapshot = None
        try:
            env_snapshot = collect_environment_snapshot()
        except Exception as exc:
            _report_telemetry_diagnostic("controlled_comparison.environment", exc)
        try:
            res_snapshot = collect_resource_snapshot()
        except Exception as exc:
            _report_telemetry_diagnostic("controlled_comparison.resources", exc)
        envelope = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id=session_id,
            started_at=utc_now().isoformat(),
            environment=env_snapshot,
            resources=res_snapshot,
        )
        telemetry_section = envelope.to_dict()
    elif collect_telemetry and telemetry_section is not None:
        if "environment" not in telemetry_section:
            try:
                env = collect_environment_snapshot()
                if env is not None:
                    telemetry_section["environment"] = env.to_dict()
            except Exception as exc:
                _report_telemetry_diagnostic("controlled_comparison.environment", exc)
        if "resources" not in telemetry_section:
            try:
                res = collect_resource_snapshot()
                if res is not None:
                    telemetry_section["resources"] = res.to_dict()
            except Exception as exc:
                _report_telemetry_diagnostic("controlled_comparison.resources", exc)

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
        persisted_telemetry = persisted.get("telemetry")
        if persisted_telemetry is not None:
            restored_telemetry = _serialize_telemetry_envelope(persisted_telemetry)
            if restored_telemetry is not None:
                telemetry_section = restored_telemetry
        persisted_execution = persisted.get("execution_config")
        if not isinstance(persisted_execution, dict):
            raise ValueError("checkpoint backend ausente ou incompatível")
        try:
            persisted_execution_config = ExecutionConfig(
                workers=cast(Any, persisted_execution.get("workers")),
                backend=cast(Any, persisted_execution.get("backend")),
                collect_telemetry=bool(persisted_execution.get("collect_telemetry", False)),
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
                telemetry=telemetry_section,
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
            telemetry_section,
            collect_telemetry=collect_telemetry,
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
            _, _, record = _execute_single_experiment(
                variant,
                seed,
                prepared_train,
                prepared_validation,
                vocabulary,
                syllable_vocab_size,
                epochs,
                learning_rate,
                syllable_unknown_id,
                backend=backend,
                collect_telemetry=collect_telemetry,
                prior_telemetry=previous.get("telemetry") if previous is not None else None,
                process_id=None,
                reraise=(checkpoint is None),
            )
            if record.get("status") == "completed":
                parameter_counts[variant] = record["parameter_count"]
            reports[variant].append(
                _preserve_run_telemetry(previous, record)
            )
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
                        telemetry=telemetry_section,
                    ),
                )

    if collect_telemetry and telemetry_section is not None:
        try:
            telemetry_section["finished_at"] = utc_now().isoformat()
            latest_resources = collect_resource_snapshot()
            if latest_resources is not None:
                telemetry_section["resources"] = latest_resources.to_dict()
        except Exception as exc:
            _report_telemetry_diagnostic("controlled_comparison.final_telemetry", exc)

    aggregates = _build_aggregates(reports, metric_names, variants)
    result = _checkpoint_report(
        configuration,
        reports,
        aggregates,
        parameter_counts,
        input_usage,
        metric_names,
        telemetry=telemetry_section,
    )
    if checkpoint is not None:
        _persist_checkpoint(checkpoint, result)
    return result


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
    telemetry_section: dict[str, Any] | None,
    collect_telemetry: bool = False,
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
                pending_record = {"seed": seed, "status": "pending", "pid": None}
                if previous is not None and previous.get("telemetry") is not None:
                    pending_record["telemetry"] = copy.deepcopy(previous["telemetry"])
                reports[variant].append(pending_record)
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
                    telemetry=telemetry_section,
                ),
            )
        for variant, seed, train_examples, validation_examples in pending_jobs:
            pending = next(run for run in reports[variant] if run.get("seed") == seed)
            pending["status"] = "running"
            if collect_telemetry:
                prior_telemetry = pending.get("telemetry")
                sub_at = utc_now().isoformat()
                sub_mono = time.monotonic_ns()
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
                    backend=None,
                    collect_telemetry=True,
                    prior_telemetry=prior_telemetry,
                    process_id=f"worker-{variant}-seed-{seed}",
                    submitted_at=sub_at,
                    submitted_monotonic_ns=sub_mono,
                    reraise=False,
                )
            else:
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
                    telemetry=telemetry_section,
                ),
            )
        for future in concurrent.futures.as_completed(futures):
            variant, seed = futures[future]
            existing = next(run for run in reports[variant] if run.get("seed") == seed)
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
                if collect_telemetry:
                    prior = existing.get("telemetry")
                    run_id = (
                        prior.get("run_id")
                        if isinstance(prior, dict) and "run_id" in prior
                        else f"{variant}-seed-{seed}"
                    )
                    existing_attempts = (
                        prior.get("attempts", [])
                        if isinstance(prior, dict)
                        else []
                    )
                    existing_ids = {
                        a.get("attempt_id")
                        for a in existing_attempts
                        if isinstance(a, dict)
                    }
                    attempt_num = len(existing_attempts) + 1
                    while f"attempt-{attempt_num}" in existing_ids:
                        attempt_num += 1
                    attempt_id = f"attempt-{attempt_num}"
                    attempt_rec = TelemetryAttemptRecord(
                        attempt_id=attempt_id,
                        process_id=f"worker-{variant}-seed-{seed}-{attempt_id}",
                        pid=None,
                        status="failed",
                        error="worker process failed",
                        cleanup_status="not_run",
                    )
                    run_rec = TelemetryRunRecord(
                        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
                        run_id=run_id,
                        attempts=[attempt_rec],
                    )
                    record["telemetry"] = run_rec.to_dict()
            replacement = _preserve_run_telemetry(existing, record)
            reports[variant][reports[variant].index(existing)] = replacement
            if replacement.get("status") == "completed":
                parameter_counts[variant] = replacement["parameter_count"]
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
                        telemetry=telemetry_section,
                    ),
                )

    if collect_telemetry and telemetry_section is not None:
        try:
            telemetry_section["finished_at"] = utc_now().isoformat()
            latest_resources = collect_resource_snapshot()
            if latest_resources is not None:
                telemetry_section["resources"] = latest_resources.to_dict()
        except Exception as exc:
            _report_telemetry_diagnostic("controlled_comparison.final_telemetry", exc)

    aggregates = _build_aggregates(reports, metric_names, variants)
    result = _checkpoint_report(
        configuration,
        reports,
        aggregates,
        parameter_counts,
        input_usage,
        metric_names,
        telemetry=telemetry_section,
    )
    if checkpoint is not None:
        _persist_checkpoint(checkpoint, result)
    return result


def _build_aggregates(
    reports: dict[str, list[dict[str, Any]]],
    metric_names: tuple[str, ...],
    variants: tuple[str, ...],
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for variant in variants:
        completed = [run for run in reports.get(variant, []) if run.get("status") == "completed"]
        if not completed:
            continue
        aggregates[variant] = {
            "mean": {},
            "std": {},
            "n_values": {},
            "n_missing": {},
            "n_invalid": {},
            "seeds": {},
        }
        for name in metric_names:
            observations = [
                (run["seed"], run["final_validation"].get(name))
                for run in completed
                if "final_validation" in run
            ]
            summary = aggregate_metric_values(observations)
            aggregates[variant]["mean"][name] = summary.mean
            aggregates[variant]["std"][name] = summary.std
            aggregates[variant]["n_values"][name] = summary.n_values
            aggregates[variant]["n_missing"][name] = summary.n_missing
            aggregates[variant]["n_invalid"][name] = summary.n_invalid
            aggregates[variant]["seeds"][name] = summary.seeds_used
    return aggregates


def _build_paired_deltas(
    reports: dict[str, list[dict[str, Any]]],
    metric_names: tuple[str, ...],
    variants: tuple[str, ...],
) -> list[dict[str, Any]]:
    potential_pairs = (("A", "C"), ("A", "D"), ("C", "D"), ("A", "B"))
    active_pairs = [
        (base, cand)
        for base, cand in potential_pairs
        if base in variants and cand in variants
    ]
    if not active_pairs and len(variants) >= 2:
        active_pairs = [(variants[0], v) for v in variants[1:]]

    results: list[dict[str, Any]] = []
    for base, cand in active_pairs:
        completed_base = {
            r["seed"]: r.get("final_validation", {})
            for r in reports.get(base, [])
            if r.get("status") == "completed"
        }
        completed_cand = {
            r["seed"]: r.get("final_validation", {})
            for r in reports.get(cand, [])
            if r.get("status") == "completed"
        }
        common_seeds = sorted(set(completed_base.keys()) & set(completed_cand.keys()))
        metrics_dict: dict[str, Any] = {}

        for name in metric_names:
            if "loss" in name:
                orientation = "base - candidate"
                unit = "loss"
            else:
                orientation = "candidate - base"
                unit = (
                    "fraction"
                    if any(k in name for k in ("accuracy", "precision", "recall", "f1", "specificity"))
                    else "count"
                )

            base_obs = {s: completed_base[s].get(name) for s in common_seeds}
            cand_obs = {s: completed_cand[s].get(name) for s in common_seeds}
            summary = summarize_paired_deltas(
                base_obs,
                cand_obs,
                orientation=orientation,
                unit=unit,
            )
            metrics_dict[name] = summary.to_dict()

        results.append({
            "base": base,
            "candidate": cand,
            "n_common_seeds": len(common_seeds),
            "common_seeds": common_seeds,
            "metrics": metrics_dict,
        })
    return results


def _sanitize_report_telemetry(
    reports: dict[str, list[dict[str, Any]]],
) -> None:
    for variant, runs in reports.items():
        for run in runs:
            if "telemetry" not in run:
                continue
            telemetry = run.get("telemetry")
            if telemetry is None:
                run.pop("telemetry", None)
                continue
            prepared = _merge_run_payloads(
                None,
                telemetry,
                component=f"experiments.comparison.report.{variant}",
            )
            if prepared is None:
                run.pop("telemetry", None)
            else:
                run["telemetry"] = prepared


def _checkpoint_report(
    configuration: dict[str, Any],
    reports: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, dict[str, float]]],
    parameter_counts: dict[str, int],
    input_usage: dict[str, dict[str, int]],
    metric_names: tuple[str, ...],
    telemetry: TelemetryEnvelope | dict[str, Any] | None = None,
) -> dict[str, Any]:
    _sanitize_report_telemetry(reports)
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
    paired_deltas = _build_paired_deltas(
        reports,
        metric_names,
        tuple(configuration["variants"]),
    )
    report = {
        **configuration,
        "status": "failed" if failed_runs else ("completed" if completed_runs == total_runs else "running"),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "runs": reports,
        "aggregates": aggregates,
        "paired_deltas": paired_deltas,
        "parameter_counts": parameter_counts,
        "input_usage": input_usage,
    }
    telemetry_section = _serialize_telemetry_envelope(telemetry)
    if telemetry_section is not None:
        report["telemetry"] = telemetry_section
    return report


def _serialize_telemetry_envelope(
    value: TelemetryEnvelope | dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _prepare_envelope_payload(
        value,
        component="experiments.comparison.envelope",
    )


def _preserve_run_telemetry(
    previous: dict[str, Any] | None,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    result = dict(replacement)
    prior = previous.get("telemetry") if previous is not None else None
    incoming = result.get("telemetry")
    merged = _merge_run_payloads(
        prior,
        incoming,
        component="experiments.comparison.run",
    )
    if merged is None:
        result.pop("telemetry", None)
    else:
        result["telemetry"] = merged
    return result


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
