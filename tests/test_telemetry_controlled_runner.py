"""Deterministic tests for controlled runner telemetry integration (Tranche C of Issue #27).

Validates:
1. Telemetry disabled by default: no clock/snapshot calls, legacy fields preserved.
2. Telemetry enabled opt-in: valid envelope, run records, attempt/segment/stage identities.
3. Partial/missing environment: procfs absence or snapshot failures do not disrupt science.
4. Error boundaries: training failures remain failed; telemetry errors do not mask status.
5. Retry and resume: previous attempts are preserved, completed runs skipped, clean round-trip.
6. CPU-only: no CUDA imports, builds, or device initializations on the collection path.
7. Parallel CPU execution: workers manage independent lifecycle without shared mutable state.
8. Backward compatibility: legacy checkpoints without telemetry load and resume cleanly.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from kinera.experiments import comparison as comparison_module
from kinera.experiments.comparison import (
    CONTROLLED_VARIANTS,
    _execute_single_experiment,
    run_controlled_experiments,
)
from kinera.experiments.config import ExecutionConfig
from kinera.language.vocabulary import CharacterVocabulary
from kinera.monitor.telemetry.records import (
    TELEMETRY_SCHEMA_VERSION,
    TelemetryEnvelope,
    TelemetryRunRecord,
)
from kinera.monitor.telemetry.timing import TimingSession
from scripts.compare import build_parser, main as compare_main


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value}")
    return parsed


class ControlledRunnerTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b", "c"])
        self.examples = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 1,
                "target_is_syllable_end": 1,
            },
            {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 2],
                "context_feature_vectors": [[1.0] + [0.0] * 9, [0.0] * 10],
                "target_char_id": 5,
                "target_syllable_id": 2,
                "target_is_syllable_end": 0,
            },
        ]

    # -------------------------------------------------------------------------
    # 1. Telemetria desativada
    # -------------------------------------------------------------------------
    def test_telemetry_disabled_by_default_performs_no_collection_and_preserves_legacy_fields(
        self,
    ) -> None:
        clock_wall_mock = MagicMock(side_effect=AssertionError("TimingSession._read_wall was called"))
        clock_mono_mock = MagicMock(side_effect=AssertionError("TimingSession._read_monotonic was called"))
        env_snapshot_mock = MagicMock(side_effect=AssertionError("collect_environment_snapshot was called"))
        res_snapshot_mock = MagicMock(side_effect=AssertionError("collect_resource_snapshot was called"))

        with (
            patch.object(TimingSession, "_read_wall", clock_wall_mock),
            patch.object(TimingSession, "_read_monotonic", clock_mono_mock),
            patch(
                "kinera.experiments.comparison.collect_environment_snapshot",
                env_snapshot_mock,
            ),
            patch(
                "kinera.experiments.comparison.collect_resource_snapshot",
                res_snapshot_mock,
            ),
        ):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=2,
                variants=("A",),
                execution_config=ExecutionConfig(workers=1),
            )

        self.assertEqual(report["status"], "completed")
        self.assertNotIn("telemetry", report)
        run_record = report["runs"]["A"][0]
        self.assertNotIn("telemetry", run_record)
        self.assertEqual(run_record["seed"], 7)
        self.assertEqual(run_record["status"], "completed")
        self.assertIsInstance(run_record["pid"], int)
        self.assertGreater(run_record["pid"], 0)
        self.assertIsInstance(run_record["max_rss_kb"], int)
        self.assertGreater(run_record["max_rss_kb"], 0)
        self.assertIsInstance(run_record["duration_seconds"], float)
        self.assertGreater(run_record["duration_seconds"], 0.0)
        self.assertIn("initial_validation", run_record)
        self.assertIn("training", run_record)
        self.assertIn("final_validation", run_record)
        self.assertIn("parameter_count", run_record)

    # -------------------------------------------------------------------------
    # 2. Telemetria ativada
    # -------------------------------------------------------------------------
    def test_telemetry_enabled_opt_in_generates_valid_envelope_and_run_records(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=3,
            seeds=[7],
            epochs=2,
            learning_rate=0.05,
            max_examples=2,
            variants=("A",),
            collect_telemetry=True,
            execution_config=ExecutionConfig(workers=1),
        )

        self.assertEqual(report["status"], "completed")
        self.assertIn("telemetry", report)
        envelope = report["telemetry"]
        self.assertEqual(envelope["telemetry_schema_version"], TELEMETRY_SCHEMA_VERSION)
        self.assertIsInstance(envelope["session_id"], str)
        self.assertTrue(envelope["session_id"].startswith("controlled-comparison-"))
        self.assertIsNotNone(envelope["started_at"])
        self.assertIsNotNone(envelope["finished_at"])
        started_dt = _parse_iso(envelope["started_at"])
        finished_dt = _parse_iso(envelope["finished_at"])
        self.assertGreaterEqual(finished_dt, started_dt)
        self.assertIn("environment", envelope)
        self.assertEqual(envelope["environment"]["environment_schema_version"], 1)
        self.assertIn("resources", envelope)
        self.assertEqual(envelope["resources"]["resources_schema_version"], 1)

        run_record = report["runs"]["A"][0]
        self.assertIn("telemetry", run_record)
        run_telemetry = run_record["telemetry"]
        self.assertEqual(run_telemetry["telemetry_schema_version"], TELEMETRY_SCHEMA_VERSION)
        self.assertEqual(run_telemetry["run_id"], "A-seed-7")
        self.assertEqual(len(run_telemetry["attempts"]), 1)

        attempt = run_telemetry["attempts"][0]
        self.assertEqual(attempt["attempt_id"], "attempt-1")
        self.assertEqual(attempt["process_id"], "controlled-A-seed-7-attempt-1")
        self.assertEqual(attempt["pid"], os.getpid())
        self.assertNotEqual(attempt["process_id"], str(attempt["pid"]))
        self.assertEqual(attempt["status"], "completed")
        self.assertIsNone(attempt["error"])
        self.assertEqual(attempt["cleanup_status"], "completed")
        self.assertIsNone(attempt["cleanup_error"])
        self.assertIsInstance(attempt["duration_seconds"], float)
        self.assertGreater(attempt["duration_seconds"], 0.0)

        self.assertEqual(len(attempt["segments"]), 1)
        segment = attempt["segments"][0]
        self.assertEqual(segment["segment_id"], "segment-attempt-1")
        self.assertEqual(segment["status"], "completed")

        stages = segment["stages"]
        stage_names = [s["name"] for s in stages]
        self.assertEqual(
            stage_names,
            [
                "setup",
                "initial_validation",
                "training",
                "epoch_sgd",
                "epoch_evaluation",
                "epoch_sgd",
                "epoch_evaluation",
                "final_validation",
                "cleanup",
            ],
        )

        training_stage = next(s for s in stages if s["name"] == "training")
        self.assertIsNone(training_stage["parent_stage_id"])
        self.assertIsInstance(training_stage["duration_seconds"], float)
        self.assertIsInstance(training_stage["exclusive_duration_seconds"], float)
        self.assertGreaterEqual(
            training_stage["duration_seconds"],
            training_stage["exclusive_duration_seconds"],
        )

        child_stages = [s for s in stages if s["parent_stage_id"] == "training"]
        self.assertEqual(len(child_stages), 4)
        for child in child_stages:
            self.assertEqual(child["status"], "completed")
            self.assertIsInstance(child["duration_seconds"], float)
            self.assertGreater(child["duration_seconds"], 0.0)

        # JSON round-trip safety check
        encoded = json.dumps(report, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        reloaded = json.loads(encoded)
        self.assertEqual(reloaded["status"], "completed")

    # -------------------------------------------------------------------------
    # 3. Ambiente parcial / falhas de snapshots
    # -------------------------------------------------------------------------
    def test_partial_environment_or_snapshot_failures_do_not_stop_training(self) -> None:
        def broken_environment(*args: Any, **kwargs: Any) -> Any:
            raise OSError("procfs unreadable: /proc/cpuinfo")

        def broken_resources(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError("permission denied: /proc/self/status")

        with (
            patch(
                "kinera.experiments.comparison.collect_environment_snapshot",
                side_effect=broken_environment,
            ),
            patch(
                "kinera.experiments.comparison.collect_resource_snapshot",
                side_effect=broken_resources,
            ),
            self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs,
        ):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=2,
                variants=("A",),
                collect_telemetry=True,
                execution_config=ExecutionConfig(workers=1),
            )

        self.assertEqual(report["status"], "completed")
        self.assertIn("telemetry", report)
        envelope = report["telemetry"]
        self.assertIsNone(envelope.get("environment"))
        self.assertIsNone(envelope.get("resources"))
        self.assertTrue(any("controlled_comparison" in line for line in logs.output))
        run_record = report["runs"]["A"][0]
        self.assertEqual(run_record["status"], "completed")
        self.assertIn("telemetry", run_record)
        self.assertEqual(run_record["telemetry"]["attempts"][0]["status"], "completed")

    # -------------------------------------------------------------------------
    # 4. Falhas reais vs erros de telemetria
    # -------------------------------------------------------------------------
    def test_scientific_failure_is_reported_faithfully_in_run_and_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fail.json"
            with patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=RuntimeError("scientific training divergence"),
            ):
                report = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=2,
                    variants=("A",),
                    checkpoint_path=checkpoint,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

        self.assertEqual(report["status"], "failed")
        run_record = report["runs"]["A"][0]
        self.assertEqual(run_record["status"], "failed")
        self.assertEqual(run_record["error"], "experiment failed")
        self.assertIn("scientific training divergence", run_record["traceback"])

        attempt = run_record["telemetry"]["attempts"][0]
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(attempt["error"], "scientific training divergence")
        self.assertEqual(attempt["cleanup_status"], "completed")

    def test_telemetry_clock_failure_is_fail_soft_without_altering_scientific_success(
        self,
    ) -> None:
        def broken_clock() -> int:
            raise OSError("monotonic clock failed")

        with patch.object(TimingSession, "_read_monotonic", side_effect=broken_clock):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=2,
                variants=("A",),
                collect_telemetry=True,
                execution_config=ExecutionConfig(workers=1),
            )

        self.assertEqual(report["status"], "completed")
        run_record = report["runs"]["A"][0]
        self.assertEqual(run_record["status"], "completed")
        attempt = run_record["telemetry"]["attempts"][0]
        self.assertEqual(attempt["status"], "completed")
        for stage in attempt["segments"][0]["stages"]:
            self.assertIn("monotonic clock failed", stage.get("timing_error", ""))

    # -------------------------------------------------------------------------
    # 5. Retry e resume com preservação de tentativas
    # -------------------------------------------------------------------------
    def test_retry_and_resume_preserves_failed_attempt_and_skips_completed_runs(
        self,
    ) -> None:
        call_count = 0
        orig_train = comparison_module.train_multitask_examples

        def conditionally_failing_trainer(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("transient failure on seed 42 attempt 1")
            return orig_train(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"

            with patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=conditionally_failing_trainer,
            ):
                first_run = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7, 42],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=2,
                    variants=("A",),
                    checkpoint_path=checkpoint,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

            self.assertEqual(first_run["status"], "failed")
            runs_first = {r["seed"]: r for r in first_run["runs"]["A"]}
            self.assertEqual(runs_first[7]["status"], "completed")
            self.assertEqual(runs_first[42]["status"], "failed")
            self.assertEqual(
                [a["attempt_id"] for a in runs_first[42]["telemetry"]["attempts"]],
                ["attempt-1"],
            )
            self.assertEqual(runs_first[42]["telemetry"]["attempts"][0]["status"], "failed")

            # Resume execution
            with patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=conditionally_failing_trainer,
            ):
                resumed_run = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7, 42],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=2,
                    variants=("A",),
                    checkpoint_path=checkpoint,
                    resume=True,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

            self.assertEqual(resumed_run["status"], "completed")
            runs_resumed = {r["seed"]: r for r in resumed_run["runs"]["A"]}
            self.assertEqual(runs_resumed[7]["status"], "completed")
            self.assertEqual(runs_resumed[42]["status"], "completed")

            # Total of 3 training calls across runs: seed 7 (ok), seed 42 attempt 1 (failed), seed 42 attempt 2 (ok)
            self.assertEqual(call_count, 3)

            telemetry_42 = runs_resumed[42]["telemetry"]
            attempts_42 = telemetry_42["attempts"]
            self.assertEqual(len(attempts_42), 2)
            self.assertEqual(attempts_42[0]["attempt_id"], "attempt-1")
            self.assertEqual(attempts_42[0]["status"], "failed")
            self.assertEqual(attempts_42[1]["attempt_id"], "attempt-2")
            self.assertEqual(attempts_42[1]["status"], "completed")
            self.assertEqual(
                attempts_42[0]["segments"][0]["segment_id"], "segment-attempt-1"
            )
            self.assertEqual(
                attempts_42[1]["segments"][0]["segment_id"], "segment-attempt-2"
            )

    # -------------------------------------------------------------------------
    # 6. CPU-only
    # -------------------------------------------------------------------------
    def test_cpu_only_guarantee_no_cuda_modules_imported_during_telemetry(self) -> None:
        cuda_keys_before = {k for k in sys.modules if "cuda" in k.lower()}

        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=3,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=2,
            variants=("A",),
            collect_telemetry=True,
            execution_config=ExecutionConfig(backend="cpu", workers=1),
        )

        cuda_keys_after = {k for k in sys.modules if "cuda" in k.lower()}
        new_cuda_keys = cuda_keys_after - cuda_keys_before
        self.assertEqual(new_cuda_keys, set())
        self.assertEqual(report["status"], "completed")

    # -------------------------------------------------------------------------
    # 7. Execução paralela CPU
    # -------------------------------------------------------------------------
    def test_parallel_execution_isolates_worker_lifecycles_and_measures_stages(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=3,
            seeds=[7, 42],
            epochs=1,
            learning_rate=0.05,
            max_examples=2,
            variants=("A", "C"),
            collect_telemetry=True,
            execution_config=ExecutionConfig(workers=2),
        )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["total_runs"], 4)
        self.assertEqual(report["completed_runs"], 4)

        for variant in ("A", "C"):
            for run_record in report["runs"][variant]:
                self.assertEqual(run_record["status"], "completed")
                self.assertIn("telemetry", run_record)
                run_tel = run_record["telemetry"]
                attempt = run_tel["attempts"][0]
                self.assertEqual(attempt["status"], "completed")
                self.assertTrue(attempt["process_id"].startswith("worker-"))
                self.assertIsNotNone(attempt["submitted_at"])
                self.assertIsNotNone(attempt["worker_started_at"])
                self.assertIsInstance(attempt["queue_duration_seconds"], float)
                self.assertGreaterEqual(attempt["queue_duration_seconds"], 0.0)

    # -------------------------------------------------------------------------
    # 8. Compatibilidade: checkpoints legados e relatórios
    # -------------------------------------------------------------------------
    def test_compatibility_with_legacy_checkpoints_without_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_checkpoint.json"
            metric_values = {
                "char_loss": 0.5,
                "syllable_loss": 0.5,
                "boundary_loss": 0.5,
                "total_loss": 0.5,
                "char_accuracy": 0.5,
                "syllable_accuracy_all": 0.5,
                "syllable_accuracy_known": 0.5,
                "boundary_accuracy": 0.5,
                "boundary_precision": 0.5,
                "boundary_recall": 0.5,
                "boundary_f1": 0.5,
            }
            legacy_data = {
                "phase": "controlled_comparison",
                "variants": ["A"],
                "seeds": [7, 42],
                "epochs": 1,
                "learning_rate": 0.05,
                "max_examples": 2,
                "syllable_vocab_size": 3,
                "syllable_unknown_id": 0,
                "loss_weights": {"char": 1.0, "syllable": 0.7, "boundary": 0.3},
                "experiment_config": {
                    "phase": "controlled_comparison",
                    "variants": ["A"],
                    "seeds": [7, 42],
                    "epochs": 1,
                    "learning_rate": 0.05,
                    "max_examples": 2,
                },
                "execution_config": {"backend": "cpu", "workers": 1},
                "workers": 1,
                "status": "running",
                "total_runs": 2,
                "completed_runs": 1,
                "failed_runs": 0,
                "runs": {
                    "A": [
                        {
                            "seed": 7,
                            "status": "completed",
                            "pid": 9999,
                            "max_rss_kb": 12345,
                            "duration_seconds": 0.42,
                            "initial_validation": dict(metric_values),
                            "training": {"epochs": 1},
                            "final_validation": dict(metric_values),
                            "parameter_count": 139432,
                        }
                    ]
                },
                "aggregates": {},
                "parameter_counts": {"A": 139432},
                "input_usage": {"A": {}},
            }
            checkpoint.write_text(json.dumps(legacy_data), encoding="utf-8")

            # Resume with telemetry enabled
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7, 42],
                epochs=1,
                learning_rate=0.05,
                max_examples=2,
                variants=("A",),
                checkpoint_path=checkpoint,
                resume=True,
                collect_telemetry=True,
                execution_config=ExecutionConfig(workers=1),
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_runs"], 2)
        runs = {r["seed"]: r for r in report["runs"]["A"]}

        # Seed 7 was pre-existing without telemetry: preserved as-is without synthetic injection
        self.assertEqual(runs[7]["pid"], 9999)
        self.assertEqual(runs[7]["max_rss_kb"], 12345)
        self.assertEqual(runs[7]["duration_seconds"], 0.42)
        self.assertNotIn("telemetry", runs[7])

        # Seed 42 was executed on resume with telemetry: has new telemetry record
        self.assertIn("telemetry", runs[42])
        self.assertEqual(runs[42]["telemetry"]["attempts"][0]["status"], "completed")

        # Top-level sections all intact
        for expected_key in (
            "phase",
            "variants",
            "seeds",
            "epochs",
            "learning_rate",
            "max_examples",
            "workers",
            "status",
            "total_runs",
            "completed_runs",
            "failed_runs",
            "runs",
            "aggregates",
            "parameter_counts",
            "input_usage",
            "telemetry",
        ):
            self.assertIn(expected_key, report)

    # -------------------------------------------------------------------------
    # CLI integration: compare.py
    # -------------------------------------------------------------------------
    def test_compare_cli_accepts_collect_telemetry_flag(self) -> None:
        parser = build_parser()
        args_without = parser.parse_args([
            "--vocabulary", "v.json",
            "--syllable-vocabulary", "s.json",
            "--train", "t.jsonl",
            "--validation", "val.jsonl",
        ])
        self.assertFalse(args_without.collect_telemetry)

        args_with = parser.parse_args([
            "--vocabulary", "v.json",
            "--syllable-vocabulary", "s.json",
            "--train", "t.jsonl",
            "--validation", "val.jsonl",
            "--collect-telemetry",
        ])
        self.assertTrue(args_with.collect_telemetry)

    # -------------------------------------------------------------------------
    # 9. Verificações aprofundadas pré-merge (Review da Issue #27 / PR #49)
    # -------------------------------------------------------------------------
    def test_session_id_uniqueness_across_rapid_and_simultaneous_sessions(self) -> None:
        """Sessões iniciadas em rápida sucessão têm session_id estritamente únicos."""
        session_ids: set[str] = set()
        for _ in range(5):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                variants=("A",),
                collect_telemetry=True,
                execution_config=ExecutionConfig(workers=1),
            )
            session_id = report["telemetry"]["session_id"]
            self.assertTrue(session_id.startswith("controlled-comparison-"))
            self.assertNotIn(session_id, session_ids)
            session_ids.add(session_id)
        self.assertEqual(len(session_ids), 5)

    def test_identities_and_history_round_trip_faithfully_after_checkpoint_persistence_and_reload(
        self,
    ) -> None:
        """Garante que identidades e histórico sobrevivem à persistência em disco e recarga no resume."""
        call_count = 0
        orig_train = comparison_module.train_multitask_examples

        def failing_on_attempt_1(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("disk round-trip test failure")
            return orig_train(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_file = Path(directory) / "checkpoint_roundtrip.json"

            # 1. Primeira execução: seed 7 tem sucesso, seed 42 falha no attempt 1
            with patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=failing_on_attempt_1,
            ):
                first_run = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7, 42],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    variants=("A",),
                    checkpoint_path=checkpoint_file,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

            self.assertEqual(first_run["status"], "failed")
            self.assertTrue(checkpoint_file.exists())

            # 2. Ler diretamente o arquivo JSON do disco
            persisted_raw = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            session_id_persisted = persisted_raw["telemetry"]["session_id"]
            runs_persisted = {r["seed"]: r for r in persisted_raw["runs"]["A"]}
            self.assertEqual(runs_persisted[7]["status"], "completed")
            self.assertEqual(runs_persisted[42]["status"], "failed")

            # Validar identidades e tipos persistidos no attempt 1 da seed 42
            attempt_1_raw = runs_persisted[42]["telemetry"]["attempts"][0]
            self.assertEqual(attempt_1_raw["attempt_id"], "attempt-1")
            self.assertEqual(attempt_1_raw["process_id"], "controlled-A-seed-42-attempt-1")
            self.assertEqual(attempt_1_raw["segments"][0]["segment_id"], "segment-attempt-1")
            self.assertNotEqual(attempt_1_raw["process_id"], str(attempt_1_raw["pid"]))

            # 3. Retomar execução a partir do disco com resume=True
            with patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=failing_on_attempt_1,
            ):
                resumed_run = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7, 42],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    variants=("A",),
                    checkpoint_path=checkpoint_file,
                    resume=True,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

            self.assertEqual(resumed_run["status"], "completed")
            self.assertEqual(call_count, 3)

            # 4. Ler novamente do disco para garantir round-trip completo após persistência
            resumed_raw = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            self.assertEqual(resumed_raw["telemetry"]["session_id"], session_id_persisted)
            resumed_runs_raw = {r["seed"]: r for r in resumed_raw["runs"]["A"]}

            # Seed 7 permanece intacto
            self.assertEqual(resumed_runs_raw[7]["status"], "completed")
            self.assertEqual(len(resumed_runs_raw[7]["telemetry"]["attempts"]), 1)

            # Seed 42 tem agora attempt-1 (failed) E attempt-2 (completed) preservados no disco
            attempts_42_disk = resumed_runs_raw[42]["telemetry"]["attempts"]
            self.assertEqual(len(attempts_42_disk), 2)
            self.assertEqual(attempts_42_disk[0]["attempt_id"], "attempt-1")
            self.assertEqual(attempts_42_disk[0]["status"], "failed")
            self.assertEqual(attempts_42_disk[0]["process_id"], "controlled-A-seed-42-attempt-1")
            self.assertEqual(attempts_42_disk[1]["attempt_id"], "attempt-2")
            self.assertEqual(attempts_42_disk[1]["status"], "completed")
            self.assertEqual(attempts_42_disk[1]["process_id"], "controlled-A-seed-42-attempt-2")
            self.assertEqual(attempts_42_disk[1]["segments"][0]["segment_id"], "segment-attempt-2")

    def test_resume_from_telemetry_checkpoint_with_telemetry_disabled(self) -> None:
        """Retomada com collect_telemetry=False preserva telemetria anterior sem coletar novas."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_file = Path(directory) / "checkpoint_mixed.json"

            # 1. Executa seed 7 com telemetria
            run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                variants=("A",),
                checkpoint_path=checkpoint_file,
                collect_telemetry=True,
                execution_config=ExecutionConfig(workers=1),
            )

            # 2. Retoma expandindo para seeds=[7, 42] com collect_telemetry=False
            resumed = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7, 42],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                variants=("A",),
                checkpoint_path=checkpoint_file,
                resume=True,
                collect_telemetry=False,
                execution_config=ExecutionConfig(workers=1),
            )

            self.assertEqual(resumed["status"], "completed")
            runs = {r["seed"]: r for r in resumed["runs"]["A"]}
            # Seed 7 manteve sua telemetria original
            self.assertIn("telemetry", runs[7])
            self.assertEqual(len(runs[7]["telemetry"]["attempts"]), 1)
            # Seed 42 executou sem telemetria
            self.assertNotIn("telemetry", runs[42])

    def test_cleanup_stage_error_does_not_mask_scientific_error_and_avoids_duplicate_stage_id(
        self,
    ) -> None:
        """Falha no cleanup após falha científica reporta ambos sem erro de duplicate stage_id."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cleanup_error.json"

            def failing_close(self_model: Any) -> None:
                raise OSError("cleanup close failed")

            with patch.object(
                comparison_module.AutoregressiveModel, "close", failing_close
            ), patch.object(
                comparison_module,
                "train_multitask_examples",
                side_effect=RuntimeError("scientific failure"),
            ):
                report = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    variants=("A",),
                    checkpoint_path=checkpoint,
                    collect_telemetry=True,
                    execution_config=ExecutionConfig(workers=1),
                )

            self.assertEqual(report["status"], "failed")
            run = report["runs"]["A"][0]
            self.assertEqual(run["status"], "failed")
            self.assertIn("scientific failure", run["traceback"])
            self.assertIn("cleanup close failed", run["traceback"])

            attempt = run["telemetry"]["attempts"][0]
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["cleanup_status"], "failed")
            self.assertEqual(attempt["cleanup_error"], "cleanup close failed")

    def test_trainer_stages_strictly_mirror_sgd_and_evaluation_intervals(self) -> None:
        """Confirma que epoch_sgd e epoch_evaluation medem exatamente os intervalos corretos sem alterar a ordem."""
        from kinera.training.trainer import train_multitask_examples
        from kinera.nn.models.hybrid import AutoregressiveModel

        model = AutoregressiveModel(len(self.vocabulary), 3, seed=7)
        timing = TimingSession()

        events: list[str] = []
        orig_step = model.multitask_train_step

        def tracked_step(*args: Any, **kwargs: Any) -> Any:
            events.append("sgd_step")
            return orig_step(*args, **kwargs)

        model.multitask_train_step = tracked_step  # type: ignore[assignment]

        with timing.stage(stage_id="training_parent", name="training"):
            train_multitask_examples(
                model,
                self.examples,
                self.vocabulary,
                epochs=2,
                learning_rate=0.05,
                max_examples=2,
                timing_session=timing,
                stage_id_prefix="training_parent",
            )

        stages_by_id = {s.stage_id: s for s in timing.stages}
        self.assertIn("training_parent", stages_by_id)
        self.assertIn("training_parent:epoch_1_sgd", stages_by_id)
        self.assertIn("training_parent:epoch_1_eval", stages_by_id)
        self.assertIn("training_parent:epoch_2_sgd", stages_by_id)
        self.assertIn("training_parent:epoch_2_eval", stages_by_id)

        # SGD steps happened (2 epochs * 2 examples = 4 steps)
        self.assertEqual(len(events), 4)

        # Each child stage has valid duration and parent
        for s in (
            stages_by_id["training_parent:epoch_1_sgd"],
            stages_by_id["training_parent:epoch_1_eval"],
            stages_by_id["training_parent:epoch_2_sgd"],
            stages_by_id["training_parent:epoch_2_eval"],
        ):
            self.assertEqual(s.parent_stage_id, "training_parent")
            self.assertEqual(s.status, "completed")
            self.assertIsNotNone(s.duration_seconds)
            self.assertGreater(s.duration_seconds, 0.0)

        # Exclusive duration of parent excludes children duration
        parent = stages_by_id["training_parent"]
        self.assertIsNotNone(parent.exclusive_duration_seconds)
        self.assertLess(parent.exclusive_duration_seconds, parent.duration_seconds)


if __name__ == "__main__":
    unittest.main()
