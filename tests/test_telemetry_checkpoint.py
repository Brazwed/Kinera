import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kinera.experiments import comparison as comparison_module
from kinera.experiments.comparison import (
    CONTROLLED_VARIANTS,
    _build_aggregates,
    _checkpoint_report,
    _persist_checkpoint,
    run_controlled_experiments,
)
from kinera.experiments.config import ExecutionConfig, ExperimentConfig
from kinera.experiments.multi_seed import prepare_seed_records, run_seed_jobs
from kinera.experiments.runner_common import aggregate_run_dicts, write_json_checkpoint
from kinera.language.vocabulary import CharacterVocabulary
from kinera.monitor.telemetry.records import (
    TELEMETRY_SCHEMA_VERSION,
    TelemetryAttemptRecord,
    TelemetryEnvelope,
    TelemetryRunRecord,
    TelemetrySegmentRecord,
)
from kinera.monitor.telemetry.timing import TimingSession


UTC = "2026-01-02T03:04:05+00:00"
METRIC_NAMES = (
    "char_loss", "syllable_loss", "boundary_loss", "total_loss",
    "char_accuracy", "syllable_accuracy_all", "syllable_accuracy_known",
    "boundary_accuracy", "boundary_precision", "boundary_recall", "boundary_f1",
)


def make_run_telemetry(run_id: str, attempt_id: str, status: str) -> dict:
    return TelemetryRunRecord(
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        run_id=run_id,
        attempts=[TelemetryAttemptRecord(
            attempt_id=attempt_id,
            process_id=f"logical-{attempt_id}",
            pid=1234,
            status=status,
            started_at=UTC,
            finished_at=UTC,
            duration_seconds=0.0,
            segments=[TelemetrySegmentRecord(
                segment_id=f"segment-{attempt_id}",
                status=status,
                started_at=UTC,
                finished_at=UTC,
                duration_seconds=0.0,
            )],
        )],
    ).to_dict()


def comparison_fixture(status: str = "completed", *, run_telemetry=None) -> tuple[dict, dict, dict, dict, dict]:
    experiment = ExperimentConfig(
        phase="controlled_comparison",
        variants=CONTROLLED_VARIANTS,
        seeds=(7,),
        epochs=1,
        learning_rate=0.05,
        max_examples=1,
        context_size=None,
    ).to_dict()
    experiment.update({
        "syllable_vocab_size": 3,
        "syllable_unknown_id": 0,
        "loss_weights": {"char": 1.0, "syllable": 0.7, "boundary": 0.3},
    })
    config = {
        **experiment,
        "experiment_config": experiment.copy(),
        "execution_config": ExecutionConfig().to_dict(),
        "workers": 1,
    }
    metrics = {name: 0.0 for name in METRIC_NAMES}
    reports = {
        variant: [{
            "seed": 7,
            "status": status,
            "initial_validation": dict(metrics),
            "training": {"epochs": 1},
            "final_validation": dict(metrics),
            "parameter_count": 12,
            **({"error": "old attempt failed"} if status == "failed" else {}),
            **({"telemetry": run_telemetry} if run_telemetry is not None else {}),
        }]
        for variant in CONTROLLED_VARIANTS
    }
    aggregates = _build_aggregates(reports, METRIC_NAMES, CONTROLLED_VARIANTS)
    parameters = {variant: 12 for variant in CONTROLLED_VARIANTS}
    input_usage = {variant: {} for variant in CONTROLLED_VARIANTS}
    return config, reports, aggregates, parameters, input_usage


class TelemetryCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a"])
        self.examples = [{
            "context_ids": [0],
            "context_syllable_ids": [0],
            "context_feature_vectors": [[0.0] * 10],
            "target_char_id": 3,
            "target_syllable_id": 0,
            "target_is_syllable_end": 0,
        }]

    def test_generic_runner_default_writes_no_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "run.json"
            with (
                patch.object(TimingSession, "_read_wall", side_effect=AssertionError("clock used")),
                patch.object(TimingSession, "_read_monotonic", side_effect=AssertionError("clock used")),
            ):
                report = run_seed_jobs(
                    seeds=[7],
                    experiment_config={"phase": "cpu-only", "epochs": 1},
                    execution_config={"workers": 1},
                    checkpoint_path=checkpoint,
                    worker=lambda seed: {"seed": seed, "status": "completed"},
                )
            persisted = json.loads(checkpoint.read_text())

        self.assertNotIn("telemetry", report)
        self.assertNotIn("telemetry", persisted)
        self.assertNotIn("telemetry", persisted["runs"][0])

    def test_generic_runner_passes_explicit_envelope_without_attaching_to_runs(self) -> None:
        envelope = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="provided-session",
            started_at=UTC,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "run.json"
            report = run_seed_jobs(
                seeds=[7],
                experiment_config={"phase": "cpu-only", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=lambda seed: {"seed": seed, "status": "completed"},
                telemetry=envelope,
            )
            persisted = json.loads(checkpoint.read_text())

        self.assertEqual(report["telemetry"], envelope.to_dict())
        self.assertEqual(persisted["telemetry"], envelope.to_dict())
        self.assertNotIn("telemetry", report["runs"][0])

    def test_generic_resume_appends_retry_attempt_and_keeps_completed_history(self) -> None:
        session = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="session-1",
            started_at=UTC,
        ).to_dict()
        completed_run = make_run_telemetry("run-7", "attempt-7", "completed")
        failed_run = make_run_telemetry("run-42", "attempt-1", "failed")
        attempt_2 = make_run_telemetry("run-42", "attempt-2", "completed")
        checkpoint_value = {
            "phase": "test",
            "epochs": 1,
            "workers": 1,
            "seeds": [7, 42],
            "telemetry": session,
            "runs": [
                {"seed": 7, "status": "completed", "telemetry": completed_run},
                {"seed": 42, "status": "failed", "telemetry": failed_run},
            ],
        }
        worker_calls = []

        def retry_worker(seed: int) -> dict:
            worker_calls.append(seed)
            return {
                "seed": seed,
                "status": "completed",
                "payload": {"value": seed},
                "telemetry": attempt_2,
            }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "seed-run.json"
            write_json_checkpoint(checkpoint, checkpoint_value)
            first = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=retry_worker,
                resume=True,
            )
            after_first = json.loads(checkpoint.read_text())
            first_retry = next(run for run in first["runs"] if run["seed"] == 42)
            self.assertEqual(
                [item["attempt_id"] for item in first_retry["telemetry"]["attempts"]],
                ["attempt-1", "attempt-2"],
            )
            self.assertEqual(first_retry["telemetry"]["attempts"][0]["status"], "failed")
            self.assertEqual(first_retry["telemetry"]["attempts"][1]["status"], "completed")
            first_attempt_record, second_attempt_record = first_retry["telemetry"]["attempts"]
            self.assertEqual(
                [segment["segment_id"] for segment in first_attempt_record["segments"]],
                ["segment-attempt-1"],
            )
            self.assertEqual(
                [segment["segment_id"] for segment in second_attempt_record["segments"]],
                ["segment-attempt-2"],
            )
            self.assertNotIn(
                "segment-attempt-1",
                [segment["segment_id"] for segment in second_attempt_record["segments"]],
            )
            completed_after_first = next(run for run in first["runs"] if run["seed"] == 7)
            self.assertEqual(completed_after_first["telemetry"], completed_run)
            self.assertEqual(after_first["telemetry"], session)

            second = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=lambda seed: self.fail("completed runs must be skipped"),
                resume=True,
            )

        self.assertEqual(worker_calls, [42])
        retry_after_second = next(run for run in second["runs"] if run["seed"] == 42)
        self.assertEqual(
            [item["attempt_id"] for item in retry_after_second["telemetry"]["attempts"]],
            ["attempt-1", "attempt-2"],
        )
        self.assertEqual(second["telemetry"], session)
        self.assertEqual(second["runs"], after_first["runs"])
        self.assertEqual(retry_after_second["telemetry"], first_retry["telemetry"])

    def test_worker_result_without_telemetry_does_not_erase_old_attempts(self) -> None:
        old_run = make_run_telemetry("run-42", "attempt-1", "failed")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "seed-run.json"
            write_json_checkpoint(checkpoint, {
                "phase": "test",
                "epochs": 1,
                "workers": 1,
                "seeds": [42],
                "runs": [{"seed": 42, "status": "failed", "telemetry": old_run}],
            })
            report = run_seed_jobs(
                seeds=[42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=lambda seed: {"seed": seed, "status": "completed"},
                resume=True,
            )

        self.assertEqual(report["runs"][0]["status"], "completed")
        self.assertEqual(report["runs"][0]["telemetry"], old_run)

    def test_duplicate_checkpoint_rows_keep_completed_record_and_merge_attempt_history(self) -> None:
        first_attempt = make_run_telemetry("run-42", "attempt-1", "failed")
        second_attempt = make_run_telemetry("run-42", "attempt-2", "completed")

        prepared = prepare_seed_records(
            [
                {"seed": 42, "status": "failed", "telemetry": first_attempt},
                {"seed": 42, "status": "completed", "telemetry": second_attempt},
            ],
            [42],
        )
        prepared_without_new_telemetry = prepare_seed_records(
            [
                {"seed": 42, "status": "failed", "telemetry": first_attempt},
                {"seed": 42, "status": "completed"},
            ],
            [42],
        )

        self.assertEqual(prepared[0]["status"], "completed")
        self.assertEqual(
            [attempt["attempt_id"] for attempt in prepared[0]["telemetry"]["attempts"]],
            ["attempt-1", "attempt-2"],
        )
        self.assertEqual(prepared_without_new_telemetry[0]["status"], "completed")
        self.assertEqual(prepared_without_new_telemetry[0]["telemetry"], first_attempt)

    def test_invalid_new_telemetry_is_diagnosed_without_changing_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid-new.json"
            with self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs:
                report = run_seed_jobs(
                    seeds=[7],
                    experiment_config={"phase": "telemetry-test", "epochs": 1},
                    execution_config={"workers": 1},
                    checkpoint_path=checkpoint,
                    worker=lambda seed: {
                        "seed": seed,
                        "status": "completed",
                        "payload": {"science": "ok"},
                        "telemetry": {"run_id": "run-7", "attempts": []},
                    },
                )

        self.assertEqual(report["runs"][0]["status"], "completed")
        self.assertEqual(report["runs"][0]["payload"], {"science": "ok"})
        self.assertNotIn("telemetry", report["runs"][0])
        self.assertTrue(any("TelemetryRecordError" in line for line in logs.output))

    def test_invalid_new_telemetry_preserves_valid_prior_history(self) -> None:
        old_telemetry = make_run_telemetry("run-7", "attempt-1", "failed")
        old_telemetry["attempts"][0]["segments"][0]["stages"] = [{
            "stage_id": "stage-attempt-1",
            "name": "fit",
            "status": "completed",
        }]
        incoming_telemetry = make_run_telemetry("other-run", "attempt-2", "completed")
        incoming_telemetry["attempts"][0]["segments"][0]["stages"] = [{
            "stage_id": "stage-attempt-2",
            "name": "fit",
            "status": "completed",
        }]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid-replacement.json"
            write_json_checkpoint(checkpoint, {
                "phase": "telemetry-test",
                "epochs": 1,
                "workers": 1,
                "seeds": [7],
                "runs": [{"seed": 7, "status": "failed", "telemetry": old_telemetry}],
            })
            with self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs:
                report = run_seed_jobs(
                    seeds=[7],
                    experiment_config={"phase": "telemetry-test", "epochs": 1},
                    execution_config={"workers": 1},
                    checkpoint_path=checkpoint,
                    resume=True,
                    worker=lambda seed: {
                        "seed": seed,
                        "status": "completed",
                        "telemetry": incoming_telemetry,
                    },
                )

        self.assertEqual(report["runs"][0]["status"], "completed")
        preserved_telemetry = report["runs"][0]["telemetry"]
        self.assertEqual(preserved_telemetry, old_telemetry)
        self.assertEqual(preserved_telemetry["run_id"], "run-7")
        attempts = preserved_telemetry["attempts"]
        self.assertEqual([item["attempt_id"] for item in attempts], ["attempt-1"])
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertEqual(
            [segment["segment_id"] for segment in attempts[0]["segments"]],
            ["segment-attempt-1"],
        )
        self.assertEqual(
            [stage["stage_id"] for stage in attempts[0]["segments"][0]["stages"]],
            ["stage-attempt-1"],
        )
        self.assertTrue(any("TelemetryRecordError" in line for line in logs.output))

    def test_nonfinite_telemetry_envelope_is_rejected_without_global_json_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "nonfinite-envelope.json"
            with self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs:
                report = run_seed_jobs(
                    seeds=[7],
                    experiment_config={"phase": "telemetry-test", "epochs": 1},
                    execution_config={"workers": 1},
                    checkpoint_path=checkpoint,
                    worker=lambda seed: {"seed": seed, "status": "completed"},
                    telemetry={
                        "telemetry_schema_version": 1,
                        "session_id": "session-1",
                        "extra": float("nan"),
                    },
                )
            persisted_text = checkpoint.read_text()

        self.assertEqual(report["status"], "completed")
        self.assertNotIn("telemetry", report)
        self.assertNotIn("NaN", persisted_text)
        self.assertTrue(any("telemetry" in line.lower() for line in logs.output))

    def test_completed_checkpoint_invalid_run_telemetry_is_diagnosed_before_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid-completed.json"
            write_json_checkpoint(checkpoint, {
                "phase": "telemetry-test",
                "epochs": 1,
                "workers": 1,
                "seeds": [7],
                "runs": [{
                    "seed": 7,
                    "status": "completed",
                    "payload": {"science": "ok"},
                    "telemetry": {"run_id": "run-7", "attempts": []},
                }],
            })
            with self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs:
                report = run_seed_jobs(
                    seeds=[7],
                    experiment_config={"phase": "telemetry-test", "epochs": 1},
                    execution_config={"workers": 1},
                    checkpoint_path=checkpoint,
                    resume=True,
                    worker=lambda seed: self.fail("completed seed must not be rerun"),
                )
            persisted = json.loads(checkpoint.read_text())

        self.assertEqual(report["status"], "completed")
        self.assertNotIn("telemetry", report["runs"][0])
        self.assertNotIn("telemetry", persisted["runs"][0])
        self.assertTrue(any("TelemetryRecordError" in line for line in logs.output))

    def test_comparison_checkpoint_diagnoses_invalid_existing_run_telemetry(self) -> None:
        config, reports, aggregates, parameters, input_usage = comparison_fixture(
            run_telemetry={"run_id": "run-7", "attempts": []}
        )
        with self.assertLogs("kinera.monitor.telemetry", level="WARNING") as logs:
            report = _checkpoint_report(
                config,
                reports,
                aggregates,
                parameters,
                input_usage,
                METRIC_NAMES,
            )

        self.assertEqual(report["status"], "completed")
        self.assertTrue(all(
            "telemetry" not in runs[0] for runs in report["runs"].values()
        ))
        self.assertTrue(any("TelemetryRecordError" in line for line in logs.output))

    def test_bad_telemetry_does_not_change_scientific_status_aggregation(self) -> None:
        summary = aggregate_run_dicts([
            {
                "seed": 7,
                "status": "completed",
                "telemetry": {"telemetry_schema_version": "broken"},
            }
        ])
        self.assertEqual(summary["completed_runs"], 1)
        self.assertEqual(summary["status"], "completed")

    def test_comparison_checkpoint_passes_through_optional_envelope_and_old_absence(self) -> None:
        envelope = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="comparison-session",
            started_at=UTC,
        ).to_dict()
        config, reports, aggregates, parameters, input_usage = comparison_fixture(
            run_telemetry=make_run_telemetry("run-7", "attempt-1", "completed")
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "comparison.json"
            report = _checkpoint_report(
                config, reports, aggregates, parameters, input_usage, METRIC_NAMES,
                telemetry=envelope,
            )
            _persist_checkpoint(checkpoint, report)
            resumed = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=3,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
                resume=True,
                variants=CONTROLLED_VARIANTS,
                execution_config=ExecutionConfig(),
            )
            persisted = json.loads(checkpoint.read_text())

            legacy_path = Path(directory) / "legacy.json"
            legacy_report = _checkpoint_report(
                config, reports, aggregates, parameters, input_usage, METRIC_NAMES
            )
            _persist_checkpoint(legacy_path, legacy_report)
            with (
                patch.object(TimingSession, "_read_wall", side_effect=AssertionError("clock used")),
                patch.object(TimingSession, "_read_monotonic", side_effect=AssertionError("clock used")),
            ):
                legacy_resumed = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    checkpoint_path=legacy_path,
                    resume=True,
                    variants=CONTROLLED_VARIANTS,
                    execution_config=ExecutionConfig(),
                )
            legacy_persisted = json.loads(legacy_path.read_text())

        self.assertEqual(resumed["telemetry"], envelope)
        self.assertEqual(persisted["telemetry"], envelope)
        self.assertNotIn("telemetry", legacy_resumed)
        self.assertNotIn("telemetry", legacy_persisted)
        for variant in CONTROLLED_VARIANTS:
            self.assertEqual(
                resumed["runs"][variant][0]["telemetry"],
                reports[variant][0]["telemetry"],
            )

    def test_comparison_retry_keeps_prior_telemetry_without_creating_new_attempt(self) -> None:
        old_telemetry = make_run_telemetry("run-7", "attempt-1", "failed")
        config, reports, aggregates, parameters, input_usage = comparison_fixture(
            status="failed", run_telemetry=old_telemetry
        )
        checkpoint = _checkpoint_report(
            config, reports, aggregates, parameters, input_usage, METRIC_NAMES,
            telemetry=None,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison-retry.json"
            _persist_checkpoint(path, checkpoint)
            fake_model = MagicMock()
            metrics = {name: 0.0 for name in METRIC_NAMES}
            with (
                patch.object(comparison_module, "AutoregressiveModel", return_value=fake_model),
                patch.object(comparison_module, "count_trainable_parameters", return_value=12),
                patch.object(comparison_module, "evaluate_multitask_examples", return_value=metrics),
                patch.object(comparison_module, "train_multitask_examples", return_value={"epochs": 1}),
            ):
                result = run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=3,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    checkpoint_path=path,
                    resume=True,
                    variants=CONTROLLED_VARIANTS,
                    execution_config=ExecutionConfig(),
                )

        self.assertEqual(result["runs"]["A"][0]["status"], "completed")
        self.assertEqual(result["runs"]["A"][0]["telemetry"], old_telemetry)
        self.assertEqual(
            [a["attempt_id"] for a in result["runs"]["A"][0]["telemetry"]["attempts"]],
            ["attempt-1"],
        )


if __name__ == "__main__":
    unittest.main()
