import unittest
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kinera.language.vocabulary import CharacterVocabulary
from kinera.experiments import comparison
from kinera.experiments.config import ExecutionConfig
from kinera.experiments.comparison import (
    BASELINE_VARIANTS,
    CONTROLLED_VARIANTS,
    DEFAULT_VARIANTS,
    OPTIONAL_ABLATION_VARIANTS,
    PRIMARY_VARIANTS,
    SUPPORTED_VARIANTS,
    count_trainable_parameters,
    prepare_variant_examples,
    run_controlled_experiments,
)
from kinera.training.autoregressive import AutoregressiveModel
from scripts.compare import build_parser, main as compare_main


class ControlledComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        self.examples = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 4],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 4,
                "target_is_syllable_end": 1,
            }
        ]

    def _run_historical(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("variants", CONTROLLED_VARIANTS)
        return run_controlled_experiments(*args, **kwargs)

    def test_compare_cli_backend_defaults_to_cpu_and_supports_explicit_cuda(self) -> None:
        arguments = [
            "--vocabulary", "vocabulary.json",
            "--syllable-vocabulary", "syllables.json",
            "--train", "train.jsonl",
            "--validation", "validation.jsonl",
        ]
        cpu_args = build_parser().parse_args(arguments)
        cuda_args = build_parser().parse_args([*arguments, "--backend", "cuda"])

        self.assertEqual(cpu_args.backend, "cpu")
        self.assertEqual(cuda_args.backend, "cuda")

    def test_compare_cli_passes_selected_execution_config_to_runner(self) -> None:
        class FakeSyllableVocabulary:
            unk_id = 0

            def __len__(self) -> int:
                return 5

        arguments = [
            "--vocabulary", "vocabulary.json",
            "--syllable-vocabulary", "syllables.json",
            "--train", "train.jsonl",
            "--validation", "validation.jsonl",
            "--seeds", "7",
            "--epochs", "1",
            "--variants", "D",
            "--max-examples", "1",
            "--backend", "cuda",
            "--workers", "1",
        ]
        with (
            patch.object(Path, "read_text", return_value="{}"),
            patch(
                "scripts.compare.CharacterVocabulary.from_dict",
                return_value=self.vocabulary,
            ),
            patch(
                "scripts.compare.SyllableVocabulary.from_dict",
                return_value=FakeSyllableVocabulary(),
            ),
            patch("scripts.compare.load_jsonl", return_value=self.examples),
            patch(
                "scripts.compare.run_controlled_experiments",
                return_value={"status": "completed"},
            ) as run_experiments,
        ):
            compare_main(arguments)

        self.assertEqual(
            run_experiments.call_args.kwargs["execution_config"],
            ExecutionConfig(backend="cuda", workers=1),
        )
        self.assertNotIn("workers", run_experiments.call_args.kwargs)

    def test_compare_cli_rejects_cuda_workers_before_file_io(self) -> None:
        arguments = [
            "--vocabulary", "vocabulary.json",
            "--syllable-vocabulary", "syllables.json",
            "--train", "train.jsonl",
            "--validation", "validation.jsonl",
            "--backend", "cuda",
            "--workers", "2",
        ]
        with patch.object(
            Path, "read_text", side_effect=AssertionError("input read before validation")
        ) as read_text:
            with self.assertRaises(SystemExit) as raised:
                compare_main(arguments)
            self.assertEqual(raised.exception.code, 2)
            read_text.assert_not_called()

    def test_variant_constants_separate_defaults_from_supported_controls(self) -> None:
        self.assertEqual(SUPPORTED_VARIANTS, ("A", "B", "C", "D"))
        self.assertEqual(CONTROLLED_VARIANTS, SUPPORTED_VARIANTS)
        self.assertEqual(DEFAULT_VARIANTS, ("A", "C", "D"))
        self.assertEqual(BASELINE_VARIANTS, ("A",))
        self.assertEqual(PRIMARY_VARIANTS, ("C", "D"))
        self.assertEqual(OPTIONAL_ABLATION_VARIANTS, ("B",))

    def test_runner_defaults_to_primary_variants(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
        )

        self.assertEqual(report["variants"], list(DEFAULT_VARIANTS))
        self.assertEqual(list(report["runs"]), list(DEFAULT_VARIANTS))
        self.assertEqual(report["completed_runs"], len(DEFAULT_VARIANTS))

    def test_runner_prepares_and_executes_only_requested_variant(self) -> None:
        with (
            patch.object(
                comparison,
                "prepare_variant_examples",
                wraps=comparison.prepare_variant_examples,
            ) as prepare,
            patch.object(
                comparison,
                "train_multitask_examples",
                wraps=comparison.train_multitask_examples,
            ) as train,
        ):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                variants=("D",),
            )

        self.assertEqual([call.args[1] for call in prepare.call_args_list], ["D", "D"])
        self.assertEqual(train.call_count, 1)
        self.assertEqual(report["variants"], ["D"])
        self.assertEqual(list(report["runs"]), ["D"])
        self.assertEqual(list(report["aggregates"]), ["D"])
        self.assertEqual(list(report["parameter_counts"]), ["D"])
        self.assertEqual(list(report["input_usage"]), ["D"])
        self.assertEqual(report["completed_runs"], 1)
        self.assertEqual(report["total_runs"], 1)

    def test_runner_rejects_empty_variant_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "variants não pode ser vazio"):
            run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                variants=(),
            )

    def test_runner_rejects_unknown_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "variante desconhecida: Z"):
            run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                variants=("Z",),
            )

    def test_runner_rejects_duplicate_variants(self) -> None:
        with self.assertRaisesRegex(ValueError, "variants não pode conter duplicatas"):
            run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                variants=("A", "A"),
            )

    def test_checkpoint_records_selection_and_rejects_incompatible_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.json"
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
                variants=("D",),
            )
            persisted = json.loads(checkpoint.read_text(encoding="utf-8"))

            self.assertEqual(report["variants"], ["D"])
            self.assertEqual(persisted["variants"], ["D"])
            self.assertEqual(list(persisted["runs"]), ["D"])
            with self.assertRaisesRegex(ValueError, "checkpoint incompatível"):
                run_controlled_experiments(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=5,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    checkpoint_path=checkpoint,
                    resume=True,
                    variants=("C", "D"),
                )

    def test_parallel_runner_respects_selected_variants(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
            workers=2,
            variants=("D",),
        )

        self.assertEqual(report["variants"], ["D"])
        self.assertEqual(list(report["runs"]), ["D"])
        self.assertEqual(report["completed_runs"], 1)
        self.assertEqual(report["total_runs"], 1)
        self.assertEqual(list(report["aggregates"]), ["D"])
        self.assertEqual(list(report["parameter_counts"]), ["D"])
        self.assertEqual(list(report["input_usage"]), ["D"])

    def test_all_variants_keep_context_shape_and_only_mask_channels(self) -> None:
        prepared = {
            name: prepare_variant_examples(
                self.examples,
                name,
                self.vocabulary,
                syllable_unknown_id=0,
            )[0]
            for name in CONTROLLED_VARIANTS
        }

        self.assertEqual(prepared["A"]["context_ids"], [0, 3])
        self.assertEqual(prepared["A"]["context_syllable_ids"], [0, 0])
        self.assertEqual(prepared["A"]["context_feature_vectors"], [[0.0] * 10, [0.0] * 10])
        self.assertEqual(prepared["B"]["context_syllable_ids"], [0, 0])
        self.assertEqual(prepared["C"]["context_syllable_ids"], [0, 4])
        self.assertEqual(prepared["C"]["context_feature_vectors"], [[0.0] * 10, [0.0] * 10])
        self.assertEqual(prepared["D"]["context_syllable_ids"], [0, 4])
        self.assertEqual(prepared["D"]["context_feature_vectors"][1][0], 1.0)

    def test_variants_have_equal_trainable_parameter_counts(self) -> None:
        counts = []
        for _ in CONTROLLED_VARIANTS:
            model = AutoregressiveModel(5, 5, seed=7)
            counts.append(count_trainable_parameters(model))

        self.assertEqual(len(set(counts)), 1)
        self.assertGreater(counts[0], 0)

    def test_parameter_count_uses_dimensions_not_iterable_weights(self) -> None:
        class NonIterable:
            def __iter__(self):
                raise AssertionError("parameter counting must not iterate weights")

        model = comparison.AutoregressiveModel(51, 3932, seed=7)
        parameter_owners = (
            (model.representation.char_embedding, "weights"),
            (model.representation.syllable_embedding, "weights"),
            (model.heads.char, "weights"),
            (model.heads.char, "bias"),
            (model.heads.syllable, "weights"),
            (model.heads.syllable, "bias"),
            (model.heads.boundary, "weights"),
            (model.heads.boundary, "bias"),
        )
        try:
            for owner, attribute in parameter_owners:
                setattr(owner, attribute, NonIterable())
            self.assertEqual(count_trainable_parameters(model), 139432)
        finally:
            model.close()

    def test_parallel_worker_closes_model_after_success_and_failure(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self) -> None:
                self.close_calls += 1

        metrics = {
            name: 0.0
            for name in (
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
        }
        for training_error in (None, RuntimeError("training failed")):
            with self.subTest(training_error=training_error):
                model = FakeModel()
                with (
                    patch.object(comparison, "AutoregressiveModel", return_value=model),
                    patch.object(comparison, "count_trainable_parameters", return_value=139432),
                    patch.object(comparison, "evaluate_multitask_examples", return_value=metrics),
                    patch.object(
                        comparison,
                        "train_multitask_examples",
                        side_effect=training_error,
                        return_value=[],
                    ),
                ):
                    _, _, record = comparison._execute_single_experiment(
                        "D",
                        7,
                        self.examples,
                        self.examples,
                        self.vocabulary,
                        5,
                        1,
                        0.05,
                        0,
                    )

                self.assertEqual(
                    record["status"], "failed" if training_error else "completed"
                )
                self.assertEqual(model.close_calls, 1)

    def test_runner_reports_execution_config_separately_with_cpu_default(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
            variants=("D",),
            execution_config=ExecutionConfig(),
        )

        self.assertEqual(
            report["execution_config"], {"backend": "cpu", "workers": 1}
        )
        self.assertEqual(report["experiment_config"]["variants"], ["D"])
        self.assertEqual(report["experiment_config"]["phase"], "controlled_comparison")
        self.assertNotIn("context_size", report["experiment_config"])
        self.assertNotIn("backend", report["experiment_config"])
        self.assertNotIn("workers", report["experiment_config"])
        self.assertEqual(report["workers"], 1)

    def test_runner_passes_explicit_cuda_backend_to_model_and_closes_model(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        fake_backend = object()
        fake_model = FakeModel()
        metrics = {
            name: 0.0
            for name in (
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
        }
        with (
            patch.object(
                comparison, "get_backend", create=True, return_value=fake_backend
            ) as backend_factory,
            patch.object(
                comparison, "AutoregressiveModel", return_value=fake_model
            ) as model_factory,
            patch.object(comparison, "count_trainable_parameters", return_value=139432),
            patch.object(comparison, "evaluate_multitask_examples", return_value=metrics),
            patch.object(comparison, "train_multitask_examples", return_value=[]),
        ):
            report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                variants=("D",),
                execution_config=ExecutionConfig(backend="cuda"),
            )

        backend_factory.assert_called_once_with("cuda")
        model_factory.assert_called_once_with(
            5, 5, seed=7, backend=fake_backend
        )
        self.assertTrue(fake_model.closed)
        self.assertEqual(
            report["execution_config"], {"backend": "cuda", "workers": 1}
        )

    def test_resume_rejects_backend_mismatch_before_backend_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runner.json"
            cpu_report = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
                variants=("D",),
                execution_config=ExecutionConfig(),
            )
            self.assertEqual(
                cpu_report["experiment_config"]["variants"], ["D"]
            )
            self.assertNotIn("backend", cpu_report["experiment_config"])
            self.assertNotIn("workers", cpu_report["experiment_config"])
            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8"))["execution_config"],
                {"backend": "cpu", "workers": 1},
            )

            for checkpoint_config, resume_config in (
                (ExecutionConfig(backend="cpu"), ExecutionConfig(backend="cuda")),
                (ExecutionConfig(backend="cuda"), ExecutionConfig(backend="cpu")),
            ):
                persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
                persisted["execution_config"] = checkpoint_config.to_dict()
                checkpoint.write_text(
                    json.dumps(persisted), encoding="utf-8"
                )
                with patch.object(
                    comparison,
                    "_create_execution_backend",
                    side_effect=AssertionError("backend created before resume validation"),
                ) as backend_factory:
                    with self.assertRaisesRegex(ValueError, "backend"):
                        run_controlled_experiments(
                            self.examples,
                            self.examples,
                            self.vocabulary,
                            syllable_vocab_size=5,
                            seeds=[7],
                            epochs=1,
                            learning_rate=0.05,
                            max_examples=1,
                            checkpoint_path=checkpoint,
                            resume=True,
                            variants=("D",),
                            execution_config=resume_config,
                        )
                    backend_factory.assert_not_called()

    def test_runner_reports_each_variant_and_seed(self) -> None:
        report = self._run_historical(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
        )

        self.assertEqual(set(report["runs"]), set(CONTROLLED_VARIANTS))
        self.assertEqual(len(report["runs"]["A"]), 1)
        self.assertIn("mean", report["aggregates"]["A"])
        self.assertIn("char_accuracy", report["aggregates"]["A"]["mean"])
        self.assertEqual(report["parameter_counts"]["A"], report["parameter_counts"]["D"])
        self.assertIn("duration_seconds", report["runs"]["A"][0])
        self.assertEqual(report["input_usage"]["A"]["train_non_unk_context_positions"], 0)
        self.assertGreater(report["input_usage"]["D"]["train_non_unk_context_positions"], 0)

    def test_runner_persists_each_run_and_resumes_completed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase5.json"
            first = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["completed_runs"], 4)
            self.assertTrue(checkpoint.exists())
            persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(persisted["completed_runs"], 4)

            resumed = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7, 8],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["completed_runs"], 8)
            for variant in CONTROLLED_VARIANTS:
                self.assertEqual([run["seed"] for run in resumed["runs"][variant]], [7, 8])

    def test_runner_records_failed_run_without_discarding_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase5.json"
            with patch(
                "kinera.experiments.comparison.train_multitask_examples",
                side_effect=RuntimeError("synthetic training failure"),
            ):
                report = self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=5,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    checkpoint_path=checkpoint,
                )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_runs"], 0)
            self.assertEqual(report["failed_runs"], 4)
            self.assertEqual(report["runs"]["A"][0]["status"], "failed")
            self.assertIn("synthetic training failure", report["runs"]["A"][0]["traceback"])
            self.assertTrue(checkpoint.exists())

    def test_parallel_runner_records_worker_pid_and_completed_status(self) -> None:
        report = self._run_historical(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
            workers=2,
        )

        self.assertEqual(report["workers"], 2)
        self.assertEqual(report["completed_runs"], 4)
        for variant in CONTROLLED_VARIANTS:
            run = report["runs"][variant][0]
            self.assertEqual(run["status"], "completed")
            self.assertIsInstance(run["pid"], int)
            self.assertGreater(run["pid"], 0)
            self.assertIsInstance(run["max_rss_kb"], int)
            self.assertGreater(run["max_rss_kb"], 0)

    def test_workers_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                workers=0,
            )

    def test_parallel_and_sequential_results_match(self) -> None:
        common = {
            "train_examples": self.examples,
            "validation_examples": self.examples,
            "vocabulary": self.vocabulary,
            "syllable_vocab_size": 5,
            "seeds": [7],
            "epochs": 1,
            "learning_rate": 0.05,
            "max_examples": 1,
        }
        sequential = self._run_historical(**common, workers=1)
        parallel = self._run_historical(**common, workers=2)
        for variant in CONTROLLED_VARIANTS:
            self.assertEqual(
                sequential["aggregates"][variant]["mean"],
                parallel["aggregates"][variant]["mean"],
            )

    def test_parallel_worker_failure_does_not_discard_other_jobs(self) -> None:
        class FakeFuture:
            def __init__(self, result=None, error=None):
                self._result = result
                self._error = error

            def result(self):
                if self._error is not None:
                    raise self._error
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, function, variant, seed, *args):
                if seed == 7:
                    return FakeFuture(error=RuntimeError("worker seven failed"))
                return FakeFuture(result=function(variant, seed, *args))

        with patch.object(comparison.concurrent.futures, "ProcessPoolExecutor", FakeExecutor), patch.object(
            comparison.concurrent.futures, "as_completed", side_effect=lambda futures: futures
        ):
            report = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                seeds=[7, 8],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                workers=2,
            )

        self.assertEqual(report["failed_runs"], 4)
        self.assertEqual(report["completed_runs"], 4)
        for variant in CONTROLLED_VARIANTS:
            statuses = [run["status"] for run in report["runs"][variant]]
            self.assertEqual(statuses, ["failed", "completed"])

    def test_parallel_checkpoint_exposes_pending_before_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase5.json"
            snapshots = []

            def capture(path, report):
                snapshots.append(json.loads(json.dumps(report)))

            with patch("kinera.experiments.comparison._persist_checkpoint", side_effect=capture):
                self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    syllable_vocab_size=5,
                    seeds=[7],
                    epochs=1,
                    learning_rate=0.05,
                    max_examples=1,
                    checkpoint_path=checkpoint,
                    workers=2,
                )

            pending_snapshots = [
                snapshot
                for snapshot in snapshots
                if any(
                    run.get("status") == "pending"
                    for runs in snapshot["runs"].values()
                    for run in runs
                )
            ]
            self.assertTrue(pending_snapshots)

    def test_aggregates_include_conservation_audit_and_paired_deltas(self) -> None:
        report = self._run_historical(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7, 42],
            epochs=1,
            learning_rate=0.05,
            max_examples=2,
            variants=("A", "C"),
        )
        # 1. Compatibilidade legada de aggregates
        self.assertIn("A", report["aggregates"])
        self.assertIn("C", report["aggregates"])
        for var in ("A", "C"):
            agg = report["aggregates"][var]
            self.assertIn("mean", agg)
            self.assertIn("std", agg)
            self.assertIn("n_values", agg)
            self.assertIn("n_missing", agg)
            self.assertIn("n_invalid", agg)
            self.assertIn("seeds", agg)

            # Invariante de conservação para cada métrica:
            for metric in agg["mean"]:
                n_v = agg["n_values"][metric]
                n_m = agg["n_missing"][metric]
                n_i = agg["n_invalid"][metric]
                self.assertEqual(n_v + n_m + n_i, 2)
                self.assertEqual(n_v, len(agg["seeds"][metric]))

        # 2. Presença e invariantes de paired_deltas
        self.assertIn("paired_deltas", report)
        deltas = report["paired_deltas"]
        self.assertTrue(len(deltas) >= 1)
        pair_ac = next(p for p in deltas if p["base"] == "A" and p["candidate"] == "C")
        self.assertEqual(pair_ac["n_common_seeds"], 2)
        self.assertEqual(pair_ac["common_seeds"], [7, 42])

        # Testar invariante por métrica pareada
        for m_name, m_delta in pair_ac["metrics"].items():
            n_p = m_delta["n_pairs"]
            n_mp = m_delta["n_missing_pairs"]
            n_ip = m_delta["n_invalid_pairs"]
            self.assertEqual(n_p + n_mp + n_ip, pair_ac["n_common_seeds"])
            self.assertEqual(n_p, len(m_delta["seeds_used"]))


if __name__ == "__main__":
    unittest.main()
