import unittest
import json
import random
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
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
        self.syllable_vocabulary = self._make_syllable_vocabulary(5)
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
        size = kwargs.get("syllable_vocab_size")
        if size is not None:
            kwargs.setdefault("syllable_vocabulary", self._make_syllable_vocabulary(size))
        return run_controlled_experiments(*args, **kwargs)

    @staticmethod
    def _make_syllable_vocabulary(size: int) -> SyllableVocabulary:
        return SyllableVocabulary([
            "<UNK_SYL>",
            *(f"<SYNTHETIC_{index}>" for index in range(1, size)),
        ])

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

    def test_compare_cli_exposes_optional_validation_cadence(self) -> None:
        arguments = [
            "--vocabulary", "vocabulary.json",
            "--syllable-vocabulary", "syllables.json",
            "--train", "train.jsonl",
            "--validation", "validation.jsonl",
        ]
        default_args = build_parser().parse_args(arguments)
        enabled_args = build_parser().parse_args(
            [*arguments, "--validation-every-n-epochs", "3"]
        )

        self.assertIsNone(default_args.validation_every_n_epochs)
        self.assertEqual(enabled_args.validation_every_n_epochs, 3)

    def test_validation_cadence_preserves_legacy_results_and_reuses_final(self) -> None:
        common = {
            "syllable_vocab_size": 5,
            "syllable_vocabulary": self.syllable_vocabulary,
            "seeds": [7],
            "epochs": 3,
            "learning_rate": 0.05,
            "max_examples": 1,
            "variants": ("A",),
        }
        rng_before = random.getstate()
        baseline = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            **common,
        )
        self.assertEqual(random.getstate(), rng_before)

        original_evaluate = comparison.evaluate_multitask_examples
        with patch.object(
            comparison,
            "evaluate_multitask_examples",
            wraps=original_evaluate,
        ) as evaluate:
            observed = run_controlled_experiments(
                self.examples,
                self.examples,
                self.vocabulary,
                validation_every_n_epochs=1,
                **common,
            )

        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(
            evaluate.call_count,
            4,  # initial + epochs 1/2 + final; epoch 3 is reused
        )
        baseline_run = baseline["runs"]["A"][0]
        observed_run = observed["runs"]["A"][0]
        self.assertEqual(observed_run["training"], baseline_run["training"])
        self.assertEqual(
            observed_run["initial_validation"], baseline_run["initial_validation"]
        )
        self.assertEqual(observed_run["final_validation"], baseline_run["final_validation"])
        for section in ("aggregates", "paired_deltas", "metrics"):
            self.assertEqual(observed[section], baseline[section])
        self.assertEqual(
            [item["total_loss"] for item in baseline_run["training"]["history"]],
            [2.9987254639778302, 2.931094460057473, 2.8646954739655675],
        )
        self.assertEqual(
            [item["char_accuracy"] for item in baseline_run["training"]["history"]],
            [0.0, 1.0, 1.0],
        )
        self.assertEqual(
            baseline_run["initial_validation"]["total_loss"], 3.067645245925604
        )
        self.assertEqual(
            baseline_run["final_validation"]["total_loss"], 2.8646954739655675
        )
        self.assertNotIn("validation_history", baseline_run)
        self.assertNotIn("observation_config", baseline)
        self.assertEqual(
            [item["epoch"] for item in observed_run["validation_history"]], [1, 2, 3]
        )
        self.assertEqual(
            {
                key: value
                for key, value in observed_run["validation_history"][-1].items()
                if key not in {"epoch", "split"}
            },
            observed_run["final_validation"],
        )
        self.assertEqual(
            observed_run["validation_history"][-1]["metrics_schema_version"], 1
        )
        self.assertEqual(
            observed_run["validation_history"][-1]["metric_results"][
                "char_accuracy"
            ]["support"],
            {"examples": 1},
        )
        self.assertEqual(
            baseline["experiment_fingerprint"], observed["experiment_fingerprint"]
        )
        reloaded = json.loads(json.dumps(observed, allow_nan=False))
        self.assertEqual(
            reloaded["runs"]["A"][0]["validation_history"],
            observed_run["validation_history"],
        )

    def test_validation_cadence_two_and_greater_than_epochs(self) -> None:
        common = {
            "syllable_vocab_size": 5,
            "seeds": [7],
            "learning_rate": 0.05,
            "max_examples": 1,
            "variants": ("A",),
        }
        every_two = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            epochs=5,
            validation_every_n_epochs=2,
            **common,
        )
        self.assertEqual(
            [item["epoch"] for item in every_two["runs"]["A"][0]["validation_history"]],
            [2, 4],
        )

        beyond_run = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            epochs=2,
            validation_every_n_epochs=3,
            **common,
        )["runs"]["A"][0]
        self.assertEqual(beyond_run["validation_history"], [])

    def test_validation_cadence_rejects_invalid_values(self) -> None:
        for value in (True, False, 0, -1, "2", 2.0):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    run_controlled_experiments(
                        self.examples,
                        self.examples,
                        self.vocabulary,
                        syllable_vocab_size=5,
                        seeds=[7],
                        epochs=1,
                        learning_rate=0.05,
                        variants=("A",),
                        validation_every_n_epochs=value,  # type: ignore[arg-type]
                    )

    def test_parallel_cpu_worker_receives_validation_cadence(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=2,
            learning_rate=0.05,
            max_examples=1,
            variants=("A",),
            execution_config=ExecutionConfig(workers=2),
            validation_every_n_epochs=1,
        )

        run = report["runs"]["A"][0]
        self.assertEqual(run["status"], "completed")
        self.assertEqual([item["epoch"] for item in run["validation_history"]], [1, 2])

    def test_validation_cadence_resume_is_strict_and_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "validation-cadence.json"
            common = {
                "syllable_vocab_size": 5,
                "seeds": [7],
                "epochs": 2,
                "learning_rate": 0.05,
                "max_examples": 1,
                "variants": ("A",),
                "checkpoint_path": checkpoint,
            }
            first = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                validation_every_n_epochs=1,
                **common,
            )
            prior_history = first["runs"]["A"][0]["validation_history"]
            raw_before_mismatch = checkpoint.read_bytes()
            with patch.object(
                comparison,
                "_execute_single_experiment",
                side_effect=AssertionError(
                    "execution began before cadence validation"
                ),
            ) as execute:
                with self.assertRaisesRegex(ValueError, "validation_every_n_epochs"):
                    self._run_historical(
                        self.examples,
                        self.examples,
                        self.vocabulary,
                        validation_every_n_epochs=2,
                        resume=True,
                        **common,
                    )
                execute.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), raw_before_mismatch)
            with self.assertRaisesRegex(ValueError, "validation_every_n_epochs"):
                self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    resume=True,
                    **common,
                )
            self.assertEqual(checkpoint.read_bytes(), raw_before_mismatch)

            malformed = json.loads(raw_before_mismatch)
            malformed["observation_config"]["validation_every_n_epochs"] = True
            checkpoint.write_text(json.dumps(malformed), encoding="utf-8")
            malformed_bytes = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "validation_every_n_epochs"):
                self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    validation_every_n_epochs=1,
                    resume=True,
                    **common,
                )
            self.assertEqual(checkpoint.read_bytes(), malformed_bytes)
            checkpoint.write_bytes(raw_before_mismatch)

            expanded = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                validation_every_n_epochs=1,
                seeds=[7, 8],
                epochs=2,
                learning_rate=0.05,
                max_examples=1,
                variants=("A", "D"),
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(
                expanded["runs"]["A"][0]["validation_history"], prior_history
            )
            self.assertEqual(expanded["variants"], ["A", "D"])
            self.assertEqual(
                [run["seed"] for run in expanded["runs"]["D"]], [7, 8]
            )
            self.assertTrue(
                all(
                    [item["epoch"] for item in run["validation_history"]] == [1, 2]
                    for run in expanded["runs"]["D"]
                )
            )
            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8"))["observation_config"],
                {"validation_every_n_epochs": 1},
            )

    def test_checkpoint_without_validation_history_cannot_resume_with_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-observation.json"
            common = {
                "syllable_vocab_size": 5,
                "seeds": [7],
                "epochs": 1,
                "learning_rate": 0.05,
                "max_examples": 1,
                "variants": ("A",),
            }
            self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                checkpoint_path=checkpoint,
                **common,
            )
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "validation_every_n_epochs"):
                self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    checkpoint_path=checkpoint,
                    resume=True,
                    validation_every_n_epochs=1,
                    **common,
                )
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_failed_partial_validation_attempt_does_not_mix_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "partial-validation.json"
            common = {
                "syllable_vocab_size": 5,
                "seeds": [7],
                "epochs": 3,
                "learning_rate": 0.05,
                "max_examples": 1,
                "variants": ("A",),
                "checkpoint_path": checkpoint,
                "validation_every_n_epochs": 1,
            }
            original_evaluate = comparison.evaluate_multitask_examples
            evaluation_calls = 0

            def fail_at_second_intermediate(
                *args: Any,
                **kwargs: Any,
            ) -> dict[str, Any]:
                nonlocal evaluation_calls
                evaluation_calls += 1
                if evaluation_calls == 3:  # initial, epoch 1, then epoch 2
                    raise RuntimeError("synthetic validation failure")
                return original_evaluate(*args, **kwargs)

            with patch.object(
                comparison,
                "evaluate_multitask_examples",
                side_effect=fail_at_second_intermediate,
            ):
                failed = self._run_historical(
                    self.examples,
                    self.examples,
                    self.vocabulary,
                    **common,
                )

            failed_run = failed["runs"]["A"][0]
            self.assertEqual(failed_run["status"], "failed")
            self.assertNotIn("validation_history", failed_run)
            resumed = self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                resume=True,
                **common,
            )
            completed_run = resumed["runs"]["A"][0]
            self.assertEqual(completed_run["status"], "completed")
            self.assertEqual(
                [item["epoch"] for item in completed_run["validation_history"]],
                [1, 2, 3],
            )

    def test_compare_cli_passes_selected_execution_config_to_runner(self) -> None:
        syllable_vocabulary = self._make_syllable_vocabulary(5)

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
            "--validation-every-n-epochs", "2",
        ]
        with (
            patch.object(Path, "read_text", return_value="{}"),
            patch(
                "scripts.compare.CharacterVocabulary.from_dict",
                return_value=self.vocabulary,
            ),
            patch(
                "scripts.compare.SyllableVocabulary.from_dict",
                return_value=syllable_vocabulary,
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
        self.assertIs(
            run_experiments.call_args.kwargs["syllable_vocabulary"],
            syllable_vocabulary,
        )
        self.assertEqual(
            run_experiments.call_args.kwargs["validation_every_n_epochs"], 2
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
            report = self._run_historical(
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

    def test_checkpoint_records_selection_and_allows_variant_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "selected.json"
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
                variants=("D",),
            )
            persisted = json.loads(checkpoint.read_text(encoding="utf-8"))

            self.assertEqual(report["variants"], ["D"])
            self.assertEqual(persisted["variants"], ["D"])
            self.assertEqual(list(persisted["runs"]), ["D"])
            fingerprint = persisted["experiment_fingerprint"]
            self.assertEqual(fingerprint["fingerprint_schema_version"], 1)
            self.assertEqual(fingerprint["algorithm"], "sha256")
            self.assertEqual(len(fingerprint["digest"]), 64)
            self.assertNotIn("training_examples", fingerprint)
            self.assertEqual(set(fingerprint["variant_inputs"]), {"D"})
            expanded = self._run_historical(
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
            self.assertEqual(expanded["variants"], ["C", "D"])
            self.assertEqual(expanded["completed_runs"], 2)
            self.assertEqual([run["seed"] for run in expanded["runs"]["D"]], [7])

    def test_checkpoint_requires_complete_syllable_vocabulary_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "requires-vocabulary.json"
            with self.assertRaisesRegex(ValueError, "syllable_vocabulary"):
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
                    variants=("D",),
                )
            self.assertFalse(checkpoint.exists())

    def test_checkpoint_rejects_changed_data_before_execution_and_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "identity.json"
            self._run_historical(
                self.examples,
                self.examples,
                self.vocabulary,
                syllable_vocab_size=5,
                syllable_vocabulary=self.syllable_vocabulary,
                seeds=[7],
                epochs=1,
                learning_rate=0.05,
                max_examples=1,
                checkpoint_path=checkpoint,
                variants=("D",),
            )
            before = checkpoint.read_bytes()
            changed = [dict(self.examples[0], target_char_id=3)]
            with patch.object(
                comparison,
                "_execute_single_experiment",
                side_effect=AssertionError("execution began before identity validation"),
            ) as execute:
                with self.assertRaisesRegex(ValueError, "training_examples_sha256"):
                    self._run_historical(
                        changed,
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
                    )
                execute.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_checkpoint_expands_seeds_and_variants_without_retraining_old_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "expand.json"
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
                variants=("D",),
            )
            executed: list[tuple[str, int]] = []

            def fake_run(variant, seed, *args, **kwargs):
                executed.append((variant, seed))
                return variant, seed, {
                    "seed": seed,
                    "status": "completed",
                    "parameter_count": 1,
                    "final_validation": {},
                }

            with patch.object(comparison, "_execute_single_experiment", side_effect=fake_run):
                expanded = self._run_historical(
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
                    variants=("C", "D"),
                )

            self.assertEqual(executed, [("C", 7), ("C", 8), ("D", 8)])
            self.assertEqual(expanded["completed_runs"], 4)
            self.assertEqual([run["seed"] for run in expanded["runs"]["D"]], [7, 8])

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
                syllable_vocabulary=self.syllable_vocabulary,
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
                            syllable_vocabulary=self.syllable_vocabulary,
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
            self.assertEqual(first["metrics_schema_version"], 1)
            self.assertEqual(
                first["metrics"]["metrics_schema_version"],
                first["metrics_schema_version"],
            )
            self.assertEqual(first["metrics"]["aggregates"], first["aggregates"])
            self.assertEqual(
                first["metrics"]["paired_deltas"], first["paired_deltas"]
            )
            self.assertTrue(checkpoint.exists())
            persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(persisted["completed_runs"], 4)
            self.assertEqual(persisted["metrics_schema_version"], 1)
            self.assertEqual(persisted["metrics"], first["metrics"])
            self.assertEqual(
                persisted["runs"]["A"][0]["final_validation"]["metrics_schema_version"],
                1,
            )

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
            self.assertEqual(resumed["metrics_schema_version"], 1)
            self.assertEqual(
                resumed["metrics"]["aggregates"], resumed["aggregates"]
            )
            self.assertEqual(
                resumed["metrics"]["paired_deltas"], resumed["paired_deltas"]
            )
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

    def test_paired_deltas_use_completed_common_seeds_and_correct_units(self) -> None:
        reports = {
            "A": [
                {"seed": 7, "status": "completed", "final_validation": {
                    "char_loss": 2.0,
                    "boundary_baseline_majority": 0.5,
                    "boundary_true_negative": 3,
                }},
                {"seed": 1, "status": "completed"},
                {"seed": 42, "status": "failed", "final_validation": {
                    "char_loss": 9.0,
                }},
            ],
            "C": [
                {"seed": 7, "status": "completed", "final_validation": {
                    "char_loss": 1.0,
                    "boundary_baseline_majority": 0.75,
                    "boundary_true_negative": 5,
                }},
                {"seed": 1, "status": "completed", "final_validation": {}},
                {"seed": 42, "status": "completed", "final_validation": {
                    "char_loss": 1.0,
                }},
            ],
        }

        paired = comparison._build_paired_deltas(
            reports,
            ("char_loss", "boundary_baseline_majority", "boundary_true_negative"),
            ("A", "C"),
        )[0]

        self.assertEqual(paired["n_common_seeds"], 2)
        self.assertEqual(paired["common_seeds"], [1, 7])
        loss = paired["metrics"]["char_loss"]
        self.assertEqual(loss["orientation"], "base - candidate")
        self.assertEqual(loss["unit"], "loss")
        self.assertEqual(loss["n_pairs"], 1)
        self.assertEqual(loss["n_missing_pairs"], 1)
        self.assertEqual(loss["mean"], 1.0)

        baseline = paired["metrics"]["boundary_baseline_majority"]
        self.assertEqual(baseline["orientation"], "candidate - base")
        self.assertEqual(baseline["unit"], "fraction")
        self.assertAlmostEqual(baseline["mean"], 0.25)
        true_negative = paired["metrics"]["boundary_true_negative"]
        self.assertEqual(true_negative["unit"], "count")
        self.assertEqual(true_negative["mean"], 2.0)

    def test_versioned_not_applicable_value_stays_missing_in_aggregate_and_pair(self) -> None:
        missing_descriptor = {
            "value": None,
            "availability": "not_applicable",
            "reason": "no_gold_or_predicted_positive",
            "support": {"gold_positive": 0, "predicted_positive": 0},
        }
        available_descriptor = {
            "value": 0.25,
            "availability": "available",
            "reason": None,
            "support": {"gold_positive": 2, "predicted_positive": 1},
        }
        reports = {
            "A": [{
                "seed": 7,
                "status": "completed",
                "final_validation": {
                    "boundary_f1": 0.0,
                    "metrics_schema_version": 1,
                    "metric_results": {"boundary_f1": missing_descriptor},
                },
            }],
            "C": [{
                "seed": 7,
                "status": "completed",
                "final_validation": {
                    "boundary_f1": 0.25,
                    "metrics_schema_version": 1,
                    "metric_results": {"boundary_f1": available_descriptor},
                },
            }],
        }

        aggregate = comparison._build_aggregates(
            reports, ("boundary_f1",), ("A",)
        )["A"]
        paired = comparison._build_paired_deltas(
            reports, ("boundary_f1",), ("A", "C")
        )[0]["metrics"]["boundary_f1"]

        self.assertEqual(aggregate["n_values"]["boundary_f1"], 0)
        self.assertEqual(aggregate["n_missing"]["boundary_f1"], 1)
        self.assertIsNone(aggregate["mean"]["boundary_f1"])
        self.assertEqual(paired["n_pairs"], 0)
        self.assertEqual(paired["n_missing_pairs"], 1)
        self.assertIsNone(paired["mean"])

    def test_majority_baseline_distribution_comes_only_from_training_examples(self) -> None:
        class ModelStub:
            def close(self) -> None:
                pass

        train_examples = [
            {"target_is_syllable_end": 1},
            {"target_is_syllable_end": 0},
            {"target_is_syllable_end": 1},
        ]
        validation_examples = [
            {"target_is_syllable_end": 0},
            {"target_is_syllable_end": 0},
            {"target_is_syllable_end": 0},
        ]

        with (
            patch.object(comparison, "AutoregressiveModel", return_value=ModelStub()),
            patch.object(comparison, "count_trainable_parameters", return_value=12),
            patch.object(comparison, "train_multitask_examples", return_value={"epochs": 1}),
            patch.object(
                comparison,
                "evaluate_multitask_examples",
                return_value={"char_loss": 0.0},
            ) as evaluate,
        ):
            _, _, record = comparison._execute_single_experiment(
                "A",
                7,
                train_examples,
                validation_examples,
                self.vocabulary,
                syllable_vocab_size=5,
                epochs=1,
                learning_rate=0.05,
                syllable_unknown_id=0,
            )

        self.assertEqual(record["status"], "completed")
        self.assertEqual(evaluate.call_count, 2)
        for call in evaluate.call_args_list:
            self.assertEqual(
                call.kwargs["train_boundary_distribution"],
                {"0": 1, "1": 2},
            )


if __name__ == "__main__":
    unittest.main()
