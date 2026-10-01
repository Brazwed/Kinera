import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from language.vocabulary import CharacterVocabulary
from training.comparison import (
    VARIANTS,
    count_trainable_parameters,
    prepare_variant_examples,
    run_controlled_experiments,
)
from training.autoregressive import AutoregressiveModel


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

    def test_all_variants_keep_context_shape_and_only_mask_channels(self) -> None:
        prepared = {
            name: prepare_variant_examples(
                self.examples,
                name,
                self.vocabulary,
                syllable_unknown_id=0,
            )[0]
            for name in VARIANTS
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
        for _ in VARIANTS:
            model = AutoregressiveModel(5, 5, seed=7)
            counts.append(count_trainable_parameters(model))

        self.assertEqual(len(set(counts)), 1)
        self.assertGreater(counts[0], 0)

    def test_runner_reports_each_variant_and_seed(self) -> None:
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

        self.assertEqual(set(report["runs"]), set(VARIANTS))
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
            first = run_controlled_experiments(
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

            resumed = run_controlled_experiments(
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
            for variant in VARIANTS:
                self.assertEqual([run["seed"] for run in resumed["runs"][variant]], [7, 8])

    def test_runner_records_failed_run_without_discarding_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase5.json"
            with patch(
                "training.comparison.train_multitask_examples",
                side_effect=RuntimeError("synthetic training failure"),
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
                    checkpoint_path=checkpoint,
                )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_runs"], 0)
            self.assertEqual(report["failed_runs"], 4)
            self.assertEqual(report["runs"]["A"][0]["status"], "failed")
            self.assertIn("synthetic training failure", report["runs"]["A"][0]["traceback"])
            self.assertTrue(checkpoint.exists())

    def test_parallel_runner_records_worker_pid_and_completed_status(self) -> None:
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
        )

        self.assertEqual(report["workers"], 2)
        self.assertEqual(report["completed_runs"], 4)
        for variant in VARIANTS:
            run = report["runs"][variant][0]
            self.assertEqual(run["status"], "completed")
            self.assertIsInstance(run["pid"], int)
            self.assertGreater(run["pid"], 0)

    def test_workers_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            run_controlled_experiments(
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
        sequential = run_controlled_experiments(**common, workers=1)
        parallel = run_controlled_experiments(**common, workers=2)
        for variant in VARIANTS:
            self.assertEqual(
                sequential["aggregates"][variant]["mean"],
                parallel["aggregates"][variant]["mean"],
            )


if __name__ == "__main__":
    unittest.main()
