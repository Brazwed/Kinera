import unittest

from training.experiments import (
    MultiSeedSummary,
    format_multi_seed_report,
    run_multi_seed,
)
from training.data import generate_samples, split_samples
from training.loop import TrainingConfig


class MultiSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_samples = [
            ([1.0, 1.0], 6.0),
            ([1.0, 2.0], 9.0),
            ([2.0, 1.0], 8.0),
            ([2.0, 2.0], 11.0),
        ]
        self.validation_samples = [([0.5, 0.5], 3.5), ([3.0, 4.0], 19.0)]
        self.config = TrainingConfig(
            learning_rate=0.01,
            max_epochs=20,
            patience=5,
            min_delta=1e-10,
            shuffle=True,
            seed=42,
        )

    def test_run_returns_one_result_per_seed(self) -> None:
        summary = run_multi_seed(
            seeds=[1, 7, 42],
            train_samples=self.train_samples,
            validation_samples=self.validation_samples,
            config=self.config,
            convergence_threshold=1e-3,
        )

        self.assertIsInstance(summary, MultiSeedSummary)
        self.assertEqual([result.seed for result in summary.results], [1, 7, 42])
        self.assertEqual(len(summary.results), 3)
        self.assertEqual(
            summary.converged_count + summary.not_converged_count + summary.error_count,
            3,
        )

    def test_same_experiment_is_reproducible(self) -> None:
        first = run_multi_seed(
            seeds=[1, 7, 42],
            train_samples=self.train_samples,
            validation_samples=self.validation_samples,
            config=self.config,
            convergence_threshold=1e-3,
        )
        second = run_multi_seed(
            seeds=[1, 7, 42],
            train_samples=self.train_samples,
            validation_samples=self.validation_samples,
            config=self.config,
            convergence_threshold=1e-3,
        )

        self.assertEqual(first, second)

    def test_non_convergence_is_reported_without_dropping_results(self) -> None:
        summary = run_multi_seed(
            seeds=[1, 7, 42],
            train_samples=self.train_samples,
            validation_samples=self.validation_samples,
            config=TrainingConfig(
                learning_rate=0.01,
                max_epochs=1,
                patience=1,
                shuffle=False,
                seed=42,
            ),
            convergence_threshold=1e-30,
        )

        self.assertEqual(summary.converged_count, 0)
        self.assertEqual(summary.error_count, 0)
        self.assertEqual(summary.not_converged_count, 3)
        self.assertTrue(all(result.status == "not_converged" for result in summary.results))

    def test_report_contains_each_seed_and_summary_counts(self) -> None:
        summary = run_multi_seed(
            seeds=[1, 7],
            train_samples=self.train_samples,
            validation_samples=self.validation_samples,
            config=self.config,
            convergence_threshold=1e-3,
        )

        report = format_multi_seed_report(summary)

        self.assertIn("seed=1", report)
        self.assertIn("seed=7", report)
        self.assertIn("converged=", report)
        self.assertIn("not_converged=", report)
        self.assertIn("errors=", report)

    def test_invalid_seed_collection_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_multi_seed(
                seeds=[],
                train_samples=self.train_samples,
                validation_samples=self.validation_samples,
                config=self.config,
            )
        with self.assertRaises(ValueError):
            run_multi_seed(
                seeds=[1, 1],
                train_samples=self.train_samples,
                validation_samples=self.validation_samples,
                config=self.config,
            )

    def test_result_records_activation_and_initial_activity(self) -> None:
        samples = generate_samples(count=100, seed=7)
        train_samples, validation_samples = split_samples(
            samples, validation_ratio=0.2, seed=7
        )

        summary = run_multi_seed(
            seeds=[7],
            train_samples=train_samples,
            validation_samples=validation_samples,
            config=self.config,
            activation="relu",
        )
        result = summary.results[0]

        self.assertEqual(result.activation, "relu")
        self.assertEqual(result.negative_slope, 0.01)
        self.assertEqual(result.initial_active_count, 0)
        self.assertEqual(result.initial_sample_count, 80)
        self.assertEqual(result.initial_active_fraction, 0.0)

    def test_leaky_relu_updates_the_known_dead_seed(self) -> None:
        samples = generate_samples(count=100, seed=7)
        train_samples, validation_samples = split_samples(
            samples, validation_ratio=0.2, seed=7
        )
        config = TrainingConfig(
            learning_rate=0.01,
            max_epochs=1,
            patience=1,
            shuffle=False,
            seed=42,
        )

        summary = run_multi_seed(
            seeds=[7],
            train_samples=train_samples,
            validation_samples=validation_samples,
            config=config,
            activation="leaky_relu",
        )
        result = summary.results[0]

        self.assertEqual(result.activation, "leaky_relu")
        self.assertEqual(result.initial_active_count, 0)
        self.assertNotEqual(
            result.final_weights,
            [-0.03523344703336753, -0.06983016521509962],
        )


if __name__ == "__main__":
    unittest.main()
