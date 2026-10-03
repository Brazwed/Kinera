import unittest

from scripts.legacy.run_experiment import DEFAULT_SEEDS, run_experiment
from kinera.training.config import TrainingConfig


class ExperimentRunnerTests(unittest.TestCase):
    def test_default_runner_uses_declared_seed_set_and_shared_data(self) -> None:
        summary = run_experiment(
            seeds=DEFAULT_SEEDS,
            sample_count=20,
            validation_ratio=0.2,
            data_seed=7,
            config=TrainingConfig(
                learning_rate=0.01,
                max_epochs=3,
                patience=3,
                shuffle=True,
                seed=42,
            ),
            convergence_threshold=1e-3,
        )

        self.assertEqual([result.seed for result in summary.results], list(DEFAULT_SEEDS))
        self.assertEqual(len(summary.results), len(DEFAULT_SEEDS))

    def test_runner_defaults_to_leaky_relu(self) -> None:
        summary = run_experiment(
            seeds=[7],
            sample_count=20,
            validation_ratio=0.2,
            data_seed=7,
            config=TrainingConfig(
                learning_rate=0.01,
                max_epochs=3,
                patience=3,
                shuffle=False,
                seed=42,
            ),
        )

        self.assertEqual(summary.results[0].activation, "leaky_relu")

    def test_runner_can_execute_relu_baseline(self) -> None:
        summary = run_experiment(
            seeds=[7],
            sample_count=20,
            validation_ratio=0.2,
            data_seed=7,
            activation="relu",
            config=TrainingConfig(
                learning_rate=0.01,
                max_epochs=1,
                patience=1,
                shuffle=False,
                seed=42,
            ),
        )

        self.assertEqual(summary.results[0].activation, "relu")


if __name__ == "__main__":
    unittest.main()
