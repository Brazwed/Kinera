import tempfile
import unittest
from pathlib import Path

from scripts.legacy.run_experiment import run_experiment
from kinera.training.checkpoint import load_neuron
from kinera.training.config import TrainingConfig


class CheckpointRunnerTests(unittest.TestCase):
    def test_runner_saves_selected_seed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints" / "seed-7.json"
            summary = run_experiment(
                seeds=[7],
                sample_count=20,
                validation_ratio=0.2,
                data_seed=7,
                config=TrainingConfig(
                    learning_rate=0.01,
                    max_epochs=5,
                    patience=5,
                    shuffle=False,
                    seed=42,
                ),
                activation="leaky_relu",
                checkpoint_path=path,
                checkpoint_seed=7,
            )

            loaded, info = load_neuron(path)

        self.assertEqual(summary.results[0].seed, 7)
        self.assertEqual(loaded.activation, "leaky_relu")
        self.assertEqual(info["metadata"]["seed"], 7)
        self.assertEqual(info["metadata"]["activation"], "leaky_relu")
        self.assertIsNotNone(loaded.forward([1.0, 2.0]))

    def test_checkpoint_seed_must_be_in_experiment_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_experiment(
                    seeds=[1],
                    sample_count=10,
                    validation_ratio=0.2,
                    data_seed=7,
                    checkpoint_path=Path(directory) / "model.json",
                    checkpoint_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
