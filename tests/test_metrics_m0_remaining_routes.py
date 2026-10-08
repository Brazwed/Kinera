import random
import unittest
from unittest.mock import patch

from kinera.experiments import runner as runner_module
from kinera.experiments.config import aggregate_metric_paths
from kinera.experiments.runner import (
    MultiSeedSummary,
    SeedExperimentResult,
    run_single_seed,
)
from kinera.monitor.metrics.core import mean_from_total, safe_ratio
from kinera.nn.layers.neuron import Neuron
from kinera.training.config import TrainingConfig
from kinera.training import evaluator as evaluator_module
from kinera.training import trainer as trainer_module


class M0RemainingRoutesTests(unittest.TestCase):
    @staticmethod
    def _neuron() -> Neuron:
        neuron = Neuron(input_size=1, seed=7)
        neuron.weights = [0.5]
        neuron.bias = 0.5
        return neuron

    @staticmethod
    def _samples() -> list[tuple[list[float], float]]:
        return [([1.0], 2.0), ([2.0], 4.0), ([3.0], 6.0)]

    @staticmethod
    def _seed_result(seed: int, status: str) -> SeedExperimentResult:
        return SeedExperimentResult(
            seed=seed,
            activation="relu",
            negative_slope=0.01,
            initial_active_count=1,
            initial_sample_count=2,
            initial_active_fraction=0.5,
            status=status,
            converged=status == "converged",
            epochs_completed=1,
            steps=2,
            best_epoch=1,
            stopped_reason="max_epochs",
            best_train_loss=0.5,
            best_validation_loss=0.25,
            final_weights=[1.0],
            final_bias=0.0,
            validation_predictions=[1.0],
        )

    def test_mean_from_total_keeps_direct_division_semantics(self) -> None:
        self.assertEqual(mean_from_total(3.0, 2), 1.5)
        with self.assertRaises(ZeroDivisionError):
            mean_from_total(1.0, 0)

    def test_evaluate_uses_shared_mean_and_restores_neuron(self) -> None:
        neuron = self._neuron()
        before = (neuron.weights.copy(), neuron.bias)
        with patch.object(
            evaluator_module, "mean_from_total", wraps=mean_from_total
        ) as aggregate:
            loss = evaluator_module.evaluate(neuron, self._samples())

        self.assertEqual(loss, 3.875)
        self.assertEqual(aggregate.call_count, 1)
        self.assertEqual(aggregate.call_args.args[1], 3)
        self.assertEqual((neuron.weights, neuron.bias), before)

    def test_train_epoch_uses_shared_mean_and_preserves_updates_and_rng(self) -> None:
        neuron = self._neuron()
        rng = random.Random(11)
        with patch.object(
            trainer_module, "mean_from_total", wraps=mean_from_total
        ) as aggregate:
            loss = trainer_module.train_epoch(
                neuron, self._samples(), 0.01, shuffle=True, rng=rng
            )

        self.assertEqual(loss, 3.581661973333333)
        self.assertEqual(aggregate.call_count, 1)
        self.assertEqual(aggregate.call_args.args[1], 3)
        self.assertEqual(neuron.weights, [0.672656])
        self.assertEqual(neuron.bias, 0.5715279999999999)
        self.assertEqual(rng.random(), 0.45183232059956535)

    def test_convergence_rate_uses_shared_ratio_for_empty_and_mixed_results(self) -> None:
        with patch.object(runner_module, "safe_ratio", wraps=safe_ratio) as ratio:
            self.assertEqual(MultiSeedSummary([]).convergence_rate, 0.0)
            self.assertEqual(ratio.call_args.args, (0, 0))
            mixed = MultiSeedSummary([
                self._seed_result(1, "converged"),
                self._seed_result(2, "not_converged"),
            ])
            self.assertEqual(mixed.convergence_rate, 0.5)
            self.assertEqual(ratio.call_args.args, (1, 2))

    def test_initial_activity_fraction_numeric_and_none_paths(self) -> None:
        samples = self._samples()
        config = TrainingConfig(
            learning_rate=0.01,
            max_epochs=2,
            patience=1,
            shuffle=False,
            seed=7,
        )
        with patch.object(runner_module, "safe_ratio", wraps=safe_ratio) as ratio:
            numeric, _ = run_single_seed(
                7, samples, samples, config, activation="relu"
            )
            self.assertEqual(numeric.initial_active_count, 0)
            self.assertEqual(numeric.initial_sample_count, 3)
            self.assertEqual(numeric.initial_active_fraction, 0.0)
            self.assertEqual(ratio.call_args.args, (0, 3))

            ratio.reset_mock()
            not_applicable, _ = run_single_seed(
                7, samples, samples, config, activation="linear"
            )
            self.assertIsNone(not_applicable.initial_active_count)
            self.assertIsNone(not_applicable.initial_active_fraction)
            ratio.assert_not_called()

    def test_aggregate_none_and_missing_paths_keep_historical_errors(self) -> None:
        with self.assertRaisesRegex(TypeError, "métrica loss não é numérica"):
            aggregate_metric_paths(
                [{"status": "completed", "metrics": {"loss": None}}],
                {"loss": ("metrics", "loss")},
            )
        with self.assertRaisesRegex(KeyError, "loss"):
            aggregate_metric_paths(
                [{"status": "completed", "metrics": {}}],
                {"loss": ("metrics", "loss")},
            )


if __name__ == "__main__":
    unittest.main()
