import unittest

from nn.neuron import Neuron
from training.loop import TrainingConfig, evaluate, fit


class TrainingLoopTests(unittest.TestCase):
    def test_fit_returns_one_metric_per_completed_epoch(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, 0.5]
        neuron.bias = 0.5
        train_samples = [
            ([1.0, 1.0], 6.0),
            ([1.0, 2.0], 9.0),
            ([2.0, 1.0], 8.0),
            ([2.0, 2.0], 11.0),
        ]
        validation_samples = [([0.5, 0.5], 3.5), ([3.0, 4.0], 19.0)]
        config = TrainingConfig(
            learning_rate=0.02,
            max_epochs=20,
            patience=20,
            min_delta=0.0,
            seed=7,
        )

        result = fit(neuron, train_samples, validation_samples, config)

        self.assertGreater(result.history.epochs_completed, 0)
        self.assertEqual(
            len(result.history.train_losses), result.history.epochs_completed
        )
        self.assertEqual(
            len(result.history.validation_losses), result.history.epochs_completed
        )
        self.assertEqual(
            result.history.steps,
            result.history.epochs_completed * len(train_samples),
        )
        self.assertAlmostEqual(
            result.history.best_validation_loss,
            min(result.history.validation_losses),
        )
        self.assertEqual(result.final_weights, neuron.weights)
        self.assertEqual(result.final_bias, neuron.bias)

    def test_evaluate_preserves_neuron_state(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, -1.0]
        neuron.bias = 0.25
        neuron.forward([2.0, -1.0])
        neuron.backward(3.0)
        snapshot = {
            "weights": neuron.weights.copy(),
            "bias": neuron.bias,
            "last_input": neuron.last_input.copy(),
            "last_z": neuron.last_z,
            "output": neuron.output,
            "grad_weights": neuron.grad_weights.copy(),
            "grad_bias": neuron.grad_bias,
            "has_backward": neuron._has_backward,
        }

        loss = evaluate(neuron, [([1.0, 1.0], 2.0)])

        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(neuron.weights, snapshot["weights"])
        self.assertEqual(neuron.bias, snapshot["bias"])
        self.assertEqual(neuron.last_input, snapshot["last_input"])
        self.assertEqual(neuron.last_z, snapshot["last_z"])
        self.assertEqual(neuron.output, snapshot["output"])
        self.assertEqual(neuron.grad_weights, snapshot["grad_weights"])
        self.assertEqual(neuron.grad_bias, snapshot["grad_bias"])
        self.assertEqual(neuron._has_backward, snapshot["has_backward"])

    def test_early_stopping_restores_best_validation_state(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.weights = [0.5]
        neuron.bias = 0.5
        config = TrainingConfig(
            learning_rate=0.1,
            max_epochs=10,
            patience=2,
            min_delta=0.0,
        )

        result = fit(
            neuron,
            train_samples=[([1.0], 3.0)],
            validation_samples=[([1.0], 1.0)],
            config=config,
        )

        self.assertEqual(result.history.stopped_reason, "early_stopping")
        self.assertEqual(result.history.epochs_completed, 3)
        self.assertEqual(result.history.best_epoch, 1)
        self.assertAlmostEqual(neuron.weights[0], 0.7)
        self.assertAlmostEqual(neuron.bias, 0.7)
        self.assertEqual(result.final_weights, neuron.weights)
        self.assertEqual(result.final_bias, neuron.bias)

    def test_zero_patience_waits_for_first_non_improving_epoch(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.weights = [0.5]
        neuron.bias = 0.5

        result = fit(
            neuron,
            train_samples=[([1.0], 3.0)],
            validation_samples=[([1.0], 1.0)],
            config=TrainingConfig(
                learning_rate=0.1,
                max_epochs=10,
                patience=0,
            ),
        )

        self.assertEqual(result.history.stopped_reason, "early_stopping")
        self.assertEqual(result.history.epochs_completed, 2)

    def test_shuffled_training_is_reproducible_with_a_seed(self) -> None:
        samples = [
            ([1.0, 1.0], 6.0),
            ([1.0, 2.0], 9.0),
            ([2.0, 1.0], 8.0),
            ([2.0, 2.0], 11.0),
        ]
        validation = [([0.5, 0.5], 3.5), ([3.0, 4.0], 19.0)]
        config = TrainingConfig(
            learning_rate=0.02,
            max_epochs=15,
            patience=15,
            shuffle=True,
            seed=42,
        )
        first_neuron = Neuron(input_size=2, seed=7)
        second_neuron = Neuron(input_size=2, seed=7)
        first_neuron.weights = second_neuron.weights = [0.5, 0.5]
        first_neuron.bias = second_neuron.bias = 0.5

        first = fit(first_neuron, samples, validation, config)
        second = fit(second_neuron, samples, validation, config)

        self.assertEqual(first.history.train_losses, second.history.train_losses)
        self.assertEqual(
            first.history.validation_losses, second.history.validation_losses
        )
        self.assertEqual(first.final_weights, second.final_weights)
        self.assertEqual(first.final_bias, second.final_bias)


class TrainingConfigTests(unittest.TestCase):
    def test_invalid_training_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.0, max_epochs=1, patience=1)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.1, max_epochs=0, patience=1)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.1, max_epochs=1, patience=-1)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.1, max_epochs=1, patience=1, min_delta=-1.0)


if __name__ == "__main__":
    unittest.main()
