import unittest

from nn.neuron import Neuron
from training.data import generate_samples, split_samples
from training.loop import TrainingConfig, fit


class LearningTests(unittest.TestCase):
    def test_neuron_learns_y_equals_2x(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.weights = [0.5]
        neuron.bias = 0.5
        samples = [(float(value), 2.0 * value) for value in (1, 2, 3, 4)]
        learning_rate = 0.05

        initial_loss = self._epoch_loss(neuron, samples, learning_rate=None)
        for _ in range(100):
            self._epoch_loss(neuron, samples, learning_rate=learning_rate)
        final_loss = self._epoch_loss(neuron, samples, learning_rate=None)

        self.assertLess(final_loss, initial_loss)
        self.assertAlmostEqual(neuron.weights[0], 2.0, delta=0.02)
        self.assertAlmostEqual(neuron.bias, 0.0, delta=0.05)
        self.assertAlmostEqual(neuron.forward([5.0]), 10.0, delta=0.1)

    def test_neuron_learns_two_input_linear_function(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, 0.5]
        neuron.bias = 0.5
        samples = [
            ([1.0, 1.0], 6.0),
            ([1.0, 2.0], 9.0),
            ([2.0, 1.0], 8.0),
            ([2.0, 2.0], 11.0),
        ]

        initial_loss = self._multi_input_epoch_loss(
            neuron, samples, learning_rate=None
        )
        for _ in range(2_000):
            self._multi_input_epoch_loss(neuron, samples, learning_rate=0.02)
        final_loss = self._multi_input_epoch_loss(
            neuron, samples, learning_rate=None
        )

        self.assertLess(final_loss, initial_loss)
        self.assertAlmostEqual(neuron.weights[0], 2.0, delta=0.01)
        self.assertAlmostEqual(neuron.weights[1], 3.0, delta=0.01)
        self.assertAlmostEqual(neuron.bias, 1.0, delta=0.01)
        self.assertAlmostEqual(neuron.forward([3.0, 4.0]), 19.0, delta=0.01)
        self.assertAlmostEqual(neuron.forward([0.5, 0.5]), 3.5, delta=0.01)

    def test_leaky_relu_learns_from_the_previously_dead_seed(self) -> None:
        samples = generate_samples(count=100, seed=7)
        train_samples, validation_samples = split_samples(
            samples, validation_ratio=0.2, seed=7
        )
        neuron = Neuron(input_size=2, seed=7, activation="leaky_relu")
        result = fit(
            neuron,
            train_samples,
            validation_samples,
            TrainingConfig(
                learning_rate=0.01,
                max_epochs=500,
                patience=20,
                min_delta=1e-10,
                shuffle=True,
                seed=42,
            ),
        )

        self.assertLess(result.history.best_validation_loss, 1e-6)
        self.assertAlmostEqual(neuron.weights[0], 2.0, delta=0.01)
        self.assertAlmostEqual(neuron.weights[1], 3.0, delta=0.01)
        self.assertAlmostEqual(neuron.bias, 1.0, delta=0.01)

    @staticmethod
    def _epoch_loss(
        neuron: Neuron,
        samples: list[tuple[float, float]],
        learning_rate: float | None,
    ) -> float:
        total_loss = 0.0
        for value, target in samples:
            prediction = neuron.forward([value])
            error = prediction - target
            total_loss += 0.5 * error**2
            if learning_rate is not None:
                neuron.backward(error)
                neuron.step(learning_rate)
        return total_loss / len(samples)

    @staticmethod
    def _multi_input_epoch_loss(
        neuron: Neuron,
        samples: list[tuple[list[float], float]],
        learning_rate: float | None,
    ) -> float:
        total_loss = 0.0
        for inputs, target in samples:
            prediction = neuron.forward(inputs)
            error = prediction - target
            total_loss += 0.5 * error**2
            if learning_rate is not None:
                neuron.backward(error)
                neuron.step(learning_rate)
        return total_loss / len(samples)


if __name__ == "__main__":
    unittest.main()
