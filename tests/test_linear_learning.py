import unittest

from kinera.nn.linear import Linear


class LinearLearningTests(unittest.TestCase):
    def test_linear_learns_two_outputs(self) -> None:
        samples = [
            ([1.0, 1.0], [6.0, 1.0]),
            ([1.0, 2.0], [9.0, 5.0]),
            ([2.0, 1.0], [8.0, 0.0]),
            ([2.0, 2.0], [11.0, 4.0]),
        ]
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, 0.5], [0.5, 0.5]]
        layer.bias = [0.5, 0.5]

        initial_loss = self._loss(layer, samples)
        for _ in range(3_000):
            for inputs, targets in samples:
                prediction = layer.forward(inputs)
                gradient = [actual - target for actual, target in zip(prediction, targets)]
                layer.backward(gradient)
                layer.step(0.01)
        final_loss = self._loss(layer, samples)

        self.assertLess(final_loss, initial_loss)
        self.assertAlmostEqual(layer.weights[0][0], 2.0, delta=0.03)
        self.assertAlmostEqual(layer.weights[0][1], 3.0, delta=0.03)
        self.assertAlmostEqual(layer.bias[0], 1.0, delta=0.05)
        self.assertAlmostEqual(layer.weights[1][0], -1.0, delta=0.03)
        self.assertAlmostEqual(layer.weights[1][1], 4.0, delta=0.03)
        self.assertAlmostEqual(layer.bias[1], -2.0, delta=0.05)
        self.assertAlmostEqual(layer.forward([3.0, 4.0])[0], 19.0, delta=0.05)
        self.assertAlmostEqual(layer.forward([3.0, 4.0])[1], 11.0, delta=0.05)

    @staticmethod
    def _loss(
        layer: Linear,
        samples: list[tuple[list[float], list[float]]],
    ) -> float:
        total = 0.0
        for inputs, targets in samples:
            prediction = layer.forward(inputs)
            total += 0.5 * sum(
                (actual - target) ** 2
                for actual, target in zip(prediction, targets)
            )
        return total / len(samples)


if __name__ == "__main__":
    unittest.main()
