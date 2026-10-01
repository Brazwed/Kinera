import unittest

from nn.neuron import Neuron


class GradientCheckingTests(unittest.TestCase):
    @staticmethod
    def _loss(neuron: Neuron, inputs: list[float], target: float) -> float:
        prediction = neuron.forward(inputs)
        return 0.5 * (prediction - target) ** 2

    def test_parameter_gradients_match_finite_difference(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.4, -0.2]
        neuron.bias = 0.3
        inputs = [1.5, -0.5]
        target = 0.7
        epsilon = 1e-6

        prediction = neuron.forward(inputs)
        neuron.backward(prediction - target)
        analytic_weights = neuron.grad_weights.copy()
        analytic_bias = neuron.grad_bias
        original_weights = neuron.weights.copy()
        original_bias = neuron.bias

        numerical_weights = []
        for index in range(len(neuron.weights)):
            neuron.weights[index] += epsilon
            loss_plus = self._loss(neuron, inputs, target)
            neuron.weights[index] -= 2.0 * epsilon
            loss_minus = self._loss(neuron, inputs, target)
            neuron.weights[index] = original_weights[index]
            numerical_weights.append((loss_plus - loss_minus) / (2.0 * epsilon))

        neuron.bias += epsilon
        loss_plus = self._loss(neuron, inputs, target)
        neuron.bias -= 2.0 * epsilon
        loss_minus = self._loss(neuron, inputs, target)
        neuron.bias = original_bias
        numerical_bias = (loss_plus - loss_minus) / (2.0 * epsilon)

        self.assertEqual(neuron.weights, original_weights)
        self.assertEqual(neuron.bias, original_bias)
        for numerical, analytic in zip(numerical_weights, analytic_weights):
            self.assertAlmostEqual(numerical, analytic, delta=1e-5)
        self.assertAlmostEqual(numerical_bias, analytic_bias, delta=1e-5)

    def test_leaky_relu_gradients_match_in_negative_region(self) -> None:
        self._assert_gradients_match(
            Neuron(input_size=2, seed=7, activation="leaky_relu"),
            weights=[-0.4, -0.2],
            bias=-0.3,
            inputs=[1.5, -0.5],
            target=-0.2,
        )

    def test_linear_gradients_match_in_negative_region(self) -> None:
        self._assert_gradients_match(
            Neuron(input_size=2, seed=7, activation="linear"),
            weights=[-0.4, -0.2],
            bias=-0.3,
            inputs=[1.5, -0.5],
            target=-0.2,
        )

    def _assert_gradients_match(
        self,
        neuron: Neuron,
        *,
        weights: list[float],
        bias: float,
        inputs: list[float],
        target: float,
    ) -> None:
        neuron.weights = weights.copy()
        neuron.bias = bias
        epsilon = 1e-6

        prediction = neuron.forward(inputs)
        neuron.backward(prediction - target)
        analytic_weights = neuron.grad_weights.copy()
        analytic_bias = neuron.grad_bias
        original_weights = neuron.weights.copy()
        original_bias = neuron.bias

        numerical_weights = []
        for index in range(len(neuron.weights)):
            neuron.weights[index] += epsilon
            loss_plus = self._loss(neuron, inputs, target)
            neuron.weights[index] -= 2.0 * epsilon
            loss_minus = self._loss(neuron, inputs, target)
            neuron.weights[index] = original_weights[index]
            numerical_weights.append((loss_plus - loss_minus) / (2.0 * epsilon))

        neuron.bias += epsilon
        loss_plus = self._loss(neuron, inputs, target)
        neuron.bias -= 2.0 * epsilon
        loss_minus = self._loss(neuron, inputs, target)
        neuron.bias = original_bias
        numerical_bias = (loss_plus - loss_minus) / (2.0 * epsilon)

        for numerical, analytic in zip(numerical_weights, analytic_weights):
            self.assertAlmostEqual(numerical, analytic, delta=1e-5)
        self.assertAlmostEqual(numerical_bias, analytic_bias, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
