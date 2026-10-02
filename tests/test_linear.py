import math
import unittest

from nn.neuron import Neuron
from nn.linear import Linear


class LinearConstructionTests(unittest.TestCase):
    def test_constructor_creates_expected_dimensions(self) -> None:
        layer = Linear(input_size=2, output_size=3, seed=7)

        self.assertEqual(len(layer.weights), 3)
        self.assertTrue(all(len(row) == 2 for row in layer.weights))
        self.assertEqual(len(layer.bias), 3)
        self.assertEqual(len(layer.grad_weights), 3)
        self.assertTrue(all(len(row) == 2 for row in layer.grad_weights))
        self.assertEqual(layer.grad_bias, [0.0, 0.0, 0.0])
        self.assertIsNone(layer.last_input)
        self.assertIsNone(layer.output)

    def test_same_seed_reproduces_parameters(self) -> None:
        first = Linear(input_size=2, output_size=3, seed=7)
        second = Linear(input_size=2, output_size=3, seed=7)

        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.bias, second.bias)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Linear(input_size=0, output_size=1)
        with self.assertRaises(ValueError):
            Linear(input_size=1, output_size=0)
        with self.assertRaises(TypeError):
            Linear(input_size=True, output_size=1)
        with self.assertRaises(TypeError):
            Linear(input_size=1, output_size=False)


class LinearForwardTests(unittest.TestCase):
    def test_forward_calculates_all_affine_outputs(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, -1.0], [2.0, 0.25]]
        layer.bias = [0.25, -0.5]

        output = layer.forward([2.0, -1.0])

        self.assertEqual(output, [2.25, 3.25])
        self.assertEqual(layer.last_input, [2.0, -1.0])
        self.assertEqual(layer.output, [2.25, 3.25])

    def test_forward_matches_generator_reference_for_both_benchmark_shapes(self) -> None:
        for output_size in (51, 3932):
            with self.subTest(output_size=output_size):
                layer = Linear(input_size=26, output_size=output_size, seed=7)
                inputs = [float(index + 1) / 10 for index in range(26)]

                actual = layer.forward(inputs)
                expected = [
                    sum(weight * value for weight, value in zip(row, inputs))
                    + bias
                    for row, bias in zip(layer.weights, layer.bias)
                ]

                self.assertEqual(actual, expected)

    def test_forward_copies_input(self) -> None:
        layer = Linear(input_size=2, output_size=1, seed=7)
        inputs = [2.0, -1.0]

        layer.forward(inputs)
        inputs[0] = 99.0

        self.assertEqual(layer.last_input, [2.0, -1.0])

    def test_forward_rejects_invalid_inputs(self) -> None:
        layer = Linear(input_size=2, output_size=1, seed=7)

        with self.assertRaises(ValueError):
            layer.forward([1.0])
        with self.assertRaises(TypeError):
            layer.forward([1.0, True])
        with self.assertRaises(ValueError):
            layer.forward([1.0, math.nan])
        with self.assertRaises(ValueError):
            layer.forward([1.0, math.inf])


class LinearBackwardTests(unittest.TestCase):
    def _configured_layer(self) -> Linear:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, -1.0], [2.0, 0.25]]
        layer.bias = [0.25, -0.5]
        layer.forward([2.0, -1.0])
        return layer

    def test_backward_calculates_parameter_and_input_gradients(self) -> None:
        layer = self._configured_layer()

        gradients = layer.backward([3.0, -2.0])

        self.assertEqual(layer.grad_weights, [[6.0, -3.0], [-4.0, 2.0]])
        self.assertEqual(layer.grad_bias, [3.0, -2.0])
        self.assertEqual(gradients, [-2.5, -3.5])

    def test_backward_matches_generator_reference_for_benchmark_shapes(self) -> None:
        input_size = 26
        for output_size in (51, 3932):
            with self.subTest(output_size=output_size):
                layer = Linear(input_size=input_size, output_size=output_size, seed=7)
                inputs = [((index % 13) - 6) / 13.0 for index in range(input_size)]
                grad_output = [((index % 19) - 9) / 19.0 for index in range(output_size)]
                layer.forward(inputs)

                expected_grad_weights = [
                    [float(grad_output[row_index]) * value for value in inputs]
                    for row_index in range(output_size)
                ]
                expected_grad_bias = [float(value) for value in grad_output]
                expected_grad_inputs = [
                    sum(
                        float(grad_output[row_index]) * layer.weights[row_index][input_index]
                        for row_index in range(output_size)
                    )
                    for input_index in range(input_size)
                ]

                actual_grad_inputs = layer.backward(grad_output)

                self.assertEqual(layer.grad_weights, expected_grad_weights)
                self.assertEqual(layer.grad_bias, expected_grad_bias)
                self.assertEqual(actual_grad_inputs, expected_grad_inputs)
                self.assertTrue(layer._has_backward)

    def test_backward_before_forward_is_rejected(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)

        with self.assertRaises(RuntimeError):
            layer.backward([1.0, 1.0])

    def test_backward_rejects_invalid_gradient_vector(self) -> None:
        layer = self._configured_layer()

        with self.assertRaises(ValueError):
            layer.backward([1.0])
        with self.assertRaises(TypeError):
            layer.backward([1.0, True])
        with self.assertRaises(ValueError):
            layer.backward([1.0, math.nan])


class LinearStepTests(unittest.TestCase):
    def _configured_layer(self) -> Linear:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, -1.0], [2.0, 0.25]]
        layer.bias = [0.25, -0.5]
        layer.forward([2.0, -1.0])
        layer.backward([3.0, -2.0])
        return layer

    def test_step_updates_all_parameters(self) -> None:
        layer = self._configured_layer()

        layer.step(0.1)

        self.assertAlmostEqual(layer.weights[0][0], -0.1)
        self.assertAlmostEqual(layer.weights[0][1], -0.7)
        self.assertAlmostEqual(layer.weights[1][0], 2.4)
        self.assertAlmostEqual(layer.weights[1][1], 0.05)
        self.assertAlmostEqual(layer.bias[0], -0.05)
        self.assertAlmostEqual(layer.bias[1], -0.3)

    def test_step_before_backward_and_twice_are_rejected(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.forward([1.0, 1.0])
        with self.assertRaises(RuntimeError):
            layer.step(0.1)

        layer = self._configured_layer()
        layer.step(0.1)
        with self.assertRaises(RuntimeError):
            layer.step(0.1)

    def test_step_rejects_negative_learning_rate(self) -> None:
        layer = self._configured_layer()

        with self.assertRaises(ValueError):
            layer.step(-0.1)


class LinearComparisonTests(unittest.TestCase):
    def test_linear_matches_multiple_linear_neurons(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, -1.0], [2.0, 0.25]]
        layer.bias = [0.25, -0.5]
        neurons = [
            Neuron(input_size=2, seed=7, activation="linear"),
            Neuron(input_size=2, seed=8, activation="linear"),
        ]
        for neuron, weights, bias in zip(
            neurons, layer.weights, layer.bias
        ):
            neuron.weights = weights.copy()
            neuron.bias = bias

        inputs = [2.0, -1.0]
        self.assertEqual(
            layer.forward(inputs),
            [neuron.forward(inputs) for neuron in neurons],
        )

        layer_input_gradients = layer.backward([3.0, -2.0])
        neuron_input_gradients = [
            neuron.backward(gradient)
            for neuron, gradient in zip(neurons, [3.0, -2.0])
        ]
        self.assertEqual(layer.grad_weights[0], neurons[0].grad_weights)
        self.assertEqual(layer.grad_weights[1], neurons[1].grad_weights)
        self.assertEqual(layer.grad_bias, [neuron.grad_bias for neuron in neurons])
        self.assertEqual(
            layer_input_gradients,
            [
                sum(gradients[index] for gradients in neuron_input_gradients)
                for index in range(2)
            ],
        )

        layer.step(0.1)
        for neuron in neurons:
            neuron.step(0.1)
        self.assertEqual(layer.weights, [neuron.weights for neuron in neurons])
        self.assertEqual(layer.bias, [neuron.bias for neuron in neurons])


class LinearGradientCheckingTests(unittest.TestCase):
    @staticmethod
    def _loss(layer: Linear, inputs: list[float], target: list[float]) -> float:
        prediction = layer.forward(inputs)
        return 0.5 * sum(
            (actual - expected) ** 2
            for actual, expected in zip(prediction, target)
        )

    def test_all_parameter_gradients_match_finite_difference(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.4, -0.2], [-0.3, 0.5]]
        layer.bias = [0.3, -0.1]
        inputs = [1.5, -0.5]
        target = [0.7, -0.2]
        epsilon = 1e-6

        prediction = layer.forward(inputs)
        layer.backward([actual - expected for actual, expected in zip(prediction, target)])
        analytic_weights = [row.copy() for row in layer.grad_weights]
        analytic_bias = layer.grad_bias.copy()
        original_weights = [row.copy() for row in layer.weights]
        original_bias = layer.bias.copy()

        for row_index in range(2):
            for column_index in range(2):
                layer.weights[row_index][column_index] += epsilon
                loss_plus = self._loss(layer, inputs, target)
                layer.weights[row_index][column_index] -= 2.0 * epsilon
                loss_minus = self._loss(layer, inputs, target)
                layer.weights[row_index][column_index] = original_weights[row_index][column_index]
                numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
                self.assertAlmostEqual(
                    numerical,
                    analytic_weights[row_index][column_index],
                    delta=1e-5,
                )

        for row_index in range(2):
            layer.bias[row_index] += epsilon
            loss_plus = self._loss(layer, inputs, target)
            layer.bias[row_index] -= 2.0 * epsilon
            loss_minus = self._loss(layer, inputs, target)
            layer.bias[row_index] = original_bias[row_index]
            numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(numerical, analytic_bias[row_index], delta=1e-5)


if __name__ == "__main__":
    unittest.main()
