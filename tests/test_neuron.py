import math
import unittest

from nn.neuron import Neuron


class NeuronConstructionTests(unittest.TestCase):
    def test_positive_input_size_creates_matching_parameters(self) -> None:
        neuron = Neuron(input_size=3, seed=7)

        self.assertEqual(len(neuron.weights), 3)
        self.assertTrue(all(isinstance(weight, float) for weight in neuron.weights))
        self.assertIsInstance(neuron.bias, float)
        self.assertTrue(all(math.isfinite(weight) for weight in neuron.weights))
        self.assertTrue(math.isfinite(neuron.bias))

    def test_same_seed_reproduces_parameters(self) -> None:
        first = Neuron(input_size=3, seed=7)
        second = Neuron(input_size=3, seed=7)

        self.assertEqual(first.weights, second.weights)
        self.assertEqual(first.bias, second.bias)

    def test_different_seed_changes_parameters(self) -> None:
        first = Neuron(input_size=3, seed=7)
        second = Neuron(input_size=3, seed=8)

        self.assertNotEqual(first.weights + [first.bias], second.weights + [second.bias])

    def test_non_positive_input_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Neuron(input_size=0)

        with self.assertRaises(ValueError):
            Neuron(input_size=-1)

    def test_boolean_input_size_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Neuron(input_size=True)

    def test_gradients_and_forward_state_start_empty(self) -> None:
        neuron = Neuron(input_size=2, seed=7)

        self.assertEqual(neuron.activation, "relu")
        self.assertEqual(neuron.grad_weights, [0.0, 0.0])
        self.assertEqual(neuron.grad_bias, 0.0)
        self.assertIsNone(neuron.last_input)
        self.assertIsNone(neuron.last_z)
        self.assertIsNone(neuron.output)


class NeuronForwardTests(unittest.TestCase):
    def test_forward_calculates_weighted_sum_and_relu(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, -1.0]
        neuron.bias = 0.25

        result = neuron.forward([2.0, -1.0])

        self.assertEqual(result, 2.25)
        self.assertEqual(neuron.last_z, 2.25)
        self.assertEqual(neuron.output, 2.25)

    def test_forward_returns_zero_when_weighted_sum_is_negative(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [-1.0, -1.0]
        neuron.bias = -0.5

        self.assertEqual(neuron.forward([1.0, 1.0]), 0.0)
        self.assertEqual(neuron.last_z, -2.5)

    def test_linear_activation_preserves_negative_weighted_sum(self) -> None:
        neuron = Neuron(input_size=2, seed=7, activation="linear")
        neuron.weights = [-1.0, -1.0]
        neuron.bias = -0.5

        self.assertEqual(neuron.forward([1.0, 1.0]), -2.5)

    def test_leaky_relu_activation_preserves_negative_gradient(self) -> None:
        neuron = Neuron(input_size=2, seed=7, activation="leaky_relu")
        neuron.weights = [-1.0, -1.0]
        neuron.bias = -0.5

        neuron.forward([1.0, 1.0])
        neuron.backward(3.0)

        self.assertAlmostEqual(neuron.output, -0.025)
        self.assertAlmostEqual(neuron.grad_weights[0], 0.03)
        self.assertAlmostEqual(neuron.grad_weights[1], 0.03)
        self.assertAlmostEqual(neuron.grad_bias, 0.03)

    def test_forward_copies_input_state(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        inputs = [2.0, -1.0]

        neuron.forward(inputs)
        inputs[0] = 99.0

        self.assertEqual(neuron.last_input, [2.0, -1.0])

    def test_forward_rejects_wrong_input_size(self) -> None:
        neuron = Neuron(input_size=2, seed=7)

        with self.assertRaises(ValueError):
            neuron.forward([1.0])

    def test_forward_rejects_invalid_input_values(self) -> None:
        neuron = Neuron(input_size=1, seed=7)

        with self.assertRaises(TypeError):
            neuron.forward(["1.0"])
        with self.assertRaises(TypeError):
            neuron.forward([True])
        with self.assertRaises(ValueError):
            neuron.forward([math.nan])
        with self.assertRaises(ValueError):
            neuron.forward([math.inf])

    def test_unknown_activation_and_invalid_slope_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Neuron(input_size=1, activation="unknown")
        with self.assertRaises(ValueError):
            Neuron(input_size=1, activation="leaky_relu", negative_slope=0.0)


class NeuronBackwardTests(unittest.TestCase):
    def test_backward_calculates_parameter_and_input_gradients(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, -1.0]
        neuron.bias = 0.25
        neuron.forward([2.0, -1.0])

        gradients = neuron.backward(3.0)

        self.assertEqual(neuron.grad_weights, [6.0, -3.0])
        self.assertEqual(neuron.grad_bias, 3.0)
        self.assertEqual(gradients, [1.5, -3.0])

    def test_backward_uses_zero_relu_derivative_when_z_is_non_positive(self) -> None:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [-1.0, -1.0]
        neuron.bias = -0.5
        neuron.forward([1.0, 1.0])

        gradients = neuron.backward(3.0)

        self.assertEqual(neuron.grad_weights, [0.0, 0.0])
        self.assertEqual(neuron.grad_bias, 0.0)
        self.assertEqual(gradients, [0.0, 0.0])

    def test_backward_before_forward_is_rejected(self) -> None:
        neuron = Neuron(input_size=1, seed=7)

        with self.assertRaises(RuntimeError):
            neuron.backward(1.0)

    def test_backward_rejects_invalid_gradient(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.forward([1.0])

        with self.assertRaises(TypeError):
            neuron.backward("1.0")
        with self.assertRaises(TypeError):
            neuron.backward(True)
        with self.assertRaises(ValueError):
            neuron.backward(math.nan)
        with self.assertRaises(ValueError):
            neuron.backward(math.inf)


class NeuronStepTests(unittest.TestCase):
    def _active_neuron(self) -> Neuron:
        neuron = Neuron(input_size=2, seed=7)
        neuron.weights = [0.5, -1.0]
        neuron.bias = 0.25
        neuron.forward([2.0, -1.0])
        neuron.backward(3.0)
        return neuron

    def test_step_updates_weights_and_bias(self) -> None:
        neuron = self._active_neuron()

        neuron.step(0.1)

        self.assertAlmostEqual(neuron.weights[0], -0.1)
        self.assertAlmostEqual(neuron.weights[1], -0.7)
        self.assertAlmostEqual(neuron.bias, -0.05)

    def test_zero_learning_rate_does_not_change_parameters(self) -> None:
        neuron = self._active_neuron()
        original_weights = neuron.weights.copy()
        original_bias = neuron.bias

        neuron.step(0.0)

        self.assertEqual(neuron.weights, original_weights)
        self.assertEqual(neuron.bias, original_bias)

    def test_step_before_backward_is_rejected(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.forward([1.0])

        with self.assertRaises(RuntimeError):
            neuron.step(0.1)

    def test_second_step_without_new_backward_is_rejected(self) -> None:
        neuron = self._active_neuron()
        neuron.step(0.1)

        with self.assertRaises(RuntimeError):
            neuron.step(0.1)

    def test_step_does_not_run_forward_implicitly(self) -> None:
        neuron = self._active_neuron()
        original_input = neuron.last_input.copy()
        original_output = neuron.output

        neuron.step(0.1)

        self.assertEqual(neuron.last_input, original_input)
        self.assertEqual(neuron.output, original_output)

    def test_step_rejects_invalid_learning_rate(self) -> None:
        neuron = self._active_neuron()

        with self.assertRaises(TypeError):
            neuron.step("0.1")
        with self.assertRaises(TypeError):
            neuron.step(True)
        with self.assertRaises(ValueError):
            neuron.step(-0.1)
        with self.assertRaises(ValueError):
            neuron.step(math.nan)
        with self.assertRaises(ValueError):
            neuron.step(math.inf)


if __name__ == "__main__":
    unittest.main()
