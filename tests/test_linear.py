import math
from copy import deepcopy
import unittest

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.neuron import Neuron
from kinera.nn.linear import Linear


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

        learning_rate = 0.05
        for output_size in (51, 3932):
            with self.subTest(output_size=output_size):
                layer = Linear(input_size=26, output_size=output_size, seed=7)
                inputs = [((index % 13) - 6) / 13.0 for index in range(26)]
                grad_output = [((index % 19) - 9) / 19.0 for index in range(output_size)]
                layer.forward(inputs)
                layer.backward(grad_output)

                expected_weights = [
                    [
                        weight - float(learning_rate) * gradient
                        for weight, gradient in zip(row, gradient_row)
                    ]
                    for row, gradient_row in zip(layer.weights, layer.grad_weights)
                ]
                expected_bias = [
                    bias - float(learning_rate) * gradient
                    for bias, gradient in zip(layer.bias, layer.grad_bias)
                ]
                old_weights = layer.weights
                old_rows = layer.weights.copy()
                old_bias = layer.bias

                layer.step(learning_rate)

                self.assertEqual(layer.weights, expected_weights)
                self.assertEqual(layer.bias, expected_bias)
                self.assertFalse(layer._has_backward)
                self.assertIsNot(layer.weights, old_weights)
                self.assertTrue(
                    all(new_row is not old_row for new_row, old_row in zip(layer.weights, old_rows))
                )
                self.assertIsNot(layer.bias, old_bias)

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


class LinearBackendIntegrationTests(unittest.TestCase):
    def test_linear_delegates_math_operations_to_injected_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def linear_forward(self, weights, bias, inputs):
                self.calls.append("forward")
                return super().linear_forward(weights, bias, inputs)

            def linear_backward(self, weights, inputs, grad_output):
                self.calls.append("backward")
                return super().linear_backward(weights, inputs, grad_output)

            def linear_step(self, weights, bias, grad_weights, grad_bias, learning_rate):
                self.calls.append("step")
                return super().linear_step(
                    weights, bias, grad_weights, grad_bias, learning_rate
                )

        backend = RecordingBackend()
        layer = Linear(input_size=2, output_size=1, seed=7, backend=backend)

        layer.forward([1.0, 2.0])
        layer.backward([1.0])
        layer.step(0.01)

        self.assertEqual(backend.calls, ["forward", "backward", "step"])


class LinearBackendBufferIntegrationTests(unittest.TestCase):
    class Buffer:
        def __init__(self, values, shape):
            self.values = deepcopy(values)
            self.shape = shape
            self.dtype = "float64"
            self.device = "mock"
            self.device_id = 0
            self.nbytes = 8 * math.prod(shape)
            self.layout = "C"
            self.owns_memory = True
            self.closed = False

    class BufferBackend(CPUBackend):
        def __init__(self) -> None:
            self.host_to_backend = 0
            self.backend_to_host = 0

        def buffer_from_host(self, values, *, shape, dtype="float64"):
            self.host_to_backend += 1
            return LinearBackendBufferIntegrationTests.Buffer(values, shape)

        def buffer_to_host(self, buffer):
            self.backend_to_host += 1
            return deepcopy(buffer.values)

        def linear_forward(self, weights, bias, inputs):
            self.assert_buffers(weights, bias, inputs)
            output = super().linear_forward(weights.values, bias.values, inputs.values)
            return LinearBackendBufferIntegrationTests.Buffer(output, (len(output),))

        @staticmethod
        def assert_buffers(*buffers) -> None:
            if not all(isinstance(buffer, LinearBackendBufferIntegrationTests.Buffer) for buffer in buffers):
                raise TypeError("backend operations require backend-owned buffers")

    def test_forward_keeps_backend_result_resident_until_explicit_export(self) -> None:
        backend = self.BufferBackend()
        layer = Linear(input_size=2, output_size=1, seed=7, backend=backend)
        layer.weights = backend.buffer_from_host([[2.0, -1.0]], shape=(1, 2))
        layer.bias = backend.buffer_from_host([0.25], shape=(1,))
        copies_before_forward = backend.host_to_backend

        output = layer.forward([4.0, 2.0])

        self.assertIsInstance(output, self.Buffer)
        self.assertEqual(backend.host_to_backend, copies_before_forward + 1)
        self.assertEqual(backend.backend_to_host, 0)
        self.assertEqual(backend.buffer_to_host(output), [6.25])

    def test_forward_accepts_resident_input_without_an_implicit_copy(self) -> None:
        backend = self.BufferBackend()
        layer = Linear(input_size=2, output_size=1, seed=7, backend=backend)
        layer.weights = backend.buffer_from_host([[2.0, -1.0]], shape=(1, 2))
        layer.bias = backend.buffer_from_host([0.25], shape=(1,))
        input_buffer = backend.buffer_from_host([4.0, 2.0], shape=(2,))
        copies_before_forward = backend.host_to_backend

        output = layer.forward(input_buffer)

        self.assertIsInstance(output, self.Buffer)
        self.assertEqual(backend.host_to_backend, copies_before_forward)
        self.assertEqual(backend.backend_to_host, 0)


if __name__ == "__main__":
    unittest.main()
