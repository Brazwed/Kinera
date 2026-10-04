"""Opt-in GPU tests for Linear.backward after the forward gate PASS."""

import math
import os
import unittest
from contextlib import ExitStack
from typing import cast

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.layers.linear import Linear


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class LinearCudaBackwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_vector_close(self, actual, expected) -> float:
        self.assertEqual(len(actual), len(expected))
        errors = [abs(got - want) for got, want in zip(actual, expected)]
        for index, (got, want, error) in enumerate(zip(actual, expected, errors)):
            limit = ATOL + RTOL * abs(want)
            with self.subTest(index=index):
                self.assertTrue(math.isfinite(got), msg=f"CUDA gradient is not finite: {got}")
                self.assertLessEqual(error, limit, msg=f"CUDA={got}, CPU={want}")
        return max(errors, default=0.0)

    def test_parameter_and_input_gradients_match_cpu_and_finite_difference(self) -> None:
        weights = [[0.5, -1.0, 0.25], [2.0, 0.25, -0.5]]
        bias = [0.25, -0.5]
        inputs = [2.0, -1.0, 0.5]
        grad_output = [3.0, -2.0]
        layer = Linear(input_size=3, output_size=2, seed=7, backend=self.backend)

        with ExitStack() as stack:
            device_weights = stack.enter_context(
                self.backend.buffer_from_host(weights, shape=(2, 3), dtype="float64")
            )
            device_bias = stack.enter_context(
                self.backend.buffer_from_host(bias, shape=(2,), dtype="float64")
            )
            device_inputs = stack.enter_context(
                self.backend.buffer_from_host(inputs, shape=(3,), dtype="float64")
            )
            layer.weights = device_weights
            layer.bias = device_bias
            raw_output = layer.forward(device_inputs)
            self.assertIsInstance(raw_output, CudaBuffer)
            output = cast(CudaBuffer, raw_output)
            stack.callback(output.close)

            raw_grad_inputs = layer.backward(grad_output)
            self.assertIsInstance(raw_grad_inputs, CudaBuffer)
            self.assertIsInstance(layer.grad_weights, CudaBuffer)
            self.assertIsInstance(layer.grad_bias, CudaBuffer)
            grad_inputs = cast(CudaBuffer, raw_grad_inputs)
            grad_weights_buffer = cast(CudaBuffer, layer.grad_weights)
            grad_bias_buffer = cast(CudaBuffer, layer.grad_bias)
            stack.callback(grad_inputs.close)
            stack.callback(grad_weights_buffer.close)
            stack.callback(grad_bias_buffer.close)

            expected_grad_weights, expected_grad_bias, expected_grad_inputs = (
                self.cpu.linear_backward(weights, inputs, grad_output)
            )
            actual_grad_weights = cast(
                list[list[float]], self.backend.buffer_to_host(grad_weights_buffer)
            )
            actual_grad_bias = cast(
                list[float], self.backend.buffer_to_host(grad_bias_buffer)
            )
            actual_grad_inputs = cast(
                list[float], self.backend.buffer_to_host(grad_inputs)
            )
            flattened_actual = [value for row in actual_grad_weights for value in row]
            flattened_expected = [value for row in expected_grad_weights for value in row]
            max_error = max(
                self._assert_vector_close(flattened_actual, flattened_expected),
                self._assert_vector_close(actual_grad_bias, expected_grad_bias),
                self._assert_vector_close(actual_grad_inputs, expected_grad_inputs),
            )

        def objective(candidate_weights, candidate_bias, candidate_inputs=inputs) -> float:
            with ExitStack() as stack:
                weights_buffer = stack.enter_context(
                    self.backend.buffer_from_host(
                        candidate_weights, shape=(2, 3), dtype="float64"
                    )
                )
                bias_buffer = stack.enter_context(
                    self.backend.buffer_from_host(
                        candidate_bias, shape=(2,), dtype="float64"
                    )
                )
                inputs_buffer = stack.enter_context(
                    self.backend.buffer_from_host(
                        candidate_inputs, shape=(3,), dtype="float64"
                    )
                )
                output_buffer = stack.enter_context(
                    self.backend.linear_forward(weights_buffer, bias_buffer, inputs_buffer)
                )
                output_values = cast(
                    list[float], self.backend.buffer_to_host(output_buffer)
                )
                return sum(
                    value * gradient
                    for value, gradient in zip(output_values, grad_output)
                )

        epsilon = 1e-6
        perturbed_weights_plus = [row.copy() for row in weights]
        perturbed_weights_minus = [row.copy() for row in weights]
        perturbed_weights_plus[0][0] += epsilon
        perturbed_weights_minus[0][0] -= epsilon
        finite_difference_weight = (
            objective(perturbed_weights_plus, bias)
            - objective(perturbed_weights_minus, bias)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(finite_difference_weight, expected_grad_weights[0][0], delta=1e-7)

        perturbed_bias_plus = bias.copy()
        perturbed_bias_minus = bias.copy()
        perturbed_bias_plus[1] += epsilon
        perturbed_bias_minus[1] -= epsilon
        finite_difference_bias = (
            objective(weights, perturbed_bias_plus)
            - objective(weights, perturbed_bias_minus)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(finite_difference_bias, expected_grad_bias[1], delta=1e-7)

        perturbed_inputs_plus = inputs.copy()
        perturbed_inputs_minus = inputs.copy()
        perturbed_inputs_plus[0] += epsilon
        perturbed_inputs_minus[0] -= epsilon
        finite_difference_input = (
            objective(weights, bias, perturbed_inputs_plus)
            - objective(weights, bias, perturbed_inputs_minus)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            finite_difference_input, expected_grad_inputs[0], delta=1e-7
        )
        print("LINEAR_BACKWARD_MAX_ABS_ERROR", max_error)


if __name__ == "__main__":
    unittest.main()
