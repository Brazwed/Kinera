"""Opt-in T4 validation for Linear.step/update after the backward gate PASS."""

from contextlib import ExitStack
import math
import os
import unittest
from typing import cast
from unittest.mock import patch

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.layers.linear import Linear


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class LinearCudaStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_parameters_match(self, cpu_layer: Linear, cuda_layer: Linear) -> float:
        self.assertIsInstance(cuda_layer.weights, CudaBuffer)
        self.assertIsInstance(cuda_layer.bias, CudaBuffer)
        actual_weights = cast(
            list[list[float]],
            self.backend.buffer_to_host(cast(CudaBuffer, cuda_layer.weights)),
        )
        actual_bias = cast(
            list[float], self.backend.buffer_to_host(cast(CudaBuffer, cuda_layer.bias))
        )
        cpu_weights = cast(list[list[float]], cpu_layer.weights)
        cpu_bias = cast(list[float], cpu_layer.bias)
        errors = [
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_weights, cpu_weights)
            for actual, expected in zip(actual_row, expected_row)
        ]
        errors.extend(
            abs(actual - expected)
            for actual, expected in zip(actual_bias, cpu_bias)
        )
        expected_count = len(cpu_bias) + sum(len(row) for row in cpu_weights)
        if len(errors) != expected_count:
            raise AssertionError("CPU/CUDA parameter shapes differ")
        for actual_row, expected_row in zip(actual_weights, cpu_weights):
            for actual, expected in zip(actual_row, expected_row):
                if not math.isfinite(actual):
                    raise AssertionError(f"CUDA weight is not finite: {actual}")
                if abs(actual - expected) > ATOL + RTOL * abs(expected):
                    raise AssertionError(f"CUDA weight {actual} != CPU {expected}")
        for actual, expected in zip(actual_bias, cpu_bias):
            if not math.isfinite(actual):
                raise AssertionError(f"CUDA bias is not finite: {actual}")
            if abs(actual - expected) > ATOL + RTOL * abs(expected):
                raise AssertionError(f"CUDA bias {actual} != CPU {expected}")
        return max(errors, default=0.0)

    def _make_layers(
        self, stack: ExitStack
    ) -> tuple[Linear, Linear, CudaBuffer, CudaBuffer]:
        initial_weights = [[0.5, -1.0, 0.25], [2.0, 0.25, -0.5]]
        initial_bias = [0.25, -0.5]
        cpu_layer = Linear(input_size=3, output_size=2, seed=7, backend=self.cpu)
        cpu_layer.weights = [row.copy() for row in initial_weights]
        cpu_layer.bias = initial_bias.copy()
        cuda_layer = Linear(input_size=3, output_size=2, seed=7, backend=self.backend)
        weights_buffer = stack.enter_context(
            self.backend.buffer_from_host(
                initial_weights, shape=(2, 3), dtype="float64"
            )
        )
        bias_buffer = stack.enter_context(
            self.backend.buffer_from_host(initial_bias, shape=(2,), dtype="float64")
        )
        cuda_layer.weights = weights_buffer
        cuda_layer.bias = bias_buffer
        return cpu_layer, cuda_layer, weights_buffer, bias_buffer

    def test_step_matches_cpu_after_one_and_multiple_consecutive_updates(self) -> None:
        learning_rate = 0.025
        step_count = 8
        input_values = [
            [0.125 * (index + 1), (-1.0) ** index * 0.5, 0.2 * (index - 2)]
            for index in range(step_count)
        ]
        output_gradients = [
            [0.3 * (index + 1), -0.15 * (index + 2)]
            for index in range(step_count)
        ]

        with ExitStack() as stack:
            cpu_layer, cuda_layer, weights_buffer, bias_buffer = self._make_layers(stack)
            input_buffers = [
                stack.enter_context(
                    self.backend.buffer_from_host(values, shape=(3,), dtype="float64")
                )
                for values in input_values
            ]
            gradient_buffers = [
                stack.enter_context(
                    self.backend.buffer_from_host(values, shape=(2,), dtype="float64")
                )
                for values in output_gradients
            ]

            max_error_after_first = None
            max_error_after_all = None
            for index, (inputs, grad_output) in enumerate(
                zip(input_values, output_gradients)
            ):
                cpu_layer.forward(inputs)
                cpu_layer.backward(grad_output)
                cpu_layer.step(learning_rate)

                output = cuda_layer.forward(input_buffers[index])
                self.assertIsInstance(output, CudaBuffer)
                stack.callback(cast(CudaBuffer, output).close)
                grad_inputs = cuda_layer.backward(gradient_buffers[index])
                self.assertIsInstance(grad_inputs, CudaBuffer)
                self.assertIsInstance(cuda_layer.grad_weights, CudaBuffer)
                self.assertIsInstance(cuda_layer.grad_bias, CudaBuffer)
                grad_inputs_buffer = cast(CudaBuffer, grad_inputs)
                grad_weights_buffer = cast(CudaBuffer, cuda_layer.grad_weights)
                grad_bias_buffer = cast(CudaBuffer, cuda_layer.grad_bias)
                stack.callback(grad_inputs_buffer.close)
                stack.callback(grad_weights_buffer.close)
                stack.callback(grad_bias_buffer.close)

                with (
                    patch.object(
                        self.backend.memory,
                        "copy_to_device",
                        side_effect=AssertionError("implicit H2D copy during Linear.step"),
                    ) as h2d,
                    patch.object(
                        self.backend.memory,
                        "copy_to_host",
                        side_effect=AssertionError("implicit D2H copy during Linear.step"),
                    ) as d2h,
                ):
                    cuda_layer.step(learning_rate)
                    h2d.assert_not_called()
                    d2h.assert_not_called()

                self.assertIs(cuda_layer.weights, weights_buffer)
                self.assertIs(cuda_layer.bias, bias_buffer)
                self.assertFalse(weights_buffer.closed)
                self.assertFalse(bias_buffer.closed)
                self.assertFalse(cuda_layer._has_backward)

                if index == 0:
                    max_error_after_first = self._assert_parameters_match(
                        cpu_layer, cuda_layer
                    )
                if index == step_count - 1:
                    max_error_after_all = self._assert_parameters_match(
                        cpu_layer, cuda_layer
                    )

            self.assertIsNotNone(max_error_after_first)
            self.assertIsNotNone(max_error_after_all)
            print("LINEAR_STEP_COUNT", step_count)
            print("LINEAR_STEP_MAX_ABS_ERROR_AFTER_1", max_error_after_first)
            print("LINEAR_STEP_MAX_ABS_ERROR_AFTER_ALL", max_error_after_all)

    def test_zero_learning_rate_preserves_parameters_exactly_without_transfer(self) -> None:
        with ExitStack() as stack:
            cpu_layer, cuda_layer, weights_buffer, bias_buffer = self._make_layers(stack)
            input_buffer = stack.enter_context(
                self.backend.buffer_from_host([1.5, -0.5, 2.0], shape=(3,))
            )
            grad_output_buffer = stack.enter_context(
                self.backend.buffer_from_host([0.75, -1.25], shape=(2,))
            )
            original_weights = self.backend.buffer_to_host(weights_buffer)
            original_bias = self.backend.buffer_to_host(bias_buffer)

            cpu_layer.forward([1.5, -0.5, 2.0])
            cpu_layer.backward([0.75, -1.25])
            cpu_layer.step(0.0)
            cuda_layer.forward(input_buffer)
            grad_inputs = cuda_layer.backward(grad_output_buffer)
            self.assertIsInstance(grad_inputs, CudaBuffer)
            stack.callback(cast(CudaBuffer, grad_inputs).close)

            with (
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=AssertionError("implicit H2D copy during zero-rate step"),
                ) as h2d,
                patch.object(
                    self.backend.memory,
                    "copy_to_host",
                    side_effect=AssertionError("implicit D2H copy during zero-rate step"),
                ) as d2h,
            ):
                cuda_layer.step(0.0)
                h2d.assert_not_called()
                d2h.assert_not_called()

            self.assertIs(cuda_layer.weights, weights_buffer)
            self.assertIs(cuda_layer.bias, bias_buffer)
            self.assertEqual(self.backend.buffer_to_host(weights_buffer), original_weights)
            self.assertEqual(self.backend.buffer_to_host(bias_buffer), original_bias)
            self.assertEqual(cpu_layer.weights, original_weights)
            self.assertEqual(cpu_layer.bias, original_bias)
            self.assertFalse(cuda_layer._has_backward)


if __name__ == "__main__":
    unittest.main()
