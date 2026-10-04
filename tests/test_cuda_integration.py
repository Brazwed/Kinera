"""Minimal integrated Linear forward/backward/step validation."""

from contextlib import ExitStack
import math
import os
import unittest
from typing import cast
from unittest.mock import patch

from kinera.backends.base import BackendUnavailableError
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer, CudaRuntime
from kinera.backends.cuda.bridge import CudaBridge
from kinera.backends.cuda.buffer import BufferReleasedError
from kinera.nn.layers.linear import Linear


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


class UnavailableCudaBridge:
    def device_count(self) -> int:
        raise BackendUnavailableError("simulated missing CUDA runtime")


class LinearCudaIntegratedUnavailableTests(unittest.TestCase):
    def test_layer_creation_fails_explicitly_without_cuda_and_never_falls_back(self) -> None:
        backend = CUDABackend(
            runtime=CudaRuntime(bridge=cast(CudaBridge, UnavailableCudaBridge()))
        )
        with self.assertRaisesRegex(
            BackendUnavailableError, "simulated missing CUDA runtime"
        ):
            Linear(input_size=3, output_size=2, seed=7, backend=backend)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class LinearCudaIntegratedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_vector_close(self, actual: list[float], expected: list[float]) -> float:
        self.assertEqual(len(actual), len(expected))
        errors = [abs(got - want) for got, want in zip(actual, expected)]
        for index, (got, want, error) in enumerate(zip(actual, expected, errors)):
            with self.subTest(index=index):
                self.assertTrue(math.isfinite(got), msg=f"CUDA result is not finite: {got}")
                self.assertLessEqual(error, ATOL + RTOL * abs(want))
        return max(errors, default=0.0)

    def test_forward_backward_step_fixture_matches_cpu_and_releases_buffers(self) -> None:
        weights = [[0.5, -1.0, 0.25], [2.0, 0.25, -0.5]]
        bias = [0.25, -0.5]
        inputs = [2.0, -1.0, 0.5]
        grad_output = [0.75, -0.25]
        learning_rate = 0.05
        cpu_layer = Linear(input_size=3, output_size=2, seed=7, backend=self.cpu)
        cpu_layer.weights = [row.copy() for row in weights]
        cpu_layer.bias = bias.copy()
        cuda_layer = Linear(input_size=3, output_size=2, seed=7, backend=self.backend)
        owned_buffers: list[CudaBuffer] = []

        with ExitStack() as stack:
            self.assertIsInstance(cuda_layer.weights, CudaBuffer)
            self.assertIsInstance(cuda_layer.bias, CudaBuffer)
            initial_cuda_weights = cast(CudaBuffer, cuda_layer.weights)
            initial_cuda_bias = cast(CudaBuffer, cuda_layer.bias)
            stack.callback(initial_cuda_weights.close)
            stack.callback(initial_cuda_bias.close)
            owned_buffers.extend([initial_cuda_weights, initial_cuda_bias])
            device_weights = stack.enter_context(
                self.backend.buffer_from_host(weights, shape=(2, 3), dtype="float64")
            )
            device_bias = stack.enter_context(
                self.backend.buffer_from_host(bias, shape=(2,), dtype="float64")
            )
            device_inputs = stack.enter_context(
                self.backend.buffer_from_host(inputs, shape=(3,), dtype="float64")
            )
            device_grad_output = stack.enter_context(
                self.backend.buffer_from_host(grad_output, shape=(2,), dtype="float64")
            )
            owned_buffers.extend(
                [device_weights, device_bias, device_inputs, device_grad_output]
            )
            cuda_layer.weights = device_weights
            cuda_layer.bias = device_bias

            cpu_output = cast(list[float], cpu_layer.forward(inputs))
            raw_cuda_output = cuda_layer.forward(device_inputs)
            self.assertIsInstance(raw_cuda_output, CudaBuffer)
            cuda_output = cast(CudaBuffer, raw_cuda_output)
            stack.callback(cuda_output.close)
            owned_buffers.append(cuda_output)
            cuda_output_host = cast(list[float], self.backend.buffer_to_host(cuda_output))
            max_error = self._assert_vector_close(cuda_output_host, cpu_output)

            cpu_grad_inputs = cast(list[float], cpu_layer.backward(grad_output))
            cpu_grad_weights = cast(list[list[float]], cpu_layer.grad_weights)
            cpu_grad_bias = cast(list[float], cpu_layer.grad_bias)
            raw_cuda_grad_inputs = cuda_layer.backward(device_grad_output)
            self.assertIsInstance(raw_cuda_grad_inputs, CudaBuffer)
            self.assertIsInstance(cuda_layer.grad_weights, CudaBuffer)
            self.assertIsInstance(cuda_layer.grad_bias, CudaBuffer)
            cuda_grad_inputs = cast(CudaBuffer, raw_cuda_grad_inputs)
            cuda_grad_weights = cast(CudaBuffer, cuda_layer.grad_weights)
            cuda_grad_bias = cast(CudaBuffer, cuda_layer.grad_bias)
            stack.callback(cuda_grad_inputs.close)
            stack.callback(cuda_grad_weights.close)
            stack.callback(cuda_grad_bias.close)
            owned_buffers.extend(
                [cuda_grad_inputs, cuda_grad_weights, cuda_grad_bias]
            )
            max_error = max(
                max_error,
                self._assert_vector_close(
                    cast(list[float], self.backend.buffer_to_host(cuda_grad_inputs)),
                    cpu_grad_inputs,
                ),
                self._assert_vector_close(
                    [
                        value
                        for row in cast(
                            list[list[float]], self.backend.buffer_to_host(cuda_grad_weights)
                        )
                        for value in row
                    ],
                    [value for row in cpu_grad_weights for value in row],
                ),
                self._assert_vector_close(
                    cast(list[float], self.backend.buffer_to_host(cuda_grad_bias)),
                    cpu_grad_bias,
                ),
            )

            with (
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=AssertionError("implicit H2D copy during integrated step"),
                ) as h2d,
                patch.object(
                    self.backend.memory,
                    "copy_to_host",
                    side_effect=AssertionError("implicit D2H copy during integrated step"),
                ) as d2h,
            ):
                cuda_layer.step(learning_rate)
                h2d.assert_not_called()
                d2h.assert_not_called()
            cpu_layer.step(learning_rate)

            self.assertIs(cuda_layer.weights, device_weights)
            self.assertIs(cuda_layer.bias, device_bias)
            self.assertFalse(cuda_layer._has_backward)
            max_error = max(
                max_error,
                self._assert_vector_close(
                    [
                        value
                        for row in cast(
                            list[list[float]], self.backend.buffer_to_host(device_weights)
                        )
                        for value in row
                    ],
                    [value for row in cast(list[list[float]], cpu_layer.weights) for value in row],
                ),
                self._assert_vector_close(
                    cast(list[float], self.backend.buffer_to_host(device_bias)),
                    cast(list[float], cpu_layer.bias),
                ),
            )
            print("LINEAR_INTEGRATED_MAX_ABS_ERROR", max_error)

        self.assertTrue(all(buffer.closed for buffer in owned_buffers))
        with self.assertRaises(BufferReleasedError):
            _ = device_weights.pointer


if __name__ == "__main__":
    unittest.main()
