"""Opt-in integrated Embedding forward/backward/step gate."""

import os
import unittest
from typing import cast
from unittest.mock import patch

from kinera.backends.base import BackendArray, BackendError, BackendUnavailableError
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer, CudaRuntime
from kinera.backends.cuda.bridge import CudaBridge
from kinera.backends.cuda.buffer import BufferReleasedError
from kinera.nn.layers.embedding import Embedding


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


class UnavailableCudaBridge(CudaBridge):
    def device_count(self) -> int:
        raise BackendUnavailableError("simulated missing CUDA runtime")


class EmbeddingCudaIntegratedUnavailableTests(unittest.TestCase):
    def test_cuda_embedding_cycle_fails_closed_without_runtime(self) -> None:
        backend = CUDABackend(runtime=CudaRuntime(bridge=UnavailableCudaBridge()))

        with self.assertRaisesRegex(
            BackendUnavailableError, "simulated missing CUDA runtime"
        ):
            Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=backend)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class EmbeddingCudaIntegratedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_matrix_close(
        self, actual: list[list[float]], expected: list[list[float]]
    ) -> float:
        self.assertEqual(len(actual), len(expected))
        errors: list[float] = []
        for row, expected_row in enumerate(expected):
            self.assertEqual(len(actual[row]), len(expected_row))
            for column, expected_value in enumerate(expected_row):
                error = abs(actual[row][column] - expected_value)
                errors.append(error)
                with self.subTest(row=row, column=column):
                    self.assertLessEqual(
                        error,
                        ATOL + RTOL * abs(expected_value),
                        msg=(
                            f"CUDA={actual[row][column]}, CPU={expected_value}, "
                            f"abs_error={error}"
                        ),
                    )
        return max(errors, default=0.0)

    def test_forward_backward_step_matches_cpu_and_releases_every_buffer(self) -> None:
        vocab_size = 4
        embedding_dim = 3
        initial_weights = [
            [0.125, -0.25, 0.5],
            [0.75, 0.125, -0.5],
            [-0.375, 0.25, 0.625],
            [0.1, -0.2, 0.3],
        ]
        ids = [2, 0, 2, 3]
        grad_output = [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.5],
            [-0.25, 2.0, 0.75],
            [0.1, -0.2, 0.3],
        ]
        learning_rate = 0.037
        cpu_layer = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            seed=73,
            backend=self.cpu,
        )
        setattr(cpu_layer, "weights", [row.copy() for row in initial_weights])
        cuda_layer = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            seed=73,
            backend=self.backend,
        )
        initial_cuda_weights = cast(CudaBuffer, cuda_layer.weights)
        initial_cuda_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
        initial_cuda_weights.close()
        initial_cuda_gradients.close()
        cuda_weights = self.backend.buffer_from_host(
            initial_weights,
            shape=(vocab_size, embedding_dim),
            dtype="float64",
        )
        cuda_layer.weights = cuda_weights
        cuda_layer.grad_weights = self.backend.buffer_from_host(
            [[0.0] * embedding_dim for _ in range(vocab_size)],
            shape=(vocab_size, embedding_dim),
            dtype="float64",
        )
        initial_device_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
        transfers: list[CudaBuffer] = []
        outputs: list[CudaBuffer] = []
        device_ids: list[CudaBuffer] = []
        cuda_output: CudaBuffer | None = None
        cuda_gradients: CudaBuffer | None = None
        max_error = 0.0

        original_copy_to_device = self.backend.memory.copy_to_device
        original_copy_zeros_to_device = self.backend.memory.copy_zeros_to_device

        def record_transfer(*args, **kwargs) -> CudaBuffer:
            buffer = original_copy_to_device(*args, **kwargs)
            transfers.append(buffer)
            return buffer

        def record_zero_transfer(*args, **kwargs) -> CudaBuffer:
            buffer = original_copy_zeros_to_device(*args, **kwargs)
            transfers.append(buffer)
            return buffer

        try:
            with (
                patch.object(self.backend.memory, "copy_to_device", side_effect=record_transfer),
                patch.object(
                    self.backend.memory,
                    "copy_zeros_to_device",
                    side_effect=record_zero_transfer,
                ),
            ):
                cpu_output = cast(list[list[float]], cpu_layer.forward(ids.copy()))
                cuda_output = cast(CudaBuffer, cuda_layer.forward(ids.copy()))
                outputs.append(cuda_output)
                ids_buffer = cast(CudaBuffer, cuda_layer.last_ids)
                device_ids.append(ids_buffer)
                gradient_buffer = cast(CudaBuffer, cuda_layer.grad_weights)
                self.assertEqual(ids_buffer.dtype, "int32")
                self.assertEqual(ids_buffer.shape, (len(ids),))
                self.assertEqual(gradient_buffer.shape, (vocab_size, embedding_dim))

                cpu_layer.backward(grad_output)
                cuda_layer.backward(grad_output)
                cuda_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
                self.assertIs(cuda_layer.grad_weights, gradient_buffer)
                self.assertEqual(len(transfers), 3)
                self.assertTrue(transfers[-1].closed, "temporary grad_output buffer leaked")

                with (
                    patch.object(
                        self.backend.memory,
                        "copy_to_device",
                        side_effect=AssertionError("implicit H2D transfer during Embedding.step"),
                    ) as h2d,
                    patch.object(
                        self.backend.memory,
                        "copy_to_host",
                        side_effect=AssertionError("implicit D2H transfer during Embedding.step"),
                    ) as d2h,
                ):
                    cpu_layer.step(learning_rate)
                    cuda_layer.step(learning_rate)
                    h2d.assert_not_called()
                    d2h.assert_not_called()

            cpu_gradients = cast(list[list[float]], cpu_layer.grad_weights)
            cuda_gradients_host = cast(
                list[list[float]], self.backend.buffer_to_host(cuda_gradients)
            )
            cpu_weights = cast(list[list[float]], cpu_layer.weights)
            cuda_weights_host = cast(
                list[list[float]], self.backend.buffer_to_host(cuda_weights)
            )
            cuda_output_host = cast(
                list[list[float]], self.backend.buffer_to_host(cuda_output)
            )
            max_error = max(
                self._assert_matrix_close(cuda_output_host, cpu_output),
                self._assert_matrix_close(cuda_gradients_host, cpu_gradients),
                self._assert_matrix_close(cuda_weights_host, cpu_weights),
            )
            print("EMBEDDING_INTEGRATED_MAX_ABS_ERROR", max_error)

            cuda_weights.close()
            ids_buffer.close()
            self.assertEqual(self.backend.buffer_to_host(cuda_output), cuda_output_host)
            cuda_gradients.close()
            cuda_output.close()
            with self.assertRaises(BufferReleasedError):
                self.backend.buffer_to_host(cuda_output)
        finally:
            buffers = [
                initial_cuda_weights,
                initial_cuda_gradients,
                cuda_weights,
                initial_device_gradients,
                cuda_gradients,
                *transfers,
                *outputs,
                *device_ids,
            ]
            if cuda_output is not None:
                buffers.append(cuda_output)
            for buffer in buffers:
                if not buffer.closed:
                    buffer.close()
        self.assertTrue(all(buffer.closed for buffer in buffers))
        self.assertTrue(cuda_layer._has_backward is False)
        self.assertTrue(max_error <= ATOL)


if __name__ == "__main__":
    unittest.main()
