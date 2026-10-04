"""Opt-in GPU equivalence tests for Embedding.step/update."""

import os
import unittest
from typing import cast

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.layers.embedding import Embedding


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12
IDS = [2, 0, 2, 3]
GRAD_OUTPUT = [
    [0.5, -1.0, 2.0],
    [1.5, 0.25, -0.5],
    [-0.25, 2.0, 0.75],
    [0.1, -0.2, 0.3],
]
LEARNING_RATE = 0.037


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class EmbeddingCudaStepEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _run_updates(self, update_count: int) -> tuple[float, float]:
        cpu_layer = Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=self.cpu)
        cuda_layer = Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=self.backend)
        cuda_weights = cast(CudaBuffer, cuda_layer.weights)
        max_errors: list[float] = []
        try:
            self.assertEqual(self.backend.buffer_to_host(cuda_weights), cpu_layer.weights)
            for step_index in range(update_count):
                cpu_layer.forward(IDS.copy())
                cuda_layer.forward(IDS.copy())
                cpu_layer.backward(GRAD_OUTPUT)
                cuda_layer.backward(GRAD_OUTPUT)

                self.assertIs(cuda_layer.weights, cuda_weights)
                pointer_before = cuda_weights.pointer
                cpu_layer.step(LEARNING_RATE)
                cuda_layer.step(LEARNING_RATE)
                self.assertIs(cuda_layer.weights, cuda_weights)
                self.assertEqual(cuda_weights.pointer, pointer_before)

                actual = cast(list[list[float]], self.backend.buffer_to_host(cuda_weights))
                expected = cast(list[list[float]], cpu_layer.weights)
                self.assertEqual(len(actual), len(expected))
                step_max_error = 0.0
                for row, expected_row in enumerate(expected):
                    self.assertEqual(len(actual[row]), len(expected_row))
                    for column, expected_value in enumerate(expected_row):
                        error = abs(actual[row][column] - expected_value)
                        step_max_error = max(step_max_error, error)
                        with self.subTest(step=step_index + 1, row=row, column=column):
                            self.assertLessEqual(
                                error,
                                ATOL + RTOL * abs(expected_value),
                                msg=f"CUDA={actual[row][column]}, CPU={expected_value}",
                            )
                max_errors.append(step_max_error)
            print("EMBEDDING_STEP_MAX_ABS_ERROR_AFTER_1", max_errors[0])
            print(
                f"EMBEDDING_STEP_MAX_ABS_ERROR_AFTER_{update_count}",
                max_errors[-1],
            )
            return max_errors[0], max_errors[-1]
        finally:
            for value in (
                cuda_layer.weights,
                cuda_layer.grad_weights,
                cuda_layer.last_ids,
                cuda_layer.output,
            ):
                if isinstance(value, CudaBuffer) and not value.closed:
                    value.close()

    def test_cpu_cuda_weights_match_after_one_and_eight_updates(self) -> None:
        self._run_updates(1)
        self._run_updates(8)

    def test_zero_learning_rate_preserves_exact_weights(self) -> None:
        cpu_layer = Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=self.cpu)
        cuda_layer = Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=self.backend)
        cuda_weights = cast(CudaBuffer, cuda_layer.weights)
        try:
            cpu_layer.forward(IDS.copy())
            cuda_layer.forward(IDS.copy())
            cpu_layer.backward(GRAD_OUTPUT)
            cuda_layer.backward(GRAD_OUTPUT)
            before = cast(list[list[float]], self.backend.buffer_to_host(cuda_weights))
            pointer_before = cuda_weights.pointer

            cpu_layer.step(0.0)
            cuda_layer.step(0.0)

            self.assertIs(cuda_layer.weights, cuda_weights)
            self.assertEqual(cuda_weights.pointer, pointer_before)
            self.assertEqual(self.backend.buffer_to_host(cuda_weights), before)
            self.assertEqual(cpu_layer.weights, before)
            self.assertFalse(cuda_layer._has_backward)
        finally:
            for value in (
                cuda_layer.weights,
                cuda_layer.grad_weights,
                cuda_layer.last_ids,
                cuda_layer.output,
            ):
                if isinstance(value, CudaBuffer) and not value.closed:
                    value.close()


if __name__ == "__main__":
    unittest.main()
