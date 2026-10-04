"""Opt-in GPU equivalence and finite-difference tests for Embedding.backward."""

import os
import unittest
from typing import Sequence, cast

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.diagnostics.gradients import compare_gradients, finite_difference_gradient
from kinera.nn.layers.embedding import Embedding


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class EmbeddingCudaBackwardEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def test_ordered_duplicate_accumulation_matches_cpu_and_finite_differences(self) -> None:
        vocab_size = 4
        embedding_dim = 3
        ids = [2, 0, 2, 3]
        grad_output = [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.5],
            [-0.25, 2.0, 0.75],
            [0.1, -0.2, 0.3],
        ]
        cpu_layer = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            seed=73,
            backend=self.cpu,
        )
        cuda_layer = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            seed=73,
            backend=self.backend,
        )
        cuda_weights = cast(CudaBuffer, cuda_layer.weights)
        initial_cuda_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
        cuda_outputs: list[CudaBuffer] = []
        cuda_ids: list[CudaBuffer] = []
        cuda_gradients: list[CudaBuffer] = [initial_cuda_gradients]

        try:
            cpu_weights = cast(list[list[float]], cpu_layer.weights)
            self.assertEqual(self.backend.buffer_to_host(cuda_weights), cpu_weights)

            cpu_output = cast(list[list[float]], cpu_layer.forward(ids.copy()))
            first_cuda_output = cast(CudaBuffer, cuda_layer.forward(ids.copy()))
            cuda_outputs.append(first_cuda_output)
            first_ids = cast(CudaBuffer, cuda_layer.last_ids)
            cuda_ids.append(first_ids)
            self.assertEqual(first_cuda_output.shape, (len(ids), embedding_dim))
            self.assertEqual(first_ids.dtype, "int32")
            self.assertEqual(self.backend.buffer_to_host(first_cuda_output), cpu_output)

            cpu_layer.backward(grad_output)
            cuda_layer.backward(grad_output)
            first_cuda_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
            cuda_gradients.append(first_cuda_gradients)
            cpu_gradient_table = cast(list[list[float]], cpu_layer.grad_weights)
            first_device_table = cast(
                list[list[float]], self.backend.buffer_to_host(first_cuda_gradients)
            )
            first_table_error = self._assert_tables_equivalent(
                first_device_table, cpu_gradient_table
            )
            self.assertEqual(first_device_table[1], [0.0] * embedding_dim)
            first_snapshot = [row.copy() for row in first_device_table]

            selected = ((2, 1), (0, 2), (1, 0))
            initial_values = [cpu_weights[row][column] for row, column in selected]

            def cuda_objective(selected_values: Sequence[float]) -> float:
                perturbed = [row.copy() for row in cpu_weights]
                for (row, column), value in zip(selected, selected_values, strict=True):
                    perturbed[row][column] = float(value)
                weights_buffer = self.backend.buffer_from_host(
                    perturbed,
                    shape=(vocab_size, embedding_dim),
                    dtype="float64",
                )
                output_buffer = None
                try:
                    output_buffer = cast(
                        CudaBuffer,
                        self.backend.embedding_forward(weights_buffer, first_ids),
                    )
                    output = cast(
                        list[list[float]], self.backend.buffer_to_host(output_buffer)
                    )
                    return sum(
                        output[position][dimension] * grad_output[position][dimension]
                        for position in range(len(ids))
                        for dimension in range(embedding_dim)
                    )
                finally:
                    if output_buffer is not None and not output_buffer.closed:
                        output_buffer.close()
                    weights_buffer.close()

            numeric = finite_difference_gradient(
                cuda_objective, initial_values, epsilon=1e-6
            )
            analytic = [first_device_table[row][column] for row, column in selected]
            finite_difference = compare_gradients(
                analytic,
                numeric,
                absolute_tolerance=1e-8,
                relative_tolerance=1e-5,
            )
            self.assertTrue(finite_difference.passed, finite_difference)
            print("EMBEDDING_BACKWARD_FD_MAX_ABS_ERROR", finite_difference.max_absolute_error)

            cpu_layer.backward(grad_output)
            cuda_layer.backward(grad_output)
            double_device_table = cast(
                list[list[float]],
                self.backend.buffer_to_host(cast(CudaBuffer, cuda_layer.grad_weights)),
            )
            double_table_error = self._assert_tables_equivalent(
                double_device_table,
                cast(list[list[float]], cpu_layer.grad_weights),
            )

            cpu_layer.forward(ids.copy())
            reset_output = cast(CudaBuffer, cuda_layer.forward(ids.copy()))
            cuda_outputs.append(reset_output)
            reset_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
            cuda_gradients.append(reset_gradients)
            self.assertEqual(
                self.backend.buffer_to_host(reset_gradients),
                [[0.0] * embedding_dim for _ in range(vocab_size)],
            )
            reset_ids = cast(CudaBuffer, cuda_layer.last_ids)
            cuda_ids.append(reset_ids)
            cpu_layer.backward(grad_output)
            cuda_layer.backward(grad_output)
            repeated_table = cast(
                list[list[float]],
                self.backend.buffer_to_host(cast(CudaBuffer, cuda_layer.grad_weights)),
            )
            repeat_table_error = self._assert_tables_equivalent(
                repeated_table,
                cast(list[list[float]], cpu_layer.grad_weights),
            )
            self.assertEqual(repeated_table, first_snapshot)
            print(
                "EMBEDDING_BACKWARD_MAX_ABS_ERROR",
                max(first_table_error, double_table_error, repeat_table_error),
            )
        finally:
            for buffer in [*cuda_outputs, *cuda_ids, *cuda_gradients, cuda_weights]:
                if not buffer.closed:
                    buffer.close()

    def _assert_tables_equivalent(
        self, actual: list[list[float]], expected: list[list[float]]
    ) -> float:
        self.assertEqual(len(actual), len(expected))
        errors = []
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


if __name__ == "__main__":
    unittest.main()
