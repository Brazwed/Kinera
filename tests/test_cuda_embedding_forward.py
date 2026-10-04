"""Opt-in Tesla/GPU equivalence tests for Embedding.forward."""

import os
import unittest
from typing import cast

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.layers.embedding import Embedding


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class EmbeddingCudaForwardEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_equivalent(self, vocab_size: int, embedding_dim: int, ids: list[int]) -> None:
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
        cuda_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
        output: CudaBuffer | None = None
        device_ids: CudaBuffer | None = None
        gradients_to_close = [cuda_gradients]
        ids_before = ids.copy()
        try:
            host_weights = self.backend.buffer_to_host(cuda_weights)
            self.assertEqual(host_weights, cpu_layer.weights)
            expected = cast(list[list[float]], cpu_layer.forward(ids_before))

            result = cuda_layer.forward(ids)
            current_gradients = cast(CudaBuffer, cuda_layer.grad_weights)
            gradients_to_close.append(current_gradients)
            self.assertEqual(current_gradients.shape, (vocab_size, embedding_dim))
            if not ids_before:
                self.assertEqual(expected, [])
                self.assertEqual(result, [])
                self.assertEqual(cuda_layer.last_ids, [])
                return

            self.assertIsInstance(result, CudaBuffer)
            output = cast(CudaBuffer, result)
            self.assertEqual(output.shape, (len(ids_before), embedding_dim))
            self.assertEqual(output.dtype, "float64")
            self.assertNotEqual(output.pointer, cuda_weights.pointer)
            self.assertIsInstance(cuda_layer.last_ids, CudaBuffer)
            device_ids = cast(CudaBuffer, cuda_layer.last_ids)
            self.assertEqual(device_ids.dtype, "int32")
            self.assertEqual(self.backend.buffer_to_host(device_ids), ids_before)

            ids[0] = (ids[0] + 1) % vocab_size
            self.assertEqual(self.backend.buffer_to_host(device_ids), ids_before)
            device_ids.close()
            actual = cast(list[list[float]], self.backend.buffer_to_host(output))
            self.assertEqual(len(actual), len(expected))
            errors = [
                abs(actual[row][column] - expected[row][column])
                for row in range(len(expected))
                for column in range(embedding_dim)
            ]
            for row, expected_row in enumerate(expected):
                for column, expected_value in enumerate(expected_row):
                    error = abs(actual[row][column] - expected_value)
                    with self.subTest(ids=ids_before, row=row, column=column):
                        self.assertLessEqual(
                            error,
                            ATOL + RTOL * abs(expected_value),
                            msg=(
                                f"CUDA={actual[row][column]}, CPU={expected_value}, "
                                f"abs_error={error}"
                            ),
                        )
            self.assertTrue(all(value == value for row in actual for value in row))
            self.assertTrue(
                all(abs(value) != float("inf") for row in actual for value in row)
            )
            print("EMBEDDING_FORWARD_MAX_ABS_ERROR", max(errors, default=0.0))

            cuda_weights.close()
            self.assertEqual(self.backend.buffer_to_host(output), expected)
        finally:
            if output is not None and not output.closed:
                output.close()
            if device_ids is not None and not device_ids.closed:
                device_ids.close()
            for gradients in gradients_to_close:
                if not gradients.closed:
                    gradients.close()
            if not cuda_weights.closed:
                cuda_weights.close()

    def test_duplicate_ids_and_seeded_weights_match_cpu(self) -> None:
        self._assert_equivalent(
            vocab_size=4,
            embedding_dim=3,
            ids=[2, 0, 2, 3],
        )

    def test_resident_out_of_range_ids_are_rejected_before_forward_lookup(self) -> None:
        vocab_size = 4
        embedding_dim = 2
        weights = self.backend.buffer_from_host(
            [[0.25, -0.5], [1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]],
            shape=(vocab_size, embedding_dim),
            dtype="float64",
        )
        invalid_ids = (-1, vocab_size, -(2**31), 2**31 - 1)
        allocated_ids: list[CudaBuffer] = []
        output: CudaBuffer | None = None
        try:
            for invalid_id in invalid_ids:
                with self.subTest(id=invalid_id):
                    device_ids = self.backend.buffer_from_host(
                        [invalid_id], shape=(1,), dtype="int32"
                    )
                    allocated_ids.append(device_ids)
                    with self.assertRaisesRegex(ValueError, "vocabulary"):
                        self.backend.embedding_forward(weights, device_ids)
                    device_ids.close()

            valid_ids = self.backend.buffer_from_host([3], shape=(1,), dtype="int32")
            allocated_ids.append(valid_ids)
            output = cast(CudaBuffer, self.backend.embedding_forward(weights, valid_ids))
            self.assertEqual(self.backend.buffer_to_host(output), [[5.0, -6.0]])
        finally:
            if output is not None and not output.closed:
                output.close()
            for device_ids in allocated_ids:
                if not device_ids.closed:
                    device_ids.close()
            if not weights.closed:
                weights.close()

    def test_single_row_and_empty_id_edge_cases_match_cpu(self) -> None:
        self._assert_equivalent(vocab_size=1, embedding_dim=1, ids=[0, 0, 0])
        self._assert_equivalent(vocab_size=1, embedding_dim=1, ids=[])


if __name__ == "__main__":
    unittest.main()
