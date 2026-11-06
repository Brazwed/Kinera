"""Opt-in RX 550 equivalence gate for OpenCL Embedding.backward (F1b)."""

import ctypes
import math
import os
import sys
import unittest

from kinera.backends import get_backend
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.nn.layers.embedding import Embedding


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on RX 550/Rusticl with FP64 enabled",
)
class EmbeddingOpenCLBackwardRX550Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = get_backend("opencl")
        try:
            cls.device = cls.backend.runtime.require_device(0, require_fp64=True)
            if cls.device.pci_bdf != "0000:03:00.0":
                raise AssertionError(
                    f"expected RX 550 at 0000:03:00.0, got {cls.device.pci_bdf}"
                )
            if "Radeon RX 550" not in cls.device.name:
                raise AssertionError(f"expected RX 550, got {cls.device.name}")
        except Exception:
            cls.backend.close()
            raise
        cls.cpu = CPUBackend()
        print(
            "OPENCL_EMBEDDING_F1B_DEVICE",
            cls.device.name,
            "PCI",
            cls.device.pci_bdf,
            "DRIVER",
            cls.device.driver_version,
            "RUNTIME",
            cls.device.version,
            "OPENCL_C",
            cls.device.opencl_c_version,
            "PYTHON",
            sys.version.split()[0],
            "FP64",
            cls.device.supports_fp64,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_ordered_duplicates_signed_values_and_cancellation_match_cpu(self) -> None:
        cases = (
            (
                [[1e16, -2.0], [3.0, 4.0], [5.0, -6.0], [7.0, 8.0]],
                [0, 0, 0, 2, 0, 2],
                [[1.0, 1.0], [1.0, -2.0], [-1e16, 3.0], [0.0, 0.0], [1.0, -1.0], [-5.0, 2.0]],
            ),
            ([[1e308]], [0, 0], [[-1e308], [1e308]]),
            ([[2.5]], [0, 0, 0], [[0.0], [-4.0], [1.5]]),
            (
                [[0.0, -1.0, 2.0], [3.5, 0.0, -4.5], [6.0, 7.0, 8.0]],
                [2, 0, 1, 2],
                [[1.25, -0.5, 2.0], [-3.0, 4.0, 0.0], [0.25, 0.75, -1.5], [-1.25, 0.5, -2.0]],
            ),
        )
        for case_index, (initial, ids_values, gradients) in enumerate(cases):
            with self.subTest(case=case_index):
                self._assert_backend_case(initial, ids_values, gradients)

    def test_consecutive_backward_calls_match_cpu_and_preserve_unvisited_rows(self) -> None:
        initial = [[0.25, -0.5], [3.0, 4.0], [1e16, -8.0], [-2.0, 1.5]]
        weights = self.backend.buffer_from_host(initial, shape=(4, 2))
        ids = self.backend.buffer_from_host([2, 2, 0], shape=(3,), dtype="int32")
        gradients = self.backend.buffer_from_host(
            [[1.0, 2.0], [1.0, -3.0], [-4.0, 5.0]], shape=(3, 2)
        )
        try:
            expected = self.cpu.embedding_backward(
                initial, [2, 2, 0], [[1.0, 2.0], [1.0, -3.0], [-4.0, 5.0]]
            )
            self.assertIs(self.backend.embedding_backward(weights, ids, gradients), weights)
            self._assert_values(self.backend.buffer_to_host(weights), expected)
            self.assertEqual(self.backend.buffer_to_host(weights)[1], initial[1])

            next_ids = self.backend.buffer_from_host([2, 0], shape=(2,), dtype="int32")
            next_gradients_values = [[-1.0, 4.0], [2.0, -5.0]]
            next_gradients = self.backend.buffer_from_host(
                next_gradients_values, shape=(2, 2)
            )
            try:
                expected = self.cpu.embedding_backward(
                    expected, [2, 0], next_gradients_values
                )
                self.assertIs(
                    self.backend.embedding_backward(weights, next_ids, next_gradients),
                    weights,
                )
                self._assert_values(self.backend.buffer_to_host(weights), expected)
            finally:
                next_gradients.close()
                next_ids.close()
        finally:
            gradients.close()
            ids.close()
            weights.close()

    def test_layer_host_and_resident_gradients_and_new_forward_reset(self) -> None:
        cpu_layer = Embedding(4, 3, seed=113, backend=self.cpu)
        gpu_layer = Embedding(4, 3, seed=113, backend=self.backend)
        resident_gradients = None
        try:
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
            )
            ids = [3, 1, 3, 0]
            cpu_layer.forward(ids.copy())
            gpu_output = gpu_layer.forward(ids.copy())
            self.assertIsInstance(gpu_output, OpenCLBuffer)
            host_gradients = [
                [1.0, -2.0, 0.5],
                [-1.0, 3.0, 2.0],
                [4.0, 0.25, -0.5],
                [0.0, -1.0, 1.0],
            ]
            cpu_layer.backward(host_gradients)
            gpu_layer.backward(host_gradients)
            self._assert_values(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                cpu_layer.grad_weights,
            )

            resident_gradients_values = [
                [-0.5, 1.0, 2.0],
                [3.5, -4.0, 0.0],
                [-1.5, 2.5, -3.0],
                [4.0, -2.0, 1.0],
            ]
            resident_gradients = self.backend.buffer_from_host(
                resident_gradients_values, shape=(len(ids), 3)
            )
            cpu_layer.backward(resident_gradients_values)
            gpu_layer.backward(resident_gradients)
            self.assertFalse(resident_gradients.closed)
            self._assert_values(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                cpu_layer.grad_weights,
            )

            cpu_layer.forward([2])
            gpu_layer.forward([2])
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                [[0.0, 0.0, 0.0] for _ in range(4)],
                "a new forward resets the full gradient table",
            )
            self.assertEqual(gpu_layer.weights.to_host(), cpu_layer.weights)
        finally:
            if resident_gradients is not None:
                resident_gradients.close()
            gpu_layer.close()
            cpu_layer.close()

    def test_empty_layer_backward_is_noop_without_resident_zero_buffers(self) -> None:
        cpu_layer = Embedding(2, 2, seed=5, backend=self.cpu)
        gpu_layer = Embedding(2, 2, seed=5, backend=self.backend)
        try:
            cpu_layer.forward([])
            gpu_layer.forward([])
            grad_weights = gpu_layer.grad_weights
            gpu_layer.backward([])
            cpu_layer.backward([])
            self.assertIs(gpu_layer.grad_weights, grad_weights)
            self.assertEqual(self.backend.buffer_to_host(grad_weights), cpu_layer.grad_weights)
            self.assertTrue(gpu_layer._has_backward)
        finally:
            gpu_layer.close()
            cpu_layer.close()

    def test_invalid_resident_ids_and_nonfinite_gradients_do_not_mutate(self) -> None:
        initial = [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]]
        gradients_values = [[0.25, -0.5], [1.5, 2.0]]
        before = [row.copy() for row in initial]
        weights = self.backend.buffer_from_host(initial, shape=(3, 2))
        gradients = self.backend.buffer_from_host(gradients_values, shape=(2, 2))
        try:
            for invalid_id in (-1, 3, -(2**31), 2**31 - 1):
                ids = self.backend.buffer_from_host([0, invalid_id], shape=(2,), dtype="int32")
                try:
                    with self.subTest(invalid_id=invalid_id):
                        with self.assertRaisesRegex(ValueError, "outside"):
                            self.backend.embedding_backward(weights, ids, gradients)
                        self.assertEqual(self.backend.buffer_to_host(weights), before)
                finally:
                    ids.close()

            ids = self.backend.buffer_from_host([0, 2], shape=(2,), dtype="int32")
            try:
                for invalid_value in (float("nan"), float("inf"), float("-inf")):
                    host_values = (ctypes.c_double * 4)(
                        invalid_value, gradients_values[0][1],
                        gradients_values[1][0], gradients_values[1][1],
                    )
                    # Controlled test-only mutation bypasses host finite validation.
                    self.backend.runtime.api.write_buffer(
                        self.backend.memory.queue,
                        gradients.handle,
                        ctypes.addressof(host_values),
                        ctypes.sizeof(host_values),
                    )
                    with self.subTest(nonfinite=invalid_value):
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            self.backend.embedding_backward(weights, ids, gradients)
                        self.assertEqual(self.backend.buffer_to_host(weights), before)
            finally:
                ids.close()
        finally:
            gradients.close()
            weights.close()

    def test_layer_resident_failure_preserves_state_and_caller_ownership(self) -> None:
        layer = Embedding(3, 2, seed=31, backend=self.backend)
        try:
            layer.forward([0, 2])
            grad_weights = layer.grad_weights
            ids = layer.last_ids
            ids_values = (ctypes.c_int32 * 2)(-1, 2)
            self.backend.runtime.api.write_buffer(
                self.backend.memory.queue,
                ids.handle,
                ctypes.addressof(ids_values),
                ctypes.sizeof(ids_values),
            )
            resident = self.backend.buffer_from_host(
                [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2)
            )
            before = self.backend.buffer_to_host(grad_weights)
            try:
                with self.assertRaisesRegex(ValueError, "outside"):
                    layer.backward(resident)
                self.assertIs(layer.grad_weights, grad_weights)
                self.assertFalse(layer._has_backward)
                self.assertFalse(resident.closed)
                self.assertEqual(self.backend.buffer_to_host(grad_weights), before)
            finally:
                resident.close()
        finally:
            layer.close()

    def _assert_backend_case(self, initial, ids_values, gradients):
        vocab_size = len(initial)
        embedding_dim = len(initial[0])
        expected = self.cpu.embedding_backward(initial, ids_values, gradients)
        weights = self.backend.buffer_from_host(
            initial, shape=(vocab_size, embedding_dim)
        )
        ids = self.backend.buffer_from_host(
            ids_values, shape=(len(ids_values),), dtype="int32"
        )
        grad_output = self.backend.buffer_from_host(
            gradients, shape=(len(gradients), embedding_dim)
        )
        try:
            result = self.backend.embedding_backward(weights, ids, grad_output)
            self.assertIs(result, weights)
            actual = self.backend.buffer_to_host(result)
            self._assert_values(actual, expected)
        finally:
            grad_output.close()
            ids.close()
            weights.close()

    def _assert_values(self, actual, expected) -> None:
        actual_flat = [value for row in actual for value in row]
        expected_flat = [value for row in expected for value in row]
        self.assertEqual(len(actual_flat), len(expected_flat))
        errors = [abs(got - want) for got, want in zip(actual_flat, expected_flat)]
        for index, (got, want) in enumerate(zip(actual_flat, expected_flat)):
            with self.subTest(element=index):
                self.assertLessEqual(
                    abs(got - want), ATOL + RTOL * abs(want),
                    msg=f"OpenCL={got}, CPU={want}",
                )
        max_error = max(errors, default=0.0)
        print("OPENCL_EMBEDDING_F1B_MAX_ABS_ERROR", max_error)
        self.assertTrue(all(math.isfinite(value) for value in actual_flat))


if __name__ == "__main__":
    unittest.main()
