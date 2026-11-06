"""Opt-in RX 550 equivalence gates for OpenCL Embedding.forward (F1a)."""

import math
import os
import sys
import unittest

from kinera.backends import get_backend
from kinera.backends.base import BackendError
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.nn.layers.embedding import Embedding


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 with RX 550/Rusticl FP64 enabled",
)
class EmbeddingOpenCLForwardRX550Tests(unittest.TestCase):
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
            "OPENCL_EMBEDDING_F1A_DEVICE",
            cls.device.name,
            cls.device.pci_bdf,
            "DRIVER",
            cls.device.driver_version,
            "RUNTIME",
            cls.device.version,
            "PYTHON",
            sys.version.split()[0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def _assert_seeded_layer_equivalence(
        self, vocab_size: int, embedding_dim: int, ids: list[int]
    ) -> None:
        cpu_layer = Embedding(vocab_size, embedding_dim, seed=73, backend=self.cpu)
        gpu_layer = Embedding(vocab_size, embedding_dim, seed=73, backend=self.backend)
        output: OpenCLBuffer | None = None
        try:
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
            )
            expected = cpu_layer.forward(ids.copy())
            result = gpu_layer.forward(ids.copy())
            if not ids:
                self.assertEqual(result, [])
                self.assertEqual(expected, [])
                self.assertEqual(gpu_layer.last_ids, [])
                self.assertEqual(gpu_layer.output, [])
                return

            self.assertIsInstance(result, OpenCLBuffer)
            output = result
            self.assertEqual(output.shape, (len(ids), embedding_dim))
            self.assertEqual(output.dtype, "float64")
            self.assertIs(output._manager, self.backend.memory)
            self.assertIsInstance(gpu_layer.last_ids, OpenCLBuffer)
            self.assertEqual(gpu_layer.last_ids.dtype, "int32")
            actual = self.backend.buffer_to_host(output)
            self.assertEqual(actual, expected)
            self.assertEqual(
                max(
                    (
                        abs(actual[row][column] - expected[row][column])
                        for row in range(len(expected))
                        for column in range(embedding_dim)
                    ),
                    default=0.0,
                ),
                0.0,
            )
            print("OPENCL_EMBEDDING_F1A_MAX_ABS_ERROR", 0.0)
        finally:
            if output is not None and not output.closed:
                output.close()
            gpu_layer.close()
            cpu_layer.close()

    def test_seeded_lookup_preserves_order_and_repeated_ids(self) -> None:
        self._assert_seeded_layer_equivalence(5, 4, [3, 0, 3, 2, 4, 0])

    def test_vocabulary_size_one_and_empty_ids(self) -> None:
        self._assert_seeded_layer_equivalence(1, 1, [0, 0, 0])
        self._assert_seeded_layer_equivalence(1, 1, [])

    def test_resident_invalid_ids_are_rejected_before_output_allocation(self) -> None:
        vocab_size, embedding_dim = 3, 2
        weights = self.backend.buffer_from_host(
            [[0.25, -0.5], [1.0, 2.0], [-3.0, 4.0]],
            shape=(vocab_size, embedding_dim),
            dtype="float64",
        )
        ids = []
        before_live = set(self.backend.memory._buffers)
        try:
            for invalid_id in (-1, vocab_size, -(2**31), 2**31 - 1):
                device_ids = self.backend.buffer_from_host(
                    [invalid_id], shape=(1,), dtype="int32"
                )
                ids.append(device_ids)
                with self.subTest(invalid_id=invalid_id):
                    with self.assertRaisesRegex(ValueError, "position 0"):
                        self.backend.embedding_forward(weights, device_ids)
                    self.assertFalse(device_ids.closed)
                    self.assertFalse(weights.closed)
                    self.assertEqual(
                        set(self.backend.memory._buffers),
                        before_live | {weights, *ids},
                    )
        finally:
            for device_ids in ids:
                device_ids.close()
            weights.close()

    def test_output_ownership_and_explicit_close(self) -> None:
        weights = self.backend.buffer_from_host(
            [[-0.0, 0.0], [2.5, -4.0]], shape=(2, 2), dtype="float64"
        )
        ids = self.backend.buffer_from_host([1, 0, 1], shape=(3,), dtype="int32")
        output = None
        try:
            output = self.backend.embedding_forward(weights, ids)
            self.assertIsInstance(output, OpenCLBuffer)
            handle = output.handle
            self.assertIs(output._manager, self.backend.memory)
            self.assertTrue(output.owns_memory)
            self.assertFalse(weights.closed)
            self.assertFalse(ids.closed)
            actual = self.backend.buffer_to_host(output)
            self.assertEqual(actual, [[2.5, -4.0], [-0.0, 0.0], [2.5, -4.0]])
            self.assertEqual(math.copysign(1.0, actual[1][0]), -1.0)
            output.close()
            output.close()
            self.assertTrue(output.closed)
            self.assertNotIn(output, self.backend.memory._buffers)
            self.assertNotEqual(handle, weights.handle)
            self.assertNotEqual(handle, ids.handle)
        finally:
            if output is not None and not output.closed:
                output.close()
            ids.close()
            weights.close()


if __name__ == "__main__":
    unittest.main()
