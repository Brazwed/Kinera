"""Opt-in RX 550 checks for resident OpenCL mean pooling (F2.2)."""

from __future__ import annotations

import ctypes
import math
import os
import platform
import sys
import unittest
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.nn.models.hybrid import AutoregressiveModel


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on RX 550/Rusticl with FP64 enabled",
)
class MeanPoolForwardOpenCLRX550Tests(unittest.TestCase):
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
            "OPENCL_F2_2_DEVICE",
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
            "OS",
            platform.platform(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_cpu_equivalence_adversarial_order_and_resident_io(self) -> None:
        cases = (
            [[-0.0, 2.5]],
            [
                [1e16, 1e16, 1e16, 1.0, 1e308, 0.125, 1e16],
                [1.0, -1e16, 1.0, 1e16, -1e308, 0.25, -1e16],
                [-1e16, 1.0, -1e16, 1.0, 1e292, 0.375, 1.0],
                [1.0, 1.0, 1.0, -1e16, 0.5, 0.5, -1e16],
            ],
            [
                [1e308, -1e308, 1e16, -1e16],
                [-1e308, 1e308, 1.0, -1.0],
                [1e292, 1e292, -1e16, 1e16],
                [1e-16, -1e-16, 1e-16, -1e-16],
            ],
        )
        long_pattern = (
            [1e16, -1e16, 1.0, 1e-16],
            [1.0, 1e16, -1e16, -1e-16],
            [-1e16, 1.0, 1e16, 1e-16],
            [-1e16, -1.0, 1e16, -1e-16],
        )
        long_context = [long_pattern[row % len(long_pattern)] for row in range(257)]
        max_absolute_error = 0.0
        max_relative_error = 0.0
        for values in (*cases, long_context):
            with self.subTest(context=len(values), features=len(values[0])):
                vectors = self.backend.buffer_from_host(
                    values, shape=(len(values), len(values[0]))
                )
                try:
                    expected = self.cpu.mean_pool_forward(values)
                    api = self.backend.runtime.api
                    with (
                        patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                        patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
                    ):
                        pooled = self.backend.mean_pool_forward(vectors)
                    try:
                        self.assertIsInstance(pooled, OpenCLBuffer)
                        self.assertEqual(pooled.shape, (len(values[0]),))
                        self.assertIs(pooled._manager, self.backend.memory)
                        read.assert_called_once()
                        self.assertEqual(read.call_args.args[3], ctypes.sizeof(ctypes.c_int32))
                        write.assert_not_called()
                        self.assertFalse(vectors.closed)
                        actual = self.backend.buffer_to_host(pooled)
                        for actual_value, expected_value in zip(actual, expected):
                            if math.isinf(expected_value):
                                self.assertEqual(actual_value, expected_value)
                                continue
                            self.assertTrue(
                                math.isclose(
                                    actual_value,
                                    expected_value,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                ),
                                (actual_value, expected_value),
                            )
                            error = abs(actual_value - expected_value)
                            max_absolute_error = max(max_absolute_error, error)
                            if expected_value != 0.0:
                                max_relative_error = max(
                                    max_relative_error,
                                    error / abs(expected_value),
                                )
                            elif error != 0.0:
                                max_relative_error = math.inf
                    finally:
                        pooled.close()
                finally:
                    vectors.close()
        print("OPENCL_F2_2_MAX_ABS_ERROR", max_absolute_error)
        print("OPENCL_F2_2_MAX_REL_ERROR", max_relative_error)

    def test_resident_nan_and_infinities_are_rejected_with_only_status_read(self) -> None:
        api = self.backend.runtime.api
        vectors = self.backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2)
        )
        try:
            live_buffers = set(self.backend.memory._buffers)
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(invalid=invalid):
                    host_invalid = (ctypes.c_double * 1)(invalid)
                    api.write_buffer(
                        self.backend.memory.queue,
                        vectors.handle,
                        ctypes.addressof(host_invalid),
                        ctypes.sizeof(host_invalid),
                    )
                    allocations_before = set(self.backend.memory._buffers)
                    with patch.object(api, "read_buffer", wraps=api.read_buffer) as read:
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            self.backend.mean_pool_forward(vectors)
                    read.assert_called_once()
                    self.assertEqual(read.call_args.args[3], ctypes.sizeof(ctypes.c_int32))
                    self.assertEqual(set(self.backend.memory._buffers), allocations_before)
                    self.assertEqual(set(self.backend.memory._buffers), live_buffers)
                    restore = (ctypes.c_double * 1)(1.0)
                    api.write_buffer(
                        self.backend.memory.queue,
                        vectors.handle,
                        ctypes.addressof(restore),
                        ctypes.sizeof(restore),
                    )
        finally:
            vectors.close()

    def test_finite_inputs_can_overflow_like_cpu_after_the_finite_preflight(self) -> None:
        values = [[1e308], [1e308]]
        vectors = self.backend.buffer_from_host(values, shape=(2, 1))
        try:
            expected = self.cpu.mean_pool_forward(values)
            pooled = self.backend.mean_pool_forward(vectors)
            try:
                actual = self.backend.buffer_to_host(pooled)
                self.assertTrue(math.isinf(expected[0]))
                self.assertEqual(actual, expected)
            finally:
                pooled.close()
        finally:
            vectors.close()

    def test_hybrid_representation_is_pooled_resident_and_releases_matrix(self) -> None:
        gpu_model = AutoregressiveModel(
            5,
            4,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=217,
            backend=self.backend,
        )
        cpu_model = AutoregressiveModel(
            5,
            4,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=217,
            backend=self.cpu,
        )
        chars, syllables = [1, 3, 2], [2, 0, 1]
        features = [[0.25, -1.0], [2.5, 0.0], [-0.5, 3.0]]
        gpu_weights = gpu_model.representation.char_embedding.weights
        original_weights = self.backend.buffer_to_host(gpu_weights)
        try:
            expected = cpu_model._pool_representation_backend(
                chars.copy(), syllables.copy(), features
            )
            api = self.backend.runtime.api
            with (
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
            ):
                actual = gpu_model._pool_representation_backend(
                    chars.copy(), syllables.copy(), features
                )
            self.assertIsInstance(actual, OpenCLBuffer)
            self.assertEqual(actual.shape, (gpu_model.representation.output_dim,))
            self.assertEqual(read.call_count, 4)
            self.assertEqual(
                [call.args[3] for call in read.call_args_list], [4, 4, 4, 4]
            )
            writes = [call.args[3] for call in write.call_args_list]
            self.assertEqual(writes, [12, 80, 12, 96, 48])
            self.assertNotIn(3 * gpu_model.representation.output_dim * 8, writes)
            self.assertNotIn(gpu_model.representation.output_dim * 8, writes)
            self.assertIsNone(gpu_model.representation.last_output)
            self.assertFalse(gpu_weights.closed)
            self.assertEqual(self.backend.buffer_to_host(gpu_weights), original_weights)
            self.assertTrue(gpu_model.representation.char_embedding.output.closed)
            self.assertTrue(gpu_model.representation.syllable_embedding.output.closed)
            try:
                actual_values = self.backend.buffer_to_host(actual)
                self.assertEqual(len(actual_values), len(expected))
                max_absolute_error = 0.0
                max_relative_error = 0.0
                for actual_value, expected_value in zip(actual_values, expected):
                    self.assertTrue(
                        math.isclose(
                            actual_value,
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        (actual_value, expected_value),
                    )
                    error = abs(actual_value - expected_value)
                    max_absolute_error = max(max_absolute_error, error)
                    if expected_value != 0.0:
                        max_relative_error = max(
                            max_relative_error, error / abs(expected_value)
                        )
                    elif error != 0.0:
                        max_relative_error = math.inf
                print("OPENCL_F2_2_INTEGRATED_MAX_ABS_ERROR", max_absolute_error)
                print("OPENCL_F2_2_INTEGRATED_MAX_REL_ERROR", max_relative_error)
            finally:
                actual.close()
        finally:
            gpu_model.close()
            cpu_model.close()


if __name__ == "__main__":
    unittest.main()
