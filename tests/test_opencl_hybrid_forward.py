"""Opt-in RX 550 gate for resident HybridRepresentation.forward (F2.1)."""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import sys
import unittest
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.nn.layers.embedding import Embedding
from kinera.nn.layers.hybrid import HybridRepresentation


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on RX 550/Rusticl with FP64 enabled",
)
class HybridRepresentationOpenCLForwardRX550Tests(unittest.TestCase):
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
            "OPENCL_F2_1_DEVICE",
            cls.device.name,
            "PCI",
            cls.device.pci_bdf,
            "DRIVER",
            cls.device.driver_version,
            "RUNTIME",
            cls.device.version,
            "PYTHON",
            sys.version.split()[0],
            "OS",
            platform.platform(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def _assert_exact(self, char, features, syllable) -> None:
        expected = self.cpu.hybrid_representation_forward(
            self.backend.buffer_to_host(char),
            self.backend.buffer_to_host(features),
            self.backend.buffer_to_host(syllable),
        )
        api = self.backend.runtime.api
        with (
            patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
            patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
        ):
            output = self.backend.hybrid_representation_forward(
                char, features, syllable
            )
        try:
            self.assertIsInstance(output, OpenCLBuffer)
            self.assertEqual(output.shape, (char.shape[0], char.shape[1] + features.shape[1] + syllable.shape[1]))
            self.assertIs(output._manager, self.backend.memory)
            read.assert_called_once()
            self.assertEqual(read.call_args.args[3], ctypes.sizeof(ctypes.c_int32))
            write.assert_not_called()
            actual = self.backend.buffer_to_host(output)
            self.assertEqual(actual, expected)
            flatten = lambda matrix: [value for row in matrix for value in row]
            expected_values = flatten(expected)
            actual_values = flatten(actual)
            self.assertEqual(
                struct.pack("<" + "d" * len(expected_values), *actual_values),
                struct.pack("<" + "d" * len(expected_values), *expected_values),
            )
            max_error = max(
                (
                    abs(actual[row][column] - expected[row][column])
                    for row in range(len(expected))
                    for column in range(len(expected[row]))
                ),
                default=0.0,
            )
            self.assertEqual(max_error, 0.0)
            print("OPENCL_F2_1_MAX_ABS_ERROR", max_error)
        finally:
            output.close()

    def test_resident_shapes_order_signed_zero_and_lifecycle(self) -> None:
        cases = (
            (
                [[-0.0]],
                [[-2.0, 3.0]],
                [[4.0]],
            ),
            (
                [[1.0, -2.0], [3.0, -0.0], [5.0, 6.0]],
                [[-7.0], [8.0], [0.0]],
                [[9.0, 10.0, -11.0], [-12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            ),
        )
        for char_host, feature_host, syllable_host in cases:
            with self.subTest(rows=len(char_host), dims=(len(char_host[0]), len(feature_host[0]), len(syllable_host[0]))):
                char = self.backend.buffer_from_host(
                    char_host, shape=(len(char_host), len(char_host[0]))
                )
                features = self.backend.buffer_from_host(
                    feature_host, shape=(len(feature_host), len(feature_host[0]))
                )
                syllable = self.backend.buffer_from_host(
                    syllable_host, shape=(len(syllable_host), len(syllable_host[0]))
                )
                try:
                    self._assert_exact(char, features, syllable)
                    self.assertFalse(char.closed)
                    self.assertFalse(features.closed)
                    self.assertFalse(syllable.closed)
                finally:
                    char.close()
                    features.close()
                    syllable.close()

    def test_resident_nan_and_both_infinities_are_rejected_in_each_channel(self) -> None:
        original_rows = (
            [[1.0, 2.0]],
            [[3.0]],
            [[4.0, 5.0]],
        )
        buffers = [
            self.backend.buffer_from_host(values, shape=(1, len(values[0])))
            for values in original_rows
        ]
        try:
            api = self.backend.runtime.api
            # The public API intentionally rejects non-finite host values. Mutate
            # the resident allocations through explicit test-only writes instead.
            original = [self.backend.buffer_to_host(buffer)[0][0] for buffer in buffers]
            live_inputs = set(self.backend.memory._buffers)
            for channel, buffer in enumerate(buffers):
                for invalid in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(channel=channel, invalid=invalid):
                        host = (ctypes.c_double * 1)(invalid)
                        api.write_buffer(
                            self.backend.memory.queue,
                            buffer.handle,
                            ctypes.addressof(host),
                            ctypes.sizeof(host),
                        )
                        with patch.object(
                            api, "read_buffer", wraps=api.read_buffer
                        ) as read:
                            with self.assertRaisesRegex(ValueError, "non-finite"):
                                self.backend.hybrid_representation_forward(*buffers)
                        read.assert_called_once()
                        self.assertEqual(read.call_args.args[3], ctypes.sizeof(ctypes.c_int32))
                        self.assertEqual(set(self.backend.memory._buffers), live_inputs)
                        self.assertTrue(all(not item.closed for item in buffers))
                        restore = (ctypes.c_double * 1)(original[channel])
                        api.write_buffer(
                            self.backend.memory.queue,
                            buffer.handle,
                            ctypes.addressof(restore),
                            ctypes.sizeof(restore),
                        )
        finally:
            for buffer in buffers:
                buffer.close()

    def test_hybrid_layer_integrates_and_preserves_prevalidation_state(self) -> None:
        cpu_layer = HybridRepresentation(
            Embedding(5, 2, seed=41, backend=self.cpu),
            Embedding(4, 3, seed=42, backend=self.cpu),
            feature_dim=2,
        )
        gpu_layer = HybridRepresentation(
            Embedding(5, 2, seed=41, backend=self.backend),
            Embedding(4, 3, seed=42, backend=self.backend),
            feature_dim=2,
        )
        try:
            chars, syllables = [1, 3], [2, 0]
            features = [[-1.5, 2.0], [3.0, -4.25]]
            expected = cpu_layer.forward(chars.copy(), syllables.copy(), features)
            api = self.backend.runtime.api
            with patch.object(api, "read_buffer", wraps=api.read_buffer) as read:
                output = gpu_layer.forward(chars.copy(), syllables.copy(), features)
            self.assertEqual(read.call_count, 3)
            self.assertTrue(
                all(
                    call.args[3] == ctypes.sizeof(ctypes.c_int32)
                    for call in read.call_args_list
                )
            )
            self.assertIsInstance(output, OpenCLBuffer)
            self.assertEqual(self.backend.buffer_to_host(output), expected)
            self.assertTrue(gpu_layer.char_embedding.output.closed)
            self.assertTrue(gpu_layer.syllable_embedding.output.closed)
            self.assertIs(gpu_layer.last_output, output)

            with self.assertRaises(ValueError):
                gpu_layer.forward(chars.copy(), syllables.copy(), [[1.0], [2.0]])
            self.assertIs(gpu_layer.last_output, output)
            self.assertFalse(output.closed)
            self.assertEqual(gpu_layer.last_context_size, 2)

            self.assertEqual(gpu_layer.forward([], [], []), [])
            self.assertTrue(output.closed)
            self.assertEqual(gpu_layer.last_output, [])
            self.assertEqual(gpu_layer.last_context_size, 0)
            self.assertEqual(gpu_layer.char_embedding.last_ids, [])
            self.assertEqual(gpu_layer.syllable_embedding.last_ids, [])
        finally:
            gpu_layer.release_output()
            gpu_layer.char_embedding.close()
            gpu_layer.syllable_embedding.close()
            cpu_layer.char_embedding.close()
            cpu_layer.syllable_embedding.close()


if __name__ == "__main__":
    unittest.main()
