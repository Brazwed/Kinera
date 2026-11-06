"""Opt-in RX 550 equivalence gate for OpenCL Embedding.step (F1c)."""

import math
import os
import sys
import unittest
from unittest.mock import patch

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
class EmbeddingOpenCLStepRX550Tests(unittest.TestCase):
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
            "OPENCL_EMBEDDING_F1C_DEVICE",
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

    def test_signed_zero_extreme_and_no_fma_match_cpu(self) -> None:
        rate = 1.0 + 2.0**-27
        weights_values = [[1.0, 0.5, -2.0], [-0.25, 1e308, 0.0]]
        gradients_values = [
            [1.0 - 2.0**-27, -0.5, 0.25],
            [0.75, 1e308, -0.0],
        ]
        expected = self.cpu.embedding_step(weights_values, gradients_values, rate)
        weights = self.backend.buffer_from_host(weights_values, shape=(2, 3))
        gradients = self.backend.buffer_from_host(gradients_values, shape=(2, 3))
        api = self.backend.runtime.api
        try:
            with (
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
            ):
                result = self.backend.embedding_step(weights, gradients, rate)
                read.assert_not_called()
                write.assert_not_called()
            self.assertIs(result, weights)
            self.assertFalse(gradients.closed)
            actual = self.backend.buffer_to_host(result)
            self._assert_values(actual, expected)
            self.assertEqual(expected[0][0], 0.0)
            self.assertEqual(actual[0][0], 0.0)
            self.assertEqual(math.fma(-rate, gradients_values[0][0], 1.0), 2.0**-54)
        finally:
            gradients.close()
            weights.close()

    def test_repeated_updates_and_zero_rate_preserve_same_resident_buffer(self) -> None:
        weights_values = [[0.25, -0.5, 1.0], [-1.5, 0.75, 0.0]]
        gradients_values = [[-0.5, 0.25, 1.5], [0.75, -1.0, 0.5]]
        expected = weights_values
        weights = self.backend.buffer_from_host(weights_values, shape=(2, 3))
        gradients = self.backend.buffer_from_host(gradients_values, shape=(2, 3))
        try:
            for index, rate in enumerate((0.1, 0.025, 0.1)):
                expected = self.cpu.embedding_step(expected, gradients_values, rate)
                api = self.backend.runtime.api
                with (
                    patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                    patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
                ):
                    result = self.backend.embedding_step(weights, gradients, rate)
                    read.assert_not_called()
                    write.assert_not_called()
                with self.subTest(iteration=index):
                    self.assertIs(result, weights)
                    self._assert_values(self.backend.buffer_to_host(weights), expected)

            before = self.backend.buffer_to_host(weights)
            enqueue_count = 0
            api = self.backend.runtime.api
            with (
                patch.object(api, "enqueue_1d", wraps=api.enqueue_1d) as enqueue,
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
            ):
                self.assertIs(
                    self.backend.embedding_step(weights, gradients, 0.0), weights
                )
                enqueue_count = enqueue.call_count
                read.assert_not_called()
                write.assert_not_called()
            self.assertEqual(enqueue_count, 0)
            self.assertEqual(self.backend.buffer_to_host(weights), before)
            self.assertEqual(
                self.backend.buffer_to_host(weights),
                self.cpu.embedding_step(expected, gradients_values, 0.0),
            )
        finally:
            gradients.close()
            weights.close()

    def test_invalid_learning_rates_do_not_launch_a_kernel(self) -> None:
        weights = self.backend.buffer_from_host([[1.0]], shape=(1, 1))
        gradients = self.backend.buffer_from_host([[2.0]], shape=(1, 1))
        try:
            original = self.backend.buffer_to_host(weights)
            api = self.backend.runtime.api
            for index, (rate, error) in enumerate(
                (
                    (True, TypeError),
                    ("0.1", TypeError),
                    (-1.0, ValueError),
                    (float("nan"), ValueError),
                    (float("inf"), ValueError),
                    (float("-inf"), ValueError),
                    (10**10000, ValueError),
                )
            ):
                with self.subTest(case=index, rate_type=type(rate).__name__):
                    with patch.object(api, "enqueue_1d", wraps=api.enqueue_1d) as enqueue:
                        with self.assertRaises(error):
                            self.backend.embedding_step(weights, gradients, rate)
                        enqueue.assert_not_called()
                    self.assertEqual(self.backend.buffer_to_host(weights), original)
        finally:
            gradients.close()
            weights.close()

    def test_layer_step_delegates_in_place_without_requiring_f1d(self) -> None:
        cpu_layer = Embedding(2, 3, seed=29, backend=self.cpu)
        gpu_layer = Embedding(2, 3, seed=29, backend=self.backend)
        supplied_weights = [[1.25, -2.5, 0.0], [3.0, 4.5, -6.0]]
        supplied_gradients = [[0.5, -1.0, 2.0], [-3.0, 0.0, 0.25]]
        new_weights = self.backend.buffer_from_host(supplied_weights, shape=(2, 3))
        new_gradients = self.backend.buffer_from_host(supplied_gradients, shape=(2, 3))
        prior_weights, prior_gradients = gpu_layer.weights, gpu_layer.grad_weights
        try:
            gpu_layer.weights = new_weights
            gpu_layer.grad_weights = new_gradients
            prior_weights.close()
            prior_gradients.close()
            gpu_layer._has_backward = True
            cpu_layer.weights = [row.copy() for row in supplied_weights]
            cpu_layer.grad_weights = [row.copy() for row in supplied_gradients]
            cpu_layer._has_backward = True

            weight_buffer = gpu_layer.weights
            expected = self.cpu.embedding_step(
                supplied_weights, supplied_gradients, 0.125
            )
            gpu_layer.step(0.125)
            cpu_layer.step(0.125)
            self.assertIs(gpu_layer.weights, weight_buffer)
            self.assertFalse(gpu_layer._has_backward)
            self.assertFalse(gpu_layer.grad_weights.closed)
            self._assert_values(self.backend.buffer_to_host(gpu_layer.weights), expected)
            self.assertEqual(cpu_layer.weights, expected)
        finally:
            gpu_layer.close()
            cpu_layer.close()

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
        print("OPENCL_EMBEDDING_F1C_MAX_ABS_ERROR", max(errors, default=0.0))


if __name__ == "__main__":
    unittest.main()
