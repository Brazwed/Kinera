"""Opt-in GPU equivalence gates for the OpenCL Linear.step operation."""

import math
import os
import sys
import unittest
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.cpu.backend import CPUBackend


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on an OpenCL FP64 GPU runner",
)
class LinearOpenCLStepEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = get_backend("opencl")
        cls.device = cls.backend.runtime.require_device(0, require_fp64=True)
        if cls.device.pci_bdf is not None:
            if cls.device.pci_bdf != "0000:03:00.0":
                raise AssertionError(
                    f"expected RX 550 at 0000:03:00.0, got {cls.device.pci_bdf}"
                )
        if "Radeon RX 550" not in cls.device.name:
            raise AssertionError(f"expected RX 550, got {cls.device.name}")
        cls.cpu = CPUBackend()
        print(
            "OPENCL_STEP_GATE_DEVICE",
            cls.device.name,
            cls.device.pci_bdf,
            "PYTHON",
            sys.version.split()[0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_single_step_matches_cpu_and_preserves_separate_rounding(self) -> None:
        rate = 1.0 + 2.0**-27
        smaller = 1.0 - 2.0**-27
        weights = [[1.0, 0.5, -2.0], [-0.25, 2.0, 1e16]]
        bias = [1.0, -0.5]
        grad_weights = [[smaller, -0.5, 0.25], [0.75, -1e16, -0.25]]
        grad_bias = [smaller, 0.125]
        expected_weights, expected_bias = self.cpu.linear_step(
            weights, bias, grad_weights, grad_bias, rate
        )

        device_weights = self.backend.buffer_from_host(
            weights, shape=(2, 3), dtype="float64"
        )
        device_bias = self.backend.buffer_from_host(
            bias, shape=(2,), dtype="float64"
        )
        device_grad_weights = self.backend.buffer_from_host(
            grad_weights, shape=(2, 3), dtype="float64"
        )
        device_grad_bias = self.backend.buffer_from_host(
            grad_bias, shape=(2,), dtype="float64"
        )
        api = self.backend.runtime.api
        try:
            with (
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
            ):
                returned_weights, returned_bias = self.backend.linear_step(
                    device_weights,
                    device_bias,
                    device_grad_weights,
                    device_grad_bias,
                    rate,
                )
                read.assert_not_called()
                write.assert_not_called()
            self.assertIs(returned_weights, device_weights)
            self.assertIs(returned_bias, device_bias)
            actual_weights = self.backend.buffer_to_host(device_weights)
            actual_bias = self.backend.buffer_to_host(device_bias)
            self._assert_values("weights", actual_weights, expected_weights)
            self._assert_values("bias", actual_bias, expected_bias)

            self.assertEqual(expected_weights[0][0], 0.0)
            self.assertEqual(expected_bias[0], 0.0)
            self.assertEqual(actual_weights[0][0], 0.0)
            self.assertEqual(actual_bias[0], 0.0)
            self.assertEqual(math.fma(-rate, smaller, 1.0), 2.0**-54)
        finally:
            device_grad_bias.close()
            device_grad_weights.close()
            device_bias.close()
            device_weights.close()

    def test_consecutive_updates_and_zero_learning_rate(self) -> None:
        weights = [[0.25, -0.5, 1.0], [-1.5, 0.75, 0.0]]
        bias = [0.125, -0.25]
        grad_weights = [[-0.5, 0.25, 1.5], [0.75, -1.0, 0.5]]
        grad_bias = [-0.25, 0.125]
        device_weights = self.backend.buffer_from_host(
            weights, shape=(2, 3), dtype="float64"
        )
        device_bias = self.backend.buffer_from_host(
            bias, shape=(2,), dtype="float64"
        )
        device_grad_weights = self.backend.buffer_from_host(
            grad_weights, shape=(2, 3), dtype="float64"
        )
        device_grad_bias = self.backend.buffer_from_host(
            grad_bias, shape=(2,), dtype="float64"
        )
        cpu_weights, cpu_bias = weights, bias
        try:
            for iteration, rate in enumerate((0.1, 0.025, 0.1)):
                cpu_weights, cpu_bias = self.cpu.linear_step(
                    cpu_weights,
                    cpu_bias,
                    grad_weights,
                    grad_bias,
                    rate,
                )
                self.backend.linear_step(
                    device_weights,
                    device_bias,
                    device_grad_weights,
                    device_grad_bias,
                    rate,
                )
                with self.subTest(iteration=iteration):
                    self._assert_values(
                        "consecutive weights",
                        self.backend.buffer_to_host(device_weights),
                        cpu_weights,
                    )
                    self._assert_values(
                        "consecutive bias",
                        self.backend.buffer_to_host(device_bias),
                        cpu_bias,
                    )

            before_weights = self.backend.buffer_to_host(device_weights)
            before_bias = self.backend.buffer_to_host(device_bias)
            zero_expected_weights, zero_expected_bias = self.cpu.linear_step(
                cpu_weights,
                cpu_bias,
                grad_weights,
                grad_bias,
                0.0,
            )
            api = self.backend.runtime.api
            with (
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
            ):
                returned_weights, returned_bias = self.backend.linear_step(
                    device_weights,
                    device_bias,
                    device_grad_weights,
                    device_grad_bias,
                    0.0,
                )
                read.assert_not_called()
                write.assert_not_called()
            self.assertIs(returned_weights, device_weights)
            self.assertIs(returned_bias, device_bias)
            after_weights = self.backend.buffer_to_host(device_weights)
            after_bias = self.backend.buffer_to_host(device_bias)
            self.assertEqual(after_weights, before_weights)
            self.assertEqual(after_bias, before_bias)
            self._assert_values("zero-rate weights", after_weights, zero_expected_weights)
            self._assert_values("zero-rate bias", after_bias, zero_expected_bias)
        finally:
            device_grad_bias.close()
            device_grad_weights.close()
            device_bias.close()
            device_weights.close()

    def _assert_values(self, label, actual, expected) -> None:
        flat_actual = self._flatten(actual)
        flat_expected = self._flatten(expected)
        self.assertEqual(len(flat_actual), len(flat_expected))
        errors = [abs(got - want) for got, want in zip(flat_actual, flat_expected)]
        for index, (got, want) in enumerate(zip(flat_actual, flat_expected)):
            with self.subTest(parameter=label, index=index):
                self.assertLessEqual(
                    abs(got - want), ATOL + RTOL * abs(want),
                    msg=f"{label}: OpenCL={got}, CPU={want}",
                )
        print(f"OPENCL_LINEAR_STEP_{label.upper()}_MAX_ABS_ERROR", max(errors, default=0.0))

    @staticmethod
    def _flatten(values):
        if not values:
            return []
        if isinstance(values[0], list):
            return [value for row in values for value in row]
        return list(values)


if __name__ == "__main__":
    unittest.main()
