"""Opt-in GPU equivalence gate for Linear.backward on OpenCL FP64 devices."""

import os
import sys
import unittest

from kinera.backends import get_backend
from kinera.backends.base import BackendBuffer
from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.layers.linear import Linear


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on an OpenCL FP64 GPU runner",
)
class LinearOpenCLBackwardEquivalenceTests(unittest.TestCase):
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
            "OPENCL_BACKWARD_GATE_DEVICE",
            cls.device.name,
            cls.device.pci_bdf,
            "PYTHON",
            sys.version.split()[0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_each_gradient_matches_cpu_with_resident_buffers(self) -> None:
        weights = [
            [1e16, -2.0, 0.0],
            [1.0, 0.5, 4.0],
            [-1e16, 1.0, -3.0],
            [0.0, 2.0, 0.25],
        ]
        inputs = [2.0, -1.0, 0.5]
        grad_output = [1.0, 1.0, 1.0, 0.0]
        self._assert_backward_case(weights, inputs, grad_output)

    def test_small_signed_head_fragments_match_cpu(self) -> None:
        cases = (
            (
                [[-2.5]],
                [0.0],
                [-0.75],
            ),
            (
                [[1.0, -2.0, 0.0], [-0.5, 0.0, 4.0]],
                [2.0, -1.0, 0.5],
                [0.25, -1.5],
            ),
            (
                [
                    [0.25, -0.5, 0.0, 1.5, -2.0],
                    [-1.0, 0.75, 2.0, 0.0, 0.5],
                    [0.0, -1.25, 0.5, -0.75, 1.0],
                ],
                [-0.5, 1.25, 0.0, -2.0, 0.75],
                [1.0, -0.5, 0.25],
            ),
        )
        for case_index, (weights, inputs, grad_output) in enumerate(cases):
            with self.subTest(case=case_index, shape=(len(weights), len(inputs))):
                self._assert_backward_case(weights, inputs, grad_output)

    def test_head_like_51_by_26_shape_matches_cpu(self) -> None:
        output_size = 51
        input_size = 26
        weights = []
        for row in range(output_size):
            values = []
            for column in range(input_size):
                if column == 0:
                    value = (1e16, 1.0, -1e16)[row] if row < 3 else 0.0
                elif (row * 7 + column * 3) % 13 == 0:
                    value = 0.0
                else:
                    value = (((row * 11 + column * 5) % 29) - 14) / 17.0
                values.append(value)
            weights.append(values)
        inputs = [((column % 9) - 4) / 7.0 for column in range(input_size)]
        grad_output = [
            1.0 if row < 3 else ((row % 7) - 3) / 11.0
            for row in range(output_size)
        ]
        self._assert_backward_case(weights, inputs, grad_output)

    def _assert_backward_case(self, weights, inputs, grad_output) -> None:
        bias = [0.0 for _ in weights]
        expected_grad_weights, expected_grad_bias, expected_grad_inputs = (
            self.cpu.linear_backward(weights, inputs, grad_output)
        )

        layer = Linear(len(inputs), len(weights), seed=31, backend=self.backend)
        old_weights, old_bias = layer.weights, layer.bias
        layer.weights = self.backend.buffer_from_host(
            weights, shape=(len(weights), len(inputs)), dtype="float64"
        )
        layer.bias = self.backend.buffer_from_host(
            bias, shape=(len(weights),), dtype="float64"
        )
        device_inputs = self.backend.buffer_from_host(
            inputs, shape=(len(inputs),), dtype="float64"
        )
        device_grad_output = self.backend.buffer_from_host(
            grad_output, shape=(len(weights),), dtype="float64"
        )
        grad_inputs = None
        try:
            forward_output = layer.forward(device_inputs)
            self.assertIsInstance(forward_output, BackendBuffer)
            grad_inputs = layer.backward(device_grad_output)
            self.assertIsInstance(grad_inputs, BackendBuffer)
            self.assertIsInstance(layer.grad_weights, BackendBuffer)
            self.assertIsInstance(layer.grad_bias, BackendBuffer)

            actual_grad_weights = self.backend.buffer_to_host(layer.grad_weights)
            actual_grad_bias = self.backend.buffer_to_host(layer.grad_bias)
            actual_grad_inputs = self.backend.buffer_to_host(grad_inputs)

            self._assert_values(
                "grad_weights", actual_grad_weights, expected_grad_weights
            )
            self._assert_values("grad_bias", actual_grad_bias, expected_grad_bias)
            self._assert_values(
                "grad_inputs", actual_grad_inputs, expected_grad_inputs
            )
            if grad_output[:3] == [1.0, 1.0, 1.0] and len(grad_output) >= 3:
                self.assertEqual(actual_grad_inputs[0], 1.0)
        finally:
            if grad_inputs is not None:
                grad_inputs.close()
            device_grad_output.close()
            device_inputs.close()
            old_bias.close()
            old_weights.close()
            layer.close()

    def _assert_values(self, label, actual, expected) -> None:
        flat_actual = self._flatten(actual)
        flat_expected = self._flatten(expected)
        self.assertEqual(len(flat_actual), len(flat_expected))
        errors = [abs(got - want) for got, want in zip(flat_actual, flat_expected)]
        for index, (got, want) in enumerate(zip(flat_actual, flat_expected)):
            with self.subTest(gradient=label, index=index):
                self.assertLessEqual(
                    abs(got - want), ATOL + RTOL * abs(want),
                    msg=f"{label}: OpenCL={got}, CPU={want}",
                )
        print(f"OPENCL_LINEAR_BACKWARD_{label.upper()}_MAX_ABS_ERROR", max(errors, default=0.0))

    @staticmethod
    def _flatten(values):
        if not values:
            return []
        if isinstance(values[0], list):
            return [value for row in values for value in row]
        return list(values)


if __name__ == "__main__":
    unittest.main()
