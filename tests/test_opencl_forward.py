"""Opt-in GPU equivalence gate for the first OpenCL Linear.forward kernel."""

import os
import sys
import unittest

from kinera.backends import get_backend
from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.layers.linear import Linear


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on an OpenCL FP64 GPU runner",
)
class LinearOpenCLForwardEquivalenceTests(unittest.TestCase):
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
            "OPENCL_FORWARD_AUDIT_DEVICE",
            cls.device.name,
            cls.device.pci_bdf,
            "PYTHON",
            sys.version.split()[0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def _assert_layer_equivalence(
        self, weights, bias, inputs, *, exact_match: bool = False
    ) -> tuple[list[float], list[float]]:
        input_size = len(inputs)
        output_size = len(weights)
        layer = Linear(input_size, output_size, seed=19, backend=self.backend)
        cpu_layer = Linear(input_size, output_size, seed=19, backend=self.cpu)
        old_weights, old_bias = layer.weights, layer.bias
        device_weights = self.backend.buffer_from_host(
            weights, shape=(output_size, input_size), dtype="float64"
        )
        device_bias = self.backend.buffer_from_host(
            bias, shape=(output_size,), dtype="float64"
        )
        layer.weights = device_weights
        layer.bias = device_bias
        cpu_layer.weights = weights
        cpu_layer.bias = bias
        output = None
        try:
            actual_buffer = layer.forward(inputs)
            self.assertNotIsInstance(actual_buffer, list)
            output = actual_buffer
            actual = self.backend.buffer_to_host(output)
            expected = cpu_layer.forward(inputs)
            self.assertEqual(len(actual), len(expected))
            errors = [abs(got - want) for got, want in zip(actual, expected)]
            for index, (got, want) in enumerate(zip(actual, expected)):
                with self.subTest(index=index):
                    self.assertLessEqual(
                        abs(got - want), ATOL + RTOL * abs(want),
                        msg=f"OpenCL={got}, CPU={want}",
                    )
                    if exact_match:
                        self.assertEqual(got, want)
            print("OPENCL_LINEAR_FORWARD_MAX_ABS_ERROR", max(errors, default=0.0))
        finally:
            if output is not None:
                output.close()
            old_bias.close()
            old_weights.close()
            layer.close()
            cpu_layer.close()
        return actual, expected

    def test_small_signed_matrix_matches_cpu(self) -> None:
        self._assert_layer_equivalence(
            [[1.0, -2.0, 0.0], [-0.5, 0.0, 4.0], [0.0, 3.0, -1.0]],
            [0.25, -1.0, 0.0],
            [2.0, -1.0, 0.5],
        )

    def test_cancellation_sensitive_sum_matches_cpu(self) -> None:
        self._assert_layer_equivalence(
            [
                [1e16, 1.0, -1e16],
                [1e100, 3.0, -1e100],
                [1e300, 1e284, -1e300],
                [-1e300, 1e284, 1e300],
            ],
            [0.0, -0.5, 0.25, -0.25],
            [1.0, 1.0, 1.0],
        )

    def test_dimensions_signs_and_zeros_match_cpu(self) -> None:
        for output_size, input_size in ((1, 1), (2, 3), (7, 19), (31, 5)):
            with self.subTest(shape=(output_size, input_size)):
                weights = [
                    [
                        0.0
                        if (row + column) % 7 == 0
                        else (((row * 11 + column * 5) % 29) - 14) / 9.0
                        for column in range(input_size)
                    ]
                    for row in range(output_size)
                ]
                bias = [((row % 5) - 2) / 13.0 for row in range(output_size)]
                inputs = [((column % 11) - 5) / 7.0 for column in range(input_size)]
                self._assert_layer_equivalence(weights, bias, inputs)

    def test_extreme_finite_magnitudes_and_subnormal_match_cpu(self) -> None:
        self._assert_layer_equivalence(
            [
                [1e150, -1e150, 1e-150, -1e-150, 0.0],
                [1e-150, -1e-150, 0.0, 0.0, 0.0],
                [5e-324, 1e-300, -1e-300, 0.0, 0.0],
            ],
            [0.5, -2.0, 0.0],
            [1e150, 1e150, 1e150, 1e150, 1e300],
        )

    def test_product_and_add_are_not_contracted_into_fma(self) -> None:
        larger = 1.0 + 2.0**-27
        smaller = 1.0 - 2.0**-27
        actual, expected = self._assert_layer_equivalence(
            [[-1.0, larger]], [0.0], [1.0, smaller], exact_match=True
        )
        self.assertEqual(expected, [0.0])
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
