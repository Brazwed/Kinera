"""Opt-in GPU tests for Linear.forward against the CPU oracle."""

import os
import unittest

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend
from kinera.nn.layers.linear import Linear


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class LinearCudaForwardEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _assert_equivalent(self, weights, bias, inputs, *, pass_host_input: bool) -> None:
        output_size = len(weights)
        input_size = len(inputs)
        layer = Linear(
            input_size=input_size,
            output_size=output_size,
            seed=7,
            backend=self.backend,
        )
        device_weights = self.backend.buffer_from_host(
            weights, shape=(output_size, input_size), dtype="float64"
        )
        device_bias = self.backend.buffer_from_host(
            bias, shape=(output_size,), dtype="float64"
        )
        device_inputs = self.backend.buffer_from_host(
            inputs, shape=(input_size,), dtype="float64"
        )
        layer.weights = device_weights
        layer.bias = device_bias
        output = None
        try:
            output = layer.forward(inputs if pass_host_input else device_inputs)
            self.assertNotIsInstance(output, list)
            actual = self.backend.buffer_to_host(output)
            expected = self.cpu.linear_forward(weights, bias, inputs)
            errors = [abs(got - want) for got, want in zip(actual, expected)]
            allowed = [ATOL + RTOL * abs(want) for want in expected]
            self.assertEqual(len(actual), len(expected))
            for index, (got, want, error, limit) in enumerate(
                zip(actual, expected, errors, allowed)
            ):
                with self.subTest(index=index):
                    self.assertLessEqual(error, limit, msg=f"CUDA={got}, CPU={want}")
            self.assertTrue(all(value == value for value in actual), "unexpected NaN")
            self.assertTrue(all(abs(value) != float("inf") for value in actual), "unexpected Inf")
            print("LINEAR_FORWARD_MAX_ABS_ERROR", max(errors, default=0.0))
        finally:
            if output is not None:
                output.close()
            device_inputs.close()
            device_bias.close()
            device_weights.close()

    def test_small_signed_and_zero_values_match_cpu(self) -> None:
        self._assert_equivalent(
            [[1.0, -2.0, 0.0], [-0.5, 0.0, 4.0], [0.0, 3.0, -1.0]],
            [0.25, -1.0, 0.0],
            [2.0, -1.0, 0.5],
            pass_host_input=False,
        )

    def test_kinera_output_shapes_match_cpu_oracle(self) -> None:
        input_size = 26
        inputs = [((index % 9) - 4) / 7.0 for index in range(input_size)]
        for output_size in (51, 3932):
            with self.subTest(output_size=output_size):
                weights = [
                    [(((row * 7 + column * 3) % 23) - 11) / 17.0 for column in range(input_size)]
                    for row in range(output_size)
                ]
                bias = [((row % 7) - 3) / 11.0 for row in range(output_size)]
                self._assert_equivalent(
                    weights, bias, inputs, pass_host_input=(output_size == 51)
                )


    def test_forward_rejects_mismatched_device_shapes(self) -> None:
        weights = self.backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2), dtype="float64"
        )
        bias = self.backend.buffer_from_host([0.0, 0.0], shape=(2,), dtype="float64")
        inputs = self.backend.buffer_from_host([1.0, 2.0, 3.0], shape=(3,), dtype="float64")
        try:
            with self.assertRaisesRegex(ValueError, "weights columns"):
                self.backend.linear_forward(weights, bias, inputs)
        finally:
            inputs.close()
            bias.close()
            weights.close()


if __name__ == "__main__":
    unittest.main()
