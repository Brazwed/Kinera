"""Exact CPU neuron arithmetic regression fixtures."""

import unittest

from kinera.backends.cpu.backend import CPUBackend


class NeuronArithmeticEquivalenceTests(unittest.TestCase):
    def test_forward_backward_and_step_match_frozen_reference_fixtures(self) -> None:
        backend = CPUBackend()
        fixtures = (
            # activation, weights, bias, inputs, slope, grad_output, rate,
            # z, output, grad_weights, grad_bias, grad_inputs, final weights, bias
            (
                "relu", [1.0, 2.0], -4.0, [1.0, 1.0], 0.01, 3.0, 0.1,
                -1.0, 0.0, [0.0, 0.0], 0.0, [0.0, 0.0], [1.0, 2.0], -4.0,
            ),
            (
                "leaky_relu", [-2.0, 1.0], 0.0, [1.0, 1.0], 0.01, 3.0, 0.1,
                -1.0, -0.01, [0.03, 0.03], 0.03, [-0.06, 0.03],
                [-2.003, 0.997], -0.003,
            ),
            (
                "linear", [-2.0, 1.0], 0.0, [1.0, 1.0], 0.01, 3.0, 0.1,
                -1.0, -1.0, [3.0, 3.0], 3.0, [-6.0, 3.0], [-2.3, 0.7],
                -0.30000000000000004,
            ),
            (
                "relu", [1.0, 2.0], 0.0, [0.0, 0.0], 0.01, 2.0, 0.25,
                0.0, 0.0, [0.0, 0.0], 0.0, [0.0, 0.0], [1.0, 2.0], 0.0,
            ),
        )

        for (
            activation, weights, bias, inputs, slope, grad_output, rate,
            expected_z, expected_output, expected_grad_weights, expected_grad_bias,
            expected_grad_inputs, expected_weights, expected_bias,
        ) in fixtures:
            with self.subTest(activation=activation, z=expected_z):
                z, output = backend.neuron_forward(
                    weights, bias, inputs, activation, slope
                )
                self.assertEqual((z, output), (expected_z, expected_output))
                grad_weights, grad_bias, grad_inputs = backend.neuron_backward(
                    weights, inputs, z, grad_output, activation, slope
                )
                self.assertEqual(
                    (grad_weights, grad_bias, grad_inputs),
                    (expected_grad_weights, expected_grad_bias, expected_grad_inputs),
                )
                updated_weights, updated_bias = backend.neuron_step(
                    weights, bias, grad_weights, grad_bias, rate
                )
                self.assertEqual(
                    (updated_weights, updated_bias), (expected_weights, expected_bias)
                )


if __name__ == "__main__":
    unittest.main()
