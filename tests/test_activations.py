import unittest

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.activations import (
    leaky_relu,
    leaky_relu_derivative,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
)


class ReLUTests(unittest.TestCase):
    def test_positive_value_passes_through(self) -> None:
        self.assertEqual(relu(3.5), 3.5)

    def test_zero_returns_zero(self) -> None:
        self.assertEqual(relu(0.0), 0.0)

    def test_negative_value_returns_zero(self) -> None:
        self.assertEqual(relu(-2.5), 0.0)

    def test_positive_derivative_is_one(self) -> None:
        self.assertEqual(relu_derivative(3.5), 1.0)

    def test_zero_and_negative_derivatives_are_zero(self) -> None:
        self.assertEqual(relu_derivative(0.0), 0.0)
        self.assertEqual(relu_derivative(-2.5), 0.0)


class LinearActivationTests(unittest.TestCase):
    def test_linear_returns_input_and_has_unit_derivative(self) -> None:
        self.assertEqual(linear(3.5), 3.5)
        self.assertEqual(linear(-2.5), -2.5)
        self.assertEqual(linear_derivative(-2.5), 1.0)


class LeakyReLUActivationTests(unittest.TestCase):
    def test_positive_value_passes_through(self) -> None:
        self.assertEqual(leaky_relu(3.5), 3.5)

    def test_negative_value_uses_default_slope(self) -> None:
        self.assertAlmostEqual(leaky_relu(-2.5), -0.025)

    def test_custom_slope_is_used_for_value_and_derivative(self) -> None:
        self.assertAlmostEqual(leaky_relu(-2.5, negative_slope=0.1), -0.25)
        self.assertEqual(leaky_relu_derivative(-2.5, negative_slope=0.1), 0.1)

    def test_zero_uses_negative_branch(self) -> None:
        self.assertEqual(leaky_relu(0.0), 0.0)
        self.assertEqual(leaky_relu_derivative(0.0), 0.01)

    def test_invalid_slope_is_rejected(self) -> None:
        for slope in (0.0, -0.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises((TypeError, ValueError)):
                leaky_relu(-1.0, negative_slope=slope)


class ActivationBackendIntegrationTests(unittest.TestCase):
    def test_public_activation_functions_use_the_selected_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def activation(self, name, value, negative_slope=0.01):
                self.calls.append(name)
                return super().activation(name, value, negative_slope)

            def activation_derivative(self, name, value, negative_slope=0.01):
                self.calls.append(f"{name}_derivative")
                return super().activation_derivative(name, value, negative_slope)

        backend = RecordingBackend()
        self.assertEqual(relu(-1.0, backend=backend), 0.0)
        self.assertEqual(linear_derivative(2.0, backend=backend), 1.0)
        self.assertEqual(backend.calls, ["relu", "linear_derivative"])


if __name__ == "__main__":
    unittest.main()
