import math
import unittest

from kinera.diagnostics.gradients import compare_gradients, finite_difference_gradient


class NumericalGradientUtilityTests(unittest.TestCase):
    def test_central_difference_matches_analytic_gradient_without_mutation(self) -> None:
        parameters = [1.25, -0.75]
        original = parameters.copy()

        def objective(values: list[float]) -> float:
            x, y = values
            return x * x + 3.0 * x * y + 2.0 * y * y

        numeric = finite_difference_gradient(objective, parameters, epsilon=1e-6)
        analytic = [2.0 * parameters[0] + 3.0 * parameters[1], 3.0 * parameters[0] + 4.0 * parameters[1]]

        self.assertEqual(parameters, original)
        result = compare_gradients(analytic, numeric, absolute_tolerance=1e-8, relative_tolerance=1e-8)
        self.assertTrue(result.passed)
        self.assertLessEqual(result.max_absolute_error, 1e-8)

    def test_finite_difference_rejects_invalid_inputs_and_non_finite_objectives(self) -> None:
        with self.assertRaises(TypeError):
            finite_difference_gradient(None, [1.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            finite_difference_gradient(lambda values: values[0] ** 2, [1.0], epsilon=0.0)
        with self.assertRaises(ValueError):
            finite_difference_gradient(lambda values: math.nan, [1.0])

    def test_gradient_comparison_rejects_shape_and_value_errors(self) -> None:
        with self.assertRaises(ValueError):
            compare_gradients([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            compare_gradients([math.inf], [1.0])
        with self.assertRaises(ValueError):
            compare_gradients([1.0], [1.0], absolute_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
