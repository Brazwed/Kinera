import math
import unittest
from typing import Any

import kinera.nn.losses as losses_module
from kinera.nn.probability import cross_entropy, cross_entropy_gradient, softmax
from kinera.training.autoregressive import AutoregressiveModel


class ProbabilityTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self) -> None:
        probabilities = softmax([1000.0, 1001.0, 999.0])

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertTrue(all(math.isfinite(value) for value in probabilities))

    def test_cross_entropy_gradient_matches_expected(self) -> None:
        logits = [0.0, 1.0, -1.0]

        gradient = cross_entropy_gradient(logits, target_id=1)

        probabilities = softmax(logits)
        self.assertAlmostEqual(cross_entropy(logits, target_id=1), -math.log(probabilities[1]))
        self.assertEqual(len(gradient), 3)
        self.assertAlmostEqual(sum(gradient), 0.0)
        self.assertAlmostEqual(gradient[1], probabilities[1] - 1.0)

    def test_invalid_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cross_entropy([0.0, 1.0], target_id=2)

    def test_public_probability_apis_reject_invalid_logits_with_exact_errors(self) -> None:
        apis: tuple[tuple[str, Any, bool], ...] = (
            ("probability.softmax", softmax, False),
            ("probability.cross_entropy", cross_entropy, True),
            ("probability.cross_entropy_gradient", cross_entropy_gradient, True),
            ("losses.cross_entropy", losses_module.cross_entropy, True),
            ("losses.cross_entropy_gradient", losses_module.cross_entropy_gradient, True),
        )
        invalid_logits: tuple[tuple[str, Any, type[Exception], str], ...] = (
            ("container", (0.0, 1.0), TypeError, "logits deve ser uma lista"),
            ("empty", [], ValueError, "logits não pode ser vazio"),
            ("nan", [math.nan], ValueError, "logits[0] deve ser finito"),
            ("positive_inf", [math.inf], ValueError, "logits[0] deve ser finito"),
            ("negative_inf", [-math.inf], ValueError, "logits[0] deve ser finito"),
            ("bool", [True], TypeError, "logits[0] deve ser numérico"),
            ("string", ["1.0"], TypeError, "logits[0] deve ser numérico"),
        )

        for api_name, function, needs_target in apis:
            for case_name, logits, error_type, expected_message in invalid_logits:
                with self.subTest(api=api_name, case=case_name):
                    with self.assertRaises(error_type) as raised:
                        if needs_target:
                            function(logits, 0)
                        else:
                            function(logits)
                    self.assertEqual(str(raised.exception), expected_message)

    def test_public_cross_entropy_apis_reject_invalid_targets_with_exact_errors(self) -> None:
        apis: tuple[tuple[str, Any], ...] = (
            ("probability.cross_entropy", cross_entropy),
            ("probability.cross_entropy_gradient", cross_entropy_gradient),
            ("losses.cross_entropy", losses_module.cross_entropy),
            ("losses.cross_entropy_gradient", losses_module.cross_entropy_gradient),
        )
        invalid_targets: tuple[tuple[str, Any, type[Exception], str], ...] = (
            ("bool", True, TypeError, "target_id deve ser inteiro"),
            ("string", "1", TypeError, "target_id deve ser inteiro"),
            ("none", None, TypeError, "target_id deve ser inteiro"),
            ("float", 1.0, TypeError, "target_id deve ser inteiro"),
            ("negative", -1, ValueError, "target_id fora do vocabulário"),
            ("out_of_range", 2, ValueError, "target_id fora do vocabulário"),
        )

        for api_name, function in apis:
            for case_name, target, error_type, expected_message in invalid_targets:
                with self.subTest(api=api_name, case=case_name):
                    with self.assertRaises(error_type) as raised:
                        function([0.0, 1.0], target)
                    self.assertEqual(str(raised.exception), expected_message)


class AutoregressiveModelTests(unittest.TestCase):
    def _model(self) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=3,
            syllable_vocab_size=2,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=7,
        )

    def _inputs(self) -> tuple[list[int], list[int], list[list[float]]]:
        return [0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]]

    def test_forward_returns_logits_and_prediction(self) -> None:
        model = self._model()

        logits = model.forward(*self._inputs())

        self.assertEqual(len(logits), 3)
        self.assertEqual(model.predict(*self._inputs()), max(range(3), key=logits.__getitem__))

    def test_backward_and_step_update_the_model(self) -> None:
        model = self._model()
        inputs = self._inputs()
        before = [row.copy() for row in model.output_layer.weights]

        loss = model.train_step(*inputs, target_id=2, learning_rate=0.1)

        self.assertGreater(loss, 0.0)
        self.assertNotEqual(before, model.output_layer.weights)

    def test_model_can_overfit_one_example(self) -> None:
        model = self._model()
        inputs = self._inputs()
        initial_loss = model.loss(*inputs, target_id=2)

        for _ in range(250):
            model.train_step(*inputs, target_id=2, learning_rate=0.2)

        final_loss = model.loss(*inputs, target_id=2)
        self.assertLess(final_loss, initial_loss * 0.1)
        self.assertEqual(model.predict(*inputs), 2)

    def test_nonfinite_inputs_are_rejected(self) -> None:
        model = self._model()

        with self.assertRaises(ValueError):
            model.forward([0, 1], [0, 1], [[math.nan, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
