import math
import unittest

from nn.probability import cross_entropy, cross_entropy_gradient, softmax
from training.autoregressive import AutoregressiveModel


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
