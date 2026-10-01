import math
import unittest

from nn.heads import MultiTaskHeads
from nn.losses import MultiTaskLoss, binary_cross_entropy_with_logits
from training.autoregressive import AutoregressiveModel


class MultiTaskLossTests(unittest.TestCase):
    def test_binary_cross_entropy_with_logits_is_stable(self) -> None:
        self.assertLess(binary_cross_entropy_with_logits(1000.0, 1.0), 1e-10)
        self.assertLess(binary_cross_entropy_with_logits(-1000.0, 0.0), 1e-10)

    def test_forward_returns_weighted_component_losses(self) -> None:
        loss = MultiTaskLoss(char_weight=1.0, syllable_weight=0.7, boundary_weight=0.3)
        outputs = {
            "char": [1.0, 0.0],
            "syllable": [0.0, 1.0, -1.0],
            "boundary": [0.5],
        }

        report = loss.forward(
            outputs,
            {"char": 0, "syllable": 1, "boundary": 1.0},
        )

        self.assertEqual(set(report), {"char_loss", "syllable_loss", "boundary_loss", "total_loss"})
        self.assertAlmostEqual(
            report["total_loss"],
            report["char_loss"] + 0.7 * report["syllable_loss"] + 0.3 * report["boundary_loss"],
        )

    def test_backward_returns_gradients_for_all_heads(self) -> None:
        loss = MultiTaskLoss()
        outputs = {
            "char": [0.0, 1.0],
            "syllable": [1.0, 0.0],
            "boundary": [-1.0],
        }
        targets = {"char": 1, "syllable": 0, "boundary": 1.0}

        gradients = loss.backward(outputs, targets)

        self.assertEqual(set(gradients), {"char", "syllable", "boundary"})
        self.assertEqual(len(gradients["char"]), 2)
        self.assertEqual(len(gradients["syllable"]), 2)
        self.assertEqual(len(gradients["boundary"]), 1)
        self.assertTrue(all(math.isfinite(value) for values in gradients.values() for value in values))

    def test_invalid_targets_are_rejected(self) -> None:
        loss = MultiTaskLoss()
        outputs = {"char": [0.0, 1.0], "syllable": [0.0, 1.0], "boundary": [0.0]}

        with self.assertRaises(ValueError):
            loss.forward(outputs, {"char": 2, "syllable": 0, "boundary": 0.0})
        with self.assertRaises(ValueError):
            loss.forward(outputs, {"char": 0, "syllable": 0, "boundary": 2.0})


class MultiTaskGradientCheckingTests(unittest.TestCase):
    def test_combined_loss_gradient_matches_finite_difference_for_each_head(self) -> None:
        heads = MultiTaskHeads(input_dim=2, char_vocab_size=2, syllable_vocab_size=2, seed=7)
        loss = MultiTaskLoss()
        inputs = [0.4, -0.3]
        targets = {"char": 1, "syllable": 0, "boundary": 1.0}
        epsilon = 1e-6

        outputs = heads.forward(inputs)
        gradients = loss.backward(outputs, targets)
        input_gradients = heads.backward(gradients)
        analytic_gradients = {
            "char": heads.char.grad_weights[0][0],
            "syllable": heads.syllable.grad_weights[0][0],
            "boundary": heads.boundary.grad_weights[0][0],
        }

        for name, layer in (
            ("char", heads.char),
            ("syllable", heads.syllable),
            ("boundary", heads.boundary),
        ):
            original = layer.weights[0][0]
            layer.weights[0][0] = original + epsilon
            loss_plus = loss.forward(heads.forward(inputs), targets)["total_loss"]
            layer.weights[0][0] = original - epsilon
            loss_minus = loss.forward(heads.forward(inputs), targets)["total_loss"]
            layer.weights[0][0] = original
            numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(
                numerical,
                analytic_gradients[name],
                delta=1e-5,
                msg=name,
            )

        self.assertEqual(len(input_gradients["char"]), 2)
        self.assertEqual(len(input_gradients["syllable"]), 2)
        self.assertEqual(len(input_gradients["boundary"]), 2)


class MultiTaskModelTests(unittest.TestCase):
    def test_multitask_train_step_updates_all_heads(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=3,
            syllable_vocab_size=4,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=7,
        )
        inputs = ([0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]])
        before = {
            name: [row.copy() for row in layer.weights]
            for name, layer in (
                ("char", model.heads.char),
                ("syllable", model.heads.syllable),
                ("boundary", model.heads.boundary),
            )
        }

        report = model.multitask_train_step(
            *inputs,
            targets={"char": 2, "syllable": 3, "boundary": 1.0},
            learning_rate=0.1,
        )

        self.assertIn("total_loss", report)
        self.assertNotEqual(before["char"], model.heads.char.weights)
        self.assertNotEqual(before["syllable"], model.heads.syllable.weights)
        self.assertNotEqual(before["boundary"], model.heads.boundary.weights)


if __name__ == "__main__":
    unittest.main()
