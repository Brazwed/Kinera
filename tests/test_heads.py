import math
import unittest
from unittest.mock import patch

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.heads import MultiTaskHeads
from kinera.training.autoregressive import AutoregressiveModel


class MultiTaskHeadsTests(unittest.TestCase):
    class TemporaryBuffer:
        owns_memory = True
        closed = False

        def close(self) -> None:
            self.closed = True

    def test_backward_failure_releases_prior_backend_input_gradients(self) -> None:
        heads = MultiTaskHeads(input_dim=2, char_vocab_size=3, syllable_vocab_size=4, seed=7)
        char_gradient = self.TemporaryBuffer()
        gradients = {
            "char": [0.0, 0.0, 0.0],
            "syllable": [0.0, 0.0, 0.0, 0.0],
            "boundary": [0.0],
        }
        with (
            patch.object(heads.char, "backward", return_value=char_gradient),
            patch.object(heads.syllable, "backward", side_effect=RuntimeError("test failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                heads.backward(gradients)

        self.assertTrue(char_gradient.closed)

    def test_forward_returns_character_syllable_and_boundary_logits(self) -> None:
        heads = MultiTaskHeads(
            input_dim=4,
            char_vocab_size=5,
            syllable_vocab_size=7,
            seed=7,
        )

        outputs = heads.forward([0.1, -0.2, 0.3, 0.4])

        self.assertEqual(set(outputs), {"char", "syllable", "boundary"})
        self.assertEqual(len(outputs["char"]), 5)
        self.assertEqual(len(outputs["syllable"]), 7)
        self.assertEqual(len(outputs["boundary"]), 1)
        self.assertTrue(all(math.isfinite(value) for values in outputs.values() for value in values))

    def test_heads_share_input_dimension_and_are_seed_reproducible(self) -> None:
        first = MultiTaskHeads(input_dim=4, char_vocab_size=5, syllable_vocab_size=7, seed=7)
        second = MultiTaskHeads(input_dim=4, char_vocab_size=5, syllable_vocab_size=7, seed=7)

        self.assertEqual(first.char.weights, second.char.weights)
        self.assertEqual(first.syllable.weights, second.syllable.weights)
        self.assertEqual(first.boundary.weights, second.boundary.weights)
        self.assertEqual(len(first.char.weights[0]), 4)
        self.assertEqual(len(first.syllable.weights[0]), 4)
        self.assertEqual(len(first.boundary.weights[0]), 4)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiTaskHeads(input_dim=0, char_vocab_size=5, syllable_vocab_size=7)
        with self.assertRaises(ValueError):
            MultiTaskHeads(input_dim=4, char_vocab_size=0, syllable_vocab_size=7)
        with self.assertRaises(ValueError):
            MultiTaskHeads(input_dim=4, char_vocab_size=5, syllable_vocab_size=0)

    def test_forward_rejects_nonfinite_shared_representation(self) -> None:
        heads = MultiTaskHeads(input_dim=2, char_vocab_size=3, syllable_vocab_size=4)

        with self.assertRaises(ValueError):
            heads.forward([math.nan, 0.0])


class AutoregressiveHeadIntegrationTests(unittest.TestCase):
    def test_model_exposes_all_heads_over_the_shared_representation(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=5,
            syllable_vocab_size=7,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=7,
        )

        outputs = model.forward_heads([0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(len(outputs["char"]), 5)
        self.assertEqual(len(outputs["syllable"]), 7)
        self.assertEqual(len(outputs["boundary"]), 1)
        self.assertIs(model.output_layer, model.heads.char)


class HeadsBackendIntegrationTests(unittest.TestCase):
    def test_all_heads_share_the_injected_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.linear_forward_calls = 0

            def linear_forward(self, weights, bias, inputs):
                self.linear_forward_calls += 1
                return super().linear_forward(weights, bias, inputs)

        backend = RecordingBackend()
        heads = MultiTaskHeads(
            input_dim=2,
            char_vocab_size=3,
            syllable_vocab_size=4,
            seed=7,
            backend=backend,
        )

        heads.forward([0.1, 0.2])

        self.assertEqual(backend.linear_forward_calls, 3)
        self.assertIs(heads.char.backend, backend)
        self.assertIs(heads.syllable.backend, backend)
        self.assertIs(heads.boundary.backend, backend)


if __name__ == "__main__":
    unittest.main()
