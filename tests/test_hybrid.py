import unittest

from kinera.nn.embedding import Embedding
from kinera.nn.hybrid import HybridRepresentation


class HybridRepresentationTests(unittest.TestCase):
    def _representation(self) -> HybridRepresentation:
        char_embedding = Embedding(vocab_size=4, embedding_dim=2, seed=7)
        syllable_embedding = Embedding(vocab_size=3, embedding_dim=3, seed=8)
        char_embedding.weights = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ]
        syllable_embedding.weights = [
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8],
        ]
        return HybridRepresentation(
            char_embedding,
            syllable_embedding,
            feature_dim=2,
        )

    def test_forward_concatenates_character_features_and_syllable_vectors(self) -> None:
        representation = self._representation()

        output = representation.forward(
            char_ids=[2, 0],
            syllable_ids=[1, 2],
            feature_vectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertEqual(
            output,
            [
                [0.5, 0.6, 1.0, 0.0, 1.3, 1.4, 1.5],
                [0.1, 0.2, 0.0, 1.0, 1.6, 1.7, 1.8],
            ],
        )

    def test_backward_splits_gradients_into_both_embeddings(self) -> None:
        representation = self._representation()
        representation.forward(
            char_ids=[2, 0],
            syllable_ids=[1, 2],
            feature_vectors=[[1.0, 0.0], [0.0, 1.0]],
        )

        feature_gradients = representation.backward(
            [[1.0] * 7, [2.0] * 7]
        )

        self.assertEqual(feature_gradients, [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(representation.char_embedding.grad_weights[2], [1.0, 1.0])
        self.assertEqual(representation.char_embedding.grad_weights[0], [2.0, 2.0])
        self.assertEqual(
            representation.syllable_embedding.grad_weights[1],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            representation.syllable_embedding.grad_weights[2],
            [2.0, 2.0, 2.0],
        )

    def test_step_updates_both_embedding_tables(self) -> None:
        representation = self._representation()
        representation.forward(
            char_ids=[2],
            syllable_ids=[1],
            feature_vectors=[[1.0, 0.0]],
        )
        representation.backward([[1.0] * 7])
        original_char = representation.char_embedding.weights[2].copy()
        original_syllable = representation.syllable_embedding.weights[1].copy()

        representation.step(0.1)

        self.assertNotEqual(representation.char_embedding.weights[2], original_char)
        self.assertNotEqual(
            representation.syllable_embedding.weights[1], original_syllable
        )

    def test_invalid_lengths_are_rejected(self) -> None:
        representation = self._representation()

        with self.assertRaises(ValueError):
            representation.forward([1], [1, 2], [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            representation.forward([1], [1], [[1.0]])


if __name__ == "__main__":
    unittest.main()
