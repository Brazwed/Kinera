import math
import unittest

from nn.embedding import Embedding


class EmbeddingConstructionTests(unittest.TestCase):
    def test_constructor_creates_matrix_and_zero_gradients(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)

        self.assertEqual(len(embedding.weights), 3)
        self.assertTrue(all(len(row) == 2 for row in embedding.weights))
        self.assertEqual(embedding.grad_weights, [[0.0, 0.0]] * 3)
        self.assertIsNone(embedding.last_ids)
        self.assertIsNone(embedding.output)

    def test_same_seed_reproduces_weights(self) -> None:
        first = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        second = Embedding(vocab_size=3, embedding_dim=2, seed=7)

        self.assertEqual(first.weights, second.weights)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Embedding(vocab_size=0, embedding_dim=2)
        with self.assertRaises(ValueError):
            Embedding(vocab_size=2, embedding_dim=0)
        with self.assertRaises(TypeError):
            Embedding(vocab_size=True, embedding_dim=2)


class EmbeddingForwardTests(unittest.TestCase):
    def test_forward_looks_up_rows_copies_output_and_resets_stale_gradients(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        output = embedding.forward([2, 0, 2])

        self.assertEqual(output, [[0.5, 0.6], [0.1, 0.2], [0.5, 0.6]])
        output[0][0] = 99.0
        self.assertEqual(embedding.weights[2], [0.5, 0.6])
        self.assertEqual(embedding.last_ids, [2, 0, 2])

        training_embedding = Embedding(vocab_size=4, embedding_dim=2, seed=11)
        first_ids = [0, 1]
        second_ids = [2, 3]
        training_embedding.forward(first_ids)
        training_embedding.backward([[1.0, 2.0], [3.0, 4.0]])
        training_embedding.step(0.1)
        weights_after_first_step = [row.copy() for row in training_embedding.weights]
        self.assertNotEqual(training_embedding.grad_weights[0], [0.0, 0.0])

        training_embedding.forward(second_ids)

        self.assertEqual(training_embedding.grad_weights, [[0.0, 0.0] for _ in range(4)])
        self.assertTrue(
            all(
                training_embedding.grad_weights[index]
                is not training_embedding.grad_weights[index + 1]
                for index in range(3)
            )
        )
        training_embedding.backward([[0.0, 0.0], [0.0, 0.0]])
        training_embedding.step(0.1)
        self.assertEqual(training_embedding.weights, weights_after_first_step)

    def test_forward_rejects_invalid_ids(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)

        with self.assertRaises(TypeError):
            embedding.forward([True])
        with self.assertRaises(TypeError):
            embedding.forward(["1"])
        with self.assertRaises(ValueError):
            embedding.forward([-1])
        with self.assertRaises(ValueError):
            embedding.forward([3])


class EmbeddingBackwardTests(unittest.TestCase):
    def test_backward_accumulates_gradients_for_repeated_ids(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.forward([2, 0, 2])

        result = embedding.backward(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )

        self.assertIsNone(result)
        self.assertEqual(embedding.grad_weights[0], [3.0, 4.0])
        self.assertEqual(embedding.grad_weights[1], [0.0, 0.0])
        self.assertEqual(embedding.grad_weights[2], [6.0, 8.0])

    def test_backward_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.forward([1, 2])

        with self.assertRaises(ValueError):
            embedding.backward([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            embedding.backward([[1.0], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            embedding.backward([[1.0, math.nan], [2.0, 3.0]])

    def test_backward_before_forward_is_rejected(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)

        with self.assertRaises(RuntimeError):
            embedding.backward([[1.0, 2.0]])


class EmbeddingStepTests(unittest.TestCase):
    def test_step_updates_only_rows_seen_by_backward(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        embedding.forward([2, 0, 2])
        embedding.backward([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        embedding.step(0.1)

        self.assertAlmostEqual(embedding.weights[0][0], -0.2)
        self.assertAlmostEqual(embedding.weights[0][1], -0.2)
        self.assertEqual(embedding.weights[1], [0.3, 0.4])
        self.assertAlmostEqual(embedding.weights[2][0], -0.1)
        self.assertAlmostEqual(embedding.weights[2][1], -0.2)

    def test_step_twice_without_backward_is_rejected(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.forward([1])
        embedding.backward([[1.0, 2.0]])
        embedding.step(0.1)

        with self.assertRaises(RuntimeError):
            embedding.step(0.1)


if __name__ == "__main__":
    unittest.main()
