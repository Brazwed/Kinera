import math
import unittest

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.embedding import Embedding


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

    def test_repeated_backward_calls_accumulate_until_the_next_forward(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        embedding.forward([2, 0, 2])

        embedding.backward([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        embedding.backward([[2.0, 1.0], [4.0, 3.0], [1.0, 2.0]])

        self.assertEqual(
            embedding.grad_weights,
            [[7.0, 7.0], [0.0, 0.0], [9.0, 11.0]],
        )

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


class EmbeddingBackendIntegrationTests(unittest.TestCase):
    def test_embedding_delegates_lookup_gradient_and_update_to_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def embedding_forward(self, weights, ids):
                self.calls.append("forward")
                return super().embedding_forward(weights, ids)

            def embedding_backward(self, grad_weights, ids, grad_output):
                self.calls.append("backward")
                return super().embedding_backward(grad_weights, ids, grad_output)

            def embedding_step(self, weights, grad_weights, learning_rate):
                self.calls.append("step")
                return super().embedding_step(weights, grad_weights, learning_rate)

        backend = RecordingBackend()
        embedding = Embedding(vocab_size=2, embedding_dim=2, seed=7, backend=backend)

        embedding.forward([0])
        embedding.backward([[1.0, 2.0]])
        embedding.step(0.01)

        self.assertEqual(backend.calls, ["forward", "backward", "step"])

    def test_only_forward_uses_fresh_backend_zero_buffers(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.zero_buffers: list[object] = []
                self.host_buffer_calls: list[tuple[tuple[int, ...], str]] = []

            def buffer_from_host(self, values, *, shape, dtype="float64"):
                self.host_buffer_calls.append((shape, dtype))
                return super().buffer_from_host(values, shape=shape, dtype=dtype)

            def buffer_zeros(self, *, shape, dtype="float64"):
                buffer = super().buffer_zeros(shape=shape, dtype=dtype)
                self.zero_buffers.append(buffer)
                return buffer

        backend = RecordingBackend()
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        initial_gradients = embedding.grad_weights

        self.assertEqual(backend.zero_buffers, [])
        self.assertEqual(
            backend.host_buffer_calls,
            [((3, 2), "float64"), ((3, 2), "float64")],
        )

        embedding.forward([2, 0])
        first_gradients = embedding.grad_weights
        self.assertEqual(backend.zero_buffers, [first_gradients])
        self.assertIsNot(first_gradients, initial_gradients)
        self.assertEqual(first_gradients, [[0.0, 0.0] for _ in range(3)])
        self.assertEqual(backend.host_buffer_calls[-1], ((2,), "int32"))

        embedding.forward([1])
        self.assertEqual(len(backend.zero_buffers), 2)
        self.assertIsNot(backend.zero_buffers[0], backend.zero_buffers[1])
        self.assertIs(embedding.grad_weights, backend.zero_buffers[1])


if __name__ == "__main__":
    unittest.main()
