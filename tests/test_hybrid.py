import math
import unittest
from copy import deepcopy
from typing import cast

from kinera.backends.base import BackendArray, BackendBuffer, BackendUnavailableError
from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.embedding import Embedding
from kinera.nn.hybrid import HybridRepresentation


class NonSequenceBackendBuffer:
    device = "cuda"
    device_id = 0
    dtype = "float64"
    layout = "C"
    owns_memory = True
    closed = False

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.nbytes = 8 * shape[0] * shape[1]

    def close(self) -> None:
        self.closed = True


class ResidentBuffer:
    device = "cuda"
    device_id = 0
    layout = "row_major"
    owns_memory = True

    def __init__(
        self, values: object, shape: tuple[int, ...], dtype: str = "float64"
    ) -> None:
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.nbytes = (4 if dtype == "int32" else 8) * math.prod(shape)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ResidentHybridBackend(CPUBackend):
    name = "cuda"

    def __init__(self) -> None:
        self.embedding_backward_inputs: list[list[list[float]]] = []

    def buffer_from_host(self, values, *, shape, dtype="float64"):  # type: ignore[override]
        return ResidentBuffer(deepcopy(values), shape, dtype)

    def buffer_to_host(self, buffer):
        raise AssertionError("resident representation backward must not download")

    def embedding_forward(self, weights, ids):  # type: ignore[override]
        rows = [weights.values[index][:] for index in ids.values]
        return ResidentBuffer(rows, (len(rows), weights.shape[1]))

    def hybrid_representation_forward(self, char_vectors, feature_vectors, syllable_vectors):  # type: ignore[override]
        values = [
            char + feature + syllable
            for char, feature, syllable in zip(
                char_vectors.values, feature_vectors.values, syllable_vectors.values
            )
        ]
        return ResidentBuffer(values, (len(values), len(values[0])))

    def hybrid_representation_backward(
        self, grad_output, *, char_dim, feature_dim, syllable_dim
    ):
        char_rows = [row[:char_dim] for row in grad_output.values]
        feature_rows = [
            row[char_dim : char_dim + feature_dim] for row in grad_output.values
        ]
        syllable_rows = [row[char_dim + feature_dim :] for row in grad_output.values]
        context = len(grad_output.values)
        return (
            ResidentBuffer(char_rows, (context, char_dim)),
            ResidentBuffer(feature_rows, (context, feature_dim)),
            ResidentBuffer(syllable_rows, (context, syllable_dim)),
        )

    def embedding_backward(self, grad_weights, ids, grad_output):
        self.embedding_backward_inputs.append(deepcopy(grad_output.values))
        return grad_weights


class RecordingHybridBackend(CPUBackend):
    def __init__(self) -> None:
        self.hybrid_calls: list[
            tuple[list[list[float]], list[list[float]], list[list[float]]]
        ] = []

    def hybrid_representation_forward(
        self,
        char_vectors: BackendArray,
        feature_vectors: BackendArray,
        syllable_vectors: BackendArray,
    ) -> list[list[float]]:
        if not all(
            isinstance(matrix, list)
            for matrix in (char_vectors, feature_vectors, syllable_vectors)
        ):
            raise TypeError("recording CPU backend accepts host matrices only")
        char_values = cast(list[list[float]], char_vectors)
        feature_values = cast(list[list[float]], feature_vectors)
        syllable_values = cast(list[list[float]], syllable_vectors)
        self.hybrid_calls.append((char_values, feature_values, syllable_values))
        return [
            char_vector + feature_vector + syllable_vector
            for char_vector, feature_vector, syllable_vector in zip(
                char_values, feature_values, syllable_values
            )
        ]


class HybridRepresentationTests(unittest.TestCase):
    def test_cpu_backend_hybrid_forward_preserves_declared_channel_order(self) -> None:
        output = CPUBackend().hybrid_representation_forward(
            [[0.1, 0.2], [0.3, 0.4]],
            [[1.0], [0.0]],
            [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        )

        self.assertEqual(
            output,
            [[0.1, 0.2, 1.0, 0.5, 0.6, 0.7], [0.3, 0.4, 0.0, 0.8, 0.9, 1.0]],
        )

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

    def test_forward_preserves_empty_aligned_inputs(self) -> None:
        representation = self._representation()

        output = representation.forward([], [], [])

        self.assertEqual(output, [])
        self.assertEqual(representation.last_output, [])

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

    def test_forward_delegates_fusion_to_the_selected_backend(self) -> None:
        backend = RecordingHybridBackend()
        char_embedding = Embedding(
            vocab_size=4, embedding_dim=2, seed=7, backend=backend
        )
        syllable_embedding = Embedding(
            vocab_size=3, embedding_dim=3, seed=8, backend=backend
        )
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
        representation = HybridRepresentation(
            char_embedding,
            syllable_embedding,
            feature_dim=2,
        )

        output = representation.forward(
            char_ids=[2],
            syllable_ids=[1],
            feature_vectors=[[1.0, 0.0]],
        )

        self.assertEqual(len(backend.hybrid_calls), 1)
        char_vectors, features, syllable_vectors = backend.hybrid_calls[0]
        self.assertEqual(char_vectors, [[0.5, 0.6]])
        self.assertEqual(features, [[1.0, 0.0]])
        self.assertEqual(syllable_vectors, [[1.3, 1.4, 1.5]])
        self.assertEqual(output, [char_vectors[0] + features[0] + syllable_vectors[0]])

    def test_cuda_backward_splits_resident_gradients_after_forward_release(self) -> None:
        backend = ResidentHybridBackend()
        representation = HybridRepresentation(
            Embedding(4, 2, seed=7, backend=backend),
            Embedding(3, 3, seed=8, backend=backend),
            feature_dim=2,
        )
        output = representation.forward(
            [2, 0], [1, 2], [[1.0, 0.0], [0.0, 1.0]]
        )
        self.assertIsInstance(output, ResidentBuffer)
        representation.release_output()
        self.assertTrue(output.closed)
        grad_output = ResidentBuffer([[1.0] * 7, [2.0] * 7], (2, 7))

        feature_gradients = representation.backward(grad_output)

        self.assertIsInstance(feature_gradients, ResidentBuffer)
        self.assertEqual(feature_gradients.values, [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(
            backend.embedding_backward_inputs,
            [[[1.0, 1.0], [2.0, 2.0]], [[1.0] * 3, [2.0] * 3]],
        )
        self.assertFalse(backend.embedding_backward_inputs == [])

    def test_cuda_backward_fails_closed_until_phase_11b(self) -> None:
        representation = self._representation()
        representation.last_output = NonSequenceBackendBuffer((2, 7))

        with self.assertRaisesRegex(
            BackendUnavailableError, "HybridRepresentation.backward"
        ):
            representation.backward([[1.0] * 7, [1.0] * 7])

    def test_cpu_backend_mean_pool_matches_python_float_sum_order(self) -> None:
        rows = [[1e16, 2.0], [1.0, 4.0], [-1e16, 6.0]]
        pooled = CPUBackend().mean_pool_forward(rows)
        expected_first = sum(row[0] for row in rows) / len(rows)

        self.assertEqual(pooled, [expected_first, 4.0])

    def test_invalid_lengths_are_rejected(self) -> None:
        representation = self._representation()

        with self.assertRaises(ValueError):
            representation.forward([1], [1, 2], [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            representation.forward([1], [1], [[1.0]])


if __name__ == "__main__":
    unittest.main()
