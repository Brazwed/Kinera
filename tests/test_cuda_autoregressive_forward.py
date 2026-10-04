"""Phase 11A CUDA forward and ownership gates."""

from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from typing import cast
from unittest.mock import patch

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.layers.embedding import Embedding
from kinera.nn.layers.hybrid import HybridRepresentation
from kinera.nn.models.hybrid import AutoregressiveModel

RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class CudaForwardOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)

    def test_embedding_forward_closes_replaced_buffers_without_d2h(self) -> None:
        embedding = Embedding(vocab_size=4, embedding_dim=3, seed=73, backend=self.backend)
        weights = cast(CudaBuffer, embedding.weights)
        initial_gradients = cast(CudaBuffer, embedding.grad_weights)
        owned = [weights, initial_gradients]
        original_copy_to_host = self.backend.memory.copy_to_host

        def reject_implicit_download(*args, **kwargs):
            raise AssertionError("Embedding.forward must not copy device data to host")

        self.backend.memory.copy_to_host = reject_implicit_download
        try:
            first_output = cast(CudaBuffer, embedding.forward([1, 2]))
            first_ids = cast(CudaBuffer, embedding.last_ids)
            first_gradients = cast(CudaBuffer, embedding.grad_weights)
            owned.extend([first_output, first_ids, first_gradients])
            self.assertTrue(initial_gradients.closed)

            second_output = cast(CudaBuffer, embedding.forward([2]))
            second_ids = cast(CudaBuffer, embedding.last_ids)
            second_gradients = cast(CudaBuffer, embedding.grad_weights)
            owned.extend([second_output, second_ids, second_gradients])

            self.assertTrue(first_output.closed)
            self.assertTrue(first_ids.closed)
            self.assertTrue(first_gradients.closed)
            self.assertFalse(second_output.closed)
            self.assertFalse(second_ids.closed)
            self.assertFalse(second_gradients.closed)
            self.assertEqual(second_output.shape, (1, 3))
            self.assertEqual(second_ids.shape, (1,))
        finally:
            self.backend.memory.copy_to_host = original_copy_to_host
            for buffer in owned:
                if not buffer.closed:
                    buffer.close()

        self.assertTrue(all(buffer.closed for buffer in owned))


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class CudaHybridRepresentationBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu_backend = CPUBackend()

    def test_resident_channel_concat_matches_cpu_without_implicit_transfers(self) -> None:
        char_values = [[0.1, 0.2], [0.3, 0.4]]
        feature_values = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]
        syllable_values = [[0.6], [0.7]]
        expected = self.cpu_backend.hybrid_representation_forward(
            char_values, feature_values, syllable_values
        )
        output: CudaBuffer | None = None
        with ExitStack() as stack:
            char_vectors = stack.enter_context(
                self.backend.buffer_from_host(char_values, shape=(2, 2), dtype="float64")
            )
            features = stack.enter_context(
                self.backend.buffer_from_host(feature_values, shape=(2, 3), dtype="float64")
            )
            syllable_vectors = stack.enter_context(
                self.backend.buffer_from_host(syllable_values, shape=(2, 1), dtype="float64")
            )
            with (
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=AssertionError("hybrid concat performed implicit H2D"),
                ),
                patch.object(
                    self.backend.memory,
                    "copy_to_host",
                    side_effect=AssertionError("hybrid concat performed implicit D2H"),
                ),
            ):
                output = self.backend.hybrid_representation_forward(
                    char_vectors, features, syllable_vectors
                )
            stack.enter_context(output)
            self.assertEqual(output.shape, (2, 6))
            self.assertEqual(output.dtype, "float64")
            self.assertEqual(output.device, "cuda")
            self.assertEqual(output.device_id, 0)
            actual = cast(list[list[float]], self.backend.buffer_to_host(output))
            self.assertEqual(actual, expected)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class CudaMeanPoolBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu_backend = CPUBackend()

    def test_mean_pool_matches_cpu_sum_without_reordering_or_transfers(self) -> None:
        values = [[1e16, 2.0], [1.0, 4.0], [-1e16, 6.0]]
        expected = self.cpu_backend.mean_pool_forward(values)
        with ExitStack() as stack:
            vectors = stack.enter_context(
                self.backend.buffer_from_host(values, shape=(3, 2), dtype="float64")
            )
            with (
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=AssertionError("mean pooling performed implicit H2D"),
                ),
                patch.object(
                    self.backend.memory,
                    "copy_to_host",
                    side_effect=AssertionError("mean pooling performed implicit D2H"),
                ),
            ):
                pooled = self.backend.mean_pool_forward(vectors)
            stack.enter_context(pooled)
            self.assertEqual(pooled.shape, (2,))
            self.assertEqual(pooled.dtype, "float64")
            self.assertEqual(pooled.device, "cuda")
            self.assertEqual(pooled.device_id, 0)
            actual = cast(list[float], self.backend.buffer_to_host(pooled))
            self.assertEqual(actual, expected)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class CudaArgmaxBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu_backend = CPUBackend()

    def test_argmax_matches_first_tie_and_downloads_only_one_int32(self) -> None:
        values = [0.0, 5.0, 5.0, -1.0]
        expected = self.cpu_backend.argmax(values)
        with ExitStack() as stack:
            logits = stack.enter_context(
                self.backend.buffer_from_host(values, shape=(4,), dtype="float64")
            )
            allocated: list[CudaBuffer] = []
            downloads: list[tuple[str, tuple[int, ...], int]] = []
            original_allocate = self.backend.memory.allocate
            original_copy_to_host = self.backend.memory.copy_to_host

            def record_allocate(*args, **kwargs) -> CudaBuffer:
                buffer = original_allocate(*args, **kwargs)
                allocated.append(buffer)
                return buffer

            def record_d2h(buffer: CudaBuffer):
                downloads.append((buffer.dtype, buffer.shape, buffer.nbytes))
                return original_copy_to_host(buffer)

            with (
                patch.object(self.backend.memory, "allocate", side_effect=record_allocate),
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=AssertionError("argmax performed implicit H2D"),
                ),
                patch.object(self.backend.memory, "copy_to_host", side_effect=record_d2h),
            ):
                predicted = self.backend.argmax(logits)

            self.assertEqual(predicted, expected)
            self.assertEqual(predicted, 1)
            self.assertEqual(len(allocated), 1)
            self.assertEqual(allocated[0].shape, (1,))
            self.assertEqual(allocated[0].dtype, "int32")
            self.assertTrue(allocated[0].closed)
            self.assertEqual(downloads, [("int32", (1,), 4)])
            self.assertFalse(logits.closed)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class HybridRepresentationCudaForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)
        cls.cpu_backend = CPUBackend()

    def test_forward_fuses_resident_embeddings_and_explicit_feature_upload(self) -> None:
        char_ids = [2, 0]
        syllable_ids = [1, 2]
        features = [[1.0, 0.0], [0.0, 1.0]]
        cpu_representation = HybridRepresentation(
            Embedding(4, 2, seed=7, backend=self.cpu_backend),
            Embedding(3, 3, seed=8, backend=self.cpu_backend),
            feature_dim=2,
        )
        cuda_representation = HybridRepresentation(
            Embedding(4, 2, seed=7, backend=self.backend),
            Embedding(3, 3, seed=8, backend=self.backend),
            feature_dim=2,
        )
        expected = cpu_representation.forward(char_ids, syllable_ids, features)
        uploads: list[CudaBuffer] = []
        output: CudaBuffer | None = None
        original_copy_to_device = self.backend.memory.copy_to_device

        def record_upload(*args, **kwargs) -> CudaBuffer:
            buffer = original_copy_to_device(*args, **kwargs)
            uploads.append(buffer)
            return buffer

        def reject_d2h(*args, **kwargs):
            raise AssertionError("hybrid representation forward performed an implicit D2H copy")

        try:
            with (
                patch.object(self.backend.memory, "copy_to_device", side_effect=record_upload),
                patch.object(self.backend.memory, "copy_to_host", side_effect=reject_d2h),
            ):
                output = cast(
                    CudaBuffer,
                    cuda_representation.forward(char_ids, syllable_ids, features),
                )
                feature_buffers = [
                    buffer
                    for buffer in uploads
                    if buffer.shape == (2, 2) and buffer.dtype == "float64"
                ]
                self.assertEqual(len(feature_buffers), 1)
                self.assertTrue(feature_buffers[0].closed)
                self.assertTrue(cast(CudaBuffer, cuda_representation.char_embedding.output).closed)
                self.assertTrue(cast(CudaBuffer, cuda_representation.syllable_embedding.output).closed)
                self.assertEqual(output.shape, (2, 7))
                self.assertEqual(output.dtype, "float64")
                self.assertEqual(output.device, "cuda")
                self.assertEqual(output.device_id, 0)
                self.assertFalse(output.closed)

            actual = cast(list[list[float]], self.backend.buffer_to_host(output))
            self.assertEqual(len(actual), len(expected))
            for row, expected_row in enumerate(expected):
                for column, expected_value in enumerate(expected_row):
                    self.assertLessEqual(
                        abs(actual[row][column] - expected_value),
                        ATOL + RTOL * abs(expected_value),
                        msg=f"row={row} column={column}",
                    )
        finally:
            owned = list(uploads)
            for embedding in (
                cuda_representation.char_embedding,
                cuda_representation.syllable_embedding,
            ):
                owned.extend(
                    value
                    for value in (
                        embedding.weights,
                        embedding.grad_weights,
                        embedding.last_ids,
                        embedding.output,
                    )
                    if isinstance(value, CudaBuffer)
                )
            if output is not None:
                owned.append(output)
            seen: set[int] = set()
            for buffer in owned:
                if id(buffer) not in seen and not buffer.closed:
                    seen.add(id(buffer))
                    buffer.close()


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class AutoregressiveModelCudaHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)

    @staticmethod
    def _model(backend) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=4,
            syllable_vocab_size=3,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=11,
            backend=backend,
        )

    def _close_model_buffers(self, model: AutoregressiveModel) -> None:
        buffers: list[CudaBuffer] = []
        for value in (
            model.last_logits,
            model.last_pooled,
            *getattr(model, "_last_head_logits_backend", {}).values(),
            model.representation.last_output,
        ):
            if isinstance(value, CudaBuffer):
                buffers.append(value)
        for embedding in (
            model.representation.char_embedding,
            model.representation.syllable_embedding,
        ):
            for name in ("weights", "grad_weights", "last_ids", "output"):
                value = getattr(embedding, name)
                if isinstance(value, CudaBuffer):
                    buffers.append(value)
        for head in (model.heads.char, model.heads.syllable, model.heads.boundary):
            for name in (
                "weights",
                "bias",
                "grad_weights",
                "grad_bias",
                "last_input",
                "output",
            ):
                value = getattr(head, name)
                if isinstance(value, CudaBuffer):
                    buffers.append(value)
        seen: set[int] = set()
        for buffer in buffers:
            if id(buffer) not in seen and not buffer.closed:
                seen.add(id(buffer))
                buffer.close()

    def test_heads_forward_keeps_logits_resident_and_shares_one_pooled_buffer(self) -> None:
        inputs = ([2, 0], [1, 2], [[1.0, 0.0], [0.0, 1.0]])
        cpu_model = self._model(CPUBackend())
        expected = cpu_model.forward_heads(*inputs)
        cuda_model = self._model(self.backend)
        outputs: dict[str, CudaBuffer] = {}
        original_copy_to_host = self.backend.memory.copy_to_host

        def reject_d2h(*args, **kwargs):
            raise AssertionError("_forward_heads_backend performed an implicit D2H copy")

        try:
            self.backend.memory.copy_to_host = reject_d2h
            outputs = cuda_model._forward_heads_backend(*inputs)
            self.assertEqual(set(outputs), {"char", "syllable", "boundary"})
            self.assertTrue(all(isinstance(value, CudaBuffer) for value in outputs.values()))
            self.assertTrue(
                all(
                    value.dtype == "float64"
                    and value.device == "cuda"
                    and value.device_id == 0
                    for value in outputs.values()
                )
            )
            self.assertEqual(outputs["char"].shape, (4,))
            self.assertEqual(outputs["syllable"].shape, (3,))
            self.assertEqual(outputs["boundary"].shape, (1,))
            pooled = cast(CudaBuffer, cuda_model.last_pooled)
            self.assertEqual(pooled.shape, (7,))
            self.assertEqual(pooled.dtype, "float64")
            self.assertEqual(pooled.device, "cuda")
            self.assertEqual(pooled.device_id, 0)
            self.assertIs(cuda_model.heads.char.last_input, pooled)
            self.assertIs(cuda_model.heads.syllable.last_input, pooled)
            self.assertIs(cuda_model.heads.boundary.last_input, pooled)
            self.assertIs(cuda_model.last_logits, outputs["char"])
            self.assertIsNone(cuda_model.representation.last_output)
            self.assertEqual(cuda_model.last_context_size, 2)
        finally:
            self.backend.memory.copy_to_host = original_copy_to_host

        try:
            actual = {
                name: cast(list[float], self.backend.buffer_to_host(value))
                for name, value in outputs.items()
            }
            for name in expected:
                self.assertEqual(len(actual[name]), len(expected[name]))
                for index, expected_value in enumerate(expected[name]):
                    self.assertLessEqual(
                        abs(actual[name][index] - expected_value),
                        ATOL + RTOL * abs(expected_value),
                        msg=f"head={name} index={index}",
                    )

            first_pooled = pooled
            first_logits = tuple(outputs.values())
            public_copy_calls: list[object] = []

            def record_d2h(*args, **kwargs):
                public_copy_calls.append(args)
                return original_copy_to_host(*args, **kwargs)

            with patch.object(
                self.backend.memory, "copy_to_host", side_effect=record_d2h
            ):
                public_heads = cuda_model.forward_heads(*inputs)
            self.assertEqual(len(public_copy_calls), 3)
            self.assertTrue(all(isinstance(value, list) for value in public_heads.values()))
            self.assertTrue(first_pooled.closed)
            self.assertTrue(all(buffer.closed for buffer in first_logits))
            for name in expected:
                for index, expected_value in enumerate(expected[name]):
                    self.assertLessEqual(
                        abs(public_heads[name][index] - expected_value),
                        ATOL + RTOL * abs(expected_value),
                        msg=f"public head={name} index={index}",
                    )

            second_pooled = cast(CudaBuffer, cuda_model.last_pooled)
            second_logits = tuple(
                cast(CudaBuffer, buffer)
                for buffer in cuda_model._last_head_logits_backend.values()
            )
            expected_char = cpu_model.forward(*inputs)
            public_copy_calls.clear()
            with patch.object(
                self.backend.memory, "copy_to_host", side_effect=record_d2h
            ):
                public_char = cuda_model.forward(*inputs)
            self.assertEqual(len(public_copy_calls), 1)
            self.assertIsInstance(public_char, list)
            self.assertTrue(second_pooled.closed)
            self.assertTrue(all(buffer.closed for buffer in second_logits))
            for index, expected_value in enumerate(expected_char):
                self.assertLessEqual(
                    abs(public_char[index] - expected_value),
                    ATOL + RTOL * abs(expected_value),
                    msg=f"public char index={index}",
                )

            current_pooled = cast(CudaBuffer, cuda_model.last_pooled)
            current_logits = cast(CudaBuffer, cuda_model.last_logits)
            cuda_model.release_forward_state()
            self.assertTrue(current_pooled.closed)
            self.assertTrue(current_logits.closed)
            self.assertIsNone(cuda_model.last_pooled)
            self.assertIsNone(cuda_model.last_logits)
        finally:
            self._close_model_buffers(cuda_model)


    def test_partial_heads_failure_releases_all_device_forward_temporaries(self) -> None:
        inputs = ([2, 0], [1, 2], [[1.0, 0.0], [0.0, 1.0]])
        model = self._model(self.backend)
        created: list[CudaBuffer] = []
        original_allocate = self.backend.memory.allocate
        original_copy_to_device = self.backend.memory.copy_to_device
        original_linear_forward = self.backend.linear_forward
        linear_calls = 0

        def track_allocate(*args, **kwargs) -> CudaBuffer:
            buffer = original_allocate(*args, **kwargs)
            created.append(buffer)
            return buffer

        def track_copy_to_device(*args, **kwargs) -> CudaBuffer:
            buffer = original_copy_to_device(*args, **kwargs)
            created.append(buffer)
            return buffer

        def fail_third_head(*args, **kwargs):
            nonlocal linear_calls
            linear_calls += 1
            if linear_calls == 3:
                raise RuntimeError("injected third-head failure")
            return original_linear_forward(*args, **kwargs)

        def reject_d2h(*args, **kwargs):
            raise AssertionError("internal head failure path performed D2H")

        try:
            with (
                patch.object(self.backend.memory, "allocate", side_effect=track_allocate),
                patch.object(
                    self.backend.memory,
                    "copy_to_device",
                    side_effect=track_copy_to_device,
                ),
                patch.object(self.backend.memory, "copy_to_host", side_effect=reject_d2h),
                patch.object(self.backend, "linear_forward", side_effect=fail_third_head),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected third-head failure"):
                    model._forward_heads_backend(*inputs)

            self.assertEqual(linear_calls, 3)
            self.assertTrue(created)
            self.assertTrue(all(buffer.closed for buffer in created))
            self.assertIsNone(model.last_pooled)
            self.assertIsNone(model.last_logits)
            self.assertEqual(model._last_head_logits_backend, {})
            self.assertIsNone(model.last_context_size)
            self.assertIsNone(model.representation.last_output)
            for embedding in (
                model.representation.char_embedding,
                model.representation.syllable_embedding,
            ):
                self.assertIsNone(embedding.last_ids)
                self.assertIsNone(embedding.output)
            for head in (model.heads.char, model.heads.syllable, model.heads.boundary):
                self.assertIsNone(head.last_input)
                self.assertIsNone(head.output)
        finally:
            self._close_model_buffers(model)


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class AutoregressiveModelCudaPredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = CUDABackend()
        cls.backend.runtime.require_device(0)

    @staticmethod
    def _model(backend) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=4,
            syllable_vocab_size=3,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=17,
            backend=backend,
        )

    def _close_model(self, model: AutoregressiveModel) -> None:
        model.release_forward_state()
        buffers: list[CudaBuffer] = []
        for embedding in (
            model.representation.char_embedding,
            model.representation.syllable_embedding,
        ):
            for name in ("weights", "grad_weights", "last_ids", "output"):
                value = getattr(embedding, name)
                if isinstance(value, CudaBuffer):
                    buffers.append(value)
        for head in (model.heads.char, model.heads.syllable, model.heads.boundary):
            for name in ("weights", "bias", "grad_weights", "grad_bias"):
                value = getattr(head, name)
                if isinstance(value, CudaBuffer):
                    buffers.append(value)
        for buffer in buffers:
            if not buffer.closed:
                buffer.close()

    def test_predict_matches_cpu_tie_and_downloads_only_the_id(self) -> None:
        inputs = ([2, 0], [1, 2], [[1.0, 0.0], [0.0, 1.0]])
        cpu_model = self._model(CPUBackend())
        cuda_model = self._model(self.backend)
        char_weights = [[0.0] * 7 for _ in range(4)]
        char_bias = [0.0, 5.0, 5.0, -1.0]
        cpu_model.output_layer.weights = [row.copy() for row in char_weights]
        cpu_model.output_layer.bias = char_bias.copy()
        old_weights = cast(CudaBuffer, cuda_model.output_layer.weights)
        old_bias = cast(CudaBuffer, cuda_model.output_layer.bias)
        cuda_model.output_layer.weights = self.backend.buffer_from_host(
            char_weights, shape=(4, 7), dtype="float64"
        )
        cuda_model.output_layer.bias = self.backend.buffer_from_host(
            char_bias, shape=(4,), dtype="float64"
        )
        old_weights.close()
        old_bias.close()
        expected = cpu_model.predict(*inputs)
        downloads: list[tuple[str, tuple[int, ...], int]] = []
        original_copy_to_host = self.backend.memory.copy_to_host

        def record_d2h(buffer: CudaBuffer):
            downloads.append((buffer.dtype, buffer.shape, buffer.nbytes))
            return original_copy_to_host(buffer)

        try:
            with patch.object(
                self.backend.memory, "copy_to_host", side_effect=record_d2h
            ):
                predicted = cuda_model.predict(*inputs)
            self.assertEqual(predicted, expected)
            self.assertEqual(predicted, 1)
            self.assertEqual(downloads, [("int32", (1,), 4)])
            self.assertIsInstance(cuda_model.last_logits, CudaBuffer)
            self.assertFalse(cast(CudaBuffer, cuda_model.last_logits).closed)
            self.assertIsInstance(cuda_model.last_pooled, CudaBuffer)
        finally:
            self._close_model(cuda_model)


if __name__ == "__main__":
    unittest.main()
