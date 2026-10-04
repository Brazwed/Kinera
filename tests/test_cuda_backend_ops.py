"""CPU-only orchestration tests for CUDA Linear gradient dispatch."""

import ctypes
import math
import struct
from contextlib import ExitStack
import unittest
from unittest.mock import patch
from typing import cast

from kinera.backends.base import BackendError
from kinera.backends.cuda import CUDABackend, CudaBridge, CudaBuffer, CudaRuntime
from kinera.nn.layers.linear import Linear


class FakeCudaBridge(CudaBridge):
    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count
        self._next_pointer = 0x1000
        self._allocations: dict[int, bytearray] = {}
        self.backward_calls: list[tuple[int, ...]] = []
        self.linear_forward_calls: list[tuple[int, ...]] = []
        self.step_calls: list[tuple[int | float, ...]] = []
        self.copy_to_device_calls = 0
        self.copy_to_host_calls = 0
        self.copy_to_host_nbytes: list[int] = []
        self.embedding_forward_calls: list[tuple[int, ...]] = []
        self.embedding_backward_calls: list[tuple[int, ...]] = []
        self.embedding_step_calls: list[tuple[int | float, ...]] = []
        self.hybrid_forward_calls: list[tuple[int, ...]] = []
        self.mean_pool_calls: list[tuple[int, ...]] = []
        self.argmax_calls: list[tuple[int, ...]] = []
        self.cross_entropy_gradient_calls: list[tuple[int | float, ...]] = []
        self.cross_entropy_forward_calls: list[tuple[int | float, ...]] = []
        self.binary_cross_entropy_forward_calls: list[tuple[int | float, ...]] = []
        self.binary_cross_entropy_gradient_calls: list[tuple[int | float, ...]] = []
        self.head_gradient_sum_calls: list[tuple[int, ...]] = []
        self.mean_pool_backward_calls: list[tuple[int, ...]] = []
        self.hybrid_backward_calls: list[tuple[int, ...]] = []

    def device_count(self) -> int:
        return self._device_count

    def allocate(self, device_id: int, nbytes: int) -> int:
        if device_id < 0 or device_id >= self._device_count:
            raise BackendError("fake device unavailable")
        pointer = self._next_pointer
        self._next_pointer += 0x1000
        self._allocations[pointer] = bytearray(nbytes)
        return pointer

    def free(self, device_id: int, pointer: int) -> None:
        del self._allocations[pointer]

    def copy_to_device(self, device_id: int, destination: int, source: int, nbytes: int) -> None:
        self.copy_to_device_calls += 1
        self._allocations[destination][:nbytes] = ctypes.string_at(source, nbytes)

    def copy_to_host(self, device_id: int, destination: int, source: int, nbytes: int) -> None:
        self.copy_to_host_calls += 1
        self.copy_to_host_nbytes.append(nbytes)
        ctypes.memmove(destination, bytes(self._allocations[source][:nbytes]), nbytes)

    def linear_step_f64(
        self,
        device_id: int,
        weights: int,
        bias: int,
        grad_weights: int,
        grad_bias: int,
        learning_rate: float,
        input_size: int,
        output_size: int,
    ) -> None:
        self.step_calls.append(
            (
                device_id,
                weights,
                bias,
                grad_weights,
                grad_bias,
                learning_rate,
                input_size,
                output_size,
            )
        )

    def linear_forward_f64(
        self,
        device_id: int,
        weights: int,
        bias: int,
        inputs: int,
        output: int,
        input_size: int,
        output_size: int,
    ) -> None:
        self.linear_forward_calls.append(
            (device_id, weights, bias, inputs, output, input_size, output_size)
        )

    def linear_backward_f64(
        self,
        device_id: int,
        weights: int,
        inputs: int,
        grad_output: int,
        grad_weights: int,
        grad_bias: int,
        grad_inputs: int,
        input_size: int,
        output_size: int,
    ) -> None:
        self.backward_calls.append(
            (
                device_id,
                weights,
                inputs,
                grad_output,
                grad_weights,
                grad_bias,
                grad_inputs,
                input_size,
                output_size,
            )
        )
    def embedding_forward_f64(
        self,
        device_id: int,
        weights: int,
        ids: int,
        output: int,
        vocab_size: int,
        embedding_dim: int,
        ids_count: int,
    ) -> None:
        self.embedding_forward_calls.append(
            (device_id, weights, ids, output, vocab_size, embedding_dim, ids_count)
        )
    def embedding_backward_f64(
        self,
        device_id: int,
        ids: int,
        grad_output: int,
        grad_weights: int,
        vocab_size: int,
        embedding_dim: int,
        ids_count: int,
    ) -> None:
        self.embedding_backward_calls.append(
            (
                device_id,
                ids,
                grad_output,
                grad_weights,
                vocab_size,
                embedding_dim,
                ids_count,
            )
        )
    def embedding_step_f64(
        self,
        device_id: int,
        weights: int,
        grad_weights: int,
        learning_rate: float,
        vocab_size: int,
        embedding_dim: int,
    ) -> None:
        self.embedding_step_calls.append(
            (device_id, weights, grad_weights, learning_rate, vocab_size, embedding_dim)
        )

    def hybrid_representation_forward_f64(
        self,
        device_id: int,
        char_vectors: int,
        feature_vectors: int,
        syllable_vectors: int,
        output: int,
        context_size: int,
        char_dim: int,
        feature_dim: int,
        syllable_dim: int,
    ) -> None:
        self.hybrid_forward_calls.append(
            (
                device_id,
                char_vectors,
                feature_vectors,
                syllable_vectors,
                output,
                context_size,
                char_dim,
                feature_dim,
                syllable_dim,
            )
        )


    def mean_pool_forward_f64(
        self,
        device_id: int,
        vectors: int,
        output: int,
        context_size: int,
        feature_dim: int,
        compensated_sum: int,
    ) -> None:
        self.mean_pool_calls.append(
            (device_id, vectors, output, context_size, feature_dim, compensated_sum)
        )


    def argmax_f64(
        self,
        device_id: int,
        logits: int,
        output: int,
        logits_count: int,
    ) -> None:
        self.argmax_calls.append((device_id, logits, output, logits_count))
        values = [
            struct.unpack_from("=d", self._allocations[logits], index * ctypes.sizeof(ctypes.c_double))[0]
            for index in range(logits_count)
        ]
        result = next(
            (-(index + 1) for index, value in enumerate(values) if not math.isfinite(value)),
            max(range(logits_count), key=values.__getitem__),
        )
        struct.pack_into("=i", self._allocations[output], 0, result)


    def cross_entropy_gradient_f64(
        self,
        device_id: int,
        logits: int,
        gradient: int,
        status: int,
        logits_count: int,
        target_id: int,
        scale: float,
        compensated_sum: int,
    ) -> None:
        self.cross_entropy_gradient_calls.append(
            (device_id, logits, gradient, status, logits_count, target_id, scale, compensated_sum)
        )
        struct.pack_into("=i", self._allocations[status], 0, 0)


    def cross_entropy_forward_f64(
        self,
        device_id: int,
        logits: int,
        loss: int,
        status: int,
        logits_count: int,
        target_id: int,
        compensated_sum: int,
    ) -> None:
        self.cross_entropy_forward_calls.append(
            (device_id, logits, loss, status, logits_count, target_id, compensated_sum)
        )
        struct.pack_into("=d", self._allocations[loss], 0, 0.75)
        struct.pack_into("=i", self._allocations[status], 0, 0)

    def binary_cross_entropy_with_logits_forward_f64(
        self, device_id: int, logit: int, loss: int, status: int, target: float
    ) -> None:
        self.binary_cross_entropy_forward_calls.append(
            (device_id, logit, loss, status, target)
        )
        struct.pack_into("=d", self._allocations[loss], 0, 0.25)
        struct.pack_into("=i", self._allocations[status], 0, 0)

    def binary_cross_entropy_with_logits_gradient_f64(
        self,
        device_id: int,
        logit: int,
        gradient: int,
        status: int,
        target: float,
        scale: float,
    ) -> None:
        self.binary_cross_entropy_gradient_calls.append(
            (device_id, logit, gradient, status, target, scale)
        )
        struct.pack_into("=i", self._allocations[status], 0, 0)


    def sum_head_input_gradients_f64(
        self,
        device_id: int,
        char_gradient: int,
        syllable_gradient: int,
        boundary_gradient: int,
        output: int,
        feature_dim: int,
        compensated_sum: int,
    ) -> None:
        self.head_gradient_sum_calls.append(
            (device_id, char_gradient, syllable_gradient, boundary_gradient, output, feature_dim, compensated_sum)
        )

    def mean_pool_backward_f64(
        self,
        device_id: int,
        grad_pooled: int,
        grad_vectors: int,
        context_size: int,
        feature_dim: int,
    ) -> None:
        self.mean_pool_backward_calls.append(
            (device_id, grad_pooled, grad_vectors, context_size, feature_dim)
        )

    def hybrid_representation_backward_f64(
        self,
        device_id: int,
        grad_output: int,
        char_gradient: int,
        feature_gradient: int,
        syllable_gradient: int,
        context_size: int,
        char_dim: int,
        feature_dim: int,
        syllable_dim: int,
    ) -> None:
        self.hybrid_backward_calls.append(
            (
                device_id,
                grad_output,
                char_gradient,
                feature_gradient,
                syllable_gradient,
                context_size,
                char_dim,
                feature_dim,
                syllable_dim,
            )
        )


class CudaBackendMeanPoolDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_mean_pool_dispatches_one_resident_matrix_without_transfers(self) -> None:
        with ExitStack() as stack:
            vectors = stack.enter_context(
                self.backend.buffer_from_host(
                    [[1e16, 2.0], [1.0, 4.0], [-1e16, 6.0]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            pooled = self.backend.mean_pool_forward(vectors)
            stack.enter_context(pooled)

            self.assertIsInstance(pooled, CudaBuffer)
            self.assertEqual(pooled.shape, (2,))
            self.assertEqual(pooled.dtype, "float64")
            self.assertIsNot(pooled, vectors)
            expected_sum_mode = int(sum([1e16, 1.0, -1e16]) == 1.0)
            self.assertEqual(
                self.bridge.mean_pool_calls,
                [(0, vectors.pointer, pooled.pointer, 3, 2, expected_sum_mode)],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)


class CudaBackendNativeDimensionBoundsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def _metadata_buffer(self, shape: tuple[int, int]) -> CudaBuffer:
        pointer = self.bridge.allocate(0, ctypes.sizeof(ctypes.c_double))
        return CudaBuffer(
            self.backend.memory,
            pointer,
            device_id=0,
            shape=shape,
            dtype="float64",
            nbytes=ctypes.sizeof(ctypes.c_double),
        )

    def test_hybrid_forward_rejects_context_size_outside_c_int_before_allocation(self) -> None:
        oversized = 2**31
        with ExitStack() as stack:
            char_vectors = stack.enter_context(self._metadata_buffer((oversized, 1)))
            feature_vectors = stack.enter_context(self._metadata_buffer((oversized, 1)))
            syllable_vectors = stack.enter_context(self._metadata_buffer((oversized, 1)))
            with patch.object(
                self.backend.memory,
                "allocate",
                side_effect=AssertionError("output must not be allocated"),
            ) as allocate:
                with self.assertRaisesRegex(ValueError, "context_size.*CUDA ABI limit"):
                    self.backend.hybrid_representation_forward(
                        char_vectors, feature_vectors, syllable_vectors
                    )
                allocate.assert_not_called()
        self.assertEqual(self.bridge.hybrid_forward_calls, [])

    def test_mean_pool_rejects_each_dimension_outside_c_int_before_allocation(self) -> None:
        oversized = 2**31
        for dimension, shape in (
            ("context_size", (oversized, 1)),
            ("feature_dim", (1, oversized)),
        ):
            with self.subTest(dimension=dimension), ExitStack() as stack:
                vectors = stack.enter_context(self._metadata_buffer(shape))
                with patch.object(
                    self.backend.memory,
                    "allocate",
                    side_effect=AssertionError("output must not be allocated"),
                ) as allocate:
                    with self.assertRaisesRegex(
                        ValueError, f"{dimension}.*CUDA ABI limit"
                    ):
                        self.backend.mean_pool_forward(vectors)
                    allocate.assert_not_called()
        self.assertEqual(self.bridge.mean_pool_calls, [])

    def test_hybrid_forward_rejects_unrepresentable_grid_before_allocation(self) -> None:
        context_size = 1_000_000_000
        feature_dim = 100
        with ExitStack() as stack:
            char_vectors = stack.enter_context(
                self._metadata_buffer((context_size, feature_dim))
            )
            feature_vectors = stack.enter_context(
                self._metadata_buffer((context_size, feature_dim))
            )
            syllable_vectors = stack.enter_context(
                self._metadata_buffer((context_size, feature_dim))
            )
            with patch.object(
                self.backend.memory,
                "allocate",
                side_effect=AssertionError("output must not be allocated"),
            ) as allocate:
                with self.assertRaisesRegex(ValueError, "CUDA grid limit"):
                    self.backend.hybrid_representation_forward(
                        char_vectors, feature_vectors, syllable_vectors
                    )
                allocate.assert_not_called()
        self.assertEqual(self.bridge.hybrid_forward_calls, [])


class CudaBackendResidentLossDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_cross_entropy_forward_returns_scalar_and_only_scalar_d2h(self) -> None:
        with ExitStack() as stack:
            logits = stack.enter_context(
                self.backend.buffer_from_host([0.25, -0.5, 1.0], shape=(3,))
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            loss = self.backend.cross_entropy_forward_buffer(logits, 2)

            self.assertEqual(loss, 0.75)
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before + 2)
            self.assertEqual(
                self.bridge.copy_to_host_nbytes[-2:],
                [ctypes.sizeof(ctypes.c_int32), ctypes.sizeof(ctypes.c_double)],
            )

    def test_binary_cross_entropy_forward_and_gradient_are_resident_except_scalars(self) -> None:
        with ExitStack() as stack:
            logit = stack.enter_context(
                self.backend.buffer_from_host([0.5], shape=(1,))
            )
            d2h_before = self.bridge.copy_to_host_calls
            loss = self.backend.binary_cross_entropy_with_logits_forward_buffer(
                logit, 1.0
            )
            gradient = self.backend.binary_cross_entropy_with_logits_gradient_buffer(
                logit, 1.0, 0.3
            )
            stack.enter_context(gradient)

            self.assertEqual(loss, 0.25)
            self.assertEqual(gradient.shape, (1,))
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before + 3)
            self.assertEqual(
                self.bridge.copy_to_host_nbytes[d2h_before:],
                [ctypes.sizeof(ctypes.c_int32), ctypes.sizeof(ctypes.c_double), ctypes.sizeof(ctypes.c_int32)],
            )
            self.assertEqual(len(self.bridge.binary_cross_entropy_gradient_calls), 1)

    def test_cross_entropy_gradient_stays_resident_and_scales_in_kernel(self) -> None:
        with ExitStack() as stack:
            logits = stack.enter_context(
                self.backend.buffer_from_host([0.25, -0.5, 1.0], shape=(3,))
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            gradient = self.backend.cross_entropy_gradient_buffer(logits, 2, 0.7)
            stack.enter_context(gradient)

            self.assertEqual(gradient.shape, (3,))
            self.assertEqual(gradient.dtype, "float64")
            self.assertEqual(len(self.bridge.cross_entropy_gradient_calls), 1)
            call = self.bridge.cross_entropy_gradient_calls[0]
            self.assertEqual(call[:3], (0, logits.pointer, gradient.pointer))
            self.assertEqual(call[4:], (3, 2, 0.7, int(sum((1e16, 1.0, -1e16)) == 1.0)))
            self.assertNotIn(call[3], self.bridge._allocations)
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before + 1)
            self.assertEqual(self.bridge.copy_to_host_nbytes[-1], ctypes.sizeof(ctypes.c_int32))


class CudaBackendResidentBackwardDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_repeated_backward_releases_replaced_device_parameter_gradients(self) -> None:
        with ExitStack() as stack:
            layer = Linear(3, 2, seed=9, backend=self.backend)
            inputs = stack.enter_context(
                self.backend.buffer_from_host([0.1, 0.2, 0.3], shape=(3,))
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host([0.4, -0.5], shape=(2,))
            )
            layer.forward(inputs)
            first_grad_inputs = cast(CudaBuffer, layer.backward(grad_output))
            stack.enter_context(first_grad_inputs)
            first_grad_weights = cast(CudaBuffer, layer.grad_weights)
            first_grad_bias = cast(CudaBuffer, layer.grad_bias)

            second_grad_inputs = cast(CudaBuffer, layer.backward(grad_output))
            stack.enter_context(second_grad_inputs)
            self.assertIsNot(first_grad_inputs, second_grad_inputs)
            self.assertTrue(first_grad_weights.closed)
            self.assertTrue(first_grad_bias.closed)
            self.assertIsInstance(layer.grad_weights, CudaBuffer)
            self.assertIsInstance(layer.grad_bias, CudaBuffer)

    def test_linear_release_backward_state_closes_device_parameter_gradients(self) -> None:
        with ExitStack() as stack:
            layer = Linear(3, 2, seed=9, backend=self.backend)
            inputs = stack.enter_context(
                self.backend.buffer_from_host([0.1, 0.2, 0.3], shape=(3,))
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host([0.4, -0.5], shape=(2,))
            )
            layer.forward(inputs)
            grad_inputs = stack.enter_context(layer.backward(grad_output))
            grad_weights = cast(CudaBuffer, layer.grad_weights)
            grad_bias = cast(CudaBuffer, layer.grad_bias)

            layer.release_backward_state()

            self.assertTrue(grad_weights.closed)
            self.assertTrue(grad_bias.closed)
            self.assertIsNone(layer.grad_weights)
            self.assertIsNone(layer.grad_bias)
            self.assertFalse(layer._has_backward)

    def test_head_sum_pool_backward_and_representation_split_stay_resident(self) -> None:
        with ExitStack() as stack:
            char = stack.enter_context(
                self.backend.buffer_from_host([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], shape=(6,))
            )
            syllable = stack.enter_context(
                self.backend.buffer_from_host([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], shape=(6,))
            )
            boundary = stack.enter_context(
                self.backend.buffer_from_host([0.05, 0.06, 0.07, 0.08, 0.09, 0.1], shape=(6,))
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            pooled_gradient = self.backend.sum_head_input_gradients(
                char, syllable, boundary
            )
            stack.enter_context(pooled_gradient)
            matrix_gradient = self.backend.mean_pool_backward(
                pooled_gradient, context_size=2
            )
            stack.enter_context(matrix_gradient)
            char_gradient, feature_gradient, syllable_gradient = (
                self.backend.hybrid_representation_backward(
                    matrix_gradient, char_dim=2, feature_dim=2, syllable_dim=2
                )
            )
            for buffer in (char_gradient, feature_gradient, syllable_gradient):
                stack.enter_context(buffer)

            self.assertEqual(pooled_gradient.shape, (6,))
            self.assertEqual(matrix_gradient.shape, (2, 6))
            self.assertEqual(char_gradient.shape, (2, 2))
            self.assertEqual(feature_gradient.shape, (2, 2))
            self.assertEqual(syllable_gradient.shape, (2, 2))
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)
            self.assertEqual(len(self.bridge.head_gradient_sum_calls), 1)
            self.assertEqual(len(self.bridge.mean_pool_backward_calls), 1)
            self.assertEqual(len(self.bridge.hybrid_backward_calls), 1)


class CudaBackendArgmaxDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_argmax_dispatches_resident_logits_and_copies_only_the_int32_id(self) -> None:
        with ExitStack() as stack:
            logits = stack.enter_context(
                self.backend.buffer_from_host([0.0, 5.0, 5.0, -1.0], shape=(4,))
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            predicted = self.backend.argmax(logits)

            self.assertEqual(predicted, 1)
            self.assertEqual(len(self.bridge.argmax_calls), 1)
            device_id, input_pointer, output_pointer, logits_count = self.bridge.argmax_calls[0]
            self.assertEqual((device_id, input_pointer, logits_count), (0, logits.pointer, 4))
            self.assertNotIn(output_pointer, self.bridge._allocations)
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before + 1)
            self.assertEqual(self.bridge.copy_to_host_nbytes[d2h_before:], [4])
            self.assertFalse(logits.closed)


class CudaBackendStepDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def _buffers(self, stack: ExitStack):
        weights = stack.enter_context(
            self.backend.buffer_from_host([[1.0, 2.0], [-1.0, 0.5]], shape=(2, 2))
        )
        bias = stack.enter_context(self.backend.buffer_from_host([0.25, -0.5], shape=(2,)))
        grad_weights = stack.enter_context(
            self.backend.buffer_from_host([[0.1, 0.2], [0.3, 0.4]], shape=(2, 2))
        )
        grad_bias = stack.enter_context(self.backend.buffer_from_host([0.5, -0.25], shape=(2,)))
        return weights, bias, grad_weights, grad_bias

    def test_step_updates_resident_buffers_with_one_native_dispatch(self) -> None:
        with ExitStack() as stack:
            weights, bias, grad_weights, grad_bias = self._buffers(stack)
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            updated_weights, updated_bias = self.backend.linear_step(
                weights, bias, grad_weights, grad_bias, 0.125
            )

            self.assertIs(updated_weights, weights)
            self.assertIs(updated_bias, bias)
            self.assertEqual(
                self.bridge.step_calls,
                [
                    (
                        0,
                        weights.pointer,
                        bias.pointer,
                        grad_weights.pointer,
                        grad_bias.pointer,
                        0.125,
                        2,
                        2,
                    )
                ],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)

    def test_zero_rate_preserves_device_buffers_without_native_dispatch(self) -> None:
        with ExitStack() as stack:
            weights, bias, grad_weights, grad_bias = self._buffers(stack)
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            updated_weights, updated_bias = self.backend.linear_step(
                weights, bias, grad_weights, grad_bias, 0.0
            )

            self.assertIs(updated_weights, weights)
            self.assertIs(updated_bias, bias)
            self.assertEqual(self.bridge.step_calls, [])
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)

    def test_step_rejects_each_parameter_and_gradient_shape_mismatch_before_native_dispatch(self) -> None:
        with ExitStack() as stack:
            weights, bias, grad_weights, grad_bias = self._buffers(stack)
            wrong_bias = stack.enter_context(
                self.backend.buffer_from_host([0.0], shape=(1,))
            )
            wrong_grad_weights = stack.enter_context(
                self.backend.buffer_from_host([[1.0], [2.0]], shape=(2, 1))
            )
            wrong_grad_bias = stack.enter_context(
                self.backend.buffer_from_host([1.0], shape=(1,))
            )
            invalid_cases = (
                ("bias", weights, wrong_bias, grad_weights, grad_bias),
                ("grad_weights", weights, bias, wrong_grad_weights, grad_bias),
                ("grad_bias", weights, bias, grad_weights, wrong_grad_bias),
            )
            for name, case_weights, case_bias, case_grad_weights, case_grad_bias in invalid_cases:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    self.backend.linear_step(
                        case_weights,
                        case_bias,
                        case_grad_weights,
                        case_grad_bias,
                        0.1,
                    )
        self.assertEqual(self.bridge.step_calls, [])

    def test_step_rejects_device_mismatch_before_native_dispatch(self) -> None:
        bridge = FakeCudaBridge(device_count=2)
        backend = CUDABackend(runtime=CudaRuntime(bridge=bridge))
        with ExitStack() as stack:
            weights = stack.enter_context(
                backend.buffer_from_host([[1.0, 2.0], [-1.0, 0.5]], shape=(2, 2))
            )
            bias = stack.enter_context(
                backend.memory.copy_to_device([0.25, -0.5], shape=(2,), device_id=1)
            )
            grad_weights = stack.enter_context(
                backend.buffer_from_host([[0.1, 0.2], [0.3, 0.4]], shape=(2, 2))
            )
            grad_bias = stack.enter_context(
                backend.buffer_from_host([0.5, -0.25], shape=(2,))
            )
            with self.assertRaisesRegex(BackendError, "same CUDA device"):
                backend.linear_step(weights, bias, grad_weights, grad_bias, 0.1)
        self.assertEqual(bridge.step_calls, [])

    def test_step_rejects_foreign_manager_and_released_buffers(self) -> None:
        other_bridge = FakeCudaBridge()
        other_backend = CUDABackend(runtime=CudaRuntime(bridge=other_bridge))
        with ExitStack() as stack:
            weights, bias, grad_weights, grad_bias = self._buffers(stack)
            foreign_weights = stack.enter_context(
                other_backend.buffer_from_host([[1.0, 2.0], [-1.0, 0.5]], shape=(2, 2))
            )
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.linear_step(
                    foreign_weights, bias, grad_weights, grad_bias, 0.1
                )
            self.assertEqual(self.bridge.step_calls, [])

        with ExitStack() as stack:
            weights, bias, grad_weights, grad_bias = self._buffers(stack)
            grad_bias.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.linear_step(weights, bias, grad_weights, grad_bias, 0.1)
        self.assertEqual(self.bridge.step_calls, [])

    def test_step_rejects_invalid_learning_rates_before_native_dispatch(self) -> None:
        invalid_rates = ((True, TypeError), ("0.1", TypeError), (float("nan"), ValueError),
                         (float("inf"), ValueError), (-0.1, ValueError))
        with ExitStack() as stack:
            buffers = self._buffers(stack)
            for rate, error_type in invalid_rates:
                with self.subTest(rate=rate), self.assertRaises(error_type):
                    self.backend.linear_step(*buffers, cast(float, rate))
        self.assertEqual(self.bridge.step_calls, [])


class CudaBackendBackwardDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_backward_allocates_device_gradients_and_dispatches_one_native_call(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host(
                    [[0.5, -1.0, 0.25], [2.0, 0.25, -0.5]],
                    shape=(2, 3),
                    dtype="float64",
                )
            )
            inputs = stack.enter_context(
                self.backend.buffer_from_host([2.0, -1.0, 0.5], shape=(3,))
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host([3.0, -2.0], shape=(2,))
            )

            grad_weights, grad_bias, grad_inputs = self.backend.linear_backward(
                weights, inputs, grad_output
            )
            stack.callback(grad_weights.close)
            stack.callback(grad_bias.close)
            stack.callback(grad_inputs.close)

            self.assertIsInstance(grad_weights, CudaBuffer)
            self.assertEqual(grad_weights.shape, (2, 3))
            self.assertEqual(grad_bias.shape, (2,))
            self.assertEqual(grad_inputs.shape, (3,))
            self.assertEqual(len(self.bridge.backward_calls), 1)
            call = self.bridge.backward_calls[0]
            self.assertEqual(call[:4], (0, weights.pointer, inputs.pointer, grad_output.pointer))
            self.assertEqual(call[7:], (3, 2))

    def test_backward_rejects_shape_mismatch_before_native_dispatch(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            )
            inputs = stack.enter_context(
                self.backend.buffer_from_host([1.0, 2.0], shape=(2,))
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host([1.0, 2.0], shape=(2,))
            )

            with self.assertRaisesRegex(ValueError, "grad_output"):
                self.backend.linear_backward(weights, inputs, grad_output)

        self.assertEqual(self.bridge.backward_calls, [])

    def test_host_lists_are_not_silently_used_as_cuda_operands(self) -> None:
        with self.assertRaisesRegex(TypeError, "CudaBuffer"):
            self.backend.linear_backward([[1.0]], [1.0], [1.0])
        self.assertEqual(self.bridge.backward_calls, [])


class CudaBackendHybridRepresentationDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_forward_dispatches_three_resident_matrices_without_transfers(self) -> None:
        with ExitStack() as stack:
            char_vectors = stack.enter_context(
                self.backend.buffer_from_host(
                    [[0.1, 0.2], [0.3, 0.4]], shape=(2, 2)
                )
            )
            features = stack.enter_context(
                self.backend.buffer_from_host(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(2, 3)
                )
            )
            syllable_vectors = stack.enter_context(
                self.backend.buffer_from_host([[0.5], [0.6]], shape=(2, 1))
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            output = self.backend.hybrid_representation_forward(
                char_vectors, features, syllable_vectors
            )
            stack.enter_context(output)

            self.assertIsInstance(output, CudaBuffer)
            self.assertEqual(output.shape, (2, 6))
            self.assertEqual(output.dtype, "float64")
            self.assertNotIn(output, (char_vectors, features, syllable_vectors))
            self.assertEqual(
                self.bridge.hybrid_forward_calls,
                [
                    (
                        0,
                        char_vectors.pointer,
                        features.pointer,
                        syllable_vectors.pointer,
                        output.pointer,
                        2,
                        2,
                        3,
                        1,
                    )
                ],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)


class CudaBackendEmbeddingForwardDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_forward_allocates_independent_output_and_dispatches_typed_ids(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host(
                    [[0.5, -1.0], [2.0, 0.25], [-0.5, 3.0]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            ids = stack.enter_context(
                self.backend.buffer_from_host(
                    [2, 0, 2], shape=(3,), dtype="int32"
                )
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            output = self.backend.embedding_forward(weights, ids)
            stack.enter_context(output)

            self.assertIsInstance(output, CudaBuffer)
            self.assertEqual(output.shape, (3, 2))
            self.assertEqual(output.dtype, "float64")
            self.assertEqual(output.nbytes, 48)
            self.assertNotEqual(output.pointer, weights.pointer)
            self.assertNotEqual(output.pointer, ids.pointer)
            self.assertEqual(
                self.bridge.embedding_forward_calls,
                [(0, weights.pointer, ids.pointer, output.pointer, 3, 2, 3)],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)


    def test_forward_rejects_wrong_id_dtype_rank_host_lists_and_released_buffers(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host([[0.5, -1.0], [2.0, 0.25]], shape=(2, 2))
            )
            float_ids = stack.enter_context(
                self.backend.buffer_from_host([1.0], shape=(1,), dtype="float64")
            )
            matrix_ids = stack.enter_context(
                self.backend.buffer_from_host([[1, 0]], shape=(1, 2), dtype="int32")
            )
            closed_ids = stack.enter_context(
                self.backend.buffer_from_host([1], shape=(1,), dtype="int32")
            )

            with self.assertRaisesRegex(ValueError, "dtype int32"):
                self.backend.embedding_forward(weights, float_ids)
            with self.assertRaisesRegex(ValueError, "rank 1"):
                self.backend.embedding_forward(weights, matrix_ids)
            with self.assertRaisesRegex(TypeError, "int32 CudaBuffer"):
                self.backend.embedding_forward(weights, [1])
            closed_ids.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_forward(weights, closed_ids)

        self.assertEqual(self.bridge.embedding_forward_calls, [])

    def test_forward_rejects_device_and_memory_owner_mismatches(self) -> None:
        bridge = FakeCudaBridge(device_count=2)
        backend = CUDABackend(runtime=CudaRuntime(bridge=bridge))
        other_bridge = FakeCudaBridge(device_count=2)
        other_backend = CUDABackend(runtime=CudaRuntime(bridge=other_bridge))
        with ExitStack() as stack:
            weights = stack.enter_context(
                backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            )
            ids_other_device = stack.enter_context(
                backend.memory.copy_to_device(
                    [0], shape=(1,), dtype="int32", device_id=1
                )
            )
            foreign_ids = stack.enter_context(
                other_backend.buffer_from_host([0], shape=(1,), dtype="int32")
            )

            with self.assertRaisesRegex(BackendError, "same CUDA device"):
                backend.embedding_forward(weights, ids_other_device)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                backend.embedding_forward(weights, foreign_ids)

        self.assertEqual(bridge.embedding_forward_calls, [])


class CudaBackendEmbeddingBackwardDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_backward_updates_resident_gradient_buffer_with_deterministic_abi_inputs(self) -> None:
        with ExitStack() as stack:
            grad_weights = stack.enter_context(
                self.backend.buffer_from_host(
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            ids = stack.enter_context(
                self.backend.buffer_from_host(
                    [2, 0, 2], shape=(3,), dtype="int32"
                )
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            result = self.backend.embedding_backward(grad_weights, ids, grad_output)

            self.assertIs(result, grad_weights)
            self.assertEqual(grad_weights.shape, (3, 2))
            self.assertEqual(
                self.bridge.embedding_backward_calls,
                [(0, ids.pointer, grad_output.pointer, grad_weights.pointer, 3, 2, 3)],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)


    def test_backward_rejects_invalid_dtype_shape_and_host_operands(self) -> None:
        with ExitStack() as stack:
            grad_weights = stack.enter_context(
                self.backend.buffer_from_host([[0.0, 0.0]] * 3, shape=(3, 2))
            )
            ids = stack.enter_context(
                self.backend.buffer_from_host([2, 0, 2], shape=(3,), dtype="int32")
            )
            grad_output = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 2.0]] * 3, shape=(3, 2))
            )
            float_ids = stack.enter_context(
                self.backend.buffer_from_host([2, 0, 2], shape=(3,), dtype="float64")
            )
            matrix_ids = stack.enter_context(
                self.backend.buffer_from_host([[2, 0, 2]], shape=(1, 3), dtype="int32")
            )
            integer_gradients = stack.enter_context(
                self.backend.buffer_from_host([[1, 2], [3, 4], [5, 6]], shape=(3, 2), dtype="int32")
            )
            wrong_shape = stack.enter_context(
                self.backend.buffer_from_host([[1.0], [2.0], [3.0]], shape=(3, 1))
            )

            invalid_cases = (
                (float_ids, grad_output, ValueError, "dtype int32"),
                (matrix_ids, grad_output, ValueError, "rank 1"),
                (ids, integer_gradients, ValueError, "dtype float64"),
                (ids, wrong_shape, ValueError, "shape"),
                (ids, [[1.0, 2.0]] * 3, TypeError, "CudaBuffer"),
            )
            for case_ids, case_gradients, error_type, message in invalid_cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    error_type, message
                ):
                    self.backend.embedding_backward(
                        grad_weights, case_ids, case_gradients
                    )
        self.assertEqual(self.bridge.embedding_backward_calls, [])

    def test_backward_rejects_device_manager_and_lifetime_mismatches(self) -> None:
        bridge = FakeCudaBridge(device_count=2)
        backend = CUDABackend(runtime=CudaRuntime(bridge=bridge))
        other_bridge = FakeCudaBridge(device_count=2)
        other_backend = CUDABackend(runtime=CudaRuntime(bridge=other_bridge))
        with ExitStack() as stack:
            grad_weights = stack.enter_context(
                backend.buffer_from_host([[0.0, 0.0]], shape=(1, 2))
            )
            grad_output = stack.enter_context(
                backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            )
            ids_other_device = stack.enter_context(
                backend.memory.copy_to_device([0], shape=(1,), dtype="int32", device_id=1)
            )
            foreign_ids = stack.enter_context(
                other_backend.buffer_from_host([0], shape=(1,), dtype="int32")
            )
            with self.assertRaisesRegex(BackendError, "same CUDA device"):
                backend.embedding_backward(grad_weights, ids_other_device, grad_output)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                backend.embedding_backward(grad_weights, foreign_ids, grad_output)
            ids_other_device.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                backend.embedding_backward(grad_weights, ids_other_device, grad_output)
        self.assertEqual(bridge.embedding_backward_calls, [])


class CudaBackendEmbeddingStepDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.backend = CUDABackend(runtime=CudaRuntime(bridge=self.bridge))

    def test_step_updates_resident_embedding_weights_with_one_native_dispatch(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host(
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            grad_weights = stack.enter_context(
                self.backend.buffer_from_host(
                    [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
                    shape=(3, 2),
                    dtype="float64",
                )
            )
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            updated_weights = self.backend.embedding_step(weights, grad_weights, 0.125)

            self.assertIs(updated_weights, weights)
            self.assertEqual(
                self.bridge.embedding_step_calls,
                [(0, weights.pointer, grad_weights.pointer, 0.125, 3, 2)],
            )
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)


    def test_zero_rate_preserves_embedding_weights_without_dispatch_or_transfer(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host([[0.1, 0.2], [0.3, 0.4]], shape=(2, 2))
            )
            grad_weights = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 2.0], [3.0, 4.0]], shape=(2, 2))
            )
            expected = self.backend.buffer_to_host(weights)
            h2d_before = self.bridge.copy_to_device_calls
            d2h_before = self.bridge.copy_to_host_calls

            result = self.backend.embedding_step(weights, grad_weights, 0.0)

            self.assertIs(result, weights)
            self.assertEqual(self.bridge.embedding_step_calls, [])
            self.assertEqual(self.bridge.copy_to_device_calls, h2d_before)
            self.assertEqual(self.bridge.copy_to_host_calls, d2h_before)
            self.assertEqual(self.backend.buffer_to_host(weights), expected)


    def test_step_rejects_shape_dtype_ownership_lifetime_and_invalid_rates(self) -> None:
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 2.0], [3.0, 4.0]], shape=(2, 2))
            )
            gradients = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 0.0], [0.0, 1.0]], shape=(2, 2))
            )
            wrong_shape = stack.enter_context(
                self.backend.buffer_from_host([[1.0], [2.0]], shape=(2, 1))
            )
            int_weights = stack.enter_context(
                self.backend.buffer_from_host([[1, 2], [3, 4]], shape=(2, 2), dtype="int32")
            )
            for rate, error in (
                (True, TypeError),
                ("0.1", TypeError),
                (float("nan"), ValueError),
                (float("inf"), ValueError),
                (-0.1, ValueError),
            ):
                with self.subTest(rate=rate), self.assertRaises(error):
                    self.backend.embedding_step(weights, gradients, cast(float, rate))
            with self.assertRaisesRegex(ValueError, "grad_weights"):
                self.backend.embedding_step(weights, wrong_shape, 0.1)
            with self.assertRaisesRegex(ValueError, "float64"):
                self.backend.embedding_step(int_weights, gradients, 0.1)
            with self.assertRaisesRegex(TypeError, "CudaBuffer"):
                self.backend.embedding_step([[1.0]], [[0.0]], 0.1)

        foreign_bridge = FakeCudaBridge()
        foreign_backend = CUDABackend(runtime=CudaRuntime(bridge=foreign_bridge))
        with ExitStack() as stack:
            weights = stack.enter_context(
                self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            )
            gradients = stack.enter_context(
                foreign_backend.buffer_from_host([[0.1, 0.2]], shape=(1, 2))
            )
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_step(weights, gradients, 0.1)
        self.assertEqual(self.bridge.embedding_step_calls, [])


    def test_step_rejects_device_mismatch_and_released_gradient_buffer(self) -> None:
        bridge = FakeCudaBridge(device_count=2)
        backend = CUDABackend(runtime=CudaRuntime(bridge=bridge))
        with ExitStack() as stack:
            weights = stack.enter_context(
                backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            )
            gradients_other_device = stack.enter_context(
                backend.memory.copy_to_device(
                    [[0.1, 0.2]], shape=(1, 2), dtype="float64", device_id=1
                )
            )
            gradients_closed = stack.enter_context(
                backend.buffer_from_host([[0.1, 0.2]], shape=(1, 2))
            )

            with self.assertRaisesRegex(BackendError, "same CUDA device"):
                backend.embedding_step(weights, gradients_other_device, 0.1)
            gradients_closed.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                backend.embedding_step(weights, gradients_closed, 0.1)

        self.assertEqual(bridge.embedding_step_calls, [])


if __name__ == "__main__":
    unittest.main()
