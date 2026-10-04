"""CPU-only orchestration tests for CUDA Linear gradient dispatch."""

import ctypes
from contextlib import ExitStack
import unittest
from typing import cast

from kinera.backends.base import BackendError
from kinera.backends.cuda import CUDABackend, CudaBridge, CudaBuffer, CudaRuntime


class FakeCudaBridge(CudaBridge):
    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count
        self._next_pointer = 0x1000
        self._allocations: dict[int, bytearray] = {}
        self.backward_calls: list[tuple[int, ...]] = []
        self.step_calls: list[tuple[int | float, ...]] = []
        self.copy_to_device_calls = 0
        self.copy_to_host_calls = 0
        self.embedding_forward_calls: list[tuple[int, ...]] = []

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


if __name__ == "__main__":
    unittest.main()
