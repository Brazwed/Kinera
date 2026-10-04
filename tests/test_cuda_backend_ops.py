"""CPU-only orchestration tests for CUDA Linear gradient dispatch."""

import ctypes
from contextlib import ExitStack
import unittest

from kinera.backends.base import BackendError
from kinera.backends.cuda import CUDABackend, CudaBridge, CudaBuffer, CudaRuntime


class FakeCudaBridge(CudaBridge):
    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count
        self._next_pointer = 0x1000
        self._allocations: dict[int, bytearray] = {}
        self.backward_calls: list[tuple[int, ...]] = []

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
        self._allocations[destination][:nbytes] = ctypes.string_at(source, nbytes)

    def copy_to_host(self, device_id: int, destination: int, source: int, nbytes: int) -> None:
        ctypes.memmove(destination, bytes(self._allocations[source][:nbytes]), nbytes)

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


if __name__ == "__main__":
    unittest.main()
