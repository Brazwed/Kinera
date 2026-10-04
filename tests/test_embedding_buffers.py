"""Phase 10: Embedding metadata and backend-buffer contracts."""

from contextlib import ExitStack
import ctypes
from typing import cast
import unittest

from kinera.backends.base import BackendArray, BackendBuffer, BackendUnavailableError, HostArray
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda.backend import CUDABackend
from kinera.backends.cuda.buffer import CudaBuffer
from kinera.backends.cuda.bridge import CudaBridge
from kinera.backends.cuda.memory import CudaMemoryManager
from kinera.backends.cuda.runtime import CudaRuntime
from kinera.nn.layers.embedding import Embedding


class ShapeOnlyBuffer:
    """Metadata-only CUDA buffer double without Python sequence behavior."""

    device = "cuda"
    device_id = 0
    dtype = "float64"
    layout = "C"
    owns_memory = True
    closed = False

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.nbytes = 8 * shape[0] * shape[1]


class RecordingCpuBufferBackend(CPUBackend):
    def __init__(self) -> None:
        self.buffer_shapes: list[tuple[tuple[int, ...], str]] = []

    def buffer_from_host(
        self, values: HostArray, *, shape: tuple[int, ...], dtype: str = "float64"
    ) -> HostArray:
        self.buffer_shapes.append((shape, dtype))
        return super().buffer_from_host(values, shape=shape, dtype=dtype)


class InMemoryCudaBridge(CudaBridge):
    def __init__(self) -> None:
        self._next_pointer = 1
        self.allocations: dict[int, bytearray] = {}
        self.copy_to_device_calls = 0
        self.copy_to_host_calls = 0

    def device_count(self) -> int:
        return 1

    def allocate(self, device_id: int, nbytes: int) -> int:
        pointer = self._next_pointer
        self._next_pointer += 1
        self.allocations[pointer] = bytearray(nbytes)
        return pointer

    def free(self, device_id: int, pointer: int) -> None:
        del self.allocations[pointer]

    def copy_to_device(
        self, device_id: int, destination: int, source: int, nbytes: int
    ) -> None:
        self.copy_to_device_calls += 1
        self.allocations[destination][:nbytes] = ctypes.string_at(source, nbytes)

    def copy_to_host(
        self, device_id: int, destination: int, source: int, nbytes: int
    ) -> None:
        self.copy_to_host_calls += 1
        ctypes.memmove(destination, bytes(self.allocations[source][:nbytes]), nbytes)


class EmbeddingCudaStorageTests(unittest.TestCase):
    def test_cuda_constructor_keeps_weights_and_gradients_in_owned_buffers(self) -> None:
        bridge = InMemoryCudaBridge()
        memory = CudaMemoryManager(bridge=bridge)
        backend = CUDABackend(runtime=memory.runtime, memory_manager=memory)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        weights = cast(CudaBuffer, embedding.weights)
        gradients = cast(CudaBuffer, embedding.grad_weights)

        with ExitStack() as stack:
            stack.enter_context(weights)
            stack.enter_context(gradients)
            self.assertEqual((embedding.vocab_size, embedding.embedding_dim), (3, 2))
            self.assertEqual(weights.shape, (3, 2))
            self.assertEqual(gradients.shape, (3, 2))
            self.assertEqual(weights.dtype, "float64")
            self.assertEqual(gradients.dtype, "float64")
            self.assertEqual(weights.nbytes, 48)
            self.assertEqual(gradients.nbytes, 48)

        self.assertEqual(bridge.allocations, {})

    def test_forward_prepares_typed_ids_and_device_gradients_without_materializing_output(self) -> None:
        bridge = InMemoryCudaBridge()
        memory = CudaMemoryManager(bridge=bridge)
        backend = CUDABackend(runtime=memory.runtime, memory_manager=memory)
        seen_weights: list[BackendArray] = []
        seen_ids: list[list[int] | BackendBuffer] = []
        seen_outputs: list[CudaBuffer] = []

        def embedding_forward(
            weights: BackendArray, ids: list[int] | BackendBuffer
        ) -> CudaBuffer:
            seen_weights.append(weights)
            seen_ids.append(ids)
            weight_buffer = cast(CudaBuffer, weights)
            count = len(ids) if isinstance(ids, list) else ids.shape[0]
            output = memory.allocate(
                (count, weight_buffer.shape[1]),
                dtype="float64",
                device_id=weight_buffer.device_id,
            )
            seen_outputs.append(output)
            return output

        setattr(backend, "embedding_forward", embedding_forward)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        original_weights = cast(CudaBuffer, embedding.weights)
        original_gradients = cast(CudaBuffer, embedding.grad_weights)
        try:
            output = embedding.forward([1, 0])
            self.assertIsInstance(seen_ids[0], CudaBuffer)
            ids_buffer = cast(CudaBuffer, seen_ids[0])
            self.assertEqual(ids_buffer.dtype, "int32")
            self.assertEqual(ids_buffer.shape, (2,))
            self.assertEqual(memory.copy_to_host(ids_buffer), [1, 0])
            self.assertIs(seen_weights[0], original_weights)
            self.assertIs(output, seen_outputs[0])
            self.assertIs(embedding.output, output)
            self.assertIsInstance(embedding.grad_weights, CudaBuffer)
            gradients = cast(CudaBuffer, embedding.grad_weights)
            self.assertEqual(gradients.shape, (3, 2))
        finally:
            buffers = [
                original_weights,
                original_gradients,
                embedding.grad_weights,
                *seen_ids,
                *seen_outputs,
            ]
            for buffer in buffers:
                if isinstance(buffer, CudaBuffer) and not buffer.closed:
                    buffer.close()
        self.assertEqual(bridge.allocations, {})

    def test_out_of_vocab_id_fails_before_device_copy_or_kernel_dispatch(self) -> None:
        bridge = InMemoryCudaBridge()
        memory = CudaMemoryManager(bridge=bridge)
        backend = CUDABackend(runtime=memory.runtime, memory_manager=memory)
        dispatched: list[object] = []

        def should_not_dispatch(
            weights: BackendArray, ids: list[int] | BackendBuffer
        ) -> list[list[float]]:
            dispatched.append((weights, ids))
            return []

        setattr(backend, "embedding_forward", should_not_dispatch)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        weights = cast(CudaBuffer, embedding.weights)
        gradients = cast(CudaBuffer, embedding.grad_weights)
        copies_before = bridge.copy_to_device_calls
        try:
            with self.assertRaisesRegex(ValueError, "fora do vocabulário"):
                embedding.forward([3])
            self.assertEqual(bridge.copy_to_device_calls, copies_before)
            self.assertEqual(dispatched, [])
        finally:
            weights.close()
            gradients.close()
        self.assertEqual(bridge.allocations, {})

    def test_backward_uploads_host_gradients_once_and_keeps_buffers_device_native(self) -> None:
        bridge = InMemoryCudaBridge()
        memory = CudaMemoryManager(bridge=bridge)
        backend = CUDABackend(runtime=memory.runtime, memory_manager=memory)
        seen_ids: list[list[int] | BackendBuffer] = []
        seen_outputs: list[CudaBuffer] = []
        seen_gradients: list[BackendArray] = []

        def embedding_forward(
            weights: BackendArray, ids: list[int] | BackendBuffer
        ) -> CudaBuffer:
            seen_ids.append(ids)
            weight_buffer = cast(CudaBuffer, weights)
            count = len(ids) if isinstance(ids, list) else ids.shape[0]
            output = memory.allocate(
                (count, weight_buffer.shape[1]),
                dtype="float64",
                device_id=weight_buffer.device_id,
            )
            seen_outputs.append(output)
            return output

        def embedding_backward(
            grad_weights: BackendArray,
            ids: list[int] | BackendBuffer,
            grad_output: BackendArray,
        ) -> BackendArray:
            seen_gradients.append(grad_output)
            return grad_weights

        setattr(backend, "embedding_forward", embedding_forward)
        setattr(backend, "embedding_backward", embedding_backward)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        weights = cast(CudaBuffer, embedding.weights)
        initial_gradients = cast(CudaBuffer, embedding.grad_weights)
        try:
            embedding.forward([2, 0, 2])
            ids_buffer = cast(CudaBuffer, embedding.last_ids)
            grad_buffer = cast(CudaBuffer, embedding.grad_weights)
            h2d_before = bridge.copy_to_device_calls
            d2h_before = bridge.copy_to_host_calls

            embedding.backward([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            self.assertIsInstance(seen_gradients[0], CudaBuffer)
            grad_output_buffer = cast(CudaBuffer, seen_gradients[0])
            self.assertEqual(grad_output_buffer.shape, (3, 2))
            self.assertEqual(grad_output_buffer.dtype, "float64")
            self.assertEqual(bridge.copy_to_device_calls, h2d_before + 1)
            self.assertEqual(bridge.copy_to_host_calls, d2h_before)
            self.assertIs(seen_ids[0], ids_buffer)
            self.assertIs(embedding.grad_weights, grad_buffer)
            self.assertTrue(embedding._has_backward)
        finally:
            buffers = [
                weights,
                initial_gradients,
                embedding.grad_weights,
                *seen_ids,
                *seen_outputs,
                *seen_gradients,
            ]
            for buffer in buffers:
                if isinstance(buffer, CudaBuffer) and not buffer.closed:
                    buffer.close()
        self.assertEqual(bridge.allocations, {})

    def test_backward_uses_existing_device_gradient_buffer_without_roundtrip(self) -> None:
        bridge = InMemoryCudaBridge()
        memory = CudaMemoryManager(bridge=bridge)
        backend = CUDABackend(runtime=memory.runtime, memory_manager=memory)
        seen_ids: list[list[int] | BackendBuffer] = []
        seen_gradients: list[BackendArray] = []
        outputs: list[CudaBuffer] = []

        def embedding_forward(
            weights: BackendArray, ids: list[int] | BackendBuffer
        ) -> CudaBuffer:
            seen_ids.append(ids)
            weight_buffer = cast(CudaBuffer, weights)
            count = len(ids) if isinstance(ids, list) else ids.shape[0]
            output = memory.allocate(
                (count, weight_buffer.shape[1]),
                dtype="float64",
                device_id=weight_buffer.device_id,
            )
            outputs.append(output)
            return output

        def embedding_backward(
            grad_weights: BackendArray,
            ids: list[int] | BackendBuffer,
            grad_output: BackendArray,
        ) -> BackendArray:
            seen_gradients.append(grad_output)
            return grad_weights

        setattr(backend, "embedding_forward", embedding_forward)
        setattr(backend, "embedding_backward", embedding_backward)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        weights = cast(CudaBuffer, embedding.weights)
        initial_gradients = cast(CudaBuffer, embedding.grad_weights)
        gradient_buffer = backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            shape=(3, 2),
            dtype="float64",
        )
        try:
            embedding.forward([2, 0, 2])
            current_gradients = cast(CudaBuffer, embedding.grad_weights)
            copies_before = bridge.copy_to_device_calls
            d2h_before = bridge.copy_to_host_calls

            embedding.backward(gradient_buffer)

            self.assertIs(seen_gradients[0], gradient_buffer)
            self.assertEqual(bridge.copy_to_device_calls, copies_before)
            self.assertEqual(bridge.copy_to_host_calls, d2h_before)
            self.assertIs(embedding.grad_weights, current_gradients)
        finally:
            buffers = [
                weights,
                initial_gradients,
                embedding.grad_weights,
                gradient_buffer,
                *seen_ids,
                *outputs,
                *seen_gradients,
            ]
            for buffer in buffers:
                if isinstance(buffer, CudaBuffer) and not buffer.closed:
                    buffer.close()
        self.assertEqual(bridge.allocations, {})


class CpuInt32IndexBoundaryTests(unittest.TestCase):
    def test_cpu_backend_keeps_int32_ids_as_python_integers(self) -> None:
        backend = CPUBackend()

        ids = backend.buffer_from_host([2, 0, 2], shape=(3,), dtype="int32")

        self.assertEqual(ids, [2, 0, 2])
        self.assertTrue(all(type(value) is int for value in ids))
        host_ids = backend.buffer_to_host(ids)
        self.assertEqual(host_ids, [2, 0, 2])
        self.assertTrue(all(type(value) is int for value in host_ids))

    def test_cpu_backend_int32_rejects_bool_float_and_overflow(self) -> None:
        backend = CPUBackend()
        invalid_cases = (
            ([True], TypeError),
            ([1.0], TypeError),
            ([2**31], ValueError),
        )
        for values, error in invalid_cases:
            with self.subTest(values=values), self.assertRaises(error):
                backend.buffer_from_host(
                    cast(HostArray, values), shape=(1,), dtype="int32"
                )


class UnavailableCudaBridge(CudaBridge):
    def device_count(self) -> int:
        raise BackendUnavailableError("simulated CUDA runtime unavailable")


class EmbeddingUnavailableBackendTests(unittest.TestCase):
    def test_cuda_embedding_construction_fails_closed_without_device(self) -> None:
        backend = CUDABackend(runtime=CudaRuntime(bridge=UnavailableCudaBridge()))

        with self.assertRaisesRegex(
            BackendUnavailableError, "simulated CUDA runtime unavailable"
        ):
            Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)


class EmbeddingShapeMetadataTests(unittest.TestCase):
    def test_constructor_initializes_parameter_arrays_through_backend_contract(self) -> None:
        backend = RecordingCpuBufferBackend()

        embedding = Embedding(vocab_size=5, embedding_dim=4, seed=7, backend=backend)

        self.assertEqual(
            backend.buffer_shapes,
            [((5, 4), "float64"), ((5, 4), "float64")],
        )
        cpu_weights = cast(list[list[float]], embedding.weights)
        self.assertEqual((len(cpu_weights), len(cpu_weights[0])), (5, 4))
        self.assertEqual(embedding.grad_weights, [[0.0] * 4 for _ in range(5)])

    def test_shape_properties_do_not_depend_on_weight_storage_sequence_api(self) -> None:
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7)
        setattr(embedding, "weights", cast(BackendBuffer, ShapeOnlyBuffer((3, 2))))

        self.assertEqual(embedding.vocab_size, 3)
        self.assertEqual(embedding.embedding_dim, 2)

    def test_forward_preserves_backend_buffer_instead_of_iterating_to_python_lists(self) -> None:
        weights = ShapeOnlyBuffer((3, 2))
        output = ShapeOnlyBuffer((1, 2))
        backend = CPUBackend()
        seen_weights: list[object] = []

        def return_buffer(weights: object, ids: list[int]) -> ShapeOnlyBuffer:
            seen_weights.append(weights)
            return output

        setattr(backend, "embedding_forward", return_buffer)
        embedding = Embedding(vocab_size=3, embedding_dim=2, seed=7, backend=backend)
        setattr(embedding, "weights", cast(BackendBuffer, weights))

        result = embedding.forward([1])

        self.assertIs(result, output)
        self.assertIs(embedding.output, output)
        self.assertIs(seen_weights[0], weights)


if __name__ == "__main__":
    unittest.main()
