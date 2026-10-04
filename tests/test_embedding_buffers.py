"""Phase 10: Embedding metadata and backend-buffer contracts."""

from contextlib import ExitStack
import ctypes
from typing import cast
import unittest

from kinera.backends.base import BackendBuffer, BackendUnavailableError, HostArray
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
        self.allocations[destination][:nbytes] = ctypes.string_at(source, nbytes)

    def copy_to_host(
        self, device_id: int, destination: int, source: int, nbytes: int
    ) -> None:
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
