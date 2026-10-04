"""Host-backed contract tests for explicit CUDA buffer ownership/transfers."""

import ctypes
import unittest

from kinera.backends.base import BackendError
from kinera.backends.cuda import CUDABackend, CudaBuffer, CudaMemoryManager


class FakeCudaBridge:
    """Deterministic memory bridge used without a CUDA toolkit or GPU."""

    def __init__(self) -> None:
        self.next_pointer = 1
        self.allocations: dict[int, bytearray] = {}
        self.freed: list[int] = []
        self.host_to_device_copies = 0
        self.device_to_host_copies = 0

    def device_count(self) -> int:
        return 1

    def device_info(self, device_id: int):
        if device_id != 0:
            raise BackendError("invalid fake device")
        return {"name": "fake CUDA device", "major": 8, "minor": 0, "total_memory_bytes": 1024}

    def allocate(self, device_id: int, nbytes: int) -> int:
        if device_id != 0:
            raise BackendError("invalid fake device")
        pointer = self.next_pointer
        self.next_pointer += 1
        self.allocations[pointer] = bytearray(nbytes)
        return pointer

    def free(self, device_id: int, pointer: int) -> None:
        del self.allocations[pointer]
        self.freed.append(pointer)

    def copy_to_device(self, device_id: int, destination: int, source: int, nbytes: int) -> None:
        self.allocations[destination][:nbytes] = ctypes.string_at(source, nbytes)
        self.host_to_device_copies += 1

    def copy_to_host(self, device_id: int, destination: int, source: int, nbytes: int) -> None:
        ctypes.memmove(destination, bytes(self.allocations[source][:nbytes]), nbytes)
        self.device_to_host_copies += 1

    def synchronize(self, device_id: int) -> None:
        return None


class CudaBufferOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = FakeCudaBridge()
        self.manager = CudaMemoryManager(bridge=self.bridge)

    def test_explicit_host_device_roundtrip_preserves_shape_dtype_and_values(self) -> None:
        buffer = self.manager.copy_to_device(
            [[1.25, -2.0], [3.5, 0.0]], shape=(2, 2), dtype="float64", device_id=0
        )

        self.assertEqual(buffer.device, "cuda")
        self.assertEqual(buffer.device_id, 0)
        self.assertEqual(buffer.shape, (2, 2))
        self.assertEqual(buffer.dtype, "float64")
        self.assertEqual(buffer.nbytes, 32)
        self.assertTrue(buffer.owns_memory)
        self.assertEqual(buffer.ownership, "owned")
        self.assertFalse(buffer.closed)
        with self.assertRaises(AttributeError):
            buffer.shape = (1,)
        with self.assertRaises(AttributeError):
            buffer.dtype = "float32"
        with self.assertRaises(AttributeError):
            buffer.device_id = 1
        self.assertEqual(self.bridge.host_to_device_copies, 1)
        self.assertEqual(self.bridge.device_to_host_copies, 0)

        host_values = self.manager.copy_to_host(buffer)

        self.assertEqual(host_values, [[1.25, -2.0], [3.5, 0.0]])
        self.assertEqual(self.bridge.device_to_host_copies, 1)

    def test_release_is_idempotent_and_use_after_release_is_rejected(self) -> None:
        buffer = self.manager.copy_to_device([1.0, 2.0], shape=(2,))
        pointer = buffer.pointer

        self.manager.release(buffer)
        self.manager.release(buffer)

        self.assertTrue(buffer.closed)
        self.assertEqual(self.bridge.freed, [pointer])
        with self.assertRaisesRegex(BackendError, "released|closed"):
            self.manager.copy_to_host(buffer)

    def test_buffer_context_manager_releases_owned_allocation(self) -> None:
        with self.manager.copy_to_device([3.0], shape=(1,)) as buffer:
            pointer = buffer.pointer
            self.assertFalse(buffer.closed)

        self.assertTrue(buffer.closed)
        self.assertEqual(self.bridge.freed, [pointer])

    def test_int32_buffer_roundtrip_preserves_integer_ids_and_uses_four_byte_elements(self) -> None:
        with self.manager.copy_to_device(
            [2, 0, 2], shape=(3,), dtype="int32", device_id=0
        ) as buffer:
            self.assertEqual(buffer.shape, (3,))
            self.assertEqual(buffer.dtype, "int32")
            self.assertEqual(buffer.nbytes, 12)
            values = self.manager.copy_to_host(buffer)
            self.assertEqual(values, [2, 0, 2])
            self.assertTrue(all(type(value) is int for value in values))

    def test_backend_host_boundary_accepts_explicit_int32_index_buffers(self) -> None:
        backend = CUDABackend(
            runtime=self.manager.runtime, memory_manager=self.manager
        )
        with backend.buffer_from_host(
            [2, 0, 2], shape=(3,), dtype="int32"
        ) as buffer:
            self.assertEqual(buffer.dtype, "int32")
            self.assertEqual(backend.buffer_to_host(buffer), [2, 0, 2])

    def test_int32_buffer_rejects_bool_float_and_out_of_range_ids(self) -> None:
        cases = (
            ([True], TypeError),
            ([1.0], TypeError),
            ([2**31], ValueError),
            ([-(2**31) - 1], ValueError),
        )
        for values, error in cases:
            with self.subTest(values=values):
                allocations_before = len(self.bridge.allocations)
                with self.assertRaises(error):
                    self.manager.copy_to_device(
                        values, shape=(1,), dtype="int32", device_id=0
                    )
                self.assertEqual(len(self.bridge.allocations), allocations_before)

    def test_invalid_shapes_values_and_dtype_fail_before_allocation(self) -> None:
        cases = (
            ([], (0,), "float64", ValueError),
            ([[1.0], [2.0, 3.0]], (2, 2), "float64", ValueError),
            ([1.0, 2.0], (3,), "float64", ValueError),
            ([True], (1,), "float64", TypeError),
            ([float("inf")], (1,), "float64", ValueError),
            ([1.0], (1,), "float32", ValueError),
        )
        for values, shape, dtype, error in cases:
            with self.subTest(values=values, shape=shape, dtype=dtype):
                allocations_before = len(self.bridge.allocations)
                with self.assertRaises(error):
                    self.manager.copy_to_device(values, shape=shape, dtype=dtype)
                self.assertEqual(len(self.bridge.allocations), allocations_before)

    def test_device_mismatch_and_double_free_are_safe(self) -> None:
        buffer = self.manager.copy_to_device([1.0], shape=(1,), device_id=0)

        with self.assertRaisesRegex(BackendError, "device"):
            self.manager.copy_to_host(buffer, device_id=1)

        self.manager.release(buffer)
        self.assertEqual(len(self.bridge.freed), 1)


if __name__ == "__main__":
    unittest.main()
