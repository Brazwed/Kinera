"""CPU-only OpenCL discovery and explicit buffer lifecycle tests."""

import ctypes
import math
import struct
import subprocess
import sys
from decimal import Decimal, localcontext
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

from kinera.backends import Backend, BackendError, BackendUnavailableError
from kinera.backends.opencl.api import (
    CL_DEVICE_NOT_FOUND,
    CL_PLATFORM_NOT_FOUND_KHR,
    OpenCLApi,
    OpenCLApiError,
)
from kinera.backends.opencl.backend import OpenCLBackend
from kinera.backends.opencl import memory as opencl_memory
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.backends.opencl.runtime import OpenCLRuntime
from kinera.nn.layers.embedding import Embedding
from kinera.nn.layers.hybrid import HybridRepresentation
from kinera.nn.models.hybrid import AutoregressiveModel


class FakeOpenCLApi:
    def __init__(
        self,
        *,
        fp64: bool = True,
        max_allocation: int = 128,
        global_memory: int = 1024,
        fail_on: str | None = None,
    ) -> None:
        self.fp64 = fp64
        self.max_allocation = max_allocation
        self.global_memory = global_memory
        self.fail_on = fail_on
        self.allocations: dict[int, bytearray] = {}
        self.released: list[int] = []
        self.released_kernels: list[int] = []
        self.programs_released: list[int] = []
        self.kernel_arguments: list[tuple[int, int]] = []
        self.created_kernels: list[str] = []
        self.enqueues: list[int] = []
        self.kernel_double_arguments: list[tuple[int, float]] = []
        self.kernel_names: dict[int, str] = {}
        self.kernel_buffers: dict[int, dict[int, int]] = {}
        self.kernel_ints: dict[int, dict[int, int]] = {}
        self.kernel_doubles: dict[int, dict[int, float]] = {}
        self.read_sizes: list[int] = []
        self.write_sizes: list[int] = []
        self.built_sources: list[str] = []
        self._next_handle = 100
        self._next_kernel = 30

    def platforms(self):
        return [1]

    def platform_info(self, platform):
        return {"name": "Fake OpenCL", "vendor": "Kinera test"}

    def devices(self, platform):
        return [2]

    def device_info(self, device, platform_info):
        extensions = ("cl_khr_fp64",) if self.fp64 else ()
        return {
            "name": "Fake Radeon",
            "vendor": "AMD",
            "vendor_id": 0x1002,
            "device_type": 1 << 2,
            "platform_name": platform_info["name"],
            "platform_vendor": platform_info["vendor"],
            "version": "OpenCL 3.0 Fake",
            "opencl_c_version": "OpenCL C 1.2 Fake",
            "driver_version": "test-driver",
            "extensions": extensions,
            "pci_bus_info": {
                "domain": 0,
                "bus": 3,
                "device": 0,
                "function": 0,
            },
            "global_mem_size": self.global_memory,
            "max_mem_alloc_size": self.max_allocation,
            "local_mem_size": 4096,
            "max_work_group_size": 256,
            "max_work_item_sizes": (256, 256, 64),
            "double_fp_config": 0x3F if self.fp64 else 0,
            "available": True,
            "compiler_available": True,
        }

    def create_context(self, device):
        return 10

    def create_command_queue(self, context, device):
        return 11

    def create_buffer(self, context, nbytes):
        if self.fail_on in {"create_buffer", f"create_buffer_{nbytes}"}:
            raise RuntimeError("injected OpenCL allocation failure")
        handle = self._next_handle
        self._next_handle += 1
        self.allocations[handle] = bytearray(nbytes)
        return handle

    def write_buffer(self, queue, buffer, host_address, nbytes):
        self.write_sizes.append(nbytes)
        self.allocations[buffer][:] = ctypes.string_at(host_address, nbytes)

    def read_buffer(self, queue, buffer, host_address, nbytes):
        if self.fail_on == "read_buffer":
            raise RuntimeError("injected OpenCL status read failure")
        self.read_sizes.append(nbytes)
        ctypes.memmove(host_address, bytes(self.allocations[buffer]), nbytes)

    def release_mem(self, handle):
        self.released.append(handle)
        del self.allocations[handle]
        if self.fail_on == "release_mem":
            raise RuntimeError("injected OpenCL memory release failure")

    def release_command_queue(self, handle):
        pass

    def release_context(self, handle):
        pass

    def build_program(self, context, device, source, options):
        if self.fail_on == "build_program":
            raise RuntimeError("injected OpenCL program build failure")
        self.built_sources.append(source)
        return 20

    def create_kernel(self, program, name):
        if self.fail_on == "create_kernel":
            raise RuntimeError("injected OpenCL kernel creation failure")
        if self.fail_on == f"create_{name}":
            raise RuntimeError(f"injected OpenCL creation failure for {name}")
        self.created_kernels.append(name)
        handle = self._next_kernel
        self._next_kernel += 1
        self.kernel_names[handle] = name
        self.kernel_buffers[handle] = {}
        self.kernel_ints[handle] = {}
        self.kernel_doubles[handle] = {}
        return handle

    def set_kernel_buffer(self, kernel, index, buffer):
        kernel_name = self.kernel_names.get(kernel, "")
        if self.fail_on in {
            f"set_kernel_buffer_{index}",
            f"set_kernel_buffer_{index}_{kernel_name}",
        }:
            raise RuntimeError("injected OpenCL argument failure")
        self.kernel_arguments.append((index, buffer))
        self.kernel_buffers[kernel][index] = buffer

    def set_kernel_int(self, kernel, index, value):
        kernel_name = self.kernel_names.get(kernel, "")
        if self.fail_on in {
            f"set_kernel_int_{index}",
            f"set_kernel_int_{index}_{kernel_name}",
        }:
            raise RuntimeError("injected OpenCL scalar argument failure")
        self.kernel_ints[kernel][index] = value

    def set_kernel_double(self, kernel, index, value):
        if self.fail_on == f"set_kernel_double_{index}":
            raise RuntimeError("injected OpenCL double argument failure")
        self.kernel_double_arguments.append((index, value))
        self.kernel_doubles[kernel][index] = value

    def enqueue_1d(self, queue, kernel, global_size):
        if self.fail_on == "enqueue_1d":
            raise RuntimeError("injected OpenCL enqueue failure")
        kernel_name = self.kernel_names[kernel]
        if self.fail_on == f"enqueue_{kernel_name}":
            raise RuntimeError(f"injected enqueue failure for {kernel_name}")
        self.enqueues.append(global_size)
        buffers = self.kernel_buffers[kernel]
        values = self.kernel_ints[kernel]
        if kernel_name == "kinera_embedding_validate_ids_i32":
            ids = self.allocations[buffers[0]]
            ids_count, vocab_size = values[2], values[3]
            invalid_position = -1
            for position in range(ids_count):
                token_id = struct.unpack_from("<i", ids, position * 4)[0]
                if token_id < 0 or token_id >= vocab_size:
                    invalid_position = position
                    break
            self.allocations[buffers[1]][:4] = struct.pack("<i", invalid_position)
        elif kernel_name == "kinera_hybrid_representation_validate_finite_f64":
            context_size = values[4]
            dimensions = (values[5], values[6], values[7])
            status = 0
            for channel, dimension in enumerate(dimensions):
                for index in range(context_size * dimension):
                    value = struct.unpack_from(
                        "<d", self.allocations[buffers[channel]], index * 8
                    )[0]
                    if not math.isfinite(value):
                        status = channel + 1
                        break
                if status:
                    break
            self.allocations[buffers[3]][:4] = struct.pack("<i", status)
        elif kernel_name == "kinera_hybrid_representation_forward_f64":
            char_values = self.allocations[buffers[0]]
            feature_values = self.allocations[buffers[1]]
            syllable_values = self.allocations[buffers[2]]
            output = self.allocations[buffers[3]]
            context_size, char_dim, feature_dim, syllable_dim = (
                values[4],
                values[5],
                values[6],
                values[7],
            )
            output_dim = char_dim + feature_dim + syllable_dim
            for row in range(context_size):
                output_offset = row * output_dim * 8
                char_start = row * char_dim * 8
                feature_start = row * feature_dim * 8
                syllable_start = row * syllable_dim * 8
                output[output_offset : output_offset + char_dim * 8] = char_values[
                    char_start : char_start + char_dim * 8
                ]
                feature_offset = output_offset + char_dim * 8
                output[feature_offset : feature_offset + feature_dim * 8] = (
                    feature_values[feature_start : feature_start + feature_dim * 8]
                )
                syllable_offset = feature_offset + feature_dim * 8
                output[syllable_offset : syllable_offset + syllable_dim * 8] = (
                    syllable_values[syllable_start : syllable_start + syllable_dim * 8]
                )
        elif kernel_name == "kinera_mean_pool_validate_finite_f64":
            vectors = self.allocations[buffers[0]]
            context_size, feature_dim = values[2], values[3]
            has_non_finite = any(
                not math.isfinite(struct.unpack_from("<d", vectors, index * 8)[0])
                for index in range(context_size * feature_dim)
            )
            self.allocations[buffers[1]][:4] = struct.pack(
                "<i", int(has_non_finite)
            )
        elif kernel_name == "kinera_mean_pool_forward_f64":
            vectors = self.allocations[buffers[0]]
            pooled = self.allocations[buffers[1]]
            context_size, feature_dim, compensated_sum = (
                values[2],
                values[3],
                values[4],
            )
            inverse_context_size = self.kernel_doubles[kernel][5]
            for feature in range(feature_dim):
                high = 0.0
                low = 0.0
                for position in range(context_size):
                    value = struct.unpack_from(
                        "<d", vectors, (position * feature_dim + feature) * 8
                    )[0]
                    next_value = high + value
                    if compensated_sum:
                        if abs(high) >= abs(value):
                            low += (high - next_value) + value
                        else:
                            low += (value - next_value) + high
                    high = next_value
                total = high
                if compensated_sum and low != 0.0 and math.isfinite(low):
                    total += low
                struct.pack_into(
                    "<d", pooled, feature * 8, total * inverse_context_size
                )
        elif kernel_name == "kinera_embedding_forward_f64":
            weights = self.allocations[buffers[0]]
            ids = self.allocations[buffers[1]]
            output = self.allocations[buffers[2]]
            embedding_dim, ids_count = values[3], values[4]
            for output_row in range(ids_count):
                token_id = struct.unpack_from("<i", ids, output_row * 4)[0]
                source = (token_id * embedding_dim) * 8
                target = output_row * embedding_dim * 8
                output[target : target + embedding_dim * 8] = weights[
                    source : source + embedding_dim * 8
                ]
        elif kernel_name == "kinera_embedding_validate_backward_inputs":
            ids = self.allocations[buffers[0]]
            grad_output = self.allocations[buffers[1]]
            ids_count, vocab_size, embedding_dim = (
                values[3],
                values[4],
                values[5],
            )
            status = 0
            for position in range(ids_count):
                token_id = struct.unpack_from("<i", ids, position * 4)[0]
                if token_id < 0 or token_id >= vocab_size:
                    status = position + 1
                    break
                for dimension in range(embedding_dim):
                    index = position * embedding_dim + dimension
                    value = struct.unpack_from("<d", grad_output, index * 8)[0]
                    if not math.isfinite(value):
                        status = -(index + 1)
                        break
                if status:
                    break
            self.allocations[buffers[2]][:4] = struct.pack("<i", status)
        elif kernel_name == "kinera_embedding_backward_f64":
            grad_weights = self.allocations[buffers[0]]
            ids = self.allocations[buffers[1]]
            grad_output = self.allocations[buffers[2]]
            vocab_size, embedding_dim, ids_count = (
                values[3],
                values[4],
                values[5],
            )
            for row in range(vocab_size):
                for dimension in range(embedding_dim):
                    element = row * embedding_dim + dimension
                    accumulated = struct.unpack_from("<d", grad_weights, element * 8)[0]
                    for position in range(ids_count):
                        token_id = struct.unpack_from("<i", ids, position * 4)[0]
                        if token_id == row:
                            index = position * embedding_dim + dimension
                            value = struct.unpack_from("<d", grad_output, index * 8)[0]
                            accumulated = accumulated + value
                    struct.pack_into("<d", grad_weights, element * 8, accumulated)
        elif kernel_name == "kinera_embedding_step_f64":
            weights = self.allocations[buffers[0]]
            gradients = self.allocations[buffers[1]]
            vocab_size, embedding_dim = values[2], values[3]
            learning_rate = self.kernel_doubles[kernel][4]
            for element in range(vocab_size * embedding_dim):
                weight = struct.unpack_from("<d", weights, element * 8)[0]
                gradient = struct.unpack_from("<d", gradients, element * 8)[0]
                update = learning_rate * gradient
                struct.pack_into("<d", weights, element * 8, weight - update)

    def release_kernel(self, handle):
        self.released_kernels.append(handle)
        if self.fail_on in {
            "release_kernel",
            f"release_{self.kernel_names.get(handle, '')}",
        }:
            raise RuntimeError("injected OpenCL kernel release failure")

    def release_program(self, handle):
        self.programs_released.append(handle)


class OpenCLScaffoldDiscoveryTests(unittest.TestCase):
    def test_api_enumeration_keeps_empty_results_distinct_from_errors(self) -> None:
        api = OpenCLApi.__new__(OpenCLApi)

        def platform_result(status: int):
            def get_platform_ids(entry_count, platforms, count_pointer):
                ctypes.cast(
                    count_pointer, ctypes.POINTER(ctypes.c_uint)
                ).contents.value = 0
                return status

            return get_platform_ids

        api._cl_get_platform_ids = platform_result(-30)
        with self.assertRaisesRegex(OpenCLApiError, "clGetPlatformIDs.*-30"):
            api.platforms()
        api._cl_get_platform_ids = platform_result(CL_PLATFORM_NOT_FOUND_KHR)
        self.assertEqual(api.platforms(), [])

        def device_result(status: int):
            def get_device_ids(platform, device_type, entry_count, devices, count_pointer):
                ctypes.cast(
                    count_pointer, ctypes.POINTER(ctypes.c_uint)
                ).contents.value = 0
                return status

            return get_device_ids

        api._cl_get_device_ids = device_result(-30)
        with self.assertRaisesRegex(OpenCLApiError, "clGetDeviceIDs.*-30"):
            api.devices(1)
        api._cl_get_device_ids = device_result(CL_DEVICE_NOT_FOUND)
        self.assertEqual(api.devices(1), [])

    def test_cpu_backend_import_does_not_load_opencl_modules(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import kinera.backends; "
                "kinera.backends.get_backend('cpu'); "
                "assert not any(name.startswith('kinera.backends.opencl') "
                "for name in sys.modules)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_implements_contract_and_is_unavailable_without_gpu(self) -> None:
        class NoGpuApi:
            def platforms(self):
                return []

        backend = OpenCLBackend(runtime=OpenCLRuntime(api=NoGpuApi()))
        self.assertIsInstance(backend, Backend)
        self.assertEqual(backend.name, "opencl")
        self.assertFalse(backend.is_available)
        with self.assertRaisesRegex(BackendUnavailableError, "no GPU devices"):
            backend.buffer_from_host([1.0], shape=(1,))
        with self.assertRaisesRegex(TypeError, "weights as an OpenCLBuffer"):
            backend.embedding_forward(None, [])
        backend.close()

    def test_device_discovery_reports_pci_and_fp64_capability(self) -> None:
        runtime = OpenCLRuntime(api=FakeOpenCLApi())
        device = runtime.require_device(0, require_fp64=True)
        self.assertEqual(device.pci_bdf, "0000:03:00.0")
        self.assertEqual(device.global_mem_size, 1024)
        self.assertEqual(device.max_mem_alloc_size, 128)

    def test_fp64_requirement_fails_closed(self) -> None:
        runtime = OpenCLRuntime(api=FakeOpenCLApi(fp64=False))
        with self.assertRaisesRegex(BackendUnavailableError, "RUSTICL_FEATURES=fp64"):
            runtime.require_device(0, require_fp64=True)

    def test_explicit_float_and_int_transfers_and_allocation_limit(self) -> None:
        api = FakeOpenCLApi(max_allocation=64)
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        floats = backend.buffer_from_host([[1.5, -2.0], [3.0, 4.25]], shape=(2, 2))
        integers = backend.buffer_from_host([1, -2, 3], shape=(3,), dtype="int32")
        try:
            self.assertEqual(backend.buffer_to_host(floats), [[1.5, -2.0], [3.0, 4.25]])
            self.assertEqual(backend.buffer_to_host(integers), [1, -2, 3])
            self.assertEqual(floats.nbytes, 32)
            self.assertEqual(integers.nbytes, 12)
            with self.assertRaisesRegex(BackendError, "exceeds device limit"):
                backend.memory.allocate((9,), dtype="float64")
        finally:
            floats.close()
            integers.close()
            backend.close()
        self.assertEqual(len(api.allocations), 0)

    def test_copy_to_device_ctypes_failure_releases_buffer_and_preserves_error(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        staging_error = MemoryError("injected ctypes array construction failure")

        class FailingArrayType:
            def __mul__(self, length):
                return self

            def __call__(self, *values):
                raise staging_error

        class CtypesProxy:
            def __init__(self):
                self.c_double_reads = 0
                self.c_int32 = ctypes.c_int32
                self.sizeof = ctypes.sizeof
                self.addressof = ctypes.addressof

            @property
            def c_double(self):
                self.c_double_reads += 1
                # copy_to_device captures the array type before allocate();
                # allocate then asks sizeof() for the real scalar type.
                if self.c_double_reads == 1:
                    return FailingArrayType()
                return ctypes.c_double

        ctypes_proxy = CtypesProxy()
        try:
            with patch.object(opencl_memory, "ctypes", ctypes_proxy):
                with self.assertRaises(MemoryError) as raised:
                    backend.buffer_from_host([1.25], shape=(1,), dtype="float64")

            self.assertIs(raised.exception, staging_error)
            self.assertEqual(ctypes_proxy.c_double_reads, 2)
            self.assertEqual(api.released, [100])
            self.assertEqual(api.allocations, {})
            self.assertEqual(len(backend.memory._buffers), 0)
        finally:
            backend.close()


class OpenCLScaffoldTests(unittest.TestCase):

    def test_copy_to_device_write_failure_releases_once_and_preserves_error(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        write_error = RuntimeError("injected blocking H2D failure")
        try:
            with patch.object(api, "write_buffer", side_effect=write_error):
                with self.assertRaises(RuntimeError) as raised:
                    backend.buffer_from_host([1.25], shape=(1,), dtype="float64")

            self.assertIs(raised.exception, write_error)
            self.assertEqual(api.released, [100])
            self.assertEqual(api.allocations, {})
            self.assertEqual(len(backend.memory._buffers), 0)
        finally:
            backend.close()

    def test_copy_to_device_preserves_transfer_error_when_release_fails(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        write_error = RuntimeError("injected blocking H2D failure")
        release_error = RuntimeError("injected release_mem failure")
        release_attempts: list[int] = []
        original_release_mem = api.release_mem

        def fail_once(handle: int) -> None:
            release_attempts.append(handle)
            if len(release_attempts) == 1:
                raise release_error
            original_release_mem(handle)

        try:
            with (
                patch.object(api, "write_buffer", side_effect=write_error),
                patch.object(api, "release_mem", side_effect=fail_once) as release_mem,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    backend.buffer_from_host([1.25], shape=(1,), dtype="float64")

                self.assertIs(raised.exception, write_error)
                self.assertIs(raised.exception.__cause__, release_error)
                self.assertEqual(release_mem.call_count, 1)
                self.assertEqual(api.allocations, {100: bytearray(8)})
                retained_buffer = next(iter(backend.memory._buffers))
                self.assertFalse(retained_buffer.closed)

                # The manager keeps the failed cleanup owned and retries it
                # when it is explicitly closed after the API recovers.
                backend.close()
                self.assertEqual(release_mem.call_count, 2)
                self.assertTrue(retained_buffer.closed)
                self.assertEqual(api.released, [100])
                self.assertEqual(api.allocations, {})

            self.assertEqual(release_attempts, [100, 100])
        finally:
            backend.close()

    def test_copy_zeros_write_failure_still_releases_once(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        write_error = RuntimeError("injected zero-buffer H2D failure")
        try:
            with patch.object(api, "write_buffer", side_effect=write_error):
                with self.assertRaises(RuntimeError) as raised:
                    backend.buffer_zeros(shape=(1,), dtype="float64")

            self.assertIs(raised.exception, write_error)
            self.assertEqual(api.released, [100])
            self.assertEqual(api.allocations, {})
            self.assertEqual(len(backend.memory._buffers), 0)
        finally:
            backend.close()

    def test_successful_copy_transfers_ownership_without_early_release(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        try:
            buffer = backend.buffer_from_host(
                [1.25, -2.5], shape=(2,), dtype="float64"
            )
            handle = buffer.handle
            self.assertFalse(buffer.closed)
            self.assertIn(handle, api.allocations)
            self.assertEqual(api.released, [])
            self.assertEqual(backend.buffer_to_host(buffer), [1.25, -2.5])
            self.assertEqual(api.released, [])

            buffer.close()
            buffer.close()
            self.assertTrue(buffer.closed)
            self.assertEqual(api.released, [handle])
            self.assertEqual(api.allocations, {})
        finally:
            backend.close()

    def test_failed_read_keeps_the_callers_buffer_open(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        buffer = backend.buffer_from_host([1.25], shape=(1,), dtype="float64")
        read_error = RuntimeError("injected blocking D2H failure")
        try:
            with patch.object(api, "read_buffer", side_effect=read_error):
                with self.assertRaises(RuntimeError) as raised:
                    backend.buffer_to_host(buffer)

            self.assertIs(raised.exception, read_error)
            self.assertFalse(buffer.closed)
            self.assertIn(buffer.handle, api.allocations)
            self.assertEqual(api.released, [])
        finally:
            buffer.close()
            backend.close()

    def test_backward_rejects_host_buffers_bad_dtype_shape_and_foreign_owner(self) -> None:
        first_api = FakeOpenCLApi()
        second_api = FakeOpenCLApi()
        first = OpenCLBackend(runtime=OpenCLRuntime(api=first_api))
        second = OpenCLBackend(runtime=OpenCLRuntime(api=second_api))
        weights = first.buffer_from_host(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=(2, 3)
        )
        inputs = first.buffer_from_host([1.0, -1.0, 2.0], shape=(3,))
        grad_output = first.buffer_from_host([0.5, -0.25], shape=(2,))
        wrong_grad_output = first.buffer_from_host([1.0], shape=(1,))
        wrong_shape = first.buffer_from_host([1.0, 2.0], shape=(2,))
        int_grad = first.buffer_from_host([1, 2], shape=(2,), dtype="int32")
        foreign_inputs = second.buffer_from_host([1.0, 2.0, 3.0], shape=(3,))
        try:
            with self.assertRaisesRegex(TypeError, "OpenCLBuffer"):
                first.linear_backward(weights, [1.0, 2.0, 3.0], grad_output)
            with self.assertRaisesRegex(ValueError, "weights columns"):
                first.linear_backward(weights, wrong_shape, grad_output)
            with self.assertRaisesRegex(ValueError, "grad_output must match"):
                first.linear_backward(weights, inputs, wrong_grad_output)
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                first.linear_backward(weights, inputs, int_grad)
            with self.assertRaisesRegex(
                BackendError, "must belong to this OpenCL backend"
            ):
                first.linear_backward(weights, foreign_inputs, grad_output)

            inputs.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                first.linear_backward(weights, inputs, grad_output)
        finally:
            foreign_inputs.close()
            int_grad.close()
            wrong_shape.close()
            wrong_grad_output.close()
            inputs.close()
            grad_output.close()
            weights.close()
            second.close()
            first.close()

    def test_backward_releases_partial_outputs_when_opencl_call_fails(self) -> None:
        for fail_on in (
            "create_kernel",
            "set_kernel_buffer_2",
            "enqueue_1d",
        ):
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi(fail_on=fail_on)
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                weights = backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
                inputs = backend.buffer_from_host([0.5, -1.0], shape=(2,))
                grad_output = backend.buffer_from_host([3.0], shape=(1,))
                try:
                    message = {
                        "create_kernel": "kernel creation",
                        "set_kernel_buffer_2": "OpenCL argument",
                        "enqueue_1d": "enqueue failure",
                    }[fail_on]
                    with self.assertRaisesRegex(RuntimeError, message):
                        backend.linear_backward(weights, inputs, grad_output)
                    self.assertEqual(len(api.allocations), 3)
                    self.assertEqual(
                        api.released_kernels,
                        [] if fail_on == "create_kernel" else [30],
                    )
                finally:
                    weights.close()
                    inputs.close()
                    grad_output.close()
                    backend.close()
                self.assertEqual(len(api.allocations), 0)

    def test_step_validates_learning_rate_without_mutating_or_launching(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        weights = backend.buffer_from_host([[1.0, -2.0]], shape=(1, 2))
        bias = backend.buffer_from_host([0.5], shape=(1,))
        grad_weights = backend.buffer_from_host([[0.25, -0.75]], shape=(1, 2))
        grad_bias = backend.buffer_from_host([-0.125], shape=(1,))
        invalid = (
            ("bool", True, TypeError, "numeric"),
            ("string", "0.1", TypeError, "numeric"),
            ("nan", float("nan"), ValueError, "finite"),
            ("positive infinity", float("inf"), ValueError, "finite"),
            ("negative infinity", float("-inf"), ValueError, "finite"),
            ("negative", -0.1, ValueError, "negative"),
            ("unrepresentable integer", 10**1000, ValueError, "finite"),
        )
        try:
            original_weights = backend.buffer_to_host(weights)
            original_bias = backend.buffer_to_host(bias)
            for label, rate, error_type, message in invalid:
                with self.subTest(rate=label):
                    with self.assertRaisesRegex(error_type, message):
                        backend.linear_step(
                            weights, bias, grad_weights, grad_bias, rate
                        )
                    self.assertEqual(backend.buffer_to_host(weights), original_weights)
                    self.assertEqual(backend.buffer_to_host(bias), original_bias)
            self.assertEqual(api.created_kernels, [])
            self.assertEqual(api.enqueues, [])
        finally:
            grad_bias.close()
            grad_weights.close()
            bias.close()
            weights.close()
            backend.close()

    def test_step_validates_shapes_dtype_closed_buffers_and_ownership(self) -> None:
        first_api = FakeOpenCLApi()
        second_api = FakeOpenCLApi()
        first = OpenCLBackend(runtime=OpenCLRuntime(api=first_api))
        second = OpenCLBackend(runtime=OpenCLRuntime(api=second_api))
        weights = first.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
        bias = first.buffer_from_host([0.5], shape=(1,))
        grad_weights = first.buffer_from_host([[0.25, -0.75]], shape=(1, 2))
        grad_bias = first.buffer_from_host([-0.125], shape=(1,))
        wrong_bias = first.buffer_from_host([0.5, 0.25], shape=(2,))
        wrong_grad_weights = first.buffer_from_host([[0.25]], shape=(1, 1))
        wrong_grad_bias = first.buffer_from_host([0.1, 0.2], shape=(2,))
        int_grad_bias = first.buffer_from_host([1], shape=(1,), dtype="int32")
        foreign_grad = second.buffer_from_host([[0.25, -0.75]], shape=(1, 2))
        try:
            original_weights = first.buffer_to_host(weights)
            original_bias = first.buffer_to_host(bias)
            with self.assertRaisesRegex(TypeError, "OpenCLBuffer"):
                first.linear_step([[1.0, 2.0]], bias, grad_weights, grad_bias, 0.1)
            with self.assertRaisesRegex(ValueError, "bias must match"):
                first.linear_step(weights, wrong_bias, grad_weights, grad_bias, 0.1)
            with self.assertRaisesRegex(ValueError, "grad_weights must match"):
                first.linear_step(weights, bias, wrong_grad_weights, grad_bias, 0.1)
            with self.assertRaisesRegex(ValueError, "grad_bias must match"):
                first.linear_step(weights, bias, grad_weights, wrong_grad_bias, 0.1)
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                first.linear_step(weights, bias, grad_weights, int_grad_bias, 0.1)
            with self.assertRaisesRegex(BackendError, "belong to this OpenCL backend"):
                first.linear_step(weights, bias, foreign_grad, grad_bias, 0.1)
            grad_weights.close()
            with self.assertRaisesRegex(BackendError, "released/closed"):
                first.linear_step(weights, bias, grad_weights, grad_bias, 0.1)
            self.assertEqual(first_api.enqueues, [])
            self.assertEqual(first.buffer_to_host(weights), original_weights)
            self.assertEqual(first.buffer_to_host(bias), original_bias)
        finally:
            foreign_grad.close()
            int_grad_bias.close()
            wrong_grad_bias.close()
            wrong_grad_weights.close()
            wrong_bias.close()
            grad_weights.close()
            grad_bias.close()
            bias.close()
            weights.close()
            second.close()
            first.close()

    def test_step_zero_returns_parameter_buffers_without_launch(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        weights = backend.buffer_from_host([[1.0, -2.0]], shape=(1, 2))
        bias = backend.buffer_from_host([0.5], shape=(1,))
        grad_weights = backend.buffer_from_host([[0.25, -0.75]], shape=(1, 2))
        grad_bias = backend.buffer_from_host([-0.125], shape=(1,))
        try:
            original_weights = backend.buffer_to_host(weights)
            original_bias = backend.buffer_to_host(bias)
            updated_weights, updated_bias = backend.linear_step(
                weights, bias, grad_weights, grad_bias, 0.0
            )
            self.assertIs(updated_weights, weights)
            self.assertIs(updated_bias, bias)
            self.assertEqual(backend.buffer_to_host(weights), original_weights)
            self.assertEqual(backend.buffer_to_host(bias), original_bias)
            self.assertEqual(api.created_kernels, [])
            self.assertEqual(api.enqueues, [])
        finally:
            grad_bias.close()
            grad_weights.close()
            bias.close()
            weights.close()
            backend.close()

    def test_step_opencl_failures_are_explicit_and_release_kernel(self) -> None:
        failures = (
            ("create_kernel", []),
            ("set_kernel_buffer_2", [30]),
            ("set_kernel_int_4", [30]),
            ("set_kernel_double_6", [30]),
            ("enqueue_1d", [30]),
        )
        for fail_on, expected_released_kernels in failures:
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi(fail_on=fail_on)
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                weights = backend.buffer_from_host([[1.0, -2.0]], shape=(1, 2))
                bias = backend.buffer_from_host([0.5], shape=(1,))
                grad_weights = backend.buffer_from_host(
                    [[0.25, -0.75]], shape=(1, 2)
                )
                grad_bias = backend.buffer_from_host([-0.125], shape=(1,))
                original_weights = backend.buffer_to_host(weights)
                original_bias = backend.buffer_to_host(bias)
                try:
                    with self.assertRaisesRegex(RuntimeError, "injected OpenCL"):
                        backend.linear_step(
                            weights, bias, grad_weights, grad_bias, 0.125
                        )
                    self.assertEqual(api.enqueues, [])
                    self.assertEqual(api.released_kernels, expected_released_kernels)
                    self.assertEqual(backend.buffer_to_host(weights), original_weights)
                    self.assertEqual(backend.buffer_to_host(bias), original_bias)
                finally:
                    grad_bias.close()
                    grad_weights.close()
                    bias.close()
                    weights.close()
                    backend.close()


class OpenCLEmbeddingForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi()
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))
        self.weights = self.backend.buffer_from_host(
            [[0.0, -0.0, 1.25], [-2.5, 3.0, 4.125], [9.5, -7.0, 0.5]],
            shape=(3, 3),
            dtype="float64",
        )

    def tearDown(self) -> None:
        self.backend.close()

    def _ids(self, values: list[int], *, shape: tuple[int, ...] | None = None):
        return self.backend.buffer_from_host(
            values,
            shape=(len(values),) if shape is None else shape,
            dtype="int32",
        )

    def test_lookup_copies_rows_exactly_and_preserves_repeated_ids(self) -> None:
        ids = self._ids([2, 0, 2, 1])
        output = None
        try:
            output = self.backend.embedding_forward(self.weights, ids)
            self.assertNotIsInstance(output, list)
            self.assertEqual(output.dtype, "float64")
            self.assertEqual(output.shape, (4, 3))
            self.assertIs(output._manager, self.backend.memory)
            self.assertEqual(self.api.read_sizes, [ctypes.sizeof(ctypes.c_int32)])
            self.assertEqual(
                self.backend.buffer_to_host(output),
                [
                    [9.5, -7.0, 0.5],
                    [0.0, -0.0, 1.25],
                    [9.5, -7.0, 0.5],
                    [-2.5, 3.0, 4.125],
                ],
            )
            self.assertEqual(
                self.api.read_sizes,
                [ctypes.sizeof(ctypes.c_int32), output.nbytes],
            )
            self.assertFalse(ids.closed)
            self.assertFalse(self.weights.closed)
        finally:
            if output is not None:
                output.close()
            ids.close()

    def test_invalid_resident_ids_fail_before_lookup_and_release_preflight(self) -> None:
        for invalid_id in (-1, 3, -(2**31), 2**31 - 1):
            with self.subTest(invalid_id=invalid_id):
                ids = self._ids([1, invalid_id])
                before_enqueues = len(self.api.enqueues)
                before_releases = list(self.api.released)
                try:
                    with self.assertRaisesRegex(ValueError, "position 1"):
                        self.backend.embedding_forward(self.weights, ids)
                    self.assertEqual(self.api.enqueues[before_enqueues:], [1])
                    self.assertEqual(
                        self.api.released[len(before_releases) :], [ids.handle + 1]
                    )
                    self.assertEqual(
                        self.api.created_kernels[-1],
                        "kinera_embedding_validate_ids_i32",
                    )
                    self.assertEqual(
                        len(self.api.allocations), 2,
                        "only caller-owned weights and IDs remain allocated",
                    )
                finally:
                    ids.close()

    def test_host_ids_must_be_explicitly_transferred_and_host_layer_validates(self) -> None:
        with self.assertRaisesRegex(TypeError, "int32 OpenCLBuffer"):
            self.backend.embedding_forward(self.weights, [-1])
        layer = Embedding(3, 3, seed=7, backend=self.backend)
        try:
            layer.forward([1, 2])
            previous = (layer.last_ids, layer.output, layer.grad_weights)
            with self.assertRaisesRegex(ValueError, "fora do vocabulário"):
                layer.forward([-1])
            self.assertEqual(
                (layer.last_ids, layer.output, layer.grad_weights), previous
            )
            self.assertFalse(previous[0].closed)
            self.assertFalse(previous[1].closed)
            self.assertFalse(previous[2].closed)
        finally:
            layer.close()

    def test_empty_host_ids_return_empty_without_zero_sized_output_or_kernel(self) -> None:
        before_allocations = set(self.api.allocations)
        before_kernels = list(self.api.created_kernels)
        before_enqueues = list(self.api.enqueues)
        self.assertEqual(self.backend.embedding_forward(self.weights, []), [])
        self.assertEqual(set(self.api.allocations), before_allocations)
        self.assertEqual(self.api.created_kernels, before_kernels)
        self.assertEqual(self.api.enqueues, before_enqueues)

        layer = Embedding(3, 3, seed=7, backend=self.backend)
        try:
            self.assertEqual(layer.forward([]), [])
            self.assertEqual(layer.last_ids, [])
            self.assertEqual(layer.output, [])
            self.assertEqual(layer.grad_weights.shape, (3, 3))
            self.assertEqual(self.api.created_kernels, before_kernels)
        finally:
            layer.close()

    def test_input_contract_rejects_shape_dtype_closed_and_foreign_owner(self) -> None:
        wrong_rank = self.backend.buffer_from_host(
            [[1, 0]], shape=(1, 2), dtype="int32"
        )
        wrong_weight_shape = self.backend.buffer_from_host([1.0], shape=(1,))
        wrong_ids_dtype = self.backend.buffer_from_host([0.0], shape=(1,))
        wrong_weight_dtype = self.backend.buffer_from_host(
            [[1, 0]], shape=(1, 2), dtype="int32"
        )
        other_api = FakeOpenCLApi()
        other_backend = OpenCLBackend(runtime=OpenCLRuntime(api=other_api))
        foreign_ids = other_backend.buffer_from_host([1], shape=(1,), dtype="int32")
        foreign_weights = other_backend.buffer_from_host(
            [[1.0, 2.0]], shape=(1, 2), dtype="float64"
        )
        closed_ids = self._ids([0])
        closed_ids.close()
        closed_weights = self.backend.buffer_from_host(
            [[1.0]], shape=(1, 1), dtype="float64"
        )
        closed_weights.close()
        try:
            with self.assertRaisesRegex(ValueError, "rank 1"):
                self.backend.embedding_forward(self.weights, wrong_rank)
            with self.assertRaisesRegex(ValueError, "rank 2"):
                self.backend.embedding_forward(wrong_weight_shape, [])
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                self.backend.embedding_forward(wrong_weight_dtype, [])
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_forward(self.weights, foreign_ids)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_forward(foreign_weights, [])
            with self.assertRaisesRegex(ValueError, "dtype int32"):
                self.backend.embedding_forward(self.weights, wrong_ids_dtype)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_forward(self.weights, closed_ids)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_forward(closed_weights, [])
            with self.assertRaisesRegex(ValueError, "outside"):
                # Valid shape and dtype, but an out-of-vocabulary device value.
                bad_id = self._ids([3])
                try:
                    self.backend.embedding_forward(self.weights, bad_id)
                finally:
                    bad_id.close()
        finally:
            closed_ids.close()
            foreign_ids.close()
            foreign_weights.close()
            other_backend.close()
            closed_weights.close()
            wrong_ids_dtype.close()
            wrong_weight_dtype.close()
            wrong_weight_shape.close()
            wrong_rank.close()

    def test_buffer_on_different_selected_device_is_rejected(self) -> None:
        backend_for_other_device = OpenCLBackend(
            runtime=self.backend.runtime,
            memory_manager=self.backend.memory,
            device_id=1,
        )
        with self.assertRaisesRegex(BackendError, "selected OpenCL GPU"):
            backend_for_other_device.embedding_forward(self.weights, [])

    def test_preflight_and_lookup_failures_release_temporaries_and_keep_inputs(self) -> None:
        failure_cases = (
            ("build_program", "build failure"),
            ("create_buffer", "allocation failure"),
            ("create_kinera_embedding_validate_ids_i32", "creation failure"),
            ("set_kernel_buffer_0", "argument failure"),
            ("enqueue_kinera_embedding_validate_ids_i32", "enqueue failure"),
            ("set_kernel_int_3", "scalar argument failure"),
            ("read_buffer", "status read failure"),
            ("create_kinera_embedding_forward_f64", "creation failure"),
            ("set_kernel_buffer_2", "argument failure"),
            ("set_kernel_int_4", "scalar argument failure"),
            ("enqueue_kinera_embedding_forward_f64", "enqueue failure"),
        )
        for fail_on, message in failure_cases:
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi()
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                weights = backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
                ids = backend.buffer_from_host([0], shape=(1,), dtype="int32")
                api.fail_on = fail_on
                before_release_mem = list(api.released)
                before_release_kernels = list(api.released_kernels)
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        backend.embedding_forward(weights, ids)
                    self.assertFalse(weights.closed)
                    self.assertFalse(ids.closed)
                    self.assertEqual(
                        set(api.allocations), {weights.handle, ids.handle}
                    )
                    newly_released = api.released[len(before_release_mem) :]
                    expected_temp_releases = {
                        "build_program": 0,
                        "create_buffer": 0,
                        "create_kinera_embedding_validate_ids_i32": 1,
                        "set_kernel_buffer_0": 1,
                        "enqueue_kinera_embedding_validate_ids_i32": 1,
                        "set_kernel_int_3": 1,
                        "read_buffer": 1,
                        "create_kinera_embedding_forward_f64": 2,
                        "set_kernel_buffer_2": 2,
                        "set_kernel_int_4": 2,
                        "enqueue_kinera_embedding_forward_f64": 2,
                    }[fail_on]
                    self.assertEqual(len(newly_released), expected_temp_releases)
                    self.assertEqual(len(set(newly_released)), len(newly_released))
                    self.assertEqual(len(api.allocations), 2)
                    expected_kernel_releases = {
                        "build_program": [],
                        "create_buffer": [],
                        "create_kinera_embedding_validate_ids_i32": [],
                        "set_kernel_buffer_0": [30],
                        "enqueue_kinera_embedding_validate_ids_i32": [30],
                        "set_kernel_int_3": [30],
                        "read_buffer": [30],
                        "create_kinera_embedding_forward_f64": [30],
                        "set_kernel_buffer_2": [30, 31],
                        "set_kernel_int_4": [30, 31],
                        "enqueue_kinera_embedding_forward_f64": [30, 31],
                    }[fail_on]
                    self.assertEqual(
                        api.released_kernels[len(before_release_kernels) :],
                        expected_kernel_releases,
                    )
                finally:
                    backend.close()

    def test_cleanup_release_errors_preserve_primary_error_and_do_not_double_release(self) -> None:
        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        weights = backend.buffer_from_host([[1.0]], shape=(1, 1))
        ids = backend.buffer_from_host([0], shape=(1,), dtype="int32")
        enqueue_error = RuntimeError("injected lookup enqueue failure")
        release_error = RuntimeError("injected temporary release failure")
        release_attempts: list[int] = []
        original_release_mem = api.release_mem
        original_release_kernel = api.release_kernel

        def fail_status_release(handle: int) -> None:
            if handle == 102:
                release_attempts.append(handle)
            if handle == 102 and len(release_attempts) == 1:
                raise release_error
            original_release_mem(handle)

        try:
            api.fail_on = "enqueue_kinera_embedding_validate_ids_i32"
            # The failed status release stays owned by the manager for close().
            with (
                patch.object(api, "enqueue_1d", side_effect=enqueue_error),
                patch.object(api, "release_mem", side_effect=fail_status_release),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    backend.embedding_forward(weights, ids)
                self.assertIs(raised.exception, enqueue_error)
                self.assertIs(raised.exception.__cause__, release_error)
                self.assertEqual(len(release_attempts), 1)
                self.assertEqual(len(api.allocations), 3)
                self.assertEqual(len(backend.memory._buffers), 3)
                backend.close()
            self.assertIs(raised.exception, enqueue_error)
            self.assertIs(raised.exception.__cause__, release_error)
            self.assertEqual(release_attempts, [102, 102])
            self.assertEqual(api.allocations, {})

            # A native kernel-release failure is attempted once; the output is
            # still released and the enqueue exception remains primary.
        finally:
            if not backend._closed:
                backend.close()

        api = FakeOpenCLApi()
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        weights = backend.buffer_from_host([[1.0]], shape=(1, 1))
        ids = backend.buffer_from_host([0], shape=(1,), dtype="int32")
        enqueue_count = 0
        release_attempts = []
        enqueue_error = RuntimeError("injected lookup enqueue failure")
        release_error = RuntimeError("injected kernel release failure")
        original_enqueue = api.enqueue_1d
        original_release_kernel = api.release_kernel

        def fail_lookup_enqueue(queue, kernel, global_size) -> None:
            nonlocal enqueue_count
            enqueue_count += 1
            if enqueue_count == 1:
                original_enqueue(queue, kernel, global_size)
                return
            raise enqueue_error

        def fail_lookup_kernel_release(handle: int) -> None:
            if handle == 31:
                release_attempts.append(handle)
                raise release_error
            original_release_kernel(handle)

        try:
            with (
                patch.object(api, "enqueue_1d", side_effect=fail_lookup_enqueue),
                patch.object(api, "release_kernel", side_effect=fail_lookup_kernel_release),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    backend.embedding_forward(weights, ids)
            self.assertIs(raised.exception, enqueue_error)
            self.assertIs(raised.exception.__cause__, release_error)
            self.assertEqual(release_attempts, [31])
            self.assertEqual(api.released_kernels, [30])
            self.assertEqual(set(api.allocations), {weights.handle, ids.handle})
        finally:
            backend.close()

    def test_failed_layer_forward_preserves_previous_state_and_releases_new_buffers(self) -> None:
        layer = Embedding(3, 3, seed=9, backend=self.backend)
        try:
            layer.forward([0, 2])
            previous = (layer.last_ids, layer.output, layer.grad_weights)
            previous_handles = {
                previous[0].handle,
                previous[1].handle,
                previous[2].handle,
            }
            before = set(self.api.allocations)
            forward_error = RuntimeError("injected forward failure")
            with patch.object(self.backend, "embedding_forward", side_effect=forward_error):
                with self.assertRaises(RuntimeError) as raised:
                    layer.forward([1])
            self.assertIs(raised.exception, forward_error)
            self.assertEqual((layer.last_ids, layer.output, layer.grad_weights), previous)
            self.assertTrue(previous_handles.issubset(self.api.allocations))
            self.assertEqual(set(self.api.allocations), before)
        finally:
            layer.close()

    def test_empty_backward_and_zero_rate_step_are_available(self) -> None:
        gradients = self.backend.buffer_zeros(shape=(3, 3), dtype="float64")
        try:
            self.assertIs(
                self.backend.embedding_backward(gradients, [], []), gradients
            )
            self.assertIs(
                self.backend.embedding_step(self.weights, gradients, 0.0),
                self.weights,
            )
        finally:
            gradients.close()


class OpenCLEmbeddingBackwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi(max_allocation=4096)
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))
        self.cpu = CPUBackend()
        self.grad_weights = self.backend.buffer_from_host(
            [[1e16, 2.0], [4.0, -3.0], [7.0, 8.0]],
            shape=(3, 2),
            dtype="float64",
        )

    def tearDown(self) -> None:
        self.backend.close()

    def _device_inputs(self, ids, grad_output):
        ids_buffer = self.backend.buffer_from_host(
            ids, shape=(len(ids),), dtype="int32"
        )
        grad_buffer = self.backend.buffer_from_host(
            grad_output, shape=(len(grad_output), 2), dtype="float64"
        ) if grad_output else None
        return ids_buffer, grad_buffer

    def _values(self, buffer):
        return self.backend.buffer_to_host(buffer)

    def test_ordered_repeated_ids_accumulate_and_calls_are_consecutive(self) -> None:
        ids_values = [0, 0, 0, 2]
        gradients = [[1.0, 1.0], [1.0, -2.0], [-1e16, 3.0], [5.0, -6.0]]
        ids, grad_output = self._device_inputs(ids_values, gradients)
        original = self._values(self.grad_weights)
        expected = self.cpu.embedding_backward(original, ids_values, gradients)
        writes_before = list(self.api.write_sizes)
        reads_before = list(self.api.read_sizes)
        releases_before = list(self.api.released)
        kernel_releases_before = list(self.api.released_kernels)
        try:
            result = self.backend.embedding_backward(
                self.grad_weights, ids, grad_output
            )
            self.assertIs(result, self.grad_weights)
            self.assertEqual(self.api.read_sizes[len(reads_before) :], [4])
            status_releases = self.api.released[len(releases_before) :]
            kernel_releases = self.api.released_kernels[len(kernel_releases_before) :]
            self.assertEqual(len(status_releases), 1)
            self.assertEqual(len(set(status_releases)), 1)
            self.assertEqual(len(kernel_releases), 2)
            self.assertEqual(len(set(kernel_releases)), 2)
            self.assertEqual(self._values(result), expected)
            self.assertEqual(self._values(result)[1], original[1])
            self.assertEqual(self._values(result)[0][0], 0.0)
            self.assertEqual(self.api.write_sizes, writes_before)
            self.assertEqual(self.api.created_kernels[-2:], [
                "kinera_embedding_validate_backward_inputs",
                "kinera_embedding_backward_f64",
            ])

            next_ids_values = [0, 2]
            next_gradients = [[0.5, -1.0], [-4.0, 2.0]]
            next_ids, next_grad_output = self._device_inputs(
                next_ids_values, next_gradients
            )
            releases_before = list(self.api.released)
            kernel_releases_before = list(self.api.released_kernels)
            try:
                expected = self.cpu.embedding_backward(
                    expected, next_ids_values, next_gradients
                )
                self.assertIs(
                    self.backend.embedding_backward(
                        result, next_ids, next_grad_output
                    ),
                    result,
                )
                self.assertEqual(self._values(result), expected)
                self.assertEqual(
                    len(self.api.released) - len(releases_before), 1
                )
                self.assertEqual(
                    len(self.api.released_kernels) - len(kernel_releases_before), 2
                )
            finally:
                next_grad_output.close()
                next_ids.close()
        finally:
            grad_output.close()
            ids.close()

    def test_empty_ids_are_noop_without_allocations_or_kernels(self) -> None:
        before = self._values(self.grad_weights)
        allocations = set(self.api.allocations)
        kernels = list(self.api.created_kernels)
        enqueues = list(self.api.enqueues)
        result = self.backend.embedding_backward(self.grad_weights, [], [])
        self.assertIs(result, self.grad_weights)
        self.assertEqual(self._values(result), before)
        self.assertEqual(set(self.api.allocations), allocations)
        self.assertEqual(self.api.created_kernels, kernels)
        self.assertEqual(self.api.enqueues, enqueues)

    def test_resident_invalid_ids_and_nonfinite_gradients_fail_before_mutation(self) -> None:
        cases = (
            ([-1], [[1.0, 2.0]], "outside"),
            ([3], [[1.0, 2.0]], "outside"),
            ([0], [[float("nan"), 2.0]], "non-finite"),
            ([0], [[float("inf"), 2.0]], "non-finite"),
            ([0], [[float("-inf"), 2.0]], "non-finite"),
        )
        original = self._values(self.grad_weights)
        for ids_values, gradients, message in cases:
            with self.subTest(ids=ids_values, gradients=gradients):
                ids, grad_output = self._device_inputs(
                    ids_values,
                    [[0.0, 0.0] for _ in ids_values],
                )
                # Inject non-finite values after a valid upload to exercise
                # device-resident validation without weakening production checks.
                if not math.isfinite(gradients[0][0]):
                    raw = struct.pack("<d", gradients[0][0]) + struct.pack(
                        "<d", gradients[0][1]
                    )
                    self.api.allocations[grad_output.handle][:] = raw
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        read_start = len(self.api.read_sizes)
                        self.backend.embedding_backward(
                            self.grad_weights, ids, grad_output
                        )
                    self.assertEqual(self.api.read_sizes[read_start:], [4])
                    self.assertEqual(self._values(self.grad_weights), original)
                    self.assertEqual(
                        self.api.created_kernels[-1],
                        "kinera_embedding_validate_backward_inputs",
                    )
                    self.assertEqual(
                        self.api.enqueues[-1], 1,
                        "only the scalar preflight may execute on invalid input",
                    )
                finally:
                    grad_output.close()
                    ids.close()

    def test_contract_rejects_wrong_shapes_dtypes_closed_foreign_and_alias_buffers(self) -> None:
        ids, grad_output = self._device_inputs([0, 1], [[1.0, 2.0], [3.0, 4.0]])
        wrong_ids_shape = self.backend.buffer_from_host(
            [[0, 1]], shape=(1, 2), dtype="int32"
        )
        wrong_ids_dtype = self.backend.buffer_from_host(
            [0, 1], shape=(2,), dtype="float64"
        )
        wrong_grad_shape = self.backend.buffer_from_host(
            [[1.0], [2.0]], shape=(2, 1), dtype="float64"
        )
        wrong_grad_dtype = self.backend.buffer_from_host(
            [[1, 2], [3, 4]], shape=(2, 2), dtype="int32"
        )
        closed = self.backend.buffer_from_host([0], shape=(1,), dtype="int32")
        closed.close()
        closed_gradients = self.backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2), dtype="float64"
        )
        closed_gradients.close()
        closed_weights = self.backend.buffer_from_host(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            shape=(3, 2),
            dtype="float64",
        )
        closed_weights.close()
        foreign_backend = OpenCLBackend(runtime=OpenCLRuntime(api=FakeOpenCLApi()))
        foreign_ids = foreign_backend.buffer_from_host([0, 1], shape=(2,), dtype="int32")
        foreign_gradients = foreign_backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2), dtype="float64"
        )
        aliasable = self.backend.buffer_from_host(
            [[0.0, 0.0], [0.0, 0.0]], shape=(2, 2), dtype="float64"
        )
        try:
            with self.assertRaisesRegex(ValueError, "rank 1"):
                self.backend.embedding_backward(self.grad_weights, wrong_ids_shape, grad_output)
            with self.assertRaisesRegex(ValueError, "dtype int32"):
                self.backend.embedding_backward(self.grad_weights, wrong_ids_dtype, grad_output)
            with self.assertRaisesRegex(ValueError, "shape"):
                self.backend.embedding_backward(self.grad_weights, ids, wrong_grad_shape)
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                self.backend.embedding_backward(self.grad_weights, ids, wrong_grad_dtype)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_backward(self.grad_weights, closed, grad_output)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_backward(
                    self.grad_weights, ids, closed_gradients
                )
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_backward(closed_weights, ids, grad_output)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_backward(self.grad_weights, foreign_ids, grad_output)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_backward(self.grad_weights, ids, foreign_gradients)
            with self.assertRaisesRegex(BackendError, "must not alias"):
                self.backend.embedding_backward(aliasable, ids, aliasable)
            self.assertEqual(self.api.created_kernels, [])
        finally:
            closed_weights.close()
            closed_gradients.close()
            aliasable.close()
            foreign_gradients.close()
            foreign_ids.close()
            foreign_backend.close()
            closed.close()
            wrong_grad_dtype.close()
            wrong_grad_shape.close()
            wrong_ids_dtype.close()
            wrong_ids_shape.close()
            grad_output.close()
            ids.close()

    def test_empty_ids_reject_nonempty_gradients_and_validate_weight_buffer(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty grad_output"):
            self.backend.embedding_backward(self.grad_weights, [], [[1.0]])
        with self.assertRaisesRegex(TypeError, "grad_weights as an OpenCLBuffer"):
            self.backend.embedding_backward([], [], [])

    def test_device_selection_and_declared_buffer_sizes_are_validated(self) -> None:
        ids, gradients = self._device_inputs([0], [[1.0, 2.0]])
        other_device_backend = OpenCLBackend(
            runtime=self.backend.runtime,
            memory_manager=self.backend.memory,
            device_id=1,
        )
        try:
            with self.assertRaisesRegex(BackendError, "selected OpenCL GPU"):
                other_device_backend.embedding_backward(
                    self.grad_weights, ids, gradients
                )

            original_size = ids._nbytes
            ids._nbytes += 4
            try:
                with self.assertRaisesRegex(ValueError, "IDs buffer size"):
                    self.backend.embedding_backward(self.grad_weights, ids, gradients)
            finally:
                ids._nbytes = original_size

            original_size = gradients._nbytes
            gradients._nbytes += 8
            try:
                with self.assertRaisesRegex(ValueError, "grad_output buffer size"):
                    self.backend.embedding_backward(self.grad_weights, ids, gradients)
            finally:
                gradients._nbytes = original_size

            original_size = self.grad_weights._nbytes
            self.grad_weights._nbytes += 8
            try:
                with self.assertRaisesRegex(ValueError, "grad_weights buffer size"):
                    self.backend.embedding_backward(self.grad_weights, ids, gradients)
            finally:
                self.grad_weights._nbytes = original_size
            self.assertEqual(self.api.created_kernels, [])
        finally:
            gradients.close()
            ids.close()

    def test_validation_failures_do_not_mutate_and_release_temporary_once(self) -> None:
        failures = (
            ("build_program", "build failure"),
            ("create_buffer", "allocation failure"),
            ("create_kinera_embedding_validate_backward_inputs", "creation failure"),
            ("set_kernel_buffer_0", "argument failure"),
            ("set_kernel_int_5", "scalar argument failure"),
            ("enqueue_kinera_embedding_validate_backward_inputs", "enqueue failure"),
            ("read_buffer", "status read failure"),
            ("release_kinera_embedding_validate_backward_inputs", "release failure"),
            ("create_kinera_embedding_backward_f64", "creation failure"),
            ("set_kernel_buffer_2_kinera_embedding_backward_f64", "argument failure"),
            ("set_kernel_int_5_kinera_embedding_backward_f64", "scalar argument failure"),
            ("enqueue_kinera_embedding_backward_f64", "enqueue failure"),
        )
        original = self._values(self.grad_weights)
        for fail_on, message in failures:
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi(max_allocation=4096)
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                weights = backend.buffer_from_host(
                    original, shape=(3, 2), dtype="float64"
                )
                ids = backend.buffer_from_host([0, 0], shape=(2,), dtype="int32")
                gradients = backend.buffer_from_host(
                    [[2.0, 3.0], [-1.0, 4.0]],
                    shape=(2, 2),
                    dtype="float64",
                )
                api.fail_on = fail_on
                reads_before = list(api.read_sizes)
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        backend.embedding_backward(weights, ids, gradients)
                    expected_status_reads = {
                        "build_program": [],
                        "create_buffer": [],
                        "create_kinera_embedding_validate_backward_inputs": [],
                        "set_kernel_buffer_0": [],
                        "set_kernel_int_5": [],
                        "enqueue_kinera_embedding_validate_backward_inputs": [],
                        "read_buffer": [],
                        "release_kinera_embedding_validate_backward_inputs": [4],
                        "create_kinera_embedding_backward_f64": [4],
                        "set_kernel_buffer_2_kinera_embedding_backward_f64": [4],
                        "set_kernel_int_5_kinera_embedding_backward_f64": [4],
                        "enqueue_kinera_embedding_backward_f64": [4],
                    }[fail_on]
                    self.assertEqual(api.read_sizes[len(reads_before):], expected_status_reads)
                    api.fail_on = None
                    self.assertEqual(backend.buffer_to_host(weights), original)
                    self.assertEqual(set(api.allocations), {weights.handle, ids.handle, gradients.handle})
                    status_releases = [handle for handle in api.released if handle not in {weights.handle, ids.handle, gradients.handle}]
                    self.assertEqual(len(status_releases), len(set(status_releases)))
                    if fail_on == "release_kinera_embedding_validate_backward_inputs":
                        self.assertEqual(api.released_kernels, [30])
                    elif fail_on == "enqueue_kinera_embedding_backward_f64":
                        self.assertEqual(api.released_kernels, [30, 31])
                finally:
                    backend.close()

    def test_primary_validation_error_survives_cleanup_error(self) -> None:
        ids, gradients = self._device_inputs([0], [[1.0, 2.0]])
        original_enqueue = self.api.enqueue_1d
        original_release = self.api.release_mem
        enqueue_error = RuntimeError("primary enqueue failure")
        release_error = RuntimeError("secondary status release failure")
        attempts = []

        def fail_enqueue(*args, **kwargs):
            raise enqueue_error

        def fail_status_release(handle):
            if handle == 103:
                attempts.append(handle)
                raise release_error
            original_release(handle)

        try:
            with (
                patch.object(self.api, "enqueue_1d", side_effect=fail_enqueue),
                patch.object(self.api, "release_mem", side_effect=fail_status_release),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self.backend.embedding_backward(self.grad_weights, ids, gradients)
            self.assertIs(raised.exception, enqueue_error)
            self.assertIs(raised.exception.__cause__, release_error)
            self.assertEqual(attempts, [103])
            self.assertEqual(self.api.released_kernels, [30])
        finally:
            gradients.close()
            ids.close()

    def test_layer_accepts_only_owned_opencl_resident_pair_and_closes_host_upload(self) -> None:
        layer = Embedding(3, 2, seed=17, backend=self.backend)
        host_grad_output = [[1.0, 2.0], [3.0, -4.0], [-5.0, 6.0]]
        try:
            layer.forward([2, 0, 2])
            current_gradients = layer.grad_weights
            writes_before = len(self.api.write_sizes)
            allocations_before = set(self.api.allocations)
            layer.backward(host_grad_output)
            self.assertIs(layer.grad_weights, current_gradients)
            self.assertTrue(layer._has_backward)
            self.assertEqual(len(self.api.write_sizes), writes_before + 1)
            self.assertEqual(
                self.backend.buffer_to_host(layer.grad_weights),
                self.cpu.embedding_backward(
                    [[0.0, 0.0] for _ in range(3)],
                    [2, 0, 2],
                    host_grad_output,
                ),
            )
            self.assertEqual(
                set(self.api.allocations), allocations_before,
                "temporary host grad_output must be closed after backward",
            )

            resident = self.backend.buffer_from_host(
                host_grad_output, shape=(3, 2), dtype="float64"
            )
            writes_before = len(self.api.write_sizes)
            reads_before = len(self.api.read_sizes)
            try:
                layer.backward(resident)
                self.assertFalse(resident.closed, "caller retains resident input ownership")
                self.assertEqual(len(self.api.write_sizes), writes_before)
                self.assertEqual(self.api.read_sizes[reads_before:], [4])
            finally:
                resident.close()
        finally:
            layer.close()

    def test_layer_empty_forward_backward_and_failed_preflight_state(self) -> None:
        layer = Embedding(3, 2, seed=19, backend=self.backend)
        try:
            layer.forward([])
            gradients = layer.grad_weights
            kernels = list(self.api.created_kernels)
            enqueues = list(self.api.enqueues)
            layer.backward([])
            self.assertIs(layer.grad_weights, gradients)
            self.assertTrue(layer._has_backward)
            self.assertEqual(self.api.created_kernels, kernels)
            self.assertEqual(self.api.enqueues, enqueues)

            layer.forward([1])
            ids = layer.last_ids
            grad_weights = layer.grad_weights
            resident = self.backend.buffer_from_host(
                [[1.0, 2.0]], shape=(1, 2), dtype="float64"
            )
            # Simulate corruption in an already resident ID buffer, bypassing
            # only the test upload path; production preflight remains unchanged.
            self.api.allocations[ids.handle][:4] = struct.pack("<i", 3)
            before = self.backend.buffer_to_host(grad_weights)
            try:
                with self.assertRaisesRegex(ValueError, "outside"):
                    layer.backward(resident)
                self.assertIs(layer.grad_weights, grad_weights)
                self.assertFalse(layer._has_backward)
                self.assertFalse(ids.closed)
                self.assertEqual(self.backend.buffer_to_host(grad_weights), before)
            finally:
                resident.close()
        finally:
            layer.close()

    def test_layer_does_not_enable_arbitrary_opencl_named_backend(self) -> None:
        layer = Embedding(3, 2, seed=23, backend=self.backend)
        resident_gradients = None
        concrete_backend = layer.backend
        try:
            layer.forward([0])
            resident_gradients = self.backend.buffer_from_host(
                [[1.0, 2.0]], shape=(1, 2), dtype="float64"
            )
            layer.backend = SimpleNamespace(
                name="opencl", memory=self.backend.memory
            )
            with self.assertRaisesRegex(TypeError, "device-backed Embedding forward"):
                layer.backward(resident_gradients)
            self.assertFalse(resident_gradients.closed)
            self.assertFalse(layer._has_backward)
        finally:
            layer.backend = concrete_backend
            if resident_gradients is not None:
                resident_gradients.close()
            layer.close()


class OpenCLEmbeddingStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi(max_allocation=4096)
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))
        self.cpu = CPUBackend()
        self.weights_values = [[1.0, 0.5, -2.0], [-0.25, 2.0, 1e16]]
        self.gradients_values = [
            [1.0 - 2.0**-27, -0.5, 0.25],
            [0.75, -1e16, -0.25],
        ]
        self.weights = self.backend.buffer_from_host(
            self.weights_values, shape=(2, 3), dtype="float64"
        )
        self.grad_weights = self.backend.buffer_from_host(
            self.gradients_values, shape=(2, 3), dtype="float64"
        )

    def tearDown(self) -> None:
        self.backend.close()

    def _values(self, buffer):
        return self.backend.buffer_to_host(buffer)

    def test_step_matches_cpu_without_fma_and_keeps_buffer_resident(self) -> None:
        learning_rate = 1.0 + 2.0**-27
        expected = self.cpu.embedding_step(
            self.weights_values, self.gradients_values, learning_rate
        )
        writes_before = list(self.api.write_sizes)
        reads_before = list(self.api.read_sizes)
        before_releases = list(self.api.released_kernels)

        result = self.backend.embedding_step(
            self.weights, self.grad_weights, learning_rate
        )

        self.assertIs(result, self.weights)
        self.assertEqual(self.api.write_sizes, writes_before)
        self.assertEqual(self.api.read_sizes, reads_before)
        self.assertEqual(
            self.api.released_kernels[len(before_releases) :], [30]
        )
        self.assertEqual(self._values(result), expected)
        self.assertEqual(expected[0][0], 0.0)
        # Decimal.from_float preserves the exact binary64 operands.  A single
        # final conversion therefore acts as an independent fused-operation
        # oracle on Python versions without math.fma.
        with localcontext() as context:
            context.prec = 200
            fused = float(
                Decimal.from_float(1.0)
                - Decimal.from_float(learning_rate)
                * Decimal.from_float(self.gradients_values[0][0])
            )
        separately_rounded = 1.0 - (
            learning_rate * self.gradients_values[0][0]
        )
        self.assertEqual(fused, 2.0**-54)
        self.assertEqual(separately_rounded, expected[0][0])
        self.assertNotEqual(fused, separately_rounded)
        self.assertFalse(self.grad_weights.closed)

    def test_consecutive_updates_and_zero_rate_have_cpu_semantics(self) -> None:
        expected = self.weights_values
        for rate in (0.1, 0.025):
            expected = self.cpu.embedding_step(
                expected, self.gradients_values, rate
            )
            returned = self.backend.embedding_step(
                self.weights, self.grad_weights, rate
            )
            self.assertIs(returned, self.weights)
            self.assertEqual(self._values(self.weights), expected)

        actual_before_zero = self._values(self.weights)
        expected_zero = self.cpu.embedding_step(
            expected, self.gradients_values, 0.0
        )
        kernels_before = list(self.api.created_kernels)
        enqueues_before = list(self.api.enqueues)
        reads_before = list(self.api.read_sizes)
        writes_before = list(self.api.write_sizes)
        returned = self.backend.embedding_step(self.weights, self.grad_weights, 0.0)
        self.assertIs(returned, self.weights)
        self.assertEqual(self.api.created_kernels, kernels_before)
        self.assertEqual(self.api.enqueues, enqueues_before)
        self.assertEqual(self.api.read_sizes, reads_before)
        self.assertEqual(self.api.write_sizes, writes_before)
        self.assertEqual(self._values(self.weights), actual_before_zero)
        self.assertEqual(self._values(self.weights), expected_zero)

    def test_learning_rate_validation_happens_before_dispatch(self) -> None:
        invalid_rates = (
            (True, TypeError),
            ("0.1", TypeError),
            (-0.1, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
            (10**10000, ValueError),
        )
        original = self._values(self.weights)
        for index, (rate, error_type) in enumerate(invalid_rates):
            with self.subTest(case=index, rate_type=type(rate).__name__):
                with self.assertRaises(error_type):
                    self.backend.embedding_step(self.weights, self.grad_weights, rate)
                self.assertEqual(self.api.created_kernels, [])
                self.assertEqual(self.api.enqueues, [])
                self.assertEqual(self._values(self.weights), original)

    def test_contract_rejects_shapes_types_closed_foreign_alias_and_layout(self) -> None:
        wrong_rank = self.backend.buffer_from_host(
            [1.0, 2.0], shape=(2,), dtype="float64"
        )
        wrong_shape = self.backend.buffer_from_host(
            [[1.0, 2.0]], shape=(1, 2), dtype="float64"
        )
        wrong_dtype = self.backend.buffer_from_host(
            [[1, 2, 3], [4, 5, 6]], shape=(2, 3), dtype="int32"
        )
        closed_weights = self.backend.buffer_from_host(
            self.weights_values, shape=(2, 3), dtype="float64"
        )
        closed_weights.close()
        closed_gradients = self.backend.buffer_from_host(
            self.gradients_values, shape=(2, 3), dtype="float64"
        )
        closed_gradients.close()
        foreign_backend = OpenCLBackend(runtime=OpenCLRuntime(api=FakeOpenCLApi()))
        foreign_gradients = foreign_backend.buffer_from_host(
            self.gradients_values, shape=(2, 3), dtype="float64"
        )
        alias = self.backend.buffer_from_host(
            self.weights_values, shape=(2, 3), dtype="float64"
        )
        try:
            with self.assertRaisesRegex(ValueError, "rank 2"):
                self.backend.embedding_step(wrong_rank, self.grad_weights, 0.1)
            with self.assertRaisesRegex(ValueError, "match weights shape"):
                self.backend.embedding_step(self.weights, wrong_shape, 0.1)
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                self.backend.embedding_step(self.weights, wrong_dtype, 0.1)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_step(closed_weights, self.grad_weights, 0.1)
            with self.assertRaisesRegex(BackendError, "released/closed"):
                self.backend.embedding_step(self.weights, closed_gradients, 0.1)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.embedding_step(self.weights, foreign_gradients, 0.1)
            with self.assertRaisesRegex(BackendError, "must not alias"):
                self.backend.embedding_step(alias, alias, 0.1)
            with patch.object(
                OpenCLBuffer, "layout", new_callable=PropertyMock, return_value="F"
            ):
                with self.assertRaisesRegex(ValueError, "row-major layout"):
                    self.backend.embedding_step(self.weights, self.grad_weights, 0.1)
            self.assertEqual(self.api.created_kernels, [])
        finally:
            alias.close()
            foreign_gradients.close()
            foreign_backend.close()
            closed_gradients.close()
            closed_weights.close()
            wrong_dtype.close()
            wrong_shape.close()
            wrong_rank.close()

    def test_device_and_declared_buffer_sizes_are_checked_before_launch(self) -> None:
        different_device = OpenCLBackend(
            runtime=self.backend.runtime,
            memory_manager=self.backend.memory,
            device_id=1,
        )
        with self.assertRaisesRegex(BackendError, "selected OpenCL GPU"):
            different_device.embedding_step(self.weights, self.grad_weights, 0.1)

        original_size = self.grad_weights._nbytes
        self.grad_weights._nbytes += 8
        try:
            with self.assertRaisesRegex(ValueError, "grad_weights buffer size"):
                self.backend.embedding_step(self.weights, self.grad_weights, 0.1)
        finally:
            self.grad_weights._nbytes = original_size
        self.assertEqual(self.api.created_kernels, [])

    def test_fp64_device_query_failure_preserves_weights_without_launch(self) -> None:
        original = self._values(self.weights)
        with patch.object(
            self.backend.runtime,
            "require_device",
            side_effect=BackendError("injected FP64 capability failure"),
        ):
            with self.assertRaisesRegex(BackendError, "FP64 capability failure"):
                self.backend.embedding_step(self.weights, self.grad_weights, 0.1)

        self.assertEqual(self.api.created_kernels, [])
        self.assertEqual(self.api.enqueues, [])
        self.assertEqual(self._values(self.weights), original)

    def test_pre_enqueue_failures_preserve_weights_and_release_created_kernel(self) -> None:
        failures = (
            ("build_program", "build failure", []),
            ("create_kinera_embedding_step_f64", "creation failure", []),
            ("set_kernel_buffer_0", "argument failure", [30]),
            ("set_kernel_buffer_1", "argument failure", [30]),
            ("set_kernel_int_2", "scalar argument failure", [30]),
            ("set_kernel_int_3", "scalar argument failure", [30]),
            ("set_kernel_double_4", "double argument failure", [30]),
            ("enqueue_kinera_embedding_step_f64", "enqueue failure", [30]),
        )
        original = self._values(self.weights)
        for fail_on, message, expected_releases in failures:
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi(max_allocation=4096, fail_on=fail_on)
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                weights = backend.buffer_from_host(
                    self.weights_values, shape=(2, 3), dtype="float64"
                )
                gradients = backend.buffer_from_host(
                    self.gradients_values, shape=(2, 3), dtype="float64"
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        backend.embedding_step(weights, gradients, 0.1)
                    self.assertEqual(backend.buffer_to_host(weights), original)
                    self.assertEqual(api.released_kernels, expected_releases)
                    self.assertEqual(api.read_sizes, [48])
                    self.assertEqual(api.write_sizes, [48, 48])
                    self.assertEqual(
                        set(api.allocations), {weights.handle, gradients.handle}
                    )
                finally:
                    backend.close()

    def test_release_failure_after_enqueue_is_reported_without_retry(self) -> None:
        api = FakeOpenCLApi(max_allocation=4096)
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        weights = backend.buffer_from_host(self.weights_values, shape=(2, 3))
        gradients = backend.buffer_from_host(self.gradients_values, shape=(2, 3))
        expected = self.cpu.embedding_step(
            self.weights_values, self.gradients_values, 0.1
        )
        api.fail_on = "release_kinera_embedding_step_f64"
        try:
            with self.assertRaisesRegex(RuntimeError, "kernel release failure"):
                backend.embedding_step(weights, gradients, 0.1)
            self.assertEqual(backend.buffer_to_host(weights), expected)
            self.assertEqual(api.released_kernels, [30])
        finally:
            backend.close()

    def test_primary_enqueue_error_survives_kernel_release_failure(self) -> None:
        original = self._values(self.weights)
        primary = RuntimeError("primary enqueue error")
        cleanup = RuntimeError("kernel cleanup error")
        releases = []

        def fail_enqueue(*args, **kwargs):
            raise primary

        def fail_release(handle):
            releases.append(handle)
            raise cleanup

        with (
            patch.object(self.api, "enqueue_1d", side_effect=fail_enqueue),
            patch.object(self.api, "release_kernel", side_effect=fail_release),
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.backend.embedding_step(self.weights, self.grad_weights, 0.1)
        self.assertIs(raised.exception, primary)
        self.assertIs(raised.exception.__cause__, cleanup)
        self.assertEqual(releases, [30])
        self.assertEqual(self._values(self.weights), original)

    def test_layer_step_delegates_in_place_and_clears_backward_flag(self) -> None:
        layer = Embedding(2, 3, seed=29, backend=self.backend)
        cpu_layer = Embedding(2, 3, seed=29, backend=self.cpu)
        new_weights = self.backend.buffer_from_host(
            self.weights_values, shape=(2, 3), dtype="float64"
        )
        new_gradients = self.backend.buffer_from_host(
            self.gradients_values, shape=(2, 3), dtype="float64"
        )
        original_weight_buffer = layer.weights
        original_gradient_buffer = layer.grad_weights
        try:
            layer.weights = new_weights
            layer.grad_weights = new_gradients
            original_weight_buffer.close()
            original_gradient_buffer.close()
            layer._has_backward = True

            cpu_layer.weights = [row.copy() for row in self.weights_values]
            cpu_layer.grad_weights = [row.copy() for row in self.gradients_values]
            cpu_layer._has_backward = True

            weights_buffer = layer.weights
            expected = self.cpu.embedding_step(
                self.weights_values, self.gradients_values, 0.125
            )
            layer.step(0.125)
            cpu_layer.step(0.125)
            self.assertIs(layer.weights, weights_buffer)
            self.assertFalse(layer._has_backward)
            self.assertEqual(self._values(layer.weights), expected)
            self.assertEqual(cpu_layer.weights, expected)
        finally:
            layer.close()
            cpu_layer.close()


class OpenCLEmbeddingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi(max_allocation=4096)
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))
        self.cpu = CPUBackend()

    def tearDown(self) -> None:
        self.backend.close()

    def _assert_values(self, actual, expected) -> float:
        if actual and isinstance(actual[0], list):
            actual_flat = [value for row in actual for value in row]
        else:
            actual_flat = list(actual)
        if expected and isinstance(expected[0], list):
            expected_flat = [value for row in expected for value in row]
        else:
            expected_flat = list(expected)
        self.assertEqual(len(actual_flat), len(expected_flat))
        errors = [abs(got - want) for got, want in zip(actual_flat, expected_flat)]
        for index, (got, want) in enumerate(zip(actual_flat, expected_flat)):
            with self.subTest(index=index, actual=got, expected=want):
                self.assertLessEqual(abs(got - want), 1e-12 + 1e-12 * abs(want))
        return max(errors, default=0.0)

    def test_repeated_forward_backward_step_cycles_and_empty_ids_match_cpu(self) -> None:
        cpu_layer = Embedding(4, 3, seed=20261106, backend=self.cpu)
        gpu_layer = Embedding(4, 3, seed=20261106, backend=self.backend)
        owned_buffers: list[OpenCLBuffer] = []
        weight_buffer = gpu_layer.weights
        initial_gradients = gpu_layer.grad_weights
        self.assertIsInstance(weight_buffer, OpenCLBuffer)
        owned_buffers.append(weight_buffer)
        owned_buffers.append(initial_gradients)
        self.assertEqual(self.backend.buffer_to_host(weight_buffer), cpu_layer.weights)
        with self.assertRaisesRegex(RuntimeError, "backward anterior"):
            gpu_layer.step(0.1)
        with self.assertRaisesRegex(RuntimeError, "backward anterior"):
            cpu_layer.step(0.1)

        cases = (
            (
                [2, 0, 2, 1],
                [
                    [1e16, 1.0, -1.0],
                    [0.5, -0.5, 0.0],
                    [-1e16, 2.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 2.0],
                    [-1.0, 2.0, 0.0],
                    [1.0, -1.0, -1.0],
                    [0.5, -0.5, 1.0],
                ],
                0.0,
            ),
            (
                [3, 3, 1],
                [[-2.0, 0.0, 1.5], [2.0, -0.0, -1.5], [0.25, -0.5, 1.0]],
                [[1.0, -2.0, 0.0], [-0.5, 1.0, 2.0], [0.0, 0.75, -1.25]],
                0.125,
            ),
            (
                [0, 2, 1, 0, 2],
                [[1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, 0.0], [0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]],
                [[-0.25, 0.5, 1.0], [0.0, 1.5, -2.0], [3.0, -4.0, 5.0], [0.75, -0.5, 0.0], [1.0, -1.0, 2.0]],
                0.03125,
            ),
        )
        maximum_errors = {
            "forward": 0.0,
            "backward_1": 0.0,
            "backward_2": 0.0,
            "step": 0.0,
        }
        previous_state = None
        try:
            for cycle, (ids, grad_one, grad_two, rate) in enumerate(cases):
                if previous_state is not None:
                    self.assertIs(gpu_layer.weights, weight_buffer)
                    self._assert_values(
                        self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
                    )

                cpu_output = cpu_layer.forward(ids.copy())
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_output = gpu_layer.forward(ids.copy())
                    materialize.assert_not_called()
                if previous_state is not None:
                    old_ids, old_output, old_gradients = previous_state
                    self.assertTrue(old_ids.closed)
                    self.assertTrue(old_output.closed)
                    self.assertTrue(old_gradients.closed)
                elif cycle == 0:
                    self.assertTrue(initial_gradients.closed)
                self.assertIsInstance(gpu_output, OpenCLBuffer)
                self.assertIs(gpu_layer.output, gpu_output)
                owned_buffers.extend(
                    (gpu_layer.last_ids, gpu_output, gpu_layer.grad_weights)
                )
                maximum_errors["forward"] = max(
                    maximum_errors["forward"],
                    self._assert_values(
                        self.backend.buffer_to_host(gpu_output), cpu_output
                    ),
                )
                self.assertEqual(
                    self.backend.buffer_to_host(gpu_layer.grad_weights),
                    [[0.0, 0.0, 0.0] for _ in range(4)],
                )

                cpu_layer.backward(grad_one)
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_layer.backward(grad_one)
                    materialize.assert_not_called()
                self.assertTrue(gpu_layer._has_backward)
                maximum_errors["backward_1"] = max(
                    maximum_errors["backward_1"],
                    self._assert_values(
                        self.backend.buffer_to_host(gpu_layer.grad_weights),
                        cpu_layer.grad_weights,
                    ),
                )

                resident_gradients = self.backend.buffer_from_host(
                    grad_two,
                    shape=(len(ids), 3),
                    dtype="float64",
                )
                owned_buffers.append(resident_gradients)
                try:
                    cpu_layer.backward(grad_two)
                    with patch.object(
                        self.backend,
                        "buffer_to_host",
                        wraps=self.backend.buffer_to_host,
                    ) as materialize:
                        gpu_layer.backward(resident_gradients)
                        materialize.assert_not_called()
                    self.assertFalse(resident_gradients.closed)
                    maximum_errors["backward_2"] = max(
                        maximum_errors["backward_2"],
                        self._assert_values(
                            self.backend.buffer_to_host(gpu_layer.grad_weights),
                            cpu_layer.grad_weights,
                        ),
                    )
                finally:
                    resident_gradients.close()

                cpu_layer.step(rate)
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_layer.step(rate)
                    materialize.assert_not_called()
                self.assertFalse(gpu_layer._has_backward)
                self.assertIs(gpu_layer.weights, weight_buffer)
                maximum_errors["step"] = max(
                    maximum_errors["step"],
                    self._assert_values(
                        self.backend.buffer_to_host(gpu_layer.weights),
                        cpu_layer.weights,
                    ),
                )
                previous_state = (
                    gpu_layer.last_ids,
                    gpu_layer.output,
                    gpu_layer.grad_weights,
                )

            kernels_before_empty = len(self.api.created_kernels)
            cpu_layer.forward([])
            self.assertEqual(gpu_layer.forward([]), [])
            self.assertEqual(gpu_layer.last_ids, [])
            self.assertEqual(gpu_layer.output, [])
            owned_buffers.append(gpu_layer.grad_weights)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                [[0.0, 0.0, 0.0] for _ in range(4)],
            )
            cpu_layer.backward([])
            gpu_layer.backward([])
            cpu_layer.step(0.0)
            gpu_layer.step(0.0)
            self.assertEqual(len(self.api.created_kernels), kernels_before_empty)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
            )
            self.assertFalse(gpu_layer._has_backward)
            print("OPENCL_F1D_FAKE_MAX_ABS_ERRORS", maximum_errors)
        finally:
            gpu_layer.close()
            gpu_layer.close()
            cpu_layer.close()

        self.assertTrue(all(buffer.closed for buffer in owned_buffers))
        self.assertEqual(self.api.allocations, {})
        self.assertFalse(any(not buffer.closed for buffer in self.backend.memory._buffers))

    def test_release_output_buffer_does_not_invalidate_backward_and_discard_clears_state(self) -> None:
        cpu_layer = Embedding(3, 2, seed=92, backend=self.cpu)
        gpu_layer = Embedding(3, 2, seed=92, backend=self.backend)
        weights = gpu_layer.weights
        try:
            cpu_layer.forward([2, 1, 2])
            output = gpu_layer.forward([2, 1, 2])
            ids = gpu_layer.last_ids
            grad_weights = gpu_layer.grad_weights
            output_handle = output.handle
            ids_handle = ids.handle
            grad_handle = grad_weights.handle
            gpu_layer.release_output_buffer()
            cpu_layer.release_output_buffer()
            self.assertTrue(output.closed)
            self.assertIs(gpu_layer.output, output)
            self.assertFalse(ids.closed)

            gradient_values = [[1.0, -2.0], [0.5, 4.0], [-1.5, 3.0]]
            cpu_layer.backward(gradient_values)
            gpu_layer.backward(gradient_values)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                cpu_layer.grad_weights,
                "backward uses IDs and gradients, not the released lookup output",
            )
            cpu_layer.step(0.25)
            gpu_layer.step(0.25)
            self.assertIs(gpu_layer.weights, weights)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
            )

            cpu_layer.forward([0])
            gpu_layer.forward([0])
            self.assertTrue(ids.closed)
            self.assertTrue(output.closed)
            self.assertTrue(grad_weights.closed)
            new_ids = gpu_layer.last_ids
            new_output = gpu_layer.output
            new_gradients = gpu_layer.grad_weights
            new_ids_handle, new_output_handle, new_grad_handle = (
                new_ids.handle,
                new_output.handle,
                new_gradients.handle,
            )
            cpu_layer.backward([[2.0, -3.0]])
            gpu_layer.backward([[2.0, -3.0]])
            gpu_layer.discard_forward_state()
            cpu_layer.discard_forward_state()
            gpu_layer.discard_forward_state()
            cpu_layer.discard_forward_state()
            self.assertIsNone(gpu_layer.last_ids)
            self.assertIsNone(gpu_layer.output)
            self.assertTrue(new_ids.closed)
            self.assertTrue(new_output.closed)
            self.assertTrue(gpu_layer.grad_weights is new_gradients)
            self.assertFalse(new_gradients.closed)
            self.assertFalse(gpu_layer._has_backward)
            with self.assertRaisesRegex(RuntimeError, "forward anterior"):
                gpu_layer.backward([])
            with self.assertRaisesRegex(RuntimeError, "backward anterior"):
                gpu_layer.step(0.1)
        finally:
            gpu_layer.close()
            gpu_layer.close()
            cpu_layer.close()

        self.assertTrue(weights.closed)
        self.assertIsNotNone(new_gradients)
        self.assertTrue(new_gradients.closed)
        for handle in (
            output_handle,
            ids_handle,
            grad_handle,
            new_ids_handle,
            new_output_handle,
            new_grad_handle,
        ):
            self.assertEqual(self.api.released.count(handle), 1)
        self.assertEqual(self.api.allocations, {})

    def test_pre_execution_failures_preserve_layer_state_and_post_submit_errors_are_ambiguous(self) -> None:
        cases = (
            ("create_buffer", False),
            ("build_program", False),
            ("create_kinera_embedding_validate_backward_inputs", False),
            ("enqueue_kinera_embedding_validate_backward_inputs", False),
            ("enqueue_kinera_embedding_backward_f64", False),
            ("release_kinera_embedding_validate_backward_inputs", False),
            ("release_kinera_embedding_backward_f64", True),
        )
        for fail_on, mutated in cases:
            with self.subTest(fail_on=fail_on):
                api = FakeOpenCLApi(max_allocation=4096)
                backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
                layer = Embedding(3, 2, seed=17, backend=backend)
                try:
                    layer.forward([2, 0, 2])
                    ids = layer.last_ids
                    output = layer.output
                    grad_weights = layer.grad_weights
                    original = backend.buffer_to_host(grad_weights)
                    grad_output_values = [[1.0, -2.0], [0.5, 4.0], [-1.5, 3.0]]
                    grad_output = backend.buffer_from_host(
                        grad_output_values, shape=(3, 2), dtype="float64"
                    )
                    api.fail_on = fail_on
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        layer.backward(grad_output)
                    api.fail_on = None
                    self.assertIs(layer.last_ids, ids)
                    self.assertIs(layer.output, output)
                    self.assertIs(layer.grad_weights, grad_weights)
                    self.assertFalse(ids.closed)
                    self.assertFalse(grad_output.closed)
                    self.assertFalse(layer._has_backward)
                    actual = backend.buffer_to_host(grad_weights)
                    if mutated:
                        expected = CPUBackend().embedding_backward(
                            original, [2, 0, 2], grad_output_values
                        )
                        self.assertEqual(actual, expected)
                    else:
                        self.assertEqual(actual, original)
                    grad_output.close()
                finally:
                    api.fail_on = None
                    layer.close()
                    backend.close()

    def test_integrated_validation_and_resident_preflight_errors_preserve_state(self) -> None:
        layer = Embedding(3, 2, seed=73, backend=self.backend)
        foreign_backend = OpenCLBackend(runtime=OpenCLRuntime(api=FakeOpenCLApi()))
        try:
            layer.forward([1, 2])
            ids = layer.last_ids
            output = layer.output
            grad_weights = layer.grad_weights
            original = self.backend.buffer_to_host(grad_weights)
            wrong_shape = self.backend.buffer_from_host(
                [[1.0, 2.0]], shape=(1, 2), dtype="float64"
            )
            foreign = foreign_backend.buffer_from_host(
                [[1.0, 2.0], [3.0, 4.0]],
                shape=(2, 2),
                dtype="float64",
            )
            resident = self.backend.buffer_from_host(
                [[1.0, 2.0], [3.0, 4.0]],
                shape=(2, 2),
                dtype="float64",
            )
            try:
                with self.assertRaisesRegex(ValueError, "shape"):
                    layer.backward(wrong_shape)
                with self.assertRaisesRegex(TypeError, "device-backed Embedding forward"):
                    layer.backward(foreign)
                self.assertIs(layer.last_ids, ids)
                self.assertIs(layer.output, output)
                self.assertIs(layer.grad_weights, grad_weights)
                self.assertFalse(layer._has_backward)

                self.api.allocations[ids.handle][:4] = struct.pack("<i", 3)
                with self.assertRaisesRegex(ValueError, "outside"):
                    layer.backward(resident)
                self.assertEqual(self.backend.buffer_to_host(grad_weights), original)
                self.assertIs(layer.last_ids, ids)
                self.assertIs(layer.output, output)
                self.assertFalse(layer._has_backward)

                self.api.allocations[ids.handle][:4] = struct.pack("<i", 1)
                self.api.allocations[resident.handle][:8] = struct.pack(
                    "<dd", float("nan"), 2.0
                )
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    layer.backward(resident)
                self.assertEqual(self.backend.buffer_to_host(grad_weights), original)
                self.assertIs(layer.grad_weights, grad_weights)
                self.assertFalse(layer._has_backward)
            finally:
                resident.close()
                foreign.close()
                wrong_shape.close()
        finally:
            layer.close()
            foreign_backend.close()

    def test_step_release_error_can_follow_update_and_keeps_layer_flag_set(self) -> None:
        api = FakeOpenCLApi(max_allocation=4096)
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        layer = Embedding(2, 2, seed=11, backend=backend)
        cpu_layer = Embedding(2, 2, seed=11, backend=self.cpu)
        try:
            ids = [1, 0, 1]
            gradients = [[1.0, -2.0], [0.5, 4.0], [-1.5, 3.0]]
            cpu_layer.forward(ids.copy())
            layer.forward(ids.copy())
            cpu_layer.backward(gradients)
            layer.backward(gradients)
            expected = self.cpu.embedding_step(
                cpu_layer.weights, cpu_layer.grad_weights, 0.125
            )
            step_kernel_handle = api._next_kernel
            api.fail_on = "release_kinera_embedding_step_f64"
            with self.assertRaisesRegex(RuntimeError, "kernel release failure"):
                layer.step(0.125)
            api.fail_on = None
            self.assertEqual(backend.buffer_to_host(layer.weights), expected)
            self.assertTrue(layer._has_backward)
            self.assertEqual(api.released_kernels.count(step_kernel_handle), 1)
        finally:
            api.fail_on = None
            layer.close()
            cpu_layer.close()
            backend.close()

    def test_step_enqueue_failure_preserves_parameters_and_backward_state(self) -> None:
        api = FakeOpenCLApi(max_allocation=4096)
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        layer = Embedding(2, 2, seed=37, backend=backend)
        cpu_layer = Embedding(2, 2, seed=37, backend=self.cpu)
        try:
            ids = [1, 0, 1]
            gradients = [[1.0, -2.0], [0.5, 4.0], [-1.5, 3.0]]
            cpu_layer.forward(ids.copy())
            layer.forward(ids.copy())
            cpu_layer.backward(gradients)
            layer.backward(gradients)
            weights_buffer = layer.weights
            weights_before = backend.buffer_to_host(weights_buffer)
            api.fail_on = "enqueue_kinera_embedding_step_f64"
            with self.assertRaisesRegex(RuntimeError, "injected enqueue failure"):
                layer.step(0.125)
            api.fail_on = None

            self.assertIs(layer.weights, weights_buffer)
            self.assertEqual(backend.buffer_to_host(layer.weights), weights_before)
            self.assertTrue(layer._has_backward)
            self.assertEqual(api.released_kernels[-1], api._next_kernel - 1)

            expected = self.cpu.embedding_step(
                cpu_layer.weights, cpu_layer.grad_weights, 0.125
            )
            cpu_layer.step(0.125)
            layer.step(0.125)
            self.assertFalse(layer._has_backward)
            self.assertIs(layer.weights, weights_buffer)
            self.assertEqual(backend.buffer_to_host(layer.weights), expected)
        finally:
            api.fail_on = None
            layer.close()
            cpu_layer.close()
            backend.close()


class OpenCLHybridRepresentationForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi(max_allocation=65536)
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))

    def tearDown(self) -> None:
        self.api.fail_on = None
        self.backend.close()

    def _buffers(self):
        char = self.backend.buffer_from_host(
            [[-0.0, 1.25], [-2.5, 0.0]], shape=(2, 2), dtype="float64"
        )
        features = self.backend.buffer_from_host(
            [[-3.0], [4.5]], shape=(2, 1), dtype="float64"
        )
        syllable = self.backend.buffer_from_host(
            [[5.0, -0.0, 7.25], [8.5, -9.0, 10.0]],
            shape=(2, 3),
            dtype="float64",
        )
        return char, features, syllable

    def test_resident_concat_preserves_channel_order_and_signed_zero(self) -> None:
        char, features, syllable = self._buffers()
        output = None
        try:
            reads_before = len(self.api.read_sizes)
            writes_before = len(self.api.write_sizes)
            output = self.backend.hybrid_representation_forward(
                char, features, syllable
            )
            self.assertEqual(self.api.read_sizes[reads_before:], [4])
            self.assertEqual(len(self.api.write_sizes), writes_before)
            self.assertEqual(output.shape, (2, 6))
            self.assertEqual(output.dtype, "float64")
            self.assertEqual(output.layout, "C")
            self.assertIs(output._manager, self.backend.memory)
            self.assertTrue(output.owns_memory)
            self.assertIsNot(output, char)
            self.assertTrue(all(not value.closed for value in (char, features, syllable)))

            expected = [
                [-0.0, 1.25, -3.0, 5.0, -0.0, 7.25],
                [-2.5, 0.0, 4.5, 8.5, -9.0, 10.0],
            ]
            self.assertEqual(self.backend.buffer_to_host(output), expected)
            output_bits = bytes(self.api.allocations[output.handle])
            expected_bits = b"".join(
                struct.pack("<d", value)
                for row in range(2)
                for value in (
                    *self.backend.buffer_to_host(char)[row],
                    *self.backend.buffer_to_host(features)[row],
                    *self.backend.buffer_to_host(syllable)[row],
                )
            )
            self.assertEqual(output_bits, expected_bits)
            self.assertEqual(math.copysign(1.0, self.backend.buffer_to_host(output)[0][0]), -1.0)
        finally:
            if output is not None:
                output.close()
            char.close()
            features.close()
            syllable.close()

    def test_shape_dtype_owner_closed_alias_and_size_validation(self) -> None:
        char, features, syllable = self._buffers()
        foreign_backend = OpenCLBackend(runtime=OpenCLRuntime(api=FakeOpenCLApi()))
        foreign = foreign_backend.buffer_from_host(
            [[1.0], [2.0]], shape=(2, 1), dtype="float64"
        )
        wrong_dtype = self.backend.buffer_from_host(
            [[1, 2], [3, 4]], shape=(2, 2), dtype="int32"
        )
        wrong_shape = self.backend.buffer_from_host(
            [[1.0], [2.0], [3.0]], shape=(3, 1), dtype="float64"
        )
        try:
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                self.backend.hybrid_representation_forward(
                    wrong_dtype, features, syllable
                )
            with self.assertRaisesRegex(ValueError, "rank 2"):
                self.backend.hybrid_representation_forward(
                    self.backend.buffer_from_host([1.0, 2.0], shape=(2,)),
                    features,
                    syllable,
                )
            with self.assertRaisesRegex(ValueError, "equal row counts"):
                self.backend.hybrid_representation_forward(
                    char, wrong_shape, syllable
                )
            with self.assertRaisesRegex(Exception, "selected OpenCL manager"):
                self.backend.hybrid_representation_forward(char, foreign, syllable)
            aliased_output = self.backend.hybrid_representation_forward(
                char, char, syllable
            )
            try:
                self.assertEqual(aliased_output.shape, (2, 7))
                self.assertEqual(
                    self.backend.buffer_to_host(aliased_output),
                    [
                        [-0.0, 1.25, -0.0, 1.25, 5.0, -0.0, 7.25],
                        [-2.5, 0.0, -2.5, 0.0, 8.5, -9.0, 10.0],
                    ],
                )
            finally:
                aliased_output.close()

            malformed = OpenCLBuffer(
                self.backend.memory,
                char.handle,
                device_id=char.device_id,
                shape=char.shape,
                dtype=char.dtype,
                nbytes=char.nbytes - 8,
            )
            with self.assertRaisesRegex(ValueError, "size does not match"):
                self.backend.hybrid_representation_forward(
                    malformed, features, syllable
                )
            malformed._mark_closed()
            oversized = OpenCLBuffer(
                self.backend.memory,
                char.handle,
                device_id=char.device_id,
                shape=(1, 2**31),
                dtype="float64",
                nbytes=2**34,
            )
            with self.assertRaisesRegex(ValueError, "signed 32-bit"):
                self.backend.hybrid_representation_forward(
                    oversized,
                    self.backend.buffer_from_host([[1.0]], shape=(1, 1)),
                    self.backend.buffer_from_host([[2.0]], shape=(1, 1)),
                )
            oversized._mark_closed()
            closed = self.backend.buffer_from_host(
                [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2), dtype="float64"
            )
            closed.close()
            with self.assertRaisesRegex(Exception, "released/closed"):
                self.backend.hybrid_representation_forward(closed, features, syllable)
        finally:
            foreign.close()
            wrong_dtype.close()
            wrong_shape.close()
            char.close()
            features.close()
            syllable.close()
            foreign_backend.close()

    def test_device_allocation_limit_is_checked_before_returning_output(self) -> None:
        api = FakeOpenCLApi(max_allocation=32)
        backend = OpenCLBackend(runtime=OpenCLRuntime(api=api))
        char = backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2)
        )
        features = backend.buffer_from_host([[5.0], [6.0]], shape=(2, 1))
        syllable = backend.buffer_from_host([[7.0], [8.0]], shape=(2, 1))
        try:
            with self.assertRaisesRegex(BackendError, "exceeds device limit"):
                backend.hybrid_representation_forward(char, features, syllable)
            self.assertEqual(set(backend.memory._buffers), {char, features, syllable})
            self.assertEqual(set(api.allocations), {char.handle, features.handle, syllable.handle})
        finally:
            char.close()
            features.close()
            syllable.close()
            backend.close()

    def test_allocator_output_alias_is_rejected_without_releasing_input(self) -> None:
        char, features, syllable = self._buffers()
        original_allocate = self.backend.memory.allocate

        def aliased_allocate(shape, *, dtype="float64", device_id=None):
            if tuple(shape) == (2, 6):
                output = OpenCLBuffer(
                    self.backend.memory,
                    char.handle,
                    device_id=self.backend.device_id,
                    shape=(2, 6),
                    dtype="float64",
                    nbytes=96,
                )
                self.backend.memory._buffers.add(output)
                return output
            return original_allocate(shape, dtype=dtype, device_id=device_id)

        try:
            with patch.object(
                self.backend.memory, "allocate", side_effect=aliased_allocate
            ):
                with self.assertRaisesRegex(Exception, "unexpectedly aliases"):
                    self.backend.hybrid_representation_forward(
                        char, features, syllable
                    )
            self.assertFalse(char.closed)
            self.assertIn(char.handle, self.api.allocations)
            self.assertEqual(len(self.backend.memory._buffers), 3)
        finally:
            char.close()
            features.close()
            syllable.close()

    def test_non_finite_resident_values_are_rejected_before_output_allocation(self) -> None:
        char, features, syllable = self._buffers()
        try:
            original_live = set(self.backend.memory._buffers)
            for channel, buffer in enumerate((char, features, syllable)):
                for invalid in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(channel=channel, invalid=invalid):
                        original_value = bytes(self.api.allocations[buffer.handle][:8])
                        self.api.allocations[buffer.handle][:8] = struct.pack(
                            "<d", invalid
                        )
                        allocations_before = set(self.api.allocations)
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            self.backend.hybrid_representation_forward(
                                char, features, syllable
                            )
                        self.assertEqual(set(self.api.allocations), allocations_before)
                        self.assertEqual(set(self.backend.memory._buffers), original_live)
                        self.assertTrue(all(not value.closed for value in (char, features, syllable)))
                        self.api.allocations[buffer.handle][:8] = original_value
        finally:
            char.close()
            features.close()
            syllable.close()

    def test_preflight_and_fusion_failures_release_temporaries_once(self) -> None:
        char, features, syllable = self._buffers()
        try:
            failure_cases = (
                ("build_program", "program build failure"),
                ("create_buffer", "allocation failure"),
                ("create_kernel", "kernel creation failure"),
                ("set_kernel_buffer_1_kinera_hybrid_representation_validate_finite_f64", "argument failure"),
                ("set_kernel_int_5_kinera_hybrid_representation_validate_finite_f64", "scalar argument failure"),
                ("enqueue_kinera_hybrid_representation_validate_finite_f64", "enqueue failure"),
                ("read_buffer", "status read failure"),
                ("release_kinera_hybrid_representation_validate_finite_f64", "kernel release failure"),
                ("create_kinera_hybrid_representation_forward_f64", "creation failure"),
                ("set_kernel_buffer_1_kinera_hybrid_representation_forward_f64", "argument failure"),
                ("enqueue_kinera_hybrid_representation_forward_f64", "enqueue failure"),
            )
            for failure, message in failure_cases:
                with self.subTest(failure=failure):
                    self.api.fail_on = failure
                    released_mem_before = len(self.api.released)
                    released_kernels_before = len(self.api.released_kernels)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.backend.hybrid_representation_forward(
                            char, features, syllable
                        )
                    self.api.fail_on = None
                    self.assertEqual(len(self.backend.memory._buffers), 3)
                    new_releases = self.api.released[released_mem_before:]
                    fusion_stage = failure in {
                        "create_kinera_hybrid_representation_forward_f64",
                        "set_kernel_buffer_1_kinera_hybrid_representation_forward_f64",
                        "enqueue_kinera_hybrid_representation_forward_f64",
                    }
                    expected_mem_releases = 2 if fusion_stage else 0 if failure in {
                        "build_program",
                        "create_buffer",
                    } else 1
                    self.assertEqual(len(new_releases), expected_mem_releases)
                    self.assertEqual(
                        len(set(new_releases)), len(new_releases)
                    )
                    new_kernel_releases = self.api.released_kernels[
                        released_kernels_before:
                    ]
                    expected_kernel_releases = (
                        2
                        if failure
                        in {
                            "set_kernel_buffer_1_kinera_hybrid_representation_forward_f64",
                            "enqueue_kinera_hybrid_representation_forward_f64",
                        }
                        else 1
                        if failure
                        in {
                            "set_kernel_buffer_1_kinera_hybrid_representation_validate_finite_f64",
                            "set_kernel_int_5_kinera_hybrid_representation_validate_finite_f64",
                            "enqueue_kinera_hybrid_representation_validate_finite_f64",
                            "read_buffer",
                            "release_kinera_hybrid_representation_validate_finite_f64",
                            "create_kinera_hybrid_representation_forward_f64",
                        }
                        else 0
                    )
                    self.assertEqual(len(new_kernel_releases), expected_kernel_releases)

            primary = RuntimeError("primary enqueue failure")
            api_release = self.api.release_kernel
            self.api.fail_on = None
            with (
                patch.object(
                    self.api,
                    "enqueue_1d",
                    side_effect=primary,
                ),
                patch.object(
                    self.api,
                    "release_kernel",
                    side_effect=RuntimeError("secondary release failure"),
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self.backend.hybrid_representation_forward(
                        char, features, syllable
                    )
            self.assertIs(raised.exception, primary)
            self.assertIsNotNone(raised.exception.__cause__)
            self.assertEqual(len(self.backend.memory._buffers), 3)
            self.api.release_kernel = api_release

            release_primary = RuntimeError("preflight enqueue remains primary")
            cleanup_failure = RuntimeError("status buffer release failure")
            def release_then_raise(handle):
                self.api.released.append(handle)
                del self.api.allocations[handle]
                raise cleanup_failure

            with (
                patch.object(self.api, "enqueue_1d", side_effect=release_primary),
                patch.object(
                    self.api, "release_mem", side_effect=release_then_raise
                ) as release_mem,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self.backend.hybrid_representation_forward(
                        char, features, syllable
                    )
            self.assertIs(raised.exception, release_primary)
            self.assertIs(raised.exception.__cause__, cleanup_failure)
            release_mem.assert_called_once()
            self.assertEqual(len(self.backend.memory._buffers), 3)
            self.assertEqual(set(self.api.allocations), {char.handle, features.handle, syllable.handle})
        finally:
            self.api.fail_on = None
            char.close()
            features.close()
            syllable.close()

    def test_hybrid_layer_retains_fusion_and_releases_intermediates(self) -> None:
        cpu = CPUBackend()
        cpu_layer = HybridRepresentation(
            Embedding(4, 2, seed=5, backend=cpu),
            Embedding(3, 3, seed=6, backend=cpu),
            feature_dim=2,
        )
        gpu_layer = HybridRepresentation(
            Embedding(4, 2, seed=5, backend=self.backend),
            Embedding(3, 3, seed=6, backend=self.backend),
            feature_dim=2,
        )
        try:
            chars, syllables = [1, 2], [0, 1]
            features = [[-0.5, 2.0], [3.25, -4.0]]
            expected = cpu_layer.forward(chars.copy(), syllables.copy(), features)
            reads_before = len(self.api.read_sizes)
            actual = gpu_layer.forward(chars.copy(), syllables.copy(), features)
            self.assertEqual(self.api.read_sizes[reads_before:], [4, 4, 4])
            self.assertEqual(self.backend.buffer_to_host(actual), expected)
            self.assertIs(gpu_layer.last_output, actual)
            self.assertTrue(gpu_layer.char_embedding.output.closed)
            self.assertTrue(gpu_layer.syllable_embedding.output.closed)
            self.assertIsInstance(gpu_layer.char_embedding.last_ids, OpenCLBuffer)
            self.assertIsInstance(gpu_layer.syllable_embedding.last_ids, OpenCLBuffer)

            previous_output = actual
            previous_context = gpu_layer.last_context_size
            with self.assertRaisesRegex(ValueError, "dimensão 2"):
                gpu_layer.forward(chars.copy(), syllables.copy(), [[1.0], [2.0]])
            self.assertIs(gpu_layer.last_output, previous_output)
            self.assertFalse(previous_output.closed)
            self.assertEqual(gpu_layer.last_context_size, previous_context)

            self.api.fail_on = "create_buffer"
            with self.assertRaisesRegex(RuntimeError, "allocation failure"):
                gpu_layer.forward(chars.copy(), syllables.copy(), features)
            self.api.fail_on = None
            self.assertTrue(previous_output.closed)
            self.assertIsNone(gpu_layer.last_output)
            self.assertIsNone(gpu_layer.last_context_size)
            self.assertIsNone(gpu_layer.char_embedding.last_ids)
            recovered = gpu_layer.forward(chars.copy(), syllables.copy(), features)
            self.assertEqual(self.backend.buffer_to_host(recovered), expected)

            self.assertEqual(gpu_layer.forward([], [], []), [])
            self.assertEqual(gpu_layer.last_output, [])
            self.assertEqual(gpu_layer.last_context_size, 0)
            self.assertTrue(previous_output.closed)
            self.assertEqual(gpu_layer.char_embedding.last_ids, [])
            self.assertEqual(gpu_layer.syllable_embedding.last_ids, [])
            self.assertTrue(all(len(data) > 0 for data in self.api.allocations.values()))
            gpu_layer.release_output()
            gpu_layer.char_embedding.close()
            gpu_layer.syllable_embedding.close()
            gpu_layer.char_embedding.close()
            gpu_layer.syllable_embedding.close()
            self.assertEqual(len(self.backend.memory._buffers), 0)
        finally:
            gpu_layer.release_output()
            gpu_layer.char_embedding.close()
            gpu_layer.syllable_embedding.close()
            cpu_layer.char_embedding.close()
            cpu_layer.syllable_embedding.close()


class OpenCLMeanPoolForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeOpenCLApi(max_allocation=65536, global_memory=65536)
        self.backend = OpenCLBackend(runtime=OpenCLRuntime(api=self.api))
        self.cpu = CPUBackend()

    def tearDown(self) -> None:
        self.api.fail_on = None
        self.backend.close()

    def test_resident_pool_matches_cpu_sum_order_and_reads_only_small_status(self) -> None:
        cases = (
            [[-0.0, 2.5]],
            [
                [1e16, 1e16, 1e16, 1.0, 1e308, 0.125, 1e16],
                [1.0, -1e16, 1.0, 1e16, -1e308, 0.25, -1e16],
                [-1e16, 1.0, -1e16, 1.0, 1e292, 0.375, 1.0],
                [1.0, 1.0, 1.0, -1e16, 0.5, 0.5, -1e16],
            ],
            [
                [1e308, -1e308, 1e16, -1e16],
                [-1e308, 1e308, 1.0, -1.0],
                [1e292, 1e292, -1e16, 1e16],
            ],
        )
        long_pattern = (
            [1e16, -1e16, 1.0, 1e-16],
            [1.0, 1e16, -1e16, -1e-16],
            [-1e16, 1.0, 1e16, 1e-16],
            [-1e16, -1.0, 1e16, -1e-16],
        )
        long_context = [long_pattern[row % len(long_pattern)] for row in range(257)]
        for values in (*cases, long_context):
            with self.subTest(shape=(len(values), len(values[0]))):
                vectors = self.backend.buffer_from_host(
                    values, shape=(len(values), len(values[0]))
                )
                try:
                    expected = self.cpu.mean_pool_forward(values)
                    input_before = bytes(self.api.allocations[vectors.handle])
                    reads_before = len(self.api.read_sizes)
                    writes_before = len(self.api.write_sizes)
                    pooled = self.backend.mean_pool_forward(vectors)
                    try:
                        self.assertEqual(
                            self.api.read_sizes[reads_before:], [ctypes.sizeof(ctypes.c_int32)]
                        )
                        self.assertEqual(len(self.api.write_sizes), writes_before)
                        self.assertEqual(bytes(self.api.allocations[vectors.handle]), input_before)
                        self.assertEqual(pooled.shape, (len(values[0]),))
                        self.assertEqual(pooled.dtype, "float64")
                        self.assertEqual(pooled.layout, "C")
                        self.assertIs(pooled._manager, self.backend.memory)
                        self.assertTrue(pooled.owns_memory)
                        actual = self.backend.buffer_to_host(pooled)
                        self.assertEqual(len(actual), len(expected))
                        for actual_value, expected_value in zip(actual, expected):
                            if math.isinf(expected_value):
                                self.assertEqual(actual_value, expected_value)
                            else:
                                self.assertTrue(
                                    math.isclose(
                                        actual_value,
                                        expected_value,
                                        rel_tol=1e-12,
                                        abs_tol=1e-12,
                                    ),
                                    (actual_value, expected_value),
                                )
                        self.assertFalse(vectors.closed)
                    finally:
                        pooled.close()
                finally:
                    vectors.close()

    def test_python_sum_policy_is_passed_to_kernel_and_naive_mode_remains_supported(self) -> None:
        vectors = self.backend.buffer_from_host(
            [[0.25, -0.5], [0.5, 1.0], [-0.125, 0.25]],
            shape=(3, 2),
        )
        try:
            with patch(
                "kinera.backends.opencl.backend._PYTHON_FLOAT_SUM_COMPENSATED",
                False,
            ):
                pooled = self.backend.mean_pool_forward(vectors)
            try:
                pool_kernel = next(
                    kernel
                    for kernel, name in self.api.kernel_names.items()
                    if name == "kinera_mean_pool_forward_f64"
                )
                self.assertEqual(self.api.kernel_ints[pool_kernel][4], 0)
                actual = self.backend.buffer_to_host(pooled)
                expected = self.cpu.mean_pool_forward(
                    [[0.25, -0.5], [0.5, 1.0], [-0.125, 0.25]]
                )
                for actual_value, expected_value in zip(actual, expected):
                    self.assertTrue(
                        math.isclose(
                            actual_value,
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )
            finally:
                pooled.close()
        finally:
            vectors.close()

    def test_finite_inputs_that_overflow_match_cpu_without_claiming_finite_output(self) -> None:
        values = [[1e308], [1e308]]
        vectors = self.backend.buffer_from_host(values, shape=(2, 1))
        try:
            expected = self.cpu.mean_pool_forward(values)
            pooled = self.backend.mean_pool_forward(vectors)
            try:
                actual = self.backend.buffer_to_host(pooled)
                self.assertTrue(math.isinf(expected[0]))
                self.assertEqual(actual, expected)
            finally:
                pooled.close()
        finally:
            vectors.close()

    def test_resident_non_finite_inputs_are_rejected_before_output_allocation(self) -> None:
        vectors = self.backend.buffer_from_host(
            [[1.0, 2.0], [3.0, 4.0]], shape=(2, 2)
        )
        try:
            original = bytes(self.api.allocations[vectors.handle])
            live = set(self.backend.memory._buffers)
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(invalid=invalid):
                    self.api.allocations[vectors.handle][:8] = struct.pack("<d", invalid)
                    allocations_before = set(self.api.allocations)
                    reads_before = len(self.api.read_sizes)
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        self.backend.mean_pool_forward(vectors)
                    self.assertEqual(set(self.api.allocations), allocations_before)
                    self.assertEqual(set(self.backend.memory._buffers), live)
                    self.assertEqual(self.api.read_sizes[reads_before:], [4])
                    self.api.allocations[vectors.handle][:] = original
        finally:
            vectors.close()

    def test_invalid_buffer_metadata_and_ownership_fail_before_dispatch(self) -> None:
        vectors = self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
        wrong_dtype = self.backend.buffer_from_host([1, 2], shape=(2,), dtype="int32")
        wrong_rank = self.backend.buffer_from_host([1.0, 2.0], shape=(2,))
        foreign_backend = OpenCLBackend(runtime=OpenCLRuntime(api=FakeOpenCLApi()))
        foreign = foreign_backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
        try:
            with self.assertRaisesRegex(ValueError, "dtype float64"):
                self.backend.mean_pool_forward(wrong_dtype)
            with self.assertRaisesRegex(ValueError, "rank 2"):
                self.backend.mean_pool_forward(wrong_rank)
            with self.assertRaisesRegex(BackendError, "different memory manager"):
                self.backend.mean_pool_forward(foreign)
            with patch.object(
                OpenCLBuffer,
                "layout",
                new_callable=PropertyMock,
                return_value="strided",
            ):
                with self.assertRaisesRegex(ValueError, "C layout"):
                    self.backend.mean_pool_forward(vectors)

            wrong_device = OpenCLBuffer(
                self.backend.memory,
                vectors.handle,
                device_id=1,
                shape=vectors.shape,
                dtype="float64",
                nbytes=vectors.nbytes,
            )
            with self.assertRaisesRegex(BackendError, "bound to GPU"):
                self.backend.mean_pool_forward(wrong_device)
            wrong_device._mark_closed()

            closed = self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
            closed.close()
            with self.assertRaisesRegex(Exception, "released/closed"):
                self.backend.mean_pool_forward(closed)

            malformed = OpenCLBuffer(
                self.backend.memory,
                vectors.handle,
                device_id=vectors.device_id,
                shape=vectors.shape,
                dtype="float64",
                nbytes=vectors.nbytes - 8,
            )
            with self.assertRaisesRegex(ValueError, "size does not match"):
                self.backend.mean_pool_forward(malformed)
            malformed._mark_closed()

            empty = OpenCLBuffer(
                self.backend.memory,
                vectors.handle,
                device_id=vectors.device_id,
                shape=(0, 2),
                dtype="float64",
                nbytes=0,
            )
            with self.assertRaisesRegex(ValueError, "positive integers"):
                self.backend.mean_pool_forward(empty)
            empty._mark_closed()

            oversized = OpenCLBuffer(
                self.backend.memory,
                vectors.handle,
                device_id=vectors.device_id,
                shape=(2**31, 1),
                dtype="float64",
                nbytes=2**31 * 8,
            )
            with self.assertRaisesRegex(ValueError, "signed 32-bit"):
                self.backend.mean_pool_forward(oversized)
            oversized._mark_closed()
            self.assertEqual(set(self.backend.memory._buffers), {vectors, wrong_dtype, wrong_rank})
        finally:
            vectors.close()
            wrong_dtype.close()
            wrong_rank.close()
            foreign.close()
            foreign_backend.close()

    def test_failure_paths_release_temporaries_once_and_keep_input_resident(self) -> None:
        vectors = self.backend.buffer_from_host([[1.0, 2.0], [3.0, 4.0]], shape=(2, 2))
        try:
            input_handles = {vectors.handle}
            failures = (
                ("build_program", "program build failure"),
                ("create_buffer", "allocation failure"),
                ("create_kinera_mean_pool_validate_finite_f64", "creation failure"),
                ("set_kernel_buffer_0_kinera_mean_pool_validate_finite_f64", "argument failure"),
                ("set_kernel_int_3_kinera_mean_pool_validate_finite_f64", "scalar argument failure"),
                ("enqueue_kinera_mean_pool_validate_finite_f64", "enqueue failure"),
                ("read_buffer", "status read failure"),
                ("release_kinera_mean_pool_validate_finite_f64", "kernel release failure"),
                ("create_buffer_16", "allocation failure"),
                ("create_kinera_mean_pool_forward_f64", "creation failure"),
                ("set_kernel_buffer_1_kinera_mean_pool_forward_f64", "argument failure"),
                ("set_kernel_int_4_kinera_mean_pool_forward_f64", "scalar argument failure"),
                ("set_kernel_double_5", "double argument failure"),
                ("enqueue_kinera_mean_pool_forward_f64", "enqueue failure"),
                ("release_kinera_mean_pool_forward_f64", "kernel release failure"),
            )
            for failure, message in failures:
                with self.subTest(failure=failure):
                    self.api.fail_on = failure
                    released_before = len(self.api.released)
                    kernels_before = len(self.api.released_kernels)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.backend.mean_pool_forward(vectors)
                    self.api.fail_on = None
                    new_releases = self.api.released[released_before:]
                    self.assertEqual(len(new_releases), len(set(new_releases)))
                    self.assertEqual(
                        set(self.api.allocations), input_handles
                    )
                    self.assertEqual(set(self.backend.memory._buffers), {vectors})
                    new_kernel_releases = self.api.released_kernels[kernels_before:]
                    self.assertEqual(
                        len(new_kernel_releases), len(set(new_kernel_releases))
                    )

            self.api.fail_on = "release_mem"
            releases_before = len(self.api.released)
            with self.assertRaisesRegex(RuntimeError, "memory release failure"):
                self.backend.mean_pool_forward(vectors)
            self.api.fail_on = None
            new_releases = self.api.released[releases_before:]
            self.assertEqual(len(new_releases), 1)
            self.assertEqual(len(set(new_releases)), 1)
            self.assertEqual(set(self.api.allocations), input_handles)
            self.assertEqual(set(self.backend.memory._buffers), {vectors})
            self.assertFalse(vectors.closed)
        finally:
            self.api.fail_on = None
            vectors.close()

    def test_primary_error_survives_a_secondary_cleanup_failure(self) -> None:
        vectors = self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
        primary = RuntimeError("primary preflight enqueue failure")
        cleanup = RuntimeError("secondary status release failure")
        original_release = self.api.release_mem

        def release_then_raise(handle: int) -> None:
            original_release(handle)
            raise cleanup

        try:
            with (
                patch.object(self.api, "enqueue_1d", side_effect=primary),
                patch.object(self.api, "release_mem", side_effect=release_then_raise) as release,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self.backend.mean_pool_forward(vectors)
            self.assertIs(raised.exception, primary)
            self.assertIs(raised.exception.__cause__, cleanup)
            release.assert_called_once()
            self.assertEqual(set(self.api.allocations), {vectors.handle})
            self.assertEqual(set(self.backend.memory._buffers), {vectors})
        finally:
            vectors.close()

    def test_allocator_alias_is_disarmed_without_releasing_borrowed_input(self) -> None:
        vectors = self.backend.buffer_from_host([[1.0, 2.0]], shape=(1, 2))
        original_allocate = self.backend.memory.allocate

        def aliased_allocate(shape, *, dtype="float64", device_id=None):
            if tuple(shape) == (2,) and dtype == "float64":
                output = OpenCLBuffer(
                    self.backend.memory,
                    vectors.handle,
                    device_id=vectors.device_id,
                    shape=(2,),
                    dtype="float64",
                    nbytes=16,
                )
                self.backend.memory._buffers.add(output)
                return output
            return original_allocate(shape, dtype=dtype, device_id=device_id)

        try:
            def status_alias_allocate(shape, *, dtype="float64", device_id=None):
                if tuple(shape) == (1,) and dtype == "int32":
                    status = OpenCLBuffer(
                        self.backend.memory,
                        vectors.handle,
                        device_id=vectors.device_id,
                        shape=(1,),
                        dtype="int32",
                        nbytes=4,
                    )
                    self.backend.memory._buffers.add(status)
                    return status
                return original_allocate(shape, dtype=dtype, device_id=device_id)

            with patch.object(
                self.backend.memory, "allocate", side_effect=status_alias_allocate
            ):
                with self.assertRaisesRegex(BackendError, "status unexpectedly aliases"):
                    self.backend.mean_pool_forward(vectors)
            self.assertFalse(vectors.closed)
            self.assertIn(vectors.handle, self.api.allocations)
            self.assertEqual(set(self.backend.memory._buffers), {vectors})

            with patch.object(self.backend.memory, "allocate", side_effect=aliased_allocate):
                with self.assertRaisesRegex(BackendError, "unexpectedly aliases"):
                    self.backend.mean_pool_forward(vectors)
            self.assertFalse(vectors.closed)
            self.assertIn(vectors.handle, self.api.allocations)
            self.assertEqual(set(self.backend.memory._buffers), {vectors})
        finally:
            vectors.close()

    def test_representation_output_is_pooled_resident_and_released_by_consumer(self) -> None:
        gpu_model = AutoregressiveModel(
            5,
            4,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=217,
            backend=self.backend,
        )
        cpu_model = AutoregressiveModel(
            5,
            4,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=217,
            backend=self.cpu,
        )
        chars, syllables = [1, 3, 2], [2, 0, 1]
        features = [[0.25, -1.0], [2.5, 0.0], [-0.5, 3.0]]
        weights = gpu_model.representation.char_embedding.weights
        original_weights = bytes(self.api.allocations[weights.handle])
        try:
            expected = cpu_model._pool_representation_backend(
                chars.copy(), syllables.copy(), features
            )
            reads_before = len(self.api.read_sizes)
            writes_before = len(self.api.write_sizes)
            actual = gpu_model._pool_representation_backend(
                chars.copy(), syllables.copy(), features
            )
            self.assertIsInstance(actual, OpenCLBuffer)
            self.assertEqual(actual.shape, (gpu_model.representation.output_dim,))
            self.assertEqual(
                self.api.read_sizes[reads_before:], [4, 4, 4, 4]
            )
            writes = self.api.write_sizes[writes_before:]
            self.assertEqual(writes, [12, 80, 12, 96, 48])
            self.assertNotIn(3 * gpu_model.representation.output_dim * 8, writes)
            self.assertNotIn(gpu_model.representation.output_dim * 8, writes)
            self.assertIsNone(gpu_model.representation.last_output)
            self.assertFalse(weights.closed)
            self.assertEqual(bytes(self.api.allocations[weights.handle]), original_weights)
            self.assertTrue(gpu_model.representation.char_embedding.output.closed)
            self.assertTrue(gpu_model.representation.syllable_embedding.output.closed)
            try:
                actual_values = self.backend.buffer_to_host(actual)
                for actual_value, expected_value in zip(actual_values, expected):
                    self.assertTrue(
                        math.isclose(
                            actual_value,
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )
            finally:
                actual.close()
        finally:
            gpu_model.close()
            cpu_model.close()


if __name__ == "__main__":
    unittest.main()
