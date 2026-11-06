"""CPU-only OpenCL discovery and explicit buffer lifecycle tests."""

import ctypes
import math
import struct
import subprocess
import sys
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


class FakeOpenCLApi:
    def __init__(
        self,
        *,
        fp64: bool = True,
        max_allocation: int = 128,
        fail_on: str | None = None,
    ) -> None:
        self.fp64 = fp64
        self.max_allocation = max_allocation
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
            "global_mem_size": 1024,
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
        if self.fail_on == "create_buffer":
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
        self.assertEqual(
            math.fma(-learning_rate, self.gradients_values[0][0], 1.0),
            2.0**-54,
        )
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


if __name__ == "__main__":
    unittest.main()
