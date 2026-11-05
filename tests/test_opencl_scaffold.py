"""CPU-only OpenCL discovery and explicit buffer lifecycle tests."""

import ctypes
import subprocess
import sys
import unittest
from unittest.mock import patch

from kinera.backends import Backend, BackendError, BackendUnavailableError
from kinera.backends.opencl.api import (
    CL_DEVICE_NOT_FOUND,
    CL_PLATFORM_NOT_FOUND_KHR,
    OpenCLApi,
    OpenCLApiError,
)
from kinera.backends.opencl.backend import OpenCLBackend
from kinera.backends.opencl import memory as opencl_memory
from kinera.backends.opencl.runtime import OpenCLRuntime


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
        self._next_handle = 100

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
        handle = self._next_handle
        self._next_handle += 1
        self.allocations[handle] = bytearray(nbytes)
        return handle

    def write_buffer(self, queue, buffer, host_address, nbytes):
        self.allocations[buffer][:] = ctypes.string_at(host_address, nbytes)

    def read_buffer(self, queue, buffer, host_address, nbytes):
        ctypes.memmove(host_address, bytes(self.allocations[buffer]), nbytes)

    def release_mem(self, handle):
        self.released.append(handle)
        del self.allocations[handle]

    def release_command_queue(self, handle):
        pass

    def release_context(self, handle):
        pass

    def build_program(self, context, device, source, options):
        return 20

    def create_kernel(self, program, name):
        if self.fail_on == "create_kernel":
            raise RuntimeError("injected OpenCL kernel creation failure")
        self.created_kernels.append(name)
        return 30

    def set_kernel_buffer(self, kernel, index, buffer):
        if self.fail_on == f"set_kernel_buffer_{index}":
            raise RuntimeError("injected OpenCL argument failure")
        self.kernel_arguments.append((index, buffer))

    def set_kernel_int(self, kernel, index, value):
        if self.fail_on == f"set_kernel_int_{index}":
            raise RuntimeError("injected OpenCL scalar argument failure")

    def set_kernel_double(self, kernel, index, value):
        if self.fail_on == f"set_kernel_double_{index}":
            raise RuntimeError("injected OpenCL double argument failure")
        self.kernel_double_arguments.append((index, value))

    def enqueue_1d(self, queue, kernel, global_size):
        if self.fail_on == "enqueue_1d":
            raise RuntimeError("injected OpenCL enqueue failure")
        self.enqueues.append(global_size)

    def release_kernel(self, handle):
        self.released_kernels.append(handle)

    def release_program(self, handle):
        self.programs_released.append(handle)


class OpenCLScaffoldTests(unittest.TestCase):
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
        with self.assertRaisesRegex(BackendUnavailableError, "current gate"):
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


if __name__ == "__main__":
    unittest.main()
