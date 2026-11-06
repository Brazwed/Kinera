"""Small, lazy ctypes binding for the OpenCL C API."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

from kinera.backends.base import BackendUnavailableError, BackendError


CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_PLATFORM_NOT_FOUND_KHR = -1001
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_WRITE = 1 << 0
CL_TRUE = 1
CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F

_CL_PLATFORM_NAME = 0x0902
_CL_PLATFORM_VENDOR = 0x0903
_CL_DEVICE_TYPE = 0x1000
_CL_DEVICE_VENDOR_ID = 0x1001
_CL_DEVICE_NAME = 0x102B
_CL_DEVICE_VENDOR = 0x102C
_CL_DRIVER_VERSION = 0x102D
_CL_DEVICE_VERSION = 0x102F
_CL_DEVICE_EXTENSIONS = 0x1030
_CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
_CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
_CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
_CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
_CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
_CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
_CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
_CL_DEVICE_AVAILABLE = 0x1027
_CL_DEVICE_COMPILER_AVAILABLE = 0x1028
_CL_PROGRAM_BUILD_LOG = 0x1183


class OpenCLApiError(BackendError):
    """An OpenCL loader or driver call failed."""


class OpenCLApi:
    """Load OpenCL only when an OpenCL backend is explicitly requested."""

    def __init__(self, library: Any | None = None) -> None:
        if library is None:
            library_path = ctypes.util.find_library("OpenCL")
            if library_path is None:
                raise BackendUnavailableError(
                    "OpenCL loader libOpenCL was not found; install an OpenCL ICD loader"
                )
            try:
                library = ctypes.CDLL(library_path)
            except OSError as error:
                raise BackendUnavailableError(
                    f"OpenCL loader could not be loaded: {error}"
                ) from error
        self._library = library
        self._bind_functions()

    def _bind(self, name: str, result: Any, arguments: list[Any]) -> Any:
        try:
            function = getattr(self._library, name)
        except AttributeError as error:
            raise BackendUnavailableError(
                f"OpenCL loader is missing required symbol {name}"
            ) from error
        function.restype = result
        function.argtypes = arguments
        return function

    def _bind_functions(self) -> None:
        void_p = ctypes.c_void_p
        uint = ctypes.c_uint
        int_ = ctypes.c_int
        size_t = ctypes.c_size_t
        ulong = ctypes.c_ulonglong

        self._cl_get_platform_ids = self._bind(
            "clGetPlatformIDs",
            int_,
            [uint, ctypes.POINTER(void_p), ctypes.POINTER(uint)],
        )
        self._cl_get_platform_info = self._bind(
            "clGetPlatformInfo",
            int_,
            [void_p, uint, size_t, void_p, ctypes.POINTER(size_t)],
        )
        self._cl_get_device_ids = self._bind(
            "clGetDeviceIDs",
            int_,
            [void_p, ulong, uint, ctypes.POINTER(void_p), ctypes.POINTER(uint)],
        )
        self._cl_get_device_info = self._bind(
            "clGetDeviceInfo",
            int_,
            [void_p, uint, size_t, void_p, ctypes.POINTER(size_t)],
        )
        self._cl_create_context = self._bind(
            "clCreateContext",
            void_p,
            [
                ctypes.POINTER(ctypes.c_ssize_t),
                uint,
                ctypes.POINTER(void_p),
                void_p,
                void_p,
                ctypes.POINTER(int_),
            ],
        )
        self._cl_create_command_queue = self._bind(
            "clCreateCommandQueue",
            void_p,
            [void_p, void_p, ulong, ctypes.POINTER(int_)],
        )
        self._cl_create_buffer = self._bind(
            "clCreateBuffer",
            void_p,
            [void_p, ulong, size_t, void_p, ctypes.POINTER(int_)],
        )
        self._cl_enqueue_write_buffer = self._bind(
            "clEnqueueWriteBuffer",
            int_,
            [
                void_p,
                void_p,
                uint,
                size_t,
                size_t,
                void_p,
                uint,
                void_p,
                void_p,
            ],
        )
        self._cl_enqueue_read_buffer = self._bind(
            "clEnqueueReadBuffer",
            int_,
            [
                void_p,
                void_p,
                uint,
                size_t,
                size_t,
                void_p,
                uint,
                void_p,
                void_p,
            ],
        )
        self._cl_create_program_with_source = self._bind(
            "clCreateProgramWithSource",
            void_p,
            [void_p, uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(size_t), ctypes.POINTER(int_)],
        )
        self._cl_build_program = self._bind(
            "clBuildProgram",
            int_,
            [void_p, uint, ctypes.POINTER(void_p), ctypes.c_char_p, void_p, void_p],
        )
        self._cl_get_program_build_info = self._bind(
            "clGetProgramBuildInfo",
            int_,
            [void_p, void_p, uint, size_t, void_p, ctypes.POINTER(size_t)],
        )
        self._cl_create_kernel = self._bind(
            "clCreateKernel", void_p, [void_p, ctypes.c_char_p, ctypes.POINTER(int_)]
        )
        self._cl_set_kernel_arg = self._bind(
            "clSetKernelArg", int_, [void_p, uint, size_t, void_p]
        )
        self._cl_enqueue_nd_range_kernel = self._bind(
            "clEnqueueNDRangeKernel",
            int_,
            [
                void_p,
                void_p,
                uint,
                ctypes.POINTER(size_t),
                ctypes.POINTER(size_t),
                ctypes.POINTER(size_t),
                uint,
                void_p,
                void_p,
            ],
        )
        self._cl_finish = self._bind("clFinish", int_, [void_p])
        self._cl_release_mem_object = self._bind("clReleaseMemObject", int_, [void_p])
        self._cl_release_kernel = self._bind("clReleaseKernel", int_, [void_p])
        self._cl_release_program = self._bind("clReleaseProgram", int_, [void_p])
        self._cl_release_command_queue = self._bind("clReleaseCommandQueue", int_, [void_p])
        self._cl_release_context = self._bind("clReleaseContext", int_, [void_p])

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != CL_SUCCESS:
            raise OpenCLApiError(f"{operation} failed with OpenCL status {status}")

    def _get_string(self, function: Any, handle: int, key: int) -> str:
        size = ctypes.c_size_t()
        self._check(function(handle, key, 0, None, ctypes.byref(size)), "OpenCL string size query")
        if size.value == 0:
            return ""
        value = ctypes.create_string_buffer(size.value)
        self._check(
            function(handle, key, size.value, value, None),
            "OpenCL string query",
        )
        return value.value.decode("utf-8", errors="replace")

    def platforms(self) -> list[int]:
        count = ctypes.c_uint()
        status = self._cl_get_platform_ids(0, None, ctypes.byref(count))
        if status == CL_PLATFORM_NOT_FOUND_KHR:
            return []
        self._check(status, "clGetPlatformIDs")
        if count.value == 0:
            return []
        result = (ctypes.c_void_p * count.value)()
        self._check(
            self._cl_get_platform_ids(count.value, result, ctypes.byref(count)),
            "clGetPlatformIDs",
        )
        return [int(result[index]) for index in range(count.value)]

    def platform_info(self, platform: int) -> dict[str, str]:
        return {
            "name": self._get_string(self._cl_get_platform_info, platform, _CL_PLATFORM_NAME),
            "vendor": self._get_string(self._cl_get_platform_info, platform, _CL_PLATFORM_VENDOR),
        }

    def devices(self, platform: int) -> list[int]:
        count = ctypes.c_uint()
        status = self._cl_get_device_ids(
            platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(count)
        )
        if status == CL_DEVICE_NOT_FOUND:
            return []
        self._check(status, "clGetDeviceIDs(GPU)")
        if count.value == 0:
            return []
        result = (ctypes.c_void_p * count.value)()
        self._check(
            self._cl_get_device_ids(
                platform, CL_DEVICE_TYPE_GPU, count.value, result, ctypes.byref(count)
            ),
            "clGetDeviceIDs(GPU)",
        )
        return [int(result[index]) for index in range(count.value)]

    def _device_scalar(self, device: int, key: int, value_type: Any) -> int:
        value = value_type()
        self._check(
            self._cl_get_device_info(
                device, key, ctypes.sizeof(value), ctypes.byref(value), None
            ),
            f"clGetDeviceInfo(0x{key:x})",
        )
        return int(value.value)

    def _optional_device_scalar(self, device: int, key: int, value_type: Any) -> int:
        try:
            return self._device_scalar(device, key, value_type)
        except OpenCLApiError:
            return 0

    def _device_string(self, device: int, key: int) -> str:
        return self._get_string(self._cl_get_device_info, device, key)

    def _work_item_sizes(self, device: int) -> tuple[int, ...]:
        dimensions = self._device_scalar(device, _CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, ctypes.c_uint)
        if dimensions == 0 or dimensions > 32:
            return ()
        sizes = (ctypes.c_size_t * dimensions)()
        self._check(
            self._cl_get_device_info(
                device,
                _CL_DEVICE_MAX_WORK_ITEM_SIZES,
                ctypes.sizeof(sizes),
                sizes,
                None,
            ),
            "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)",
        )
        return tuple(int(value) for value in sizes)

    def _pci_bus_info(self, device: int, extensions: tuple[str, ...]) -> dict[str, int] | None:
        if "cl_khr_pci_bus_info" not in extensions:
            return None

        class PciBusInfo(ctypes.Structure):
            _fields_ = [
                ("domain", ctypes.c_uint),
                ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint),
                ("function", ctypes.c_uint),
            ]

        value = PciBusInfo()
        self._check(
            self._cl_get_device_info(
                device,
                CL_DEVICE_PCI_BUS_INFO_KHR,
                ctypes.sizeof(value),
                ctypes.byref(value),
                None,
            ),
            "clGetDeviceInfo(CL_DEVICE_PCI_BUS_INFO_KHR)",
        )
        return {
            "domain": int(value.domain),
            "bus": int(value.bus),
            "device": int(value.device),
            "function": int(value.function),
        }

    def device_info(self, device: int, platform_info: dict[str, str]) -> dict[str, Any]:
        extensions = tuple(self._device_string(device, _CL_DEVICE_EXTENSIONS).split())
        return {
            "name": self._device_string(device, _CL_DEVICE_NAME),
            "vendor": self._device_string(device, _CL_DEVICE_VENDOR),
            "vendor_id": self._device_scalar(device, _CL_DEVICE_VENDOR_ID, ctypes.c_uint),
            "device_type": self._device_scalar(device, _CL_DEVICE_TYPE, ctypes.c_ulonglong),
            "driver_version": self._device_string(device, _CL_DRIVER_VERSION),
            "version": self._device_string(device, _CL_DEVICE_VERSION),
            "extensions": extensions,
            "pci_bus_info": self._pci_bus_info(device, extensions),
            "global_mem_size": self._device_scalar(device, _CL_DEVICE_GLOBAL_MEM_SIZE, ctypes.c_ulonglong),
            "max_mem_alloc_size": self._device_scalar(device, _CL_DEVICE_MAX_MEM_ALLOC_SIZE, ctypes.c_ulonglong),
            "local_mem_size": self._device_scalar(device, _CL_DEVICE_LOCAL_MEM_SIZE, ctypes.c_ulonglong),
            "max_work_group_size": self._device_scalar(device, _CL_DEVICE_MAX_WORK_GROUP_SIZE, ctypes.c_size_t),
            "max_work_item_sizes": self._work_item_sizes(device),
            "double_fp_config": self._optional_device_scalar(device, _CL_DEVICE_DOUBLE_FP_CONFIG, ctypes.c_ulonglong),
            "available": bool(self._device_scalar(device, _CL_DEVICE_AVAILABLE, ctypes.c_uint)),
            "compiler_available": bool(self._device_scalar(device, _CL_DEVICE_COMPILER_AVAILABLE, ctypes.c_uint)),
            "opencl_c_version": self._device_string(device, 0x103D),
            "platform_name": platform_info["name"],
            "platform_vendor": platform_info["vendor"],
        }

    def create_context(self, device: int) -> int:
        selected_device = ctypes.c_void_p(device)
        status = ctypes.c_int()
        context = self._cl_create_context(
            None, 1, ctypes.byref(selected_device), None, None, ctypes.byref(status)
        )
        self._check(status.value, "clCreateContext")
        if not context:
            raise OpenCLApiError("clCreateContext returned a null context")
        return int(context)

    def create_command_queue(self, context: int, device: int) -> int:
        status = ctypes.c_int()
        queue = self._cl_create_command_queue(context, device, 0, ctypes.byref(status))
        self._check(status.value, "clCreateCommandQueue")
        if not queue:
            raise OpenCLApiError("clCreateCommandQueue returned a null queue")
        return int(queue)

    def create_buffer(self, context: int, nbytes: int) -> int:
        status = ctypes.c_int()
        buffer = self._cl_create_buffer(
            context, CL_MEM_READ_WRITE, nbytes, None, ctypes.byref(status)
        )
        self._check(status.value, "clCreateBuffer")
        if not buffer:
            raise OpenCLApiError("clCreateBuffer returned a null buffer")
        return int(buffer)

    def write_buffer(self, queue: int, buffer: int, host_address: int, nbytes: int) -> None:
        self._check(
            self._cl_enqueue_write_buffer(
                queue,
                buffer,
                CL_TRUE,
                0,
                nbytes,
                host_address,
                0,
                None,
                None,
            ),
            "clEnqueueWriteBuffer",
        )

    def read_buffer(self, queue: int, buffer: int, host_address: int, nbytes: int) -> None:
        self._check(
            self._cl_enqueue_read_buffer(
                queue,
                buffer,
                CL_TRUE,
                0,
                nbytes,
                host_address,
                0,
                None,
                None,
            ),
            "clEnqueueReadBuffer",
        )

    def build_program(self, context: int, device: int, source: str, options: str) -> int:
        encoded_source = source.encode("utf-8")
        source_pointer = ctypes.c_char_p(encoded_source)
        source_length = ctypes.c_size_t(len(encoded_source))
        status = ctypes.c_int()
        program = self._cl_create_program_with_source(
            context,
            1,
            ctypes.byref(source_pointer),
            ctypes.byref(source_length),
            ctypes.byref(status),
        )
        self._check(status.value, "clCreateProgramWithSource")
        if not program:
            raise OpenCLApiError("clCreateProgramWithSource returned a null program")

        selected_device = ctypes.c_void_p(device)
        build_status = self._cl_build_program(
            program,
            1,
            ctypes.byref(selected_device),
            options.encode("utf-8"),
            None,
            None,
        )
        if build_status != CL_SUCCESS:
            log_size = ctypes.c_size_t()
            self._cl_get_program_build_info(
                program, device, _CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size)
            )
            build_log = ""
            if log_size.value:
                log = ctypes.create_string_buffer(log_size.value)
                if (
                    self._cl_get_program_build_info(
                        program,
                        device,
                        _CL_PROGRAM_BUILD_LOG,
                        log_size.value,
                        log,
                        None,
                    )
                    == CL_SUCCESS
                ):
                    build_log = log.value.decode("utf-8", errors="replace")
            self.release_program(int(program))
            detail = f": {build_log.strip()}" if build_log.strip() else ""
            raise OpenCLApiError(
                f"clBuildProgram failed with OpenCL status {build_status}{detail}"
            )
        return int(program)

    def create_kernel(self, program: int, name: str) -> int:
        status = ctypes.c_int()
        kernel = self._cl_create_kernel(program, name.encode("ascii"), ctypes.byref(status))
        self._check(status.value, "clCreateKernel")
        if not kernel:
            raise OpenCLApiError("clCreateKernel returned a null kernel")
        return int(kernel)

    def set_kernel_buffer(self, kernel: int, index: int, buffer: int) -> None:
        value = ctypes.c_void_p(buffer)
        self._check(
            self._cl_set_kernel_arg(
                kernel, index, ctypes.sizeof(value), ctypes.byref(value)
            ),
            "clSetKernelArg(buffer)",
        )

    def set_kernel_int(self, kernel: int, index: int, value: int) -> None:
        argument = ctypes.c_int(value)
        self._check(
            self._cl_set_kernel_arg(
                kernel, index, ctypes.sizeof(argument), ctypes.byref(argument)
            ),
            "clSetKernelArg(int32)",
        )

    def set_kernel_double(self, kernel: int, index: int, value: float) -> None:
        argument = ctypes.c_double(value)
        self._check(
            self._cl_set_kernel_arg(
                kernel, index, ctypes.sizeof(argument), ctypes.byref(argument)
            ),
            "clSetKernelArg(double)",
        )

    def enqueue_1d(self, queue: int, kernel: int, global_size: int) -> None:
        work_size = ctypes.c_size_t(global_size)
        self._check(
            self._cl_enqueue_nd_range_kernel(
                queue, kernel, 1, None, ctypes.byref(work_size), None, 0, None, None
            ),
            "clEnqueueNDRangeKernel",
        )
        self._check(self._cl_finish(queue), "clFinish")

    def release_mem(self, handle: int) -> None:
        self._check(self._cl_release_mem_object(handle), "clReleaseMemObject")

    def release_kernel(self, handle: int) -> None:
        self._check(self._cl_release_kernel(handle), "clReleaseKernel")

    def release_program(self, handle: int) -> None:
        self._check(self._cl_release_program(handle), "clReleaseProgram")

    def release_command_queue(self, handle: int) -> None:
        self._check(self._cl_release_command_queue(handle), "clReleaseCommandQueue")

    def release_context(self, handle: int) -> None:
        self._check(self._cl_release_context(handle), "clReleaseContext")
