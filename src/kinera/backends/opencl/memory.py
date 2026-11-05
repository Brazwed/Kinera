"""Explicit allocation and host/device transfer for OpenCL buffers."""

from __future__ import annotations

import ctypes
import math
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from kinera.backends.base import BackendError, HostArray
from kinera.backends.opencl.buffer import OpenCLBuffer

if TYPE_CHECKING:
    from kinera.backends.opencl.runtime import OpenCLRuntime


class OpenCLMemoryManager:
    """Manage typed allocations and their context/queue on one selected GPU."""

    def __init__(self, runtime: OpenCLRuntime, *, device_id: int = 0) -> None:
        self.runtime = runtime
        self.device_id = device_id
        self._device = None
        self.api = runtime.api
        self._context: int | None = None
        self._queue: int | None = None
        self._buffers: weakref.WeakSet[OpenCLBuffer] = weakref.WeakSet()
        self._closed = False

    @property
    def context(self) -> int:
        self._ensure_context()
        assert self._context is not None
        return self._context

    @property
    def queue(self) -> int:
        self._ensure_context()
        assert self._queue is not None
        return self._queue

    @property
    def device(self):
        if self._device is None:
            self._device = self.runtime.require_device(self.device_id)
            self.api = self.runtime.api
        return self._device

    def allocate(
        self,
        shape: Sequence[int],
        *,
        dtype: str = "float64",
        device_id: int | None = None,
    ) -> OpenCLBuffer:
        normalized_shape = self._normalize_shape(shape)
        self._validate_dtype(dtype)
        selected_device = self.device_id if device_id is None else device_id
        self._validate_device(selected_device)
        element_size = ctypes.sizeof(
            ctypes.c_double if dtype == "float64" else ctypes.c_int32
        )
        nbytes = math.prod(normalized_shape) * element_size
        device = self.device
        if nbytes > device.max_mem_alloc_size:
            raise BackendError(
                f"OpenCL allocation of {nbytes} bytes exceeds device limit "
                f"{device.max_mem_alloc_size} bytes"
            )
        if nbytes > device.global_mem_size:
            raise BackendError(
                f"OpenCL allocation of {nbytes} bytes exceeds device global memory "
                f"{device.global_mem_size} bytes"
            )
        self._ensure_context()
        assert self._context is not None
        handle = self.api.create_buffer(self._context, nbytes)
        buffer = OpenCLBuffer(
            self,
            handle,
            device_id=selected_device,
            shape=normalized_shape,
            dtype=dtype,
            nbytes=nbytes,
        )
        self._buffers.add(buffer)
        return buffer

    def copy_to_device(
        self,
        values: object,
        *,
        shape: Sequence[int] | None = None,
        dtype: str = "float64",
        device_id: int | None = None,
    ) -> OpenCLBuffer:
        self._validate_dtype(dtype)
        if dtype == "float64":
            flat_values, inferred_shape = self._flatten_float_values(values)
            element_type = ctypes.c_double
        else:
            flat_values, inferred_shape = self._flatten_int32_values(values)
            element_type = ctypes.c_int32
        normalized_shape = (
            inferred_shape if shape is None else self._normalize_shape(shape)
        )
        if normalized_shape != inferred_shape:
            raise ValueError(
                f"shape {normalized_shape} does not match host data shape "
                f"{inferred_shape}"
            )
        device_buffer = self.allocate(
            normalized_shape, dtype=dtype, device_id=device_id
        )
        try:
            host_array = (element_type * len(flat_values))(*flat_values)
            self.api.write_buffer(
                self.queue,
                device_buffer.handle,
                ctypes.addressof(host_array),
                device_buffer.nbytes,
            )
        except BaseException as error:
            try:
                self.release(device_buffer)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        return device_buffer

    def copy_zeros_to_device(
        self,
        shape: Sequence[int],
        *,
        dtype: str = "float64",
        device_id: int | None = None,
    ) -> OpenCLBuffer:
        normalized_shape = self._normalize_shape(shape)
        self._validate_dtype(dtype)
        if len(normalized_shape) not in {1, 2}:
            raise ValueError("OpenCL zero buffers support rank 1 or rank 2")
        element_type = ctypes.c_double if dtype == "float64" else ctypes.c_int32
        host_array = (element_type * math.prod(normalized_shape))()
        device_buffer = self.allocate(
            normalized_shape, dtype=dtype, device_id=device_id
        )
        try:
            self.api.write_buffer(
                self.queue,
                device_buffer.handle,
                ctypes.addressof(host_array),
                device_buffer.nbytes,
            )
        except Exception:
            self.release(device_buffer)
            raise
        return device_buffer

    def copy_to_host(
        self, device_buffer: OpenCLBuffer, *, device_id: int | None = None
    ) -> HostArray:
        self._validate_buffer(device_buffer)
        selected_device = device_buffer.device_id if device_id is None else device_id
        if selected_device != device_buffer.device_id:
            raise BackendError(
                f"OpenCL buffer belongs to device {device_buffer.device_id}, "
                f"not device {selected_device}"
            )
        if len(device_buffer.shape) not in {1, 2}:
            raise ValueError("OpenCL public host export supports rank 1 or rank 2")
        count = math.prod(device_buffer.shape)
        element_type = (
            ctypes.c_double if device_buffer.dtype == "float64" else ctypes.c_int32
        )
        host_array = (element_type * count)()
        self.api.read_buffer(
            self.queue,
            device_buffer.handle,
            ctypes.addressof(host_array),
            device_buffer.nbytes,
        )
        return self._reshape_host_values(list(host_array), device_buffer.shape)

    def release(self, device_buffer: OpenCLBuffer) -> None:
        if not isinstance(device_buffer, OpenCLBuffer):
            raise TypeError("device_buffer must be an OpenCLBuffer")
        if device_buffer.closed:
            return
        if device_buffer._manager is not self:
            raise BackendError("OpenCL buffer is owned by a different memory manager")
        self.api.release_mem(device_buffer.handle)
        device_buffer._mark_closed()
        self._buffers.discard(device_buffer)

    def close(self) -> None:
        """Release live allocations, then the queue and context."""
        if self._closed:
            return
        errors: list[Exception] = []
        for buffer in list(self._buffers):
            try:
                self.release(buffer)
            except Exception as error:
                errors.append(error)
        if self._queue is not None:
            try:
                self.api.release_command_queue(self._queue)
            except Exception as error:
                errors.append(error)
            self._queue = None
        if self._context is not None:
            try:
                self.api.release_context(self._context)
            except Exception as error:
                errors.append(error)
            self._context = None
        self._closed = True
        if errors:
            raise BackendError(f"failed to close OpenCL memory manager: {errors[0]}")

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError("shape must be a sequence of integer dimensions")
        normalized: list[int] = []
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"shape[{axis}] must be an integer")
            if dimension <= 0:
                raise ValueError(f"shape[{axis}] must be positive")
            normalized.append(dimension)
        if not normalized:
            raise ValueError("shape cannot be empty")
        return tuple(normalized)

    @staticmethod
    def _validate_dtype(dtype: str) -> None:
        if dtype not in {"float64", "int32"}:
            raise ValueError("OpenCL buffers support only dtype='float64' or 'int32'")

    def _validate_device(self, device_id: int) -> None:
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError("OpenCL device_id must be an integer")
        if device_id != self.device_id:
            raise BackendError(
                f"OpenCL memory manager is bound to GPU {self.device_id}, not {device_id}"
            )
        self.runtime.require_device(device_id)
        self._device = self.runtime.require_device(device_id)
        self.api = self.runtime.api

    @classmethod
    def _flatten_float_values(
        cls, value: object
    ) -> tuple[list[float], tuple[int, ...]]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("host data cannot contain an empty dimension")
            children = [cls._flatten_float_values(item) for item in value]
            child_shape = children[0][1]
            if any(shape != child_shape for _, shape in children[1:]):
                raise ValueError("host data must be rectangular")
            return (
                [number for child, _ in children for number in child],
                (len(value), *child_shape),
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("OpenCL float64 host data must contain numeric values")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("OpenCL float64 host data must contain finite values")
        return [number], ()

    @classmethod
    def _flatten_int32_values(
        cls, value: object
    ) -> tuple[list[int], tuple[int, ...]]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("host data cannot contain an empty dimension")
            children = [cls._flatten_int32_values(item) for item in value]
            child_shape = children[0][1]
            if any(shape != child_shape for _, shape in children[1:]):
                raise ValueError("host data must be rectangular")
            return (
                [number for child, _ in children for number in child],
                (len(value), *child_shape),
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("OpenCL int32 host data must contain integer values")
        if value < -(2**31) or value > (2**31) - 1:
            raise ValueError("OpenCL int32 host data is outside the signed 32-bit range")
        return [value], ()

    @staticmethod
    def _reshape_host_values(
        values: list[float] | list[int], shape: tuple[int, ...]
    ) -> HostArray:
        if len(shape) == 1:
            return cast(HostArray, values)
        stride = math.prod(shape[1:])
        children = [
            OpenCLMemoryManager._reshape_host_values(
                values[index * stride : (index + 1) * stride], shape[1:]
            )
            for index in range(shape[0])
        ]
        return cast(HostArray, children)

    def _validate_buffer(self, buffer: OpenCLBuffer) -> None:
        if not isinstance(buffer, OpenCLBuffer):
            raise TypeError("device_buffer must be an OpenCLBuffer")
        buffer._assert_open()
        if buffer._manager is not self:
            raise BackendError("OpenCL buffer is owned by a different memory manager")
        self._validate_dtype(buffer.dtype)
        self._validate_device(buffer.device_id)

    def _ensure_context(self) -> None:
        if self._closed:
            raise BackendError("OpenCL memory manager is closed")
        if self._context is not None and self._queue is not None:
            return
        device = self.device
        context = self.api.create_context(device.handle)
        try:
            queue = self.api.create_command_queue(context, device.handle)
        except Exception:
            self.api.release_context(context)
            raise
        self._context = context
        self._queue = queue
