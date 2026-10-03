"""Explicit CUDA allocation and host/device transfer management."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kinera.backends.base import BackendError
from kinera.backends.cuda.buffer import CudaBuffer

if TYPE_CHECKING:
    from kinera.backends.cuda.bridge import CudaBridge
    from kinera.backends.cuda.runtime import CudaRuntime


class CudaMemoryManager:
    """Allocate owned device buffers and perform explicit float64 transfers."""

    def __init__(
        self,
        bridge: CudaBridge | None = None,
        *,
        runtime: CudaRuntime | None = None,
    ) -> None:
        if runtime is not None and bridge is not None and runtime.bridge is not bridge:
            raise ValueError("bridge e runtime devem compartilhar a mesma instância")
        if runtime is None:
            if bridge is None:
                from kinera.backends.cuda.bridge import CudaBridge

                bridge = CudaBridge()
            from kinera.backends.cuda.runtime import CudaRuntime

            runtime = CudaRuntime(bridge=bridge)
        if runtime is None:
            raise AssertionError("CUDA runtime initialization failed")
        self.runtime = runtime
        self.bridge = runtime.bridge

    def allocate(
        self,
        shape: Sequence[int],
        *,
        dtype: str = "float64",
        device_id: int = 0,
    ) -> CudaBuffer:
        normalized_shape = self._normalize_shape(shape)
        self._validate_dtype(dtype)
        self._validate_device(device_id)
        nbytes = math.prod(normalized_shape) * ctypes.sizeof(ctypes.c_double)
        pointer = self.bridge.allocate(device_id, nbytes)
        if not pointer:
            raise BackendError("CUDA allocator returned a null pointer")
        return CudaBuffer(
            self,
            pointer,
            device_id=device_id,
            shape=normalized_shape,
            dtype=dtype,
            nbytes=nbytes,
        )

    def copy_to_device(
        self,
        host_buffer: object,
        *,
        shape: Sequence[int] | None = None,
        dtype: str = "float64",
        device_id: int = 0,
    ) -> CudaBuffer:
        """Flatten and copy host numeric data into a newly owned CUDA buffer."""
        flat_values, inferred_shape = self._flatten_host_values(host_buffer)
        normalized_shape = (
            inferred_shape if shape is None else self._normalize_shape(shape)
        )
        if normalized_shape != inferred_shape:
            raise ValueError(
                f"shape {normalized_shape} does not match host data shape {inferred_shape}"
            )
        self._validate_dtype(dtype)
        buffer = self.allocate(normalized_shape, dtype=dtype, device_id=device_id)
        host_array = (ctypes.c_double * len(flat_values))(*flat_values)
        try:
            self.bridge.copy_to_device(
                device_id,
                buffer.pointer,
                ctypes.addressof(host_array),
                buffer.nbytes,
            )
        except Exception:
            self.release(buffer)
            raise
        return buffer

    def copy_to_host(
        self, device_buffer: CudaBuffer, *, device_id: int | None = None
    ) -> list[float] | list[list[float]]:
        """Explicitly copy a live CUDA buffer to a nested Python list."""
        self._validate_buffer(device_buffer)
        selected_device = device_buffer.device_id if device_id is None else device_id
        if selected_device != device_buffer.device_id:
            raise BackendError(
                f"CUDA buffer belongs to device {device_buffer.device_id}, "
                f"not device {selected_device}"
            )
        count = math.prod(device_buffer.shape)
        host_array = (ctypes.c_double * count)()
        self.bridge.copy_to_host(
            selected_device,
            ctypes.addressof(host_array),
            device_buffer.pointer,
            device_buffer.nbytes,
        )
        return self._reshape_host_values(list(host_array), device_buffer.shape)

    def release(self, device_buffer: CudaBuffer) -> None:
        """Free an owned allocation once; released buffers remain detectable."""
        if not isinstance(device_buffer, CudaBuffer):
            raise TypeError("device_buffer deve ser um CudaBuffer")
        if device_buffer.closed:
            return
        if device_buffer._manager is not self:
            raise BackendError("CUDA buffer is owned by a different memory manager")
        self.bridge.free(device_buffer.device_id, device_buffer.pointer)
        device_buffer._mark_closed()

    def _validate_device(self, device_id: int) -> None:
        self.runtime.require_device(device_id)

    @staticmethod
    def _validate_dtype(dtype: str) -> None:
        if dtype != "float64":
            raise ValueError("Phase 9 CUDA buffers support only dtype='float64'")

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError("shape deve ser uma sequência de dimensões inteiras")
        normalized: list[int] = []
        for axis, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"shape[{axis}] deve ser um inteiro")
            if dimension <= 0:
                raise ValueError(f"shape[{axis}] deve ser positivo")
            normalized.append(dimension)
        if not normalized:
            raise ValueError("shape não pode ser vazio")
        return tuple(normalized)

    @classmethod
    def _flatten_host_values(cls, value: object) -> tuple[list[float], tuple[int, ...]]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("host data cannot contain an empty dimension")
            flattened_children = [cls._flatten_host_values(item) for item in value]
            child_shape = flattened_children[0][1]
            if any(shape != child_shape for _, shape in flattened_children[1:]):
                raise ValueError("host data must be rectangular")
            flat = [number for child, _ in flattened_children for number in child]
            return flat, (len(value), *child_shape)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("CUDA host data must contain only numeric values")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CUDA host data must contain only finite values")
        return [number], ()

    @staticmethod
    def _reshape_host_values(
        values: list[float], shape: tuple[int, ...]
    ) -> list[float] | list[list[float]]:
        if len(shape) == 1:
            return values
        stride = math.prod(shape[1:])
        children = [
            CudaMemoryManager._reshape_host_values(
                values[index * stride : (index + 1) * stride], shape[1:]
            )
            for index in range(shape[0])
        ]
        return children

    def _validate_buffer(self, buffer: CudaBuffer) -> None:
        if not isinstance(buffer, CudaBuffer):
            raise TypeError("device_buffer deve ser um CudaBuffer")
        buffer._assert_open()
        if buffer._manager is not self:
            raise BackendError("CUDA buffer is owned by a different memory manager")
