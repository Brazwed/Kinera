"""Owned, explicit CUDA device-buffer handle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kinera.backends.base import BackendError, HostArray

if TYPE_CHECKING:
    from kinera.backends.cuda.memory import CudaMemoryManager


class BufferReleasedError(BackendError):
    """Raised when an operation tries to use a released CUDA allocation."""


class CudaBuffer:
    """A contiguous float64 or int32 allocation owned by one CUDA memory manager.

    Device buffers deliberately do not implement Python's sequence protocol.
    Host/device movement is available only through explicit copy methods.
    """

    __slots__ = (
        "_manager",
        "_pointer",
        "_device_id",
        "_shape",
        "_dtype",
        "_nbytes",
        "_closed",
        "__weakref__",
    )

    def __init__(
        self,
        manager: CudaMemoryManager,
        pointer: int,
        *,
        device_id: int,
        shape: tuple[int, ...],
        dtype: str,
        nbytes: int,
    ) -> None:
        self._manager = manager
        self._pointer = int(pointer)
        self._device_id = device_id
        self._shape = tuple(shape)
        self._dtype = dtype
        self._nbytes = nbytes
        self._closed = False

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def layout(self) -> str:
        return "C"

    @property
    def owns_memory(self) -> bool:
        return True

    @property
    def ownership(self) -> str:
        return "owned"

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pointer(self) -> int:
        self._assert_open()
        return self._pointer

    def _assert_open(self) -> None:
        if self._closed:
            raise BufferReleasedError("CUDA buffer has been released/closed")

    def _mark_closed(self) -> None:
        self._closed = True
        self._pointer = 0

    def to_host(self) -> HostArray:
        """Explicitly copy this allocation to a Python list structure."""
        self._assert_open()
        return self._manager.copy_to_host(self)

    def close(self) -> None:
        """Release the owned allocation; repeated calls are safe."""
        self._manager.release(self)

    def __enter__(self) -> CudaBuffer:
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __copy__(self) -> CudaBuffer:
        raise TypeError("CudaBuffer cannot be shallow-copied; copy data explicitly")

    def __deepcopy__(self, memo) -> CudaBuffer:
        raise TypeError("CudaBuffer cannot be deep-copied; copy data explicitly")

    def __del__(self) -> None:
        try:
            if not self._closed:
                self.close()
        except Exception:
            # Destructors must not leak errors during interpreter shutdown.
            pass

    def __repr__(self) -> str:
        return (
            f"CudaBuffer(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"device_id={self.device_id}, closed={self.closed})"
        )
