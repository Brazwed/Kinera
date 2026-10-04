"""Lazy CUDA runtime/device discovery with explicit fail-closed operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kinera.backends.base import BackendUnavailableError

if TYPE_CHECKING:
    from kinera.backends.cuda.bridge import CudaBridge


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    """Stable Python representation of CUDA device capabilities."""

    index: int
    name: str
    compute_capability_major: int
    compute_capability_minor: int
    total_memory_bytes: int


class CudaRuntime:
    """Detect devices lazily; CPU imports never load or compile CUDA code."""

    def __init__(self, bridge: CudaBridge | None = None) -> None:
        if bridge is None:
            from kinera.backends.cuda.bridge import CudaBridge

            bridge = CudaBridge()
        self.bridge = bridge
        self._last_error: str | None = None

    def is_available(self) -> bool:
        return self.device_count() > 0

    def device_count(self) -> int:
        """Load/build the native bridge on explicit device-count queries."""
        try:
            count = self.bridge.device_count()
        except BackendUnavailableError as error:
            self._last_error = str(error)
            return 0
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            self._last_error = "CUDA bridge returned an invalid device count"
            return 0
        self._last_error = None
        return count

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def require_device(self, device_id: int = 0) -> None:
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError("device_id deve ser um inteiro")
        try:
            count = self.bridge.device_count()
        except BackendUnavailableError as error:
            self._last_error = str(error)
            raise BackendUnavailableError(
                f"CUDA runtime/device unavailable: {error}"
            ) from error
        if device_id < 0 or device_id >= count:
            raise BackendUnavailableError(
                f"CUDA device {device_id} is unavailable; detected {count} device(s)"
            )
        self._last_error = None

    def devices(self) -> list[CudaDeviceInfo]:
        count = self.device_count()
        return [self.device_info(index) for index in range(count)]

    def device_info(self, device_id: int) -> CudaDeviceInfo:
        self.require_device(device_id)
        info = self.bridge.device_info(device_id)
        if isinstance(info, CudaDeviceInfo):
            return info
        if not isinstance(info, dict):
            raise BackendUnavailableError("CUDA bridge returned invalid device information")
        try:
            return CudaDeviceInfo(
                index=device_id,
                name=str(info["name"]),
                compute_capability_major=int(info["major"]),
                compute_capability_minor=int(info["minor"]),
                total_memory_bytes=int(info["total_memory_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BackendUnavailableError(
                f"CUDA bridge returned incomplete device information: {error}"
            ) from error

    def set_device(self, device_id: int = 0) -> None:
        self.require_device(device_id)
        self.bridge.set_device(device_id)

    def synchronize(self, device_id: int = 0) -> None:
        self.require_device(device_id)
        self.bridge.synchronize(device_id)
