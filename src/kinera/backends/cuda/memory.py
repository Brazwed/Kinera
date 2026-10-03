"""CUDA memory-management placeholder with no host/device allocation."""

from typing import NoReturn

from kinera.backends.base import BackendUnavailableError


class CudaMemoryManager:
    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise BackendUnavailableError(
            f"CUDA memory scaffold only; {operation} is unavailable"
        )

    def allocate(self, size: int):
        self._unavailable("allocate")

    def copy_to_device(self, host_buffer):
        self._unavailable("copy_to_device")

    def copy_to_host(self, device_buffer):
        self._unavailable("copy_to_host")

    def release(self, device_buffer) -> None:
        self._unavailable("release")
