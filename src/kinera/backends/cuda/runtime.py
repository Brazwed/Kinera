"""CUDA runtime placeholder; no CUDA library is imported in this phase."""

from kinera.backends.base import BackendUnavailableError


class CudaRuntime:
    def is_available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        raise BackendUnavailableError("CUDA runtime scaffold only; synchronization is unavailable")
