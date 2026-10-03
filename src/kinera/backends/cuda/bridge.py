"""Python/native CUDA bridge placeholder; no library is loaded or invoked."""

from typing import NoReturn

from kinera.backends.base import BackendUnavailableError


class CudaBridge:
    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise BackendUnavailableError(
            f"CUDA bridge scaffold only; {operation} is unavailable"
        )

    def load(self):
        self._unavailable("native module loading")

    def invoke(self, operation: str, *args, **kwargs):
        self._unavailable(f"native operation {operation}")
