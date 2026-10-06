"""Backend discovery and dependency-injection helpers."""

from __future__ import annotations

from .base import (
    Backend,
    BackendError,
    BackendUnavailableError,
    SupportsBufferZeros,
    create_zero_buffer,
)


def get_backend(name: str = "cpu") -> Backend:
    """Create a backend by name; implementations are imported lazily."""
    if not isinstance(name, str):
        raise TypeError("backend name deve ser uma string")
    normalized = name.casefold()
    if normalized == "cpu":
        from .cpu.backend import CPUBackend

        return CPUBackend()
    if normalized == "cuda":
        from .cuda.backend import CUDABackend

        return CUDABackend()
    raise ValueError(f"backend desconhecido: {name}")


def resolve_backend(backend: Backend | None = None) -> Backend:
    """Validate an injected backend or select the CPU reference backend."""
    if backend is None:
        return get_backend("cpu")
    if not isinstance(backend, Backend):
        raise TypeError("backend deve implementar o contrato Backend")
    return backend


__all__ = [
    "Backend",
    "BackendError",
    "BackendUnavailableError",
    "SupportsBufferZeros",
    "create_zero_buffer",
    "get_backend",
    "resolve_backend",
]
