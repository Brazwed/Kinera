"""Optional CUDA scaffold; this module does not load CUDA libraries."""

from .backend import CUDABackend
from .memory import CudaMemoryManager
from .runtime import CudaRuntime
from .bridge import CudaBridge

__all__ = ["CUDABackend", "CudaBridge", "CudaMemoryManager", "CudaRuntime"]
