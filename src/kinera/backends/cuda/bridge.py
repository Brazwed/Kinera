"""Lazy ctypes bridge for the optional native CUDA library."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Any, Callable

from kinera.backends.base import BackendUnavailableError


ABI_VERSION = 1
_BUILD_FLAGS = (
    "--shared",
    "--cudart=shared",
    "--std=c++17",
    "--fmad=false",
    "-Xcompiler=-fPIC",
)


class CudaBridge:
    """Loads or optionally builds a narrow, versioned C ABI on explicit use."""

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        nvcc_path: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        loader: Callable[[str], object] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._explicit_library_path = (
            None if library_path is None else Path(library_path).expanduser()
        )
        self._nvcc_path = None if nvcc_path is None else os.fspath(nvcc_path)
        configured_cache = os.environ.get("KINERA_CUDA_CACHE")
        default_cache = Path.home() / ".cache" / "kinera" / "cuda"
        self._cache_dir = Path(cache_dir or configured_cache or default_cache).expanduser()
        self._loader = loader or ctypes.CDLL
        self._runner = runner or subprocess.run
        self._library: object | None = None
        self._functions: dict[str, Any] = {}
        self._abi_version: int | None = None
        self._loaded_path: Path | None = None
        self._built = False

    @property
    def is_loaded(self) -> bool:
        return self._library is not None

    @property
    def abi_version(self) -> int | None:
        return self._abi_version

    @property
    def loaded_path(self) -> Path | None:
        return self._loaded_path

    @property
    def built_library(self) -> bool:
        return self._built

    def load(self, *, build_if_missing: bool = True) -> None:
        """Load and validate the ABI; build from packaged sources only if asked."""
        if self._library is not None:
            return
        library_path = self._resolve_library_path(build_if_missing=build_if_missing)
        try:
            library = self._loader(str(library_path))
        except (OSError, ValueError) as error:
            raise BackendUnavailableError(
                f"could not load CUDA native library at {library_path}: {error}"
            ) from error

        functions = self._bind_functions(library)
        try:
            abi_version = int(functions["kinera_cuda_abi_version"]())
        except Exception as error:
            raise BackendUnavailableError(
                f"could not read CUDA native ABI version: {error}"
            ) from error
        if abi_version != ABI_VERSION:
            raise BackendUnavailableError(
                f"CUDA native ABI mismatch: Python expects {ABI_VERSION}, "
                f"library reports {abi_version}"
            )

        self._library = library
        self._functions = functions
        self._abi_version = abi_version
        self._loaded_path = library_path

    def probe_device_count(self) -> int:
        """Probe an existing native library without compiling from CPU-only CI."""
        self.load(build_if_missing=False)
        return self.device_count()

    def device_count(self) -> int:
        self.load()
        count = ctypes.c_int()
        self._check(
            "device discovery",
            self._functions["kinera_cuda_get_device_count"](ctypes.byref(count)),
        )
        return count.value

    def device_info(self, device_id: int) -> dict[str, int | str]:
        self.load()
        name = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        memory_bytes = ctypes.c_ulonglong()
        self._check(
            "device information",
            self._functions["kinera_cuda_get_device_info"](
                device_id,
                name,
                len(name),
                ctypes.byref(major),
                ctypes.byref(minor),
                ctypes.byref(memory_bytes),
            ),
        )
        return {
            "name": name.value.decode("utf-8", errors="replace"),
            "major": major.value,
            "minor": minor.value,
            "total_memory_bytes": memory_bytes.value,
        }

    def set_device(self, device_id: int) -> None:
        self.load()
        self._check(
            "device selection",
            self._functions["kinera_cuda_set_device"](device_id),
        )

    def synchronize(self, device_id: int) -> None:
        self.load()
        self._check(
            "device synchronization",
            self._functions["kinera_cuda_synchronize"](device_id),
        )

    def allocate(self, device_id: int, nbytes: int) -> int:
        self.load()
        pointer = ctypes.c_void_p()
        self._check(
            "device allocation",
            self._functions["kinera_cuda_allocate"](
                device_id, nbytes, ctypes.byref(pointer)
            ),
        )
        if not pointer.value:
            raise BackendUnavailableError("CUDA native allocator returned a null pointer")
        return int(pointer.value)

    def free(self, device_id: int, pointer: int) -> None:
        self.load()
        self._check(
            "device release",
            self._functions["kinera_cuda_free"](device_id, pointer),
        )

    def copy_to_device(
        self, device_id: int, destination: int, source: int, nbytes: int
    ) -> None:
        self.load()
        self._check(
            "host-to-device copy",
            self._functions["kinera_cuda_copy_to_device"](
                device_id, destination, source, nbytes
            ),
        )

    def copy_to_host(
        self, device_id: int, destination: int, source: int, nbytes: int
    ) -> None:
        self.load()
        self._check(
            "device-to-host copy",
            self._functions["kinera_cuda_copy_to_host"](
                device_id, destination, source, nbytes
            ),
        )

    def _check(self, operation: str, result: object) -> None:
        status = int(result)
        if status == 0:
            return
        try:
            message = self._functions["kinera_cuda_error_string"](status)
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
        except Exception:
            message = "unknown CUDA runtime error"
        raise BackendUnavailableError(
            f"CUDA {operation} failed with status {status}: {message}"
        )

    def _bind_functions(self, library: object) -> dict[str, Any]:
        signature = {
            "kinera_cuda_abi_version": ([], ctypes.c_int),
            "kinera_cuda_error_string": ([ctypes.c_int], ctypes.c_char_p),
            "kinera_cuda_get_device_count": (
                [ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
            "kinera_cuda_get_device_info": (
                [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_char),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_ulonglong),
                ],
                ctypes.c_int,
            ),
            "kinera_cuda_set_device": ([ctypes.c_int], ctypes.c_int),
            "kinera_cuda_synchronize": ([ctypes.c_int], ctypes.c_int),
            "kinera_cuda_allocate": (
                [ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int,
            ),
            "kinera_cuda_free": ([ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            "kinera_cuda_copy_to_device": (
                [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
            "kinera_cuda_copy_to_host": (
                [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
                ctypes.c_int,
            ),
        }
        functions: dict[str, Any] = {}
        for symbol, (arguments, result) in signature.items():
            try:
                function = getattr(library, symbol)
            except AttributeError as error:
                raise BackendUnavailableError(
                    f"CUDA native library is missing required symbol {symbol}"
                ) from error
            try:
                function.argtypes = arguments
                function.restype = result
            except (AttributeError, TypeError) as error:
                raise BackendUnavailableError(
                    f"could not bind CUDA native symbol {symbol}: {error}"
                ) from error
            functions[symbol] = function
        return functions

    def _resolve_library_path(self, *, build_if_missing: bool) -> Path:
        if self._explicit_library_path is not None:
            return self._require_file(self._explicit_library_path, "explicit CUDA library")

        environment_path = os.environ.get("KINERA_CUDA_LIBRARY")
        if environment_path:
            return self._require_file(Path(environment_path).expanduser(), "KINERA_CUDA_LIBRARY")

        source_root = Path(__file__).resolve().parent
        digest = self._source_digest(source_root)
        suffix = ".dll" if os.name == "nt" else ".so"
        build_dir = self._cache_dir / digest
        output_path = build_dir / f"libkinera_cuda{suffix}"
        if output_path.is_file():
            return output_path
        if not build_if_missing:
            raise BackendUnavailableError(
                "CUDA native library is not cached; runtime probe did not invoke nvcc"
            )
        if sys.platform != "linux":
            raise BackendUnavailableError(
                "on-demand CUDA native builds currently require Linux; set "
                "KINERA_CUDA_LIBRARY to use a prebuilt compatible library"
            )

        compiler = self._resolve_nvcc()
        build_dir.mkdir(parents=True, exist_ok=True)
        sources = self._source_files(source_root)
        temporary_path = build_dir / f".kinera_cuda.{os.getpid()}.{uuid.uuid4().hex}.so"
        command = [
            compiler,
            *_BUILD_FLAGS,
            "-o",
            str(temporary_path),
            *(str(path) for path in sources),
        ]
        try:
            result = self._runner(
                command,
                cwd=source_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            temporary_path.unlink(missing_ok=True)
            raise BackendUnavailableError(f"CUDA native build failed to start: {error}") from error
        if result.returncode != 0 or not temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip()[:4000] or "nvcc produced no diagnostic output"
            raise BackendUnavailableError(f"CUDA native build failed: {output}")
        os.replace(temporary_path, output_path)
        self._built = True
        return output_path

    @staticmethod
    def _require_file(path: Path, label: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise BackendUnavailableError(f"{label} does not exist or is not a file: {resolved}")
        return resolved

    @staticmethod
    def _source_files(source_root: Path) -> list[Path]:
        sources = sorted((source_root / "native").glob("*.cu"))
        sources.extend(sorted((source_root / "kernels").rglob("*.cu")))
        if not sources:
            raise BackendUnavailableError("no CUDA .cu sources were packaged")
        return sources

    @classmethod
    def _source_digest(cls, source_root: Path) -> str:
        digest = hashlib.sha256()
        digest.update(f"kinera-cuda-abi-{ABI_VERSION}\n".encode())
        digest.update("\n".join(_BUILD_FLAGS).encode())
        files = cls._source_files(source_root)
        for path in files:
            digest.update(path.relative_to(source_root).as_posix().encode())
            digest.update(path.read_bytes())
        for path in sorted((source_root / "kernels").rglob("*.cuh")):
            digest.update(path.relative_to(source_root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _resolve_nvcc(self) -> str:
        configured = self._nvcc_path or os.environ.get("NVCC")
        compiler = shutil.which(configured) if configured else shutil.which("nvcc")
        if compiler is None:
            raise BackendUnavailableError(
                "CUDA was explicitly requested, but nvcc is unavailable and no "
                "compatible cached/native library was found"
            )
        return compiler
