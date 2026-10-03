"""Lazy CUDA runtime/ABI discovery tests with no GPU dependency."""

import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path

from kinera.backends.base import BackendError, BackendUnavailableError
from kinera.backends.cuda.runtime import CudaRuntime


class FakeRuntimeBridge:
    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.probes = 0
        self.requirements = 0
        self.selected_devices: list[int] = []
        self.synchronized: list[int] = []
        self.load_error: str | None = None

    def probe_device_count(self) -> int:
        self.probes += 1
        if self.load_error:
            raise BackendUnavailableError(self.load_error)
        return self.count

    def device_count(self) -> int:
        self.requirements += 1
        if self.load_error:
            raise BackendUnavailableError(self.load_error)
        return self.count

    def device_info(self, device_id: int):
        if device_id < 0 or device_id >= self.count:
            raise BackendUnavailableError("invalid device")
        return {
            "name": f"Fake CUDA {device_id}",
            "major": 8,
            "minor": 0,
            "total_memory_bytes": 4096,
        }

    def set_device(self, device_id: int) -> None:
        self.selected_devices.append(device_id)

    def synchronize(self, device_id: int) -> None:
        self.synchronized.append(device_id)


class FakeFunction:
    def __init__(self, result: object = 0, callback=None) -> None:
        self.result = result
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self.callback is not None:
            return self.callback(*args)
        return self.result


class FakeLibrary:
    def __init__(self, abi_version: int = 1, omit: str | None = None) -> None:
        self.kinera_cuda_abi_version = FakeFunction(abi_version)
        self.kinera_cuda_error_string = FakeFunction(b"fake CUDA error")
        self.kinera_cuda_get_device_count = FakeFunction(
            callback=lambda count: self._set_count(count)
        )
        self.kinera_cuda_get_device_info = FakeFunction(0)
        self.kinera_cuda_set_device = FakeFunction(0)
        self.kinera_cuda_synchronize = FakeFunction(0)
        self.kinera_cuda_allocate = FakeFunction(0)
        self.kinera_cuda_free = FakeFunction(0)
        self.kinera_cuda_copy_to_device = FakeFunction(0)
        self.kinera_cuda_copy_to_host = FakeFunction(0)
        if omit:
            delattr(self, omit)

    @staticmethod
    def _set_count(pointer) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0


class CudaRuntimeTests(unittest.TestCase):
    def test_runtime_construction_is_lazy_and_reports_device_details(self) -> None:
        bridge = FakeRuntimeBridge(count=2)
        runtime = CudaRuntime(bridge=bridge)

        self.assertEqual(bridge.requirements, 0)
        self.assertTrue(runtime.is_available())
        self.assertEqual(runtime.device_count(), 2)
        self.assertEqual(bridge.requirements, 2)
        device = runtime.device_info(1)
        self.assertEqual(device.index, 1)
        self.assertEqual(device.name, "Fake CUDA 1")
        self.assertEqual((device.compute_capability_major, device.compute_capability_minor), (8, 0))
        self.assertEqual(device.total_memory_bytes, 4096)

    def test_missing_runtime_is_unavailable_without_cpu_fallback(self) -> None:
        bridge = FakeRuntimeBridge()
        bridge.load_error = "nvcc and native CUDA library are unavailable"
        runtime = CudaRuntime(bridge=bridge)

        self.assertFalse(runtime.is_available())
        self.assertEqual(runtime.device_count(), 0)
        with self.assertRaisesRegex(BackendUnavailableError, "nvcc.*unavailable"):
            runtime.require_device(0)

    def test_invalid_device_and_invalid_id_type_are_rejected(self) -> None:
        runtime = CudaRuntime(bridge=FakeRuntimeBridge(count=1))

        with self.assertRaisesRegex(BackendUnavailableError, "device 2"):
            runtime.require_device(2)
        with self.assertRaises(TypeError):
            runtime.require_device(True)

    def test_device_selection_and_synchronization_use_explicit_id(self) -> None:
        bridge = FakeRuntimeBridge(count=2)
        runtime = CudaRuntime(bridge=bridge)

        runtime.set_device(1)
        runtime.synchronize(1)

        self.assertEqual(bridge.selected_devices, [1])
        self.assertEqual(bridge.synchronized, [1])


class CudaBridgeAbiTests(unittest.TestCase):
    def _load_with_fake_library(self, library: FakeLibrary):
        from kinera.backends.cuda.bridge import CudaBridge

        temporary = tempfile.TemporaryDirectory()
        library_path = Path(temporary.name) / "libkinera_cuda.so"
        library_path.write_bytes(b"not a real shared object; fake loader is injected")
        bridge = CudaBridge(
            library_path=library_path,
            loader=lambda _: library,
        )
        return temporary, bridge

    def test_bridge_load_is_lazy_and_binds_versioned_abi(self) -> None:
        from kinera.backends.cuda.bridge import CudaBridge

        library = FakeLibrary()
        temporary, bridge = self._load_with_fake_library(library)
        with temporary:
            self.assertFalse(bridge.is_loaded)
            bridge.load()
            self.assertTrue(bridge.is_loaded)
            self.assertEqual(bridge.abi_version, 1)
            self.assertEqual(bridge.device_count(), 1)
            self.assertIsNotNone(library.kinera_cuda_get_device_count.argtypes)

    def test_bridge_rejects_incompatible_abi(self) -> None:
        from kinera.backends.cuda.bridge import CudaBridge

        library = FakeLibrary(abi_version=2)
        temporary, bridge = self._load_with_fake_library(library)
        with temporary:
            with self.assertRaisesRegex(BackendUnavailableError, "ABI"):
                bridge.load()
            self.assertFalse(bridge.is_loaded)

    def test_bridge_rejects_missing_required_symbol(self) -> None:
        from kinera.backends.cuda.bridge import CudaBridge

        library = FakeLibrary(omit="kinera_cuda_allocate")
        temporary, bridge = self._load_with_fake_library(library)
        with temporary:
            with self.assertRaisesRegex(BackendUnavailableError, "symbol"):
                bridge.load()

    def test_explicit_load_builds_packaged_shared_library_on_demand(self) -> None:
        from kinera.backends.cuda.bridge import CudaBridge

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            fake_nvcc = root / "nvcc"
            fake_nvcc.write_text("#!/bin/sh\\n", encoding="utf-8")
            fake_nvcc.chmod(0o755)
            commands: list[list[str]] = []

            def runner(command, **kwargs):
                commands.append(list(command))
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_bytes(b"test-only shared library")
                return subprocess.CompletedProcess(command, 0, "", "")

            bridge = CudaBridge(
                nvcc_path=fake_nvcc,
                cache_dir=root / "cache",
                loader=lambda _: FakeLibrary(),
                runner=runner,
            )
            self.assertFalse(bridge.is_loaded)
            self.assertEqual(commands, [])

            bridge.load()

            self.assertTrue(bridge.is_loaded)
            self.assertTrue(bridge.built_library)
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertIn("--shared", command)
            self.assertIn("--cudart=shared", command)
            self.assertIn("--fmad=false", command)
            source_paths = [Path(argument) for argument in command if argument.endswith(".cu")]
            self.assertTrue(
                any(path.parent.name == "native" and path.name == "runtime.cu" for path in source_paths)
            )


if __name__ == "__main__":
    unittest.main()
