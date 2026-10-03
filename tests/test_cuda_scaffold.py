"""CUDA scaffolding must fail closed until real kernels exist."""

from pathlib import Path
import tempfile
import unittest

from kinera.backends import Backend, BackendUnavailableError, get_backend


class CudaBackendScaffoldTests(unittest.TestCase):
    def test_cuda_backend_implements_contract_but_does_not_run_operations(self) -> None:
        try:
            backend = get_backend("cuda")
        except ModuleNotFoundError as error:
            self.fail(f"CUDA scaffold is missing: {error}")

        self.assertIsInstance(backend, Backend)
        self.assertEqual(backend.name, "cuda")
        self.assertFalse(backend.is_available)
        with self.assertRaisesRegex(BackendUnavailableError, "scaffold only"):
            backend.linear_forward([], [], [])

    def test_missing_runtime_memory_and_library_fail_closed(self) -> None:
        from kinera.backends.cuda.bridge import CudaBridge
        from kinera.backends.cuda.memory import CudaMemoryManager
        from kinera.backends.cuda.runtime import CudaRuntime

        class UnavailableBridge(CudaBridge):
            def probe_device_count(self) -> int:
                raise BackendUnavailableError("no native CUDA runtime in this test")

            def device_count(self) -> int:
                raise BackendUnavailableError("no native CUDA runtime in this test")

        unavailable = UnavailableBridge()
        runtime = CudaRuntime(bridge=unavailable)
        self.assertFalse(runtime.is_available())
        self.assertEqual(runtime.device_count(), 0)
        with self.assertRaises(BackendUnavailableError):
            runtime.synchronize()

        memory = CudaMemoryManager(bridge=unavailable)
        with self.assertRaises(BackendUnavailableError):
            memory.allocate((16,))

        with tempfile.TemporaryDirectory() as temporary_dir:
            bridge = CudaBridge(library_path=Path(temporary_dir) / "missing.so")
            with self.assertRaises(BackendUnavailableError):
                bridge.load()

    def test_cuda_source_files_are_placeholders_not_production_kernels(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src/kinera/backends/cuda/kernels"
        expected = (
            "linear_forward.cu",
            "linear_backward.cu",
            "embedding_forward.cu",
            "embedding_backward.cu",
            "activations.cu",
            "losses.cu",
        )
        for name in expected:
            with self.subTest(kernel=name):
                source = source_root / name
                self.assertTrue(source.is_file())
                self.assertNotIn("__global__", source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
