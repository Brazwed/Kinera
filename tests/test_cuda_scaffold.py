"""CUDA scaffolding must fail closed until real kernels exist."""

from pathlib import Path
import tempfile
import unittest

from kinera.backends import Backend, BackendUnavailableError, get_backend


class CudaBackendScaffoldTests(unittest.TestCase):
    def test_cuda_backend_is_contractful_and_fail_closed_without_device(self) -> None:
        from kinera.backends.cuda.backend import CUDABackend
        from kinera.backends.cuda.bridge import CudaBridge
        from kinera.backends.cuda.runtime import CudaRuntime

        class UnavailableBridge(CudaBridge):
            def device_count(self) -> int:
                raise BackendUnavailableError("no native CUDA runtime in this test")

        backend = CUDABackend(runtime=CudaRuntime(bridge=UnavailableBridge()))
        self.assertIsInstance(backend, Backend)
        self.assertEqual(backend.name, "cuda")
        self.assertFalse(backend.is_available)
        with self.assertRaisesRegex(BackendUnavailableError, "no native CUDA runtime"):
            backend.buffer_from_host([1.0], shape=(1,))
        with self.assertRaises(BackendUnavailableError):
            backend.linear_backward(None, None, None)

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

    def test_linear_forward_is_the_only_real_cuda_kernel_in_this_gate(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src/kinera/backends/cuda/kernels"
        forward_source = source_root / "linear_forward.cu"
        self.assertTrue(forward_source.is_file())
        forward_text = forward_source.read_text(encoding="utf-8")
        self.assertIn("__global__", forward_text)
        self.assertIn("kinera_cuda_linear_forward_f64", forward_text)

        placeholders = (
            "linear_backward.cu",
            "embedding_forward.cu",
            "embedding_backward.cu",
            "activations.cu",
            "losses.cu",
        )
        for name in placeholders:
            with self.subTest(kernel=name):
                source = source_root / name
                self.assertTrue(source.is_file())
                self.assertNotIn("__global__", source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
