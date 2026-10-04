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
        with self.assertRaisesRegex(TypeError, "CudaBuffer"):
            backend.linear_step(None, None, None, None, 0.1)

    def test_embedding_scaffold_fails_closed_with_phase_independent_message(self) -> None:
        from kinera.backends.cuda.backend import CUDABackend

        backend = CUDABackend()
        with self.assertRaisesRegex(BackendUnavailableError, "current CUDA gate"):
            backend.embedding_forward(None, None)

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

    def test_linear_forward_backward_and_step_are_the_real_kernels_in_this_gate(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src/kinera/backends/cuda/kernels"
        for filename, symbol in (
            ("linear_forward.cu", "kinera_cuda_linear_forward_f64"),
            ("linear_backward.cu", "kinera_cuda_linear_backward_f64"),
            ("linear_step.cu", "kinera_cuda_linear_step_f64"),
        ):
            with self.subTest(kernel=filename):
                source = source_root / filename
                self.assertTrue(source.is_file())
                text = source.read_text(encoding="utf-8")
                self.assertIn("__global__", text)
                self.assertIn(symbol, text)

        placeholders = (
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
