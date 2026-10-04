"""CUDA backend: explicit buffers and the gated Linear forward/backward/step kernels."""

from __future__ import annotations

import math
from typing import NoReturn, cast

from kinera.backends.base import (
    BackendArray,
    BackendBuffer,
    BackendError,
    BackendUnavailableError,
    HostArray,
)
from kinera.backends.cuda.bridge import KINERA_CUDA_INVALID_NUMERIC_STATUS
from kinera.backends.cuda.buffer import CudaBuffer
from kinera.backends.cuda.memory import CudaMemoryManager
from kinera.backends.cuda.runtime import CudaRuntime

# Keep CUDA pooling aligned with the active interpreter's built-in float sum.
_PYTHON_FLOAT_SUM_COMPENSATED = sum((1e16, 1.0, -1e16)) == 1.0
_CUDA_ABI_INT_MAX = 2**31 - 1
_CUDA_GRID_MAX_BLOCKS = 2**31 - 1
_CUDA_GRID_THREADS_PER_BLOCK = 128


class CUDABackend:
    """Fail-closed optional CUDA backend with explicit device-array transfers."""

    name = "cuda"

    def __init__(
        self,
        *,
        runtime: CudaRuntime | None = None,
        memory_manager: CudaMemoryManager | None = None,
    ) -> None:
        self.runtime = CudaRuntime() if runtime is None else runtime
        if memory_manager is not None and memory_manager.runtime is not self.runtime:
            raise ValueError("memory_manager and runtime must share the same instance")
        self.memory = (
            CudaMemoryManager(runtime=self.runtime)
            if memory_manager is None
            else memory_manager
        )
        self.bridge = self.memory.bridge

    @property
    def is_available(self) -> bool:
        return self.runtime.is_available()

    def buffer_from_host(
        self,
        values: HostArray,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> CudaBuffer:
        """Explicitly allocate a device buffer and upload host values."""
        return self.memory.copy_to_device(
            values, shape=shape, dtype=dtype, device_id=0
        )

    def buffer_to_host(self, buffer: BackendArray) -> HostArray:
        """Explicitly download a CUDA buffer to independent Python lists."""
        if not isinstance(buffer, CudaBuffer):
            raise TypeError("CUDA buffer_to_host requires a CudaBuffer")
        values = self.memory.copy_to_host(buffer)
        if len(buffer.shape) in {1, 2}:
            return cast(HostArray, values)
        raise ValueError("CUDA public host export supports rank 1 or rank 2")

    def release_buffer(self, buffer: CudaBuffer) -> None:
        """Release a buffer explicitly through its owning memory manager."""
        self.memory.release(buffer)

    def linear_forward(
        self, weights: BackendArray, bias: BackendArray, inputs: BackendArray
    ) -> CudaBuffer:
        """Execute row-major y = W x + b without materializing Python lists."""
        weights_buffer = self._require_buffer(weights, "weights", rank=2)
        bias_buffer = self._require_buffer(bias, "bias", rank=1)
        inputs_buffer = self._require_buffer(inputs, "inputs", rank=1)
        if weights_buffer.shape[0] != bias_buffer.shape[0]:
            raise ValueError("weights rows must match bias length")
        if weights_buffer.shape[1] != inputs_buffer.shape[0]:
            raise ValueError("weights columns must match input length")
        device_ids = {
            weights_buffer.device_id,
            bias_buffer.device_id,
            inputs_buffer.device_id,
        }
        if len(device_ids) != 1:
            raise BackendError("Linear.forward buffers must reside on the same CUDA device")
        device_id = inputs_buffer.device_id
        self.runtime.require_device(device_id)
        output = self.memory.allocate(
            (weights_buffer.shape[0],), dtype="float64", device_id=device_id
        )
        try:
            self.bridge.linear_forward_f64(
                device_id,
                weights_buffer.pointer,
                bias_buffer.pointer,
                inputs_buffer.pointer,
                output.pointer,
                inputs_buffer.shape[0],
                weights_buffer.shape[0],
            )
        except Exception:
            self.memory.release(output)
            raise
        return output

    @staticmethod
    def _require_buffer(
        value: BackendArray,
        name: str,
        *,
        rank: int,
        operation: str = "Linear.forward",
    ) -> CudaBuffer:
        if not isinstance(value, CudaBuffer):
            raise TypeError(
                f"CUDA {operation} requires {name} as a CudaBuffer; "
                "call buffer_from_host explicitly for host data"
            )
        value._assert_open()
        if value.device != "cuda":
            raise BackendError(f"{name} must reside on CUDA")
        if value.dtype != "float64":
            raise ValueError(f"{name} must use dtype float64")
        if len(value.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}")
        return value

    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise BackendUnavailableError(
            f"CUDA operation {operation} is not available in the current CUDA gate"
        )

    def linear_backward(
        self, weights: BackendArray, inputs: BackendArray, grad_output: BackendArray
    ) -> tuple[CudaBuffer, CudaBuffer, CudaBuffer]:
        """Compute Linear parameter/input gradients without leaving device memory."""
        weights_buffer = self._require_buffer(weights, "weights", rank=2)
        inputs_buffer = self._require_buffer(inputs, "inputs", rank=1)
        gradient_buffer = self._require_buffer(grad_output, "grad_output", rank=1)
        output_size, input_size = weights_buffer.shape
        if inputs_buffer.shape != (input_size,):
            raise ValueError("weights columns must match input length")
        if gradient_buffer.shape != (output_size,):
            raise ValueError("grad_output must match the number of Linear outputs")
        device_ids = {
            weights_buffer.device_id,
            inputs_buffer.device_id,
            gradient_buffer.device_id,
        }
        if len(device_ids) != 1:
            raise BackendError("Linear.backward buffers must reside on the same CUDA device")
        device_id = inputs_buffer.device_id
        self.runtime.require_device(device_id)

        allocated: list[CudaBuffer] = []
        try:
            grad_weights = self.memory.allocate(
                (output_size, input_size), dtype="float64", device_id=device_id
            )
            allocated.append(grad_weights)
            grad_bias = self.memory.allocate(
                (output_size,), dtype="float64", device_id=device_id
            )
            allocated.append(grad_bias)
            grad_inputs = self.memory.allocate(
                (input_size,), dtype="float64", device_id=device_id
            )
            allocated.append(grad_inputs)
            self.bridge.linear_backward_f64(
                device_id,
                weights_buffer.pointer,
                inputs_buffer.pointer,
                gradient_buffer.pointer,
                grad_weights.pointer,
                grad_bias.pointer,
                grad_inputs.pointer,
                input_size,
                output_size,
            )
        except Exception:
            for buffer in reversed(allocated):
                try:
                    self.memory.release(buffer)
                except Exception:
                    pass
            raise
        return grad_weights, grad_bias, grad_inputs

    def linear_step(
        self,
        weights: BackendArray,
        bias: BackendArray,
        grad_weights: BackendArray,
        grad_bias: BackendArray,
        learning_rate: float,
    ) -> tuple[CudaBuffer, CudaBuffer]:
        """Update resident Linear parameters in place with no host transfers."""
        weights_buffer = self._require_buffer(weights, "weights", rank=2)
        bias_buffer = self._require_buffer(bias, "bias", rank=1)
        grad_weights_buffer = self._require_buffer(grad_weights, "grad_weights", rank=2)
        grad_bias_buffer = self._require_buffer(grad_bias, "grad_bias", rank=1)
        self.memory._validate_buffer(weights_buffer)
        self.memory._validate_buffer(bias_buffer)
        self.memory._validate_buffer(grad_weights_buffer)
        self.memory._validate_buffer(grad_bias_buffer)

        output_size, input_size = weights_buffer.shape
        if bias_buffer.shape != (output_size,):
            raise ValueError("bias must match the number of Linear outputs")
        if grad_weights_buffer.shape != (output_size, input_size):
            raise ValueError("grad_weights must match weights shape")
        if grad_bias_buffer.shape != (output_size,):
            raise ValueError("grad_bias must match the number of Linear outputs")
        device_ids = {
            weights_buffer.device_id,
            bias_buffer.device_id,
            grad_weights_buffer.device_id,
            grad_bias_buffer.device_id,
        }
        if len(device_ids) != 1:
            raise BackendError("Linear.step buffers must reside on the same CUDA device")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be numeric")
        if not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite")
        if learning_rate < 0.0:
            raise ValueError("learning_rate cannot be negative")

        device_id = weights_buffer.device_id
        self.runtime.require_device(device_id)
        rate = float(learning_rate)
        if rate == 0.0:
            return weights_buffer, bias_buffer

        self.bridge.linear_step_f64(
            device_id,
            weights_buffer.pointer,
            bias_buffer.pointer,
            grad_weights_buffer.pointer,
            grad_bias_buffer.pointer,
            rate,
            input_size,
            output_size,
        )
        return weights_buffer, bias_buffer

    def hybrid_representation_forward(
        self,
        char_vectors: BackendArray,
        feature_vectors: BackendArray,
        syllable_vectors: BackendArray,
    ) -> CudaBuffer:
        """Concatenate resident vector channels without materializing host lists."""
        char_buffer = self._require_buffer(
            char_vectors, "char_vectors", rank=2, operation="HybridRepresentation.forward"
        )
        feature_buffer = self._require_buffer(
            feature_vectors, "feature_vectors", rank=2, operation="HybridRepresentation.forward"
        )
        syllable_buffer = self._require_buffer(
            syllable_vectors, "syllable_vectors", rank=2, operation="HybridRepresentation.forward"
        )
        for buffer in (char_buffer, feature_buffer, syllable_buffer):
            self.memory._validate_buffer(buffer)

        context_size, char_dim = char_buffer.shape
        feature_context, feature_dim = feature_buffer.shape
        syllable_context, syllable_dim = syllable_buffer.shape
        if feature_context != context_size or syllable_context != context_size:
            raise ValueError("hybrid representation matrices must have equal context sizes")
        for name, dimension in (
            ("context_size", context_size),
            ("char_dim", char_dim),
            ("feature_dim", feature_dim),
            ("syllable_dim", syllable_dim),
        ):
            if dimension > _CUDA_ABI_INT_MAX:
                raise ValueError(
                    f"hybrid representation {name} exceeds the CUDA ABI limit"
                )
        device_ids = {
            char_buffer.device_id,
            feature_buffer.device_id,
            syllable_buffer.device_id,
        }
        if len(device_ids) != 1:
            raise BackendError(
                "HybridRepresentation.forward buffers must reside on the same CUDA device"
            )
        output_dim = char_dim + feature_dim + syllable_dim
        if output_dim > 2**31 - 1:
            raise ValueError("hybrid representation output dimension exceeds the CUDA ABI limit")
        output_element_count = context_size * output_dim
        grid_block_count = (
            output_element_count + _CUDA_GRID_THREADS_PER_BLOCK - 1
        ) // _CUDA_GRID_THREADS_PER_BLOCK
        if grid_block_count > _CUDA_GRID_MAX_BLOCKS:
            raise ValueError("hybrid representation output exceeds the CUDA grid limit")
        device_id = char_buffer.device_id
        self.runtime.require_device(device_id)
        output = self.memory.allocate(
            (context_size, output_dim), dtype="float64", device_id=device_id
        )
        try:
            self.bridge.hybrid_representation_forward_f64(
                device_id,
                char_buffer.pointer,
                feature_buffer.pointer,
                syllable_buffer.pointer,
                output.pointer,
                context_size,
                char_dim,
                feature_dim,
                syllable_dim,
            )
        except Exception:
            self.memory.release(output)
            raise
        return output

    def mean_pool_forward(self, vectors: BackendArray) -> CudaBuffer:
        """Average a resident matrix using CPU-compatible sequential sums."""
        vector_buffer = self._require_buffer(
            vectors, "vectors", rank=2, operation="mean_pool_forward"
        )
        self.memory._validate_buffer(vector_buffer)
        context_size, feature_dim = vector_buffer.shape
        for name, dimension in (
            ("context_size", context_size),
            ("feature_dim", feature_dim),
        ):
            if dimension > _CUDA_ABI_INT_MAX:
                raise ValueError(f"mean_pool_forward {name} exceeds the CUDA ABI limit")
        device_id = vector_buffer.device_id
        self.runtime.require_device(device_id)
        pooled = self.memory.allocate(
            (feature_dim,), dtype="float64", device_id=device_id
        )
        try:
            self.bridge.mean_pool_forward_f64(
                device_id,
                vector_buffer.pointer,
                pooled.pointer,
                context_size,
                feature_dim,
                int(_PYTHON_FLOAT_SUM_COMPENSATED),
            )
        except Exception:
            self.memory.release(pooled)
            raise
        return pooled

    def cross_entropy_forward_buffer(
        self, logits: BackendArray, target_id: int
    ) -> float:
        """Compute CE on resident logits and export only its scalar result."""
        logits_buffer = self._require_buffer(
            logits, "logits", rank=1, operation="cross_entropy_forward"
        )
        self.memory._validate_buffer(logits_buffer)
        logits_count = logits_buffer.shape[0]
        if logits_count > _CUDA_ABI_INT_MAX:
            raise ValueError("cross_entropy input exceeds the CUDA ABI limit")
        if isinstance(target_id, bool) or not isinstance(target_id, int):
            raise TypeError("target_id must be an integer")
        if not 0 <= target_id < logits_count:
            raise ValueError("target_id outside vocabulary")
        device_id = logits_buffer.device_id
        self.runtime.require_device(device_id)
        loss_buffer: CudaBuffer | None = None
        status_buffer: CudaBuffer | None = None
        try:
            loss_buffer = self.memory.allocate((1,), dtype="float64", device_id=device_id)
            status_buffer = self.memory.allocate((1,), dtype="int32", device_id=device_id)
            self.bridge.cross_entropy_forward_f64(
                device_id,
                logits_buffer.pointer,
                loss_buffer.pointer,
                status_buffer.pointer,
                logits_count,
                target_id,
                int(_PYTHON_FLOAT_SUM_COMPENSATED),
            )
            self._check_numeric_status(status_buffer, "cross_entropy_forward")
            host_loss = self.memory.copy_to_host(loss_buffer)
            if not isinstance(host_loss, list) or len(host_loss) != 1:
                raise BackendError("CUDA cross_entropy_forward returned an invalid scalar")
            result = cast(list[float], host_loss)[0]
            if not math.isfinite(result):
                raise ValueError("CUDA cross_entropy_forward returned a non-finite loss")
            return result
        finally:
            if status_buffer is not None:
                self.memory.release(status_buffer)
            if loss_buffer is not None:
                self.memory.release(loss_buffer)

    def cross_entropy_gradient_buffer(
        self,
        logits: BackendArray,
        target_id: int,
        scale: float,
    ) -> CudaBuffer:
        """Compute a weighted resident cross-entropy gradient without logits D2H."""
        logits_buffer = self._require_buffer(
            logits, "logits", rank=1, operation="cross_entropy_gradient"
        )
        self.memory._validate_buffer(logits_buffer)
        logits_count = logits_buffer.shape[0]
        if logits_count > _CUDA_ABI_INT_MAX:
            raise ValueError("cross_entropy_gradient input exceeds the CUDA ABI limit")
        if isinstance(target_id, bool) or not isinstance(target_id, int):
            raise TypeError("target_id must be an integer")
        if not 0 <= target_id < logits_count:
            raise ValueError("target_id outside vocabulary")
        self._validate_gradient_scale(scale)
        device_id = logits_buffer.device_id
        self.runtime.require_device(device_id)
        gradient: CudaBuffer | None = None
        status_buffer: CudaBuffer | None = None
        try:
            gradient = self.memory.allocate(
                (logits_count,), dtype="float64", device_id=device_id
            )
            status_buffer = self.memory.allocate(
                (1,), dtype="int32", device_id=device_id
            )
            self.bridge.cross_entropy_gradient_f64(
                device_id,
                logits_buffer.pointer,
                gradient.pointer,
                status_buffer.pointer,
                logits_count,
                target_id,
                float(scale),
                int(_PYTHON_FLOAT_SUM_COMPENSATED),
            )
            self._check_numeric_status(status_buffer, "cross_entropy_gradient")
            return gradient
        except Exception:
            if gradient is not None:
                self.memory.release(gradient)
            raise
        finally:
            if status_buffer is not None:
                self.memory.release(status_buffer)

    @staticmethod
    def _validate_gradient_scale(scale: float) -> None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("gradient scale must be numeric")
        if not math.isfinite(scale):
            raise ValueError("gradient scale must be finite")
        if scale < 0.0:
            raise ValueError("gradient scale cannot be negative")

    def _check_numeric_status(self, status_buffer: CudaBuffer, operation: str) -> None:
        status_values = self.memory.copy_to_host(status_buffer)
        if not isinstance(status_values, list) or len(status_values) != 1:
            raise BackendError(f"CUDA {operation} returned an invalid status buffer")
        status = cast(list[int], status_values)[0]
        if status == KINERA_CUDA_INVALID_NUMERIC_STATUS:
            raise ValueError(f"CUDA {operation} requires finite input and output values")
        if status != 0:
            raise BackendError(f"CUDA {operation} returned invalid numeric status {status}")

    @staticmethod
    def _validate_binary_target(target: float) -> float:
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("target must be numeric")
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target must be between 0 and 1")
        return float(target)

    def binary_cross_entropy_with_logits_forward_buffer(
        self, logit: BackendArray, target: float
    ) -> float:
        """Compute resident BCE-with-logits and export only its scalar result."""
        logit_buffer = self._require_buffer(
            logit, "logit", rank=1, operation="binary_cross_entropy_with_logits_forward"
        )
        self.memory._validate_buffer(logit_buffer)
        if logit_buffer.shape != (1,):
            raise ValueError("binary_cross_entropy_with_logits requires one logit")
        target_value = self._validate_binary_target(target)
        device_id = logit_buffer.device_id
        self.runtime.require_device(device_id)
        loss_buffer: CudaBuffer | None = None
        status_buffer: CudaBuffer | None = None
        try:
            loss_buffer = self.memory.allocate((1,), dtype="float64", device_id=device_id)
            status_buffer = self.memory.allocate((1,), dtype="int32", device_id=device_id)
            self.bridge.binary_cross_entropy_with_logits_forward_f64(
                device_id,
                logit_buffer.pointer,
                loss_buffer.pointer,
                status_buffer.pointer,
                target_value,
            )
            self._check_numeric_status(
                status_buffer, "binary_cross_entropy_with_logits_forward"
            )
            host_loss = self.memory.copy_to_host(loss_buffer)
            if not isinstance(host_loss, list) or len(host_loss) != 1:
                raise BackendError("CUDA BCE forward returned an invalid scalar")
            result = cast(list[float], host_loss)[0]
            if not math.isfinite(result):
                raise ValueError("CUDA BCE forward returned a non-finite loss")
            return result
        finally:
            if status_buffer is not None:
                self.memory.release(status_buffer)
            if loss_buffer is not None:
                self.memory.release(loss_buffer)

    def binary_cross_entropy_with_logits_gradient_buffer(
        self, logit: BackendArray, target: float, scale: float
    ) -> CudaBuffer:
        """Compute weighted resident BCE gradient without downloading logits."""
        logit_buffer = self._require_buffer(
            logit, "logit", rank=1, operation="binary_cross_entropy_with_logits_gradient"
        )
        self.memory._validate_buffer(logit_buffer)
        if logit_buffer.shape != (1,):
            raise ValueError("binary_cross_entropy_with_logits requires one logit")
        target_value = self._validate_binary_target(target)
        self._validate_gradient_scale(scale)
        device_id = logit_buffer.device_id
        self.runtime.require_device(device_id)
        gradient: CudaBuffer | None = None
        status_buffer: CudaBuffer | None = None
        try:
            gradient = self.memory.allocate((1,), dtype="float64", device_id=device_id)
            status_buffer = self.memory.allocate((1,), dtype="int32", device_id=device_id)
            self.bridge.binary_cross_entropy_with_logits_gradient_f64(
                device_id,
                logit_buffer.pointer,
                gradient.pointer,
                status_buffer.pointer,
                target_value,
                float(scale),
            )
            self._check_numeric_status(
                status_buffer, "binary_cross_entropy_with_logits_gradient"
            )
            return gradient
        except Exception:
            if gradient is not None:
                self.memory.release(gradient)
            raise
        finally:
            if status_buffer is not None:
                self.memory.release(status_buffer)

    @staticmethod
    def _validate_grid_count(element_count: int, operation: str) -> None:
        if element_count <= 0:
            raise ValueError(f"{operation} requires a positive element count")
        blocks = (element_count + _CUDA_GRID_THREADS_PER_BLOCK - 1) // _CUDA_GRID_THREADS_PER_BLOCK
        if blocks > _CUDA_GRID_MAX_BLOCKS:
            raise ValueError(f"{operation} output exceeds the CUDA grid limit")

    def sum_head_input_gradients(
        self,
        char_gradient: BackendArray,
        syllable_gradient: BackendArray,
        boundary_gradient: BackendArray,
    ) -> CudaBuffer:
        """Sum head input gradients in CPU semantic order without D2H."""
        buffers = (
            self._require_buffer(
                char_gradient, "char_gradient", rank=1,
                operation="sum_head_input_gradients",
            ),
            self._require_buffer(
                syllable_gradient, "syllable_gradient", rank=1,
                operation="sum_head_input_gradients",
            ),
            self._require_buffer(
                boundary_gradient, "boundary_gradient", rank=1,
                operation="sum_head_input_gradients",
            ),
        )
        for buffer in buffers:
            self.memory._validate_buffer(buffer)
        feature_dim = buffers[0].shape[0]
        if any(buffer.shape != (feature_dim,) for buffer in buffers[1:]):
            raise ValueError("head input gradients must have equal shapes")
        if feature_dim > _CUDA_ABI_INT_MAX:
            raise ValueError("sum_head_input_gradients dimension exceeds the CUDA ABI limit")
        if len({buffer.device_id for buffer in buffers}) != 1:
            raise BackendError("head input gradients must reside on the same CUDA device")
        self._validate_grid_count(feature_dim, "sum_head_input_gradients")
        device_id = buffers[0].device_id
        self.runtime.require_device(device_id)
        output = self.memory.allocate((feature_dim,), dtype="float64", device_id=device_id)
        try:
            self.bridge.sum_head_input_gradients_f64(
                device_id,
                buffers[0].pointer,
                buffers[1].pointer,
                buffers[2].pointer,
                output.pointer,
                feature_dim,
                int(_PYTHON_FLOAT_SUM_COMPENSATED),
            )
        except Exception:
            self.memory.release(output)
            raise
        return output

    def mean_pool_backward(
        self, grad_pooled: BackendArray, *, context_size: int
    ) -> CudaBuffer:
        """Repeat the pooled gradient across context rows with CPU division order."""
        gradient_buffer = self._require_buffer(
            grad_pooled, "grad_pooled", rank=1, operation="mean_pool_backward"
        )
        self.memory._validate_buffer(gradient_buffer)
        if isinstance(context_size, bool) or not isinstance(context_size, int):
            raise TypeError("context_size must be an integer")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        feature_dim = gradient_buffer.shape[0]
        for name, dimension in (("context_size", context_size), ("feature_dim", feature_dim)):
            if dimension > _CUDA_ABI_INT_MAX:
                raise ValueError(f"mean_pool_backward {name} exceeds the CUDA ABI limit")
        element_count = context_size * feature_dim
        self._validate_grid_count(element_count, "mean_pool_backward")
        device_id = gradient_buffer.device_id
        self.runtime.require_device(device_id)
        output = self.memory.allocate(
            (context_size, feature_dim), dtype="float64", device_id=device_id
        )
        try:
            self.bridge.mean_pool_backward_f64(
                device_id,
                gradient_buffer.pointer,
                output.pointer,
                context_size,
                feature_dim,
            )
        except Exception:
            self.memory.release(output)
            raise
        return output

    def hybrid_representation_backward(
        self,
        grad_output: BackendArray,
        *,
        char_dim: int,
        feature_dim: int,
        syllable_dim: int,
    ) -> tuple[CudaBuffer, CudaBuffer, CudaBuffer]:
        """Split resident concatenated gradients into the three input channels."""
        gradient_buffer = self._require_buffer(
            grad_output, "grad_output", rank=2,
            operation="hybrid_representation_backward",
        )
        self.memory._validate_buffer(gradient_buffer)
        context_size, output_dim = gradient_buffer.shape
        for name, dimension in (
            ("context_size", context_size),
            ("char_dim", char_dim),
            ("feature_dim", feature_dim),
            ("syllable_dim", syllable_dim),
        ):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(f"{name} must be an integer")
            if dimension <= 0:
                raise ValueError(f"{name} must be positive")
            if dimension > _CUDA_ABI_INT_MAX:
                raise ValueError(f"hybrid backward {name} exceeds the CUDA ABI limit")
        expected_output_dim = char_dim + feature_dim + syllable_dim
        if output_dim != expected_output_dim:
            raise ValueError("grad_output columns must equal the hybrid output dimension")
        if expected_output_dim > _CUDA_ABI_INT_MAX:
            raise ValueError("hybrid backward output dimension exceeds the CUDA ABI limit")
        self._validate_grid_count(
            context_size * expected_output_dim, "hybrid_representation_backward"
        )
        device_id = gradient_buffer.device_id
        self.runtime.require_device(device_id)
        allocated: list[CudaBuffer] = []
        try:
            char_gradient = self.memory.allocate(
                (context_size, char_dim), dtype="float64", device_id=device_id
            )
            allocated.append(char_gradient)
            feature_gradient = self.memory.allocate(
                (context_size, feature_dim), dtype="float64", device_id=device_id
            )
            allocated.append(feature_gradient)
            syllable_gradient = self.memory.allocate(
                (context_size, syllable_dim), dtype="float64", device_id=device_id
            )
            allocated.append(syllable_gradient)
            self.bridge.hybrid_representation_backward_f64(
                device_id,
                gradient_buffer.pointer,
                char_gradient.pointer,
                feature_gradient.pointer,
                syllable_gradient.pointer,
                context_size,
                char_dim,
                feature_dim,
                syllable_dim,
            )
        except Exception:
            for buffer in reversed(allocated):
                self.memory.release(buffer)
            raise
        return char_gradient, feature_gradient, syllable_gradient

    def argmax(self, logits: BackendArray) -> int:
        """Find the first finite maximum and copy back only its int32 index."""
        logits_buffer = self._require_buffer(
            logits, "logits", rank=1, operation="argmax"
        )
        self.memory._validate_buffer(logits_buffer)
        logits_count = logits_buffer.shape[0]
        if logits_count > 2**31 - 1:
            raise ValueError("argmax input length exceeds the CUDA ABI limit")
        device_id = logits_buffer.device_id
        self.runtime.require_device(device_id)
        output_id = self.memory.allocate((1,), dtype="int32", device_id=device_id)
        try:
            self.bridge.argmax_f64(
                device_id,
                logits_buffer.pointer,
                output_id.pointer,
                logits_count,
            )
            host_id = self.memory.copy_to_host(output_id)
        finally:
            self.memory.release(output_id)

        if not isinstance(host_id, list) or len(host_id) != 1:
            raise BackendError("CUDA argmax did not return one index")
        predicted_id = cast(list[int], host_id)[0]
        if predicted_id < 0:
            raise ValueError(f"output[{-predicted_id - 1}] deve ser finito")
        if predicted_id >= logits_count:
            raise BackendError("CUDA argmax returned an out-of-range index")
        return predicted_id

    def embedding_forward(
        self, weights: BackendArray, ids: list[int] | BackendBuffer
    ) -> CudaBuffer | list[list[float]]:
        """Copy row-major weight rows selected by an owned int32 ID buffer."""
        weights_buffer = self._require_buffer(
            weights, "weights", rank=2, operation="Embedding.forward"
        )
        self.memory._validate_buffer(weights_buffer)
        vocab_size, embedding_dim = weights_buffer.shape
        device_id = weights_buffer.device_id
        self.runtime.require_device(device_id)

        if isinstance(ids, list):
            if ids:
                raise TypeError(
                    "CUDA Embedding.forward requires IDs in an int32 CudaBuffer; "
                    "call buffer_from_host explicitly after layer validation"
                )
            return []
        if not isinstance(ids, CudaBuffer):
            raise TypeError("CUDA Embedding.forward requires IDs as a CudaBuffer")
        ids._assert_open()
        if ids.dtype != "int32":
            raise ValueError("Embedding IDs must use dtype int32")
        if len(ids.shape) != 1:
            raise ValueError("Embedding IDs must have rank 1")
        self.memory._validate_buffer(ids)
        if ids.device_id != device_id:
            raise BackendError("Embedding.forward buffers must reside on the same CUDA device")

        ids_count = ids.shape[0]
        output = self.memory.allocate(
            (ids_count, embedding_dim), dtype="float64", device_id=device_id
        )
        try:
            self.bridge.embedding_forward_f64(
                device_id,
                weights_buffer.pointer,
                ids.pointer,
                output.pointer,
                vocab_size,
                embedding_dim,
                ids_count,
            )
        except Exception:
            self.memory.release(output)
            raise
        return output

    def embedding_backward(
        self,
        grad_weights: BackendArray,
        ids: list[int] | BackendBuffer,
        grad_output: BackendArray,
    ) -> CudaBuffer | list[list[float]]:
        """Accumulate Embedding gradients deterministically into the resident table."""
        gradients_buffer = self._require_buffer(
            grad_weights, "grad_weights", rank=2, operation="Embedding.backward"
        )
        self.memory._validate_buffer(gradients_buffer)
        vocab_size, embedding_dim = gradients_buffer.shape
        device_id = gradients_buffer.device_id
        self.runtime.require_device(device_id)

        if isinstance(ids, list):
            if ids:
                raise TypeError(
                    "CUDA Embedding.backward requires IDs in an int32 CudaBuffer"
                )
            if not isinstance(grad_output, list) or grad_output:
                raise ValueError("empty Embedding IDs require empty grad_output")
            return gradients_buffer
        if not isinstance(ids, CudaBuffer):
            raise TypeError("CUDA Embedding.backward requires IDs as a CudaBuffer")
        ids._assert_open()
        if ids.dtype != "int32":
            raise ValueError("Embedding IDs must use dtype int32")
        if len(ids.shape) != 1:
            raise ValueError("Embedding IDs must have rank 1")
        self.memory._validate_buffer(ids)

        gradient_buffer = self._require_buffer(
            grad_output, "grad_output", rank=2, operation="Embedding.backward"
        )
        self.memory._validate_buffer(gradient_buffer)
        ids_count = ids.shape[0]
        if gradient_buffer.shape != (ids_count, embedding_dim):
            raise ValueError(
                "grad_output must have shape (len(ids), embedding_dim)"
            )
        device_ids = {
            gradients_buffer.device_id,
            ids.device_id,
            gradient_buffer.device_id,
        }
        if len(device_ids) != 1:
            raise BackendError(
                "Embedding.backward buffers must reside on the same CUDA device"
            )
        self.bridge.embedding_backward_f64(
            device_id,
            ids.pointer,
            gradient_buffer.pointer,
            gradients_buffer.pointer,
            vocab_size,
            embedding_dim,
            ids_count,
        )
        return gradients_buffer

    def embedding_step(
        self,
        weights: BackendArray,
        grad_weights: BackendArray,
        learning_rate: float,
    ) -> CudaBuffer | list[list[float]]:
        """Apply gradient descent to the resident Embedding table in place."""
        weights_buffer = self._require_buffer(
            weights, "weights", rank=2, operation="Embedding.step"
        )
        grad_weights_buffer = self._require_buffer(
            grad_weights, "grad_weights", rank=2, operation="Embedding.step"
        )
        self.memory._validate_buffer(weights_buffer)
        self.memory._validate_buffer(grad_weights_buffer)
        if grad_weights_buffer.shape != weights_buffer.shape:
            raise ValueError("grad_weights must match weights shape")
        if weights_buffer.device_id != grad_weights_buffer.device_id:
            raise BackendError("Embedding.step buffers must reside on the same CUDA device")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be numeric")
        if not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite")
        if learning_rate < 0.0:
            raise ValueError("learning_rate cannot be negative")

        device_id = weights_buffer.device_id
        self.runtime.require_device(device_id)
        rate = float(learning_rate)
        if rate == 0.0:
            return weights_buffer
        vocab_size, embedding_dim = weights_buffer.shape
        self.bridge.embedding_step_f64(
            device_id,
            weights_buffer.pointer,
            grad_weights_buffer.pointer,
            rate,
            vocab_size,
            embedding_dim,
        )
        return weights_buffer

    def neuron_forward(self, *args, **kwargs):
        self._unavailable("neuron_forward")

    def neuron_backward(self, *args, **kwargs):
        self._unavailable("neuron_backward")

    def neuron_step(self, *args, **kwargs):
        self._unavailable("neuron_step")

    def activation(self, *args, **kwargs):
        self._unavailable("activation")

    def activation_derivative(self, *args, **kwargs):
        self._unavailable("activation_derivative")

    def softmax(self, *args, **kwargs):
        self._unavailable("softmax")

    def cross_entropy(self, *args, **kwargs):
        self._unavailable("cross_entropy")

    def cross_entropy_gradient(self, *args, **kwargs):
        self._unavailable("cross_entropy_gradient")

    def binary_cross_entropy_with_logits(self, *args, **kwargs):
        self._unavailable("binary_cross_entropy_with_logits")

    def binary_cross_entropy_with_logits_gradient(self, *args, **kwargs):
        self._unavailable("binary_cross_entropy_with_logits_gradient")
