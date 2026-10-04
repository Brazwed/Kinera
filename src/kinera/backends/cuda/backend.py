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
from kinera.backends.cuda.buffer import CudaBuffer
from kinera.backends.cuda.memory import CudaMemoryManager
from kinera.backends.cuda.runtime import CudaRuntime


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
        if len(buffer.shape) == 1:
            return cast(list[float], values)
        if len(buffer.shape) == 2:
            return cast(list[list[float]], values)
        raise ValueError("Phase 9 public host export supports rank 1 or rank 2")

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
    def _require_buffer(value: BackendArray, name: str, *, rank: int) -> CudaBuffer:
        if not isinstance(value, CudaBuffer):
            raise TypeError(
                f"CUDA Linear.forward requires {name} as a CudaBuffer; "
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
            f"CUDA operation {operation} is not available in the current Phase 9 gate"
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

    def embedding_forward(self, *args, **kwargs):
        self._unavailable("embedding_forward")

    def embedding_backward(self, *args, **kwargs):
        self._unavailable("embedding_backward")

    def embedding_step(self, *args, **kwargs):
        self._unavailable("embedding_step")

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
