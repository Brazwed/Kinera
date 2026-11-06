"""Optional OpenCL backend with gated Linear and Embedding operations."""

from __future__ import annotations

import ctypes
import math
from importlib import resources
from typing import NoReturn

from kinera.backends.base import (
    BackendArray,
    BackendBuffer,
    BackendError,
    BackendUnavailableError,
    HostArray,
)
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.backends.opencl.memory import OpenCLMemoryManager
from kinera.backends.opencl.runtime import OpenCLRuntime

_PYTHON_FLOAT_SUM_COMPENSATED = sum((1e16, 1.0, -1e16)) == 1.0
_OPENCL_INT_MAX = 2**31 - 1


class OpenCLBackend:
    """Fail-closed backend with gated Linear and Embedding.forward operations."""

    name = "opencl"

    def __init__(
        self,
        *,
        runtime: OpenCLRuntime | None = None,
        memory_manager: OpenCLMemoryManager | None = None,
        device_id: int = 0,
    ) -> None:
        self.runtime = OpenCLRuntime() if runtime is None else runtime
        if memory_manager is not None and memory_manager.runtime is not self.runtime:
            raise ValueError("memory_manager and runtime must share the same instance")
        self.memory = (
            OpenCLMemoryManager(self.runtime, device_id=device_id)
            if memory_manager is None
            else memory_manager
        )
        self.device_id = device_id
        self._programs: dict[str, int] = {}
        self._closed = False

    @property
    def is_available(self) -> bool:
        return self.runtime.is_available

    def buffer_zeros(
        self, *, shape: tuple[int, ...], dtype: str = "float64"
    ) -> OpenCLBuffer:
        return self.memory.copy_zeros_to_device(
            shape, dtype=dtype, device_id=self.device_id
        )

    def buffer_from_host(
        self,
        values: HostArray,
        *,
        shape: tuple[int, ...],
        dtype: str = "float64",
    ) -> OpenCLBuffer:
        return self.memory.copy_to_device(
            values, shape=shape, dtype=dtype, device_id=self.device_id
        )

    def buffer_to_host(self, buffer: BackendArray) -> HostArray:
        if not isinstance(buffer, OpenCLBuffer):
            raise TypeError("OpenCL buffer_to_host requires an OpenCLBuffer")
        return self.memory.copy_to_host(buffer)

    def release_buffer(self, buffer: OpenCLBuffer) -> None:
        self.memory.release(buffer)

    def linear_forward(
        self, weights: BackendArray, bias: BackendArray, inputs: BackendArray
    ) -> OpenCLBuffer:
        """Execute row-major y = W x + b on the selected GPU."""
        self._ensure_open()
        weights_buffer = self._require_buffer(weights, "weights", rank=2)
        bias_buffer = self._require_buffer(bias, "bias", rank=1)
        inputs_buffer = self._require_buffer(inputs, "inputs", rank=1)
        if weights_buffer.shape[0] != bias_buffer.shape[0]:
            raise ValueError("weights rows must match bias length")
        if weights_buffer.shape[1] != inputs_buffer.shape[0]:
            raise ValueError("weights columns must match input length")
        buffers = (weights_buffer, bias_buffer, inputs_buffer)
        if any(buffer._manager is not self.memory for buffer in buffers):
            raise BackendError("Linear.forward buffers must belong to this OpenCL backend")
        if any(buffer.device_id != self.device_id for buffer in buffers):
            raise BackendError("Linear.forward buffers must reside on the selected OpenCL GPU")
        output_size, input_size = weights_buffer.shape
        if output_size > _OPENCL_INT_MAX or input_size > _OPENCL_INT_MAX:
            raise ValueError("OpenCL Linear dimensions must fit in a signed 32-bit integer")
        self.runtime.require_device(self.device_id, require_fp64=True)

        output = self.memory.allocate(
            (output_size,), dtype="float64", device_id=self.device_id
        )
        kernel: int | None = None
        try:
            program = self._get_program("linear_forward.cl")
            kernel = self.runtime.api.create_kernel(program, "kinera_linear_forward_f64")
            for index, buffer in enumerate(
                (weights_buffer, bias_buffer, inputs_buffer, output)
            ):
                self.runtime.api.set_kernel_buffer(kernel, index, buffer.handle)
            self.runtime.api.set_kernel_int(kernel, 4, input_size)
            self.runtime.api.set_kernel_int(
                kernel, 5, int(_PYTHON_FLOAT_SUM_COMPENSATED)
            )
            self.runtime.api.enqueue_1d(self.memory.queue, kernel, output_size)
        except Exception:
            if kernel is not None:
                try:
                    self.runtime.api.release_kernel(kernel)
                except Exception:
                    pass
            self.memory.release(output)
            raise
        if kernel is not None:
            try:
                self.runtime.api.release_kernel(kernel)
            except Exception:
                self.memory.release(output)
                raise
        return output

    def linear_backward(
        self, weights: BackendArray, inputs: BackendArray, grad_output: BackendArray
    ) -> tuple[OpenCLBuffer, OpenCLBuffer, OpenCLBuffer]:
        """Compute parameter and input gradients without leaving device memory."""
        self._ensure_open()
        weights_buffer = self._require_buffer(
            weights, "weights", rank=2, operation="Linear.backward"
        )
        inputs_buffer = self._require_buffer(
            inputs, "inputs", rank=1, operation="Linear.backward"
        )
        grad_output_buffer = self._require_buffer(
            grad_output, "grad_output", rank=1, operation="Linear.backward"
        )
        output_size, input_size = weights_buffer.shape
        if inputs_buffer.shape != (input_size,):
            raise ValueError("weights columns must match input length")
        if grad_output_buffer.shape != (output_size,):
            raise ValueError("grad_output must match the number of Linear outputs")
        buffers = (weights_buffer, inputs_buffer, grad_output_buffer)
        if any(buffer._manager is not self.memory for buffer in buffers):
            raise BackendError(
                "Linear.backward buffers must belong to this OpenCL backend"
            )
        if any(buffer.device_id != self.device_id for buffer in buffers):
            raise BackendError(
                "Linear.backward buffers must reside on the selected OpenCL GPU"
            )
        if output_size > _OPENCL_INT_MAX or input_size > _OPENCL_INT_MAX:
            raise ValueError("OpenCL Linear dimensions must fit in a signed 32-bit integer")
        self.runtime.require_device(self.device_id, require_fp64=True)

        allocated: list[OpenCLBuffer] = []
        kernel: int | None = None
        try:
            grad_weights = self.memory.allocate(
                (output_size, input_size), dtype="float64", device_id=self.device_id
            )
            allocated.append(grad_weights)
            grad_bias = self.memory.allocate(
                (output_size,), dtype="float64", device_id=self.device_id
            )
            allocated.append(grad_bias)
            grad_inputs = self.memory.allocate(
                (input_size,), dtype="float64", device_id=self.device_id
            )
            allocated.append(grad_inputs)

            program = self._get_program("linear_backward.cl")
            kernel = self.runtime.api.create_kernel(
                program, "kinera_linear_backward_f64"
            )
            for index, buffer in enumerate(
                (
                    weights_buffer,
                    inputs_buffer,
                    grad_output_buffer,
                    grad_weights,
                    grad_bias,
                    grad_inputs,
                )
            ):
                self.runtime.api.set_kernel_buffer(kernel, index, buffer.handle)
            self.runtime.api.set_kernel_int(kernel, 6, output_size)
            self.runtime.api.set_kernel_int(kernel, 7, input_size)
            self.runtime.api.set_kernel_int(
                kernel, 8, int(_PYTHON_FLOAT_SUM_COMPENSATED)
            )
            self.runtime.api.enqueue_1d(
                self.memory.queue, kernel, output_size * input_size
            )
            self.runtime.api.release_kernel(kernel)
            kernel = None
            return grad_weights, grad_bias, grad_inputs
        except Exception:
            if kernel is not None:
                try:
                    self.runtime.api.release_kernel(kernel)
                except Exception:
                    pass
            for buffer in reversed(allocated):
                try:
                    self.memory.release(buffer)
                except Exception:
                    pass
            raise

    def linear_step(
        self,
        weights: BackendArray,
        bias: BackendArray,
        grad_weights: BackendArray,
        grad_bias: BackendArray,
        learning_rate: float,
    ) -> tuple[OpenCLBuffer, OpenCLBuffer]:
        """Update resident Linear parameters in place on the selected GPU."""
        self._ensure_open()
        weights_buffer = self._require_buffer(
            weights, "weights", rank=2, operation="Linear.step"
        )
        bias_buffer = self._require_buffer(
            bias, "bias", rank=1, operation="Linear.step"
        )
        grad_weights_buffer = self._require_buffer(
            grad_weights, "grad_weights", rank=2, operation="Linear.step"
        )
        grad_bias_buffer = self._require_buffer(
            grad_bias, "grad_bias", rank=1, operation="Linear.step"
        )
        output_size, input_size = weights_buffer.shape
        if bias_buffer.shape != (output_size,):
            raise ValueError("bias must match the number of Linear outputs")
        if grad_weights_buffer.shape != (output_size, input_size):
            raise ValueError("grad_weights must match weights shape")
        if grad_bias_buffer.shape != (output_size,):
            raise ValueError("grad_bias must match the number of Linear outputs")
        buffers = (
            weights_buffer,
            bias_buffer,
            grad_weights_buffer,
            grad_bias_buffer,
        )
        if any(buffer._manager is not self.memory for buffer in buffers):
            raise BackendError("Linear.step buffers must belong to this OpenCL backend")
        if any(buffer.device_id != self.device_id for buffer in buffers):
            raise BackendError(
                "Linear.step buffers must reside on the selected OpenCL GPU"
            )
        if output_size > _OPENCL_INT_MAX or input_size > _OPENCL_INT_MAX:
            raise ValueError("OpenCL Linear dimensions must fit in a signed 32-bit integer")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise TypeError("learning_rate must be numeric")
        try:
            rate = float(learning_rate)
        except (OverflowError, ValueError) as error:
            raise ValueError("learning_rate must be finite") from error
        if not math.isfinite(rate):
            raise ValueError("learning_rate must be finite")
        if rate < 0.0:
            raise ValueError("learning_rate cannot be negative")

        self.runtime.require_device(self.device_id, require_fp64=True)
        if rate == 0.0:
            return weights_buffer, bias_buffer

        kernel: int | None = None
        try:
            program = self._get_program("linear_step.cl")
            kernel = self.runtime.api.create_kernel(
                program, "kinera_linear_step_f64"
            )
            for index, buffer in enumerate(buffers):
                self.runtime.api.set_kernel_buffer(kernel, index, buffer.handle)
            self.runtime.api.set_kernel_int(kernel, 4, output_size)
            self.runtime.api.set_kernel_int(kernel, 5, input_size)
            self.runtime.api.set_kernel_double(kernel, 6, rate)
            self.runtime.api.enqueue_1d(
                self.memory.queue, kernel, output_size * input_size
            )
            self.runtime.api.release_kernel(kernel)
            kernel = None
        except Exception:
            if kernel is not None:
                try:
                    self.runtime.api.release_kernel(kernel)
                except Exception:
                    pass
            raise
        return weights_buffer, bias_buffer

    def embedding_forward(
        self, weights: BackendArray, ids: list[int] | BackendBuffer
    ) -> OpenCLBuffer | list[list[float]]:
        """Look up resident row-major embedding weights by resident int32 IDs."""
        self._ensure_open()
        weights_buffer = self._require_buffer(
            weights, "weights", rank=2, operation="Embedding.forward"
        )
        self.memory._validate_buffer(weights_buffer)
        if weights_buffer.device_id != self.device_id:
            raise BackendError(
                "Embedding.forward weights must reside on the selected OpenCL GPU"
            )
        vocab_size, embedding_dim = weights_buffer.shape
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding.forward weights dimensions must be positive")
        if weights_buffer.nbytes != vocab_size * embedding_dim * ctypes.sizeof(ctypes.c_double):
            raise ValueError("Embedding.forward weights buffer size does not match its shape")

        if isinstance(ids, list):
            if ids:
                raise TypeError(
                    "OpenCL Embedding.forward requires IDs in an int32 OpenCLBuffer; "
                    "call buffer_from_host explicitly after layer validation"
                )
            # Match CPU/CUDA's empty lookup contract without creating a zero-sized
            # allocation or requiring FP64 kernel support for a no-op.
            self.runtime.require_device(self.device_id)
            return []
        if not isinstance(ids, OpenCLBuffer):
            raise TypeError("OpenCL Embedding.forward requires IDs as an OpenCLBuffer")
        ids._assert_open()
        if ids.dtype != "int32":
            raise ValueError("Embedding IDs must use dtype int32")
        if len(ids.shape) != 1:
            raise ValueError("Embedding IDs must have rank 1")
        self.memory._validate_buffer(ids)
        if ids.device_id != self.device_id:
            raise BackendError(
                "Embedding.forward IDs must reside on the selected OpenCL GPU"
            )
        ids_count = ids.shape[0]
        if ids_count <= 0:
            raise ValueError("resident Embedding IDs buffer must not be empty")
        if ids.nbytes != ids_count * ctypes.sizeof(ctypes.c_int32):
            raise ValueError("Embedding IDs buffer size does not match its shape")
        if any(
            dimension > _OPENCL_INT_MAX
            for dimension in (vocab_size, embedding_dim, ids_count)
        ):
            raise ValueError("Embedding dimensions must fit in a signed 32-bit integer")

        self.runtime.require_device(self.device_id, require_fp64=True)
        program = self._get_program("embedding_forward.cl")

        # Validate arbitrary resident IDs before allocating the result or enqueueing
        # the lookup. The validation kernel scans only the ID buffer and returns a
        # single int32 position; the full ID vector remains resident on the device.
        status: OpenCLBuffer | None = None
        validation_kernel: int | None = None
        try:
            status = self.memory.allocate((1,), dtype="int32", device_id=self.device_id)
            validation_kernel = self.runtime.api.create_kernel(
                program, "kinera_embedding_validate_ids_i32"
            )
            self.runtime.api.set_kernel_buffer(validation_kernel, 0, ids.handle)
            self.runtime.api.set_kernel_buffer(validation_kernel, 1, status.handle)
            self.runtime.api.set_kernel_int(validation_kernel, 2, ids_count)
            self.runtime.api.set_kernel_int(validation_kernel, 3, vocab_size)
            self.runtime.api.enqueue_1d(self.memory.queue, validation_kernel, 1)
            status_host = (ctypes.c_int32 * 1)()
            self.runtime.api.read_buffer(
                self.memory.queue,
                status.handle,
                ctypes.addressof(status_host),
                ctypes.sizeof(status_host),
            )
            invalid_position = int(status_host[0])
        except BaseException as error:
            cleanup_error = self._release_embedding_temporaries(
                validation_kernel, status
            )
            if cleanup_error is not None:
                raise error from cleanup_error
            raise

        cleanup_error = self._release_embedding_temporaries(
            validation_kernel, status
        )
        validation_kernel = None
        status = None
        if cleanup_error is not None:
            raise cleanup_error
        if invalid_position != -1:
            raise ValueError(
                f"Embedding ID at position {invalid_position} is outside "
                f"[0, {vocab_size})"
            )

        output: OpenCLBuffer | None = None
        lookup_kernel: int | None = None
        try:
            output = self.memory.allocate(
                (ids_count, embedding_dim),
                dtype="float64",
                device_id=self.device_id,
            )
            lookup_kernel = self.runtime.api.create_kernel(
                program, "kinera_embedding_forward_f64"
            )
            for index, buffer in enumerate((weights_buffer, ids, output)):
                self.runtime.api.set_kernel_buffer(lookup_kernel, index, buffer.handle)
            self.runtime.api.set_kernel_int(lookup_kernel, 3, embedding_dim)
            self.runtime.api.set_kernel_int(lookup_kernel, 4, ids_count)
            self.runtime.api.enqueue_1d(
                self.memory.queue, lookup_kernel, ids_count * embedding_dim
            )
            # A release call is attempted exactly once. Do not retry an ambiguous
            # native release error, which could turn into a double release.
            kernel_to_release, lookup_kernel = lookup_kernel, None
            self.runtime.api.release_kernel(kernel_to_release)
            result, output = output, None
            return result
        except BaseException as error:
            cleanup_error = self._release_embedding_temporaries(
                lookup_kernel, output
            )
            if cleanup_error is not None:
                raise error from cleanup_error
            raise

    def embedding_backward(
        self,
        grad_weights: BackendArray,
        ids: list[int] | BackendBuffer,
        grad_output: BackendArray,
    ) -> OpenCLBuffer:
        """Accumulate Embedding gradients in place, preserving ID order."""
        self._ensure_open()
        grad_weights_buffer = self._require_buffer(
            grad_weights, "grad_weights", rank=2, operation="Embedding.backward"
        )
        self.memory._validate_buffer(grad_weights_buffer)
        if grad_weights_buffer.device_id != self.device_id:
            raise BackendError(
                "Embedding.backward grad_weights must reside on the selected OpenCL GPU"
            )
        vocab_size, embedding_dim = grad_weights_buffer.shape
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding.backward dimensions must be positive")
        expected_grad_weights_nbytes = (
            vocab_size * embedding_dim * ctypes.sizeof(ctypes.c_double)
        )
        if grad_weights_buffer.nbytes != expected_grad_weights_nbytes:
            raise ValueError(
                "Embedding.backward grad_weights buffer size does not match its shape"
            )

        # Empty Embedding forwards are represented as Python lists by the layer.
        # Preserve the CPU/CUDA no-op contract without creating zero-sized buffers.
        if isinstance(ids, list) and not ids:
            if grad_output != []:
                raise ValueError("empty IDs require empty grad_output")
            self.runtime.require_device(self.device_id)
            return grad_weights_buffer
        if isinstance(ids, list):
            raise TypeError(
                "OpenCL Embedding.backward requires IDs in an int32 OpenCLBuffer; "
                "call buffer_from_host explicitly after layer validation"
            )
        if not isinstance(ids, OpenCLBuffer):
            raise TypeError("OpenCL Embedding.backward requires IDs as an OpenCLBuffer")
        ids._assert_open()
        if ids.dtype != "int32":
            raise ValueError("Embedding IDs must use dtype int32")
        if len(ids.shape) != 1:
            raise ValueError("Embedding IDs must have rank 1")
        self.memory._validate_buffer(ids)
        if ids.device_id != self.device_id:
            raise BackendError(
                "Embedding.backward IDs must reside on the selected OpenCL GPU"
            )
        ids_count = ids.shape[0]
        if ids_count <= 0:
            raise ValueError("resident Embedding IDs buffer must not be empty")
        expected_ids_nbytes = ids_count * ctypes.sizeof(ctypes.c_int32)
        if ids.nbytes != expected_ids_nbytes:
            raise ValueError("Embedding IDs buffer size does not match its shape")

        grad_output_buffer = self._require_buffer(
            grad_output, "grad_output", rank=2, operation="Embedding.backward"
        )
        self.memory._validate_buffer(grad_output_buffer)
        if grad_output_buffer.device_id != self.device_id:
            raise BackendError(
                "Embedding.backward grad_output must reside on the selected OpenCL GPU"
            )
        if grad_output_buffer.shape != (ids_count, embedding_dim):
            raise ValueError(
                "grad_output must have shape (len(ids), embedding_dim)"
            )
        expected_grad_output_nbytes = (
            ids_count * embedding_dim * ctypes.sizeof(ctypes.c_double)
        )
        if grad_output_buffer.nbytes != expected_grad_output_nbytes:
            raise ValueError(
                "Embedding.backward grad_output buffer size does not match its shape"
            )

        buffers = (grad_weights_buffer, ids, grad_output_buffer)
        if any(buffer._manager is not self.memory for buffer in buffers):
            raise BackendError(
                "Embedding.backward buffers must belong to this OpenCL backend"
            )
        if any(buffer.device_id != self.device_id for buffer in buffers):
            raise BackendError(
                "Embedding.backward buffers must reside on the selected OpenCL GPU"
            )
        handles = [buffer.handle for buffer in buffers]
        if len(set(handles)) != len(handles):
            raise BackendError("Embedding.backward buffers must not alias")
        if any(
            dimension > _OPENCL_INT_MAX
            for dimension in (vocab_size, embedding_dim, ids_count)
        ):
            raise ValueError(
                "Embedding.backward dimensions must fit in a signed 32-bit integer"
            )
        total_gradient_elements = vocab_size * embedding_dim
        if total_gradient_elements > ctypes.c_size_t(-1).value:
            raise ValueError("Embedding.backward dispatch exceeds the host size_t range")
        grad_output_elements = ids_count * embedding_dim
        if grad_output_elements > _OPENCL_INT_MAX - 1:
            raise ValueError(
                "Embedding.backward grad_output is too large for validation status encoding"
            )

        self.runtime.require_device(self.device_id, require_fp64=True)
        program = self._get_program("embedding_backward.cl")

        # Device-side preflight validates both resident IDs and every incoming
        # gradient before the in-place accumulation kernel can be enqueued. Only
        # its single int32 status value is copied back to the host.
        status: OpenCLBuffer | None = None
        validation_kernel: int | None = None
        try:
            status = self.memory.allocate(
                (1,), dtype="int32", device_id=self.device_id
            )
            validation_kernel = self.runtime.api.create_kernel(
                program, "kinera_embedding_validate_backward_inputs"
            )
            for index, buffer in enumerate((ids, grad_output_buffer, status)):
                self.runtime.api.set_kernel_buffer(
                    validation_kernel, index, buffer.handle
                )
            self.runtime.api.set_kernel_int(validation_kernel, 3, ids_count)
            self.runtime.api.set_kernel_int(validation_kernel, 4, vocab_size)
            self.runtime.api.set_kernel_int(validation_kernel, 5, embedding_dim)
            self.runtime.api.enqueue_1d(self.memory.queue, validation_kernel, 1)
            status_host = (ctypes.c_int32 * 1)()
            self.runtime.api.read_buffer(
                self.memory.queue,
                status.handle,
                ctypes.addressof(status_host),
                ctypes.sizeof(status_host),
            )
            validation_status = int(status_host[0])
        except BaseException as error:
            cleanup_error = self._release_embedding_temporaries(
                validation_kernel, status
            )
            if cleanup_error is not None:
                raise error from cleanup_error
            raise

        kernel_to_release, validation_kernel = validation_kernel, None
        status_to_release, status = status, None
        cleanup_error = self._release_embedding_temporaries(
            kernel_to_release, status_to_release
        )
        if cleanup_error is not None:
            raise cleanup_error
        if validation_status > 0:
            raise ValueError(
                f"Embedding ID at position {validation_status - 1} is outside "
                f"[0, {vocab_size})"
            )
        if validation_status < 0:
            raise ValueError(
                "Embedding grad_output contains a non-finite value at position "
                f"{(-validation_status - 1) // embedding_dim}, dimension "
                f"{(-validation_status - 1) % embedding_dim}"
            )

        accumulation_kernel: int | None = None
        try:
            accumulation_kernel = self.runtime.api.create_kernel(
                program, "kinera_embedding_backward_f64"
            )
            for index, buffer in enumerate(
                (grad_weights_buffer, ids, grad_output_buffer)
            ):
                self.runtime.api.set_kernel_buffer(
                    accumulation_kernel, index, buffer.handle
                )
            self.runtime.api.set_kernel_int(accumulation_kernel, 3, vocab_size)
            self.runtime.api.set_kernel_int(
                accumulation_kernel, 4, embedding_dim
            )
            self.runtime.api.set_kernel_int(accumulation_kernel, 5, ids_count)
            self.runtime.api.enqueue_1d(
                self.memory.queue, accumulation_kernel, total_gradient_elements
            )
            kernel_to_release, accumulation_kernel = accumulation_kernel, None
            self.runtime.api.release_kernel(kernel_to_release)
        except BaseException as error:
            cleanup_error = self._release_embedding_temporaries(
                accumulation_kernel
            )
            if cleanup_error is not None:
                raise error from cleanup_error
            raise
        return grad_weights_buffer

    def embedding_step(
        self,
        weights: BackendArray,
        grad_weights: BackendArray,
        learning_rate: float,
    ) -> OpenCLBuffer:
        self._unavailable("Embedding.step")

    def neuron_forward(
        self,
        weights: list[float],
        bias: float,
        inputs: list[float],
        activation: str,
        negative_slope: float,
    ) -> tuple[float, float]:
        self._unavailable("Neuron.forward")

    def neuron_backward(
        self,
        weights: list[float],
        inputs: list[float],
        z: float,
        grad_output: float,
        activation: str,
        negative_slope: float,
    ) -> tuple[list[float], float, list[float]]:
        self._unavailable("Neuron.backward")

    def neuron_step(
        self,
        weights: list[float],
        bias: float,
        grad_weights: list[float],
        grad_bias: float,
        learning_rate: float,
    ) -> tuple[list[float], float]:
        self._unavailable("Neuron.step")

    def activation(
        self, name: str, value: float, negative_slope: float = 0.01
    ) -> float:
        self._unavailable("activation")

    def activation_derivative(
        self, name: str, value: float, negative_slope: float = 0.01
    ) -> float:
        self._unavailable("activation derivative")

    def softmax(self, logits: list[float]) -> list[float]:
        self._unavailable("softmax")

    def cross_entropy(self, logits: list[float], target_id: int) -> float:
        self._unavailable("cross entropy")

    def cross_entropy_gradient(
        self, logits: list[float], target_id: int
    ) -> list[float]:
        self._unavailable("cross entropy gradient")

    def binary_cross_entropy_with_logits(self, logit: float, target: float) -> float:
        self._unavailable("binary cross entropy")

    def binary_cross_entropy_with_logits_gradient(
        self, logit: float, target: float
    ) -> float:
        self._unavailable("binary cross entropy gradient")

    def close(self) -> None:
        if self._closed:
            return
        programs, self._programs = self._programs, {}
        errors: list[Exception] = []
        for program in programs.values():
            try:
                self.runtime.api.release_program(program)
            except Exception as error:
                errors.append(error)
        try:
            self.memory.close()
        except Exception as error:
            errors.append(error)
        self._closed = True
        if errors:
            raise BackendError(f"failed to close OpenCL backend: {errors[0]}")

    def _get_program(self, kernel_filename: str) -> int:
        self._ensure_open()
        if kernel_filename not in {
            "linear_forward.cl",
            "linear_backward.cl",
            "linear_step.cl",
            "embedding_forward.cl",
            "embedding_backward.cl",
        }:
            raise ValueError(f"unsupported OpenCL kernel program: {kernel_filename}")
        program = self._programs.get(kernel_filename)
        if program is None:
            self.runtime.require_device(self.device_id, require_fp64=True)
            source = (
                resources.files("kinera.backends.opencl")
                .joinpath(f"kernels/{kernel_filename}")
                .read_text(encoding="utf-8")
            )
            program = self.runtime.api.build_program(
                self.memory.context,
                self.memory.device.handle,
                source,
                "",
            )
            self._programs[kernel_filename] = program
        return program

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendError("OpenCL backend is closed")

    def _release_embedding_temporaries(
        self, kernel: int | None, *buffers: OpenCLBuffer | None
    ) -> BaseException | None:
        """Attempt every temporary cleanup once and return the first failure."""
        first_error: BaseException | None = None
        if kernel is not None:
            try:
                self.runtime.api.release_kernel(kernel)
            except BaseException as error:
                first_error = error
        for buffer in buffers:
            if buffer is None:
                continue
            try:
                self.memory.release(buffer)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        return first_error

    @staticmethod
    def _require_buffer(
        value: BackendArray,
        name: str,
        *,
        rank: int,
        operation: str = "Linear.forward",
    ) -> OpenCLBuffer:
        if not isinstance(value, OpenCLBuffer):
            raise TypeError(
                f"OpenCL {operation} requires {name} as an OpenCLBuffer; "
                "call buffer_from_host explicitly for host data"
            )
        value._assert_open()
        if value.device != "opencl":
            raise BackendError(f"{name} must reside on OpenCL")
        if value.dtype != "float64":
            raise ValueError(f"{name} must use dtype float64")
        if len(value.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}")
        return value

    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise BackendUnavailableError(
            f"OpenCL operation {operation} is not available in the current gate"
        )
