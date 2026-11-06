"""Opt-in RX 550 integration gate for Linear.forward -> backward -> step."""

import os
import sys
import unittest
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.base import BackendBuffer
from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.layers.linear import Linear


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on an OpenCL FP64 GPU runner",
)
class LinearOpenCLIntegratedEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = get_backend("opencl")
        cls.device = cls.backend.runtime.require_device(0, require_fp64=True)
        if cls.device.pci_bdf != "0000:03:00.0":
            raise AssertionError(
                f"expected RX 550 at 0000:03:00.0, got {cls.device.pci_bdf}"
            )
        if "Radeon RX 550" not in cls.device.name:
            raise AssertionError(f"expected RX 550, got {cls.device.name}")
        cls.cpu = CPUBackend()
        print(
            "OPENCL_INTEGRATION_GATE_DEVICE",
            cls.device.name,
            cls.device.pci_bdf,
            "PYTHON",
            sys.version.split()[0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_seeded_layers_complete_repeated_cycles_for_requested_shapes(self) -> None:
        for output_size, input_size in ((1, 1), (51, 26), (3932, 26)):
            with self.subTest(shape=(output_size, input_size)):
                self._exercise_repeated_cycles(output_size, input_size, cycles=2)

    def _exercise_repeated_cycles(
        self, output_size: int, input_size: int, *, cycles: int
    ) -> None:
        seed = 20261105 + output_size + input_size
        cpu_layer = Linear(input_size, output_size, seed=seed, backend=self.cpu)
        gpu_layer = Linear(input_size, output_size, seed=seed, backend=self.backend)
        current_gpu_output = None
        current_gpu_grad_inputs = None
        setup_buffers: list[BackendBuffer] = []
        try:
            self.assertEqual(cpu_layer.weights, self.backend.buffer_to_host(gpu_layer.weights))
            self.assertEqual(cpu_layer.bias, self.backend.buffer_to_host(gpu_layer.bias))
            self.assertEqual(gpu_layer.weights.dtype, "float64")
            self.assertEqual(gpu_layer.bias.dtype, "float64")

            previous_input = None
            previous_output = None
            previous_grad_weights = None
            previous_grad_bias = None

            for cycle in range(cycles):
                inputs = [
                    (((column * 7 + cycle * 3) % 19) - 9) / 11.0
                    for column in range(input_size)
                ]
                grad_output = [
                    (((row * 13 + cycle * 5) % 23) - 11) / 17.0
                    for row in range(output_size)
                ]
                rate = (0.125, 0.03125)[cycle % 2]

                cpu_output = cpu_layer.forward(inputs)
                cpu_grad_inputs = cpu_layer.backward(grad_output)
                cpu_grad_weights = cpu_layer.grad_weights
                cpu_grad_bias = cpu_layer.grad_bias
                cpu_layer.step(rate)

                api = self.backend.runtime.api
                with patch.object(api, "write_buffer", wraps=api.write_buffer) as write:
                    device_inputs = self.backend.buffer_from_host(
                        inputs, shape=(input_size,), dtype="float64"
                    )
                    device_grad_output = self.backend.buffer_from_host(
                        grad_output, shape=(output_size,), dtype="float64"
                    )
                    self.assertEqual(write.call_count, 2)
                setup_buffers.extend((device_inputs, device_grad_output))
                parameters_before_step = (gpu_layer.weights, gpu_layer.bias)

                with (
                    patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                    patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
                ):
                    gpu_output = gpu_layer.forward(device_inputs)
                    self.assertIsInstance(gpu_output, BackendBuffer)
                    self.assertIs(gpu_layer.output, gpu_output)
                    self.assertIs(gpu_layer.last_input, device_inputs)
                    self._assert_opencl_buffer(
                        gpu_output, (output_size,), "float64"
                    )
                    self._assert_opencl_buffer(
                        gpu_layer.weights, (output_size, input_size), "float64"
                    )
                    self._assert_opencl_buffer(
                        gpu_layer.bias, (output_size,), "float64"
                    )
                    self._assert_opencl_buffer(
                        gpu_layer.last_input, (input_size,), "float64"
                    )
                    gpu_grad_inputs = gpu_layer.backward(device_grad_output)
                    self.assertIsInstance(gpu_grad_inputs, BackendBuffer)
                    self.assertIsInstance(gpu_layer.grad_weights, BackendBuffer)
                    self.assertIsInstance(gpu_layer.grad_bias, BackendBuffer)
                    self._assert_opencl_buffer(
                        gpu_grad_inputs, (input_size,), "float64"
                    )
                    self._assert_opencl_buffer(
                        gpu_layer.grad_weights,
                        (output_size, input_size),
                        "float64",
                    )
                    self._assert_opencl_buffer(
                        gpu_layer.grad_bias, (output_size,), "float64"
                    )
                    self.assertIs(gpu_layer.weights, parameters_before_step[0])
                    self.assertIs(gpu_layer.bias, parameters_before_step[1])
                    gpu_layer.step(rate)
                    self.assertIs(gpu_layer.weights, parameters_before_step[0])
                    self.assertIs(gpu_layer.bias, parameters_before_step[1])
                    read.assert_not_called()
                    write.assert_not_called()

                if previous_grad_weights is not None:
                    self.assertTrue(previous_grad_weights.closed)
                    self.assertTrue(previous_grad_bias.closed)
                    self.assertFalse(previous_input.closed)
                    self.assertFalse(previous_output.closed)
                    previous_input.close()
                    previous_output.close()

                current_gpu_output = gpu_output
                current_gpu_grad_inputs = gpu_grad_inputs
                with patch.object(api, "read_buffer", wraps=api.read_buffer) as read:
                    actual_output = self.backend.buffer_to_host(gpu_output)
                    actual_grad_weights = self.backend.buffer_to_host(
                        gpu_layer.grad_weights
                    )
                    actual_grad_bias = self.backend.buffer_to_host(gpu_layer.grad_bias)
                    actual_grad_inputs = self.backend.buffer_to_host(gpu_grad_inputs)
                    actual_weights = self.backend.buffer_to_host(gpu_layer.weights)
                    actual_bias = self.backend.buffer_to_host(gpu_layer.bias)
                    self.assertEqual(read.call_count, 6)

                shape_label = f"{output_size}x{input_size}/cycle-{cycle + 1}"
                self._assert_values(f"{shape_label}/output", actual_output, cpu_output)
                self._assert_values(
                    f"{shape_label}/grad_weights",
                    actual_grad_weights,
                    cpu_grad_weights,
                )
                self._assert_values(
                    f"{shape_label}/grad_bias", actual_grad_bias, cpu_grad_bias
                )
                self._assert_values(
                    f"{shape_label}/grad_inputs", actual_grad_inputs, cpu_grad_inputs
                )
                self._assert_values(
                    f"{shape_label}/final_weights", actual_weights, cpu_layer.weights
                )
                self._assert_values(
                    f"{shape_label}/final_bias", actual_bias, cpu_layer.bias
                )

                previous_input = device_inputs
                previous_output = gpu_output
                previous_grad_weights = gpu_layer.grad_weights
                previous_grad_bias = gpu_layer.grad_bias
                device_grad_output.close()
                setup_buffers.remove(device_grad_output)
                gpu_grad_inputs.close()
                current_gpu_grad_inputs = None

            current_buffers = (
                gpu_layer.weights,
                gpu_layer.bias,
                gpu_layer.last_input,
                gpu_layer.output,
                gpu_layer.grad_weights,
                gpu_layer.grad_bias,
            )
            gpu_layer.close()
            gpu_layer.close()
            self.assertTrue(all(buffer.closed for buffer in current_buffers))
            self.assertIsNone(current_gpu_grad_inputs)
            self.assertIsNotNone(current_gpu_output)
            self.assertTrue(current_gpu_output.closed)
            for buffer in setup_buffers:
                if not buffer.closed:
                    buffer.close()
            with self.assertRaisesRegex(RuntimeError, "Linear is closed"):
                gpu_layer.forward([0.0] * input_size)
            with self.assertRaisesRegex(RuntimeError, "Linear is closed"):
                gpu_layer.backward([0.0] * output_size)
            with self.assertRaisesRegex(RuntimeError, "Linear is closed"):
                gpu_layer.step(0.1)
        finally:
            if not gpu_layer._closed:
                gpu_layer.close()
            for buffer in setup_buffers:
                if not buffer.closed:
                    buffer.close()
            if current_gpu_grad_inputs is not None and not current_gpu_grad_inputs.closed:
                current_gpu_grad_inputs.close()
            cpu_layer.close()

    def test_failed_backward_and_step_leave_layer_retryable(self) -> None:
        cpu_layer = Linear(3, 2, seed=914, backend=self.cpu)
        gpu_layer = Linear(3, 2, seed=914, backend=self.backend)
        inputs = [0.5, -1.0, 0.25]
        grad_output = [1.0, -0.5]
        device_inputs = self.backend.buffer_from_host(
            inputs, shape=(3,), dtype="float64"
        )
        device_grad_output = self.backend.buffer_from_host(
            grad_output, shape=(2,), dtype="float64"
        )
        try:
            cpu_output = cpu_layer.forward(inputs)
            cpu_grad_inputs = cpu_layer.backward(grad_output)
            gpu_output = gpu_layer.forward(device_inputs)
            api = self.backend.runtime.api
            with (
                patch.object(
                    api,
                    "enqueue_1d",
                    side_effect=RuntimeError("injected backward enqueue"),
                ),
                patch.object(api, "release_mem", wraps=api.release_mem) as release_mem,
                patch.object(
                    api, "read_buffer", wraps=api.read_buffer
                ) as read,
                patch.object(
                    api, "write_buffer", wraps=api.write_buffer
                ) as write,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected backward enqueue"):
                    gpu_layer.backward(device_grad_output)
                self.assertEqual(release_mem.call_count, 3)
                read.assert_not_called()
                write.assert_not_called()
            self.assertFalse(gpu_layer._has_backward)
            self.assertIsNone(gpu_layer.grad_weights)
            self.assertIsNone(gpu_layer.grad_bias)
            self.assertFalse(gpu_layer.output.closed)
            self.assertFalse(gpu_layer.last_input.closed)

            gpu_grad_inputs = gpu_layer.backward(device_grad_output)
            self._assert_values(
                "failure-retry/output",
                self.backend.buffer_to_host(gpu_output),
                cpu_output,
            )
            self._assert_values(
                "failure-retry/grad_inputs",
                self.backend.buffer_to_host(gpu_grad_inputs),
                cpu_grad_inputs,
            )
            cpu_layer.step(0.0625)
            before_weights = self.backend.buffer_to_host(gpu_layer.weights)
            before_bias = self.backend.buffer_to_host(gpu_layer.bias)
            release_kernel = api.release_kernel
            with (
                patch.object(
                    api, "enqueue_1d", side_effect=RuntimeError("injected step enqueue")
                ),
                patch.object(api, "read_buffer", wraps=api.read_buffer) as read,
                patch.object(api, "write_buffer", wraps=api.write_buffer) as write,
                patch.object(api, "release_kernel", wraps=release_kernel) as release,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected step enqueue"):
                    gpu_layer.step(0.0625)
                read.assert_not_called()
                write.assert_not_called()
                release.assert_called_once()

            self.assertTrue(gpu_layer._has_backward)
            self.assertEqual(self.backend.buffer_to_host(gpu_layer.weights), before_weights)
            self.assertEqual(self.backend.buffer_to_host(gpu_layer.bias), before_bias)
            gpu_layer.step(0.0625)
            self._assert_values(
                "failure-retry/final_weights",
                self.backend.buffer_to_host(gpu_layer.weights),
                cpu_layer.weights,
            )
            self._assert_values(
                "failure-retry/final_bias",
                self.backend.buffer_to_host(gpu_layer.bias),
                cpu_layer.bias,
            )
            gpu_grad_inputs.close()
            device_grad_output.close()
            gpu_layer.close()
            self.assertTrue(device_inputs.closed)
            self.assertTrue(gpu_output.closed)
            with self.assertRaisesRegex(RuntimeError, "Linear is closed"):
                gpu_layer.forward(device_inputs)
        finally:
            if not gpu_layer._closed:
                gpu_layer.close()
            if not device_grad_output.closed:
                device_grad_output.close()
            cpu_layer.close()

    def _assert_values(self, label, actual, expected) -> None:
        flat_actual = self._flatten(actual)
        flat_expected = self._flatten(expected)
        self.assertEqual(len(flat_actual), len(flat_expected))
        errors = [abs(got - want) for got, want in zip(flat_actual, flat_expected)]
        for index, (got, want) in enumerate(zip(flat_actual, flat_expected)):
            with self.subTest(value=label, index=index):
                self.assertLessEqual(
                    abs(got - want), ATOL + RTOL * abs(want),
                    msg=f"{label}: OpenCL={got}, CPU={want}",
                )
        print(f"OPENCL_LINEAR_INTEGRATION_{label.upper()}_MAX_ABS_ERROR", max(errors, default=0.0))

    def _assert_opencl_buffer(self, buffer, shape, dtype) -> None:
        self.assertIsInstance(buffer, BackendBuffer)
        self.assertFalse(buffer.closed)
        self.assertTrue(buffer.owns_memory)
        self.assertEqual(buffer.device, "opencl")
        self.assertEqual(buffer.device_id, 0)
        self.assertEqual(buffer.shape, shape)
        self.assertEqual(buffer.dtype, dtype)
        self.assertIs(buffer._manager, self.backend.memory)

    @staticmethod
    def _flatten(values):
        if not values:
            return []
        if isinstance(values[0], list):
            return [value for row in values for value in row]
        return list(values)


if __name__ == "__main__":
    unittest.main()
