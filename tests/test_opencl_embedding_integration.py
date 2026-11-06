"""Opt-in RX 550 integration gate for the full Embedding layer cycle (F1d)."""

import os
import sys
import unittest
from collections import Counter
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.base import BackendBuffer
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.buffer import OpenCLBuffer
from kinera.nn.layers.embedding import Embedding


RUN_OPENCL_TESTS = os.environ.get("KINERA_RUN_OPENCL_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12


@unittest.skipUnless(
    RUN_OPENCL_TESTS,
    "set KINERA_RUN_OPENCL_TESTS=1 on RX 550/Rusticl with FP64 enabled",
)
class EmbeddingOpenCLIntegrationRX550Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = get_backend("opencl")
        try:
            cls.device = cls.backend.runtime.require_device(0, require_fp64=True)
            if cls.device.pci_bdf != "0000:03:00.0":
                raise AssertionError(
                    f"expected RX 550 at 0000:03:00.0, got {cls.device.pci_bdf}"
                )
            if "Radeon RX 550" not in cls.device.name:
                raise AssertionError(f"expected RX 550, got {cls.device.name}")
        except Exception:
            cls.backend.close()
            raise
        cls.cpu = CPUBackend()
        print(
            "OPENCL_F1D_DEVICE",
            cls.device.name,
            "PCI",
            cls.device.pci_bdf,
            "DRIVER",
            cls.device.driver_version,
            "RUNTIME",
            cls.device.version,
            "OPENCL_C",
            cls.device.opencl_c_version,
            "PYTHON",
            sys.version.split()[0],
            "FP64",
            cls.device.supports_fp64,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_repeated_layer_cycles_match_cpu_at_each_checkpoint(self) -> None:
        baseline_live = set(self.backend.memory._buffers)
        cpu_layer = Embedding(4, 3, seed=20261106, backend=self.cpu)
        gpu_layer = Embedding(4, 3, seed=20261106, backend=self.backend)
        weight_buffer = gpu_layer.weights
        self.assertIsInstance(weight_buffer, OpenCLBuffer)
        self.assertEqual(self.backend.buffer_to_host(weight_buffer), cpu_layer.weights)
        owned_buffers: list[OpenCLBuffer] = [weight_buffer, gpu_layer.grad_weights]
        errors = {
            "forward": 0.0,
            "gradients_after_backward_1": 0.0,
            "gradients_after_backward_2": 0.0,
            "weights_after_step": 0.0,
        }

        with self.assertRaisesRegex(RuntimeError, "backward anterior"):
            gpu_layer.step(0.1)
        with self.assertRaisesRegex(RuntimeError, "backward anterior"):
            cpu_layer.step(0.1)

        cases = (
            (
                [2, 0, 2, 1],
                [
                    [1e16, 1.0, -1.0],
                    [0.5, -0.5, 0.0],
                    [-1e16, 2.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 2.0],
                    [-1.0, 2.0, 0.0],
                    [1.0, -1.0, -1.0],
                    [0.5, -0.5, 1.0],
                ],
                0.0,
            ),
            (
                [3, 3, 1],
                [[-2.0, 0.0, 1.5], [2.0, -0.0, -1.5], [0.25, -0.5, 1.0]],
                [[1.0, -2.0, 0.0], [-0.5, 1.0, 2.0], [0.0, 0.75, -1.25]],
                0.125,
            ),
            (
                [0, 2, 1, 0, 2],
                [
                    [1.0, 2.0, -3.0],
                    [4.0, -5.0, 6.0],
                    [-7.0, 8.0, 0.0],
                    [0.0, -1.0, 2.0],
                    [-4.0, 5.0, -6.0],
                ],
                [
                    [-0.25, 0.5, 1.0],
                    [0.0, 1.5, -2.0],
                    [3.0, -4.0, 5.0],
                    [0.75, -0.5, 0.0],
                    [1.0, -1.0, 2.0],
                ],
                0.03125,
            ),
        )
        previous_state = None
        try:
            for cycle, (ids, grad_one, grad_two, rate) in enumerate(cases):
                if previous_state is not None:
                    self.assertIs(gpu_layer.weights, weight_buffer)
                    self._assert_close(
                        self.backend.buffer_to_host(gpu_layer.weights),
                        cpu_layer.weights,
                    )

                cpu_output = cpu_layer.forward(ids.copy())
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_output = gpu_layer.forward(ids.copy())
                    materialize.assert_not_called()
                if previous_state is not None:
                    old_ids, old_output, old_gradients = previous_state
                    self.assertTrue(old_ids.closed)
                    self.assertTrue(old_output.closed)
                    self.assertTrue(old_gradients.closed)
                self.assertIsInstance(gpu_output, BackendBuffer)
                self.assertIs(gpu_layer.output, gpu_output)
                owned_buffers.extend(
                    (gpu_layer.last_ids, gpu_output, gpu_layer.grad_weights)
                )
                errors["forward"] = max(
                    errors["forward"],
                    self._assert_close(
                        self.backend.buffer_to_host(gpu_output), cpu_output
                    ),
                )
                self.assertEqual(
                    self.backend.buffer_to_host(gpu_layer.grad_weights),
                    [[0.0, 0.0, 0.0] for _ in range(4)],
                )

                cpu_layer.backward(grad_one)
                live_before_backward = set(self.backend.memory._buffers)
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_layer.backward(grad_one)
                    materialize.assert_not_called()
                self.assertEqual(
                    set(self.backend.memory._buffers), live_before_backward,
                    "host grad_output and backward preflight temporaries are released",
                )
                self.assertTrue(gpu_layer._has_backward)
                errors["gradients_after_backward_1"] = max(
                    errors["gradients_after_backward_1"],
                    self._assert_close(
                        self.backend.buffer_to_host(gpu_layer.grad_weights),
                        cpu_layer.grad_weights,
                    ),
                )

                resident_gradients = self.backend.buffer_from_host(
                    grad_two, shape=(len(ids), 3), dtype="float64"
                )
                owned_buffers.append(resident_gradients)
                try:
                    cpu_layer.backward(grad_two)
                    live_before_resident_backward = set(self.backend.memory._buffers)
                    with patch.object(
                        self.backend,
                        "buffer_to_host",
                        wraps=self.backend.buffer_to_host,
                    ) as materialize:
                        gpu_layer.backward(resident_gradients)
                        materialize.assert_not_called()
                    self.assertFalse(resident_gradients.closed)
                    self.assertEqual(
                        set(self.backend.memory._buffers),
                        live_before_resident_backward,
                        "resident input remains caller-owned and status temporaries close",
                    )
                    errors["gradients_after_backward_2"] = max(
                        errors["gradients_after_backward_2"],
                        self._assert_close(
                            self.backend.buffer_to_host(gpu_layer.grad_weights),
                            cpu_layer.grad_weights,
                        ),
                    )
                finally:
                    resident_gradients.close()

                cpu_layer.step(rate)
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_layer.step(rate)
                    materialize.assert_not_called()
                self.assertFalse(gpu_layer._has_backward)
                self.assertIs(gpu_layer.weights, weight_buffer)
                errors["weights_after_step"] = max(
                    errors["weights_after_step"],
                    self._assert_close(
                        self.backend.buffer_to_host(gpu_layer.weights),
                        cpu_layer.weights,
                    ),
                )
                previous_state = (
                    gpu_layer.last_ids,
                    gpu_layer.output,
                    gpu_layer.grad_weights,
                )

            # Empty IDs use the list contract: no zero-sized output and no
            # lookup, validation, backward, or step kernel dispatch.
            empty_previous_state = previous_state
            kernels_before_empty = len(self.backend._programs)
            cpu_layer.forward([])
            with (
                patch.object(
                    self.backend.runtime.api,
                    "create_kernel",
                    wraps=self.backend.runtime.api.create_kernel,
                ) as create_kernel,
                patch.object(
                    self.backend.runtime.api,
                    "enqueue_1d",
                    wraps=self.backend.runtime.api.enqueue_1d,
                ) as enqueue,
            ):
                self.assertEqual(gpu_layer.forward([]), [])
                self.assertEqual(gpu_layer.last_ids, [])
                self.assertEqual(gpu_layer.output, [])
                self.assertFalse(
                    any(0 in buffer.shape for buffer in self.backend.memory._buffers)
                )
                cpu_layer.backward([])
                gpu_layer.backward([])
                cpu_layer.step(0.0)
                gpu_layer.step(0.0)
                create_kernel.assert_not_called()
                enqueue.assert_not_called()

            self.assertTrue(empty_previous_state[0].closed)
            self.assertTrue(empty_previous_state[1].closed)
            self.assertTrue(empty_previous_state[2].closed)
            owned_buffers.append(gpu_layer.grad_weights)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.grad_weights),
                [[0.0, 0.0, 0.0] for _ in range(4)],
            )
            self.assertIs(gpu_layer.weights, weight_buffer)
            self.assertEqual(
                self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
            )
            self.assertEqual(len(self.backend._programs), kernels_before_empty)
            print("OPENCL_F1D_MAX_ABS_ERRORS", errors)
        finally:
            if not gpu_layer._closed:
                gpu_layer.close()
            if not cpu_layer._closed:
                cpu_layer.close()

        self.assertTrue(all(buffer.closed for buffer in owned_buffers))
        self.assertFalse(set(self.backend.memory._buffers) - baseline_live)

    def test_release_output_backward_discard_and_close_lifecycle(self) -> None:
        baseline_live = set(self.backend.memory._buffers)
        api = self.backend.runtime.api
        release_counts = Counter()
        allocation_counts = Counter()
        generation_by_handle = {}
        original_create = api.create_buffer
        original_release = api.release_mem

        def track_create(context: int, nbytes: int) -> int:
            handle = original_create(context, nbytes)
            allocation_counts[handle] += 1
            generation_by_handle[handle] = allocation_counts[handle]
            return handle

        def count_release(handle: int) -> None:
            release_counts[(handle, generation_by_handle[handle])] += 1
            original_release(handle)

        cpu_layer = None
        gpu_layer = None
        owned: list[OpenCLBuffer] = []
        try:
            with (
                patch.object(api, "create_buffer", side_effect=track_create),
                patch.object(api, "release_mem", side_effect=count_release),
            ):
                cpu_layer = Embedding(3, 2, seed=92, backend=self.cpu)
                gpu_layer = Embedding(3, 2, seed=92, backend=self.backend)
                owned.extend((gpu_layer.weights, gpu_layer.grad_weights))
                weights = gpu_layer.weights
                cpu_layer.forward([2, 1, 2])
                output = gpu_layer.forward([2, 1, 2])
                ids = gpu_layer.last_ids
                grad_weights = gpu_layer.grad_weights
                owned.extend((ids, output, grad_weights))
                output_key, ids_key, grad_key = (
                    (output.handle, generation_by_handle[output.handle]),
                    (ids.handle, generation_by_handle[ids.handle]),
                    (grad_weights.handle, generation_by_handle[grad_weights.handle]),
                )

                gpu_layer.release_output_buffer()
                cpu_layer.release_output_buffer()
                self.assertTrue(output.closed)
                self.assertIs(gpu_layer.output, output)
                self.assertFalse(ids.closed)

                gradients = [[1.0, -2.0], [0.5, 4.0], [-1.5, 3.0]]
                cpu_layer.backward(gradients)
                with patch.object(
                    self.backend, "buffer_to_host", wraps=self.backend.buffer_to_host
                ) as materialize:
                    gpu_layer.backward(gradients)
                    materialize.assert_not_called()
                self.assertEqual(
                    self.backend.buffer_to_host(gpu_layer.grad_weights),
                    cpu_layer.grad_weights,
                )
                cpu_layer.step(0.25)
                gpu_layer.step(0.25)
                self.assertIs(gpu_layer.weights, weights)
                self.assertEqual(
                    self.backend.buffer_to_host(gpu_layer.weights), cpu_layer.weights
                )

                cpu_layer.forward([0])
                gpu_layer.forward([0])
                self.assertTrue(ids.closed)
                self.assertTrue(output.closed)
                self.assertTrue(grad_weights.closed)
                new_ids = gpu_layer.last_ids
                new_output = gpu_layer.output
                new_gradients = gpu_layer.grad_weights
                owned.extend((new_ids, new_output, new_gradients))
                new_keys = tuple(
                    (buffer.handle, generation_by_handle[buffer.handle])
                    for buffer in (new_ids, new_output, new_gradients)
                )
                cpu_layer.backward([[2.0, -3.0]])
                gpu_layer.backward([[2.0, -3.0]])

                gpu_layer.discard_forward_state()
                cpu_layer.discard_forward_state()
                gpu_layer.discard_forward_state()
                cpu_layer.discard_forward_state()
                self.assertIsNone(gpu_layer.last_ids)
                self.assertIsNone(gpu_layer.output)
                self.assertTrue(new_ids.closed)
                self.assertTrue(new_output.closed)
                self.assertIs(gpu_layer.grad_weights, new_gradients)
                self.assertFalse(new_gradients.closed)
                self.assertFalse(gpu_layer._has_backward)
                with self.assertRaisesRegex(RuntimeError, "forward anterior"):
                    gpu_layer.backward([])
                with self.assertRaisesRegex(RuntimeError, "backward anterior"):
                    gpu_layer.step(0.1)

                gpu_layer.close()
                gpu_layer.close()
                cpu_layer.close()
                cpu_layer.close()

            self.assertEqual(release_counts[output_key], 1)
            self.assertEqual(release_counts[ids_key], 1)
            self.assertEqual(release_counts[grad_key], 1)
            for key in new_keys:
                self.assertEqual(release_counts[key], 1)
            self.assertTrue(all(buffer.closed for buffer in owned))
            self.assertFalse(set(self.backend.memory._buffers) - baseline_live)
        finally:
            if gpu_layer is not None and not gpu_layer._closed:
                gpu_layer.close()
            if cpu_layer is not None and not cpu_layer._closed:
                cpu_layer.close()

    def _assert_close(self, actual, expected) -> float:
        if actual and isinstance(actual[0], list):
            actual_flat = [value for row in actual for value in row]
        else:
            actual_flat = list(actual)
        if expected and isinstance(expected[0], list):
            expected_flat = [value for row in expected for value in row]
        else:
            expected_flat = list(expected)
        self.assertEqual(len(actual_flat), len(expected_flat))
        errors = [abs(got - want) for got, want in zip(actual_flat, expected_flat)]
        for index, (got, want) in enumerate(zip(actual_flat, expected_flat)):
            with self.subTest(index=index):
                self.assertLessEqual(
                    abs(got - want), ATOL + RTOL * abs(want),
                    msg=f"OpenCL={got}, CPU={want}",
                )
        return max(errors, default=0.0)


if __name__ == "__main__":
    unittest.main()
