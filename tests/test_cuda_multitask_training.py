"""CUDA equivalence gate for one resident multi-objective training step."""

from __future__ import annotations

import math
import os
import unittest
from collections import Counter
from typing import cast
from unittest.mock import patch

from kinera.backends.base import BackendBuffer, release_backend_buffers
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.nn.losses.multitask import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12
INPUTS = ([0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]])
TARGETS = {"char": 2, "syllable": 3, "boundary": 1.0}


@unittest.skipUnless(RUN_CUDA_TESTS, "set KINERA_RUN_CUDA_TESTS=1 on a CUDA runner")
class ResidentMultiTaskCudaEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = CUDABackend()
        cls.cuda.runtime.require_device(0)
        cls.cpu = CPUBackend()

    def _model(self, backend: CPUBackend | CUDABackend) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=3,
            syllable_vocab_size=4,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=731,
            backend=backend,
        )

    def _release_cuda_model(self, model: AutoregressiveModel) -> None:
        model.release_forward_state()
        release_backend_buffers(
            model.representation.char_embedding.weights,
            model.representation.syllable_embedding.weights,
            model.heads.char.weights,
            model.heads.char.bias,
            model.heads.syllable.weights,
            model.heads.syllable.bias,
            model.heads.boundary.weights,
            model.heads.boundary.bias,
        )

    def _host(self, backend: CUDABackend, value: object) -> object:
        if isinstance(value, BackendBuffer):
            return backend.buffer_to_host(value)
        return value

    @staticmethod
    def _flat(value: object) -> list[float]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [float(value)]
        if isinstance(value, list) and value and isinstance(value[0], list):
            return [
                float(cast(float | int, item))
                for row in cast(list[list[object]], value)
                for item in row
            ]
        if isinstance(value, list):
            return [float(cast(float | int, item)) for item in cast(list[object], value)]
        raise TypeError(f"expected a list-backed numeric array, got {type(value).__name__}")

    def _assert_close(self, actual: object, expected: object, label: str) -> float:
        actual_values = self._flat(actual)
        expected_values = self._flat(expected)
        self.assertEqual(len(actual_values), len(expected_values), label)
        errors = []
        for index, (got, want) in enumerate(zip(actual_values, expected_values)):
            with self.subTest(value=label, index=index):
                self.assertTrue(math.isfinite(got), f"{label}[{index}] is non-finite")
                error = abs(got - want)
                self.assertLessEqual(error, ATOL + RTOL * abs(want), label)
                errors.append(error)
        return max(errors, default=0.0)

    def test_one_multitask_train_step_matches_cpu_oracle(self) -> None:
        cpu_model = self._model(self.cpu)
        cuda_model = self._model(self.cuda)
        self.addCleanup(self._release_cuda_model, cuda_model)
        owned: list[object] = []
        max_error = 0.0
        try:
            cpu_logits = cpu_model._forward_heads_backend(*INPUTS)
            cuda_logits = cuda_model._forward_heads_backend(*INPUTS)
            owned.extend(cuda_logits.values())
            for name in ("char", "syllable", "boundary"):
                max_error = max(
                    max_error,
                    self._assert_close(
                        self._host(self.cuda, cuda_logits[name]),
                        cpu_logits[name],
                        f"{name}_logits",
                    ),
                )

            cpu_loss = MultiTaskLoss(backend=self.cpu)
            cuda_loss = MultiTaskLoss(backend=self.cuda)
            cpu_report = cpu_loss.forward(cpu_logits, TARGETS)
            cuda_report = cuda_loss.forward(cuda_logits, TARGETS)
            for name in ("char_loss", "syllable_loss", "boundary_loss", "total_loss"):
                max_error = max(
                    max_error,
                    self._assert_close(cuda_report[name], cpu_report[name], name),
                )

            cpu_logit_gradients = cpu_loss.backward(cpu_logits, TARGETS)
            cuda_logit_gradients = cuda_loss.backward(cuda_logits, TARGETS)
            owned.extend(cuda_logit_gradients.values())
            for name in ("char", "syllable", "boundary"):
                max_error = max(
                    max_error,
                    self._assert_close(
                        self._host(self.cuda, cuda_logit_gradients[name]),
                        cpu_logit_gradients[name],
                        f"{name}_logit_gradient",
                    ),
                )

            cpu_input_gradients = cpu_model.heads.backward(cpu_logit_gradients)
            cuda_input_gradients = cuda_model.heads.backward(cuda_logit_gradients)
            owned.extend(cuda_input_gradients.values())
            for name in ("char", "syllable", "boundary"):
                max_error = max(
                    max_error,
                    self._assert_close(
                        self._host(self.cuda, cuda_input_gradients[name]),
                        cpu_input_gradients[name],
                        f"{name}_head_input_gradient",
                    ),
                )

            cpu_char_input = cast(list[float], cpu_input_gradients["char"])
            cpu_syllable_input = cast(list[float], cpu_input_gradients["syllable"])
            cpu_boundary_input = cast(list[float], cpu_input_gradients["boundary"])
            cpu_context_size = len(INPUTS[0])
            cpu_pooled_gradient = [
                sum(
                    (
                        cpu_char_input[index],
                        cpu_syllable_input[index],
                        cpu_boundary_input[index],
                    )
                )
                for index in range(cpu_model.representation.output_dim)
            ]
            cpu_representation_gradient = [
                [value / cpu_context_size for value in cpu_pooled_gradient]
                for _ in range(cpu_context_size)
            ]
            cuda_pooled_gradient = self.cuda.sum_head_input_gradients(
                cast(BackendBuffer, cuda_input_gradients["char"]),
                cast(BackendBuffer, cuda_input_gradients["syllable"]),
                cast(BackendBuffer, cuda_input_gradients["boundary"]),
            )
            owned.append(cuda_pooled_gradient)
            max_error = max(
                max_error,
                self._assert_close(
                    self._host(self.cuda, cuda_pooled_gradient),
                    cpu_pooled_gradient,
                    "aggregated_head_gradient",
                ),
            )
            cuda_representation_gradient = self.cuda.mean_pool_backward(
                cuda_pooled_gradient, context_size=cpu_context_size
            )
            owned.append(cuda_representation_gradient)
            max_error = max(
                max_error,
                self._assert_close(
                    self._host(self.cuda, cuda_representation_gradient),
                    cpu_representation_gradient,
                    "mean_pool_backward",
                ),
            )

            cpu_feature_gradient = cpu_model.representation.backward(
                cpu_representation_gradient
            )
            cuda_feature_gradient = cuda_model.representation.backward(
                cuda_representation_gradient
            )
            owned.append(cuda_feature_gradient)
            max_error = max(
                max_error,
                self._assert_close(
                    self._host(self.cuda, cuda_feature_gradient),
                    cpu_feature_gradient,
                    "feature_gradient",
                ),
            )
            for name, cpu_embedding, cuda_embedding in (
                (
                    "char_embedding_gradient",
                    cpu_model.representation.char_embedding,
                    cuda_model.representation.char_embedding,
                ),
                (
                    "syllable_embedding_gradient",
                    cpu_model.representation.syllable_embedding,
                    cuda_model.representation.syllable_embedding,
                ),
            ):
                max_error = max(
                    max_error,
                    self._assert_close(
                        self._host(self.cuda, cuda_embedding.grad_weights),
                        cpu_embedding.grad_weights,
                        name,
                    ),
                )
        finally:
            release_backend_buffers(*owned)
            cuda_model.release_forward_state()

        step_cpu = self._model(self.cpu)
        step_cuda = self._model(self.cuda)
        self.addCleanup(self._release_cuda_model, step_cuda)
        downloads: list[tuple[tuple[int, ...], str]] = []
        copy_to_host = self.cuda.memory.copy_to_host

        def record_scalar_or_status_copy(buffer: CudaBuffer) -> object:
            downloads.append((buffer.shape, buffer.dtype))
            return copy_to_host(buffer)

        with patch.object(
            self.cuda.memory, "copy_to_host", side_effect=record_scalar_or_status_copy
        ):
            cpu_step_report = step_cpu.multitask_train_step(
                *INPUTS, targets=TARGETS, learning_rate=0.05
            )
            cuda_step_report = step_cuda.multitask_train_step(
                *INPUTS, targets=TARGETS, learning_rate=0.05
            )

        for name in ("char_loss", "syllable_loss", "boundary_loss", "total_loss"):
            max_error = max(
                max_error,
                self._assert_close(cuda_step_report[name], cpu_step_report[name], name),
            )
        for name, cpu_parameter, cuda_parameter in (
            ("char_embedding", step_cpu.representation.char_embedding.weights, step_cuda.representation.char_embedding.weights),
            ("syllable_embedding", step_cpu.representation.syllable_embedding.weights, step_cuda.representation.syllable_embedding.weights),
            ("char_head_weights", step_cpu.heads.char.weights, step_cuda.heads.char.weights),
            ("char_head_bias", step_cpu.heads.char.bias, step_cuda.heads.char.bias),
            ("syllable_head_weights", step_cpu.heads.syllable.weights, step_cuda.heads.syllable.weights),
            ("syllable_head_bias", step_cpu.heads.syllable.bias, step_cuda.heads.syllable.bias),
            ("boundary_head_weights", step_cpu.heads.boundary.weights, step_cuda.heads.boundary.weights),
            ("boundary_head_bias", step_cpu.heads.boundary.bias, step_cuda.heads.boundary.bias),
        ):
            max_error = max(
                max_error,
                self._assert_close(self._host(self.cuda, cuda_parameter), cpu_parameter, name),
            )
        self.assertEqual(
            Counter(downloads),
            Counter({((1,), "float64"): 3, ((1,), "int32"): 6}),
        )
        self.assertIsNone(step_cuda.last_logits)
        self.assertIsNone(step_cuda.last_pooled)
        self.assertIsNone(step_cuda.heads.char.grad_weights)
        self.assertIsNone(step_cuda.heads.syllable.grad_weights)
        self.assertIsNone(step_cuda.heads.boundary.grad_weights)
        self.assertTrue(math.isfinite(max_error))
        print("PHASE11B_MULTITASK_MAX_ABS_ERROR", format(max_error, ".17g"))


if __name__ == "__main__":
    unittest.main()
