"""CUDA equivalence gate for one resident multi-objective training step."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from kinera.backends.base import BackendBuffer, release_backend_buffers
from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.cuda import CUDABackend, CudaBuffer
from kinera.experiments import comparison
from kinera.experiments.comparison import run_controlled_experiments
from kinera.experiments.config import ExecutionConfig
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.losses.multitask import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples


RUN_CUDA_TESTS = os.environ.get("KINERA_RUN_CUDA_TESTS") == "1"
ATOL = 1e-12
RTOL = 1e-12
INPUTS = ([0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]])
TARGETS = {"char": 2, "syllable": 3, "boundary": 1.0}


def _assert_nested_close(
    testcase: unittest.TestCase,
    actual: object,
    expected: object,
    label: str,
) -> float:
    errors: list[float] = []

    def compare(got: object, want: object, path: tuple[str, ...] = ()) -> None:
        label_at_path = f"{label}{path}"
        if isinstance(got, dict) or isinstance(want, dict):
            testcase.assertIsInstance(got, dict, label_at_path)
            testcase.assertIsInstance(want, dict, label_at_path)
            got_dict = cast(dict[object, object], got)
            want_dict = cast(dict[object, object], want)
            testcase.assertEqual(got_dict.keys(), want_dict.keys(), label_at_path)
            for key in got_dict:
                compare(got_dict[key], want_dict[key], path + (str(key),))
            return
        if isinstance(got, list) or isinstance(want, list):
            testcase.assertIsInstance(got, list, label_at_path)
            testcase.assertIsInstance(want, list, label_at_path)
            got_list = cast(list[object], got)
            want_list = cast(list[object], want)
            testcase.assertEqual(len(got_list), len(want_list), label_at_path)
            for index, (got_item, want_item) in enumerate(zip(got_list, want_list)):
                compare(got_item, want_item, path + (str(index),))
            return
        if (
            isinstance(got, bool)
            or isinstance(want, bool)
            or got is None
            or want is None
        ):
            testcase.assertEqual(got, want, label_at_path)
            return
        if isinstance(got, int) and isinstance(want, int):
            testcase.assertEqual(got, want, label_at_path)
            return
        if isinstance(got, float) and isinstance(want, float):
            testcase.assertTrue(math.isfinite(got), f"{label_at_path} is non-finite")
            testcase.assertTrue(
                math.isfinite(want), f"expected {label_at_path} is non-finite"
            )
            error = abs(got - want)
            errors.append(error)
            testcase.assertLessEqual(error, ATOL + RTOL * abs(want), label_at_path)
            return
        if isinstance(got, (int, float)) or isinstance(want, (int, float)):
            # Numeric type changes are contract changes (especially for IDs).
            testcase.assertIs(type(got), type(want), label_at_path)
            testcase.assertEqual(got, want, label_at_path)
            return
        testcase.assertEqual(got, want, label_at_path)

    compare(actual, expected)
    return max(errors, default=0.0)


class CudaReportComparisonHelperTests(unittest.TestCase):
    def test_nested_descriptors_keep_numeric_and_metadata_semantics(self) -> None:
        expected = {
            "metric_results": {
                "char_accuracy": {
                    "value": 0.25,
                    "availability": "available",
                    "reason": None,
                    "support": {"examples": 4},
                }
            },
            "baseline_selection": {"classes": {"char": 3}, "version": 1},
            "validation_history": [
                {"epoch": 1, "split": "validation"},
                {"epoch": 2, "split": "validation"},
            ],
        }
        actual = {
            "metric_results": {
                "char_accuracy": {
                    "value": 0.25 + 1e-13,
                    "availability": "available",
                    "reason": None,
                    "support": {"examples": 4},
                }
            },
            "baseline_selection": {"classes": {"char": 3}, "version": 1},
            "validation_history": [
                {"epoch": 1, "split": "validation"},
                {"epoch": 2, "split": "validation"},
            ],
        }
        self.assertLessEqual(
            _assert_nested_close(self, actual, expected, "report"), 1e-12
        )
        descriptor = actual["metric_results"]["char_accuracy"]
        invalid_values = [
            {
                **actual,
                "metric_results": {
                    "char_accuracy": {**descriptor, "availability": "unavailable"}
                },
            },
            {
                **actual,
                "metric_results": {
                    "char_accuracy": {**descriptor, "support": {"examples": 5}}
                },
            },
            {
                **actual,
                "metric_results": {
                    "char_accuracy": {**descriptor, "value": None}
                },
            },
            {
                **actual,
                "validation_history": list(reversed(actual["validation_history"])),
            },
        ]
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(AssertionError):
                    _assert_nested_close(self, invalid_value, expected, "report")


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

    def _trainer_model(self, backend: CPUBackend | CUDABackend) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=5,
            syllable_vocab_size=3,
            seed=7,
            backend=backend,
        )

    def _release_cuda_model(self, model: AutoregressiveModel) -> None:
        model.close()

    def _host(self, backend: CUDABackend, value: object) -> object:
        if isinstance(value, BackendBuffer):
            return backend.buffer_to_host(value)
        return value

    def _assert_close(self, actual: object, expected: object, label: str) -> float:
        return _assert_nested_close(self, actual, expected, label)

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

    def test_trainer_and_evaluator_two_epochs_match_cpu_oracle(self) -> None:
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        train_examples = [
            {
                "context_ids": [0, 0],
                "context_syllable_ids": [0, 0],
                "context_feature_vectors": [[0.0] * 10, [0.0] * 10],
                "target_char_id": 3,
                "target_syllable_id": 1,
                "target_is_syllable_end": 1,
            },
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 2,
                "target_is_syllable_end": 1,
            },
        ]
        validation_examples = [
            {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 2],
                "context_feature_vectors": [[1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8],
                "target_char_id": 2,
                "target_syllable_id": 0,
                "target_is_syllable_end": 0,
            }
        ]
        cpu_model = self._trainer_model(self.cpu)

        def run_training(model: AutoregressiveModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            captured: list[dict[str, Any]] = []
            eval_step = model.multitask_eval_step

            def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
                result = eval_step(*args, **kwargs)
                captured.append(result)
                return result

            with patch.object(model, "multitask_eval_step", side_effect=capture):
                report = train_multitask_examples(
                    model,
                    train_examples,
                    vocabulary,
                    epochs=2,
                    learning_rate=0.1,
                )
            return report, captured

        def run_validation(
            model: AutoregressiveModel,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            captured: list[dict[str, Any]] = []
            eval_step = model.multitask_eval_step

            def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
                result = eval_step(*args, **kwargs)
                captured.append(result)
                return result

            with patch.object(model, "multitask_eval_step", side_effect=capture):
                report = evaluate_multitask_examples(
                    model,
                    validation_examples,
                    vocabulary,
                    syllable_unknown_id=0,
                )
            return report, captured

        cpu_training, cpu_train_steps = run_training(cpu_model)
        cpu_validation, cpu_validation_steps = run_validation(cpu_model)
        live_allocations: Counter[int] = Counter()
        allocation_sizes: dict[int, int] = {}
        unmatched_frees: list[int] = []
        freed_pointers: list[int] = []
        original_allocate = self.cuda.bridge.allocate
        original_free = self.cuda.bridge.free

        def tracked_allocate(device_id: int, nbytes: int) -> int:
            pointer = original_allocate(device_id, nbytes)
            live_allocations[pointer] += 1
            allocation_sizes[pointer] = nbytes
            return pointer

        def tracked_free(device_id: int, pointer: int) -> None:
            original_free(device_id, pointer)
            freed_pointers.append(pointer)
            if live_allocations[pointer] == 0:
                unmatched_frees.append(pointer)
            else:
                live_allocations[pointer] -= 1
                if live_allocations[pointer] == 0:
                    del live_allocations[pointer]
                    allocation_sizes.pop(pointer, None)

        cuda_model: AutoregressiveModel | None = None
        try:
            with (
                patch.object(self.cuda.bridge, "allocate", side_effect=tracked_allocate),
                patch.object(self.cuda.bridge, "free", side_effect=tracked_free),
            ):
                cuda_model = self._trainer_model(self.cuda)
                downloads: list[tuple[tuple[int, ...], str, str]] = []
                original_copy_to_host = self.cuda.memory.copy_to_host

                def record_copy_to_host(
                    buffer: CudaBuffer, *args: Any, **kwargs: Any
                ) -> object:
                    caller = sys._getframe(1).f_code.co_name
                    downloads.append((buffer.shape, buffer.dtype, caller))
                    return original_copy_to_host(buffer, *args, **kwargs)

                with patch.object(
                    self.cuda.memory,
                    "copy_to_host",
                    new=record_copy_to_host,
                ):
                    cuda_training, cuda_train_steps = run_training(cuda_model)
                    self.assertEqual(
                        Counter((shape, dtype) for shape, dtype, _ in downloads),
                        # Four train steps have six status scalars each. Four
                        # epoch-evaluation examples export three loss statuses
                        # and two argmax IDs apiece.
                        Counter({((1,), "float64"): 28, ((1,), "int32"): 44}),
                    )
                    downloads.clear()
                    cuda_validation, cuda_validation_steps = run_validation(cuda_model)
                    self.assertEqual(
                        Counter(downloads),
                        # Exact scalar D2H purpose for one validation example.
                        Counter(
                            {
                                ((1,), "float64", "cross_entropy_forward_buffer"): 2,
                                ((1,), "float64", "binary_cross_entropy_with_logits_forward_buffer"): 1,
                                ((1,), "float64", "buffer_to_host"): 1,
                                ((1,), "int32", "_check_numeric_status"): 3,
                                ((1,), "int32", "argmax"): 2,
                            }
                        ),
                    )

                self.assertEqual(cpu_training["epochs"], 2)
                self.assertEqual(cuda_training["epochs"], 2)
                self.assertEqual(len(cpu_training["history"]), 2)
                self.assertEqual(len(cuda_training["history"]), 2)
                max_error = 0.0
                for cpu_epoch, cuda_epoch in zip(
                    cpu_training["history"], cuda_training["history"]
                ):
                    for name, cpu_value in cpu_epoch.items():
                        if name == "boundary_target_distribution":
                            self.assertEqual(cuda_epoch[name], cpu_value)
                        else:
                            max_error = max(
                                max_error,
                                self._assert_close(cuda_epoch[name], cpu_value, name),
                            )
                for cpu_step, cuda_step in zip(cpu_train_steps, cuda_train_steps):
                    self.assertEqual(cpu_step["predictions"], cuda_step["predictions"])
                    for name, cpu_value in cpu_step["losses"].items():
                        max_error = max(
                            max_error,
                            self._assert_close(cuda_step["losses"][name], cpu_value, name),
                        )
                for name, cpu_value in cpu_validation.items():
                    if name == "boundary_target_distribution":
                        self.assertEqual(cuda_validation[name], cpu_value)
                    else:
                        max_error = max(
                            max_error,
                            self._assert_close(cuda_validation[name], cpu_value, name),
                        )
                self.assertEqual(len(cpu_validation_steps), 1)
                self.assertEqual(len(cuda_validation_steps), 1)
                self.assertEqual(
                    cpu_validation_steps[0]["predictions"],
                    cuda_validation_steps[0]["predictions"],
                )
                for name, cpu_value in cpu_validation_steps[0]["losses"].items():
                    max_error = max(
                        max_error,
                        self._assert_close(
                            cuda_validation_steps[0]["losses"][name], cpu_value, name
                        ),
                    )

                parameter_pairs = (
                    ("char_embedding", cpu_model.representation.char_embedding.weights, cuda_model.representation.char_embedding.weights),
                    ("syllable_embedding", cpu_model.representation.syllable_embedding.weights, cuda_model.representation.syllable_embedding.weights),
                    ("char_head_weights", cpu_model.heads.char.weights, cuda_model.heads.char.weights),
                    ("char_head_bias", cpu_model.heads.char.bias, cuda_model.heads.char.bias),
                    ("syllable_head_weights", cpu_model.heads.syllable.weights, cuda_model.heads.syllable.weights),
                    ("syllable_head_bias", cpu_model.heads.syllable.bias, cuda_model.heads.syllable.bias),
                    ("boundary_head_weights", cpu_model.heads.boundary.weights, cuda_model.heads.boundary.weights),
                    ("boundary_head_bias", cpu_model.heads.boundary.bias, cuda_model.heads.boundary.bias),
                )
                for name, cpu_value, cuda_value in parameter_pairs:
                    max_error = max(
                        max_error,
                        self._assert_close(self._host(self.cuda, cuda_value), cpu_value, name),
                    )
                print("PHASE12_GATE_A_MAX_ABS_ERROR", format(max_error, ".17g"))

                primary_parameter_values = (
                    cuda_model.representation.char_embedding.weights,
                    cuda_model.representation.syllable_embedding.weights,
                    cuda_model.heads.char.weights,
                    cuda_model.heads.char.bias,
                    cuda_model.heads.syllable.weights,
                    cuda_model.heads.syllable.bias,
                    cuda_model.heads.boundary.weights,
                    cuda_model.heads.boundary.bias,
                )
                primary_parameter_buffers = [
                    value
                    for value in primary_parameter_values
                    if isinstance(value, CudaBuffer)
                ]
                primary_parameter_pointers = {
                    buffer.pointer for buffer in primary_parameter_buffers
                }
                self.assertEqual(len(primary_parameter_buffers), 8)
                live_before_failure_model = live_allocations.copy()
                failed_model: AutoregressiveModel | None = None
                try:
                    failed_model = self._trainer_model(self.cuda)
                    failed_parameter_values = (
                        failed_model.representation.char_embedding.weights,
                        failed_model.representation.syllable_embedding.weights,
                        failed_model.heads.char.weights,
                        failed_model.heads.char.bias,
                        failed_model.heads.syllable.weights,
                        failed_model.heads.syllable.bias,
                        failed_model.heads.boundary.weights,
                        failed_model.heads.boundary.bias,
                    )
                    failed_parameter_buffers = [
                        value
                        for value in failed_parameter_values
                        if isinstance(value, CudaBuffer)
                    ]
                    failed_parameter_pointers = {
                        buffer.pointer for buffer in failed_parameter_buffers
                    }
                    self.assertEqual(len(failed_parameter_buffers), 8)
                    self.assertFalse(
                        primary_parameter_pointers & failed_parameter_pointers,
                        "separately owned models unexpectedly share CUDA parameter pointers",
                    )
                    example = validation_examples[0]
                    with patch.object(
                        self.cuda,
                        "argmax",
                        side_effect=RuntimeError("injected evaluation failure"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "injected evaluation failure"):
                            failed_model.multitask_eval_step(
                                example["context_ids"],
                                example["context_syllable_ids"],
                                example["context_feature_vectors"],
                                targets={
                                    "char": example["target_char_id"],
                                    "syllable": example["target_syllable_id"],
                                    "boundary": float(example["target_is_syllable_end"]),
                                },
                            )
                    self.assertIsNone(failed_model.last_logits)
                    self.assertIsNone(failed_model.last_pooled)
                    self.assertEqual(failed_model._last_head_logits_backend, {})
                finally:
                    if failed_model is not None:
                        failed_model.close()
                self.assertTrue(
                    all(not buffer.closed for buffer in primary_parameter_buffers),
                    "closing the failed model closed another model's parameter buffer",
                )
                self.assertEqual(
                    live_allocations,
                    live_before_failure_model,
                    "failed-model close changed live buffers owned by the primary model",
                )
        finally:
            if cuda_model is not None:
                with (
                    patch.object(self.cuda.bridge, "allocate", side_effect=tracked_allocate),
                    patch.object(self.cuda.bridge, "free", side_effect=tracked_free),
                ):
                    parameter_values = (
                        cuda_model.representation.char_embedding.weights,
                        cuda_model.representation.syllable_embedding.weights,
                        cuda_model.heads.char.weights,
                        cuda_model.heads.char.bias,
                        cuda_model.heads.syllable.weights,
                        cuda_model.heads.syllable.bias,
                        cuda_model.heads.boundary.weights,
                        cuda_model.heads.boundary.bias,
                    )
                    parameter_buffers = [
                        value for value in parameter_values if isinstance(value, CudaBuffer)
                    ]
                    parameter_pointers = {buffer.pointer for buffer in parameter_buffers}
                    managers_match = all(
                        buffer._manager is self.cuda.memory for buffer in parameter_buffers
                    )
                    backend_matches = cuda_model.backend is self.cuda
                    cuda_model.close()
                    buffers_closed = all(buffer.closed for buffer in parameter_buffers)
                    freed_parameter_pointers = parameter_pointers.issubset(
                        set(freed_pointers)
                    )
                    self.assertTrue(
                        managers_match and backend_matches and buffers_closed,
                        "model close ownership diagnostic: "
                        f"managers_match={managers_match}, backend_matches={backend_matches}, "
                        f"buffers_closed={buffers_closed}",
                    )
                    self.assertTrue(
                        freed_parameter_pointers,
                        "model.close() did not free every saved parameter pointer",
                    )
                    self.assertEqual(unmatched_frees, [])
                    remaining = [
                        (pointer, count, allocation_sizes.get(pointer))
                        for pointer, count in live_allocations.items()
                    ]
                    self.assertEqual(
                        live_allocations,
                        Counter(),
                        f"unreleased CUDA allocations after model.close(): {remaining}",
                    )
            cpu_model.close()

    def test_epoch_validation_and_b009_baselines_preserve_cuda_training(self) -> None:
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        syllable_vocabulary = SyllableVocabulary(
            ["<UNK_SYL>", "known-a", "known-b"]
        )

        def example(
            index: int, target_char_id: int, target_syllable_id: int
        ) -> dict[str, object]:
            return {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 2],
                "context_feature_vectors": [
                    [1.0] + [0.0] * 9,
                    [0.0, 1.0] + [0.0] * 8,
                ],
                "target_char_id": target_char_id,
                "target_syllable_id": target_syllable_id,
                "target_is_syllable_end": index % 2,
            }

        train_examples = [
            example(0, 3, 1),
            example(1, 3, 1),
            example(2, 4, syllable_vocabulary.unk_id),
            example(3, vocabulary.end_id, 2),
        ]
        validation_examples = [
            example(0, 3, 1),
            example(1, 4, 1),
            example(2, 4, syllable_vocabulary.unk_id),
            example(3, vocabulary.end_id, 2),
        ]
        run_args = {
            "syllable_vocab_size": len(syllable_vocabulary),
            "syllable_vocabulary": syllable_vocabulary,
            "seeds": [41],
            "epochs": 2,
            "learning_rate": 0.05,
            "max_examples": 4,
            "variants": ("A",),
        }

        counts: dict[str, Counter[str]] = {
            "disabled": Counter(),
            "enabled": Counter(),
        }
        update_orders: dict[str, list[object]] = {"disabled": [], "enabled": []}
        parameter_snapshots: dict[str, list[object]] = {}
        d2h_shapes: list[tuple[tuple[int, ...], str]] = []
        live_allocations: Counter[int] = Counter()
        unmatched_frees: list[int] = []
        case = {"name": "disabled"}
        capturing_parameters = {"active": False}
        models_closed = 0

        original_train_step = AutoregressiveModel.multitask_train_step
        original_eval_step = AutoregressiveModel.multitask_eval_step
        original_backward = MultiTaskLoss.backward
        original_close = AutoregressiveModel.close
        original_allocate = self.cuda.bridge.allocate
        original_free = self.cuda.bridge.free
        original_copy_to_host = self.cuda.memory.copy_to_host

        def tracked_train_step(
            model: AutoregressiveModel, *args: Any, **kwargs: Any
        ) -> object:
            name = case["name"]
            counts[name]["train_step"] += 1
            update_orders[name].append(
                (args[0], args[1], args[2], kwargs["targets"], kwargs["learning_rate"])
            )
            return original_train_step(model, *args, **kwargs)

        def tracked_eval_step(
            model: AutoregressiveModel, *args: Any, **kwargs: Any
        ) -> object:
            counts[case["name"]]["eval_step"] += 1
            return original_eval_step(model, *args, **kwargs)

        def tracked_backward(loss: MultiTaskLoss, *args: Any, **kwargs: Any) -> object:
            counts[case["name"]]["backward"] += 1
            return original_backward(loss, *args, **kwargs)

        def tracked_allocate(device_id: int, nbytes: int) -> int:
            pointer = original_allocate(device_id, nbytes)
            live_allocations[pointer] += 1
            return pointer

        def tracked_free(device_id: int, pointer: int) -> None:
            original_free(device_id, pointer)
            if live_allocations[pointer] == 0:
                unmatched_frees.append(pointer)
            else:
                live_allocations[pointer] -= 1
                if live_allocations[pointer] == 0:
                    del live_allocations[pointer]

        def model_parameters(model: AutoregressiveModel) -> list[CudaBuffer]:
            return [
                model.representation.char_embedding.weights,
                model.representation.syllable_embedding.weights,
                model.heads.char.weights,
                model.heads.char.bias,
                model.heads.syllable.weights,
                model.heads.syllable.bias,
                model.heads.boundary.weights,
                model.heads.boundary.bias,
            ]

        def tracked_close(model: AutoregressiveModel) -> None:
            nonlocal models_closed
            capturing_parameters["active"] = True
            try:
                parameter_snapshots[case["name"]] = [
                    self.cuda.buffer_to_host(buffer)
                    for buffer in model_parameters(model)
                ]
            finally:
                capturing_parameters["active"] = False
            original_close(model)
            models_closed += 1
            self.assertTrue(
                all(buffer.closed for buffer in model_parameters(model)),
                "model.close() deixou buffers de parâmetros abertos",
            )
            self.assertEqual(
                live_allocations,
                Counter(),
                "model.close() deixou alocações CUDA ativas",
            )

        def tracked_copy_to_host(
            buffer: CudaBuffer, *args: Any, **kwargs: Any
        ) -> object:
            if not capturing_parameters["active"]:
                d2h_shapes.append((buffer.shape, buffer.dtype))
            return original_copy_to_host(buffer, *args, **kwargs)

        def run_case(name: str, cadence: int | None) -> tuple[dict[str, Any], int]:
            case["name"] = name
            with patch.object(
                comparison,
                "evaluate_multitask_examples",
                wraps=evaluate_multitask_examples,
            ) as validation_evaluator:
                report = run_controlled_experiments(
                    train_examples,
                    validation_examples,
                    vocabulary,
                    **run_args,
                    validation_every_n_epochs=cadence,
                    execution_config=ExecutionConfig(backend="cuda", workers=1),
                )
            return report, validation_evaluator.call_count

        with (
            patch.object(self.cuda.memory, "copy_to_host", new=tracked_copy_to_host),
            patch.object(self.cuda.bridge, "allocate", side_effect=tracked_allocate),
            patch.object(self.cuda.bridge, "free", side_effect=tracked_free),
            patch.object(AutoregressiveModel, "multitask_train_step", new=tracked_train_step),
            patch.object(AutoregressiveModel, "multitask_eval_step", new=tracked_eval_step),
            patch.object(MultiTaskLoss, "backward", new=tracked_backward),
            patch.object(AutoregressiveModel, "close", new=tracked_close),
            patch.object(comparison, "get_backend", return_value=self.cuda),
        ):
            disabled_report, disabled_validation_calls = run_case("disabled", None)
            enabled_report, enabled_validation_calls = run_case("enabled", 1)

        self.assertEqual(disabled_validation_calls, 2)  # initial + final
        self.assertEqual(enabled_validation_calls, 3)  # initial + epoch 1 + final
        self.assertEqual(models_closed, 2)
        self.assertEqual(live_allocations, Counter())
        self.assertEqual(unmatched_frees, [])
        self.assertTrue(d2h_shapes)
        self.assertTrue(
            all(shape == (1,) and dtype in {"float64", "int32"} for shape, dtype in d2h_shapes),
            f"validação exportou D2H não escalar: {d2h_shapes}",
        )

        disabled_run = disabled_report["runs"]["A"][0]
        enabled_run = enabled_report["runs"]["A"][0]
        self.assertNotIn("validation_history", disabled_run)
        self.assertEqual(
            [item["epoch"] for item in enabled_run["validation_history"]], [1, 2]
        )
        self.assertEqual(
            enabled_run["validation_history"][-1]["split"], "validation"
        )
        self.assertEqual(
            {
                key: value
                for key, value in enabled_run["validation_history"][-1].items()
                if key not in {"epoch", "split"}
            },
            enabled_run["final_validation"],
        )
        self.assertEqual(disabled_run["training"], enabled_run["training"])
        self.assertEqual(
            disabled_run["initial_validation"], enabled_run["initial_validation"]
        )
        self.assertEqual(
            disabled_run["final_validation"], enabled_run["final_validation"]
        )
        self.assertEqual(disabled_report["metrics"], enabled_report["metrics"])
        self.assertEqual(update_orders["disabled"], update_orders["enabled"])
        self.assertEqual(
            counts["disabled"]["train_step"], counts["enabled"]["train_step"]
        )
        self.assertEqual(counts["disabled"]["train_step"], 8)
        self.assertEqual(
            counts["disabled"]["backward"], counts["enabled"]["backward"]
        )
        self.assertEqual(counts["disabled"]["backward"], 8)
        self.assertEqual(counts["disabled"]["eval_step"], 16)
        self.assertEqual(counts["enabled"]["eval_step"], 20)
        for disabled_parameters, enabled_parameters in zip(
            parameter_snapshots["disabled"], parameter_snapshots["enabled"]
        ):
            self._assert_close(enabled_parameters, disabled_parameters, "final_parameters")

        validation = enabled_run["final_validation"]
        self.assertEqual(validation["metrics_schema_version"], 1)
        self.assertEqual(
            validation["baseline_selection"]["classes"],
            {
                "char_baseline_majority": 3,
                "syllable_baseline_majority_all": 1,
                "syllable_baseline_majority_known": 1,
            },
        )
        self.assertEqual(
            validation["baseline_selection"]["syllable_known_scope"],
            {
                "excluded_syllable_id": syllable_vocabulary.unk_id,
                "excluded_character_id": vocabulary.end_id,
            },
        )
        expected_baseline_supports = {
            "char_baseline_majority": {"training_targets": 4, "evaluation_targets": 4},
            "syllable_baseline_majority_all": {
                "training_targets": 4,
                "evaluation_targets": 4,
            },
            "syllable_baseline_majority_known": {
                "training_targets": 2,
                "evaluation_targets": 2,
            },
        }
        for name, support in expected_baseline_supports.items():
            result = validation["metric_results"][name]
            self.assertEqual(result["availability"], "available")
            self.assertEqual(result["support"], support)
        self.assertEqual(
            validation["metric_results"]["char_baseline_majority"]["value"], 0.25
        )
        self.assertEqual(
            validation["metric_results"]["syllable_baseline_majority_all"]["value"],
            0.5,
        )
        self.assertEqual(
            validation["metric_results"]["syllable_baseline_majority_known"]["value"],
            1.0,
        )
        json.dumps(enabled_report, allow_nan=False)

    def test_controlled_runner_matches_cpu_cuda_and_resumes_without_duplicate_runs(self) -> None:
        vocabulary = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", *(f"char-{index}" for index in range(48))]
        )
        self.assertEqual(len(vocabulary), 51)
        train_examples = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 1],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 2,
                "target_is_syllable_end": 1,
            },
            {
                "context_ids": [0, 4],
                "context_syllable_ids": [0, 2],
                "context_feature_vectors": [[0.0] * 10, [0.0, 1.0] + [0.0] * 8],
                "target_char_id": 5,
                "target_syllable_id": 3,
                "target_is_syllable_end": 0,
            },
        ]
        validation_examples = [
            {
                "context_ids": [4, 5],
                "context_syllable_ids": [2, 3],
                "context_feature_vectors": [[1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8],
                "target_char_id": 6,
                "target_syllable_id": 4,
                "target_is_syllable_end": 1,
            },
            {
                "context_ids": [5, 6],
                "context_syllable_ids": [3, 4],
                "context_feature_vectors": [[0.0, 1.0] + [0.0] * 8, [0.0, 0.0, 1.0] + [0.0] * 7],
                "target_char_id": 7,
                "target_syllable_id": 5,
                "target_is_syllable_end": 0,
            },
        ]
        run_args = {
            "syllable_vocab_size": 3932,
            "syllable_vocabulary": SyllableVocabulary([
                "<UNK_SYL>",
                *(f"<FIXTURE_SYL_{index}>" for index in range(1, 3932)),
            ]),
            "seeds": [7],
            "epochs": 1,
            "learning_rate": 0.05,
            "max_examples": 2,
            "variants": ("A", "C", "D"),
        }
        cpu_report = run_controlled_experiments(
            train_examples,
            validation_examples,
            vocabulary,
            **run_args,
            execution_config=ExecutionConfig(backend="cpu", workers=1),
        )
        self.assertEqual(cpu_report["execution_config"], {"backend": "cpu", "workers": 1})
        for variant in ("A", "C", "D"):
            self.assertEqual(cpu_report["parameter_counts"][variant], 139432)

        live_allocations: Counter[int] = Counter()
        unmatched_frees: list[int] = []
        original_allocate = self.cuda.bridge.allocate
        original_free = self.cuda.bridge.free
        original_init = AutoregressiveModel.__init__
        original_close = AutoregressiveModel.close
        model_counts = {"created": 0, "closed": 0}

        def tracked_allocate(device_id: int, nbytes: int) -> int:
            pointer = original_allocate(device_id, nbytes)
            live_allocations[pointer] += 1
            return pointer

        def tracked_free(device_id: int, pointer: int) -> None:
            original_free(device_id, pointer)
            if live_allocations[pointer] == 0:
                unmatched_frees.append(pointer)
            else:
                live_allocations[pointer] -= 1
                if live_allocations[pointer] == 0:
                    del live_allocations[pointer]

        def tracked_init(model: AutoregressiveModel, *args: Any, **kwargs: Any) -> None:
            if model_counts["created"]:
                self.assertEqual(
                    model_counts["closed"],
                    model_counts["created"],
                    "next CUDA model started before prior model.close()",
                )
                self.assertEqual(
                    live_allocations,
                    Counter(),
                    "next CUDA model started with buffers owned by a previous run",
                )
            original_init(model, *args, **kwargs)
            model_counts["created"] += 1

        def tracked_close(model: AutoregressiveModel) -> None:
            original_close(model)
            model_counts["closed"] += 1
            self.assertEqual(
                live_allocations,
                Counter(),
                "model.close() left CUDA allocations live",
            )

        def assert_report_equivalent(expected: object, actual: object) -> list[float]:
            maximum_error = [0.0]
            volatile_fields = {
                "duration_seconds",
                "pid",
                "max_rss_kb",
                "hardware_fingerprint",
                "runtime_fingerprint",
            }

            def compare(left: object, right: object, path: tuple[str, ...] = ()) -> None:
                if isinstance(left, dict) and isinstance(right, dict):
                    left_keys = set(left) - volatile_fields
                    right_keys = set(right) - volatile_fields
                    if path == ("execution_config",):
                        left_keys.discard("backend")
                        right_keys.discard("backend")
                    self.assertEqual(left_keys, right_keys, f"report keys differ at {path}")
                    for key in left_keys:
                        compare(left[key], right[key], path + (str(key),))
                    return
                if isinstance(left, list) and isinstance(right, list):
                    self.assertEqual(len(left), len(right), f"report list length differs at {path}")
                    for index, (left_item, right_item) in enumerate(zip(left, right)):
                        compare(left_item, right_item, path + (str(index),))
                    return
                if (
                    isinstance(left, bool)
                    or isinstance(right, bool)
                    or isinstance(left, str)
                    or isinstance(right, str)
                    or left is None
                    or right is None
                ):
                    self.assertEqual(left, right, f"report value differs at {path}")
                    return
                if isinstance(left, int) or isinstance(right, int):
                    self.assertIs(type(left), type(right), f"report numeric type differs at {path}")
                    self.assertEqual(left, right, f"report integer differs at {path}")
                    return
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    self.assertTrue(math.isfinite(float(left)), f"report value is non-finite at {path}")
                    self.assertTrue(math.isfinite(float(right)), f"expected report value is non-finite at {path}")
                    error = abs(float(left) - float(right))
                    maximum_error[0] = max(maximum_error[0], error)
                    self.assertLessEqual(
                        error,
                        ATOL + RTOL * abs(float(right)),
                        f"report numeric value differs at {path}",
                    )
                    return
                self.assertEqual(left, right, f"report value differs at {path}")

            compare(expected, actual)
            return maximum_error

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cuda-report.json"
            with (
                patch.object(comparison, "get_backend", return_value=self.cuda) as backend_factory,
                patch.object(self.cuda.bridge, "allocate", side_effect=tracked_allocate),
                patch.object(self.cuda.bridge, "free", side_effect=tracked_free),
                patch.object(AutoregressiveModel, "__init__", new=tracked_init),
                patch.object(AutoregressiveModel, "close", new=tracked_close),
            ):
                cuda_report = run_controlled_experiments(
                    train_examples,
                    validation_examples,
                    vocabulary,
                    **run_args,
                    checkpoint_path=checkpoint,
                    execution_config=ExecutionConfig(backend="cuda", workers=1),
                )
            backend_factory.assert_called_once_with("cuda")
            self.assertEqual(model_counts, {"created": 3, "closed": 3})
            self.assertEqual(live_allocations, Counter())
            self.assertEqual(unmatched_frees, [])
            self.assertEqual(cuda_report["execution_config"], {"backend": "cuda", "workers": 1})
            self.assertEqual(cuda_report["completed_runs"], 3)
            for variant in ("A", "C", "D"):
                self.assertEqual(cuda_report["parameter_counts"][variant], 139432)
                self.assertEqual(len(cuda_report["runs"][variant]), 1)
                self.assertEqual(cuda_report["runs"][variant][0]["status"], "completed")

            max_error = assert_report_equivalent(cpu_report, cuda_report)[0]
            persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["execution_config"], {"backend": "cuda", "workers": 1}
            )
            self.assertEqual(
                persisted["experiment_config"], cpu_report["experiment_config"]
            )
            print("PHASE12_GATE_B_MAX_ABS_ERROR", format(max_error, ".17g"))

            with patch.object(
                comparison,
                "_create_execution_backend",
                side_effect=AssertionError("backend created before resume validation"),
            ) as backend_factory:
                with self.assertRaisesRegex(ValueError, "backend"):
                    run_controlled_experiments(
                        train_examples,
                        validation_examples,
                        vocabulary,
                        **run_args,
                        checkpoint_path=checkpoint,
                        resume=True,
                        execution_config=ExecutionConfig(backend="cpu", workers=1),
                    )
                backend_factory.assert_not_called()

            with (
                patch.object(comparison, "get_backend", return_value=self.cuda),
                patch.object(
                    comparison,
                    "AutoregressiveModel",
                    side_effect=AssertionError("completed CUDA run was duplicated"),
                ),
                patch.object(
                    comparison,
                    "train_multitask_examples",
                    side_effect=AssertionError("completed CUDA training was repeated"),
                ),
            ):
                resumed_report = run_controlled_experiments(
                    train_examples,
                    validation_examples,
                    vocabulary,
                    **run_args,
                    checkpoint_path=checkpoint,
                    resume=True,
                    execution_config=ExecutionConfig(backend="cuda", workers=1),
                )
            self.assertEqual(resumed_report["completed_runs"], 3)
            self.assertEqual(
                {variant: len(resumed_report["runs"][variant]) for variant in ("A", "C", "D")},
                {"A": 1, "C": 1, "D": 1},
            )


if __name__ == "__main__":
    unittest.main()
