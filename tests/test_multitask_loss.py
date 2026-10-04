import math
import unittest
from unittest.mock import patch
from typing import Any

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.heads import MultiTaskHeads
from kinera.nn.losses import MultiTaskLoss, binary_cross_entropy_with_logits
from kinera.training.autoregressive import AutoregressiveModel


class MultiTaskLossTests(unittest.TestCase):
    def test_binary_cross_entropy_with_logits_is_stable(self) -> None:
        self.assertLess(binary_cross_entropy_with_logits(1000.0, 1.0), 1e-10)
        self.assertLess(binary_cross_entropy_with_logits(-1000.0, 0.0), 1e-10)

    def test_forward_returns_weighted_component_losses(self) -> None:
        loss = MultiTaskLoss(char_weight=1.0, syllable_weight=0.7, boundary_weight=0.3)
        outputs = {
            "char": [1.0, 0.0],
            "syllable": [0.0, 1.0, -1.0],
            "boundary": [0.5],
        }

        report = loss.forward(
            outputs,
            {"char": 0, "syllable": 1, "boundary": 1.0},
        )

        self.assertEqual(set(report), {"char_loss", "syllable_loss", "boundary_loss", "total_loss"})
        self.assertAlmostEqual(
            report["total_loss"],
            report["char_loss"] + 0.7 * report["syllable_loss"] + 0.3 * report["boundary_loss"],
        )

    def test_backward_returns_gradients_for_all_heads(self) -> None:
        loss = MultiTaskLoss()
        outputs = {
            "char": [0.0, 1.0],
            "syllable": [1.0, 0.0],
            "boundary": [-1.0],
        }
        targets = {"char": 1, "syllable": 0, "boundary": 1.0}

        gradients = loss.backward(outputs, targets)

        self.assertEqual(set(gradients), {"char", "syllable", "boundary"})
        self.assertEqual(len(gradients["char"]), 2)
        self.assertEqual(len(gradients["syllable"]), 2)
        self.assertEqual(len(gradients["boundary"]), 1)
        self.assertTrue(all(math.isfinite(value) for values in gradients.values() for value in values))

    def test_invalid_targets_are_rejected(self) -> None:
        loss = MultiTaskLoss()
        outputs = {"char": [0.0, 1.0], "syllable": [0.0, 1.0], "boundary": [0.0]}

        with self.assertRaises(ValueError):
            loss.forward(outputs, {"char": 2, "syllable": 0, "boundary": 0.0})
        with self.assertRaises(ValueError):
            loss.forward(outputs, {"char": 0, "syllable": 0, "boundary": 2.0})

    def test_forward_and_backward_reject_invalid_inputs_with_exact_errors(self) -> None:
        valid_outputs: dict[str, Any] = {
            "char": [0.0, 1.0],
            "syllable": [0.0, 1.0],
            "boundary": [0.0],
        }
        valid_targets: dict[str, Any] = {
            "char": 0,
            "syllable": 1,
            "boundary": 0.0,
        }
        empty_char = {**valid_outputs, "char": []}
        container_char = {**valid_outputs, "char": (0.0, 1.0)}
        cases: tuple[tuple[str, Any, Any, type[Exception], str], ...] = (
            (
                "outputs_container",
                [],
                valid_targets,
                ValueError,
                "outputs deve conter char, syllable e boundary",
            ),
            (
                "char_container",
                container_char,
                valid_targets,
                ValueError,
                "outputs[char] deve ser uma lista não vazia",
            ),
            (
                "char_empty",
                empty_char,
                valid_targets,
                ValueError,
                "outputs[char] deve ser uma lista não vazia",
            ),
            (
                "char_nan",
                {**valid_outputs, "char": [math.nan, 1.0]},
                valid_targets,
                ValueError,
                "outputs[char][0] deve ser finito",
            ),
            (
                "char_inf",
                {**valid_outputs, "char": [math.inf, 1.0]},
                valid_targets,
                ValueError,
                "outputs[char][0] deve ser finito",
            ),
            (
                "char_bool",
                {**valid_outputs, "char": [True, 1.0]},
                valid_targets,
                TypeError,
                "outputs[char][0] deve ser numérico",
            ),
            (
                "char_string",
                {**valid_outputs, "char": ["0", 1.0]},
                valid_targets,
                TypeError,
                "outputs[char][0] deve ser numérico",
            ),
            (
                "target_out_of_range",
                valid_outputs,
                {**valid_targets, "char": 2},
                ValueError,
                "targets[char] fora do vocabulário",
            ),
        )

        for method_name in ("forward", "backward"):
            for case_name, outputs, targets, error_type, expected_message in cases:
                with self.subTest(method=method_name, case=case_name):
                    method = getattr(MultiTaskLoss(), method_name)
                    with self.assertRaises(error_type) as raised:
                        method(outputs, targets)
                    self.assertEqual(str(raised.exception), expected_message)


class MultiTaskGradientCheckingTests(unittest.TestCase):
    def test_combined_loss_gradient_matches_finite_difference_for_each_head(self) -> None:
        heads = MultiTaskHeads(input_dim=2, char_vocab_size=2, syllable_vocab_size=2, seed=7)
        loss = MultiTaskLoss()
        inputs = [0.4, -0.3]
        targets = {"char": 1, "syllable": 0, "boundary": 1.0}
        epsilon = 1e-6

        outputs = heads.forward(inputs)
        gradients = loss.backward(outputs, targets)
        input_gradients = heads.backward(gradients)
        analytic_gradients = {
            "char": heads.char.grad_weights[0][0],
            "syllable": heads.syllable.grad_weights[0][0],
            "boundary": heads.boundary.grad_weights[0][0],
        }

        for name, layer in (
            ("char", heads.char),
            ("syllable", heads.syllable),
            ("boundary", heads.boundary),
        ):
            original = layer.weights[0][0]
            layer.weights[0][0] = original + epsilon
            loss_plus = loss.forward(heads.forward(inputs), targets)["total_loss"]
            layer.weights[0][0] = original - epsilon
            loss_minus = loss.forward(heads.forward(inputs), targets)["total_loss"]
            layer.weights[0][0] = original
            numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
            self.assertAlmostEqual(
                numerical,
                analytic_gradients[name],
                delta=1e-5,
                msg=name,
            )

        self.assertEqual(len(input_gradients["char"]), 2)
        self.assertEqual(len(input_gradients["syllable"]), 2)
        self.assertEqual(len(input_gradients["boundary"]), 2)


class MultiTaskModelTests(unittest.TestCase):
    def test_multitask_train_step_updates_all_heads(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=3,
            syllable_vocab_size=4,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=7,
        )
        inputs = ([0, 1], [0, 1], [[1.0, 0.0], [0.0, 1.0]])
        before = {
            name: [row.copy() for row in layer.weights]
            for name, layer in (
                ("char", model.heads.char),
                ("syllable", model.heads.syllable),
                ("boundary", model.heads.boundary),
            )
        }

        report = model.multitask_train_step(
            *inputs,
            targets={"char": 2, "syllable": 3, "boundary": 1.0},
            learning_rate=0.1,
        )

        self.assertIn("total_loss", report)
        self.assertNotEqual(before["char"], model.heads.char.weights)
        self.assertNotEqual(before["syllable"], model.heads.syllable.weights)
        self.assertNotEqual(before["boundary"], model.heads.boundary.weights)


class MultiTaskBackendIntegrationTests(unittest.TestCase):
    def test_multitask_loss_delegates_probability_operations_to_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def cross_entropy(self, logits, target_id):
                self.calls.append("cross_entropy")
                return super().cross_entropy(logits, target_id)

            def cross_entropy_gradient(self, logits, target_id):
                self.calls.append("cross_entropy_gradient")
                return super().cross_entropy_gradient(logits, target_id)

            def binary_cross_entropy_with_logits(self, logit, target):
                self.calls.append("binary_cross_entropy_with_logits")
                return super().binary_cross_entropy_with_logits(logit, target)

            def binary_cross_entropy_with_logits_gradient(self, logit, target):
                self.calls.append("binary_cross_entropy_with_logits_gradient")
                return super().binary_cross_entropy_with_logits_gradient(logit, target)

        backend = RecordingBackend()
        loss = MultiTaskLoss(backend=backend)
        outputs = {
            "char": [0.0, 1.0],
            "syllable": [1.0, 0.0],
            "boundary": [-1.0],
        }
        targets = {"char": 1, "syllable": 0, "boundary": 1.0}

        loss.forward(outputs, targets)
        loss.backward(outputs, targets)

        self.assertEqual(
            backend.calls,
            [
                "cross_entropy",
                "cross_entropy",
                "binary_cross_entropy_with_logits",
                "cross_entropy_gradient",
                "cross_entropy_gradient",
                "binary_cross_entropy_with_logits_gradient",
            ],
        )


class AutoregressiveModelBackendIntegrationTests(unittest.TestCase):
    def test_model_injects_one_backend_into_embeddings_heads_and_losses(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def linear_forward(self, weights, bias, inputs):
                self.calls.append("linear_forward")
                return super().linear_forward(weights, bias, inputs)

            def embedding_forward(self, weights, ids):
                self.calls.append("embedding_forward")
                return super().embedding_forward(weights, ids)

            def cross_entropy(self, logits, target_id):
                self.calls.append("cross_entropy")
                return super().cross_entropy(logits, target_id)

        backend = RecordingBackend()
        model = AutoregressiveModel(
            char_vocab_size=4,
            syllable_vocab_size=4,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            feature_dim=2,
            seed=7,
            backend=backend,
        )

        model.multitask_train_step(
            [0, 1],
            [0, 1],
            [[1.0, 0.0], [0.0, 1.0]],
            targets={"char": 2, "syllable": 3, "boundary": 1.0},
            learning_rate=0.01,
        )

        self.assertIs(model.representation.char_embedding.backend, backend)
        self.assertIs(model.representation.syllable_embedding.backend, backend)
        self.assertIs(model.heads.char.backend, backend)
        self.assertIs(model.heads.syllable.backend, backend)
        self.assertIs(model.heads.boundary.backend, backend)
        self.assertGreaterEqual(backend.calls.count("linear_forward"), 3)
        self.assertGreaterEqual(backend.calls.count("embedding_forward"), 2)
        self.assertGreaterEqual(backend.calls.count("cross_entropy"), 2)


class ResidentMultiTaskLossTests(unittest.TestCase):
    class Buffer:
        def __init__(self, shape: tuple[int, ...], device_id: int = 0) -> None:
            self.device = "cuda"
            self.device_id = device_id
            self.shape = shape
            self.dtype = "float64"
            self.nbytes = math.prod(shape) * 8
            self.layout = "row_major"
            self.owns_memory = False
            self.closed = False

    class Backend(CPUBackend):
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []
            self.resident_calls: list[tuple[object, ...]] = []

        def cross_entropy_forward_buffer(self, logits, target_id):
            self.calls.append(("ce", logits, target_id))
            return 0.2

        def binary_cross_entropy_with_logits_forward_buffer(self, logits, target):
            self.calls.append(("bce", logits, target))
            return 0.4

        def cross_entropy_gradient_buffer(self, logits, target_id, scale):
            self.calls.append(("ce_gradient", logits, target_id, scale))
            return ResidentMultiTaskLossTests.Buffer(logits.shape, logits.device_id)

        def binary_cross_entropy_with_logits_gradient_buffer(self, logits, target, scale):
            self.calls.append(("bce_gradient", logits, target, scale))
            return ResidentMultiTaskLossTests.Buffer(logits.shape, logits.device_id)

        def sum_head_input_gradients(self, char_gradient, syllable_gradient, boundary_gradient):
            self.resident_calls.append(("sum_heads", char_gradient, syllable_gradient, boundary_gradient))
            return ResidentMultiTaskLossTests.Buffer(char_gradient.shape, char_gradient.device_id)

        def mean_pool_backward(self, grad_pooled, *, context_size):
            self.resident_calls.append(("mean_pool_backward", grad_pooled, context_size))
            return ResidentMultiTaskLossTests.Buffer(
                (context_size, grad_pooled.shape[0]), grad_pooled.device_id
            )

        def buffer_to_host(self, buffer):
            raise AssertionError("resident loss path must not download logits")

    def test_forward_and_backward_keep_all_logits_and_gradients_backend_native(self) -> None:
        backend = self.Backend()
        loss = MultiTaskLoss(backend=backend)
        outputs = {
            "char": self.Buffer((3,)),
            "syllable": self.Buffer((4,)),
            "boundary": self.Buffer((1,)),
        }
        targets = {"char": 1, "syllable": 2, "boundary": 1.0}

        report = loss.forward(outputs, targets)
        gradients = loss.backward(outputs, targets)

        self.assertEqual(report["char_loss"], 0.2)
        self.assertEqual(report["syllable_loss"], 0.2)
        self.assertEqual(report["boundary_loss"], 0.4)
        self.assertAlmostEqual(report["total_loss"], 0.46)
        self.assertEqual(set(gradients), {"char", "syllable", "boundary"})
        self.assertTrue(all(isinstance(value, self.Buffer) for value in gradients.values()))
        self.assertEqual(
            [(call[0], call[-1]) for call in backend.calls
             if isinstance(call[0], str) and call[0].endswith("gradient")],
            [("ce_gradient", 1.0), ("ce_gradient", 0.7), ("bce_gradient", 0.3)],
        )
        self.assertEqual([call[0] for call in backend.calls], [
            "ce", "ce", "bce", "ce_gradient", "ce_gradient", "bce_gradient"
        ])

    def test_model_multitask_step_uses_backend_native_forward_and_backward_chain(self) -> None:
        backend = self.Backend()
        model = AutoregressiveModel(
            char_vocab_size=3,
            syllable_vocab_size=4,
            char_embedding_dim=2,
            syllable_embedding_dim=3,
            feature_dim=2,
            seed=7,
            backend=backend,
        )
        loss = MultiTaskLoss(backend=backend)
        outputs = {
            "char": self.Buffer((3,)),
            "syllable": self.Buffer((4,)),
            "boundary": self.Buffer((1,)),
        }
        head_gradients = {
            name: self.Buffer((model.representation.output_dim,))
            for name in ("char", "syllable", "boundary")
        }

        def resident_forward(*args):
            model.last_context_size = 2
            return outputs

        with (
            patch.object(model, "_forward_heads_backend", side_effect=resident_forward) as forward_backend,
            patch.object(
                model,
                "forward_heads",
                side_effect=AssertionError("training must not export logits"),
            ),
            patch.object(model.heads, "backward", return_value=head_gradients) as heads_backward,
            patch.object(model.representation, "backward") as representation_backward,
            patch.object(model.heads, "step") as heads_step,
            patch.object(model.representation, "step") as representation_step,
        ):
            report = model.multitask_train_step(
                [0, 1],
                [0, 1],
                [[1.0, 0.0], [0.0, 1.0]],
                targets={"char": 1, "syllable": 2, "boundary": 1.0},
                learning_rate=0.1,
                loss_function=loss,
            )

        self.assertAlmostEqual(report["total_loss"], 0.46)
        forward_backend.assert_called_once()
        heads_backward.assert_called_once()
        self.assertEqual([call[0] for call in backend.resident_calls], [
            "sum_heads", "mean_pool_backward"
        ])
        representation_backward.assert_called_once()
        self.assertEqual(representation_backward.call_args.args[0].shape, (2, model.representation.output_dim))
        heads_step.assert_called_once_with(0.1)
        representation_step.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()
