import copy
import random
import unittest
from typing import Any
from unittest.mock import Mock, patch

from kinera.experiments.learning_memory import (
    ExampleRef,
    LearningMemory,
    LearningMemoryRecorder,
)
from kinera.language.vocabulary import CharacterVocabulary
from kinera.training.autoregressive import (
    AutoregressiveModel,
    evaluate_multitask_examples,
    train_multitask_examples,
)


class MultitaskTrainingWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        self.examples = [
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
            {
                "context_ids": [3, 4],
                "context_syllable_ids": [1, 2],
                "context_feature_vectors": [[1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8],
                "target_char_id": 1,
                "target_syllable_id": 0,
                "target_is_syllable_end": 1,
            },
        ]

    def _model(self) -> AutoregressiveModel:
        return AutoregressiveModel(
            char_vocab_size=len(self.vocabulary),
            syllable_vocab_size=3,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            seed=7,
        )

    def test_training_reports_all_losses_and_accuracies(self) -> None:
        report = train_multitask_examples(
            self._model(),
            self.examples,
            self.vocabulary,
            epochs=2,
            learning_rate=0.1,
        )

        metrics = report["history"][-1]
        self.assertEqual(report["epochs"], 2)
        for name in (
            "char_loss",
            "syllable_loss",
            "boundary_loss",
            "total_loss",
            "char_accuracy",
            "syllable_accuracy",
            "boundary_accuracy",
        ):
            self.assertIn(name, metrics)

    def test_evaluation_reports_known_syllable_accuracy_and_boundary_metrics(self) -> None:
        report = evaluate_multitask_examples(
            self._model(), self.examples, self.vocabulary, syllable_unknown_id=0
        )

        self.assertIn("syllable_accuracy_all", report)
        self.assertIn("syllable_accuracy_known", report)
        self.assertIn("boundary_target_distribution", report)
        self.assertIn("boundary_precision", report)
        self.assertIn("boundary_recall", report)
        self.assertIn("boundary_f1", report)
        self.assertEqual(report["boundary_target_distribution"], {"0": 0, "1": 3})

    def test_evaluator_uses_scalar_model_step_without_exporting_head_vectors(self) -> None:
        model = self._model()
        with patch.object(
            model,
            "forward_heads",
            side_effect=AssertionError("evaluation must not export complete logits"),
        ):
            report = evaluate_multitask_examples(
                model, self.examples, self.vocabulary, syllable_unknown_id=0
            )

        self.assertEqual(report["examples"], len(self.examples))
        self.assertEqual(report["boundary_target_distribution"], {"0": 0, "1": 3})

    def test_evaluation_does_not_update_any_parameters(self) -> None:
        model = self._model()
        before = {
            "char": [row.copy() for row in model.heads.char.weights],
            "syllable": [row.copy() for row in model.heads.syllable.weights],
            "boundary": [row.copy() for row in model.heads.boundary.weights],
        }

        report = evaluate_multitask_examples(model, self.examples, self.vocabulary)

        self.assertEqual(report["examples"], 3)
        self.assertEqual(before["char"], model.heads.char.weights)
        self.assertEqual(before["syllable"], model.heads.syllable.weights)
        self.assertEqual(before["boundary"], model.heads.boundary.weights)

    def test_passive_recorder_is_equivalent_to_baseline_and_observes_each_update(self) -> None:
        def run(observer_enabled: bool) -> tuple[Any, ...]:
            model = self._model()
            counters = {"train": 0, "eval": 0, "forward_heads": 0}
            train_reports: list[dict[str, float]] = []
            update_order: list[list[int]] = []
            observer_calls: list[tuple[int, int, int, dict[str, float]]] = []
            train_step = model.multitask_train_step
            eval_step = model.multitask_eval_step
            forward_heads = model._forward_heads_backend

            def counted_train(*args: Any, **kwargs: Any) -> dict[str, float]:
                counters["train"] += 1
                update_order.append(list(args[0]))
                report = train_step(*args, **kwargs)
                train_reports.append(report.copy())
                return report

            def counted_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
                counters["eval"] += 1
                return eval_step(*args, **kwargs)

            def counted_forward(*args: Any, **kwargs: Any) -> Any:
                counters["forward_heads"] += 1
                return forward_heads(*args, **kwargs)

            memory = LearningMemory(capacity=3)
            refs = tuple(ExampleRef("fixture:multitask", "train", i) for i in range(3))
            recorder = LearningMemoryRecorder(refs, memory)

            def observe(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
                observer_calls.append((epoch, index, step, report.copy()))
                recorder(epoch, index, step, report)

            random_state_before = random.getstate()
            with (
                patch.object(model, "multitask_train_step", side_effect=counted_train),
                patch.object(model, "multitask_eval_step", side_effect=counted_eval),
                patch.object(model, "_forward_heads_backend", side_effect=counted_forward),
            ):
                kwargs = {"step_observer": observe} if observer_enabled else {}
                training_report = train_multitask_examples(
                    model,
                    self.examples,
                    self.vocabulary,
                    epochs=2,
                    learning_rate=0.1,
                    **kwargs,
                )
                evaluation = evaluate_multitask_examples(model, self.examples, self.vocabulary)
            self.assertEqual(random.getstate(), random_state_before)
            parameters = copy.deepcopy((
                model.representation.char_embedding.weights,
                model.representation.syllable_embedding.weights,
                model.heads.char.weights,
                model.heads.char.bias,
                model.heads.syllable.weights,
                model.heads.syllable.bias,
                model.heads.boundary.weights,
                model.heads.boundary.bias,
            ))
            return (
                training_report,
                parameters,
                observer_calls,
                counters,
                evaluation,
                memory.entries(),
                train_reports,
                update_order,
            )

        baseline = run(observer_enabled=False)
        observed = run(observer_enabled=True)

        self.assertEqual(observed[0], baseline[0])
        self.assertEqual(observed[1], baseline[1])
        self.assertEqual(observed[3], baseline[3])
        self.assertEqual(observed[4], baseline[4])
        self.assertEqual(observed[6], baseline[6])
        self.assertEqual(observed[7], baseline[7])
        self.assertEqual([call[3] for call in observed[2]], observed[6])
        self.assertEqual(observed[3], {"train": 6, "eval": 9, "forward_heads": 15})
        self.assertEqual(len(observed[2]), 6)
        self.assertEqual(
            [(epoch, index, step) for epoch, index, step, _ in observed[2]],
            [(epoch, index, (epoch - 1) * 3 + index + 1) for epoch in (1, 2) for index in range(3)],
        )
        self.assertEqual(len(observed[5]), 3)
        for index, entry in enumerate(observed[5]):
            self.assertEqual(entry.example_ref, ExampleRef("fixture:multitask", "train", index))
            self.assertEqual(entry.seen_count, 2)
            self.assertEqual(entry.first_seen_step, index + 1)
            self.assertEqual(entry.last_seen_step, index + 4)
            self.assertEqual(entry.latest_event.char_loss, observed[6][index + 3]["char_loss"])
            self.assertEqual(entry.latest_event.syllable_loss, observed[6][index + 3]["syllable_loss"])
            self.assertEqual(entry.latest_event.boundary_loss, observed[6][index + 3]["boundary_loss"])
            self.assertEqual(entry.latest_event.total_loss, observed[6][index + 3]["total_loss"])
            self.assertEqual(observed[2][index + 3][3], observed[6][index + 3])

    def test_step_observer_errors_propagate_after_one_successful_update(self) -> None:
        model = self._model()
        calls = 0

        def failing_observer(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            nonlocal calls
            calls += 1
            self.assertEqual((epoch, index, step), (1, 0, 1))
            self.assertEqual(set(report), {"char_loss", "syllable_loss", "boundary_loss", "total_loss"})
            raise RuntimeError("observer failed")

        with patch.object(model, "multitask_train_step", wraps=model.multitask_train_step) as train_step:
            with self.assertRaisesRegex(RuntimeError, "observer failed"):
                train_multitask_examples(
                    model,
                    self.examples,
                    self.vocabulary,
                    epochs=2,
                    learning_rate=0.1,
                    step_observer=failing_observer,
                )
        self.assertEqual(calls, 1)
        train_step.assert_called_once()

    def test_step_observer_is_not_called_when_training_step_fails(self) -> None:
        model = self._model()
        observer = Mock()
        with patch.object(
            model,
            "multitask_train_step",
            side_effect=RuntimeError("step failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                train_multitask_examples(
                    model,
                    self.examples,
                    self.vocabulary,
                    epochs=1,
                    learning_rate=0.1,
                    step_observer=observer,
                )
        observer.assert_not_called()

    def test_training_is_reproducible_with_same_seed(self) -> None:
        first = train_multitask_examples(
            self._model(), self.examples, self.vocabulary, epochs=2, learning_rate=0.1
        )
        second = train_multitask_examples(
            self._model(), self.examples, self.vocabulary, epochs=2, learning_rate=0.1
        )

        self.assertEqual(first, second)

    def test_model_can_overfit_small_multitask_dataset(self) -> None:
        model = self._model()
        initial = evaluate_multitask_examples(model, self.examples, self.vocabulary)

        report = train_multitask_examples(
            model,
            self.examples,
            self.vocabulary,
            epochs=120,
            learning_rate=0.2,
        )
        final = evaluate_multitask_examples(model, self.examples, self.vocabulary)

        self.assertLess(final["total_loss"], initial["total_loss"])
        self.assertGreaterEqual(final["char_accuracy"], report["history"][0]["char_accuracy"])

    def test_micro_overfit_updates_shared_embeddings(self) -> None:
        model = self._model()
        before_char = [row.copy() for row in model.representation.char_embedding.weights]
        before_syllable = [row.copy() for row in model.representation.syllable_embedding.weights]

        train_multitask_examples(
            model,
            self.examples[:1],
            self.vocabulary,
            epochs=10,
            learning_rate=0.2,
        )

        self.assertNotEqual(before_char, model.representation.char_embedding.weights)
        self.assertNotEqual(before_syllable, model.representation.syllable_embedding.weights)

    def test_multitask_eval_step_returns_scalars_and_releases_forward_state(self) -> None:
        model = self._model()
        example = self.examples[0]
        result = model.multitask_eval_step(
            example["context_ids"],
            example["context_syllable_ids"],
            example["context_feature_vectors"],
            targets={
                "char": example["target_char_id"],
                "syllable": example["target_syllable_id"],
                "boundary": float(example["target_is_syllable_end"]),
            },
        )

        self.assertEqual(set(result), {"losses", "predictions"})
        self.assertEqual(
            set(result["losses"]),
            {"char_loss", "syllable_loss", "boundary_loss", "total_loss"},
        )
        self.assertEqual(set(result["predictions"]), {"char", "syllable", "boundary"})
        self.assertTrue(
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in result["losses"].values()
            )
        )
        self.assertTrue(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in result["predictions"].values()
            )
        )
        self.assertIsNone(model.last_logits)
        self.assertIsNone(model.last_pooled)
        self.assertEqual(model._last_head_logits_backend, {})
        self.assertIsNone(model.last_context_size)
        self.assertIsNone(model.representation.last_output)
        self.assertIsNone(model.representation.char_embedding.output)
        self.assertIsNone(model.representation.syllable_embedding.output)
        for head in (model.heads.char, model.heads.syllable, model.heads.boundary):
            self.assertIsNone(head.last_input)
            self.assertIsNone(head.output)

        model.close()
        model.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.multitask_eval_step(
                example["context_ids"],
                example["context_syllable_ids"],
                example["context_feature_vectors"],
                targets={
                    "char": example["target_char_id"],
                    "syllable": example["target_syllable_id"],
                    "boundary": float(example["target_is_syllable_end"]),
                },
            )


if __name__ == "__main__":
    unittest.main()
