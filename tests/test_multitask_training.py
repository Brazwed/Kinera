import unittest
from unittest.mock import patch

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
