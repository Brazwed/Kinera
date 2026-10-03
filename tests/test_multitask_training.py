import unittest

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


if __name__ == "__main__":
    unittest.main()
