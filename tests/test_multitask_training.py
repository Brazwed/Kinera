import unittest

from language.vocabulary import CharacterVocabulary
from training.autoregressive import (
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


if __name__ == "__main__":
    unittest.main()
