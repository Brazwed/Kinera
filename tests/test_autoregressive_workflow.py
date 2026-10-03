import unittest

from kinera.language.vocabulary import CharacterVocabulary
from kinera.training.autoregressive import (
    example_to_inputs,
    evaluate_examples,
    generate_text,
    train_examples,
)
from kinera.training.autoregressive import AutoregressiveModel


class AutoregressiveWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        self.examples = [
            {
                "context_ids": [0, 0],
                "target_char_id": 3,
            },
            {
                "context_ids": [0, 3],
                "target_char_id": 4,
            },
            {
                "context_ids": [3, 4],
                "target_char_id": 1,
            },
        ]

    def test_example_adapter_builds_hybrid_inputs(self) -> None:
        char_ids, syllable_ids, features = example_to_inputs(
            self.examples[1], self.vocabulary, syllable_unknown_id=0, feature_dim=10
        )

        self.assertEqual(char_ids, [0, 3])
        self.assertEqual(syllable_ids, [0, 0])
        self.assertEqual(len(features), 2)
        self.assertEqual(len(features[0]), 10)
        self.assertEqual(features[1][0], 1.0)

    def test_example_adapter_prefers_persisted_syllable_context(self) -> None:
        example = {
            "context_ids": [0, 3],
            "context_syllable_ids": [0, 4],
            "context_feature_vectors": [[0.0] * 10, [0.0, 1.0] + [0.0] * 8],
        }

        _, syllable_ids, features = example_to_inputs(
            example, self.vocabulary, syllable_unknown_id=0
        )

        self.assertEqual(syllable_ids, [0, 4])
        self.assertEqual(features[1][1], 1.0)

    def test_training_reports_loss_and_accuracy(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=len(self.vocabulary),
            syllable_vocab_size=1,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            seed=7,
        )

        report = train_examples(
            model,
            self.examples,
            self.vocabulary,
            epochs=3,
            learning_rate=0.1,
        )

        self.assertEqual(report["epochs"], 3)
        self.assertEqual(len(report["history"]), 3)
        self.assertIn("loss", report["history"][-1])
        self.assertIn("accuracy", report["history"][-1])

    def test_evaluation_does_not_update_parameters(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=len(self.vocabulary),
            syllable_vocab_size=1,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            seed=7,
        )
        before = [row.copy() for row in model.output_layer.weights]

        report = evaluate_examples(model, self.examples, self.vocabulary)

        self.assertEqual(report["examples"], 3)
        self.assertIn("loss", report)
        self.assertEqual(before, model.output_layer.weights)

    def test_generation_stops_at_end_token(self) -> None:
        model = AutoregressiveModel(
            char_vocab_size=len(self.vocabulary),
            syllable_vocab_size=1,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            seed=7,
        )
        model.output_layer.weights = [[0.0] * model.representation.output_dim for _ in range(len(self.vocabulary))]
        model.output_layer.bias = [0.0] * len(self.vocabulary)
        model.output_layer.bias[self.vocabulary.end_id] = 1.0

        generated = generate_text(model, self.vocabulary, context_size=2, max_tokens=5)

        self.assertEqual(generated, "")


if __name__ == "__main__":
    unittest.main()
