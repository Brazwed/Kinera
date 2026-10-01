import unittest

from language.vocabulary import CharacterVocabulary
from training.comparison import (
    VARIANTS,
    count_trainable_parameters,
    prepare_variant_examples,
    run_controlled_experiments,
)
from training.autoregressive import AutoregressiveModel


class ControlledComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        self.examples = [
            {
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 4],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_syllable_id": 4,
                "target_is_syllable_end": 1,
            }
        ]

    def test_all_variants_keep_context_shape_and_only_mask_channels(self) -> None:
        prepared = {
            name: prepare_variant_examples(
                self.examples,
                name,
                self.vocabulary,
                syllable_unknown_id=0,
            )[0]
            for name in VARIANTS
        }

        self.assertEqual(prepared["A"]["context_ids"], [0, 3])
        self.assertEqual(prepared["A"]["context_syllable_ids"], [0, 0])
        self.assertEqual(prepared["A"]["context_feature_vectors"], [[0.0] * 10, [0.0] * 10])
        self.assertEqual(prepared["B"]["context_syllable_ids"], [0, 0])
        self.assertEqual(prepared["C"]["context_syllable_ids"], [0, 4])
        self.assertEqual(prepared["C"]["context_feature_vectors"], [[0.0] * 10, [0.0] * 10])
        self.assertEqual(prepared["D"]["context_syllable_ids"], [0, 4])
        self.assertEqual(prepared["D"]["context_feature_vectors"][1][0], 1.0)

    def test_variants_have_equal_trainable_parameter_counts(self) -> None:
        counts = []
        for _ in VARIANTS:
            model = AutoregressiveModel(5, 5, seed=7)
            counts.append(count_trainable_parameters(model))

        self.assertEqual(len(set(counts)), 1)
        self.assertGreater(counts[0], 0)

    def test_runner_reports_each_variant_and_seed(self) -> None:
        report = run_controlled_experiments(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
        )

        self.assertEqual(set(report["runs"]), set(VARIANTS))
        self.assertEqual(len(report["runs"]["A"]), 1)
        self.assertIn("mean", report["aggregates"]["A"])
        self.assertIn("char_accuracy", report["aggregates"]["A"]["mean"])
        self.assertEqual(report["parameter_counts"]["A"], report["parameter_counts"]["D"])


if __name__ == "__main__":
    unittest.main()
