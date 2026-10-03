import unittest

from kinera.language.partial_syllable import (
    CausalPartialSyllableEncoder,
    prepare_partial_variant_examples,
)
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


class CausalPartialSyllableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = CharacterVocabulary(
            ["<START>", "<END>", "<UNK>", "c", "o", "m"]
        )
        self.syllables = SyllableVocabulary(
            ["<UNK_SYL>", "com", *[f"synthetic-{index}" for index in range(100)]]
        )
        self.encoder = CausalPartialSyllableEncoder(self.characters, self.syllables)

    def test_prefix_id_changes_only_when_observed_character_is_added(self) -> None:
        short = self.encoder.encode_context(
            [0, 0, 3, 4],
            [0, 0, 0, 0],
        )
        longer = self.encoder.encode_context(
            [0, 0, 3, 4, 5],
            [0, 0, 0, 0, 0],
        )

        self.assertEqual(short, longer[:4])
        self.assertNotEqual(short[2], short[3])
        self.assertNotEqual(short[3], longer[4])

    def test_completed_syllable_is_preserved_and_partial_prefix_resets(self) -> None:
        encoded = self.encoder.encode_context(
            [3, 4, 5, 3],
            [1, 1, 1, 0],
        )

        self.assertEqual(encoded[:3], [1, 1, 1])
        self.assertNotEqual(encoded[3], 1)
        self.assertNotEqual(encoded[3], 0)

    def test_ids_stay_inside_the_fixed_embedding_table(self) -> None:
        encoded = self.encoder.encode_context([3, 4, 5], [0, 0, 0])

        self.assertEqual(len(encoded), 3)
        self.assertTrue(all(0 <= value < len(self.syllables) for value in encoded))

    def test_variants_preserve_shapes_and_only_f_masks_features(self) -> None:
        examples = [{
            "context_ids": [0, 3, 4],
            "context_syllable_ids": [0, 0, 0],
            "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9, [0.0] * 10],
        }]
        partial = prepare_partial_variant_examples(examples, "E", self.characters, self.syllables)[0]
        hybrid = prepare_partial_variant_examples(examples, "F", self.characters, self.syllables)[0]

        self.assertEqual(len(partial["context_syllable_ids"]), 3)
        self.assertEqual(partial["context_feature_vectors"], [[0.0] * 10] * 3)
        self.assertEqual(hybrid["context_feature_vectors"], examples[0]["context_feature_vectors"])


if __name__ == "__main__":
    unittest.main()
