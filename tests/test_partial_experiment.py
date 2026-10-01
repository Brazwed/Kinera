import unittest

from language.partial_syllable import CausalPartialSyllableEncoder
from language.vocabulary import CharacterVocabulary, SyllableVocabulary
from training.partial_experiment import PARTIAL_VARIANTS, run_partial_comparison


class PartialSyllableExperimentTests(unittest.TestCase):
    def test_smoke_compares_original_and_partial_variants(self) -> None:
        chars = CharacterVocabulary(["<START>", "<END>", "<UNK>", "c", "o", "m"])
        syllables = SyllableVocabulary(
            ["<UNK_SYL>", "com", *[f"synthetic-{index}" for index in range(100)]]
        )
        examples = [{
            "context_ids": [0, 0, 3, 4],
            "context_syllable_ids": [0, 0, 0, 0],
            "context_feature_vectors": [[0.0] * 10] * 4,
            "target_char_id": 5,
            "target_char": "m",
            "target_syllable_id": 1,
            "target_syllable_index": 0,
            "target_is_syllable_end": 1,
        }]

        report = run_partial_comparison(
            examples,
            examples,
            chars,
            syllables,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
        )

        self.assertEqual(report["variants"], list(PARTIAL_VARIANTS))
        self.assertIn("E", report["aggregates"])
        self.assertIn("F", report["aggregates"])
        counts = {
            variant: report["runs"][0]["metrics"][variant]["parameter_count"]
            for variant in PARTIAL_VARIANTS
        }
        self.assertEqual(len(set(counts.values())), 1)


if __name__ == "__main__":
    unittest.main()
