import unittest

from kinera.language.vocabulary import CharacterVocabulary
from kinera.diagnostics.syllables import run_syllable_gain_analysis


class SyllableGainAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        self.examples = [
            {
                "word_key": "ab",
                "context_ids": [0, 3],
                "context_syllable_ids": [0, 4],
                "context_feature_vectors": [[0.0] * 10, [1.0] + [0.0] * 9],
                "target_char_id": 4,
                "target_char": "b",
                "target_syllable_id": 4,
                "target_syllable_index": 0,
                "target_is_syllable_end": 1,
            },
            {
                "word_key": "ab",
                "context_ids": [3, 4],
                "context_syllable_ids": [4, 0],
                "context_feature_vectors": [[1.0] + [0.0] * 9, [0.0] * 10],
                "target_char_id": 1,
                "target_char": "<END>",
                "target_syllable_id": 0,
                "target_syllable_index": None,
                "target_is_syllable_end": 1,
            },
        ]

    def test_report_contains_causal_coverage_buckets_and_pair_deltas(self) -> None:
        report = run_syllable_gain_analysis(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            syllable_unknown_id=0,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=2,
            top_examples=2,
        )

        self.assertEqual(report["completed_runs"], 1)
        self.assertIn("active", report["runs"][0]["coverage"])
        self.assertIn("unavailable", report["runs"][0]["coverage"])
        self.assertIn("A_vs_C", report["runs"][0]["pair_deltas"])
        self.assertIn("B_vs_D", report["runs"][0]["pair_deltas"])
        self.assertIn("word_position", report["runs"][0]["buckets"])
        self.assertIn("gain", report["runs"][0]["examples"]["A_vs_C"])
        self.assertIn("regression", report["runs"][0]["examples"]["B_vs_D"])

    def test_pair_buckets_use_augmented_causal_metadata(self) -> None:
        report = run_syllable_gain_analysis(
            self.examples,
            self.examples,
            self.vocabulary,
            syllable_vocab_size=5,
            syllable_unknown_id=0,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=2,
        )
        run = report["runs"][0]
        self.assertEqual(run["coverage"]["active"], 2)
        active = run["pair_deltas"]["A_vs_C"]["by_syllable_info"]["active"]
        self.assertEqual(active["examples"], 2)
        proximity = run["buckets"]["boundary_proximity"]["A_vs_C"]
        self.assertIn("after_completed_or_completed_context", proximity)
        self.assertIn("inside_next_syllable", proximity)
        example = run["examples"]["A_vs_C"]["gain"][0]
        self.assertAlmostEqual(
            example["delta_loss"],
            example["base_loss"] - example["augmented_loss"],
        )


if __name__ == "__main__":
    unittest.main()
