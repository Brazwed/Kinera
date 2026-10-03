import unittest

from kinera.language.features import character_features
from kinera.language.vocabulary import CharacterVocabulary
from kinera.diagnostics.orthography import classify_target, run_orthographic_class_analysis


class OrthographicAnalysisTests(unittest.TestCase):
    def test_classification_keeps_special_and_orthographic_buckets_separate(self) -> None:
        self.assertEqual(classify_target("<END>"), "<END>")
        self.assertEqual(classify_target("a"), "vogal")
        self.assertEqual(classify_target("á"), "vogal_acentuada")
        self.assertEqual(classify_target("b"), "consoante")
        self.assertEqual(classify_target("-"), "separador_pontuacao")

    def test_report_contains_pair_metrics_per_class(self) -> None:
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        examples = [{
            "context_ids": [0, 0],
            "context_syllable_ids": [0, 0],
            "context_feature_vectors": [[0.0] * 10, [0.0] * 10],
            "target_char_id": 3,
            "target_char": "a",
            "target_syllable_id": 1,
            "target_syllable_index": 0,
            "target_is_syllable_end": 1,
        }]
        report = run_orthographic_class_analysis(
            examples,
            examples,
            vocabulary,
            syllable_vocab_size=3,
            syllable_unknown_id=0,
            seeds=[7],
            epochs=1,
            learning_rate=0.05,
            max_examples=1,
        )
        self.assertIn("A_vs_C", report["runs"][0]["comparisons"])
        self.assertIn("vogal", report["runs"][0]["comparisons"]["A_vs_C"])
        self.assertIn("count", report["runs"][0]["comparisons"]["A_vs_C"]["vogal"])


if __name__ == "__main__":
    unittest.main()
