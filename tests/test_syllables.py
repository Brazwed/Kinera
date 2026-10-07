import unittest

from kinera.language.syllables import (
    evaluate_records,
    segment_word,
    syllable_boundaries,
)


class SyllableSegmenterTests(unittest.TestCase):
    def test_common_portuguese_words(self) -> None:
        expected = {
            "banana": ["ba", "na", "na"],
            "computador": ["com", "pu", "ta", "dor"],
            "programação": ["pro", "gra", "ma", "ção"],
            "casa": ["ca", "sa"],
            "a": ["a"],
        }

        for word, syllables in expected.items():
            with self.subTest(word=word):
                self.assertEqual(segment_word(word), syllables)

    def test_m0_keeps_zero_support_fallbacks_and_segmenter_counts(self) -> None:
        self.assertEqual(evaluate_records([]), {
            "total": 0,
            "scored": 0,
            "exact_matches": 0,
            "exact_accuracy": 0.0,
            "boundary_correct": 0,
            "boundary_total": 0,
            "boundary_accuracy": 1.0,
            "errors": {},
        })
        report = evaluate_records([{
            "word_normalized": "banana",
            "syllables": ["ba", "nana"],
        }])
        self.assertEqual(report["exact_accuracy"], 0.0)
        self.assertEqual(report["boundary_correct"], 1)
        self.assertEqual(report["boundary_total"], 1)
        self.assertEqual(report["boundary_accuracy"], 1.0)

    def test_digraph_and_diphthong_cases(self) -> None:
        self.assertEqual(segment_word("chuva"), ["chu", "va"])
        self.assertEqual(segment_word("mãe"), ["mãe"])
        self.assertEqual(segment_word("quente"), ["quen", "te"])

    def test_boundaries_are_character_offsets(self) -> None:
        self.assertEqual(syllable_boundaries("computador", ["com", "pu", "ta", "dor"]), [3, 5, 7])

    def test_evaluation_reports_exact_matches_and_boundaries(self) -> None:
        records = [
            {
                "word_normalized": "banana",
                "syllables": ["ba", "na", "na"],
            },
            {
                "word_normalized": "casa",
                "syllables": ["ca", "sa"],
            },
        ]

        report = evaluate_records(records)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["exact_matches"], 2)
        self.assertEqual(report["exact_accuracy"], 1.0)
        self.assertEqual(report["boundary_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
