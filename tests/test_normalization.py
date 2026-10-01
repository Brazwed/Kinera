import unittest

from language.records import (
    normalize_silabas_row,
    normalize_word,
    parse_syllable_markup,
)


class NormalizationTests(unittest.TestCase):
    def test_word_normalization_preserves_accents_and_creates_key(self) -> None:
        normalized, key = normalize_word("  Àção  ")

        self.assertEqual(normalized, "Àção")
        self.assertEqual(key, "àção")

    def test_syllable_markup_removes_source_annotations(self) -> None:
        syllables, stressed_index = parse_syllable_markup(
            "__a__-pro|**pó**|si|to"
        )

        self.assertEqual(syllables, ["a", "pro", "pó", "si", "to"])
        self.assertEqual(stressed_index, 2)

    def test_single_syllable_markup_is_supported(self) -> None:
        self.assertEqual(parse_syllable_markup("**a**"), (["a"], 0))
        self.assertEqual(parse_syllable_markup("a"), (["a"], None))

    def test_row_is_converted_to_normalized_record(self) -> None:
        record = normalize_silabas_row(
            {
                "word": "a-tempo",
                "category": "nome masculino",
                "syl": "__a__-**tem**|po",
                "syl_positions": "[1, 5]",
                "syl_strong": "1",
            },
            source="silabas-pt",
        )

        self.assertEqual(record["word_raw"], "a-tempo")
        self.assertEqual(record["word_normalized"], "a-tempo")
        self.assertEqual(record["word_key"], "a-tempo")
        self.assertEqual(record["syllables"], ["a", "tem", "po"])
        self.assertEqual(record["stressed_syllable_index"], 1)
        self.assertEqual(record["category"], "nome masculino")
        self.assertEqual(record["source"], "silabas-pt")
        self.assertEqual(record["alignment_status"], "valid")
        self.assertEqual(record["quality_flags"], [])

    def test_malformed_alignment_is_flagged_without_dropping_record(self) -> None:
        record = normalize_silabas_row(
            {
                "word": "banana",
                "category": "nome",
                "syl": "ba|na",
                "syl_positions": "[2]",
                "syl_strong": "0",
            },
            source="silabas-pt",
        )

        self.assertEqual(record["syllables"], ["ba", "na"])
        self.assertEqual(record["alignment_status"], "mismatch")
        self.assertIn("syllable_reconstruction_mismatch", record["quality_flags"])


if __name__ == "__main__":
    unittest.main()
