import unittest

from kinera.language.alignment import (
    align_characters,
    alignment_summary,
    reconstruct_from_alignment,
)


class CharacterSyllableAlignmentTests(unittest.TestCase):
    def test_computador_maps_each_character_to_a_syllable(self) -> None:
        aligned = align_characters(
            "computador",
            ["com", "pu", "ta", "dor"],
        )

        self.assertEqual(len(aligned), 10)
        self.assertEqual(
            [item["syllable_index"] for item in aligned],
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
        )
        self.assertTrue(aligned[0]["is_syllable_start"])
        self.assertTrue(aligned[2]["is_syllable_end"])
        self.assertTrue(aligned[7]["is_syllable_start"])
        self.assertTrue(aligned[9]["is_syllable_end"])

    def test_hyphen_is_preserved_as_unaligned_separator(self) -> None:
        aligned = align_characters("a-tempo", ["a", "tem", "po"])

        hyphen = aligned[1]
        self.assertEqual(hyphen["char"], "-")
        self.assertIsNone(hyphen["syllable_index"])
        self.assertTrue(hyphen["is_separator"])
        self.assertEqual(reconstruct_from_alignment(aligned), "atempo")

    def test_alignment_summary_reports_mismatch_without_crashing(self) -> None:
        summary = alignment_summary("banana", ["ba", "na"])

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["word_without_separators"], "banana")
        self.assertEqual(summary["syllable_reconstruction"], "bana")
        self.assertIn("reconstruction_mismatch", summary["flags"])

    def test_empty_word_and_empty_syllables_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            align_characters("", [])
        with self.assertRaises(ValueError):
            align_characters("casa", [])


if __name__ == "__main__":
    unittest.main()
