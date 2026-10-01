import unittest

from language.features import (
    ORTHOGRAPHIC_FEATURE_NAMES,
    annotate_record,
    annotate_word,
    character_features,
    features_to_vector,
)


class CharacterFeatureTests(unittest.TestCase):
    def test_accented_vowel_features(self) -> None:
        features = character_features("á")

        self.assertTrue(features["is_vowel"])
        self.assertFalse(features["is_consonant"])
        self.assertTrue(features["is_accented"])
        self.assertTrue(features["is_alpha"])
        self.assertFalse(features["is_digit"])

    def test_consonant_features(self) -> None:
        features = character_features("r")

        self.assertFalse(features["is_vowel"])
        self.assertTrue(features["is_consonant"])
        self.assertTrue(features["is_alpha"])

    def test_digit_space_and_punctuation_features(self) -> None:
        digit = character_features("3")
        space = character_features(" ")
        punctuation = character_features("!")

        self.assertTrue(digit["is_digit"])
        self.assertTrue(space["is_space"])
        self.assertTrue(punctuation["is_punctuation"])

    def test_case_and_unicode_category_are_preserved(self) -> None:
        features = character_features("A")

        self.assertTrue(features["is_upper"])
        self.assertFalse(features["is_lower"])
        self.assertEqual(features["unicode_category"], "Lu")

    def test_annotate_word_combines_features_and_alignment(self) -> None:
        annotated = annotate_word("casa", ["ca", "sa"])

        self.assertEqual([item["char"] for item in annotated], list("casa"))
        self.assertEqual([item["syllable_index"] for item in annotated], [0, 0, 1, 1])
        self.assertTrue(annotated[0]["is_vowel"] is False)
        self.assertTrue(annotated[1]["is_vowel"])

    def test_annotate_record_preserves_source_fields(self) -> None:
        record = {
            "word_raw": "casa",
            "word_normalized": "casa",
            "syllables": ["ca", "sa"],
            "source": "silabas-pt",
        }

        annotated = annotate_record(record)

        self.assertEqual(annotated["source"], "silabas-pt")
        self.assertEqual(len(annotated["characters"]), 4)
        self.assertIn("alignment", annotated)

    def test_features_to_vector_has_stable_order_and_binary_values(self) -> None:
        features = character_features("á")
        vector = features_to_vector(features)

        self.assertEqual(len(vector), len(ORTHOGRAPHIC_FEATURE_NAMES))
        self.assertTrue(all(value in (0.0, 1.0) for value in vector))
        self.assertEqual(vector[ORTHOGRAPHIC_FEATURE_NAMES.index("is_vowel")], 1.0)


if __name__ == "__main__":
    unittest.main()
