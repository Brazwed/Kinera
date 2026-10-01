import unittest

from language.hybrid_inputs import record_to_hybrid_inputs
from language.vocabulary import CharacterVocabulary, SyllableVocabulary


class HybridInputTests(unittest.TestCase):
    def test_record_maps_characters_syllables_and_features_per_position(self) -> None:
        record = {
            "word_normalized": "a-b",
            "syllables": ["a", "b"],
            "characters": [
                {
                    "char": "a",
                    "syllable_index": 0,
                    "is_vowel": True,
                    "is_alpha": True,
                },
                {
                    "char": "-",
                    "syllable_index": None,
                    "is_separator": True,
                },
                {
                    "char": "b",
                    "syllable_index": 1,
                    "is_consonant": True,
                    "is_alpha": True,
                },
            ],
        }
        char_vocabulary = CharacterVocabulary.from_records([record])
        syllable_vocabulary = SyllableVocabulary.from_records([record])

        char_ids, syllable_ids, feature_vectors = record_to_hybrid_inputs(
            record, char_vocabulary, syllable_vocabulary
        )

        self.assertEqual(len(char_ids), 3)
        self.assertEqual(syllable_ids[0], syllable_vocabulary.encode("a"))
        self.assertEqual(syllable_ids[1], syllable_vocabulary.unk_id)
        self.assertEqual(syllable_ids[2], syllable_vocabulary.encode("b"))
        self.assertEqual(len(feature_vectors), 3)
        self.assertEqual(len(feature_vectors[0]), 10)


if __name__ == "__main__":
    unittest.main()
