import unittest

from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


class CharacterVocabularyTests(unittest.TestCase):
    def test_special_tokens_are_stable(self) -> None:
        vocabulary = CharacterVocabulary.from_records(
            [{"word_normalized": "casa"}]
        )

        self.assertEqual(vocabulary.token_to_id["<START>"], 0)
        self.assertEqual(vocabulary.token_to_id["<END>"], 1)
        self.assertEqual(vocabulary.token_to_id["<UNK>"], 2)

    def test_encode_decode_round_trip_with_boundaries(self) -> None:
        vocabulary = CharacterVocabulary.from_records(
            [{"word_normalized": "ação"}]
        )

        encoded = vocabulary.encode("ação", add_boundaries=True)
        decoded = vocabulary.decode(encoded, skip_special=True)

        self.assertEqual(decoded, "ação")
        self.assertEqual(encoded[0], vocabulary.start_id)
        self.assertEqual(encoded[-1], vocabulary.end_id)

    def test_unknown_character_uses_unk(self) -> None:
        vocabulary = CharacterVocabulary.from_records(
            [{"word_normalized": "casa"}]
        )

        encoded = vocabulary.encode("z")

        self.assertEqual(encoded, [vocabulary.unk_id])
        self.assertEqual(vocabulary.decode(encoded), "<UNK>")

    def test_vocabulary_serializes_and_restores(self) -> None:
        vocabulary = CharacterVocabulary.from_records(
            [{"word_normalized": "casa"}, {"word_normalized": "ação"}]
        )

        restored = CharacterVocabulary.from_dict(vocabulary.to_dict())

        self.assertEqual(restored.to_dict(), vocabulary.to_dict())
        self.assertEqual(restored.encode("ação"), vocabulary.encode("ação"))


class SyllableVocabularyTests(unittest.TestCase):
    def test_special_unknown_token_and_known_syllables(self) -> None:
        vocabulary = SyllableVocabulary.from_records(
            [
                {"syllables": ["ca", "sa"]},
                {"syllables": ["ação"]},
            ]
        )

        self.assertEqual(vocabulary.unk_id, 0)
        self.assertEqual(vocabulary.encode("ca"), vocabulary.token_to_id["ca"])
        self.assertEqual(vocabulary.encode("desconhecida"), vocabulary.unk_id)

    def test_syllable_vocabulary_round_trip(self) -> None:
        vocabulary = SyllableVocabulary.from_records(
            [{"syllables": ["ca", "sa"]}]
        )

        restored = SyllableVocabulary.from_dict(vocabulary.to_dict())

        self.assertEqual(restored.to_dict(), vocabulary.to_dict())


if __name__ == "__main__":
    unittest.main()
