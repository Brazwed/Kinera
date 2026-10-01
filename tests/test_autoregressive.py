import unittest

from language.dataset import build_autoregressive_examples
from language.vocabulary import CharacterVocabulary


class AutoregressiveDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "word_key": "casa",
                "word_normalized": "casa",
                "syllables": ["ca", "sa"],
                "characters": [
                    {
                        "char": "c",
                        "syllable_index": 0,
                        "is_syllable_end": False,
                    },
                    {
                        "char": "a",
                        "syllable_index": 0,
                        "is_syllable_end": True,
                    },
                    {
                        "char": "s",
                        "syllable_index": 1,
                        "is_syllable_end": False,
                    },
                    {
                        "char": "a",
                        "syllable_index": 1,
                        "is_syllable_end": True,
                    },
                ],
            }
        ]
        self.vocabulary = CharacterVocabulary.from_records(self.records)

    def test_examples_use_fixed_context_and_end_token(self) -> None:
        examples = build_autoregressive_examples(
            self.records,
            self.vocabulary,
            context_size=2,
        )

        self.assertEqual(len(examples), 5)
        self.assertTrue(all(len(item["context_ids"]) == 2 for item in examples))
        self.assertEqual(examples[0]["target_char"], "c")
        self.assertEqual(examples[-1]["target_char"], "<END>")
        self.assertEqual(examples[-1]["target_is_syllable_end"], True)

    def test_examples_preserve_syllable_targets_and_word_key(self) -> None:
        examples = build_autoregressive_examples(
            self.records,
            self.vocabulary,
            context_size=2,
        )

        target_a = next(item for item in examples if item["target_char"] == "a")
        self.assertEqual(target_a["word_key"], "casa")
        self.assertIn(target_a["target_syllable_index"], (0, 1))

    def test_invalid_context_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_autoregressive_examples(self.records, self.vocabulary, context_size=0)


if __name__ == "__main__":
    unittest.main()
