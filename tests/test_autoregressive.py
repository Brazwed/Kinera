import unittest

from language.dataset import build_autoregressive_examples
from language.vocabulary import CharacterVocabulary, SyllableVocabulary


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

    def test_examples_include_aligned_syllable_context_when_requested(self) -> None:
        syllable_vocabulary = SyllableVocabulary(["<UNK_SYL>", "ca", "sa"])

        examples = build_autoregressive_examples(
            self.records,
            self.vocabulary,
            syllable_vocabulary=syllable_vocabulary,
            context_size=2,
        )

        first = examples[1]
        self.assertEqual(first["context_ids"], [0, 4])
        self.assertEqual(first["context_syllable_ids"], [0, 0])
        self.assertEqual(len(first["context_feature_vectors"]), 2)
        self.assertEqual(len(first["context_feature_vectors"][0]), 10)

    def test_context_syllable_ids_never_reveal_an_incomplete_syllable(self) -> None:
        syllable_vocabulary = SyllableVocabulary(["<UNK_SYL>", "ca", "sa"])

        examples = build_autoregressive_examples(
            self.records,
            self.vocabulary,
            syllable_vocabulary=syllable_vocabulary,
            context_size=2,
        )

        # Predicting the first character of "casa": no syllable is complete.
        self.assertEqual(examples[0]["context_syllable_ids"], [0, 0])
        # Predicting "a" after only observing "c": "ca" is still incomplete.
        self.assertEqual(examples[1]["context_syllable_ids"], [0, 0])
        # Predicting "s": "ca" is now fully observed and may be exposed.
        self.assertEqual(examples[2]["context_syllable_ids"], [1, 1])
        # Predicting the final "a": "sa" is incomplete, so only "ca" is known.
        self.assertEqual(examples[3]["context_syllable_ids"], [1, 0])
        # At <END>, the complete final syllable is observable.
        self.assertEqual(examples[4]["context_syllable_ids"], [2, 2])

    def test_examples_include_syllable_targets_and_unknown_end_target(self) -> None:
        syllable_vocabulary = SyllableVocabulary(["<UNK_SYL>", "ca", "sa"])

        examples = build_autoregressive_examples(
            self.records,
            self.vocabulary,
            syllable_vocabulary=syllable_vocabulary,
            context_size=2,
        )

        first_character = examples[0]
        end_example = examples[-1]
        self.assertEqual(first_character["target_syllable_id"], 1)
        self.assertEqual(end_example["target_char"], "<END>")
        self.assertEqual(end_example["target_syllable_id"], 0)

    def test_invalid_syllable_index_uses_unknown_target(self) -> None:
        records = [
            {
                "word_key": "x",
                "characters": [
                    {"char": "x", "syllable_index": 99, "is_syllable_end": True}
                ],
                "syllables": ["x"],
            }
        ]
        vocabulary = CharacterVocabulary.from_records(records)
        syllable_vocabulary = SyllableVocabulary(["<UNK_SYL>", "x"])

        examples = build_autoregressive_examples(
            records,
            vocabulary,
            syllable_vocabulary=syllable_vocabulary,
            context_size=1,
        )

        self.assertEqual(examples[0]["target_syllable_id"], 0)

    def test_syllable_targets_handle_accent_and_separator(self) -> None:
        records = [
            {
                "word_key": "á-bê",
                "word_normalized": "á-bê",
                "characters": [
                    {"char": "á", "syllable_index": 0, "is_syllable_end": True},
                    {"char": "-", "syllable_index": None, "is_syllable_end": False},
                    {"char": "b", "syllable_index": 1, "is_syllable_end": False},
                    {"char": "ê", "syllable_index": 1, "is_syllable_end": True},
                ],
                "syllables": ["á", "bê"],
            }
        ]
        vocabulary = CharacterVocabulary.from_records(records)
        syllable_vocabulary = SyllableVocabulary(["<UNK_SYL>", "bê", "á"])

        examples = build_autoregressive_examples(
            records,
            vocabulary,
            syllable_vocabulary=syllable_vocabulary,
            context_size=1,
        )

        accent_target = next(item for item in examples if item["target_char"] == "á")
        separator_target = next(item for item in examples if item["target_char"] == "-")
        self.assertEqual(accent_target["target_syllable_id"], 2)
        self.assertEqual(separator_target["target_syllable_id"], 0)


if __name__ == "__main__":
    unittest.main()
