import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_character_dataset import build_character_dataset


class CharacterDatasetBuilderTests(unittest.TestCase):
    def test_builder_writes_vocab_and_autoregressive_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotated = root / "annotated.jsonl"
            groups_train = root / "train_groups.jsonl"
            groups_validation = root / "validation_groups.jsonl"
            annotated.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in [
                        {
                            "word_key": "casa",
                            "word_normalized": "casa",
                            "characters": [
                                {"char": "c", "syllable_index": 0, "is_syllable_end": False},
                                {"char": "a", "syllable_index": 0, "is_syllable_end": True},
                                {"char": "s", "syllable_index": 1, "is_syllable_end": False},
                                {"char": "a", "syllable_index": 1, "is_syllable_end": True},
                            ],
                        },
                        {
                            "word_key": "pato",
                            "word_normalized": "pato",
                            "characters": [
                                {"char": "p", "syllable_index": 0, "is_syllable_end": False},
                                {"char": "a", "syllable_index": 0, "is_syllable_end": True},
                                {"char": "t", "syllable_index": 1, "is_syllable_end": False},
                                {"char": "o", "syllable_index": 1, "is_syllable_end": True},
                            ],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            groups_train.write_text(json.dumps({"word_key": "casa"}) + "\n", encoding="utf-8")
            groups_validation.write_text(json.dumps({"word_key": "pato"}) + "\n", encoding="utf-8")
            vocab = root / "vocabulary.json"
            train_output = root / "train.jsonl"
            validation_output = root / "validation.jsonl"
            report_path = root / "report.json"

            report = build_character_dataset(
                annotated,
                groups_train,
                groups_validation,
                vocab,
                train_output,
                validation_output,
                report_path,
                context_size=2,
            )

            train_examples = train_output.read_text(encoding="utf-8").splitlines()
            validation_examples = validation_output.read_text(encoding="utf-8").splitlines()
            vocabulary = json.loads(vocab.read_text(encoding="utf-8"))

        self.assertEqual(report["train_records"], 1)
        self.assertEqual(report["validation_records"], 1)
        self.assertGreater(len(train_examples), 0)
        self.assertGreater(len(validation_examples), 0)
        self.assertEqual(vocabulary["version"], 1)


if __name__ == "__main__":
    unittest.main()
