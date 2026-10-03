import unittest
import json
import tempfile
from pathlib import Path

from kinera.language.dataset import group_records, split_by_word
from kinera.scripts.build_syllable_dataset import build_dataset


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "word_raw": "banana",
                "word_key": "banana",
                "syllables": ["ba", "na", "na"],
                "source": "silabas-pt",
                "category": "nome",
                "quality_flags": [],
            },
            {
                "word_raw": "banana",
                "word_key": "banana",
                "syllables": ["ba", "na", "na"],
                "source": "portal_lingua",
                "category": None,
                "quality_flags": [],
            },
            {
                "word_raw": "ação",
                "word_key": "ação",
                "syllables": ["a", "ção"],
                "source": "silabas-pt",
                "category": "nome",
                "quality_flags": [],
            },
            {
                "word_raw": "ação",
                "word_key": "ação",
                "syllables": ["a", "ção"],
                "source": "portal_lingua",
                "category": None,
                "quality_flags": [],
            },
            {
                "word_raw": "casa",
                "word_key": "casa",
                "syllables": ["ca", "sa"],
                "source": "silabas-pt",
                "category": "nome",
                "quality_flags": [],
            },
        ]

    def test_group_records_preserves_sources_and_no_conflict(self) -> None:
        groups = group_records(self.records)

        banana = next(group for group in groups if group["word_key"] == "banana")
        self.assertEqual(len(banana["analyses"]), 2)
        self.assertFalse(banana["conflict"])

    def test_conflicting_syllabification_is_marked(self) -> None:
        records = self.records + [
            {
                "word_raw": "banana",
                "word_key": "banana",
                "syllables": ["ban", "ana"],
                "source": "other",
                "category": None,
                "quality_flags": [],
            }
        ]

        groups = group_records(records)
        banana = next(group for group in groups if group["word_key"] == "banana")

        self.assertTrue(banana["conflict"])
        self.assertEqual(banana["distinct_syllabifications"], 2)

    def test_split_keeps_same_word_in_one_partition(self) -> None:
        groups = group_records(self.records)

        train, validation = split_by_word(groups, validation_ratio=0.5, seed=42)
        train_keys = {group["word_key"] for group in train}
        validation_keys = {group["word_key"] for group in validation}

        self.assertTrue(train_keys.isdisjoint(validation_keys))
        self.assertEqual(train_keys | validation_keys, {"banana", "ação", "casa"})

    def test_split_is_reproducible(self) -> None:
        groups = group_records(self.records)

        first = split_by_word(groups, validation_ratio=0.5, seed=42)
        second = split_by_word(groups, validation_ratio=0.5, seed=42)

        self.assertEqual(first, second)

    def test_build_dataset_writes_grouped_splits_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "records.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(record) for record in self.records) + "\n",
                encoding="utf-8",
            )
            grouped_path = root / "grouped.jsonl"
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            report_path = root / "report.json"

            report = build_dataset(
                [input_path],
                grouped_path,
                train_path,
                validation_path,
                report_path,
                validation_ratio=0.5,
                seed=42,
            )

            grouped = grouped_path.read_text(encoding="utf-8").splitlines()
            train = train_path.read_text(encoding="utf-8").splitlines()
            validation = validation_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(report["raw_records"], 5)
        self.assertEqual(report["groups"], 3)
        self.assertEqual(len(grouped), 3)
        self.assertEqual(len(train) + len(validation), 3)


if __name__ == "__main__":
    unittest.main()
