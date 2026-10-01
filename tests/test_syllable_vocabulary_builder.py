import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_syllable_vocabulary import build_syllable_vocabulary


class SyllableVocabularyBuilderTests(unittest.TestCase):
    def test_builder_writes_deterministic_vocabulary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.jsonl"
            vocabulary_path = root / "vocabulary.json"
            report_path = root / "report.json"
            records = [
                {"syllables": ["sa", "ca"]},
                {"syllables": ["ação"]},
            ]
            source.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            report = build_syllable_vocabulary(source, vocabulary_path, report_path)

            vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            self.assertEqual(vocabulary["tokens"], ["<UNK_SYL>", "ação", "ca", "sa"])
            self.assertEqual(report["records_read"], 2)
            self.assertEqual(report["syllable_count"], 3)


if __name__ == "__main__":
    unittest.main()
