import json
import tempfile
import unittest
from pathlib import Path

from scripts.annotate_syllable_dataset import annotate_jsonl


class AnnotationPipelineTests(unittest.TestCase):
    def test_annotate_jsonl_writes_character_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "word_raw": "casa",
                        "word_normalized": "casa",
                        "syllables": ["ca", "sa"],
                        "source": "test",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "annotated.jsonl"
            report_path = root / "report.json"

            report = annotate_jsonl(input_path, output_path, report_path)
            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["records"], 1)
        self.assertEqual(len(record["characters"]), 4)
        self.assertEqual(record["characters"][1]["is_vowel"], True)
        self.assertTrue(record["alignment"]["valid"])


if __name__ == "__main__":
    unittest.main()
