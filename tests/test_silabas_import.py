import csv
import json
import tempfile
import unittest
from pathlib import Path

from kinera.scripts.inspect_silabas_pt import inspect_csv
from kinera.scripts.import_silabas_pt import import_csv


class SilabasImportTests(unittest.TestCase):
    def _write_fixture(self, directory: str) -> Path:
        path = Path(directory) / "syls.csv"
        rows = [
            {
                "word": "a-tempo",
                "category": "nome masculino",
                "syl": "__a__-**tem**|po",
                "syl_positions": "[1, 5]",
                "syl_strong": "1",
            },
            {
                "word": "banana",
                "category": "nome",
                "syl": "ba|**na**|na",
                "syl_positions": "[2, 4]",
                "syl_strong": "1",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_inspector_reports_schema_and_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = self._write_fixture(directory)

            report = inspect_csv(input_path)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(
            report["fields"],
            ["word", "category", "syl", "syl_positions", "syl_strong"],
        )
        self.assertEqual(report["empty_fields"]["word"], 0)

    def test_importer_writes_jsonl_and_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = self._write_fixture(directory)
            output_path = Path(directory) / "processed.jsonl"
            report_path = Path(directory) / "report.json"

            report = import_csv(input_path, output_path, report_path)
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["records_written"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["source"], "silabas-pt")
        self.assertEqual(records[0]["syllables"], ["a", "tem", "po"])
        self.assertEqual(records[1]["stressed_syllable_index"], 1)
        self.assertEqual(records[0]["alignment_status"], "valid")
        self.assertEqual(report["alignment_status"], {"valid": 2})

    def test_importer_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "bad.csv"
            input_path.write_text("word,category\nabc,nome\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                import_csv(input_path, Path(directory) / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
