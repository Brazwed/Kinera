import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_portal_lingua import import_export


class PortalImportTests(unittest.TestCase):
    def test_jsonl_export_is_normalized_and_preserves_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "export.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "word": "computador",
                                "syllables": ["com", "pu", "ta", "dor"],
                                "stress": 3,
                                "ipa": "kũputɐˈdoɾ",
                                "category": "nome masculino",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "word": "ação",
                                "syl": "a|**ção**",
                                "syl_strong": "1",
                                "variant": "pt",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = Path(directory) / "processed.jsonl"

            report = import_export(input_path, output_path)
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(report["rows"], 2)
        self.assertEqual(records[0]["source"], "portal_lingua")
        self.assertEqual(records[0]["syllables"], ["com", "pu", "ta", "dor"])
        self.assertEqual(records[0]["stressed_syllable_index"], 3)
        self.assertEqual(records[0]["ipa"], "kũputɐˈdoɾ")
        self.assertEqual(records[1]["variant"], "pt")

    def test_csv_export_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "export.csv"
            input_path.write_text(
                "word,syllables,stress,ipa\n"
                "banana,ba|**na**|na,1,baˈnɐnɐ\n",
                encoding="utf-8",
            )
            output_path = Path(directory) / "processed.jsonl"

            import_export(input_path, output_path)
            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(record["word_normalized"], "banana")
        self.assertEqual(record["syllables"], ["ba", "na", "na"])
        self.assertEqual(record["stressed_syllable_index"], 1)

    def test_unknown_extension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "export.txt"
            input_path.write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                import_export(input_path, Path(directory) / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
