import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.compare as run_comparison
from kinera.experiments.comparison import CONTROLLED_VARIANTS, DEFAULT_VARIANTS


class RunComparisonCliTests(unittest.TestCase):
    def _base_args(self) -> list[str]:
        return [
            "--vocabulary", "chars.json",
            "--syllable-vocabulary", "syllables.json",
            "--train", "train.jsonl",
            "--validation", "validation.jsonl",
        ]

    def test_cli_defaults_to_primary_variants(self) -> None:
        args = run_comparison.build_parser().parse_args(self._base_args())

        self.assertEqual(args.variants, list(DEFAULT_VARIANTS))

    def test_cli_accepts_explicit_historical_control_set(self) -> None:
        args = run_comparison.build_parser().parse_args(
            self._base_args() + ["--variants", *CONTROLLED_VARIANTS]
        )

        self.assertEqual(args.variants, list(CONTROLLED_VARIANTS))

    def test_cli_rejects_unknown_variant_choice(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_comparison.build_parser().parse_args(
                self._base_args() + ["--variants", "Z"]
            )

    def test_main_forwards_selected_variants_to_runner(self) -> None:
        class FakeSyllableVocabulary:
            unk_id = 0

            def __len__(self) -> int:
                return 2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            char_path = root / "chars.json"
            syllable_path = root / "syllables.json"
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            for path in (char_path, syllable_path):
                path.write_text("{}\n", encoding="utf-8")
            for path in (train_path, validation_path):
                path.write_text("{}\n", encoding="utf-8")

            with (
                patch.object(
                    run_comparison.CharacterVocabulary,
                    "from_dict",
                    return_value=object(),
                ),
                patch.object(
                    run_comparison.SyllableVocabulary,
                    "from_dict",
                    return_value=FakeSyllableVocabulary(),
                ),
                patch.object(run_comparison, "load_jsonl", return_value=[{}]),
                patch.object(
                    run_comparison,
                    "run_controlled_experiments",
                    return_value={"ok": True},
                ) as runner,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_comparison.main([
                    "--vocabulary", str(char_path),
                    "--syllable-vocabulary", str(syllable_path),
                    "--train", str(train_path),
                    "--validation", str(validation_path),
                    "--variants", "D",
                ])

        self.assertEqual(runner.call_args.kwargs["variants"], ("D",))


if __name__ == "__main__":
    unittest.main()
