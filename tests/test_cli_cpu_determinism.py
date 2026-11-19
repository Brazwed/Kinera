"""CPU-only regression for seeded end-to-end execution of the public training CLI.

The temporary fixture is synthetic; no historical datasets or artifacts are written.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TrainCliCpuDeterminismTests(unittest.TestCase):
    def test_two_independent_processes_repeat_same_seed(self) -> None:
        vocabulary = {
            "version": 1,
            "tokens": ["<START>", "<END>", "<UNK>", "a", "b", "c"],
        }
        examples = [
            {"context_ids": [0, 0], "target_char_id": 3},
            {"context_ids": [0, 3], "target_char_id": 4},
            {"context_ids": [3, 4], "target_char_id": 5},
            {"context_ids": [4, 5], "target_char_id": 1},
        ]

        with tempfile.TemporaryDirectory(prefix="kinera-cli-cpu-") as directory:
            root = Path(directory)
            vocabulary_path = root / "vocabulary.json"
            training_path = root / "training.jsonl"

            vocabulary_path.write_text(
                json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8"
            )
            training_path.write_text(
                "".join(json.dumps(example, ensure_ascii=False) + "\n"
                        for example in examples),
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train.py"),
                "--vocabulary", str(vocabulary_path),
                "--train", str(training_path),
                "--epochs", "3",
                "--learning-rate", "0.1",
                "--max-examples", "4",
                "--context-size", "2",
                "--max-tokens", "5",
                "--seed", "7",
            ]
            outputs = []
            for _ in range(2):
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=90,
                )
                outputs.append(completed.stdout)

        self.assertEqual(outputs[0], outputs[1])
        report = json.loads(outputs[0])
        self.assertEqual(report["epochs"], 3)
        self.assertEqual(len(report["history"]), 3)
        self.assertIsInstance(report["generated_text"], str)


if __name__ == "__main__":
    unittest.main()
