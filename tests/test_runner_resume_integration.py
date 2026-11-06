import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunnerResumeIntegrationTests(unittest.TestCase):
    def _run_resume(self, script: str, report: dict, extra_args: list[str], resume_seeds: list[int]) -> dict:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text(json.dumps(report), encoding="utf-8")
            command = [
                sys.executable,
                str(root / script),
                "--vocabulary", "missing-vocabulary.json",
                "--syllable-vocabulary", "missing-syllables.json",
                "--train", "missing-train.jsonl",
                "--validation", "missing-validation.jsonl",
                "--seeds", *[str(seed) for seed in resume_seeds],
                "--epochs", "1",
                "--learning-rate", "0.05",
                "--max-examples", "10",
                "--workers", "1",
                "--checkpoint", str(checkpoint),
                "--resume",
                *extra_args,
            ]
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(result["workers"], 1)
            self.assertEqual(result["completed_runs"], len(resume_seeds))
            return result

    def test_phase54_resume_allows_operational_worker_change(self) -> None:
        metrics = {variant: {"validation": {"char_loss": 1.0, "char_accuracy": 0.0, "total_loss": 1.0}} for variant in "ABCDEF"}
        report = {
            "phase": "5.4_expanded_partial_syllable", "variants": list("ABCDEF"),
            "seeds": [7], "epochs": 1, "learning_rate": 0.05, "max_examples": 10,
            "context_size": 8, "workers": 3,
            "runs": [{"seed": seed, "status": "completed", "run": {"metrics": metrics}} for seed in [7, 42]],
        }
        self._run_resume("scripts/legacy/phase5_4_expanded.py", report, [], [7, 42])


if __name__ == "__main__":
    unittest.main()
