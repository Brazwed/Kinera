import tempfile
import unittest
from pathlib import Path

from training.multi_seed_runner import run_seed_jobs


class MultiSeedRunnerTests(unittest.TestCase):
    def test_resume_skips_completed_seed_and_executes_new_seed_once(self) -> None:
        events = []

        def worker(seed):
            events.append(seed)
            return {"seed": seed, "status": "completed", "payload": {"value": seed}}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runner.json"
            first = run_seed_jobs(
                seeds=[7],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=worker,
            )
            events.clear()
            resumed = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=worker,
                resume=True,
            )

        self.assertEqual(first["completed_runs"], 1)
        self.assertEqual(events, [42])
        self.assertEqual(resumed["completed_runs"], 2)
        self.assertEqual([run["seed"] for run in resumed["runs"]], [7, 42])

    def test_worker_failure_is_persisted_without_losing_completed_jobs(self) -> None:
        def worker(seed):
            if seed == 42:
                raise RuntimeError("boom")
            return {"seed": seed, "status": "completed"}

        with tempfile.TemporaryDirectory() as directory:
            report = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=Path(directory) / "runner.json",
                worker=worker,
            )

        self.assertEqual(report["completed_runs"], 1)
        self.assertEqual(report["failed_runs"], 1)
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
