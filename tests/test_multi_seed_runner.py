import json
import tempfile
import unittest
from pathlib import Path

from kinera.experiments.multi_seed import prepare_seed_records, run_seed_jobs


class MultiSeedRunnerTests(unittest.TestCase):
    def test_prepare_records_reuses_interrupted_and_failed_seeds_without_duplicates(self) -> None:
        records = [
            {"seed": 7, "status": "completed", "payload": "kept"},
            {"seed": 42, "status": "running", "payload": "stale"},
            {"seed": 123, "status": "failed", "payload": "retry"},
        ]

        prepared = prepare_seed_records(records, [7, 42, 123, 999])

        self.assertEqual([record["seed"] for record in prepared], [7, 42, 123, 999])
        self.assertEqual(prepared[0]["status"], "completed")
        self.assertEqual([record["status"] for record in prepared[1:]], ["pending", "pending", "pending"])
        self.assertEqual(len({record["seed"] for record in prepared}), 4)

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

    def test_resume_retries_interrupted_running_seed_without_duplicate_record(self) -> None:
        events = []

        def worker(seed):
            events.append(seed)
            return {"seed": seed, "status": "completed"}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runner.json"
            checkpoint.write_text(
                json.dumps({
                    "phase": "test",
                    "epochs": 1,
                    "seeds": [7, 42],
                    "workers": 1,
                    "runs": [
                        {"seed": 7, "status": "completed"},
                        {"seed": 42, "status": "running"},
                    ],
                }),
                encoding="utf-8",
            )
            report = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=worker,
                resume=True,
            )

        self.assertEqual(events, [42])
        self.assertEqual([run["seed"] for run in report["runs"]], [7, 42])
        self.assertEqual(report["completed_runs"], 2)

    def test_resume_retries_failed_seed_without_duplicate_record(self) -> None:
        events = []

        def worker(seed):
            events.append(seed)
            return {"seed": seed, "status": "completed"}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runner.json"
            checkpoint.write_text(
                json.dumps({
                    "phase": "test",
                    "epochs": 1,
                    "seeds": [7, 42],
                    "workers": 1,
                    "runs": [
                        {"seed": 7, "status": "completed"},
                        {"seed": 42, "status": "failed", "traceback": "old"},
                    ],
                }),
                encoding="utf-8",
            )
            report = run_seed_jobs(
                seeds=[7, 42],
                experiment_config={"phase": "test", "epochs": 1},
                execution_config={"workers": 1},
                checkpoint_path=checkpoint,
                worker=worker,
                resume=True,
            )

        self.assertEqual(events, [42])
        self.assertEqual([run["seed"] for run in report["runs"]], [7, 42])
        self.assertEqual(report["completed_runs"], 2)


if __name__ == "__main__":
    unittest.main()
