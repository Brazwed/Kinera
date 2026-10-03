import unittest

from kinera.experiments.config import (
    ExecutionConfig,
    ExperimentConfig,
    RunRecord,
    aggregate_metric_paths,
    aggregate_run_status,
    mean_std,
)


class ExperimentTypesTests(unittest.TestCase):
    def test_config_serializes_common_experiment_controls(self) -> None:
        config = ExperimentConfig(
            phase="test",
            variants=("A", "C"),
            seeds=(7, 42),
            epochs=2,
            learning_rate=0.05,
            max_examples=1000,
            context_size=8,
        )
        execution = ExecutionConfig(workers=2)

        self.assertEqual(config.to_dict()["variants"], ["A", "C"])
        self.assertEqual(config.to_dict()["seeds"], [7, 42])
        self.assertNotIn("workers", config.to_dict())
        self.assertEqual(execution.workers, 2)

    def test_run_record_round_trip_and_status_aggregation(self) -> None:
        records = [
            RunRecord(seed=7, status="completed", payload={"loss": 1.0}),
            RunRecord(seed=42, status="failed", error="boom"),
            RunRecord(seed=123, status="running"),
        ]

        restored = [RunRecord.from_dict(record.to_dict()) for record in records]
        summary = aggregate_run_status(restored)

        self.assertEqual(restored[0].payload["loss"], 1.0)
        self.assertEqual(summary, {
            "total_runs": 3,
            "completed_runs": 1,
            "failed_runs": 1,
            "pending_runs": 0,
            "running_runs": 1,
            "status": "failed",
        })

    def test_mean_std_uses_population_dispersion(self) -> None:
        self.assertEqual(mean_std([1.0, 2.0, 3.0]), {"mean": 2.0, "std": 0.816496580927726})

    def test_nested_metric_aggregation_ignores_failed_runs(self) -> None:
        records = [
            {"status": "completed", "metrics": {"loss": 1.0}},
            {"status": "completed", "metrics": {"loss": 3.0}},
            {"status": "failed", "metrics": {"loss": 100.0}},
        ]

        report = aggregate_metric_paths(records, {"loss": ("metrics", "loss")})

        self.assertEqual(report["completed_runs"], 2)
        self.assertEqual(report["metrics"]["loss"], {"mean": 2.0, "std": 1.0})


if __name__ == "__main__":
    unittest.main()
