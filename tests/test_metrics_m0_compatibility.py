import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from kinera.monitor.metrics import mean_std, safe_ratio
from kinera.training.metrics import _safe_ratio
from kinera.experiments.config import mean_std as historical_mean_std


class MetricsM0CompatibilityTests(unittest.TestCase):
    def test_population_statistics_are_bitwise_equal_to_legacy_formula(self) -> None:
        fixtures = ([1.0], [1.0, 2.0, 3.0], [1e16, 1.0, -1e16, 3.0])
        for values in fixtures:
            with self.subTest(values=values):
                legacy_mean = sum(values) / len(values)
                legacy_variance = sum(
                    (value - legacy_mean) ** 2 for value in values
                ) / len(values)
                expected = {
                    "mean": legacy_mean,
                    "std": math.sqrt(legacy_variance),
                }
                self.assertEqual(mean_std(values), expected)
                self.assertEqual(historical_mean_std(values), expected)

    def test_empty_statistics_keep_historical_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "values não pode ser vazio"):
            mean_std([])

    def test_ratio_and_historical_alias_preserve_zero_policy(self) -> None:
        self.assertEqual(safe_ratio(1, 2), 0.5)
        self.assertEqual(safe_ratio(1, 0), 0.0)
        self.assertEqual(safe_ratio(1, 0, zero_denominator=1.0), 1.0)
        self.assertIs(_safe_ratio, safe_ratio)

    def test_importing_monitor_parent_has_no_cuda_or_telemetry_side_effects(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ,
            "PYTHONPATH": str(project_root / "src"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import kinera.monitor; "
                "assert not any(name.startswith('kinera.backends.cuda') "
                "for name in sys.modules); "
                "assert 'kinera.monitor.telemetry' not in sys.modules",
            ],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
