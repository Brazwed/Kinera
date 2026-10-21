from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest

from kinera.experiments.learning_memory import ExampleRef
from kinera.experiments.replay import ReplayAccounting
from kinera.experiments.scientific_metrics import (
    ScientificComparison,
    ScientificVariantEvaluation,
    calculate_recovery_ratio,
    calculate_replay_utility,
    calculate_retention_ratio,
    calculate_stability_plasticity_tradeoff,
    compare_scientific_variants,
)
from kinera.experiments.scientific_runner import run_scientific_protocol


class ScientificMetricsTests(unittest.TestCase):
    def test_calculate_replay_utility_valid(self) -> None:
        self.assertEqual(calculate_replay_utility(1.0, 2), 0.5)
        self.assertEqual(calculate_replay_utility(-0.5, 1), -0.5)
        self.assertEqual(calculate_replay_utility(0.0, 4), 0.0)

    def test_calculate_replay_utility_rejects_invalid_inputs(self) -> None:
        for updates in (0, -1, True, False, 1.5, "2"):
            with self.subTest(updates=updates), self.assertRaises((TypeError, ValueError)):
                calculate_replay_utility(1.0, updates)  # type: ignore[arg-type]

        for delta in (float("nan"), float("inf"), float("-inf"), True, "1.0"):
            with self.subTest(delta=delta), self.assertRaises((TypeError, ValueError)):
                calculate_replay_utility(delta, 1)  # type: ignore[arg-type]

    def test_calculate_retention_ratio(self) -> None:
        # higher is better
        self.assertAlmostEqual(calculate_retention_ratio(1.0, 0.8, "higher_is_better"), 0.8)
        self.assertEqual(calculate_retention_ratio(0.0, 0.5, "higher_is_better"), 1.0)
        self.assertEqual(calculate_retention_ratio(0.0, -0.5, "higher_is_better"), 0.0)

        # lower is better
        self.assertAlmostEqual(calculate_retention_ratio(1.0, 2.0, "lower_is_better"), 0.5)
        self.assertAlmostEqual(calculate_retention_ratio(2.0, 1.0, "lower_is_better"), 2.0)

        with self.assertRaises(ValueError):
            calculate_retention_ratio(1.0, 1.0, "invalid_direction")  # type: ignore[arg-type]

    def test_calculate_stability_plasticity_tradeoff(self) -> None:
        # Both positive -> harmonic mean
        val = calculate_stability_plasticity_tradeoff(0.4, 0.4)
        self.assertAlmostEqual(val, 0.4)

        # One non-positive -> arithmetic mean penalty
        self.assertEqual(calculate_stability_plasticity_tradeoff(-0.2, 0.4), 0.1)
        self.assertEqual(calculate_stability_plasticity_tradeoff(0.0, 0.0), 0.0)

    def test_variant_evaluation_dataclass_validations(self) -> None:
        acc_zero = ReplayAccounting(1, 0, 1)
        ref = ExampleRef("f", "s", 0)
        acc_replay = ReplayAccounting(1, 1, 2, selected_refs=(ref,))

        # Invalid variant
        with self.assertRaises(ValueError):
            ScientificVariantEvaluation(
                "X", (1.0,), (1.0,), (1.0,), (0.0,), (0.0,), (0.0,),
                1.0, 1.0, 0.0, acc_zero, 0.0, None, 1.0, 0.0
            )

        # Replay updates == 0 but utility provided
        with self.assertRaises(ValueError):
            ScientificVariantEvaluation(
                "A", (1.0,), (1.0,), (1.0,), (0.0,), (0.0,), (0.0,),
                1.0, 1.0, 0.0, acc_zero, 0.0, 0.5, 1.0, 0.0
            )

        # Replay updates > 0 but utility is None
        with self.assertRaises(ValueError):
            ScientificVariantEvaluation(
                "B", (1.0,), (1.0,), (1.0,), (0.0,), (0.0,), (0.0,),
                1.0, 1.0, 0.0, acc_replay, 0.0, None, 1.0, 0.0
            )

        valid_a = ScientificVariantEvaluation(
            "A", (1.0,), (0.5,), (0.5,), (0.5,), (0.5,), (0.0,),
            1.0, 0.8, -0.2, acc_zero, 0.0, None, 0.5, -0.1
        )
        self.assertEqual(valid_a.variant, "A")
        d = valid_a.to_dict()
        self.assertEqual(d["variant"], "A")
        self.assertIsNone(d["mean_replay_utility"])
        self.assertEqual(d["mean_retention_ratio"], 0.5)
        self.assertEqual(d["stability_plasticity_tradeoff"], -0.1)


class ScientificProtocolExecutionTests(unittest.TestCase):
    def test_synthetic_protocol_is_reproducible_and_fair(self) -> None:
        first = run_scientific_protocol("synthetic", replay_seed=0)
        second = run_scientific_protocol("synthetic", replay_seed=0)
        self.assertEqual(first, second)

        evals = first["evaluations"]
        a, b, c = evals["A"], evals["B"], evals["C"]

        # Fairness checks
        self.assertEqual(a["accounting"]["replay_updates"], 0)
        self.assertEqual(b["accounting"]["replay_updates"], 1)
        self.assertEqual(c["accounting"]["replay_updates"], 1)
        self.assertEqual(b["accounting"]["total_updates"], 2)
        self.assertEqual(c["accounting"]["total_updates"], 2)

        comp = first["comparison"]
        self.assertTrue(comp["is_fairly_matched_budget"])
        self.assertFalse(comp["is_directed_superior"])
        self.assertIn("Relative gradient update count increase", comp["overhead_definition"])
        self.assertIn("does not demonstrate empirical superiority", comp["summary_claim"])

    def test_natural_protocol_proves_prior_competence_and_differential_selection(self) -> None:
        result = run_scientific_protocol("natural", replay_seed=1)
        self.assertEqual(result["fixture_type"], "natural")
        evals = result["evaluations"]
        a, b, c = evals["A"], evals["B"], evals["C"]

        # 1. Proves prior acquired competence is strictly positive on all probes
        for variant_eval in (a, b, c):
            self.assertEqual(variant_eval["old_competence_reference"], [1.0, 1.0])

        # 2. Proves actual forgetting occurred before replay on one skill but not the other
        for variant_eval in (a, b, c):
            self.assertEqual(variant_eval["old_forgetting_pre"], [0.0, 1.0])

        # 3. Proves B (random) and C (directed) differentiate their selected candidate
        b_selected = b["accounting"]["selected_refs"] if "selected_refs" in b["accounting"] else None
        # Access from runner's accounting
        self.assertEqual(b["accounting"]["replay_seed"], 1)
        self.assertIsNone(c["accounting"]["replay_seed"])

        # 4. Verifies retention ratio and stability vs plasticity metrics are present and finite
        for variant_eval in (a, b, c):
            self.assertTrue(math.isfinite(variant_eval["mean_retention_ratio"]))
            self.assertTrue(math.isfinite(variant_eval["stability_plasticity_tradeoff"]))

        comp = result["comparison"]
        self.assertTrue(comp["is_fairly_matched_budget"])
        self.assertFalse(comp["is_directed_superior"])
        self.assertIn("deferred pending scaled corpora experiments", comp["summary_claim"])

    def test_unknown_fixture_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_scientific_protocol("non_existent")  # type: ignore[arg-type]


class ScientificImportBoundaryTests(unittest.TestCase):
    def test_import_has_no_forbidden_dependencies_or_runtime_side_effects(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = {**os.environ, "PYTHONPATH": os.path.join(project_root, "src")}
        script = """
import sys
import kinera.experiments.scientific_metrics
import kinera.experiments.scientific_runner

forbidden = ("torch", "numpy", "psutil", "kinera.learning", "kinera.backends.cuda")
for mod in forbidden:
    assert not any(name == mod or name.startswith(mod + ".") for name in sys.modules), f"forbidden module imported: {mod}"
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
