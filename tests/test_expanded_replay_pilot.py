from __future__ import annotations

import json
import math
import unittest

from kinera.experiments.expanded_pilot import (
    AggregatedMetric,
    ConditionAggregateResult,
    InterferenceLevel,
    PilotConditionSpec,
    PilotCostProfile,
    PilotSeedResult,
    compute_cost_profile,
    get_condition_spec,
    run_condition_seed,
    run_expanded_pilot,
    _build_unified_vocabularies,
)
from kinera.experiments.learning_memory import ExampleRef


class AggregatedMetricTests(unittest.TestCase):
    def test_aggregation_valid_sequence(self) -> None:
        agg = AggregatedMetric.from_sequence([1.0, 2.0, 3.0])
        self.assertAlmostEqual(agg.mean, 2.0)
        self.assertAlmostEqual(agg.min, 1.0)
        self.assertAlmostEqual(agg.max, 3.0)
        # population std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) approx 0.81649658
        self.assertAlmostEqual(agg.std, math.sqrt(2.0 / 3.0))

    def test_aggregation_single_value(self) -> None:
        agg = AggregatedMetric.from_sequence([4.5])
        self.assertEqual(agg.mean, 4.5)
        self.assertEqual(agg.std, 0.0)
        self.assertEqual(agg.min, 4.5)
        self.assertEqual(agg.max, 4.5)

    def test_aggregation_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            AggregatedMetric.from_sequence([])


class PilotConditionSpecTests(unittest.TestCase):
    def test_all_conditions_exist_and_valid(self) -> None:
        for level in ("low", "medium", "high"):
            spec = get_condition_spec(level)  # type: ignore[arg-type]
            self.assertEqual(spec.level, level)
            self.assertGreater(spec.old_epochs, 0)
            self.assertGreater(spec.new_epochs, 0)
            self.assertGreater(spec.replay_epochs, 0)
            self.assertGreater(spec.replay_budget, 0)
            self.assertGreater(spec.old_learning_rate, 0.0)
            self.assertGreater(spec.new_learning_rate, 0.0)
            self.assertGreater(spec.replay_learning_rate, 0.0)

    def test_unknown_condition_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_condition_spec("unknown")  # type: ignore[arg-type]


class ExpandedPilotExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.char_vocab, cls.syl_vocab = _build_unified_vocabularies()

    def test_budget_fairness_and_accounting(self) -> None:
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        comp = result.comparison
        self.assertTrue(comp.is_fairly_matched_budget)
        self.assertEqual(result.eval_b.accounting.replay_updates, result.eval_c.accounting.replay_updates)
        self.assertEqual(result.eval_b.accounting.new_updates, result.eval_c.accounting.new_updates)
        self.assertEqual(result.eval_b.accounting.total_updates, result.eval_c.accounting.total_updates)
        self.assertEqual(result.eval_a.accounting.replay_updates, 0)

    def test_strict_leakage_prevention_audit(self) -> None:
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        
        # Verify candidate pool contains no holdout probe references
        for cand in result.eval_c.candidate_pool:
            self.assertNotIn("holdout", cand.example_ref.split)
            self.assertNotIn("holdout", cand.example_ref.dataset_artifact_id)
            for gid in cand.group_ids:
                self.assertNotIn("hold", gid)

        # Verify selected candidates are also clean
        for sel in result.eval_c.selected_candidates:
            self.assertNotIn("holdout", sel.example_ref.split)
            self.assertNotIn("holdout", sel.example_ref.dataset_artifact_id)

        # Verify holdout metrics exist and are in [0.0, 1.0]
        for variant in ("A", "B", "C"):
            self.assertIn(variant, result.holdout_retention_ratio)
            h_ret = result.holdout_retention_ratio[variant]
            self.assertTrue(0.0 <= h_ret <= 1.0)

    def test_deterministic_reproducibility(self) -> None:
        res1 = run_condition_seed("low", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        res2 = run_condition_seed("low", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        self.assertEqual(res1.comparison.c_vs_b_recovery_advantage, res2.comparison.c_vs_b_recovery_advantage)
        self.assertEqual(res1.eval_c.mean_recovery_delta, res2.eval_c.mean_recovery_delta)
        self.assertEqual(res1.eval_b.mean_recovery_delta, res2.eval_b.mean_recovery_delta)
        self.assertEqual(res1.eval_a.mean_recovery_delta, res2.eval_a.mean_recovery_delta)

    def test_cost_profile_and_resource_tracking(self) -> None:
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        self.assertGreater(result.cpu_time_seconds, 0.0)
        self.assertGreater(result.peak_ram_bytes, 0)
        self.assertTrue(math.isfinite(result.cpu_time_seconds))

        profile = compute_cost_profile(
            total_cpu_seconds=result.cpu_time_seconds,
            peak_ram_bytes=result.peak_ram_bytes,
            pilot_seeds_count=1,
            pilot_words_count=6,
        )
        self.assertIn(profile.scaling_recommendation, ("local_cpu", "google_colab_pro"))
        self.assertTrue(len(profile.recommendation_rationale) > 10)
        self.assertTrue(math.isfinite(profile.projected_scaled_hours_cpu))

    def test_full_expanded_pilot_execution_and_schema(self) -> None:
        # Run small smoke pilot with 2 seeds and 2 conditions to test complete report generation
        report = run_expanded_pilot(
            seeds=(42, 43),
            conditions=("low", "high"),
            baseline_commit="5dba2cc",
            runner_commit="test_commit",
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["phase"], "42-expanded-pilot")
        self.assertTrue(report["protocol_validated"])
        self.assertTrue(report["leakage_prevented"])
        self.assertEqual(report["conditions_evaluated"], ["low", "high"])
        self.assertEqual(report["seeds_evaluated"], [42, 43])
        self.assertIn("condition_aggregates", report)
        self.assertIn("cost_profile", report)
        self.assertIn("conclusions", report)
        self.assertTrue(report["conclusions"]["fairness_confirmed_all_seeds"])

        # Check JSON serializability
        dumped = json.dumps(report)
        self.assertTrue(len(dumped) > 100)


if __name__ == "__main__":
    unittest.main()
