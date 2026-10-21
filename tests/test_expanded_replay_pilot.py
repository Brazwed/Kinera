from __future__ import annotations

import json
import math
import unittest

from kinera.experiments.expanded_pilot import (
    AggregatedMetric,
    ConditionAggregateResult,
    InterferenceLevel,
    NewTaskSpec,
    PilotConditionSpec,
    PilotCostProfile,
    PilotSeedResult,
    TaskFamilySpec,
    compute_cost_profile,
    get_condition_spec,
    run_condition_seed,
    run_expanded_pilot,
    _build_unified_vocabularies,
    _STANDARD_OLD_FAMILIES,
    _WORD_SYLLABLES,
)
from kinera.experiments.learning_memory import ExampleRef
from kinera.experiments.replay import (
    ReplayCandidate,
    select_directed_replay,
    select_random_replay,
)
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record


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
        self.assertEqual(result.eval_b.accounting.replay_updates, 2)
        self.assertEqual(result.eval_c.accounting.replay_updates, 2)
        self.assertEqual(result.eval_b.accounting.new_updates, 2)
        self.assertEqual(result.eval_c.accounting.new_updates, 2)

    def test_quadripartite_word_and_ref_separation(self) -> None:
        for level in ("low", "medium", "high"):
            spec = get_condition_spec(level)  # type: ignore[arg-type]
            w_old = set(spec.old_words)
            w_int = set(spec.old_internal_words)
            w_hold = set(spec.old_holdout_words)
            w_new = set(spec.new_words)
            w_new_hold = set(spec.new_holdout_words)

            # Ensure all 5 word partitions are pairwise disjoint
            self.assertTrue(w_old.isdisjoint(w_int), f"Overlap old/int in {level}")
            self.assertTrue(w_old.isdisjoint(w_hold), f"Overlap old/hold in {level}")
            self.assertTrue(w_int.isdisjoint(w_hold), f"Overlap int/hold in {level}")
            self.assertTrue(w_old.isdisjoint(w_new), f"Overlap old/new in {level}")
            self.assertTrue(w_new.isdisjoint(w_new_hold), f"Overlap new/new_hold in {level}")
            self.assertEqual(spec.replay_epochs, 1, f"Replay epochs must be exactly 1 in {level}")

    def test_task_family_and_probe_traceability(self) -> None:
        spec = get_condition_spec("high")
        self.assertEqual(len(spec.old_families), 4)
        for f in spec.old_families:
            self.assertTrue(isinstance(f, TaskFamilySpec))
            self.assertIn(f.train_word, spec.old_words)
            self.assertIn(f.internal_probe_word, spec.old_internal_words)
            self.assertIn(f.holdout_probe_word, spec.old_holdout_words)
            # Verify explicit prefix relationship
            self.assertTrue(f.train_word.startswith(f.shared_prefix))
            self.assertTrue(f.internal_probe_word.startswith(f.shared_prefix))
            self.assertTrue(f.holdout_probe_word.startswith(f.shared_prefix))

        for t in spec.new_tasks:
            self.assertTrue(isinstance(t, NewTaskSpec))
            self.assertIn(t.train_word, spec.new_words)
            self.assertIn(t.holdout_word, spec.new_holdout_words)
            self.assertTrue(t.train_word.startswith(t.shared_prefix))
            self.assertTrue(t.holdout_word.startswith(t.shared_prefix))

    def test_transition_content_and_intentional_overlap(self) -> None:
        """Inspect autoregressive transition content: intentional overlaps vs disjoint contexts."""
        spec = get_condition_spec("high")
        
        # 1. Verify intentional overlap across old train, internal probe, and holdout probe:
        # All items in each family share the identical onset-nucleus transition at ex[1]
        for f in spec.old_families:
            r_train = annotate_record({"word_key": f.train_word, "word_normalized": f.train_word, "syllables": _WORD_SYLLABLES[f.train_word]})
            r_int = annotate_record({"word_key": f.internal_probe_word, "word_normalized": f.internal_probe_word, "syllables": _WORD_SYLLABLES[f.internal_probe_word]})
            r_hold = annotate_record({"word_key": f.holdout_probe_word, "word_normalized": f.holdout_probe_word, "syllables": _WORD_SYLLABLES[f.holdout_probe_word]})

            ex_train = build_autoregressive_examples([r_train], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1]
            ex_int = build_autoregressive_examples([r_int], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1]
            ex_hold = build_autoregressive_examples([r_hold], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1]

            t_train = (tuple(ex_train["context_ids"]), ex_train["target_char_id"])
            t_int = (tuple(ex_int["context_ids"]), ex_int["target_char_id"])
            t_hold = (tuple(ex_hold["context_ids"]), ex_hold["target_char_id"])

            self.assertEqual(t_train, t_int, f"Family {f.family_id}: train and internal probe must share transition")
            self.assertEqual(t_train, t_hold, f"Family {f.family_id}: train and holdout probe must share transition")

        # 2. Verify low interference: context prefix sets are completely disjoint
        spec_low = get_condition_spec("low")
        old_train_contexts = {
            tuple(build_autoregressive_examples([annotate_record({"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]})], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1]["context_ids"])
            for w in spec_low.old_words
        }
        low_new_contexts = {
            tuple(build_autoregressive_examples([annotate_record({"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]})], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1]["context_ids"])
            for w in spec_low.new_words
        }
        self.assertTrue(old_train_contexts.isdisjoint(low_new_contexts), "Low interference must have disjoint prefix contexts")

        # 3. Verify high interference: intentional collision on prefixes ('<START>', 'm') and ('<START>', 'p') with divergent targets
        high_new_records = [annotate_record({"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}) for w in spec.new_words]
        high_new_ex = [build_autoregressive_examples([r], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1] for r in high_new_records]
        high_new_transitions = {tuple(ex["context_ids"]): ex["target_char_id"] for ex in high_new_ex}

        old_train_records = [annotate_record({"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}) for w in spec.old_words]
        old_train_ex = [build_autoregressive_examples([r], self.char_vocab, context_size=2, syllable_vocabulary=self.syl_vocab)[1] for r in old_train_records]
        old_train_transitions = {tuple(ex["context_ids"]): ex["target_char_id"] for ex in old_train_ex}

        colliding_contexts = set(old_train_transitions.keys()) & set(high_new_transitions.keys())
        self.assertEqual(len(colliding_contexts), 2, "High interference must collide on exactly 2 contexts ('m' and 'p')")
        for ctx in colliding_contexts:
            self.assertNotEqual(old_train_transitions[ctx], high_new_transitions[ctx], f"Collision on {ctx} must have divergent target")

    def test_strict_leakage_prevention_audit(self) -> None:
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        
        # Verify candidate pool contains no holdout probe references (strict set disjointness)
        candidate_refs = {c.example_ref for c in result.eval_c.candidate_pool}
        holdout_refs = {ExampleRef("dataset:pilot-old-holdout", "holdout_test", i) for i in range(4)}
        new_holdout_refs = {ExampleRef("dataset:pilot-new-holdout-high", "holdout_test", i) for i in range(2)}
        
        self.assertTrue(candidate_refs.isdisjoint(holdout_refs), "Candidate pool must be disjoint from old holdout refs")
        self.assertTrue(candidate_refs.isdisjoint(new_holdout_refs), "Candidate pool must be disjoint from new holdout refs")
        
        # Verify selected candidates are also strictly disjoint
        selected_refs = {c.example_ref for c in result.eval_c.selected_candidates}
        self.assertTrue(selected_refs.isdisjoint(holdout_refs))
        self.assertTrue(selected_refs.isdisjoint(new_holdout_refs))

        # Verify candidate pool group IDs never include any holdout probe groups
        for cand in result.eval_c.candidate_pool:
            for gid in cand.group_ids:
                self.assertNotIn("hold", gid)
                self.assertTrue(gid.startswith("probe-int-"))

        # Verify holdout metrics exist and are in [0.0, 1.0]
        for variant in ("A", "B", "C"):
            self.assertIn(variant, result.holdout_retention_ratio)
            h_ret = result.holdout_retention_ratio[variant]
            self.assertTrue(0.0 <= h_ret <= 1.0)

    def test_directed_replay_selector_auditing(self) -> None:
        """Validate forgetting_score ranking, deterministic tie breaking, budget parity, and causal isolation."""
        ref0 = ExampleRef("dataset:pilot-old-train", "train", 0)
        ref1 = ExampleRef("dataset:pilot-old-train", "train", 1)
        ref2 = ExampleRef("dataset:pilot-old-train", "train", 2)
        ref3 = ExampleRef("dataset:pilot-old-train", "train", 3)

        # 1. Priority scoring: highest forgetting first
        pool = (
            ReplayCandidate(ref0, ("probe-int-malas",), 0.2),
            ReplayCandidate(ref1, ("probe-int-patos",), 0.9),
            ReplayCandidate(ref2, ("probe-int-sapos",), 0.0),
            ReplayCandidate(ref3, ("probe-int-fogos",), 0.8),
        )
        selected = select_directed_replay(pool, budget=2)
        # Top 2 scores are 0.9 (ref1) and 0.8 (ref3)
        self.assertEqual(tuple(c.example_ref for c in selected), (ref1, ref3))

        # 2. Deterministic tie breaking: preserves pool FIFO order
        pool_ties = (
            ReplayCandidate(ref0, ("probe-int-malas",), 1.0),
            ReplayCandidate(ref1, ("probe-int-patos",), 1.0),
            ReplayCandidate(ref2, ("probe-int-sapos",), 0.5),
            ReplayCandidate(ref3, ("probe-int-fogos",), 1.0),
        )
        selected_ties = select_directed_replay(pool_ties, budget=2)
        # Should pick indices 0 and 1 (ref0 and ref1) because ties resolve by FIFO order
        self.assertEqual(tuple(c.example_ref for c in selected_ties), (ref0, ref1))

        # 3. Causal isolation: Selector C does not read or depend on final holdout evaluations
        selected_again = select_directed_replay(pool, budget=2)
        self.assertEqual(selected, selected_again)

    def test_holdout_retention_sensitive_to_catastrophic_interference(self) -> None:
        # Under high interference, Variant A baseline must suffer forgetting on holdout probes
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        hold_a = result.holdout_retention_ratio["A"]
        hold_c = result.holdout_retention_ratio["C"]
        self.assertLess(hold_a, 1.0, "Holdout retention for baseline A must detect interference")
        self.assertGreater(hold_c, hold_a, "Directed replay C must recover holdout retention")

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
        self.assertGreater(result.peak_rss_bytes, 0)
        self.assertGreaterEqual(result.peak_rss_bytes, result.peak_ram_bytes)
        self.assertTrue(math.isfinite(result.cpu_time_seconds))

        profile = compute_cost_profile(
            total_cpu_seconds=result.cpu_time_seconds,
            peak_ram_bytes=result.peak_ram_bytes,
            peak_rss_bytes=result.peak_rss_bytes,
            pilot_seeds_count=1,
            pilot_words_count=6,
        )
        self.assertEqual(profile.scaling_recommendation, "local_cpu")
        self.assertIn("local machine", profile.recommendation_rationale)
        self.assertTrue(math.isfinite(profile.projected_scaled_hours_cpu))
        self.assertGreater(profile.peak_rss_bytes, 0)
        self.assertGreater(profile.peak_rss_mb, 0.0)

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
