from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

from kinera.backends import get_backend
from kinera.backends.base import BackendBuffer
from kinera.backends.cpu.backend import CPUBackend
import kinera.experiments.expanded_pilot as expanded_pilot
from kinera.experiments.expanded_pilot import (
    AggregatedMetric,
    ConditionAggregateResult,
    InterferenceLevel,
    NewTaskSpec,
    PilotConditionSpec,
    PilotCostProfile,
    PilotSeedResult,
    ProbePartitionBreakdown,
    TaskFamilySpec,
    compute_cost_profile,
    compute_probe_partition_breakdown,
    get_condition_spec,
    run_condition_seed,
    run_cpu_vs_cuda_microbenchmark,
    run_expanded_pilot,
    verify_scientific_equivalence,
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

    def test_positive_reference_retention_vs_zero_reference_acquisition(self) -> None:
        """Verify retention strictly evaluates ref > 0.0 probes while tracking ref <= 0.0 acquisition separately."""
        groups = ("probe-int-malas", "probe-int-patos", "probe-int-sapos", "probe-int-fogos")
        ref_comp = (1.0, 1.0, 1.0, 0.0)
        # Variant B in high-interference seed 42/44: loses all 3 learned probes, gains unlearned 4th probe
        post_comp_b = (0.0, 0.0, 0.0, 1.0)
        bd_b = compute_probe_partition_breakdown(groups, ref_comp, post_comp_b)

        self.assertEqual(
            bd_b.positive_ref_group_ids,
            ("probe-int-malas", "probe-int-patos", "probe-int-sapos"),
        )
        self.assertEqual(bd_b.zero_ref_group_ids, ("probe-int-fogos",))
        self.assertEqual(bd_b.positive_ref_retention_ratios, (0.0, 0.0, 0.0))
        self.assertEqual(bd_b.mean_positive_ref_retention, 0.0)
        self.assertEqual(bd_b.zero_ref_acquisition_deltas, (1.0,))
        self.assertEqual(bd_b.mean_zero_ref_acquisition, 1.0)

        # Verify actual high-interference seed 42 execution matches this exact breakdown
        result = run_condition_seed("high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab)
        self.assertEqual(result.eval_a.mean_retention_ratio, 0.0)
        self.assertEqual(result.eval_b.mean_retention_ratio, 0.0)
        self.assertEqual(result.eval_c.mean_retention_ratio, 1.0)
        self.assertEqual(result.holdout_retention_ratio["A"], 0.0)
        self.assertEqual(result.holdout_retention_ratio["B"], 0.0)
        self.assertEqual(result.holdout_retention_ratio["C"], 1.0)
        self.assertEqual(result.zero_ref_acquisition["A"], 1.0)
        self.assertEqual(result.zero_ref_acquisition["B"], 1.0)
        self.assertEqual(result.zero_ref_acquisition["C"], 0.0)
        self.assertEqual(result.holdout_zero_ref_acquisition["A"], 1.0)
        self.assertEqual(result.holdout_zero_ref_acquisition["B"], 1.0)
        self.assertEqual(result.holdout_zero_ref_acquisition["C"], 0.0)
        self.assertEqual(
            result.probe_breakdowns["C"]["internal"].positive_ref_group_ids,
            ("probe-int-malas", "probe-int-patos", "probe-int-sapos"),
        )
        self.assertEqual(
            result.probe_breakdowns["C"]["internal"].zero_ref_group_ids,
            ("probe-int-fogos",),
        )
        self.assertEqual(
            result.probe_breakdowns["C"]["holdout"].positive_ref_group_ids,
            ("probe-hold-maleta", "probe-hold-pateta", "probe-hold-sapato"),
        )
        self.assertEqual(
            result.probe_breakdowns["C"]["holdout"].zero_ref_group_ids,
            ("probe-hold-foguete",),
        )

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
        # 1. Default execution has tracemalloc disabled (profile_heap=False)
        with patch.object(tracemalloc, "start") as tm_start:
            default_res = run_condition_seed(
                "high", seed=42, char_vocab=self.char_vocab, syl_vocab=self.syl_vocab
            )
            tm_start.assert_not_called()
        self.assertGreater(default_res.cpu_time_seconds, 0.0)
        self.assertEqual(default_res.peak_ram_bytes, 0)
        self.assertGreater(default_res.peak_rss_bytes, 0)

        # 2. Explicit profile_heap=True enables tracemalloc and stops it cleanly
        result = run_condition_seed(
            "high",
            seed=42,
            char_vocab=self.char_vocab,
            syl_vocab=self.syl_vocab,
            profile_heap=True,
        )
        self.assertFalse(tracemalloc.is_tracing())
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
            profile_heap_enabled=True,
        )
        self.assertEqual(profile.scaling_recommendation, "local_cpu")
        self.assertIn("local machine", profile.recommendation_rationale)
        self.assertTrue(profile.profile_heap_enabled)
        self.assertEqual(profile.backend, "cpu")
        self.assertTrue(math.isfinite(profile.projected_scaled_hours_cpu))
        self.assertGreater(profile.peak_rss_bytes, 0)
        self.assertGreater(profile.peak_rss_mb, 0.0)

    def test_full_expanded_pilot_execution_and_schema(self) -> None:
        # Run small smoke pilot with 2 seeds and 2 conditions to test complete report generation
        report = run_expanded_pilot(
            seeds=(42, 43),
            conditions=("low", "high"),
            baseline_commit="5dba2ccd6fadd3c2406223e3ec28657cc1fb9138",
            runner_commit="test_commit",
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["phase"], "42-expanded-pilot")
        self.assertEqual(report["baseline_commit"], "5dba2ccd6fadd3c2406223e3ec28657cc1fb9138")
        self.assertEqual(report["runner_commit"], "test_commit")
        self.assertTrue(report["protocol_validated"])
        self.assertTrue(report["leakage_prevented"])
        self.assertEqual(report["conditions_evaluated"], ["low", "high"])
        self.assertEqual(report["seeds_evaluated"], [42, 43])
        self.assertEqual(
            report["execution_config"],
            {"backend": "cpu", "workers": 1, "profile_heap": False},
        )
        self.assertEqual(report["execution_metadata"]["status"], "completed")
        self.assertEqual(report["execution_metadata"]["backend"], "cpu")
        self.assertFalse(report["execution_metadata"]["resumed"])
        self.assertEqual(len(report["execution_metadata"]["completed_units"]), 4)
        self.assertEqual(report["execution_metadata"]["pending_units"], [])
        self.assertEqual(report["execution_metadata"]["failed_runs"], [])
        self.assertIn("condition_aggregates", report)
        high_agg = report["condition_aggregates"]["high"]
        self.assertIn("c_vs_b_retention_advantage", high_agg)
        self.assertIn("c_vs_b_holdout_retention_advantage", high_agg)
        self.assertIn("zero_ref_acquisition", high_agg)
        self.assertIn("holdout_zero_ref_acquisition", high_agg)
        self.assertIn("probe_partition_sets", high_agg)
        self.assertEqual(
            high_agg["probe_partition_sets"]["positive_ref_internal_groups"],
            ["probe-int-malas", "probe-int-patos", "probe-int-sapos"],
        )
        self.assertEqual(
            high_agg["probe_partition_sets"]["zero_ref_internal_groups"],
            ["probe-int-fogos"],
        )
        self.assertIn("new_competence_delta_during_replay", high_agg)
        self.assertIn("A", high_agg["new_competence_delta_during_replay"])
        self.assertIn("B", high_agg["new_competence_delta_during_replay"])
        self.assertIn("C", high_agg["new_competence_delta_during_replay"])
        self.assertIn("cost_profile", report)
        self.assertIn("conclusions", report)
        self.assertIn("high_interference_mean_recovery_advantage", report["conclusions"])
        self.assertIn("high_interference_mean_retention_advantage", report["conclusions"])
        self.assertIn("high_interference_zero_ref_acquisition_b", report["conclusions"])
        self.assertIn("high_interference_zero_ref_acquisition_c", report["conclusions"])
        self.assertTrue(report["conclusions"]["fairness_confirmed_all_seeds"])

        # Check JSON serializability
        dumped = json.dumps(report)
        self.assertTrue(len(dumped) > 100)

    def test_backend_selection_and_cuda_resource_release_lifecycle(self) -> None:
        """Confirm explicit backend selection and that each CUDA model releases its buffers before the next starts."""
        with self.assertRaisesRegex(ValueError, "backend"):
            run_expanded_pilot(seeds=(42,), conditions=("low",), backend="invalid")  # type: ignore[arg-type]

        class SimulatedCudaBridge:
            def __init__(self) -> None:
                self._next_ptr = 1000
                self.live: set[int] = set()

            def allocate(self, device_id: int, nbytes: int) -> int:
                ptr = self._next_ptr
                self._next_ptr += 1
                self.live.add(ptr)
                return ptr

            def free(self, device_id: int, pointer: int) -> None:
                self.live.remove(pointer)

            def device_info(self, device_id: int) -> dict[str, Any]:
                return {"name": "Simulated T4", "major": 7, "minor": 5, "total_memory_bytes": 16 * 1024**3}

        class SimulatedCudaBackend(CPUBackend):
            name = "cuda"

            def __init__(self) -> None:
                super().__init__()
                self.bridge = SimulatedCudaBridge()

        sim_backend = SimulatedCudaBackend()
        original_init = expanded_pilot.AutoregressiveModel.__init__
        original_close = expanded_pilot.AutoregressiveModel.close
        created_models: list[int] = []
        closed_models: list[int] = []

        def tracked_init(model: Any, *args: Any, **kwargs: Any) -> None:
            # Confirm previous model closed and freed all pointers BEFORE next model initializes
            self.assertEqual(
                len(created_models),
                len(closed_models),
                "Next model initialized before previous model was closed!",
            )
            self.assertEqual(
                len(sim_backend.bridge.live),
                0,
                "Next model initialized while previous CUDA allocations were still live!",
            )
            original_init(model, *args, **kwargs)
            # Allocate a simulated device parameter pointer owned by this model
            model._sim_ptr = sim_backend.bridge.allocate(0, 256)
            created_models.append(model._sim_ptr)

        def tracked_close(model: Any) -> None:
            original_close(model)
            ptr = getattr(model, "_sim_ptr", None)
            if ptr is not None and ptr in sim_backend.bridge.live:
                sim_backend.bridge.free(0, ptr)
                closed_models.append(ptr)

        with (
            patch.object(expanded_pilot, "get_backend", return_value=sim_backend) as mock_get_backend,
            patch.object(expanded_pilot.AutoregressiveModel, "__init__", tracked_init),
            patch.object(expanded_pilot.AutoregressiveModel, "close", tracked_close),
        ):
            report = run_expanded_pilot(
                seeds=(42, 43),
                conditions=("high",),
                backend="cuda",
            )

        mock_get_backend.assert_called_once_with("cuda")
        # 2 seeds * 3 variants (A, B, C) = 6 models created and closed sequentially
        self.assertEqual(len(created_models), 6)
        self.assertEqual(created_models, closed_models)
        self.assertEqual(len(sim_backend.bridge.live), 0)
        self.assertEqual(report["execution_config"]["backend"], "cuda")
        self.assertEqual(report["execution_metadata"]["backend"], "cuda")

    def test_cuda_model_closes_and_stops_tracemalloc_on_training_exception(self) -> None:
        """Ensure try/finally closes the model and stops tracemalloc even when training raises an exception."""
        class SimulatedCudaBridge:
            def __init__(self) -> None:
                self.live: set[int] = set()

            def allocate(self, device_id: int, nbytes: int) -> int:
                ptr = len(self.live) + 1
                self.live.add(ptr)
                return ptr

            def free(self, device_id: int, pointer: int) -> None:
                self.live.remove(pointer)

        class SimulatedCudaBackend(CPUBackend):
            name = "cuda"

            def __init__(self) -> None:
                super().__init__()
                self.bridge = SimulatedCudaBridge()

        sim_backend = SimulatedCudaBackend()
        original_init = expanded_pilot.AutoregressiveModel.__init__
        original_close = expanded_pilot.AutoregressiveModel.close
        close_calls = 0

        def tracked_init(model: Any, *args: Any, **kwargs: Any) -> None:
            original_init(model, *args, **kwargs)
            model._sim_ptr = sim_backend.bridge.allocate(0, 128)

        def tracked_close(model: Any) -> None:
            nonlocal close_calls
            close_calls += 1
            original_close(model)
            ptr = getattr(model, "_sim_ptr", None)
            if ptr is not None and ptr in sim_backend.bridge.live:
                sim_backend.bridge.free(0, ptr)

        with (
            patch.object(expanded_pilot.AutoregressiveModel, "__init__", tracked_init),
            patch.object(expanded_pilot.AutoregressiveModel, "close", tracked_close),
            patch.object(
                expanded_pilot,
                "train_multitask_examples",
                side_effect=RuntimeError("simulated training failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated training failure"):
                run_condition_seed(
                    "high",
                    seed=42,
                    char_vocab=self.char_vocab,
                    syl_vocab=self.syl_vocab,
                    backend=sim_backend,
                    profile_heap=True,
                )

        self.assertEqual(close_calls, 1)
        self.assertEqual(len(sim_backend.bridge.live), 0)
        self.assertFalse(tracemalloc.is_tracing())

    def test_incremental_checkpoint_and_resume_after_failure_and_seed_expansion(self) -> None:
        """Verify incremental checkpointing, failure recording, non-duplicating resume, and config checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "pilot_ckpt.json"
            real_run_seed = expanded_pilot.run_condition_seed
            executed_units: list[tuple[str, int]] = []

            def fail_on_second_unit(
                level: InterferenceLevel,
                seed: int,
                char_vocab: Any,
                syl_vocab: Any,
                **kwargs: Any,
            ) -> PilotSeedResult:
                executed_units.append((level, seed))
                if (level, seed) == ("low", 43):
                    raise RuntimeError("transient interruption on seed 43")
                return real_run_seed(level, seed, char_vocab, syl_vocab, **kwargs)

            with patch.object(expanded_pilot, "run_condition_seed", side_effect=fail_on_second_unit):
                with self.assertRaisesRegex(RuntimeError, "transient interruption on seed 43"):
                    run_expanded_pilot(
                        seeds=(42, 43),
                        conditions=("low", "high"),
                        checkpoint_path=ckpt_path,
                        resume=False,
                    )

            # Verify partial checkpoint persisted unit (low, 42) and recorded failure on (low, 43)
            self.assertTrue(ckpt_path.exists())
            partial = json.loads(ckpt_path.read_text(encoding="utf-8"))
            self.assertEqual(partial["execution_metadata"]["status"], "failed")
            self.assertEqual(
                partial["execution_metadata"]["completed_units"],
                [{"level": "low", "seed": 42}],
            )
            self.assertEqual(len(partial["seed_results"]), 1)
            self.assertEqual(len(partial["execution_metadata"]["failed_runs"]), 1)
            self.assertEqual(
                partial["execution_metadata"]["failed_runs"][0]["seed"], 43
            )

            # Reject backend mismatch BEFORE creating/compiling backend
            with patch.object(
                expanded_pilot,
                "_prepare_execution_backend",
                side_effect=AssertionError("backend initialized before resume check"),
            ) as mock_prep:
                with self.assertRaisesRegex(ValueError, "backend"):
                    run_expanded_pilot(
                        seeds=(42, 43),
                        conditions=("low", "high"),
                        backend="cuda",
                        checkpoint_path=ckpt_path,
                        resume=True,
                    )
                mock_prep.assert_not_called()

            # Reject baseline_commit mismatch and shrinking seeds
            with self.assertRaisesRegex(ValueError, "baseline_commit"):
                run_expanded_pilot(
                    seeds=(42, 43),
                    conditions=("low", "high"),
                    baseline_commit="wrong_commit",
                    checkpoint_path=ckpt_path,
                    resume=True,
                )
            with self.assertRaisesRegex(ValueError, "seeds"):
                run_expanded_pilot(
                    seeds=(99,),
                    conditions=("low", "high"),
                    checkpoint_path=ckpt_path,
                    resume=True,
                )

            # Now resume normally and expand seeds to (42, 43, 44):
            # (low, 42) must NOT be re-executed or duplicated!
            resumed_executed: list[tuple[str, int]] = []

            def track_resumed(
                level: InterferenceLevel,
                seed: int,
                char_vocab: Any,
                syl_vocab: Any,
                **kwargs: Any,
            ) -> PilotSeedResult:
                resumed_executed.append((level, seed))
                return real_run_seed(level, seed, char_vocab, syl_vocab, **kwargs)

            with patch.object(expanded_pilot, "run_condition_seed", side_effect=track_resumed):
                resumed_report = run_expanded_pilot(
                    seeds=(42, 43, 44),
                    conditions=("low", "high"),
                    checkpoint_path=ckpt_path,
                    resume=True,
                )

            self.assertNotIn(("low", 42), resumed_executed)
            self.assertEqual(
                resumed_executed,
                [("low", 43), ("low", 44), ("high", 42), ("high", 43), ("high", 44)],
            )
            self.assertEqual(resumed_report["execution_metadata"]["status"], "completed")
            self.assertTrue(resumed_report["execution_metadata"]["resumed"])
            self.assertEqual(resumed_report["execution_metadata"]["resume_count"], 1)
            self.assertEqual(
                resumed_report["execution_metadata"]["resumed_completed_units"],
                [{"level": "low", "seed": 42}],
            )
            self.assertEqual(len(resumed_report["seed_results"]), 6)

            # Compare resumed report against fresh uninterrupted report
            fresh_report = run_expanded_pilot(
                seeds=(42, 43, 44),
                conditions=("low", "high"),
            )
            eq = verify_scientific_equivalence(fresh_report, resumed_report, atol=0.0)
            self.assertTrue(eq["equivalent"])
            self.assertEqual(eq["max_absolute_error"], 0.0)
            self.assertEqual(eq["checked_seed_units"], 6)

    def test_verify_scientific_equivalence_rejects_per_seed_divergence_with_equal_aggregates(self) -> None:
        """Ensure aggregate equality is never sufficient when individual seed results diverge."""
        base_report = run_expanded_pilot(
            seeds=(42, 43),
            conditions=("high",),
        )
        mutated_report = copy.deepcopy(base_report)
        # Perturb seed 42 by +0.1 and seed 43 by -0.1 so the mean aggregate remains identical
        mutated_report["seed_results"][0]["evaluations"]["C"]["mean_recovery_delta"] += 0.1
        mutated_report["seed_results"][1]["evaluations"]["C"]["mean_recovery_delta"] -= 0.1

        with self.assertRaisesRegex(ValueError, r"seed_results\[high:42\]\.evaluations\.C\.mean_recovery_delta"):
            verify_scientific_equivalence(base_report, mutated_report)

    def test_cpu_vs_cuda_microbenchmark_separates_init_and_verifies_parity(self) -> None:
        """Verify microbenchmark separates backend_init_seconds from training_wall_time_seconds."""
        class SimulatedCudaBridge:
            def __init__(self) -> None:
                self.live: set[int] = set()
                self.allocs = 0
                self.frees = 0

            def allocate(self, device_id: int, nbytes: int) -> int:
                self.allocs += 1
                ptr = self.allocs
                self.live.add(ptr)
                return ptr

            def free(self, device_id: int, pointer: int) -> None:
                self.frees += 1
                self.live.remove(pointer)

            def device_info(self, device_id: int) -> dict[str, Any]:
                return {"name": "Simulated T4", "major": 7, "minor": 5, "total_memory_bytes": 16 * 1024**3}

        class SimulatedCudaBackend(CPUBackend):
            name = "cuda"

            def __init__(self) -> None:
                super().__init__()
                self.bridge = SimulatedCudaBridge()

        sim_cuda = SimulatedCudaBackend()
        bench = run_cpu_vs_cuda_microbenchmark(
            seeds=(42,),
            conditions=("high",),
            cuda_backend=sim_cuda,
            require_cuda=True,
        )
        self.assertEqual(bench["phase"], "42-cpu-vs-cuda-microbenchmark")
        self.assertFalse(bench["workload"]["profile_heap"])
        self.assertTrue(bench["cpu"]["available"])
        self.assertGreaterEqual(bench["cpu"]["backend_init_seconds"], 0.0)
        self.assertGreater(bench["cpu"]["training_wall_time_seconds"], 0.0)
        self.assertTrue(bench["cuda"]["available"])
        self.assertGreaterEqual(bench["cuda"]["backend_init_seconds"], 0.0)
        self.assertGreater(bench["cuda"]["training_wall_time_seconds"], 0.0)
        self.assertEqual(bench["cuda"]["live_allocations_after_close"], 0)
        self.assertEqual(bench["cuda"]["unmatched_frees"], 0)
        self.assertTrue(bench["comparison"]["scientific_equivalence_verified"])
        self.assertEqual(bench["comparison"]["max_absolute_error"], 0.0)
        self.assertGreater(bench["comparison"]["training_speedup_cuda_vs_cpu"], 0.0)

    def test_resource_leak_detector_raises_if_model_leaves_cuda_buffer_open(self) -> None:
        """Confirm that if a model fails to release its CUDA buffer on close(), an error is raised immediately."""
        class LeakingCudaBridge:
            def __init__(self) -> None:
                self.live: set[int] = set()

            def allocate(self, device_id: int, nbytes: int) -> int:
                ptr = len(self.live) + 1
                self.live.add(ptr)
                return ptr

            def free(self, device_id: int, pointer: int) -> None:
                self.live.remove(pointer)

        class LeakingCudaBackend(CPUBackend):
            name = "cuda"

            def __init__(self) -> None:
                super().__init__()
                self.bridge = LeakingCudaBridge()

        leaking_backend = LeakingCudaBackend()
        original_init = expanded_pilot.AutoregressiveModel.__init__

        def leaking_init(model: Any, *args: Any, **kwargs: Any) -> None:
            original_init(model, *args, **kwargs)
            leaking_backend.bridge.allocate(0, 64)

        with patch.object(expanded_pilot.AutoregressiveModel, "__init__", leaking_init):
            with self.assertRaisesRegex(RuntimeError, "unreleased device buffer"):
                run_condition_seed(
                    "low",
                    seed=42,
                    char_vocab=self.char_vocab,
                    syl_vocab=self.syl_vocab,
                    backend=leaking_backend,
                )

    def test_cli_runner_seeds_resume_and_verification(self) -> None:
        """Verify CLI script flags (--seeds, --conditions, --resume, --verify-against, --benchmark-backends)."""
        from scripts.run_expanded_replay_pilot import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "cli_out.json"
            bench_file = Path(tmpdir) / "cli_bench.json"

            # 1. Run CLI with seed 42
            rc1 = cli_main(
                [
                    "--seeds",
                    "42",
                    "--conditions",
                    "high",
                    "--output",
                    str(out_file),
                ]
            )
            self.assertEqual(rc1, 0)
            self.assertTrue(out_file.exists())

            # 2. Resume CLI adding seed 43 and verify against itself
            rc2 = cli_main(
                [
                    "--seeds",
                    "42",
                    "43",
                    "--conditions",
                    "high",
                    "--output",
                    str(out_file),
                    "--resume",
                    "--verify-against",
                    str(out_file),
                ]
            )
            self.assertEqual(rc2, 0)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["seeds_evaluated"], [42, 43])
            self.assertTrue(payload["execution_metadata"]["resumed"])

            # 3. Run CLI --benchmark-backends (CPU-only fallback on non-CUDA host)
            rc3 = cli_main(
                [
                    "--seeds",
                    "42",
                    "--conditions",
                    "high",
                    "--benchmark-backends",
                    "--output",
                    str(bench_file),
                ]
            )
            self.assertEqual(rc3, 0)
            bench_payload = json.loads(bench_file.read_text(encoding="utf-8"))
            self.assertEqual(bench_payload["phase"], "42-cpu-vs-cuda-microbenchmark")

    def test_summary_claim_dynamically_reflects_evaluated_seeds_and_metrics(self) -> None:
        """Verify summary_claim derives from actual metrics and seed counts without hardcoded literals."""
        # 1. Test 3-seed report
        p3_path = Path("experiments/phase42_pilot_results.json")
        self.assertTrue(p3_path.exists())
        d3 = json.loads(p3_path.read_text(encoding="utf-8"))
        completed_map_3 = {
            (res.level, res.seed): res
            for item in d3["seed_results"]
            if (res := PilotSeedResult.from_dict(item)) is not None
        }
        payload_3 = expanded_pilot._build_pilot_report_payload(
            baseline_commit=d3["baseline_commit"],
            runner_commit=d3["runner_commit"],
            conditions=tuple(d3["conditions_evaluated"]),
            seeds=tuple(d3["seeds_evaluated"]),
            experiment_config=d3.get("experiment_config", {}),
            execution_config=d3.get("execution_config", {}),
            completed_map=completed_map_3,
            resumed=False,
            resume_count=0,
            resumed_units=[],
            failed_runs=[],
            status="completed",
            backend_name="cpu",
            profile_heap=False,
            backend_init_seconds=0.0,
            elapsed_wall_seconds=10.0,
        )
        claim_3 = payload_3["conclusions"]["summary_claim"]
        self.assertIn("budget parity (3 seeds)", claim_3)
        self.assertIn("with C retaining 1.0000 vs B retaining 0.3333", claim_3)
        self.assertIn("in 2 of 3 seeds (seeds 42 and 44)", claim_3)
        self.assertIn("new task (-0.50)", claim_3)
        self.assertIn("strict empirical superiority across seeds was 0.0%", claim_3)
        self.assertIn("Because this pilot evaluated 3 exploratory seeds", claim_3)

        # 2. Test 30-seed report
        p30_path = Path("experiments/phase42_30seeds_t4_results.json")
        self.assertTrue(p30_path.exists())
        d30 = json.loads(p30_path.read_text(encoding="utf-8"))
        completed_map_30 = {
            (res.level, res.seed): res
            for item in d30["seed_results"]
            if (res := PilotSeedResult.from_dict(item)) is not None
        }
        payload_30 = expanded_pilot._build_pilot_report_payload(
            baseline_commit=d30["baseline_commit"],
            runner_commit=d30["runner_commit"],
            conditions=tuple(d30["conditions_evaluated"]),
            seeds=tuple(d30["seeds_evaluated"]),
            experiment_config=d30.get("experiment_config", {}),
            execution_config=d30.get("execution_config", {}),
            completed_map=completed_map_30,
            resumed=True,
            resume_count=1,
            resumed_units=[],
            failed_runs=[],
            status="completed",
            backend_name="cuda",
            profile_heap=False,
            backend_init_seconds=0.23,
            elapsed_wall_seconds=113.96,
        )
        claim_30 = payload_30["conclusions"]["summary_claim"]
        # Assertions for 30-seed dynamic metrics
        self.assertIn("budget parity (30 seeds)", claim_30)
        self.assertIn("+0.2417 across all 4 old task probes", claim_30)
        self.assertIn("+0.4778, with C retaining 0.9778 vs B retaining 0.5000", claim_30)
        self.assertIn("mean 0.5000 for B and 0.0333 for C", claim_30)
        self.assertIn(
            "wastes 50% of its budget on the unforgotten item in 14 of 30 seeds, with retention falling to 0.0 in 15 of 30 seeds",
            claim_30,
        )
        self.assertIn("new task (-0.4833)", claim_30)
        self.assertIn("strict empirical superiority across seeds was 3.3%", claim_30)
        self.assertIn("Because this pilot evaluated 30 seeds on synthetic micro-fixtures", claim_30)

        # Explicitly assert 3-seed literals are NEVER present in 30-seed report
        self.assertNotIn("1.0000 vs B retaining 0.3333", claim_30)
        self.assertNotIn("2 of 3 seeds", claim_30)
        self.assertNotIn("seeds 42 and 44", claim_30)
        self.assertNotIn("new task (-0.50)", claim_30)
        self.assertNotIn("superiority across seeds was 0.0%", claim_30)
        self.assertNotIn("3 exploratory seeds", claim_30)


if __name__ == "__main__":
    unittest.main()
