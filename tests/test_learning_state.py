from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import sys
import unittest
from typing import Any
from unittest.mock import ANY, Mock, patch

from kinera.experiments.learning_memory import (
    ExampleRef,
    LearningEvent,
    LearningMemory,
    LearningMemoryRecorder,
)
from kinera.experiments.learning_state import (
    LearningState,
    build_learning_state,
    calculate_forgetting,
    oriented_competence_delta,
)
from kinera.experiments.retention import (
    ProbeObservation,
    RetentionProbeGroup,
    evaluate_retention_probe,
    resolve_probe_examples,
)
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples


class RetentionProbeGroupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = ExampleRef("fixture:l2", "probe", 0)
        self.second = ExampleRef("fixture:l2", "probe", 1)

    def test_group_is_immutable_and_keeps_explicit_metric_direction(self) -> None:
        group = RetentionProbeGroup(
            "vowels",
            (self.first, self.second),
            "char_accuracy",
            "higher_is_better",
        )
        self.assertEqual(group.example_refs, (self.first, self.second))
        with self.assertRaises((AttributeError, TypeError)):
            group.group_id = "other"  # type: ignore[misc]

    def test_group_rejects_empty_duplicate_or_invalid_contract_fields(self) -> None:
        invalid = (
            ("", (self.first,), "char_accuracy", "higher_is_better"),
            ("vowels", (), "char_accuracy", "higher_is_better"),
            ("vowels", (self.first, self.first), "char_accuracy", "higher_is_better"),
            ("vowels", ("not-a-ref",), "char_accuracy", "higher_is_better"),
            ("vowels", (self.first,), "", "higher_is_better"),
            ("vowels", (self.first,), "char_accuracy", "best_is_better"),
            ("vowels", [self.first], "char_accuracy", "higher_is_better"),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                RetentionProbeGroup(*args)  # type: ignore[arg-type]

    def test_probe_resolution_follows_group_order_and_rejects_missing_or_duplicate_refs(self) -> None:
        group = RetentionProbeGroup(
            "vowels",
            (self.second, self.first),
            "char_accuracy",
            "higher_is_better",
        )
        example_first = {"id": "first"}
        example_second = {"id": "second"}
        resolved = resolve_probe_examples(
            group,
            (self.first, self.second),
            (example_first, example_second),
        )
        self.assertEqual(resolved, (example_second, example_first))
        with self.assertRaisesRegex(KeyError, "missing"):
            resolve_probe_examples(group, (self.first,), (example_first,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_probe_examples(
                group,
                (self.first, self.first),
                (example_first, example_second),
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            resolve_probe_examples(group, (self.first,), (example_first, example_second))


class ProbeObservationTests(unittest.TestCase):
    def test_observation_requires_finite_competence_and_positive_support(self) -> None:
        observation = ProbeObservation(
            "vowels", "char_accuracy", "higher_is_better", 0.75, 3
        )
        self.assertEqual(observation.competence, 0.75)
        self.assertEqual(observation.support, 3)
        with self.assertRaises((AttributeError, TypeError)):
            observation.support = 4  # type: ignore[misc]
        invalid = (
            ("", "char_accuracy", "higher_is_better", 0.75, 3),
            ("vowels", "", "higher_is_better", 0.75, 3),
            ("vowels", "char_accuracy", "higher_is_better", math.nan, 3),
            ("vowels", "char_accuracy", "higher_is_better", math.inf, 3),
            ("vowels", "char_accuracy", "higher_is_better", 0.75, 0),
            ("vowels", "char_accuracy", "higher_is_better", 0.75, True),
            ("vowels", "char_accuracy", "unknown", 0.75, 3),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ProbeObservation(*args)  # type: ignore[arg-type]

    def test_retention_evaluation_extracts_declared_metric_and_actual_support(self) -> None:
        ref_a = ExampleRef("fixture:l2", "probe", 0)
        ref_b = ExampleRef("fixture:l2", "probe", 1)
        group = RetentionProbeGroup(
            "vowels", (ref_b, ref_a), "char_accuracy", "higher_is_better"
        )
        refs = (ref_a, ref_b)
        examples = ({"example": "a"}, {"example": "b"})
        model = object()
        vocabulary = object()
        report = {"char_accuracy": 0.625, "examples": 2}

        with patch(
            "kinera.training.evaluator.evaluate_multitask_examples",
            return_value=report,
        ) as evaluator:
            observation = evaluate_retention_probe(
                model,
                group,
                refs,
                examples,
                vocabulary,
            )

        self.assertEqual(
            observation,
            ProbeObservation("vowels", "char_accuracy", "higher_is_better", 0.625, 2),
        )
        evaluator.assert_called_once_with(model, [examples[1], examples[0]], vocabulary)

    def test_probe_requires_available_scalar_metric_and_exact_positive_support(self) -> None:
        ref = ExampleRef("fixture:l2", "probe", 0)
        example = {"example": "a"}
        group = RetentionProbeGroup(
            "vowels", (ref,), "char_accuracy", "higher_is_better"
        )
        for report in (
            {"char_accuracy": 0.5},
            {"char_accuracy": 0.5, "examples": 0},
            {"char_accuracy": 0.5, "examples": True},
            {"char_accuracy": 0.5, "examples": 2},
        ):
            with self.subTest(report=report), patch(
                "kinera.training.evaluator.evaluate_multitask_examples",
                return_value=report,
            ), self.assertRaises(ValueError):
                evaluate_retention_probe(object(), group, (ref,), (example,), object())
        support_group = RetentionProbeGroup(
            "support", (ref,), "examples", "higher_is_better"
        )
        with patch(
            "kinera.training.evaluator.evaluate_multitask_examples",
            return_value={"examples": 1},
        ), self.assertRaisesRegex(ValueError, "support metadata"):
            evaluate_retention_probe(
                object(), support_group, (ref,), (example,), object()
            )

    def test_missing_or_non_scalar_metric_is_an_explicit_error_not_zero(self) -> None:
        ref = ExampleRef("fixture:l2", "probe", 0)
        examples = ({"example": "a"},)
        for report, error in (
            ({"examples": 1}, KeyError),
            ({"char_accuracy": math.nan, "examples": 1}, ValueError),
            ({"char_accuracy": {"nested": 0.5}, "examples": 1}, TypeError),
        ):
            group = RetentionProbeGroup(
                "vowels", (ref,), "char_accuracy", "higher_is_better"
            )
            with self.subTest(report=report), patch(
                "kinera.training.evaluator.evaluate_multitask_examples",
                return_value=report,
            ), self.assertRaises(error):
                evaluate_retention_probe(object(), group, (ref,), examples, object())


class LearningStateTests(unittest.TestCase):
    def _observation(
        self,
        competence: float,
        *,
        group_id: str = "vowels",
        metric: str = "char_accuracy",
        direction: str = "higher_is_better",
    ) -> ProbeObservation:
        return ProbeObservation(group_id, metric, direction, competence, 4)  # type: ignore[arg-type]

    def test_forgetting_uses_exact_metric_direction_without_thresholds(self) -> None:
        cases = (
            (0.8, 0.9, "higher_is_better", 0.0),
            (0.8, 0.8, "higher_is_better", 0.0),
            (0.8, 0.55, "higher_is_better", 0.25),
            (0.3, 0.2, "lower_is_better", 0.0),
            (0.2, 0.2, "lower_is_better", 0.0),
            (0.2, 0.4, "lower_is_better", 0.2),
        )
        for reference, current, direction, expected in cases:
            with self.subTest(direction=direction, reference=reference, current=current):
                self.assertAlmostEqual(
                    calculate_forgetting(reference, current, direction), expected
                )

    def test_competence_delta_is_oriented_with_positive_meaning_improvement(self) -> None:
        self.assertAlmostEqual(
            oriented_competence_delta(0.4, 0.7, "higher_is_better"), 0.3
        )
        self.assertAlmostEqual(
            oriented_competence_delta(0.4, 0.2, "lower_is_better"), 0.2
        )
        self.assertAlmostEqual(
            oriented_competence_delta(0.4, 0.7, "lower_is_better"), -0.3
        )

    def test_forgetting_rejects_invalid_direction_and_non_finite_values(self) -> None:
        for args in (
            (math.nan, 0.5, "higher_is_better"),
            (0.5, math.inf, "higher_is_better"),
            (True, 0.5, "higher_is_better"),
            (0.5, 0.4, "unknown"),
        ):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                calculate_forgetting(*args)  # type: ignore[arg-type]

    def test_state_is_immutable_and_can_represent_new_and_unresolved(self) -> None:
        initial = LearningState(
            "vowels", None, None, None, 0, 0, None, "NEW"
        )
        unresolved = LearningState(
            "vowels", 0.8, 0.5, 0.3, 2, 0, 12, "UNRESOLVED", -0.3
        )
        self.assertEqual(initial.evidence_count, 0)
        self.assertEqual(unresolved.status, "UNRESOLVED")
        self.assertEqual(unresolved.replay_count, 0)
        with self.assertRaises((AttributeError, TypeError)):
            unresolved.status = "STABLE"  # type: ignore[misc]

    def test_state_validates_evidence_replay_status_and_finite_values(self) -> None:
        valid = ("vowels", 0.8, 0.5, 0.3, 2, 0, 12, "UNRESOLVED")
        invalid = (
            ("vowels", 0.8, 0.5, 0.3, -1, 0, 12, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, 0.3, True, 0, 12, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, 0.3, 2, 1, 12, "UNRESOLVED"),
            ("vowels", math.nan, 0.5, 0.3, 2, 0, 12, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, math.inf, 2, 0, 12, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, -0.1, 2, 0, 12, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, 0.3, 2, 0, -1, "UNRESOLVED"),
            ("vowels", 0.8, 0.5, 0.3, 2, 0, 12, "UNKNOWN"),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                LearningState(*args)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LearningState(*valid, competence_delta=math.nan)  # type: ignore[arg-type]

    def test_pair_build_uses_two_evidences_and_preserves_harness_status(self) -> None:
        reference = self._observation(0.8)
        current = self._observation(0.5)
        state = build_learning_state(
            reference,
            current,
            status="STABLE",
            last_practiced_step=160,
        )
        self.assertEqual(state.reference_competence, 0.8)
        self.assertEqual(state.current_competence, 0.5)
        self.assertAlmostEqual(state.forgetting_score or 0.0, 0.3)
        self.assertAlmostEqual(state.competence_delta or 0.0, -0.3)
        self.assertEqual(state.evidence_count, 2)
        self.assertEqual(state.replay_count, 0)
        self.assertEqual(state.last_practiced_step, 160)
        # The caller-provided status is retained; no numeric threshold is applied.
        self.assertEqual(state.status, "STABLE")

    def test_pair_build_rejects_reused_observation_object(self) -> None:
        observation = self._observation(0.8)
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_learning_state(
                observation,
                observation,
                status="LEARNING",
                last_practiced_step=10,
            )

    def test_pair_build_rejects_mismatched_group_metric_or_direction(self) -> None:
        reference = self._observation(0.8)
        mismatches = (
            self._observation(0.5, group_id="other"),
            self._observation(0.5, metric="total_loss"),
            self._observation(0.5, direction="lower_is_better"),
        )
        for current in mismatches:
            with self.subTest(current=current), self.assertRaises(ValueError):
                build_learning_state(
                    reference,
                    current,
                    status="LEARNING",
                    last_practiced_step=10,
                )


class RetentionProbeExecutionTests(unittest.TestCase):
    @staticmethod
    def _parameter_snapshot(model: AutoregressiveModel) -> object:
        return copy.deepcopy(
            (
                model.representation.char_embedding.weights,
                model.representation.syllable_embedding.weights,
                model.heads.char.weights,
                model.heads.char.bias,
                model.heads.syllable.weights,
                model.heads.syllable.bias,
                model.heads.boundary.weights,
                model.heads.boundary.bias,
            )
        )

    @staticmethod
    def _example(context_ids: list[int], features: list[list[float]], target: int) -> dict[str, Any]:
        return {
            "context_ids": context_ids,
            "context_syllable_ids": [0, 1],
            "context_feature_vectors": features,
            "target_char_id": target,
            "target_syllable_id": 1,
            "target_is_syllable_end": 1,
        }

    def _evaluate_without_training_side_effects(
        self,
        model: AutoregressiveModel,
        group: RetentionProbeGroup,
        refs: tuple[ExampleRef, ...],
        examples: tuple[dict[str, Any], ...],
        vocabulary: CharacterVocabulary,
        memory: LearningMemory,
    ) -> ProbeObservation:
        parameters_before = self._parameter_snapshot(model)
        rng_before = random.getstate()
        memory_before = memory.entries()
        with (
            patch.object(
                model,
                "multitask_train_step",
                side_effect=AssertionError("retention probe called training step"),
            ) as train_step,
            patch.object(
                model.heads,
                "backward",
                side_effect=AssertionError("retention probe called backward"),
            ) as heads_backward,
            patch.object(
                model.heads,
                "step",
                side_effect=AssertionError("retention probe called step"),
            ) as heads_step,
            patch.object(
                model.representation,
                "backward",
                side_effect=AssertionError("retention probe called backward"),
            ) as representation_backward,
            patch.object(
                model.representation,
                "step",
                side_effect=AssertionError("retention probe called step"),
            ) as representation_step,
            patch.object(model, "multitask_eval_step", wraps=model.multitask_eval_step) as eval_step,
            patch.object(memory, "observe", wraps=memory.observe) as memory_write,
            patch(
                "kinera.training.evaluator.evaluate_multitask_examples",
                wraps=evaluate_multitask_examples,
            ) as evaluator,
        ):
            observation = evaluate_retention_probe(
                model,
                group,
                refs,
                examples,
                vocabulary,
            )
        evaluator.assert_called_once()
        eval_step.assert_called_once_with(
            examples[refs.index(group.example_refs[0])]["context_ids"],
            examples[refs.index(group.example_refs[0])]["context_syllable_ids"],
            examples[refs.index(group.example_refs[0])]["context_feature_vectors"],
            targets={
                "char": examples[refs.index(group.example_refs[0])]["target_char_id"],
                "syllable": examples[refs.index(group.example_refs[0])]["target_syllable_id"],
                "boundary": float(examples[refs.index(group.example_refs[0])]["target_is_syllable_end"]),
            },
            loss_function=ANY,
        )
        train_step.assert_not_called()
        heads_backward.assert_not_called()
        heads_step.assert_not_called()
        representation_backward.assert_not_called()
        representation_step.assert_not_called()
        memory_write.assert_not_called()
        self.assertEqual(memory.entries(), memory_before)
        self.assertEqual(self._parameter_snapshot(model), parameters_before)
        self.assertEqual(random.getstate(), rng_before)
        self.assertTrue(all(not memory.contains(ref) for ref in group.example_refs))
        return observation

    def test_probe_uses_evaluator_only_and_preserves_model_rng_and_memory(self) -> None:
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        model = AutoregressiveModel(
            len(vocabulary),
            3,
            char_embedding_dim=2,
            syllable_embedding_dim=2,
            seed=17,
        )
        self.addCleanup(model.close)
        ref = ExampleRef("fixture:l2-isolation", "probe", 0)
        training_ref = ExampleRef("fixture:l2-isolation", "train", 0)
        example = self._example([0, 0], [[0.0] * 10, [0.0] * 10], 3)
        group = RetentionProbeGroup(
            "isolation", (ref,), "char_accuracy", "higher_is_better"
        )
        memory = LearningMemory(capacity=2)
        memory.observe(LearningEvent(training_ref, 1, 1, "pre_update", 1, 1, 1, 3))

        observation = self._evaluate_without_training_side_effects(
            model, group, (ref,), (example,), vocabulary, memory
        )

        self.assertEqual(observation.support, 1)
        self.assertEqual(observation.group_id, "isolation")
        self.assertFalse(memory.contains(ref))

    def test_controlled_synthetic_interference_is_reproducible_and_not_language_evidence(self) -> None:
        """Toy A/probe-A/B interference checks mechanics, not linguistic forgetting."""
        vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b"])
        zeros = [[0.0] * 10, [0.0] * 10]
        feature_one = [[0.0] * 10, [1.0] + [0.0] * 9]
        train_a = (
            self._example([0, 0], zeros, 3),
            self._example([0, 3], feature_one, 3),
        )
        probe_example = self._example([3, 3], feature_one, 3)
        train_b = (
            self._example([0, 0], zeros, 4),
            self._example([0, 4], feature_one, 4),
        )
        refs_a = tuple(
            ExampleRef("fixture:interference", "train-a", index) for index in range(2)
        )
        probe_ref = ExampleRef("fixture:interference", "probe-a", 0)
        refs_b = tuple(
            ExampleRef("fixture:interference", "train-b", index) for index in range(2)
        )
        group = RetentionProbeGroup(
            "synthetic-a", (probe_ref,), "char_accuracy", "higher_is_better"
        )

        def run() -> tuple[ProbeObservation, ProbeObservation, LearningState]:
            model = AutoregressiveModel(
                len(vocabulary),
                3,
                char_embedding_dim=2,
                syllable_embedding_dim=2,
                seed=7,
            )
            memory = LearningMemory(capacity=4)
            try:
                train_multitask_examples(
                    model,
                    list(train_a),
                    vocabulary,
                    epochs=40,
                    learning_rate=0.1,
                    step_observer=LearningMemoryRecorder(refs_a, memory),
                )
                reference = self._evaluate_without_training_side_effects(
                    model,
                    group,
                    (probe_ref,),
                    (probe_example,),
                    vocabulary,
                    memory,
                )
                train_multitask_examples(
                    model,
                    list(train_b),
                    vocabulary,
                    epochs=40,
                    learning_rate=0.1,
                    step_observer=LearningMemoryRecorder(refs_b, memory),
                )
                current = self._evaluate_without_training_side_effects(
                    model,
                    group,
                    (probe_ref,),
                    (probe_example,),
                    vocabulary,
                    memory,
                )
                state = build_learning_state(
                    reference,
                    current,
                    status="LEARNING",
                    last_practiced_step=160,
                )
                self.assertTrue(all(memory.contains(ref) for ref in (*refs_a, *refs_b)))
                self.assertFalse(memory.contains(probe_ref))
                return reference, current, state
            finally:
                model.close()

        first_run = run()
        second_run = run()
        self.assertEqual(first_run, second_run)
        reference, current, state = first_run
        self.assertEqual(reference.support, 1)
        self.assertEqual(current.support, 1)
        self.assertEqual(reference.competence, 1.0)
        self.assertEqual(current.competence, 0.0)
        self.assertEqual(state.reference_competence, reference.competence)
        self.assertEqual(state.current_competence, current.competence)
        self.assertEqual(state.forgetting_score, 1.0)
        self.assertEqual(state.competence_delta, -1.0)
        self.assertEqual(state.evidence_count, 2)
        self.assertEqual(state.replay_count, 0)

    def test_natural_word_fixture_smokes_probe_mapping_without_language_claim(self) -> None:
        """Hand-authored Portuguese words exercise the pipeline, not corpus evidence."""
        train_records = [
            annotate_record({
                "word_key": "casa",
                "word_normalized": "casa",
                "syllables": ["ca", "sa"],
            }),
            annotate_record({
                "word_key": "pato",
                "word_normalized": "pato",
                "syllables": ["pa", "to"],
            }),
        ]
        probe_record = annotate_record({
            "word_key": "capa",
            "word_normalized": "capa",
            "syllables": ["ca", "pa"],
        })
        char_vocabulary = CharacterVocabulary.from_records(train_records)
        syllable_vocabulary = SyllableVocabulary.from_records(train_records)
        probe_examples = tuple(
            build_autoregressive_examples(
                [probe_record],
                char_vocabulary,
                context_size=2,
                syllable_vocabulary=syllable_vocabulary,
            )
        )
        refs = tuple(
            ExampleRef("fixture:natural-words:casa-pato-capa", "probe", index)
            for index in range(len(probe_examples))
        )
        group = RetentionProbeGroup(
            "natural-word-smoke", refs, "char_accuracy", "higher_is_better"
        )
        model = AutoregressiveModel(
            len(char_vocabulary),
            len(syllable_vocabulary),
            seed=23,
        )
        self.addCleanup(model.close)

        observation = evaluate_retention_probe(
            model,
            group,
            refs,
            probe_examples,
            char_vocabulary,
        )

        self.assertEqual(observation.support, len(probe_examples))
        self.assertGreater(observation.support, 0)
        self.assertTrue(math.isfinite(observation.competence))


class L2ImportBoundaryTests(unittest.TestCase):
    def test_import_has_no_cuda_telemetry_clock_proc_thread_or_kema_side_effects(self) -> None:
        project_root = __import__("pathlib").Path(__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
        script = r"""
import builtins, sys, threading, time
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith(("kinera.backends", "kinera.monitor", "kinera.training", "kinera.nn", "kinera.kema")):
        raise AssertionError("forbidden L2 import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
for clock in ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns"):
    setattr(time, clock, lambda *a, **k: (_ for _ in ()).throw(AssertionError("clock called")))
original_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if str(file).startswith("/proc/"):
        raise AssertionError("/proc read")
    return original_open(file, *args, **kwargs)
builtins.open = guarded_open
threading.Thread.__init__ = lambda *a, **k: (_ for _ in ()).throw(AssertionError("thread created"))
import kinera.experiments.retention
import kinera.experiments.learning_state
assert not any(name.startswith("kinera.backends.cuda") for name in sys.modules)
assert not any(name.startswith("kinera.monitor.telemetry") for name in sys.modules)
assert not any(name.startswith("kinera.training") for name in sys.modules)
assert not any(name.startswith("kinera.kema") for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
