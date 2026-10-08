from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import os
import random
import subprocess
import sys
import unittest
from typing import Any

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
from kinera.experiments.replay import (
    ReplayAccounting,
    ReplayCandidate,
    ReplayPlanItem,
    build_interleaved_plan,
    build_replay_candidates,
    resolve_replay_examples,
    select_directed_replay,
    select_random_replay,
)
from kinera.experiments.retention import (
    ProbeObservation,
    RetentionProbeGroup,
    evaluate_retention_probe,
)
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.trainer import train_multitask_examples


def _event(ref: ExampleRef, groups: tuple[str, ...]) -> LearningEvent:
    return LearningEvent(ref, 1, 1, "pre_update", 1.0, 0.5, 0.25, 1.75, groups)


def _state(group_id: str, forgetting: float | None) -> LearningState:
    return LearningState(
        group_id,
        0.8,
        0.8 - (forgetting or 0.0),
        forgetting,
        2,
        0,
        2,
        "LEARNING",
        -(forgetting or 0.0),
    )


def _candidate(index: int, score: float, *groups: str) -> ReplayCandidate:
    return ReplayCandidate(
        ExampleRef("fixture:replay", "old-train", index), tuple(groups), score
    )


class ReplayCandidateTests(unittest.TestCase):
    def test_candidate_priority_is_max_forgetting_across_explicit_groups(self) -> None:
        memory = LearningMemory(4)
        refs = (ExampleRef("fixture:replay", "old-train", 0),
                ExampleRef("fixture:replay", "old-train", 1))
        memory.observe(_event(refs[0], ("g1", "g2")))
        memory.observe(_event(refs[1], ("g2",)))
        states = {"g1": _state("g1", 0.2), "g2": _state("g2", 0.7)}
        before = memory.entries()

        candidates = build_replay_candidates(memory, states)

        self.assertEqual(
            candidates,
            (ReplayCandidate(refs[0], ("g1", "g2"), 0.7),
             ReplayCandidate(refs[1], ("g2",), 0.7)),
        )
        self.assertEqual(memory.entries(), before)

    def test_unassociated_or_unmeasurable_memory_entries_fail_explicitly(self) -> None:
        ref = ExampleRef("fixture:replay", "old-train", 0)
        memory = LearningMemory(2)
        memory.observe(_event(ref, ()))
        with self.assertRaisesRegex(ValueError, "group association"):
            build_replay_candidates(memory, {})

        memory = LearningMemory(2)
        memory.observe(_event(ref, ("missing",)))
        with self.assertRaisesRegex(KeyError, "missing"):
            build_replay_candidates(memory, {})

        with self.assertRaisesRegex(ValueError, "forgetting_score"):
            build_replay_candidates(memory, {"missing": _state("missing", None)})

    def test_candidate_contract_rejects_invalid_association_and_priority(self) -> None:
        ref = ExampleRef("fixture:replay", "old-train", 0)
        invalid = (
            ("not-a-ref", ("g",), 0.1),
            (ref, (), 0.1),
            (ref, ("",), 0.1),
            (ref, ("g", "g"), 0.1),
            (ref, ("g",), math.nan),
            (ref, ("g",), math.inf),
            (ref, ("g",), -0.1),
            (ref, ("g",), True),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ReplayCandidate(*args)  # type: ignore[arg-type]

    def test_candidate_pool_duplicate_refs_are_rejected(self) -> None:
        candidate = _candidate(0, 0.2, "g")
        with self.assertRaisesRegex(ValueError, "unique"):
            select_directed_replay((candidate, candidate), 1)


class ReplaySelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = (
            _candidate(0, 0.2, "g1"),
            _candidate(1, 0.9, "g2"),
            _candidate(2, 0.9, "g3"),
        )

    def test_random_selection_is_seeded_local_without_replacement_and_score_blind(self) -> None:
        state_before = random.getstate()
        first = select_random_replay(self.pool, 2, replay_seed=41)
        second = select_random_replay(self.pool, 2, replay_seed=41)
        score_changed = tuple(
            ReplayCandidate(item.example_ref, item.group_ids, 100.0 - item.priority_score)
            for item in self.pool
        )
        third = select_random_replay(score_changed, 2, replay_seed=41)

        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.example_ref for item in first),
            tuple(item.example_ref for item in third),
        )
        self.assertEqual(len(first), 2)
        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(first, tuple(item for item in self.pool if item in first))

    def test_directed_selects_largest_score_ties_follow_pool_order(self) -> None:
        state_before = random.getstate()
        selected = select_directed_replay(self.pool, 2)
        self.assertEqual(
            tuple(item.example_ref.example_index for item in selected), (1, 2)
        )
        self.assertEqual(
            tuple(item.example_ref.example_index for item in select_directed_replay(self.pool, 1)),
            (1,),
        )
        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(select_directed_replay(self.pool, 0), ())
        self.assertEqual(select_directed_replay(self.pool, len(self.pool)), self.pool)

    def test_budget_validation_for_both_selectors(self) -> None:
        for budget in (-1, True, 1.5, len(self.pool) + 1):
            with self.subTest(budget=budget), self.assertRaises((TypeError, ValueError)):
                select_random_replay(self.pool, budget, replay_seed=1)  # type: ignore[arg-type]
            with self.subTest(selector="directed", budget=budget), self.assertRaises((TypeError, ValueError)):
                select_directed_replay(self.pool, budget)  # type: ignore[arg-type]
        for budget in (0, 1, len(self.pool)):
            self.assertEqual(len(select_random_replay(self.pool, budget, replay_seed=2)), budget)
            self.assertEqual(len(select_directed_replay(self.pool, budget)), budget)
        for seed in (True, 1.2, "seed"):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                select_random_replay(self.pool, 1, replay_seed=seed)  # type: ignore[arg-type]


class ReplayPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.new_refs = tuple(ExampleRef("fixture:new", "train", i) for i in range(4))
        self.replay_refs = tuple(ExampleRef("fixture:old", "train", i) for i in range(2))

    def test_plan_interleaves_final_zero_based_positions_and_preserves_both_orders(self) -> None:
        plan = build_interleaved_plan(
            self.new_refs, self.replay_refs, replay_positions=(1, 4)
        )
        self.assertEqual(
            plan,
            (
                ReplayPlanItem("new", self.new_refs[0]),
                ReplayPlanItem("replay", self.replay_refs[0]),
                ReplayPlanItem("new", self.new_refs[1]),
                ReplayPlanItem("new", self.new_refs[2]),
                ReplayPlanItem("replay", self.replay_refs[1]),
                ReplayPlanItem("new", self.new_refs[3]),
            ),
        )
        self.assertEqual(
            tuple(item.example_ref for item in plan if item.source == "new"),
            self.new_refs,
        )
        self.assertEqual(
            tuple(item.example_ref for item in plan if item.source == "replay"),
            self.replay_refs,
        )
        no_replay = build_interleaved_plan(self.new_refs, (), replay_positions=())
        self.assertEqual(tuple(item.example_ref for item in no_replay), self.new_refs)

    def test_plan_rejects_invalid_positions_and_duplicate_input_refs(self) -> None:
        for positions in ((1,), (1, 1), (-1, 4), (1, 6), (True, 4), (1.0, 4)):
            with self.subTest(positions=positions), self.assertRaises((TypeError, ValueError)):
                build_interleaved_plan(
                    self.new_refs, self.replay_refs, replay_positions=positions
                )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_interleaved_plan(
                (self.new_refs[0], self.new_refs[0]), (), replay_positions=()
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_interleaved_plan(
                self.new_refs[:1], (self.replay_refs[0], self.replay_refs[0]),
                replay_positions=(0, 1),
            )

    def test_resolution_is_plan_ordered_and_rejects_missing_duplicate_or_unaligned_refs(self) -> None:
        plan = build_interleaved_plan(
            self.new_refs[:1], self.replay_refs[:1], replay_positions=(1,)
        )
        aligned = {
            self.new_refs[0]: {"value": "new"},
            self.replay_refs[0]: {"value": "old"},
        }
        self.assertEqual(
            resolve_replay_examples(
                plan,
                tuple(aligned),
                tuple(aligned.values()),
            ),
            ({"value": "new"}, {"value": "old"}),
        )
        duplicate_plan = (
            ReplayPlanItem("new", self.new_refs[0]),
            ReplayPlanItem("replay", self.new_refs[0]),
        )
        with self.assertRaisesRegex(ValueError, "plan.*unique"):
            resolve_replay_examples(
                duplicate_plan,
                (self.new_refs[0],),
                ({"value": "same-example"},),
            )
        with self.assertRaisesRegex(KeyError, "missing"):
            resolve_replay_examples(plan, (self.new_refs[0],), ({"value": "new"},))
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_replay_examples(
                plan,
                (self.new_refs[0], self.new_refs[0]),
                ({"value": "new"}, {"value": "duplicate"}),
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            resolve_replay_examples(plan, (self.new_refs[0],), ())


class ReplayAccountingTests(unittest.TestCase):
    def test_accounting_is_immutable_and_consistent(self) -> None:
        refs = (ExampleRef("fixture:old", "train", 0),)
        accounting = ReplayAccounting(4, 1, 5, refs, 12)
        self.assertEqual(accounting.total_updates, accounting.new_updates + accounting.replay_updates)
        self.assertEqual(accounting.selected_refs, refs)
        with self.assertRaises((AttributeError, TypeError)):
            accounting.total_updates = 6  # type: ignore[misc]
        invalid = (
            (True, 0, 1, (), None),
            (1, -1, 0, (), None),
            (1, 1, 3, refs, None),
            (1, 1, 2, (), None),
            (1, 0, 1, refs, None),
            (1, 1, 2, refs, True),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ReplayAccounting(*args)  # type: ignore[arg-type]


def _example(context_ids: list[int], feature_index: int, target: int) -> dict[str, Any]:
    features = [0.0] * 10
    if feature_index >= 0:
        features[feature_index] = 1.0
    return {
        "context_ids": context_ids,
        "context_syllable_ids": [0, 1],
        "context_feature_vectors": [features.copy(), features.copy()],
        "target_char_id": target,
        "target_syllable_id": 1,
        "target_is_syllable_end": 1,
    }


def _parameters(model: AutoregressiveModel) -> tuple[Any, ...]:
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


@dataclass(frozen=True)
class _ProtocolReport:
    prefix_snapshot: tuple[Any, ...]
    old_reference: tuple[ProbeObservation, ...]
    old_pre: tuple[ProbeObservation, ...]
    old_post: tuple[ProbeObservation, ...]
    new_reference: ProbeObservation
    new_post: ProbeObservation
    states: tuple[LearningState, ...]
    candidates: tuple[ReplayCandidate, ...]
    selected_refs: tuple[ExampleRef, ...]
    plan: tuple[ReplayPlanItem, ...]
    accounting: ReplayAccounting
    actual_update_sources: tuple[str, ...]
    old_forgetting_after: tuple[float, ...]
    old_recovery_delta: tuple[float, ...]
    new_competence_delta: float
    source_memory_after: tuple[Any, ...]
    final_parameters: tuple[Any, ...]
    post_reports: tuple[dict[str, Any], ...]


def _run_synthetic_variant(variant: str) -> _ProtocolReport:
    """Mechanistic CPU fixture: G1 conflicts with B; G2 uses a separate input."""
    vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b", "c"])
    old_refs = (
        ExampleRef("fixture:l3-synthetic", "old-train", 0),
        ExampleRef("fixture:l3-synthetic", "old-train", 1),
    )
    old_groups = ("g1-interfered", "g2-stable")
    old_train = (
        _example([0, 0], -1, 3),
        _example([1, 2], 1, 4),
    )
    old_probe_refs = (
        ExampleRef("fixture:l3-synthetic", "probe-g1", 0),
        ExampleRef("fixture:l3-synthetic", "probe-g2", 0),
    )
    old_probe_groups = tuple(
        RetentionProbeGroup(group, (ref,), "char_accuracy", "higher_is_better")
        for group, ref in zip(old_groups, old_probe_refs)
    )
    old_probe_examples = (old_train[0], old_train[1])
    new_ref = ExampleRef("fixture:l3-synthetic", "new-train", 0)
    new_probe_ref = ExampleRef("fixture:l3-synthetic", "probe-new", 0)
    new_example = _example([0, 0], -1, 5)
    new_probe_group = RetentionProbeGroup(
        "new-task", (new_probe_ref,), "char_accuracy", "higher_is_better"
    )
    learning_rate = 0.1
    interference_epochs = 35
    model = AutoregressiveModel(
        len(vocabulary), 3, char_embedding_dim=2, syllable_embedding_dim=2, seed=19
    )
    memory = LearningMemory(8)
    try:
        common_update_trace: list[tuple[ExampleRef, int, int]] = []

        def observer(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            common_update_trace.append((old_refs[index], epoch, step))
            memory.observe(
                LearningEvent(
                    old_refs[index], epoch, step, "pre_update",
                    report["char_loss"], report["syllable_loss"],
                    report["boundary_loss"], report["total_loss"],
                    (old_groups[index],),
                )
            )

        old_history = train_multitask_examples(
            model, list(old_train), vocabulary,
            epochs=interference_epochs, learning_rate=learning_rate,
            step_observer=observer,
        )
        old_reference = tuple(
            evaluate_retention_probe(
                model, group, (ref,), (example,), vocabulary
            )
            for group, ref, example in zip(
                old_probe_groups, old_probe_refs, old_probe_examples
            )
        )
        def new_observer(epoch: int, index: int, step: int, _report: dict[str, float]) -> None:
            if index != 0:
                raise AssertionError("interference fixture unexpectedly changed example order")
            common_update_trace.append((new_ref, epoch, step))

        new_history = train_multitask_examples(
            model, [new_example], vocabulary,
            epochs=interference_epochs, learning_rate=learning_rate,
            step_observer=new_observer,
        )
        old_pre = tuple(
            evaluate_retention_probe(
                model, group, (ref,), (example,), vocabulary
            )
            for group, ref, example in zip(
                old_probe_groups, old_probe_refs, old_probe_examples
            )
        )
        new_reference = evaluate_retention_probe(
            model, new_probe_group, (new_probe_ref,), (new_example,), vocabulary
        )
        states = tuple(
            build_learning_state(
                reference, current, status="LEARNING",
                last_practiced_step=interference_epochs * 3,
            )
            for reference, current in zip(old_reference, old_pre)
        )
        state_by_group = {state.group_id: state for state in states}
        candidates = build_replay_candidates(memory, state_by_group)
        memory_snapshot = memory.entries()
        prefix_snapshot = (
            _parameters(model), old_history, new_history, old_reference, old_pre,
            new_reference, states, candidates, memory_snapshot,
            tuple(common_update_trace),
        )

        if variant == "A":
            selected: tuple[ReplayCandidate, ...] = ()
        elif variant == "B":
            selected = select_random_replay(candidates, 1, replay_seed=0)
        elif variant == "C":
            selected = select_directed_replay(candidates, 1)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        if _parameters(model) != prefix_snapshot[0]:
            raise AssertionError("replay selection mutated model parameters")
        selected_refs = tuple(candidate.example_ref for candidate in selected)
        plan = build_interleaved_plan(
            (new_ref,), selected_refs,
            replay_positions=tuple(range(1, 1 + len(selected_refs))),
        )
        all_refs = (*old_refs, new_ref)
        all_examples = (*old_train, new_example)
        replay_examples = resolve_replay_examples(plan, all_refs, all_examples)
        source_before_replay = memory.entries()
        actual_update_sources: list[str] = []

        def replay_observer(
            _epoch: int, index: int, _step: int, _report: dict[str, float]
        ) -> None:
            actual_update_sources.append(plan[index].source)

        post_result = train_multitask_examples(
            model, list(replay_examples), vocabulary,
            epochs=1, learning_rate=learning_rate,
            step_observer=replay_observer,
        )
        if tuple(actual_update_sources) != tuple(item.source for item in plan):
            raise AssertionError("completed trainer updates diverged from frozen replay plan")
        if memory.entries() != source_before_replay:
            raise AssertionError("replay mutated source LearningMemory")
        old_post = tuple(
            evaluate_retention_probe(
                model, group, (ref,), (example,), vocabulary
            )
            for group, ref, example in zip(
                old_probe_groups, old_probe_refs, old_probe_examples
            )
        )
        new_post = evaluate_retention_probe(
            model, new_probe_group, (new_probe_ref,), (new_example,), vocabulary
        )
        accounting = ReplayAccounting(
            actual_update_sources.count("new"),
            actual_update_sources.count("replay"),
            len(actual_update_sources),
            selected_refs,
            0 if variant == "B" else None,
        )
        old_forgetting_after = tuple(
            calculate_forgetting(reference.competence, post.competence, reference.direction)
            for reference, post in zip(old_reference, old_post)
        )
        old_recovery_delta = tuple(
            oriented_competence_delta(before.competence, post.competence, before.direction)
            for before, post in zip(old_pre, old_post)
        )
        new_competence_delta = oriented_competence_delta(
            new_reference.competence, new_post.competence, new_reference.direction
        )
        return _ProtocolReport(
            prefix_snapshot, old_reference, old_pre, old_post, new_reference,
            new_post, states, candidates, selected_refs, plan, accounting,
            tuple(actual_update_sources), old_forgetting_after, old_recovery_delta,
            new_competence_delta, memory.entries(), _parameters(model),
            tuple(post_result["history"]),
        )
    finally:
        model.close()


def _run_natural_smoke_variant(variant: str) -> _ProtocolReport:
    """Hand-authored Portuguese records exercise Kinera builders, not a corpus."""
    records = [
        annotate_record({"word_key": "casa", "word_normalized": "casa", "syllables": ["ca", "sa"]}),
        annotate_record({"word_key": "pato", "word_normalized": "pato", "syllables": ["pa", "to"]}),
        annotate_record({"word_key": "capa", "word_normalized": "capa", "syllables": ["ca", "pa"]}),
    ]
    char_vocabulary = CharacterVocabulary.from_records(records)
    syllable_vocabulary = SyllableVocabulary.from_records(records)

    def build_one(record: dict[str, Any]) -> dict[str, Any]:
        return next(iter(build_autoregressive_examples(
            [record], char_vocabulary, context_size=2,
            syllable_vocabulary=syllable_vocabulary,
        )))

    old_train = build_one(records[0])
    new_train = build_one(records[1])
    old_probe_example = build_one(records[2])
    new_probe_example = build_one(records[1])
    old_ref = ExampleRef("fixture:l3-natural:casa-pato-capa", "old-train", 0)
    new_ref = ExampleRef("fixture:l3-natural:casa-pato-capa", "new-train", 0)
    old_probe_ref = ExampleRef("fixture:l3-natural:casa-pato-capa", "probe-old", 0)
    new_probe_ref = ExampleRef("fixture:l3-natural:casa-pato-capa", "probe-new", 0)
    old_group = RetentionProbeGroup(
        "old-casa-smoke", (old_probe_ref,), "char_accuracy", "higher_is_better"
    )
    new_group = RetentionProbeGroup(
        "new-pato-smoke", (new_probe_ref,), "char_accuracy", "higher_is_better"
    )
    model = AutoregressiveModel(
        len(char_vocabulary), len(syllable_vocabulary), seed=23
    )
    memory = LearningMemory(4)
    try:
        old_history = train_multitask_examples(
            model, [old_train], char_vocabulary, epochs=2, learning_rate=0.03,
            step_observer=LearningMemoryRecorder(
                (old_ref,), memory, group_ids=(old_group.group_id,)
            ),
        )
        old_reference = evaluate_retention_probe(
            model, old_group, (old_probe_ref,), (old_probe_example,), char_vocabulary
        )
        new_history = train_multitask_examples(
            model, [new_train], char_vocabulary, epochs=2, learning_rate=0.03
        )
        old_pre = evaluate_retention_probe(
            model, old_group, (old_probe_ref,), (old_probe_example,), char_vocabulary
        )
        new_reference = evaluate_retention_probe(
            model, new_group, (new_probe_ref,), (new_probe_example,), char_vocabulary
        )
        state = build_learning_state(
            old_reference, old_pre, status="LEARNING", last_practiced_step=4
        )
        candidates = build_replay_candidates(memory, {state.group_id: state})
        memory_snapshot = memory.entries()
        prefix_snapshot = (
            _parameters(model), old_history, new_history, old_reference, old_pre,
            new_reference, (state,), candidates, memory_snapshot,
        )
        if variant == "A":
            selected: tuple[ReplayCandidate, ...] = ()
        elif variant == "B":
            selected = select_random_replay(candidates, 1, replay_seed=0)
        elif variant == "C":
            selected = select_directed_replay(candidates, 1)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        if _parameters(model) != prefix_snapshot[0]:
            raise AssertionError("replay selection mutated model parameters")
        selected_refs = tuple(candidate.example_ref for candidate in selected)
        plan = build_interleaved_plan(
            (new_ref,), selected_refs,
            replay_positions=tuple(range(1, 1 + len(selected_refs))),
        )
        replay_examples = resolve_replay_examples(
            plan, (old_ref, new_ref), (old_train, new_train)
        )
        update_sources: list[str] = []
        source_before = memory.entries()

        def observer(_epoch: int, index: int, _step: int, _report: dict[str, float]) -> None:
            update_sources.append(plan[index].source)

        post_report = train_multitask_examples(
            model, list(replay_examples), char_vocabulary, epochs=1,
            learning_rate=0.03, step_observer=observer,
        )
        if tuple(update_sources) != tuple(item.source for item in plan):
            raise AssertionError("natural smoke updates diverged from replay plan")
        if memory.entries() != source_before:
            raise AssertionError("natural smoke replay mutated source memory")
        old_post = evaluate_retention_probe(
            model, old_group, (old_probe_ref,), (old_probe_example,), char_vocabulary
        )
        new_post = evaluate_retention_probe(
            model, new_group, (new_probe_ref,), (new_probe_example,), char_vocabulary
        )
        accounting = ReplayAccounting(
            update_sources.count("new"), update_sources.count("replay"),
            len(update_sources), selected_refs, 0 if variant == "B" else None,
        )
        return _ProtocolReport(
            prefix_snapshot, (old_reference,), (old_pre,), (old_post,),
            new_reference, new_post, (state,), candidates, selected_refs, plan,
            accounting, tuple(update_sources),
            (calculate_forgetting(old_reference.competence, old_post.competence,
                                  old_reference.direction),),
            (oriented_competence_delta(old_pre.competence, old_post.competence,
                                       old_pre.direction),),
            oriented_competence_delta(new_reference.competence,
                                      new_post.competence, new_reference.direction),
            memory.entries(), _parameters(model), tuple(post_report["history"]),
        )
    finally:
        model.close()


class ControlledABCTests(unittest.TestCase):
    def test_synthetic_abc_is_frozen_fair_deterministic_and_mechanistic(self) -> None:
        first = {name: _run_synthetic_variant(name) for name in ("A", "B", "C")}
        second = {name: _run_synthetic_variant(name) for name in ("A", "B", "C")}
        self.assertEqual(first, second)

        a, b, c = (first[name] for name in ("A", "B", "C"))
        self.assertEqual(a.prefix_snapshot, b.prefix_snapshot)
        self.assertEqual(b.prefix_snapshot, c.prefix_snapshot)
        self.assertEqual(b.candidates, c.candidates)
        self.assertEqual(b.accounting.replay_updates, c.accounting.replay_updates)
        self.assertEqual(b.accounting.total_updates, c.accounting.total_updates)
        self.assertEqual(b.accounting.new_updates, c.accounting.new_updates)
        self.assertEqual((a.accounting.new_updates, a.accounting.replay_updates,
                          a.accounting.total_updates), (1, 0, 1))
        self.assertEqual((b.accounting.new_updates, b.accounting.replay_updates,
                          b.accounting.total_updates), (1, 1, 2))
        self.assertEqual((c.accounting.new_updates, c.accounting.replay_updates,
                          c.accounting.total_updates), (1, 1, 2))
        self.assertEqual(tuple(item.source for item in b.plan), ("new", "replay"))
        self.assertEqual(tuple(item.source for item in c.plan), ("new", "replay"))
        self.assertEqual(
            tuple(item.example_ref for item in b.plan if item.source == "new"),
            tuple(item.example_ref for item in c.plan if item.source == "new"),
        )
        self.assertEqual(
            tuple(index for index, item in enumerate(b.plan) if item.source == "replay"),
            tuple(index for index, item in enumerate(c.plan) if item.source == "replay"),
        )
        self.assertEqual(
            tuple((item.source, item.example_ref if item.source == "new" else None)
                  for item in b.plan),
            tuple((item.source, item.example_ref if item.source == "new" else None)
                  for item in c.plan),
        )
        self.assertEqual(b.selected_refs, tuple(item.example_ref for item in b.candidates if item.example_ref in b.selected_refs))
        self.assertEqual(c.selected_refs, tuple(item.example_ref for item in c.candidates if item.example_ref in c.selected_refs))
        self.assertNotEqual(b.selected_refs, c.selected_refs)
        self.assertEqual(c.selected_refs[0], c.candidates[0].example_ref)
        self.assertGreater(a.states[0].forgetting_score or 0.0,
                           a.states[1].forgetting_score or 0.0)
        self.assertEqual(a.source_memory_after, a.prefix_snapshot[-2])
        self.assertEqual(b.source_memory_after, b.prefix_snapshot[-2])
        self.assertEqual(c.source_memory_after, c.prefix_snapshot[-2])
        self.assertEqual(len(a.actual_update_sources), a.accounting.total_updates)
        self.assertEqual(len(b.actual_update_sources), b.accounting.total_updates)
        self.assertEqual(len(c.actual_update_sources), c.accounting.total_updates)
        self.assertEqual(a.actual_update_sources, ("new",))
        self.assertEqual(b.actual_update_sources, ("new", "replay"))
        self.assertEqual(c.actual_update_sources, ("new", "replay"))
        probe_refs = {obs.example_refs[0] for obs in (*b.old_reference, *b.old_pre, b.new_reference)}
        self.assertTrue(probe_refs.isdisjoint(item.example_ref for item in b.plan if item.source == "replay"))
        self.assertEqual(b.accounting.replay_seed, 0)
        self.assertIsNone(c.accounting.replay_seed)

        # Store raw outcomes and L2-derived deltas only; no winner claim is made.
        for report in (a, b, c):
            for reference, before, after, state in zip(
                report.old_reference, report.old_pre, report.old_post, report.states
            ):
                self.assertEqual(
                    calculate_forgetting(reference.competence, before.competence,
                                         reference.direction),
                    state.forgetting_score,
                )
                self.assertTrue(math.isfinite(
                    oriented_competence_delta(before.competence, after.competence,
                                              before.direction)
                ))
            self.assertTrue(math.isfinite(oriented_competence_delta(
                report.new_reference.competence, report.new_post.competence,
                report.new_reference.direction,
            )))



    def test_natural_hand_authored_kinera_records_smoke_all_variants_without_claim(self) -> None:
        reports = {
            name: _run_natural_smoke_variant(name) for name in ("A", "B", "C")
        }
        a, b, c = (reports[name] for name in ("A", "B", "C"))
        self.assertEqual(a.prefix_snapshot, b.prefix_snapshot)
        self.assertEqual(b.prefix_snapshot, c.prefix_snapshot)
        self.assertEqual(len(a.candidates), 1)
        self.assertEqual(b.selected_refs, c.selected_refs)
        self.assertEqual((a.accounting.new_updates, a.accounting.replay_updates,
                          a.accounting.total_updates), (1, 0, 1))
        self.assertEqual((b.accounting.new_updates, b.accounting.replay_updates,
                          b.accounting.total_updates), (1, 1, 2))
        self.assertEqual((c.accounting.new_updates, c.accounting.replay_updates,
                          c.accounting.total_updates), (1, 1, 2))
        for report in (a, b, c):
            self.assertEqual(len(report.old_reference), 1)
            self.assertEqual(report.old_reference[0].support, 1)
            self.assertEqual(report.old_post[0].support, 1)
            self.assertEqual(report.new_reference.support, 1)
            self.assertEqual(report.new_post.support, 1)
            self.assertEqual(report.source_memory_after, report.prefix_snapshot[-1])
            self.assertEqual(len(report.actual_update_sources), report.accounting.total_updates)
            self.assertTrue(math.isfinite(report.new_competence_delta))
        probe_refs = {report.old_reference[0].example_refs[0] for report in (a, b, c)}
        probe_refs.add(a.new_reference.example_refs[0])
        self.assertTrue(probe_refs.isdisjoint(item.example_ref for item in c.plan if item.source == "replay"))


class ReplayImportBoundaryTests(unittest.TestCase):
    def test_import_has_no_forbidden_dependencies_or_runtime_side_effects(self) -> None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        environment = {**os.environ, "PYTHONPATH": os.path.join(project_root, "src")}
        script = r"""
import builtins, random, sys, threading, time
random_state = random.getstate()
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith(("kinera.backends", "kinera.monitor", "kinera.training", "kinera.nn", "kinera.kema")):
        raise AssertionError("forbidden import: " + name)
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
original_random = random.Random
class ForbiddenRandom:
    def __init__(self, *args, **kwargs):
        raise AssertionError("RNG constructed during import")
random.Random = ForbiddenRandom
import kinera.experiments.replay
assert random.getstate() == random_state
assert not any(name.startswith(("kinera.backends", "kinera.monitor", "kinera.training", "kinera.nn", "kinera.kema")) for name in sys.modules)
random.Random = original_random
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project_root, env=environment,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
