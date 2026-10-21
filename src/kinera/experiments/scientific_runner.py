"""Controlled experimental runner for Phase 34 (Tranche L4).

Executes variants A, B, and C across synthetic and natural Kinera fixtures,
evaluates learning retention/forgetting, and generates a structured scientific report.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Literal

from kinera.experiments.learning_memory import (
    ExampleRef,
    LearningEvent,
    LearningMemory,
    LearningMemoryRecorder,
)
from kinera.experiments.learning_state import (
    build_learning_state,
    calculate_forgetting,
    oriented_competence_delta,
)
from kinera.experiments.replay import (
    ReplayAccounting,
    ReplayCandidate,
    build_interleaved_plan,
    build_replay_candidates,
    resolve_replay_examples,
    select_directed_replay,
    select_random_replay,
)
from kinera.experiments.retention import (
    RetentionProbeGroup,
    evaluate_retention_probe,
)
from kinera.experiments.scientific_metrics import (
    ScientificComparison,
    ScientificVariantEvaluation,
    calculate_replay_utility,
    compare_scientific_variants,
)
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.trainer import train_multitask_examples

FixtureType = Literal["synthetic", "natural"]


def _make_example(
    context_ids: list[int],
    feature_index: int,
    target: int,
) -> dict[str, Any]:
    features = [0.0] * 10
    if feature_index >= 0:
        features[feature_index] = 1.0
    return {
        "context_ids": list(context_ids),
        "context_syllable_ids": [0, 1],
        "context_feature_vectors": [features.copy(), features.copy()],
        "target_char_id": target,
        "target_syllable_id": 1,
        "target_is_syllable_end": 1,
    }


def _run_synthetic_protocol(variant: str, replay_seed: int = 0) -> ScientificVariantEvaluation:
    vocabulary = CharacterVocabulary(["<START>", "<END>", "<UNK>", "a", "b", "c"])
    old_refs = (
        ExampleRef("fixture:l4-synthetic", "old-train", 0),
        ExampleRef("fixture:l4-synthetic", "old-train", 1),
    )
    old_groups = ("g1-interfered", "g2-stable")
    old_train = (
        _make_example([0, 0], -1, 3),
        _make_example([1, 2], 1, 4),
    )
    old_probe_refs = (
        ExampleRef("fixture:l4-synthetic", "probe-g1", 0),
        ExampleRef("fixture:l4-synthetic", "probe-g2", 0),
    )
    old_probe_groups = tuple(
        RetentionProbeGroup(group, (ref,), "char_accuracy", "higher_is_better")
        for group, ref in zip(old_groups, old_probe_refs)
    )
    old_probe_examples = (old_train[0], old_train[1])
    new_ref = ExampleRef("fixture:l4-synthetic", "new-train", 0)
    new_probe_ref = ExampleRef("fixture:l4-synthetic", "probe-new", 0)
    new_example = _make_example([0, 0], -1, 5)
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
        def observer(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            memory.observe(
                LearningEvent(
                    old_refs[index], epoch, step, "pre_update",
                    report["char_loss"], report["syllable_loss"],
                    report["boundary_loss"], report["total_loss"],
                    (old_groups[index],),
                )
            )

        train_multitask_examples(
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

        train_multitask_examples(
            model, [new_example], vocabulary,
            epochs=interference_epochs, learning_rate=learning_rate,
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

        if variant == "A":
            selected: tuple[ReplayCandidate, ...] = ()
            used_seed = None
        elif variant == "B":
            selected = select_random_replay(candidates, 1, replay_seed=replay_seed)
            used_seed = replay_seed
        elif variant == "C":
            selected = select_directed_replay(candidates, 1)
            used_seed = None
        else:
            raise ValueError(f"unknown variant {variant!r}")

        selected_refs = tuple(candidate.example_ref for candidate in selected)
        plan = build_interleaved_plan(
            (new_ref,), selected_refs,
            replay_positions=tuple(range(1, 1 + len(selected_refs))),
        )
        all_refs = (*old_refs, new_ref)
        all_examples = (*old_train, new_example)
        replay_examples = resolve_replay_examples(plan, all_refs, all_examples)

        train_multitask_examples(
            model, list(replay_examples), vocabulary,
            epochs=1, learning_rate=learning_rate,
        )

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

        old_ref_comp = tuple(obs.competence for obs in old_reference)
        old_pre_comp = tuple(obs.competence for obs in old_pre)
        old_post_comp = tuple(obs.competence for obs in old_post)
        old_forgetting_pre = tuple(state.forgetting_score or 0.0 for state in states)
        old_forgetting_post = tuple(
            calculate_forgetting(ref.competence, post.competence, ref.direction) or 0.0
            for ref, post in zip(old_reference, old_post)
        )
        old_recovery = tuple(
            oriented_competence_delta(pre.competence, post.competence, pre.direction)
            for pre, post in zip(old_pre, old_post)
        )
        new_delta = oriented_competence_delta(
            new_reference.competence, new_post.competence, new_reference.direction
        )

        accounting = ReplayAccounting(
            new_updates=1,
            replay_updates=len(selected_refs),
            total_updates=1 + len(selected_refs),
            selected_refs=selected_refs,
            replay_seed=used_seed,
        )

        mean_recovery = sum(old_recovery) / len(old_recovery) if old_recovery else 0.0
        if accounting.replay_updates > 0:
            mean_utility = calculate_replay_utility(mean_recovery, accounting.replay_updates)
        else:
            mean_utility = None

        return ScientificVariantEvaluation(
            variant=variant,
            old_competence_reference=old_ref_comp,
            old_competence_pre=old_pre_comp,
            old_competence_post=old_post_comp,
            old_forgetting_pre=old_forgetting_pre,
            old_forgetting_post=old_forgetting_post,
            old_recovery_delta=old_recovery,
            new_competence_reference=new_reference.competence,
            new_competence_post=new_post.competence,
            new_competence_delta=new_delta,
            accounting=accounting,
            mean_recovery_delta=mean_recovery,
            mean_replay_utility=mean_utility,
        )
    finally:
        model.close()


def _run_natural_protocol(variant: str, replay_seed: int = 0) -> ScientificVariantEvaluation:
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

    old_ref = ExampleRef("fixture:l4-natural:casa-pato-capa", "old-train", 0)
    new_ref = ExampleRef("fixture:l4-natural:casa-pato-capa", "new-train", 0)
    old_probe_ref = ExampleRef("fixture:l4-natural:casa-pato-capa", "probe-old", 0)
    new_probe_ref = ExampleRef("fixture:l4-natural:casa-pato-capa", "probe-new", 0)

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
        train_multitask_examples(
            model, [old_train], char_vocabulary, epochs=2, learning_rate=0.03,
            step_observer=LearningMemoryRecorder(
                (old_ref,), memory, group_ids=(old_group.group_id,)
            ),
        )
        old_reference = evaluate_retention_probe(
            model, old_group, (old_probe_ref,), (old_probe_example,), char_vocabulary
        )

        train_multitask_examples(
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

        if variant == "A":
            selected = ()
            used_seed = None
        elif variant == "B":
            selected = select_random_replay(candidates, 1, replay_seed=replay_seed)
            used_seed = replay_seed
        elif variant == "C":
            selected = select_directed_replay(candidates, 1)
            used_seed = None
        else:
            raise ValueError(f"unknown variant {variant!r}")

        selected_refs = tuple(candidate.example_ref for candidate in selected)
        plan = build_interleaved_plan(
            (new_ref,), selected_refs,
            replay_positions=tuple(range(1, 1 + len(selected_refs))),
        )
        all_refs = (old_ref, new_ref)
        all_examples = (old_train, new_train)
        replay_examples = resolve_replay_examples(plan, all_refs, all_examples)

        train_multitask_examples(
            model, list(replay_examples), char_vocabulary,
            epochs=1, learning_rate=0.03,
        )

        old_post = evaluate_retention_probe(
            model, old_group, (old_probe_ref,), (old_probe_example,), char_vocabulary
        )
        new_post = evaluate_retention_probe(
            model, new_group, (new_probe_ref,), (new_probe_example,), char_vocabulary
        )

        old_ref_comp = (old_reference.competence,)
        old_pre_comp = (old_pre.competence,)
        old_post_comp = (old_post.competence,)
        old_forgetting_pre = (state.forgetting_score or 0.0,)
        old_forgetting_post = (
            calculate_forgetting(
                old_reference.competence, old_post.competence, old_reference.direction
            ) or 0.0,
        )
        old_recovery = (
            oriented_competence_delta(
                old_pre.competence, old_post.competence, old_pre.direction
            ),
        )
        new_delta = oriented_competence_delta(
            new_reference.competence, new_post.competence, new_reference.direction
        )

        accounting = ReplayAccounting(
            new_updates=1,
            replay_updates=len(selected_refs),
            total_updates=1 + len(selected_refs),
            selected_refs=selected_refs,
            replay_seed=used_seed,
        )

        mean_recovery = old_recovery[0]
        if accounting.replay_updates > 0:
            mean_utility = calculate_replay_utility(mean_recovery, accounting.replay_updates)
        else:
            mean_utility = None

        return ScientificVariantEvaluation(
            variant=variant,
            old_competence_reference=old_ref_comp,
            old_competence_pre=old_pre_comp,
            old_competence_post=old_post_comp,
            old_forgetting_pre=old_forgetting_pre,
            old_forgetting_post=old_forgetting_post,
            old_recovery_delta=old_recovery,
            new_competence_reference=new_reference.competence,
            new_competence_post=new_post.competence,
            new_competence_delta=new_delta,
            accounting=accounting,
            mean_recovery_delta=mean_recovery,
            mean_replay_utility=mean_utility,
        )
    finally:
        model.close()


def run_scientific_protocol(
    fixture_type: FixtureType,
    replay_seed: int = 0,
) -> dict[str, Any]:
    """Run all 3 variants (A, B, C) and compile the scientific evaluation report."""
    if fixture_type == "synthetic":
        runner_fn = _run_synthetic_protocol
    elif fixture_type == "natural":
        runner_fn = _run_natural_protocol
    else:
        raise ValueError(f"unknown fixture_type: {fixture_type!r}")

    eval_a = runner_fn("A", replay_seed=replay_seed)
    eval_b = runner_fn("B", replay_seed=replay_seed)
    eval_c = runner_fn("C", replay_seed=replay_seed)

    comparison = compare_scientific_variants(eval_a, eval_b, eval_c)

    return {
        "fixture_type": fixture_type,
        "replay_seed": replay_seed,
        "evaluations": {
            "A": eval_a.to_dict(),
            "B": eval_b.to_dict(),
            "C": eval_c.to_dict(),
        },
        "comparison": comparison.to_dict(),
    }


__all__ = [
    "FixtureType",
    "run_scientific_protocol",
]
