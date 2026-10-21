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
    CandidateAuditRecord,
    ScientificComparison,
    ScientificVariantEvaluation,
    calculate_replay_utility,
    calculate_retention_ratio,
    calculate_stability_plasticity_tradeoff,
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
        new_initial = evaluate_retention_probe(
            model, new_probe_group, (new_probe_ref,), (new_example,), vocabulary
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
        new_acquired = evaluate_retention_probe(
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

        replay_budget = 0 if variant == "A" else 1

        if variant == "A":
            selected: tuple[ReplayCandidate, ...] = ()
            used_seed = None
        elif variant == "B":
            selected = select_random_replay(candidates, replay_budget, replay_seed=replay_seed)
            used_seed = replay_seed
        elif variant == "C":
            selected = select_directed_replay(candidates, replay_budget)
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
        net_stability = tuple(
            oriented_competence_delta(ref.competence, post.competence, ref.direction)
            for ref, post in zip(old_reference, old_post)
        )
        plasticity_acquisition = oriented_competence_delta(
            new_initial.competence, new_acquired.competence, new_acquired.direction
        )
        new_retention_during_replay = oriented_competence_delta(
            new_acquired.competence, new_post.competence, new_acquired.direction
        )

        accounting = ReplayAccounting(
            new_updates=1,
            replay_updates=len(selected_refs),
            total_updates=1 + len(selected_refs),
            selected_refs=selected_refs,
            replay_seed=used_seed,
        )

        mean_recovery = sum(old_recovery) / len(old_recovery) if old_recovery else 0.0
        mean_net_stability = sum(net_stability) / len(net_stability) if net_stability else 0.0

        if accounting.replay_updates > 0:
            mean_utility = calculate_replay_utility(mean_recovery, accounting.replay_updates)
        else:
            mean_utility = None

        retention_ratios = tuple(
            calculate_retention_ratio(ref.competence, post.competence, ref.direction)
            for ref, post in zip(old_reference, old_post)
        )
        mean_retention = sum(retention_ratios) / len(retention_ratios) if retention_ratios else 0.0
        tradeoff = calculate_stability_plasticity_tradeoff(mean_retention, plasticity_acquisition)

        # Audit candidate records
        audit_pool: list[CandidateAuditRecord] = []
        for cand in candidates:
            is_sel = cand.example_ref in selected_refs
            reason = None
            if is_sel:
                reason = "uniform_random_sample" if variant == "B" else "max_forgetting_score"
            audit_pool.append(
                CandidateAuditRecord(
                    example_ref=cand.example_ref,
                    group_ids=cand.group_ids,
                    forgetting_score=cand.priority_score,
                    selected=is_sel,
                    selection_reason=reason,
                )
            )
        audit_selected = tuple(c for c in audit_pool if c.selected)

        return ScientificVariantEvaluation(
            variant=variant,
            old_competence_reference=old_ref_comp,
            old_competence_pre=old_pre_comp,
            old_competence_post=old_post_comp,
            old_forgetting_pre=old_forgetting_pre,
            old_forgetting_post=old_forgetting_post,
            old_recovery_delta=old_recovery,
            net_stability_delta=net_stability,
            new_competence_initial=new_initial.competence,
            new_competence_acquired=new_acquired.competence,
            new_competence_post=new_post.competence,
            plasticity_acquisition=plasticity_acquisition,
            new_retention_during_replay=new_retention_during_replay,
            new_competence_reference=new_acquired.competence,
            new_competence_delta=new_retention_during_replay,
            accounting=accounting,
            replay_budget=replay_budget,
            mean_recovery_delta=mean_recovery,
            mean_net_stability_delta=mean_net_stability,
            mean_replay_utility=mean_utility,
            mean_retention_ratio=mean_retention,
            stability_plasticity_tradeoff=tradeoff,
            candidate_pool=tuple(audit_pool),
            selected_candidates=audit_selected,
        )
    finally:
        model.close()


def _run_natural_protocol(variant: str, replay_seed: int = 1) -> ScientificVariantEvaluation:
    """Natural Portuguese fixture testing genuine retention and differential candidate selection.

    Trains initial competence on two distinct words ('sapo' and 'bola'), establishing high baseline
    competence (1.0) on both. An interfering task ('pato') degrades 'bola' while leaving 'sapo' stable.
    Memory stores both candidates with distinct forgetting scores (0.0 vs 1.0), enabling directed replay
    (C) to target the degraded skill while random replay (B) samples uniformly.
    """
    records = [
        annotate_record({"word_key": "sapo", "word_normalized": "sapo", "syllables": ["sa", "po"]}),
        annotate_record({"word_key": "bola", "word_normalized": "bola", "syllables": ["bo", "la"]}),
        annotate_record({"word_key": "pato", "word_normalized": "pato", "syllables": ["pa", "to"]}),
    ]
    char_vocabulary = CharacterVocabulary.from_records(records)
    syllable_vocabulary = SyllableVocabulary.from_records(records)

    all_sapo = list(build_autoregressive_examples([records[0]], char_vocabulary, context_size=2, syllable_vocabulary=syllable_vocabulary))
    all_bola = list(build_autoregressive_examples([records[1]], char_vocabulary, context_size=2, syllable_vocabulary=syllable_vocabulary))
    all_pato = list(build_autoregressive_examples([records[2]], char_vocabulary, context_size=2, syllable_vocabulary=syllable_vocabulary))

    # Second character examples provide non-overlapping contexts ['<START>', 's'] vs ['<START>', 'b']
    ex_sapo = all_sapo[1]
    ex_bola = all_bola[1]
    ex_interf = all_pato[1]

    ref_sapo = ExampleRef("fixture:l4-natural", "old-train", 0)
    ref_bola = ExampleRef("fixture:l4-natural", "old-train", 1)
    ref_interf = ExampleRef("fixture:l4-natural", "new-train", 0)

    group_sapo = RetentionProbeGroup("probe-sapo", (ref_sapo,), "char_accuracy", "higher_is_better")
    group_bola = RetentionProbeGroup("probe-bola", (ref_bola,), "char_accuracy", "higher_is_better")
    group_interf = RetentionProbeGroup("probe-interf", (ref_interf,), "char_accuracy", "higher_is_better")

    model = AutoregressiveModel(
        len(char_vocabulary), len(syllable_vocabulary), char_embedding_dim=8, syllable_embedding_dim=4, seed=42
    )
    memory = LearningMemory(4)
    try:
        def observer(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            ref = ref_sapo if index == 0 else ref_bola
            group = group_sapo.group_id if index == 0 else group_bola.group_id
            memory.observe(
                LearningEvent(
                    ref, epoch, step, "pre_update",
                    report["char_loss"], report["syllable_loss"],
                    report["boundary_loss"], report["total_loss"],
                    (group,),
                )
            )

        # 1. Acquire initial competence on both sapo and bola
        train_multitask_examples(
            model, [ex_sapo, ex_bola], char_vocabulary, epochs=30, learning_rate=0.08,
            step_observer=observer,
        )
        old_ref_sapo = evaluate_retention_probe(
            model, group_sapo, (ref_sapo,), (ex_sapo,), char_vocabulary
        )
        old_ref_bola = evaluate_retention_probe(
            model, group_bola, (ref_bola,), (ex_bola,), char_vocabulary
        )
        new_initial = evaluate_retention_probe(
            model, group_interf, (ref_interf,), (ex_interf,), char_vocabulary
        )

        # 2. Interfering learning task
        train_multitask_examples(
            model, [ex_interf], char_vocabulary, epochs=25, learning_rate=0.1
        )
        old_pre_sapo = evaluate_retention_probe(
            model, group_sapo, (ref_sapo,), (ex_sapo,), char_vocabulary
        )
        old_pre_bola = evaluate_retention_probe(
            model, group_bola, (ref_bola,), (ex_bola,), char_vocabulary
        )
        new_acquired = evaluate_retention_probe(
            model, group_interf, (ref_interf,), (ex_interf,), char_vocabulary
        )

        # 3. Assess state and candidate pool
        state_sapo = build_learning_state(old_ref_sapo, old_pre_sapo, status="LEARNING", last_practiced_step=55)
        state_bola = build_learning_state(old_ref_bola, old_pre_bola, status="LEARNING", last_practiced_step=55)
        state_by_group = {state_sapo.group_id: state_sapo, state_bola.group_id: state_bola}
        candidates = build_replay_candidates(memory, state_by_group)

        replay_budget = 0 if variant == "A" else 1

        if variant == "A":
            selected = ()
            used_seed = None
        elif variant == "B":
            selected = select_random_replay(candidates, replay_budget, replay_seed=replay_seed)
            used_seed = replay_seed
        elif variant == "C":
            selected = select_directed_replay(candidates, replay_budget)
            used_seed = None
        else:
            raise ValueError(f"unknown variant {variant!r}")

        selected_refs = tuple(candidate.example_ref for candidate in selected)
        plan = build_interleaved_plan(
            (ref_interf,), selected_refs,
            replay_positions=tuple(range(1, 1 + len(selected_refs))),
        )
        all_refs = (ref_sapo, ref_bola, ref_interf)
        all_examples = (ex_sapo, ex_bola, ex_interf)
        replay_examples = resolve_replay_examples(plan, all_refs, all_examples)

        # 4. Execute replay
        train_multitask_examples(
            model, list(replay_examples), char_vocabulary,
            epochs=1, learning_rate=0.1,
        )

        old_post_sapo = evaluate_retention_probe(
            model, group_sapo, (ref_sapo,), (ex_sapo,), char_vocabulary
        )
        old_post_bola = evaluate_retention_probe(
            model, group_bola, (ref_bola,), (ex_bola,), char_vocabulary
        )
        new_post = evaluate_retention_probe(
            model, group_interf, (ref_interf,), (ex_interf,), char_vocabulary
        )

        old_reference = (old_ref_sapo, old_ref_bola)
        old_pre = (old_pre_sapo, old_pre_bola)
        old_post = (old_post_sapo, old_post_bola)

        old_ref_comp = (old_ref_sapo.competence, old_ref_bola.competence)
        old_pre_comp = (old_pre_sapo.competence, old_pre_bola.competence)
        old_post_comp = (old_post_sapo.competence, old_post_bola.competence)
        old_forgetting_pre = (state_sapo.forgetting_score or 0.0, state_bola.forgetting_score or 0.0)
        old_forgetting_post = (
            calculate_forgetting(old_ref_sapo.competence, old_post_sapo.competence, old_ref_sapo.direction) or 0.0,
            calculate_forgetting(old_ref_bola.competence, old_post_bola.competence, old_ref_bola.direction) or 0.0,
        )
        old_recovery = (
            oriented_competence_delta(old_pre_sapo.competence, old_post_sapo.competence, old_pre_sapo.direction),
            oriented_competence_delta(old_pre_bola.competence, old_post_bola.competence, old_pre_bola.direction),
        )
        net_stability = (
            oriented_competence_delta(old_ref_sapo.competence, old_post_sapo.competence, old_ref_sapo.direction),
            oriented_competence_delta(old_ref_bola.competence, old_post_bola.competence, old_ref_bola.direction),
        )
        plasticity_acquisition = oriented_competence_delta(
            new_initial.competence, new_acquired.competence, new_acquired.direction
        )
        new_retention_during_replay = oriented_competence_delta(
            new_acquired.competence, new_post.competence, new_acquired.direction
        )

        accounting = ReplayAccounting(
            new_updates=1,
            replay_updates=len(selected_refs),
            total_updates=1 + len(selected_refs),
            selected_refs=selected_refs,
            replay_seed=used_seed,
        )

        mean_recovery = sum(old_recovery) / len(old_recovery)
        mean_net_stability = sum(net_stability) / len(net_stability)

        if accounting.replay_updates > 0:
            mean_utility = calculate_replay_utility(mean_recovery, accounting.replay_updates)
        else:
            mean_utility = None

        retention_ratios = tuple(
            calculate_retention_ratio(ref.competence, post.competence, ref.direction)
            for ref, post in zip(old_reference, old_post)
        )
        mean_retention = sum(retention_ratios) / len(retention_ratios)
        tradeoff = calculate_stability_plasticity_tradeoff(mean_retention, plasticity_acquisition)

        # Audit candidate records
        audit_pool: list[CandidateAuditRecord] = []
        for cand in candidates:
            is_sel = cand.example_ref in selected_refs
            reason = None
            if is_sel:
                reason = "uniform_random_sample" if variant == "B" else "max_forgetting_score"
            audit_pool.append(
                CandidateAuditRecord(
                    example_ref=cand.example_ref,
                    group_ids=cand.group_ids,
                    forgetting_score=cand.priority_score,
                    selected=is_sel,
                    selection_reason=reason,
                )
            )
        audit_selected = tuple(c for c in audit_pool if c.selected)

        return ScientificVariantEvaluation(
            variant=variant,
            old_competence_reference=old_ref_comp,
            old_competence_pre=old_pre_comp,
            old_competence_post=old_post_comp,
            old_forgetting_pre=old_forgetting_pre,
            old_forgetting_post=old_forgetting_post,
            old_recovery_delta=old_recovery,
            net_stability_delta=net_stability,
            new_competence_initial=new_initial.competence,
            new_competence_acquired=new_acquired.competence,
            new_competence_post=new_post.competence,
            plasticity_acquisition=plasticity_acquisition,
            new_retention_during_replay=new_retention_during_replay,
            new_competence_reference=new_acquired.competence,
            new_competence_delta=new_retention_during_replay,
            accounting=accounting,
            replay_budget=replay_budget,
            mean_recovery_delta=mean_recovery,
            mean_net_stability_delta=mean_net_stability,
            mean_replay_utility=mean_utility,
            mean_retention_ratio=mean_retention,
            stability_plasticity_tradeoff=tradeoff,
            candidate_pool=tuple(audit_pool),
            selected_candidates=audit_selected,
        )
    finally:
        model.close()


def run_scientific_protocol(
    fixture_type: FixtureType,
    replay_seed: int = 1,
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
