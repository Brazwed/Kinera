#!/usr/bin/env python3
"""Run Phase 45 Economical Pilot - Replay LR Calibration (Issue #45).

Evaluates replay learning rate calibration across [0.15, 0.16, 0.17, 0.18, 0.20]
under condition 'high' for seeds [42, 43, 44] with strict budget parity (4 steps),
continuous probability tracking (P(target)), log-loss per task, and explicit
retention denominators.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from kinera.backends.cpu.ops.probability import cross_entropy, softmax
from kinera.experiments.expanded_pilot import (
    _build_unified_vocabularies,
    _WORD_SYLLABLES,
    AutoregressiveModel,
    compute_probe_partition_breakdown,
    get_condition_spec,
)
from kinera.experiments.learning_memory import ExampleRef, LearningEvent, LearningMemory
from kinera.experiments.learning_state import build_learning_state
from kinera.experiments.replay import (
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
from kinera.experiments.scientific_metrics import calculate_retention_ratio
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record
from kinera.training.evaluator import example_to_inputs
from kinera.training.trainer import train_multitask_examples


def evaluate_probe_continuous(
    model: AutoregressiveModel,
    example: dict[str, Any],
    char_vocab: Any,
) -> dict[str, Any]:
    """Evaluate continuous prediction metrics (accuracy, P(target), log_loss)."""
    char_ids, syl_ids, feat_vecs = example_to_inputs(
        example, char_vocab, syllable_unknown_id=0
    )
    char_logits = model.forward(char_ids, syl_ids, feat_vecs)
    target_id = example["target_char_id"]
    probs = softmax(char_logits)
    target_prob = float(probs[target_id])
    loss = float(cross_entropy(char_logits, target_id))
    pred_id = max(range(len(char_logits)), key=char_logits.__getitem__)
    acc = 1.0 if pred_id == target_id else 0.0
    return {
        "accuracy": acc,
        "target_prob": target_prob,
        "log_loss": loss,
        "predicted_token": char_vocab.id_to_token[pred_id],
        "target_token": char_vocab.id_to_token[target_id],
        "predicted_prob": float(probs[pred_id]),
    }


def run_single_condition_seed(
    condition: str,
    seed: int,
    variant: str,
    replay_lr: float,
) -> dict[str, Any]:
    """Run one experimental unit with strict budget tracking and continuous logging."""
    char_vocab, syl_vocab = _build_unified_vocabularies()
    spec = get_condition_spec(condition)  # type: ignore[arg-type]

    old_train_records = [
        annotate_record(
            {"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}
        )
        for w in spec.old_words
    ]
    old_internal_records = [
        annotate_record(
            {"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}
        )
        for w in spec.old_internal_words
    ]
    old_holdout_records = [
        annotate_record(
            {"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}
        )
        for w in spec.old_holdout_words
    ]
    new_train_records = [
        annotate_record(
            {"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}
        )
        for w in spec.new_words
    ]
    new_holdout_records = [
        annotate_record(
            {"word_key": w, "word_normalized": w, "syllables": _WORD_SYLLABLES[w]}
        )
        for w in spec.new_holdout_words
    ]

    ex_old_train = [
        build_autoregressive_examples(
            [r], char_vocab, context_size=2, syllable_vocabulary=syl_vocab
        )[1]
        for r in old_train_records
    ]
    ex_old_internal = [
        build_autoregressive_examples(
            [r], char_vocab, context_size=2, syllable_vocabulary=syl_vocab
        )[1]
        for r in old_internal_records
    ]
    ex_old_holdout = [
        build_autoregressive_examples(
            [r], char_vocab, context_size=2, syllable_vocabulary=syl_vocab
        )[1]
        for r in old_holdout_records
    ]
    ex_new_train = [
        build_autoregressive_examples(
            [r], char_vocab, context_size=2, syllable_vocabulary=syl_vocab
        )[1]
        for r in new_train_records
    ]
    ex_new_holdout = [
        build_autoregressive_examples(
            [r], char_vocab, context_size=2, syllable_vocabulary=syl_vocab
        )[1]
        for r in new_holdout_records
    ]

    old_refs = tuple(
        ExampleRef("dataset:pilot-old-train", "train", i)
        for i in range(len(old_train_records))
    )
    internal_probe_refs = tuple(
        ExampleRef("dataset:pilot-old-internal", "internal_probe", i)
        for i in range(len(old_internal_records))
    )
    holdout_probe_refs = tuple(
        ExampleRef("dataset:pilot-old-holdout", "holdout_test", i)
        for i in range(len(old_holdout_records))
    )
    new_refs = tuple(
        ExampleRef(f"dataset:pilot-new-train-{condition}", "train", i)
        for i in range(len(new_train_records))
    )
    new_holdout_refs = tuple(
        ExampleRef(f"dataset:pilot-new-holdout-{condition}", "holdout_test", i)
        for i in range(len(new_holdout_records))
    )

    internal_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-int-{f.internal_probe_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for f, ref in zip(spec.old_families, internal_probe_refs)
    )
    holdout_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-hold-{f.holdout_probe_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for f, ref in zip(spec.old_families, holdout_probe_refs)
    )
    new_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-new-{condition}-{t.train_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for t, ref in zip(spec.new_tasks, new_refs)
    )
    new_holdout_groups = tuple(
        RetentionProbeGroup(
            f"probe-new-hold-{condition}-{t.holdout_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for t, ref in zip(spec.new_tasks, new_holdout_refs)
    )

    model = AutoregressiveModel(
        len(char_vocab),
        len(syl_vocab),
        char_embedding_dim=8,
        syllable_embedding_dim=4,
        seed=seed,
    )
    memory = LearningMemory(8)

    try:
        def observer(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            memory.observe(
                LearningEvent(
                    old_refs[index],
                    epoch,
                    step,
                    "pre_update",
                    report["char_loss"],
                    report["syllable_loss"],
                    report["boundary_loss"],
                    report["total_loss"],
                    (internal_probe_groups[index].group_id,),
                )
            )

        # Phase 1: Old task acquisition
        train_multitask_examples(
            model,
            ex_old_train,
            char_vocab,
            epochs=spec.old_epochs,
            learning_rate=spec.old_learning_rate,
            step_observer=observer,
        )

        old_ref_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(internal_probe_groups, internal_probe_refs, ex_old_internal)
        ]
        old_holdout_ref_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(holdout_probe_groups, holdout_probe_refs, ex_old_holdout)
        ]
        new_init_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
        ]

        task_metrics_ref: dict[str, Any] = {}
        for w, ex in zip(spec.old_internal_words, ex_old_internal):
            task_metrics_ref[f"old_internal_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.old_holdout_words, ex_old_holdout):
            task_metrics_ref[f"old_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_words, ex_new_train):
            task_metrics_ref[f"new_train_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_holdout_words, ex_new_holdout):
            task_metrics_ref[f"new_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)

        # Phase 2: Interfering learning
        train_multitask_examples(
            model,
            ex_new_train,
            char_vocab,
            epochs=spec.new_epochs,
            learning_rate=spec.new_learning_rate,
        )

        old_pre_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(internal_probe_groups, internal_probe_refs, ex_old_internal)
        ]
        old_holdout_pre_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(holdout_probe_groups, holdout_probe_refs, ex_old_holdout)
        ]
        new_acq_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
        ]

        task_metrics_pre: dict[str, Any] = {}
        for w, ex in zip(spec.old_internal_words, ex_old_internal):
            task_metrics_pre[f"old_internal_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.old_holdout_words, ex_old_holdout):
            task_metrics_pre[f"old_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_words, ex_new_train):
            task_metrics_pre[f"new_train_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_holdout_words, ex_new_holdout):
            task_metrics_pre[f"new_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)

        # Phase 3: State assessment & candidate selection
        states = tuple(
            build_learning_state(
                ProbeObservation(g.group_id, (r,), "char_accuracy", "higher_is_better", ref_c, 1),
                ProbeObservation(g.group_id, (r,), "char_accuracy", "higher_is_better", pre_c, 1),
                status="LEARNING",
                last_practiced_step=spec.old_epochs + spec.new_epochs,
            )
            for g, r, ref_c, pre_c in zip(internal_probe_groups, internal_probe_refs, old_ref_comp, old_pre_comp)
        )
        state_by_group = {s.group_id: s for s in states}
        candidates = build_replay_candidates(memory, state_by_group)

        if variant == "B":
            selected = select_random_replay(candidates, 2, replay_seed=seed)
        elif variant == "C":
            selected = select_directed_replay(candidates, 2)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        sel_refs = tuple(c.example_ref for c in selected)
        plan = build_interleaved_plan(new_refs, sel_refs, replay_positions=(2, 3))
        all_refs = (*old_refs, *new_refs)
        all_examples = (*ex_old_train, *ex_new_train)
        replay_ex = resolve_replay_examples(plan, all_refs, all_examples)

        # Phase 4: Replay consolidation with physical step tracking
        actual_new_updates = 0
        actual_replay_updates = 0

        def step_tracker(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
            nonlocal actual_new_updates, actual_replay_updates
            if plan[index].source == "replay":
                actual_replay_updates += 1
            else:
                actual_new_updates += 1

        train_multitask_examples(
            model,
            list(replay_ex),
            char_vocab,
            epochs=spec.replay_epochs,
            learning_rate=replay_lr,
            step_observer=step_tracker,
        )

        assert actual_new_updates == 2, f"Expected 2 new updates, got {actual_new_updates}"
        assert actual_replay_updates == 2, f"Expected 2 replay updates, got {actual_replay_updates}"
        assert actual_new_updates + actual_replay_updates == 4, f"Expected 4 updates, got {actual_new_updates + actual_replay_updates}"

        # Phase 5: Post-replay evaluation
        old_post_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(internal_probe_groups, internal_probe_refs, ex_old_internal)
        ]
        old_holdout_post_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(holdout_probe_groups, holdout_probe_refs, ex_old_holdout)
        ]
        new_post_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
        ]
        new_holdout_post_comp = [
            evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
            for g, r, e in zip(new_holdout_groups, new_holdout_refs, ex_new_holdout)
        ]

        task_metrics_post: dict[str, Any] = {}
        for w, ex in zip(spec.old_internal_words, ex_old_internal):
            task_metrics_post[f"old_internal_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.old_holdout_words, ex_old_holdout):
            task_metrics_post[f"old_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_words, ex_new_train):
            task_metrics_post[f"new_train_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)
        for w, ex in zip(spec.new_holdout_words, ex_new_holdout):
            task_metrics_post[f"new_holdout_{w}"] = evaluate_probe_continuous(model, ex, char_vocab)

        # Standard and partitioned retention metrics
        internal_breakdown = compute_probe_partition_breakdown(
            tuple(g.group_id for g in internal_probe_groups), old_ref_comp, old_post_comp
        )
        holdout_breakdown = compute_probe_partition_breakdown(
            tuple(g.group_id for g in holdout_probe_groups), old_holdout_ref_comp, old_holdout_post_comp
        )

        # Explicit denominative definitions:
        r_pos_internal = internal_breakdown.mean_positive_ref_retention
        r_pos_internal_pos_count = len(internal_breakdown.positive_ref_group_ids)
        r_pos_internal_total_count = len(internal_probe_groups)

        r_pos_holdout = holdout_breakdown.mean_positive_ref_retention
        r_pos_holdout_pos_count = len(holdout_breakdown.positive_ref_group_ids)
        r_pos_holdout_total_count = len(holdout_probe_groups)

        # Phase 42 original overall retention across all 4 internal probes:
        mean_old_ref = sum(old_ref_comp) / float(len(old_ref_comp))
        mean_old_post = sum(old_post_comp) / float(len(old_post_comp))
        overall_retention_ratio = (
            sum(calculate_retention_ratio(r, p) for r, p in zip(old_ref_comp, old_post_comp))
            / float(len(old_ref_comp))
        )

        mean_new_acq = sum(new_acq_comp) / float(len(new_acq_comp))
        mean_new_post = sum(new_post_comp) / float(len(new_post_comp))
        delta_new = mean_new_post - mean_new_acq

        # Task-specific probabilities and log-loss
        p_medo_post = task_metrics_post["new_train_medo"]["target_prob"]
        p_medo_acq = task_metrics_pre["new_train_medo"]["target_prob"]
        p_povo_post = task_metrics_post["new_train_povo"]["target_prob"]
        p_povo_acq = task_metrics_pre["new_train_povo"]["target_prob"]

        p_malas_post = task_metrics_post["old_internal_malas"]["target_prob"]
        p_patos_post = task_metrics_post["old_internal_patos"]["target_prob"]
        p_sapos_post = task_metrics_post["old_internal_sapos"]["target_prob"]
        p_fogos_post = task_metrics_post["old_internal_fogos"]["target_prob"]

        loss_new_post = (
            task_metrics_post["new_train_medo"]["log_loss"]
            + task_metrics_post["new_train_povo"]["log_loss"]
        ) / 2.0
        loss_new_acq = (
            task_metrics_pre["new_train_medo"]["log_loss"]
            + task_metrics_pre["new_train_povo"]["log_loss"]
        ) / 2.0

        loss_old_post = (
            task_metrics_post["old_internal_malas"]["log_loss"]
            + task_metrics_post["old_internal_patos"]["log_loss"]
            + task_metrics_post["old_internal_sapos"]["log_loss"]
            + task_metrics_post["old_internal_fogos"]["log_loss"]
        ) / 4.0

        criteria_met = bool(r_pos_internal >= 0.70 and delta_new >= -0.25)

        return {
            "seed": seed,
            "condition": condition,
            "variant": variant,
            "replay_lr": replay_lr,
            "selected_refs": [str(r) for r in sel_refs],
            "actual_new_updates": actual_new_updates,
            "actual_replay_updates": actual_replay_updates,
            "total_updates": actual_new_updates + actual_replay_updates,
            "r_pos_internal": r_pos_internal,
            "r_pos_internal_numerator": sum(internal_breakdown.positive_ref_retention_ratios),
            "r_pos_internal_denominator": r_pos_internal_pos_count,
            "r_pos_internal_total_probes": r_pos_internal_total_count,
            "r_pos_holdout": r_pos_holdout,
            "r_pos_holdout_denominator": r_pos_holdout_pos_count,
            "overall_retention_ratio": overall_retention_ratio,
            "mean_old_ref_comp": mean_old_ref,
            "mean_old_pre_comp": sum(old_pre_comp) / float(len(old_pre_comp)),
            "mean_old_post_comp": mean_old_post,
            "c_new_init": sum(new_init_comp) / float(len(new_init_comp)),
            "c_new_acq": mean_new_acq,
            "c_new_post": mean_new_post,
            "delta_new": delta_new,
            "p_target_new_acq": (p_medo_acq + p_povo_acq) / 2.0,
            "p_target_new_post": (p_medo_post + p_povo_post) / 2.0,
            "delta_p_target_new": ((p_medo_post + p_povo_post) / 2.0) - ((p_medo_acq + p_povo_acq) / 2.0),
            "p_target_old_post": (p_malas_post + p_patos_post + p_sapos_post + p_fogos_post) / 4.0,
            "loss_new_acq": loss_new_acq,
            "loss_new_post": loss_new_post,
            "delta_loss_new": loss_new_post - loss_new_acq,
            "loss_old_post": loss_old_post,
            "criteria_met": criteria_met,
            "probes": {
                "old_ref_comp": old_ref_comp,
                "old_pre_comp": old_pre_comp,
                "old_post_comp": old_post_comp,
                "new_acq_comp": new_acq_comp,
                "new_post_comp": new_post_comp,
            },
            "task_metrics_ref": task_metrics_ref,
            "task_metrics_pre": task_metrics_pre,
            "task_metrics_post": task_metrics_post,
        }
    finally:
        model.close()


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate runs across seeds."""
    n = len(runs)
    if n == 0:
        return {}

    def stats(key: str) -> dict[str, float]:
        vals = [float(r[key]) for r in runs]
        m = sum(vals) / float(n)
        v = sum((x - m) ** 2 for x in vals) / float(n)
        return {
            "mean": m,
            "std": math.sqrt(v),
            "min": min(vals),
            "max": max(vals),
        }

    agg = {
        "r_pos_internal": stats("r_pos_internal"),
        "r_pos_holdout": stats("r_pos_holdout"),
        "overall_retention_ratio": stats("overall_retention_ratio"),
        "c_new_acq": stats("c_new_acq"),
        "c_new_post": stats("c_new_post"),
        "delta_new": stats("delta_new"),
        "p_target_new_acq": stats("p_target_new_acq"),
        "p_target_new_post": stats("p_target_new_post"),
        "delta_p_target_new": stats("delta_p_target_new"),
        "p_target_old_post": stats("p_target_old_post"),
        "loss_new_acq": stats("loss_new_acq"),
        "loss_new_post": stats("loss_new_post"),
        "delta_loss_new": stats("delta_loss_new"),
        "loss_old_post": stats("loss_old_post"),
        "all_seeds_criteria_met": all(r["criteria_met"] for r in runs),
        "mean_criteria_met": bool(
            stats("r_pos_internal")["mean"] >= 0.70
            and stats("delta_new")["mean"] >= -0.25
        ),
        "budget_verified_all_seeds": all(
            r["actual_new_updates"] == 2
            and r["actual_replay_updates"] == 2
            and r["total_updates"] == 4
            for r in runs
        ),
    }
    return agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 45 Economical Pilot - Replay LR Calibration"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds to evaluate",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.15, 0.16, 0.17, 0.18, 0.20],
        help="Replay learning rates to calibrate for C",
    )
    parser.add_argument(
        "--condition",
        choices=["low", "medium", "high"],
        default="high",
        help="Interference condition (default: high)",
    )
    parser.add_argument(
        "--output",
        default="experiments/phase45_pilot_results.json",
        help="Output JSON file path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    output_path = (
        Path(args.output)
        if Path(args.output).is_absolute()
        else (project_root / args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    print("=" * 80)
    print("Kinera Phase 45: Economical Pilot — Replay LR Calibration")
    print(f"Condition:        {args.condition}")
    print(f"Seeds:            {args.seeds}")
    print(f"Replay LRs:       {args.learning_rates}")
    print(f"Git Commit:       {git_commit[:10]}")
    print(f"Output File:      {output_path}")
    print("=" * 80)

    t0 = time.perf_counter()

    # Formulate configurations: B_ref (0.20), C_baseline (0.20), and C_lr for each lr
    configs = [("B_ref", "B", 0.20)]
    for lr in args.learning_rates:
        tag = f"C_{lr:.2f}"
        configs.append((tag, "C", lr))

    results_by_config: dict[str, Any] = {}
    for name, variant, lr in configs:
        print(f"\nRunning {name} (variant={variant}, lr={lr:.2f})...")
        runs = []
        for s in args.seeds:
            res = run_single_condition_seed(args.condition, s, variant, lr)
            runs.append(res)
            print(
                f"  Seed {s:<3} | R_pos={res['r_pos_internal']:.4f} "
                f"({res['r_pos_internal_numerator']:.1f}/{res['r_pos_internal_denominator']}) | "
                f"Delta_new={res['delta_new']:+.4f} | "
                f"P(new)={res['p_target_new_post']:.4f} | "
                f"P(old)={res['p_target_old_post']:.4f} | "
                f"Gate={'PASS' if res['criteria_met'] else 'FAIL'}"
            )
        agg = aggregate_runs(runs)
        results_by_config[name] = {
            "variant": variant,
            "replay_lr": lr,
            "seeds": args.seeds,
            "runs": runs,
            "aggregates": agg,
        }

    total_time = time.perf_counter() - t0

    report = {
        "metadata": {
            "experiment": "Phase 45 Economical Pilot - Replay LR Calibration",
            "condition": args.condition,
            "seeds": args.seeds,
            "learning_rates": args.learning_rates,
            "commit": git_commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_execution_seconds": total_time,
            "decision_criteria": {
                "r_pos_target": 0.70,
                "delta_new_target": -0.25,
                "rule": "Both criteria must be met simultaneously to consider expansion",
            },
        },
        "configurations": results_by_config,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport successfully saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 115)
    print(
        f"{'Configuration':<16} | {'LR':<4} | {'Mean R_pos':<10} | {'Mean Delta_new':<14} | "
        f"{'Mean C_new_post':<15} | {'Mean P(new)':<11} | {'Mean Loss(new)':<14} | {'Mean Gate'}"
    )
    print("-" * 115)
    for name, cdata in results_by_config.items():
        agg = cdata["aggregates"]
        lr = cdata["replay_lr"]
        rpos_m = agg["r_pos_internal"]["mean"]
        rpos_s = agg["r_pos_internal"]["std"]
        dnew_m = agg["delta_new"]["mean"]
        dnew_s = agg["delta_new"]["std"]
        cpost_m = agg["c_new_post"]["mean"]
        pnew_m = agg["p_target_new_post"]["mean"]
        lnew_m = agg["loss_new_post"]["mean"]
        gate = "PASS" if agg["mean_criteria_met"] else "FAIL"
        print(
            f"{name:<16} | {lr:<4.2f} | {rpos_m:.4f}±{rpos_s:.2f} | "
            f"{dnew_m:+.4f}±{dnew_s:.2f}    | {cpost_m:.4f}          | "
            f"{pnew_m:.4f}      | {lnew_m:.4f}         | {gate}"
        )
    print("=" * 115)

    return 0


if __name__ == "__main__":
    sys.exit(main())
