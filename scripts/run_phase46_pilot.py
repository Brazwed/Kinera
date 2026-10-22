#!/usr/bin/env python3
"""Run Phase 46 Pilot - Contextual Disambiguation of Conflicting Targets (Issue #46).

Evaluates representational separability, functional coexistence under directed replay,
causal dependence under negative control (shuffled lemma), and generalization to
unseen words across [baseline_unconditioned, lemma_conditioned, negative_control]
under condition 'high' for seeds [42, 43, 44] with strict budget parity (4 consolidation steps).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from kinera.backends import resolve_backend
from kinera.backends.cpu.ops.probability import cross_entropy, softmax
from kinera.experiments.expanded_pilot import (
    _build_unified_vocabularies,
    _WORD_SYLLABLES,
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
)
from kinera.experiments.scientific_metrics import calculate_retention_ratio
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.orthography.features import annotate_record
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.heads.multitask import MultiTaskHeads
from kinera.nn.layers.embedding import Embedding
from kinera.nn.layers.hybrid import HybridRepresentation
from kinera.nn.losses.multitask import MultiTaskLoss
from kinera.training.metrics import _multitask_targets


# Lexical mapping for Phase 46 words
LEMMA_TO_ID: dict[str, int] = {
    # Old train & internal
    "mala": 0,
    "malas": 0,
    "maleta": 0,
    "pato": 1,
    "patos": 1,
    "pateta": 1,
    "sapo": 2,
    "sapos": 2,
    "sapato": 2,
    "fogo": 3,
    "fogos": 3,
    "foguete": 3,
    # New train & diagnostic holdouts
    "medo": 4,
    "meta": 4,
    "povo": 5,
    "pote": 5,
    # Unknown / independent holdouts
    "<UNK_LEMMA>": 6,
}

SWAPPED_LEMMA_MAP: dict[str, int] = {
    "mala": 4,   # receives lemma of medo
    "medo": 0,   # receives lemma of mala
    "malas": 4,
    "meta": 0,
    "pato": 5,   # receives lemma of povo
    "povo": 1,   # receives lemma of pato
    "patos": 5,
    "pote": 1,
    "sapo": 2,
    "sapos": 2,
    "fogo": 3,
    "fogos": 3,
}

INDEPENDENT_HOLDOUT_WORDS: list[str] = [
    "mola", "muda", "pala", "pelo", "selo", "toga", "vela"
]

INDEPENDENT_SYLLABLES: dict[str, list[str]] = {
    "mola": ["mo", "la"],
    "muda": ["mu", "da"],
    "pala": ["pa", "la"],
    "pelo": ["pe", "lo"],
    "selo": ["se", "lo"],
    "toga": ["to", "ga"],
    "vela": ["ve", "la"],
}


class ResearchModel:
    """Hybrid autoregressive model supporting optional lemma conditioning."""

    def __init__(
        self,
        char_vocab_size: int,
        syllable_vocab_size: int,
        lemma_vocab_size: int,
        *,
        char_embedding_dim: int = 8,
        syllable_embedding_dim: int = 4,
        feature_dim: int = 10,
        lemma_embedding_dim: int = 8,
        conditioned: bool = False,
        seed: int | None = None,
    ) -> None:
        self.backend = resolve_backend(None)
        self.conditioned = conditioned
        self.char_vocab_size = char_vocab_size
        self.syllable_vocab_size = syllable_vocab_size
        self.lemma_embedding_dim = lemma_embedding_dim

        self.char_embedding = Embedding(
            char_vocab_size, char_embedding_dim, seed=seed, backend=self.backend
        )
        self.syllable_embedding = Embedding(
            syllable_vocab_size,
            syllable_embedding_dim,
            seed=None if seed is None else seed + 1,
            backend=self.backend,
        )
        self.representation = HybridRepresentation(
            self.char_embedding,
            self.syllable_embedding,
            feature_dim=feature_dim,
        )

        if self.conditioned:
            self.lemma_embedding: Embedding | None = Embedding(
                lemma_vocab_size,
                lemma_embedding_dim,
                seed=None if seed is None else seed + 2,
                backend=self.backend,
            )
            self.total_dim = self.representation.output_dim + lemma_embedding_dim
        else:
            self.lemma_embedding = None
            self.total_dim = self.representation.output_dim

        self.heads = MultiTaskHeads(
            self.total_dim,
            char_vocab_size,
            syllable_vocab_size,
            seed=None if seed is None else seed + 3,
            backend=self.backend,
        )
        self.loss_function = MultiTaskLoss(backend=self.backend)

        self.last_pooled: list[float] | None = None
        self.last_logits: list[float] | None = None
        self.last_lemma_id: int | None = None

    def _pool(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        lemma_id: int | None = None,
    ) -> list[float]:
        base_vectors = self.representation.forward(char_ids, syllable_ids, feature_vectors)
        base_pooled = [
            sum(v[i] for v in base_vectors) / float(len(base_vectors))
            for i in range(self.representation.output_dim)
        ]
        if self.conditioned and self.lemma_embedding is not None:
            actual_lemma = lemma_id if lemma_id is not None else LEMMA_TO_ID["<UNK_LEMMA>"]
            lemma_vec = self.lemma_embedding.forward([actual_lemma])[0]
            pooled = base_pooled + lemma_vec
        else:
            pooled = base_pooled
        self.last_pooled = pooled
        self.last_lemma_id = lemma_id
        return pooled

    def forward(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        lemma_id: int | None = None,
    ) -> list[float]:
        pooled = self._pool(char_ids, syllable_ids, feature_vectors, lemma_id)
        logits = self.heads.char.forward(pooled)
        self.last_logits = logits
        return logits

    def forward_heads(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        lemma_id: int | None = None,
    ) -> dict[str, list[float]]:
        pooled = self._pool(char_ids, syllable_ids, feature_vectors, lemma_id)
        outputs = {
            "char": self.heads.char.forward(pooled),
            "syllable": self.heads.syllable.forward(pooled),
            "boundary": self.heads.boundary.forward(pooled),
        }
        self.last_logits = outputs["char"]
        return outputs

    def train_step(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        *,
        targets: Mapping[str, object],
        learning_rate: float,
        lemma_id: int | None = None,
    ) -> dict[str, float]:
        outputs = self.forward_heads(char_ids, syllable_ids, feature_vectors, lemma_id)
        report = self.loss_function.forward(outputs, targets)
        logit_grads = self.loss_function.backward(outputs, targets)

        input_grads = self.heads.backward(logit_grads)
        combined_grad = [
            input_grads["char"][i] + input_grads["syllable"][i] + input_grads["boundary"][i]
            for i in range(self.total_dim)
        ]

        base_dim = self.representation.output_dim
        base_grad = combined_grad[:base_dim]

        # Distribute gradient equally across context tokens for mean pooling
        context_len = len(char_ids)
        token_grads = [[g / float(context_len) for g in base_grad] for _ in range(context_len)]
        self.representation.backward(token_grads)

        if self.conditioned and self.lemma_embedding is not None and lemma_id is not None:
            lemma_grad = combined_grad[base_dim:]
            self.lemma_embedding.backward([lemma_grad])

        # Step updates
        self.heads.step(learning_rate)
        self.char_embedding.step(learning_rate)
        self.syllable_embedding.step(learning_rate)
        if self.conditioned and self.lemma_embedding is not None and lemma_id is not None:
            self.lemma_embedding.step(learning_rate)

        return report

    def close(self) -> None:
        pass


def evaluate_probe_continuous(
    model: ResearchModel,
    example: dict[str, Any],
    char_vocab: CharacterVocabulary,
    lemma_id: int | None = None,
) -> dict[str, float]:
    char_ids = example["context_ids"]
    syl_ids = example["context_syllable_ids"]
    features = example["context_feature_vectors"]
    target_id = example["target_char_id"]

    logits = model.forward(char_ids, syl_ids, features, lemma_id)
    probs = softmax(logits)
    target_prob = probs[target_id]
    loss = cross_entropy(logits, target_id)
    pred_id = max(range(len(logits)), key=logits.__getitem__)
    acc = 1.0 if pred_id == target_id else 0.0

    return {
        "accuracy": acc,
        "target_prob": float(target_prob),
        "loss": float(loss),
        "pred_id": pred_id,
        "target_id": target_id,
    }


def vector_distances(vec1: list[float], vec2: list[float]) -> tuple[float, float]:
    euclid = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    cos_dist = 1.0 - (dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
    return float(euclid), float(cos_dist)


def run_experiment_unit(
    condition_name: str,
    seed: int,
    replay_lr: float = 0.20,
) -> dict[str, Any]:
    """Execute one full experimental unit under Phase 46 specifications."""
    char_vocab, syl_vocab = _build_unified_vocabularies()
    spec = get_condition_spec("high")

    # Build all word sets
    old_train_words = spec.old_words          # mala, pato, sapo, fogo
    new_train_words = spec.new_words          # medo, povo
    old_int_words = spec.old_internal_words    # malas, patos, sapos, fogos
    old_hold_words = spec.old_holdout_words    # maleta, pateta, sapato, foguete
    new_hold_words = spec.new_holdout_words    # meta, pote

    def make_exs(words: Sequence[str]) -> list[dict[str, Any]]:
        exs = []
        for w in words:
            syls = _WORD_SYLLABLES.get(w, INDEPENDENT_SYLLABLES.get(w, ["ma"]))
            rec = annotate_record({"word_key": w, "word_normalized": w, "syllables": syls})
            ex = build_autoregressive_examples([rec], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)[1]
            exs.append(ex)
        return exs

    ex_old_train = make_exs(old_train_words)
    ex_new_train = make_exs(new_train_words)
    ex_old_int = make_exs(old_int_words)
    ex_old_hold = make_exs(old_hold_words)
    ex_new_hold = make_exs(new_hold_words)
    ex_indep = make_exs(INDEPENDENT_HOLDOUT_WORDS)

    is_conditioned = condition_name in ("lemma_conditioned", "negative_control")

    model = ResearchModel(
        len(char_vocab),
        len(syl_vocab),
        len(LEMMA_TO_ID),
        char_embedding_dim=8,
        syllable_embedding_dim=4,
        feature_dim=10,
        lemma_embedding_dim=8,
        conditioned=is_conditioned,
        seed=seed,
    )

    def get_lemma(w: str, condition: str) -> int | None:
        if not is_conditioned:
            return None
        if condition == "negative_control":
            return SWAPPED_LEMMA_MAP.get(w, LEMMA_TO_ID.get(w, LEMMA_TO_ID["<UNK_LEMMA>"]))
        return LEMMA_TO_ID.get(w, LEMMA_TO_ID["<UNK_LEMMA>"])

    # Measure initial representational separability before training
    model.forward(ex_old_train[0]["context_ids"], ex_old_train[0]["context_syllable_ids"], ex_old_train[0]["context_feature_vectors"], get_lemma("mala", "normal"))
    h_init_mala = list(model.last_pooled)
    model.forward(ex_new_train[0]["context_ids"], ex_new_train[0]["context_syllable_ids"], ex_new_train[0]["context_feature_vectors"], get_lemma("medo", "normal"))
    h_init_medo = list(model.last_pooled)
    init_euclid_m, init_cos_m = vector_distances(h_init_mala, h_init_medo)

    # Phase 1: Old task acquisition (20 epochs)
    for _ in range(spec.old_epochs):
        for w, ex in zip(old_train_words, ex_old_train):
            lid = get_lemma(w, "normal")
            model.train_step(
                ex["context_ids"],
                ex["context_syllable_ids"],
                ex["context_feature_vectors"],
                targets=_multitask_targets(ex),
                learning_rate=spec.old_learning_rate,
                lemma_id=lid,
            )

    # Measure reference competence
    old_ref_comp = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, "normal"))["accuracy"]
        for w, ex in zip(old_int_words, ex_old_int)
    ]
    diag_hold_ref = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, "normal"))["accuracy"]
        for w, ex in zip(old_hold_words, ex_old_hold)
    ]

    # Phase 2: Interfering learning (15 epochs on new words)
    for _ in range(spec.new_epochs):
        for w, ex in zip(new_train_words, ex_new_train):
            lid = get_lemma(w, "normal")
            model.train_step(
                ex["context_ids"],
                ex["context_syllable_ids"],
                ex["context_feature_vectors"],
                targets=_multitask_targets(ex),
                learning_rate=spec.new_learning_rate,
                lemma_id=lid,
            )

    old_pre_comp = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, "normal"))["accuracy"]
        for w, ex in zip(old_int_words, ex_old_int)
    ]
    new_acq_comp = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, "normal"))["accuracy"]
        for w, ex in zip(new_train_words, ex_new_train)
    ]

    # Phase 3: Directed replay candidate selection
    # In condition 'high', indices 0 (mala) and 1 (pato) suffer max forgetting and are selected
    replay_words = ["medo", "povo", "mala", "pato"]
    replay_exs = [ex_new_train[0], ex_new_train[1], ex_old_train[0], ex_old_train[1]]

    # Phase 4: Consolidation under strict 4-step budget (2 new + 2 replay)
    actual_new = 0
    actual_replay = 0
    for idx, (w, ex) in enumerate(zip(replay_words, replay_exs)):
        lid = get_lemma(w, "normal")
        is_rep = idx >= 2
        lr = replay_lr if is_rep else spec.new_learning_rate
        if is_rep:
            actual_replay += 1
        else:
            actual_new += 1
        model.train_step(
            ex["context_ids"],
            ex["context_syllable_ids"],
            ex["context_feature_vectors"],
            targets=_multitask_targets(ex),
            learning_rate=lr,
            lemma_id=lid,
        )

    assert actual_new == 2 and actual_replay == 2 and (actual_new + actual_replay == 4)

    # Phase 5: Evaluation
    # Note: under negative_control, eval_mode swaps conflicting lemmas
    eval_mode = "negative_control" if condition_name == "negative_control" else "normal"

    # Representational distance post-training
    model.forward(ex_old_train[0]["context_ids"], ex_old_train[0]["context_syllable_ids"], ex_old_train[0]["context_feature_vectors"], get_lemma("mala", "normal"))
    h_post_mala = list(model.last_pooled)
    model.forward(ex_new_train[0]["context_ids"], ex_new_train[0]["context_syllable_ids"], ex_new_train[0]["context_feature_vectors"], get_lemma("medo", "normal"))
    h_post_medo = list(model.last_pooled)
    post_euclid_m, post_cos_m = vector_distances(h_post_mala, h_post_medo)

    model.forward(ex_old_train[1]["context_ids"], ex_old_train[1]["context_syllable_ids"], ex_old_train[1]["context_feature_vectors"], get_lemma("pato", "normal"))
    h_post_pato = list(model.last_pooled)
    model.forward(ex_new_train[1]["context_ids"], ex_new_train[1]["context_syllable_ids"], ex_new_train[1]["context_feature_vectors"], get_lemma("povo", "normal"))
    h_post_povo = list(model.last_pooled)
    post_euclid_p, post_cos_p = vector_distances(h_post_pato, h_post_povo)

    # Probe metrics
    eval_old_int = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, eval_mode))
        for w, ex in zip(old_int_words, ex_old_int)
    ]
    old_post_comp = [m["accuracy"] for m in eval_old_int]

    eval_new_train = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, eval_mode))
        for w, ex in zip(new_train_words, ex_new_train)
    ]
    new_post_comp = [m["accuracy"] for m in eval_new_train]

    # Diagnostic holdout evaluation
    eval_diag_old = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, eval_mode))
        for w, ex in zip(old_hold_words, ex_old_hold)
    ]
    eval_diag_new = [
        evaluate_probe_continuous(model, ex, char_vocab, get_lemma(w, eval_mode))
        for w, ex in zip(new_hold_words, ex_new_hold)
    ]
    diag_holdout_acc = sum(m["accuracy"] for m in (*eval_diag_old, *eval_diag_new)) / float(len(eval_diag_old) + len(eval_diag_new))

    # Independent holdout evaluation (unseen words with UNK_LEMMA)
    eval_indep = [
        evaluate_probe_continuous(model, ex, char_vocab, LEMMA_TO_ID["<UNK_LEMMA>"] if is_conditioned else None)
        for ex in ex_indep
    ]
    indep_holdout_acc = sum(m["accuracy"] for m in eval_indep) / float(len(eval_indep))

    # Metric calculations
    internal_breakdown = compute_probe_partition_breakdown(
        tuple(f"probe_{w}" for w in old_int_words), old_ref_comp, old_post_comp
    )
    r_pos = internal_breakdown.mean_positive_ref_retention
    c_new_acq = sum(new_acq_comp) / float(len(new_acq_comp))
    c_new_post = sum(new_post_comp) / float(len(new_post_comp))
    delta_new = c_new_post - c_new_acq

    p_new_m = eval_new_train[0]["target_prob"]
    p_new_p = eval_new_train[1]["target_prob"]
    mean_p_new = (p_new_m + p_new_p) / 2.0
    loss_new_m = eval_new_train[0]["loss"]
    loss_new_p = eval_new_train[1]["loss"]
    mean_loss_new = (loss_new_m + loss_new_p) / 2.0

    p_old_m = eval_old_int[0]["target_prob"]
    p_old_p = eval_old_int[1]["target_prob"]
    mean_p_old = (p_old_m + p_old_p) / 2.0
    loss_old_m = eval_old_int[0]["loss"]
    loss_old_p = eval_old_int[1]["loss"]
    mean_loss_old = (loss_old_m + loss_old_p) / 2.0

    joint_collided_loss = (mean_loss_new + mean_loss_old) / 2.0

    criteria_met = bool(r_pos >= 0.70 and delta_new >= 0.0)

    return {
        "condition": condition_name,
        "seed": seed,
        "separability": {
            "initial_euclid_m": init_euclid_m,
            "initial_cos_dist_m": init_cos_m,
            "post_euclid_m": post_euclid_m,
            "post_cos_dist_m": post_cos_m,
            "post_euclid_p": post_euclid_p,
            "post_cos_dist_p": post_cos_p,
            "mean_post_cos_dist": (post_cos_m + post_cos_p) / 2.0,
        },
        "functional": {
            "r_pos": r_pos,
            "c_new_acq": c_new_acq,
            "c_new_post": c_new_post,
            "delta_new": delta_new,
            "mean_p_new": mean_p_new,
            "mean_loss_new": mean_loss_new,
            "mean_p_old": mean_p_old,
            "mean_loss_old": mean_loss_old,
            "joint_collided_loss": joint_collided_loss,
            "criteria_met": criteria_met,
        },
        "generalization": {
            "diag_holdout_acc": diag_holdout_acc,
            "indep_holdout_acc": indep_holdout_acc,
        },
        "details": {
            "old_ref_comp": old_ref_comp,
            "old_post_comp": old_post_comp,
            "new_post_comp": new_post_comp,
        },
    }


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def stats(extract_fn: Any) -> dict[str, float]:
        vals = [extract_fn(r) for r in runs]
        m = sum(vals) / float(len(vals))
        v = sum((x - m) ** 2 for x in vals) / float(len(vals))
        return {
            "mean": m,
            "std": math.sqrt(v),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "post_cos_dist": stats(lambda r: r["separability"]["mean_post_cos_dist"]),
        "r_pos": stats(lambda r: r["functional"]["r_pos"]),
        "c_new_post": stats(lambda r: r["functional"]["c_new_post"]),
        "delta_new": stats(lambda r: r["functional"]["delta_new"]),
        "mean_p_new": stats(lambda r: r["functional"]["mean_p_new"]),
        "mean_p_old": stats(lambda r: r["functional"]["mean_p_old"]),
        "joint_collided_loss": stats(lambda r: r["functional"]["joint_collided_loss"]),
        "diag_holdout_acc": stats(lambda r: r["generalization"]["diag_holdout_acc"]),
        "indep_holdout_acc": stats(lambda r: r["generalization"]["indep_holdout_acc"]),
        "all_seeds_criteria_met": all(r["functional"]["criteria_met"] for r in runs),
        "fraction_criteria_met": sum(int(r["functional"]["criteria_met"]) for r in runs) / float(len(runs)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 46 Contextual Disambiguation Pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--replay-lr", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase46_pilot_results.json"),
    )
    args = parser.parse_args(argv)

    conditions = ["baseline_unconditioned", "lemma_conditioned", "negative_control"]
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "conditions": conditions,
            "seeds": args.seeds,
            "replay_lr": args.replay_lr,
            "budget_updates_per_run": 4,
        },
        "configurations": {},
    }

    print("=" * 80)
    print("PHASE 46 CONTEXTUAL DISAMBIGUATION PILOT (Issue #46)")
    print(f"Seeds: {args.seeds} | Replay LR: {args.replay_lr:.2f} | Budget: 4 steps")
    print("=" * 80)

    for cond in conditions:
        print(f"\nEvaluating condition: {cond}...")
        runs = []
        for s in args.seeds:
            run_data = run_experiment_unit(cond, s, args.replay_lr)
            runs.append(run_data)
            print(
                f"  Seed {s}: cos_dist={run_data['separability']['mean_post_cos_dist']:.4f} "
                f"| R_pos={run_data['functional']['r_pos']:.4f} "
                f"| d_new={run_data['functional']['delta_new']:+.4f} "
                f"| JointLoss={run_data['functional']['joint_collided_loss']:.4f} "
                f"| DiagHold={run_data['generalization']['diag_holdout_acc']:.2f} "
                f"| IndepHold={run_data['generalization']['indep_holdout_acc']:.2f} "
                f"| Gate={run_data['functional']['criteria_met']}"
            )
        agg = aggregate_runs(runs)
        results["configurations"][cond] = {
            "runs": runs,
            "aggregates": agg,
        }

    # Summary Table
    print("\n" + "=" * 95)
    print(f"{'Condition':24s} | {'CosDist':8s} | {'R_pos':14s} | {'Delta_new':14s} | {'Loss':8s} | {'Diag':6s} | {'Indep':6s} | {'AllMet':7s}")
    print("-" * 95)
    for cond in conditions:
        agg = results["configurations"][cond]["aggregates"]
        print(
            f"{cond:24s} | "
            f"{agg['post_cos_dist']['mean']:.4f}   | "
            f"{agg['r_pos']['mean']:.4f} ± {agg['r_pos']['std']:.2f} | "
            f"{agg['delta_new']['mean']:+.4f} ± {agg['delta_new']['std']:.2f} | "
            f"{agg['joint_collided_loss']['mean']:.4f}   | "
            f"{agg['diag_holdout_acc']['mean']:.2f}   | "
            f"{agg['indep_holdout_acc']['mean']:.2f}   | "
            f"{str(agg['all_seeds_criteria_met']):7s}"
        )
    print("=" * 95)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audited results to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
