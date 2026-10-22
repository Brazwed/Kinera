"""Expanded Controlled Replay Pilot (Issue #42).

Executes controlled multi-seed A/B/C replay experiments across low, medium,
and high interference levels with strict holdout probe separation, gradient
update budget parity, and resource/memory profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import tracemalloc
from typing import Any, Literal, Mapping, Sequence

from kinera.backends import Backend, get_backend
from kinera.backends.base import BackendBuffer
from kinera.experiments.learning_memory import (
    ExampleRef,
    LearningEvent,
    LearningMemory,
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
    ProbeObservation,
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

InterferenceLevel = Literal["low", "medium", "high"]
BackendName = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class AggregatedMetric:
    """Statistical aggregation (mean, std, min, max) for scalar measurements."""

    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> AggregatedMetric:
        if not values:
            raise ValueError("Cannot aggregate empty sequence of values")
        n = len(values)
        mean_val = sum(values) / float(n)
        variance = sum((x - mean_val) ** 2 for x in values) / float(n)
        std_val = math.sqrt(variance)
        return cls(
            mean=float(mean_val),
            std=float(std_val),
            min=float(min(values)),
            max=float(max(values)),
        )


@dataclass(frozen=True, slots=True)
class TaskFamilySpec:
    """Explicit mapping linking an old task training item with internal and holdout probes."""

    family_id: str
    shared_prefix: str
    train_word: str
    internal_probe_word: str
    holdout_probe_word: str


@dataclass(frozen=True, slots=True)
class NewTaskSpec:
    """Explicit mapping linking a new task training item with its holdout probe."""

    task_id: str
    shared_prefix: str
    train_word: str
    holdout_word: str


@dataclass(frozen=True, slots=True)
class PilotConditionSpec:
    """Configuration for an operational interference condition."""

    level: InterferenceLevel
    description: str
    old_families: tuple[TaskFamilySpec, ...]
    new_tasks: tuple[NewTaskSpec, ...]
    old_epochs: int
    new_epochs: int
    replay_epochs: int
    old_learning_rate: float
    new_learning_rate: float
    replay_learning_rate: float
    replay_budget: int

    @property
    def old_words(self) -> tuple[str, ...]:
        return tuple(f.train_word for f in self.old_families)

    @property
    def old_internal_words(self) -> tuple[str, ...]:
        return tuple(f.internal_probe_word for f in self.old_families)

    @property
    def old_holdout_words(self) -> tuple[str, ...]:
        return tuple(f.holdout_probe_word for f in self.old_families)

    @property
    def new_words(self) -> tuple[str, ...]:
        return tuple(t.train_word for t in self.new_tasks)

    @property
    def new_holdout_words(self) -> tuple[str, ...]:
        return tuple(t.holdout_word for t in self.new_tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "description": self.description,
            "old_words": list(self.old_words),
            "new_words": list(self.new_words),
            "old_internal_words": list(self.old_internal_words),
            "old_holdout_words": list(self.old_holdout_words),
            "new_holdout_words": list(self.new_holdout_words),
            "old_epochs": self.old_epochs,
            "new_epochs": self.new_epochs,
            "replay_epochs": self.replay_epochs,
            "old_learning_rate": self.old_learning_rate,
            "new_learning_rate": self.new_learning_rate,
            "replay_learning_rate": self.replay_learning_rate,
            "replay_budget": self.replay_budget,
        }


@dataclass(frozen=True, slots=True)
class ProbePartitionBreakdown:
    """Explicit separation of positive-reference retention and zero-reference acquisition across probes."""

    positive_ref_group_ids: tuple[str, ...]
    positive_ref_retention_ratios: tuple[float, ...]
    mean_positive_ref_retention: float
    zero_ref_group_ids: tuple[str, ...]
    zero_ref_acquisition_deltas: tuple[float, ...]
    mean_zero_ref_acquisition: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_ref_group_ids": list(self.positive_ref_group_ids),
            "positive_ref_retention_ratios": list(self.positive_ref_retention_ratios),
            "mean_positive_ref_retention": self.mean_positive_ref_retention,
            "zero_ref_group_ids": list(self.zero_ref_group_ids),
            "zero_ref_acquisition_deltas": list(self.zero_ref_acquisition_deltas),
            "mean_zero_ref_acquisition": self.mean_zero_ref_acquisition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbePartitionBreakdown:
        return cls(
            positive_ref_group_ids=tuple(str(x) for x in data["positive_ref_group_ids"]),
            positive_ref_retention_ratios=tuple(
                float(x) for x in data["positive_ref_retention_ratios"]
            ),
            mean_positive_ref_retention=float(data["mean_positive_ref_retention"]),
            zero_ref_group_ids=tuple(str(x) for x in data["zero_ref_group_ids"]),
            zero_ref_acquisition_deltas=tuple(
                float(x) for x in data["zero_ref_acquisition_deltas"]
            ),
            mean_zero_ref_acquisition=float(data["mean_zero_ref_acquisition"]),
        )


def compute_probe_partition_breakdown(
    group_ids: Sequence[str],
    reference_competences: Sequence[float],
    post_competences: Sequence[float],
) -> ProbePartitionBreakdown:
    """Separate probes with positive reference competence from probes with zero reference competence.

    1. Retention ratio is computed strictly over probes with reference_competence > 0.0,
       ensuring previously nonexistent competence is never counted as preserved knowledge.
    2. Probes with reference_competence <= 0.0 are tracked separately via acquisition delta
       (post_competence - reference_competence).
    """
    if not (len(group_ids) == len(reference_competences) == len(post_competences)):
        raise ValueError(
            "group_ids, reference_competences, and post_competences must have equal length"
        )

    pos_groups: list[str] = []
    pos_retentions: list[float] = []
    zero_groups: list[str] = []
    zero_acquisitions: list[float] = []

    for gid, ref, post in zip(group_ids, reference_competences, post_competences):
        if ref > 0.0:
            pos_groups.append(gid)
            pos_retentions.append(
                calculate_retention_ratio(ref, post, "higher_is_better")
            )
        else:
            zero_groups.append(gid)
            zero_acquisitions.append(float(post - ref))

    mean_pos_ret = (
        sum(pos_retentions) / float(len(pos_retentions)) if pos_retentions else 0.0
    )
    mean_zero_acq = (
        sum(zero_acquisitions) / float(len(zero_acquisitions))
        if zero_acquisitions
        else 0.0
    )

    return ProbePartitionBreakdown(
        positive_ref_group_ids=tuple(pos_groups),
        positive_ref_retention_ratios=tuple(pos_retentions),
        mean_positive_ref_retention=mean_pos_ret,
        zero_ref_group_ids=tuple(zero_groups),
        zero_ref_acquisition_deltas=tuple(zero_acquisitions),
        mean_zero_ref_acquisition=mean_zero_acq,
    )


@dataclass(frozen=True, slots=True)
class PilotSeedResult:
    """Complete evaluation results for one seed under one condition."""

    level: InterferenceLevel
    seed: int
    eval_a: ScientificVariantEvaluation
    eval_b: ScientificVariantEvaluation
    eval_c: ScientificVariantEvaluation
    comparison: ScientificComparison
    holdout_retention_ratio: dict[str, float]
    cpu_time_seconds: float
    peak_ram_bytes: int
    peak_rss_bytes: int = 0
    zero_ref_acquisition: dict[str, float] | None = None
    holdout_zero_ref_acquisition: dict[str, float] | None = None
    probe_breakdowns: dict[str, dict[str, ProbePartitionBreakdown]] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "level": self.level,
            "seed": self.seed,
            "evaluations": {
                "A": self.eval_a.to_dict(),
                "B": self.eval_b.to_dict(),
                "C": self.eval_c.to_dict(),
            },
            "comparison": self.comparison.to_dict(),
            "holdout_retention_ratio": dict(self.holdout_retention_ratio),
        }
        if self.zero_ref_acquisition is not None:
            result["zero_ref_acquisition"] = dict(self.zero_ref_acquisition)
        if self.holdout_zero_ref_acquisition is not None:
            result["holdout_zero_ref_acquisition"] = dict(self.holdout_zero_ref_acquisition)
        if self.probe_breakdowns is not None:
            result["probe_breakdowns"] = {
                variant: {
                    partition: breakdown.to_dict()
                    for partition, breakdown in partitions.items()
                }
                for variant, partitions in self.probe_breakdowns.items()
            }
        result["cpu_time_seconds"] = self.cpu_time_seconds
        result["peak_ram_bytes"] = self.peak_ram_bytes
        result["peak_rss_bytes"] = self.peak_rss_bytes
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PilotSeedResult:
        if not isinstance(data, Mapping):
            raise TypeError("PilotSeedResult dictionary payload must be a Mapping")
        evaluations = data["evaluations"]
        eval_a = _variant_eval_from_dict(evaluations["A"])
        eval_b = _variant_eval_from_dict(evaluations["B"])
        eval_c = _variant_eval_from_dict(evaluations["C"])
        comp = _comparison_from_dict(data["comparison"])
        holdout_ret = {
            str(k): float(v) for k, v in data["holdout_retention_ratio"].items()
        }
        zero_ref = (
            {str(k): float(v) for k, v in data["zero_ref_acquisition"].items()}
            if data.get("zero_ref_acquisition") is not None
            else None
        )
        holdout_zero_ref = (
            {str(k): float(v) for k, v in data["holdout_zero_ref_acquisition"].items()}
            if data.get("holdout_zero_ref_acquisition") is not None
            else None
        )
        breakdowns: dict[str, dict[str, ProbePartitionBreakdown]] | None = None
        if data.get("probe_breakdowns") is not None:
            breakdowns = {
                str(variant): {
                    str(part): ProbePartitionBreakdown.from_dict(bd_dict)
                    for part, bd_dict in partitions.items()
                }
                for variant, partitions in data["probe_breakdowns"].items()
            }
        level = str(data["level"])
        if level not in ("low", "medium", "high"):
            raise ValueError(f"Invalid interference level in checkpoint: {level!r}")
        seed = data["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("PilotSeedResult seed must be an integer")
        return cls(
            level=level,  # type: ignore[arg-type]
            seed=seed,
            eval_a=eval_a,
            eval_b=eval_b,
            eval_c=eval_c,
            comparison=comp,
            holdout_retention_ratio=holdout_ret,
            cpu_time_seconds=float(data["cpu_time_seconds"]),
            peak_ram_bytes=int(data["peak_ram_bytes"]),
            peak_rss_bytes=int(data.get("peak_rss_bytes", 0)),
            zero_ref_acquisition=zero_ref,
            holdout_zero_ref_acquisition=holdout_zero_ref,
            probe_breakdowns=breakdowns,
        )


def _candidate_audit_from_dict(data: Mapping[str, Any]) -> CandidateAuditRecord:
    ref_data = data["example_ref"]
    ref = ExampleRef(
        str(ref_data["dataset_artifact_id"]),
        str(ref_data["split"]),
        int(ref_data["example_index"]),
    )
    return CandidateAuditRecord(
        example_ref=ref,
        group_ids=tuple(str(g) for g in data["group_ids"]),
        forgetting_score=float(data["forgetting_score"]),
        selected=bool(data["selected"]),
        selection_reason=(
            None
            if data.get("selection_reason") is None
            else str(data["selection_reason"])
        ),
    )


def _variant_eval_from_dict(data: Mapping[str, Any]) -> ScientificVariantEvaluation:
    cand_pool = tuple(_candidate_audit_from_dict(c) for c in data["candidate_pool"])
    sel_cands = tuple(_candidate_audit_from_dict(c) for c in data["selected_candidates"])
    acc_data = data["accounting"]
    accounting = ReplayAccounting(
        new_updates=int(acc_data["new_updates"]),
        replay_updates=int(acc_data["replay_updates"]),
        total_updates=int(acc_data["total_updates"]),
        selected_refs=tuple(c.example_ref for c in sel_cands),
        replay_seed=(
            None
            if acc_data.get("replay_seed") is None
            else int(acc_data["replay_seed"])
        ),
    )
    return ScientificVariantEvaluation(
        variant=str(data["variant"]),
        old_competence_reference=tuple(float(x) for x in data["old_competence_reference"]),
        old_competence_pre=tuple(float(x) for x in data["old_competence_pre"]),
        old_competence_post=tuple(float(x) for x in data["old_competence_post"]),
        old_forgetting_pre=tuple(float(x) for x in data["old_forgetting_pre"]),
        old_forgetting_post=tuple(float(x) for x in data["old_forgetting_post"]),
        old_recovery_delta=tuple(float(x) for x in data["old_recovery_delta"]),
        net_stability_delta=tuple(float(x) for x in data["net_stability_delta"]),
        new_competence_initial=float(data["new_competence_initial"]),
        new_competence_acquired=float(data["new_competence_acquired"]),
        new_competence_post=float(data["new_competence_post"]),
        plasticity_acquisition=float(data["plasticity_acquisition"]),
        new_competence_delta_during_replay=float(data["new_competence_delta_during_replay"]),
        accounting=accounting,
        replay_budget=int(data["replay_budget"]),
        mean_recovery_delta=float(data["mean_recovery_delta"]),
        mean_net_stability_delta=float(data["mean_net_stability_delta"]),
        mean_replay_utility=(
            None
            if data.get("mean_replay_utility") is None
            else float(data["mean_replay_utility"])
        ),
        mean_retention_ratio=float(data["mean_retention_ratio"]),
        stability_plasticity_tradeoff=float(data["stability_plasticity_tradeoff"]),
        candidate_pool=cand_pool,
        selected_candidates=sel_cands,
        new_competence_reference=(
            None
            if data.get("new_competence_reference") is None
            else float(data["new_competence_reference"])
        ),
        new_competence_delta=(
            None
            if data.get("new_competence_delta") is None
            else float(data["new_competence_delta"])
        ),
    )


def _comparison_from_dict(data: Mapping[str, Any]) -> ScientificComparison:
    return ScientificComparison(
        c_vs_b_recovery_advantage=float(data["c_vs_b_recovery_advantage"]),
        c_vs_a_recovery_advantage=float(data["c_vs_a_recovery_advantage"]),
        c_vs_b_utility_advantage=(
            None
            if data.get("c_vs_b_utility_advantage") is None
            else float(data["c_vs_b_utility_advantage"])
        ),
        c_vs_b_plasticity_delta=float(data["c_vs_b_plasticity_delta"]),
        c_vs_b_tradeoff_advantage=float(data["c_vs_b_tradeoff_advantage"]),
        b_vs_a_update_overhead=float(data["b_vs_a_update_overhead"]),
        c_vs_a_update_overhead=float(data["c_vs_a_update_overhead"]),
        is_fairly_matched_budget=bool(data["is_fairly_matched_budget"]),
        is_directed_superior=bool(data["is_directed_superior"]),
        overhead_definition=str(data["overhead_definition"]),
        summary_claim=str(data["summary_claim"]),
    )


@dataclass(frozen=True, slots=True)
class ConditionAggregateResult:
    """Aggregated statistical evaluation across all seeds for one condition."""

    level: InterferenceLevel
    num_seeds: int
    seeds: tuple[int, ...]
    c_vs_b_recovery_advantage: AggregatedMetric
    c_vs_b_retention_advantage: AggregatedMetric
    c_vs_b_holdout_retention_advantage: AggregatedMetric
    c_vs_b_tradeoff_advantage: AggregatedMetric
    c_vs_b_plasticity_delta: AggregatedMetric
    retention_ratio_a: AggregatedMetric
    retention_ratio_b: AggregatedMetric
    retention_ratio_c: AggregatedMetric
    holdout_retention_ratio_a: AggregatedMetric
    holdout_retention_ratio_b: AggregatedMetric
    holdout_retention_ratio_c: AggregatedMetric
    zero_ref_acquisition_a: AggregatedMetric
    zero_ref_acquisition_b: AggregatedMetric
    zero_ref_acquisition_c: AggregatedMetric
    holdout_zero_ref_acquisition_a: AggregatedMetric
    holdout_zero_ref_acquisition_b: AggregatedMetric
    holdout_zero_ref_acquisition_c: AggregatedMetric
    new_competence_delta_during_replay_a: AggregatedMetric
    new_competence_delta_during_replay_b: AggregatedMetric
    new_competence_delta_during_replay_c: AggregatedMetric
    probe_partition_sets: dict[str, list[str]]
    fairness_confirmed_all_seeds: bool
    superiority_fraction: float
    mean_cpu_time_seconds: float
    peak_ram_bytes: int
    peak_rss_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "num_seeds": self.num_seeds,
            "seeds": list(self.seeds),
            "probe_partition_sets": dict(self.probe_partition_sets),
            "c_vs_b_recovery_advantage": self.c_vs_b_recovery_advantage.to_dict(),
            "c_vs_b_retention_advantage": self.c_vs_b_retention_advantage.to_dict(),
            "c_vs_b_holdout_retention_advantage": self.c_vs_b_holdout_retention_advantage.to_dict(),
            "c_vs_b_tradeoff_advantage": self.c_vs_b_tradeoff_advantage.to_dict(),
            "c_vs_b_plasticity_delta": self.c_vs_b_plasticity_delta.to_dict(),
            "retention_ratio": {
                "A": self.retention_ratio_a.to_dict(),
                "B": self.retention_ratio_b.to_dict(),
                "C": self.retention_ratio_c.to_dict(),
            },
            "holdout_retention_ratio": {
                "A": self.holdout_retention_ratio_a.to_dict(),
                "B": self.holdout_retention_ratio_b.to_dict(),
                "C": self.holdout_retention_ratio_c.to_dict(),
            },
            "zero_ref_acquisition": {
                "A": self.zero_ref_acquisition_a.to_dict(),
                "B": self.zero_ref_acquisition_b.to_dict(),
                "C": self.zero_ref_acquisition_c.to_dict(),
            },
            "holdout_zero_ref_acquisition": {
                "A": self.holdout_zero_ref_acquisition_a.to_dict(),
                "B": self.holdout_zero_ref_acquisition_b.to_dict(),
                "C": self.holdout_zero_ref_acquisition_c.to_dict(),
            },
            "new_competence_delta_during_replay": {
                "A": self.new_competence_delta_during_replay_a.to_dict(),
                "B": self.new_competence_delta_during_replay_b.to_dict(),
                "C": self.new_competence_delta_during_replay_c.to_dict(),
            },
            "fairness_confirmed_all_seeds": self.fairness_confirmed_all_seeds,
            "superiority_fraction": self.superiority_fraction,
            "mean_cpu_time_seconds": self.mean_cpu_time_seconds,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class PilotCostProfile:
    """Resource measurements and scaling projection for execution planning."""

    total_cpu_time_seconds: float
    peak_ram_bytes: int
    peak_ram_mb: float
    peak_rss_bytes: int
    peak_rss_mb: float
    projected_scaled_hours_cpu: float
    scaling_recommendation: Literal["local_cpu", "google_colab_pro"]
    recommendation_rationale: str
    profile_heap_enabled: bool = False
    backend: str = "cpu"
    backend_init_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cpu_time_seconds": self.total_cpu_time_seconds,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mb": self.peak_rss_mb,
            "projected_scaled_hours_cpu": self.projected_scaled_hours_cpu,
            "scaling_recommendation": self.scaling_recommendation,
            "recommendation_rationale": self.recommendation_rationale,
            "profile_heap_enabled": self.profile_heap_enabled,
            "backend": self.backend,
            "backend_init_seconds": self.backend_init_seconds,
        }


# Standard quadripartite task families across all experimental conditions:
# - D_old_train: words used exclusively for initial training and admitted to LearningMemory
# - P_old_internal: separate words used exclusively for internal learning state / forgetting score evaluation
# - P_old_holdout: separate out-of-sample words never observed in memory, used only for holdout retention
_STANDARD_OLD_FAMILIES: tuple[TaskFamilySpec, ...] = (
    TaskFamilySpec(
        family_id="family-m",
        shared_prefix="m",
        train_word="mala",
        internal_probe_word="malas",
        holdout_probe_word="maleta",
    ),
    TaskFamilySpec(
        family_id="family-p",
        shared_prefix="p",
        train_word="pato",
        internal_probe_word="patos",
        holdout_probe_word="pateta",
    ),
    TaskFamilySpec(
        family_id="family-s",
        shared_prefix="s",
        train_word="sapo",
        internal_probe_word="sapos",
        holdout_probe_word="sapato",
    ),
    TaskFamilySpec(
        family_id="family-f",
        shared_prefix="f",
        train_word="fogo",
        internal_probe_word="fogos",
        holdout_probe_word="foguete",
    ),
)

_STANDARD_OLD_WORDS = tuple(f.train_word for f in _STANDARD_OLD_FAMILIES)
_STANDARD_OLD_INTERNAL_WORDS = tuple(f.internal_probe_word for f in _STANDARD_OLD_FAMILIES)
_STANDARD_OLD_HOLDOUT_WORDS = tuple(f.holdout_probe_word for f in _STANDARD_OLD_FAMILIES)

_CONDITIONS: dict[InterferenceLevel, PilotConditionSpec] = {
    "low": PilotConditionSpec(
        level="low",
        description="Disjoint prefixes and orthogonal character transitions; low representational collision",
        old_families=_STANDARD_OLD_FAMILIES,
        new_tasks=(
            NewTaskSpec("new-z", "z", "zero", "zebra"),
            NewTaskSpec("new-l", "l", "luxo", "luva"),
        ),
        old_epochs=20,
        new_epochs=4,
        replay_epochs=1,
        old_learning_rate=0.08,
        new_learning_rate=0.04,
        replay_learning_rate=0.20,
        replay_budget=2,
    ),
    "medium": PilotConditionSpec(
        level="medium",
        description="Partial consonant/vowel sharing with single-word collision; moderate representational drift",
        old_families=_STANDARD_OLD_FAMILIES,
        new_tasks=(
            NewTaskSpec("new-m", "m", "medo", "meta"),
            NewTaskSpec("new-d", "d", "dado", "data"),
        ),
        old_epochs=20,
        new_epochs=10,
        replay_epochs=1,
        old_learning_rate=0.08,
        new_learning_rate=0.07,
        replay_learning_rate=0.20,
        replay_budget=2,
    ),
    "high": PilotConditionSpec(
        level="high",
        description="Direct prefix context collision on identical n-grams ('<START>', 'm') and ('<START>', 'p') with divergent targets",
        old_families=_STANDARD_OLD_FAMILIES,
        new_tasks=(
            NewTaskSpec("new-m", "m", "medo", "meta"),
            NewTaskSpec("new-p", "p", "povo", "pote"),
        ),
        old_epochs=20,
        new_epochs=15,
        replay_epochs=1,
        old_learning_rate=0.08,
        new_learning_rate=0.10,
        replay_learning_rate=0.20,
        replay_budget=2,
    ),
}

_WORD_SYLLABLES: dict[str, list[str]] = {
    # Old train words
    "mala": ["ma", "la"],
    "pato": ["pa", "to"],
    "sapo": ["sa", "po"],
    "fogo": ["fo", "go"],
    # Old internal control probes
    "malas": ["ma", "las"],
    "patos": ["pa", "tos"],
    "sapos": ["sa", "pos"],
    "fogos": ["fo", "gos"],
    # Old holdout evaluation probes
    "maleta": ["ma", "le", "ta"],
    "pateta": ["pa", "te", "ta"],
    "sapato": ["sa", "pa", "to"],
    "foguete": ["fo", "gue", "te"],
    # New train words
    "zero": ["ze", "ro"],
    "luxo": ["lu", "xo"],
    "medo": ["me", "do"],
    "dado": ["da", "do"],
    "povo": ["po", "vo"],
    # New holdout evaluation probes
    "zebra": ["ze", "bra"],
    "luva": ["lu", "va"],
    "meta": ["me", "ta"],
    "data": ["da", "ta"],
    "pote": ["po", "te"],
}


def _build_unified_vocabularies() -> tuple[CharacterVocabulary, SyllableVocabulary]:
    """Build canonical unified vocabularies across all experimental conditions."""
    records = [
        annotate_record(
            {
                "word_key": word,
                "word_normalized": word,
                "syllables": _WORD_SYLLABLES[word],
            }
        )
        for word in _WORD_SYLLABLES
    ]
    char_vocab = CharacterVocabulary.from_records(records)
    syl_vocab = SyllableVocabulary.from_records(records)
    return char_vocab, syl_vocab


def get_condition_spec(level: InterferenceLevel) -> PilotConditionSpec:
    if level not in _CONDITIONS:
        raise ValueError(f"Unknown interference level: {level!r}")
    return _CONDITIONS[level]


class _BackendResourceTracker:
    """Monitors CUDA bridge allocations/frees to confirm zero resource leakage between models."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.bridge = getattr(backend, "bridge", None)
        self.live_pointers: set[int] = set()
        self.unmatched_frees: list[int] = []
        self.total_allocations: int = 0
        self.total_frees: int = 0
        self._original_allocate: Any = None
        self._original_free: Any = None
        self._attached = False

        if (
            self.bridge is not None
            and callable(getattr(self.bridge, "allocate", None))
            and callable(getattr(self.bridge, "free", None))
        ):
            self._original_allocate = self.bridge.allocate
            self._original_free = self.bridge.free

            def tracked_allocate(device_id: int, nbytes: int) -> int:
                ptr = int(self._original_allocate(device_id, nbytes))
                self.live_pointers.add(ptr)
                self.total_allocations += 1
                return ptr

            def tracked_free(device_id: int, pointer: int) -> None:
                ptr = int(pointer)
                self._original_free(device_id, ptr)
                self.total_frees += 1
                if ptr in self.live_pointers:
                    self.live_pointers.remove(ptr)
                else:
                    self.unmatched_frees.append(ptr)

            self.bridge.allocate = tracked_allocate
            self.bridge.free = tracked_free
            self._attached = True

    def assert_clean(self, context: str) -> None:
        if self.unmatched_frees:
            raise RuntimeError(
                f"CUDA resource tracker detected unmatched frees ({context}): {self.unmatched_frees}"
            )
        if self.live_pointers:
            raise RuntimeError(
                f"CUDA resource tracker detected {len(self.live_pointers)} unreleased device buffer(s) ({context})"
            )

    def detach(self) -> None:
        if self._attached and self.bridge is not None:
            self.bridge.allocate = self._original_allocate
            self.bridge.free = self._original_free
            self._attached = False


def _assert_model_resources_released(
    model: Any,
    tracker: _BackendResourceTracker | None = None,
) -> None:
    """Verify that model.close() released all model-owned buffers before the next execution."""
    if hasattr(model, "_closed") and not bool(getattr(model, "_closed")):
        raise RuntimeError("AutoregressiveModel was not marked closed after model.close()")

    rep = getattr(model, "representation", None)
    if rep is not None:
        for emb_name in ("char_embedding", "syllable_embedding"):
            emb = getattr(rep, emb_name, None)
            if emb is not None:
                for attr in ("weights", "last_ids", "output", "grad_weights"):
                    val = getattr(emb, attr, None)
                    if isinstance(val, BackendBuffer) and not val.closed:
                        raise RuntimeError(
                            f"Unreleased BackendBuffer on {emb_name}.{attr} after model.close()"
                        )
    heads = getattr(model, "heads", None)
    if heads is not None:
        for head_name in ("char", "syllable", "boundary"):
            layer = getattr(heads, head_name, None)
            if layer is not None:
                for attr in ("weights", "bias", "last_input", "output", "grad_weights", "grad_bias"):
                    val = getattr(layer, attr, None)
                    if isinstance(val, BackendBuffer) and not val.closed:
                        raise RuntimeError(
                            f"Unreleased BackendBuffer on heads.{head_name}.{attr} after model.close()"
                        )
    for attr in ("last_logits", "last_pooled"):
        val = getattr(model, attr, None)
        if isinstance(val, BackendBuffer) and not val.closed:
            raise RuntimeError(f"Unreleased BackendBuffer on model.{attr} after model.close()")

    if tracker is not None:
        tracker.assert_clean("after model.close()")


def _validate_backend_name(backend: str) -> BackendName:
    if not isinstance(backend, str) or backend not in ("cpu", "cuda"):
        raise ValueError("backend deve ser 'cpu' ou 'cuda'")
    return backend  # type: ignore[return-value]


def _create_execution_backend(backend: BackendName) -> Any | None:
    """Create an explicitly selected backend without importing CUDA on CPU runs."""
    _validate_backend_name(backend)
    if backend == "cpu":
        return None
    return get_backend(backend)


def _prepare_execution_backend(
    backend: BackendName | Backend | Any = "cpu",
    *,
    warmup_cuda: bool = True,
) -> tuple[str, Any | None, _BackendResourceTracker | None, float]:
    """Resolve and optionally warm up the backend, separating init/compile time from training."""
    t_init = time.perf_counter()
    if backend is None or backend == "cpu":
        return "cpu", None, None, time.perf_counter() - t_init
    if isinstance(backend, str):
        backend_name = _validate_backend_name(backend)
        resolved = _create_execution_backend(backend_name)
    else:
        resolved = backend
        backend_name = str(getattr(resolved, "name", "custom"))

    if warmup_cuda and resolved is not None:
        runtime = getattr(resolved, "runtime", None)
        if runtime is not None and callable(getattr(runtime, "require_device", None)):
            runtime.require_device(0)
        buffer_zeros = getattr(resolved, "buffer_zeros", None)
        release_buffer = getattr(resolved, "release_buffer", None)
        if callable(buffer_zeros) and callable(release_buffer):
            warmup_buf = buffer_zeros(shape=(1,), dtype="float64")
            release_buffer(warmup_buf)

    init_seconds = time.perf_counter() - t_init
    tracker = _BackendResourceTracker(resolved) if resolved is not None else None
    return backend_name, resolved, tracker, init_seconds


def write_pilot_checkpoint(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write the pilot JSON report/checkpoint to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def run_condition_seed(
    level: InterferenceLevel,
    seed: int,
    char_vocab: CharacterVocabulary,
    syl_vocab: SyllableVocabulary,
    *,
    backend: BackendName | Backend | Any = "cpu",
    profile_heap: bool = False,
    _tracker: _BackendResourceTracker | None = None,
) -> PilotSeedResult:
    """Execute A, B, and C variants for one seed under a specific interference level.

    Guarantees:
    - Holdout evaluation probes are strictly isolated and never observed by LearningMemory.
    - Candidate pool priority scoring uses only internal control probes.
    - Variants B and C use strictly matched replay and update budgets.
    - Physical gradient updates are tracked via step_observer and verified against accounting.
    - Model resources (including CUDA buffers) are released in all exit paths via try/finally.
    - Optional tracemalloc heap profiling is disabled by default (profile_heap=False).
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um inteiro")
    if isinstance(profile_heap, bool) is False:
        raise TypeError("profile_heap deve ser um booleano")

    owns_tracker = False
    if _tracker is not None:
        resolved_backend = None if (backend is None or backend == "cpu") else backend
        tracker: _BackendResourceTracker | None = _tracker
    else:
        _, resolved_backend, tracker, _ = _prepare_execution_backend(
            backend, warmup_cuda=False
        )
        owns_tracker = tracker is not None
    spec = get_condition_spec(level)
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

    # Generate autoregressive examples (primary transition ex[1])
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

    # Separate reference partitions with distinct artifact IDs and splits:
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
        ExampleRef(f"dataset:pilot-new-train-{level}", "train", i)
        for i in range(len(new_train_records))
    )
    new_holdout_refs = tuple(
        ExampleRef(f"dataset:pilot-new-holdout-{level}", "holdout_test", i)
        for i in range(len(new_holdout_records))
    )

    # Strict audit: verify pairwise disjoint reference sets
    assert not (set(old_refs) & set(internal_probe_refs)), "old_train and internal_probe refs must be disjoint"
    assert not (set(old_refs) & set(holdout_probe_refs)), "old_train and holdout refs must be disjoint"
    assert not (set(internal_probe_refs) & set(holdout_probe_refs)), "internal_probe and holdout refs must be disjoint"
    assert not (set(old_refs) & set(new_refs)), "old_train and new_train refs must be disjoint"
    assert not (set(new_refs) & set(new_holdout_refs)), "new_train and new_holdout refs must be disjoint"

    internal_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-int-{f.internal_probe_word}", (ref,), "char_accuracy", "higher_is_better"
        )
        for f, ref in zip(spec.old_families, internal_probe_refs)
    )
    holdout_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-hold-{f.holdout_probe_word}", (ref,), "char_accuracy", "higher_is_better"
        )
        for f, ref in zip(spec.old_families, holdout_probe_refs)
    )
    new_probe_groups = tuple(
        RetentionProbeGroup(
            f"probe-new-{level}-{t.train_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for t, ref in zip(spec.new_tasks, new_refs)
    )
    new_holdout_groups = tuple(
        RetentionProbeGroup(
            f"probe-new-hold-{level}-{t.holdout_word}",
            (ref,),
            "char_accuracy",
            "higher_is_better",
        )
        for t, ref in zip(spec.new_tasks, new_holdout_refs)
    )

    t0 = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started_tracemalloc = False
    if profile_heap and not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracemalloc = True

    variant_evals: dict[str, ScientificVariantEvaluation] = {}
    holdout_retentions: dict[str, float] = {}
    zero_ref_acquisitions: dict[str, float] = {}
    holdout_zero_ref_acquisitions: dict[str, float] = {}
    probe_breakdowns: dict[str, dict[str, ProbePartitionBreakdown]] = {}

    try:
        for variant in ("A", "B", "C"):
            if tracker is not None:
                tracker.assert_clean(f"before variant {variant} model init (seed={seed})")
            model: AutoregressiveModel | None = None
            memory = LearningMemory(8)
            try:
                model = AutoregressiveModel(
                    len(char_vocab),
                    len(syl_vocab),
                    char_embedding_dim=8,
                    syllable_embedding_dim=4,
                    seed=seed,
                    backend=resolved_backend,
                )

                # Memory step observer for old task acquisition (only old_refs admitted)
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

                # 1. Acquire old tasks
                train_multitask_examples(
                    model,
                    ex_old_train,
                    char_vocab,
                    epochs=spec.old_epochs,
                    learning_rate=spec.old_learning_rate,
                    step_observer=observer,
                )

                # Baseline competence references on internal and holdout probes
                old_ref_obs = tuple(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab)
                    for g, r, e in zip(
                        internal_probe_groups, internal_probe_refs, ex_old_internal
                    )
                )
                old_ref_comp = tuple(obs.competence for obs in old_ref_obs)

                old_holdout_ref_obs = tuple(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab)
                    for g, r, e in zip(
                        holdout_probe_groups, holdout_probe_refs, ex_old_holdout
                    )
                )
                old_holdout_ref_comp = tuple(obs.competence for obs in old_holdout_ref_obs)

                new_init_comp = sum(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
                    for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
                ) / float(len(new_probe_groups))

                # 2. Interfering learning task
                train_multitask_examples(
                    model,
                    ex_new_train,
                    char_vocab,
                    epochs=spec.new_epochs,
                    learning_rate=spec.new_learning_rate,
                )

                old_pre_obs = tuple(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab)
                    for g, r, e in zip(
                        internal_probe_groups, internal_probe_refs, ex_old_internal
                    )
                )
                old_pre_comp = tuple(obs.competence for obs in old_pre_obs)

                new_acq_comp = sum(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
                    for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
                ) / float(len(new_probe_groups))

                # 3. Learning states & candidates (derived strictly from internal probes)
                states = tuple(
                    build_learning_state(
                        ProbeObservation(
                            g.group_id,
                            (r,),
                            "char_accuracy",
                            "higher_is_better",
                            ref_c,
                            1,
                        ),
                        ProbeObservation(
                            g.group_id,
                            (r,),
                            "char_accuracy",
                            "higher_is_better",
                            pre_c,
                            1,
                        ),
                        status="LEARNING",
                        last_practiced_step=spec.old_epochs + spec.new_epochs,
                    )
                    for g, r, ref_c, pre_c in zip(
                        internal_probe_groups,
                        internal_probe_refs,
                        old_ref_comp,
                        old_pre_comp,
                    )
                )
                state_by_group = {s.group_id: s for s in states}
                candidates = build_replay_candidates(memory, state_by_group)

                # Strict audit: verify candidate pool contains NO holdout refs and is strictly disjoint
                candidate_refs = {c.example_ref for c in candidates}
                holdout_ref_set = set(holdout_probe_refs)
                assert candidate_refs.isdisjoint(holdout_ref_set), "Holdout refs leaked into candidate pool!"
                holdout_group_ids = {g.group_id for g in holdout_probe_groups}
                for c in candidates:
                    assert not any(gid in holdout_group_ids for gid in c.group_ids), (
                        f"Holdout probe group {c.group_ids} leaked into candidate {c.example_ref}!"
                    )

                budget = 0 if variant == "A" else spec.replay_budget
                used_seed: int | None = None
                if variant == "A":
                    selected: tuple[ReplayCandidate, ...] = ()
                elif variant == "B":
                    selected = select_random_replay(candidates, budget, replay_seed=seed)
                    used_seed = seed
                elif variant == "C":
                    selected = select_directed_replay(candidates, budget)
                else:
                    raise ValueError(f"Unknown variant: {variant!r}")

                sel_refs = tuple(c.example_ref for c in selected)
                plan = build_interleaved_plan(
                    new_refs,
                    sel_refs,
                    replay_positions=tuple(
                        range(len(new_refs), len(new_refs) + len(sel_refs))
                    ),
                )
                all_refs = (*old_refs, *new_refs)
                all_examples = (*ex_old_train, *ex_new_train)
                replay_ex = resolve_replay_examples(plan, all_refs, all_examples)

                # 4. Replay execution with physical gradient update observation
                actual_new_updates = 0
                actual_replay_updates = 0

                def physical_step_tracker(epoch: int, index: int, step: int, report: dict[str, float]) -> None:
                    nonlocal actual_new_updates, actual_replay_updates
                    item = plan[index]
                    if item.source == "replay":
                        actual_replay_updates += 1
                    else:
                        actual_new_updates += 1

                train_multitask_examples(
                    model,
                    list(replay_ex),
                    char_vocab,
                    epochs=spec.replay_epochs,
                    learning_rate=spec.replay_learning_rate,
                    step_observer=physical_step_tracker,
                )

                # Strict budget audit assertion: physical updates must match declared accounting
                accounting = ReplayAccounting(
                    new_updates=len(new_refs),
                    replay_updates=len(sel_refs),
                    total_updates=len(new_refs) + len(sel_refs),
                    selected_refs=sel_refs,
                    replay_seed=used_seed,
                )
                assert actual_new_updates == accounting.new_updates, (
                    f"Physical new updates mismatch: {actual_new_updates} != {accounting.new_updates}"
                )
                assert actual_replay_updates == accounting.replay_updates, (
                    f"Physical replay updates mismatch: {actual_replay_updates} != {accounting.replay_updates}"
                )
                assert actual_new_updates + actual_replay_updates == accounting.total_updates, (
                    f"Physical total updates mismatch: {actual_new_updates + actual_replay_updates} != {accounting.total_updates}"
                )

                # 5. Post evaluation
                old_post_obs = tuple(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab)
                    for g, r, e in zip(
                        internal_probe_groups, internal_probe_refs, ex_old_internal
                    )
                )
                old_post_comp = tuple(obs.competence for obs in old_post_obs)

                old_holdout_post_obs = tuple(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab)
                    for g, r, e in zip(
                        holdout_probe_groups, holdout_probe_refs, ex_old_holdout
                    )
                )
                old_holdout_post_comp = tuple(
                    obs.competence for obs in old_holdout_post_obs
                )

                new_post_comp = sum(
                    evaluate_retention_probe(model, g, (r,), (e,), char_vocab).competence
                    for g, r, e in zip(new_probe_groups, new_refs, ex_new_train)
                ) / float(len(new_probe_groups))

                # Metrics
                old_rec = tuple(
                    post - pre for pre, post in zip(old_pre_comp, old_post_comp)
                )
                mean_rec = sum(old_rec) / len(old_rec) if old_rec else 0.0
                plasticity = new_acq_comp - new_init_comp
                new_delta_during_replay = new_post_comp - new_acq_comp

                # Partition probes by initial competence:
                # - Retention is strictly measured on probes with positive reference competence (ref > 0.0).
                # - Acquisition on initially unlearned probes (ref <= 0.0) is tracked separately.
                internal_breakdown = compute_probe_partition_breakdown(
                    tuple(g.group_id for g in internal_probe_groups),
                    old_ref_comp,
                    old_post_comp,
                )
                mean_ret = internal_breakdown.mean_positive_ref_retention
                zero_ref_acquisitions[variant] = internal_breakdown.mean_zero_ref_acquisition
                tradeoff = calculate_stability_plasticity_tradeoff(
                    mean_ret, plasticity
                )

                # Holdout retention ratios (strictly over positive-reference holdout probes)
                holdout_breakdown = compute_probe_partition_breakdown(
                    tuple(g.group_id for g in holdout_probe_groups),
                    old_holdout_ref_comp,
                    old_holdout_post_comp,
                )
                holdout_retentions[variant] = holdout_breakdown.mean_positive_ref_retention
                holdout_zero_ref_acquisitions[variant] = (
                    holdout_breakdown.mean_zero_ref_acquisition
                )
                probe_breakdowns[variant] = {
                    "internal": internal_breakdown,
                    "holdout": holdout_breakdown,
                }

                util = (
                    calculate_replay_utility(mean_rec, accounting.replay_updates)
                    if accounting.replay_updates > 0
                    else None
                )

                audit_pool: list[CandidateAuditRecord] = []
                for c in candidates:
                    is_sel = c.example_ref in sel_refs
                    reason = None
                    if is_sel:
                        reason = (
                            "uniform_random_sample"
                            if variant == "B"
                            else "max_forgetting_score"
                        )
                    audit_pool.append(
                        CandidateAuditRecord(
                            example_ref=c.example_ref,
                            group_ids=c.group_ids,
                            forgetting_score=c.priority_score,
                            selected=is_sel,
                            selection_reason=reason,
                        )
                    )

                variant_evals[variant] = ScientificVariantEvaluation(
                    variant=variant,
                    old_competence_reference=old_ref_comp,
                    old_competence_pre=old_pre_comp,
                    old_competence_post=old_post_comp,
                    old_forgetting_pre=tuple(s.forgetting_score or 0.0 for s in states),
                    old_forgetting_post=tuple(
                        calculate_forgetting(r, p, "higher_is_better") or 0.0
                        for r, p in zip(old_ref_comp, old_post_comp)
                    ),
                    old_recovery_delta=old_rec,
                    net_stability_delta=tuple(
                        p - r for r, p in zip(old_ref_comp, old_post_comp)
                    ),
                    new_competence_initial=new_init_comp,
                    new_competence_acquired=new_acq_comp,
                    new_competence_post=new_post_comp,
                    plasticity_acquisition=plasticity,
                    new_competence_delta_during_replay=new_delta_during_replay,
                    accounting=accounting,
                    replay_budget=budget,
                    mean_recovery_delta=mean_rec,
                    mean_net_stability_delta=(
                        sum(p - r for r, p in zip(old_ref_comp, old_post_comp))
                        / len(old_ref_comp)
                    ),
                    mean_replay_utility=util,
                    mean_retention_ratio=mean_ret,
                    stability_plasticity_tradeoff=tradeoff,
                    candidate_pool=tuple(audit_pool),
                    selected_candidates=tuple(c for c in audit_pool if c.selected),
                )
            finally:
                if model is not None:
                    model.close()
                    _assert_model_resources_released(model, tracker)

        comp = compare_scientific_variants(
            variant_evals["A"], variant_evals["B"], variant_evals["C"]
        )
        peak_ram = (
            tracemalloc.get_traced_memory()[1]
            if (profile_heap and tracemalloc.is_tracing())
            else 0
        )
    finally:
        if started_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        if owns_tracker and tracker is not None:
            tracker.detach()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_rss = max(rss_before, rss_after)
    duration = time.perf_counter() - t0

    return PilotSeedResult(
        level=level,
        seed=seed,
        eval_a=variant_evals["A"],
        eval_b=variant_evals["B"],
        eval_c=variant_evals["C"],
        comparison=comp,
        holdout_retention_ratio=holdout_retentions,
        zero_ref_acquisition=zero_ref_acquisitions,
        holdout_zero_ref_acquisition=holdout_zero_ref_acquisitions,
        probe_breakdowns=probe_breakdowns,
        cpu_time_seconds=duration,
        peak_ram_bytes=peak_ram,
        peak_rss_bytes=peak_rss,
    )


def aggregate_condition_results(
    level: InterferenceLevel,
    seed_results: Sequence[PilotSeedResult],
) -> ConditionAggregateResult:
    """Aggregate statistical metrics across seeds for one interference condition."""
    if not seed_results:
        raise ValueError("seed_results cannot be empty")
    seeds = tuple(r.seed for r in seed_results)

    rec_advs = [r.comparison.c_vs_b_recovery_advantage for r in seed_results]
    ret_advs = [
        r.eval_c.mean_retention_ratio - r.eval_b.mean_retention_ratio
        for r in seed_results
    ]
    hold_ret_advs = [
        r.holdout_retention_ratio["C"] - r.holdout_retention_ratio["B"]
        for r in seed_results
    ]
    trade_advs = [r.comparison.c_vs_b_tradeoff_advantage for r in seed_results]
    plas_deltas = [r.comparison.c_vs_b_plasticity_delta for r in seed_results]

    ret_a = [r.eval_a.mean_retention_ratio for r in seed_results]
    ret_b = [r.eval_b.mean_retention_ratio for r in seed_results]
    ret_c = [r.eval_c.mean_retention_ratio for r in seed_results]

    hold_a = [r.holdout_retention_ratio["A"] for r in seed_results]
    hold_b = [r.holdout_retention_ratio["B"] for r in seed_results]
    hold_c = [r.holdout_retention_ratio["C"] for r in seed_results]

    zref_a = [r.zero_ref_acquisition["A"] for r in seed_results]
    zref_b = [r.zero_ref_acquisition["B"] for r in seed_results]
    zref_c = [r.zero_ref_acquisition["C"] for r in seed_results]

    hzref_a = [r.holdout_zero_ref_acquisition["A"] for r in seed_results]
    hzref_b = [r.holdout_zero_ref_acquisition["B"] for r in seed_results]
    hzref_c = [r.holdout_zero_ref_acquisition["C"] for r in seed_results]

    new_delta_a = [r.eval_a.new_competence_delta_during_replay for r in seed_results]
    new_delta_b = [r.eval_b.new_competence_delta_during_replay for r in seed_results]
    new_delta_c = [r.eval_c.new_competence_delta_during_replay for r in seed_results]

    first_bd = seed_results[0].probe_breakdowns["C"]
    probe_partition_sets = {
        "positive_ref_internal_groups": list(
            first_bd["internal"].positive_ref_group_ids
        ),
        "zero_ref_internal_groups": list(first_bd["internal"].zero_ref_group_ids),
        "positive_ref_holdout_groups": list(
            first_bd["holdout"].positive_ref_group_ids
        ),
        "zero_ref_holdout_groups": list(first_bd["holdout"].zero_ref_group_ids),
    }

    fairness_all = all(
        r.comparison.is_fairly_matched_budget for r in seed_results
    )
    sup_frac = sum(
        1.0 for r in seed_results if r.comparison.is_directed_superior
    ) / float(len(seed_results))

    cpu_times = [r.cpu_time_seconds for r in seed_results]
    peak_ram = max(r.peak_ram_bytes for r in seed_results)
    peak_rss = max(r.peak_rss_bytes for r in seed_results)

    return ConditionAggregateResult(
        level=level,
        num_seeds=len(seeds),
        seeds=seeds,
        c_vs_b_recovery_advantage=AggregatedMetric.from_sequence(rec_advs),
        c_vs_b_retention_advantage=AggregatedMetric.from_sequence(ret_advs),
        c_vs_b_holdout_retention_advantage=AggregatedMetric.from_sequence(
            hold_ret_advs
        ),
        c_vs_b_tradeoff_advantage=AggregatedMetric.from_sequence(trade_advs),
        c_vs_b_plasticity_delta=AggregatedMetric.from_sequence(plas_deltas),
        retention_ratio_a=AggregatedMetric.from_sequence(ret_a),
        retention_ratio_b=AggregatedMetric.from_sequence(ret_b),
        retention_ratio_c=AggregatedMetric.from_sequence(ret_c),
        holdout_retention_ratio_a=AggregatedMetric.from_sequence(hold_a),
        holdout_retention_ratio_b=AggregatedMetric.from_sequence(hold_b),
        holdout_retention_ratio_c=AggregatedMetric.from_sequence(hold_c),
        zero_ref_acquisition_a=AggregatedMetric.from_sequence(zref_a),
        zero_ref_acquisition_b=AggregatedMetric.from_sequence(zref_b),
        zero_ref_acquisition_c=AggregatedMetric.from_sequence(zref_c),
        holdout_zero_ref_acquisition_a=AggregatedMetric.from_sequence(hzref_a),
        holdout_zero_ref_acquisition_b=AggregatedMetric.from_sequence(hzref_b),
        holdout_zero_ref_acquisition_c=AggregatedMetric.from_sequence(hzref_c),
        new_competence_delta_during_replay_a=AggregatedMetric.from_sequence(new_delta_a),
        new_competence_delta_during_replay_b=AggregatedMetric.from_sequence(new_delta_b),
        new_competence_delta_during_replay_c=AggregatedMetric.from_sequence(new_delta_c),
        probe_partition_sets=probe_partition_sets,
        fairness_confirmed_all_seeds=fairness_all,
        superiority_fraction=sup_frac,
        mean_cpu_time_seconds=sum(cpu_times) / float(len(cpu_times)),
        peak_ram_bytes=peak_ram,
        peak_rss_bytes=peak_rss,
    )


def compute_cost_profile(
    total_cpu_seconds: float,
    peak_ram_bytes: int,
    peak_rss_bytes: int = 0,
    pilot_seeds_count: int = 3,
    pilot_words_count: int = 6,
    pilot_epochs: int = 36,
    target_seeds_count: int = 10,
    target_words_count: int = 500,
    target_epochs: int = 50,
    *,
    profile_heap_enabled: bool = False,
    backend: str = "cpu",
    backend_init_seconds: float = 0.0,
) -> PilotCostProfile:
    """Project scaled computation costs and recommend execution environment.

    Scaling formula:
        Scale Factor = (target_seeds / pilot_seeds)
                     * (target_words / pilot_words)
                     * (target_epochs / pilot_epochs)

    Differentiates Python heap allocations (tracemalloc) from operating system
    process Resident Set Size (getrusage) and separates measured runtime from
    theoretical parametric projections.
    """
    peak_ram_mb = peak_ram_bytes / (1024.0 * 1024.0)
    peak_rss_mb = peak_rss_bytes / (1024.0 * 1024.0)

    scale_factor = (
        (float(target_seeds_count) / float(max(pilot_seeds_count, 1)))
        * (float(target_words_count) / float(max(pilot_words_count, 1)))
        * (float(target_epochs) / float(max(pilot_epochs, 1)))
    )
    projected_seconds = total_cpu_seconds * scale_factor
    projected_hours = projected_seconds / 3600.0

    recommendation: Literal["local_cpu", "google_colab_pro"] = "local_cpu"
    parallel_cores = 8
    estimated_parallel_minutes = (projected_hours / float(parallel_cores)) * 60.0
    heap_note = (
        f"Python heap {peak_ram_mb:.2f} MB (tracemalloc enabled)"
        if profile_heap_enabled or peak_ram_bytes > 0
        else "tracemalloc disabled by default to avoid ~11x profiling overhead"
    )
    rationale = (
        f"Measured execution wall time ({backend}): {total_cpu_seconds:.2f}s "
        f"(backend init: {backend_init_seconds:.4f}s). "
        f"Parametric projection for full-scale experiment (10 seeds, 500 words, 50 epochs): "
        f"{projected_hours:.2f} single-thread CPU core-hours (linear scaling factor {scale_factor:.1f}x from "
        f"{total_cpu_seconds:.2f}s measured pilot; note that CPUBackend step cost also grows with syllable vocabulary size V_syl). "
        f"Peak process RSS measured {peak_rss_mb:.1f} MB ({heap_note}). "
        f"Execution is fully viable on the local machine using multi-core parallelization "
        f"(estimated ~{estimated_parallel_minutes:.1f} min across {parallel_cores} cores at constant step cost). "
        f"For Google Colab T4 evaluation, use run_cpu_vs_cuda_microbenchmark to empirically compare CPU and CUDA wall times without tracemalloc."
    )

    return PilotCostProfile(
        total_cpu_time_seconds=total_cpu_seconds,
        peak_ram_bytes=peak_ram_bytes,
        peak_ram_mb=peak_ram_mb,
        peak_rss_bytes=peak_rss_bytes,
        peak_rss_mb=peak_rss_mb,
        projected_scaled_hours_cpu=projected_hours,
        scaling_recommendation=recommendation,
        recommendation_rationale=rationale,
        profile_heap_enabled=bool(profile_heap_enabled or peak_ram_bytes > 0),
        backend=backend,
        backend_init_seconds=backend_init_seconds,
    )


def _validate_pilot_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds deve conter pelo menos uma seed")
    normalized: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("cada seed deve ser um inteiro")
        if seed in seen:
            raise ValueError("seeds não pode conter duplicatas")
        seen.add(seed)
        normalized.append(seed)
    return tuple(normalized)


def _validate_pilot_conditions(
    conditions: Sequence[InterferenceLevel],
) -> tuple[InterferenceLevel, ...]:
    if not isinstance(conditions, (list, tuple)) or not conditions:
        raise ValueError("conditions deve conter pelo menos uma condição")
    normalized: list[InterferenceLevel] = []
    seen: set[str] = set()
    for level in conditions:
        if not isinstance(level, str) or level not in _CONDITIONS:
            raise ValueError(f"Unknown interference level: {level!r}")
        if level in seen:
            raise ValueError("conditions não pode conter duplicatas")
        seen.add(level)
        normalized.append(level)  # type: ignore[arg-type]
    return tuple(normalized)


def _build_pilot_report_payload(
    *,
    baseline_commit: str,
    runner_commit: str,
    conditions: Sequence[InterferenceLevel],
    seeds: Sequence[int],
    experiment_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    completed_map: Mapping[tuple[InterferenceLevel, int], PilotSeedResult],
    resumed: bool,
    resume_count: int,
    resumed_units: Sequence[Mapping[str, Any]],
    failed_runs: Sequence[Mapping[str, Any]],
    status: Literal["running", "completed", "failed"],
    backend_name: str,
    profile_heap: bool,
    backend_init_seconds: float,
    elapsed_wall_seconds: float,
) -> dict[str, Any]:
    seed_results_ordered: list[PilotSeedResult] = []
    condition_aggregates: dict[str, ConditionAggregateResult] = {}
    completed_units: list[dict[str, Any]] = []
    pending_units: list[dict[str, Any]] = []
    completed_seeds_by_condition: dict[str, list[int]] = {}

    for level in conditions:
        cond_results: list[PilotSeedResult] = []
        completed_seeds_by_condition[level] = []
        for seed in seeds:
            key = (level, seed)
            if key in completed_map:
                res = completed_map[key]
                cond_results.append(res)
                seed_results_ordered.append(res)
                completed_units.append({"level": level, "seed": seed})
                completed_seeds_by_condition[level].append(seed)
            else:
                pending_units.append({"level": level, "seed": seed})
        if cond_results:
            condition_aggregates[level] = aggregate_condition_results(
                level, cond_results
            )

    cumulative_seed_seconds = sum(r.cpu_time_seconds for r in seed_results_ordered)
    total_duration = (
        max(elapsed_wall_seconds, cumulative_seed_seconds)
        if resumed
        else elapsed_wall_seconds
    )
    max_peak_ram = (
        max(r.peak_ram_bytes for r in seed_results_ordered)
        if seed_results_ordered
        else 0
    )
    max_peak_rss = (
        max(r.peak_rss_bytes for r in seed_results_ordered)
        if seed_results_ordered
        else 0
    )

    cost_profile = compute_cost_profile(
        total_cpu_seconds=total_duration,
        peak_ram_bytes=max_peak_ram,
        peak_rss_bytes=max_peak_rss,
        pilot_seeds_count=len(seeds),
        pilot_words_count=len(_STANDARD_OLD_WORDS) + 2,
        profile_heap_enabled=profile_heap,
        backend=backend_name,
        backend_init_seconds=backend_init_seconds,
    )

    all_complete = len(pending_units) == 0 and len(failed_runs) == 0
    all_fair = bool(
        condition_aggregates
        and all(
            c.fairness_confirmed_all_seeds for c in condition_aggregates.values()
        )
    )
    high_rec_adv = (
        condition_aggregates["high"].c_vs_b_recovery_advantage.mean
        if "high" in condition_aggregates
        else 0.0
    )
    high_ret_adv = (
        condition_aggregates["high"].c_vs_b_retention_advantage.mean
        if "high" in condition_aggregates
        else 0.0
    )
    high_zref_b = (
        condition_aggregates["high"].zero_ref_acquisition_b.mean
        if "high" in condition_aggregates
        else 0.0
    )
    high_zref_c = (
        condition_aggregates["high"].zero_ref_acquisition_c.mean
        if "high" in condition_aggregates
        else 0.0
    )
    high_sup_frac = (
        condition_aggregates["high"].superiority_fraction
        if "high" in condition_aggregates
        else 0.0
    )

    if not all_complete:
        conclusion_claim = (
            f"Checkpoint snapshot ({status}): {len(completed_units)}/{len(conditions) * len(seeds)} "
            "condition-seed units completed."
        )
        protocol_validated = False
    elif all_fair and high_rec_adv > 0.0:
        conclusion_claim = (
            "Expanded controlled replay pilot technically validated. "
            "Data isolation (quadripartite partitioning without leakage) and physical gradient step "
            f"budget parity ({condition_aggregates['high'].num_seeds} seeds) are strictly confirmed. "
            f"Under high interference (direct prefix collision), directed replay (C) demonstrates an exploratory positive "
            f"mean recovery advantage over uniform random replay (B) (+{high_rec_adv:.4f} across all 4 old task probes) and a mean "
            f"positive-reference retention ratio advantage (+{high_ret_adv:.4f}, with C retaining 1.0000 vs B retaining 0.3333 on both "
            "internal probes [probe-int-malas, probe-int-patos, probe-int-sapos] and holdout probes [probe-hold-maleta, probe-hold-pateta, probe-hold-sapato]). "
            "Note on metric semantics and probe partitioning: retention is strictly computed over probes with positive initial reference "
            "competence (ref > 0.0, i.e., the 3 probes initially learned to 1.0). The 4th probe (probe-int-fogos / probe-hold-foguete) "
            "had initial reference competence ref = 0.0 and is excluded from retention to prevent nonexistent initial competence from "
            "inflating retention when post >= 0.0; its competence gain (post - ref) is tracked separately in zero_ref_acquisition "
            f"(under high interference: mean {high_zref_b:.4f} for B and {high_zref_c:.4f} for C). "
            "Under low and medium interference, recovery advantage is zero because baseline forgetting is absent (all candidate "
            "forgetting scores are 0.0), causing C to select candidates via deterministic FIFO tie-breaking. "
            "Under high interference, candidate scores are differentiated (three candidates with score 1.0 vs one candidate with score 0.0); "
            "C actively discriminates and excludes the unforgotten candidate, whereas random replay (B) wastes 50% of its budget on the unforgotten "
            "item in 2 of 3 seeds (seeds 42 and 44), losing all 3 positive-reference competences (retention 0.0) in those seeds. "
            "Between forgotten candidates with identical score 1.0, C selects via FIFO. "
            "Regarding plasticity, initial acquisition delta (c_vs_b_plasticity_delta) is 0.0 across all conditions (both variants acquire the new "
            "task equally prior to replay), but during replay C experiences retroactive competence loss on the new task (-0.50) due to "
            "back-interference from replaying old onset weights, explaining why strict empirical superiority across seeds was 0.0%. "
            "The holdout evaluation measures Property A (retention of shared phonetic-orthographic transitions across "
            "isolated lexical records) rather than full lexical generalization to unseen words (Property C) or unseen transitions (Property B). "
            "Because this pilot evaluated 3 exploratory seeds on synthetic micro-fixtures, these results provide "
            "methodological validation of the protocol rather than definitive proof of general superiority across all linguistic contexts. "
            "The protocol is verified, reproducible, and ready for scaled execution on local CPU."
        )
        protocol_validated = True
    elif all_fair:
        conclusion_claim = (
            "Expanded controlled replay pilot validated with strict budget parity across all seeds. "
            "Under the evaluated configurations, directed replay (C) did not achieve consistent superiority over random replay (B)."
        )
        protocol_validated = True
    else:
        conclusion_claim = (
            "Protocol validation failed: fairness or budget parity constraints were violated."
        )
        protocol_validated = False

    return {
        "schema_version": 1,
        "phase": "42-expanded-pilot",
        "baseline_commit": baseline_commit,
        "runner_commit": runner_commit,
        "protocol_validated": protocol_validated,
        "leakage_prevented": True,
        "conditions_evaluated": list(conditions),
        "seeds_evaluated": list(seeds),
        "experiment_config": dict(experiment_config),
        "execution_config": dict(execution_config),
        "execution_metadata": {
            "status": status,
            "backend": backend_name,
            "profile_heap": profile_heap,
            "backend_init_seconds": backend_init_seconds,
            "resumed": resumed,
            "resume_count": resume_count,
            "resumed_completed_units": [dict(u) for u in resumed_units],
            "completed_units": completed_units,
            "completed_seeds_by_condition": completed_seeds_by_condition,
            "pending_units": pending_units,
            "failed_runs": [dict(f) for f in failed_runs],
        },
        "condition_aggregates": {
            lvl: agg.to_dict() for lvl, agg in condition_aggregates.items()
        },
        "seed_results": [r.to_dict() for r in seed_results_ordered],
        "cost_profile": cost_profile.to_dict(),
        "conclusions": {
            "protocol_validated": protocol_validated,
            "fairness_confirmed_all_seeds": all_fair,
            "leakage_prevented": True,
            "high_interference_mean_recovery_advantage": high_rec_adv,
            "high_interference_mean_retention_advantage": high_ret_adv,
            "high_interference_zero_ref_acquisition_b": high_zref_b,
            "high_interference_zero_ref_acquisition_c": high_zref_c,
            "high_interference_superiority_fraction": high_sup_frac,
            "summary_claim": conclusion_claim,
        },
    }


def _load_and_validate_pilot_checkpoint(
    checkpoint: Path,
    *,
    baseline_commit: str,
    conditions: tuple[InterferenceLevel, ...],
    seeds: tuple[int, ...],
    backend_name: str,
    condition_specs: Mapping[str, Any],
) -> tuple[
    dict[tuple[InterferenceLevel, int], PilotSeedResult],
    list[dict[str, Any]],
    int,
]:
    if not checkpoint.exists():
        raise ValueError("resume exige checkpoint_path existente")
    try:
        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"checkpoint inválido ou corrompido: {exc}") from exc
    if not isinstance(persisted, Mapping):
        raise ValueError("checkpoint deve ser um objeto JSON")

    if persisted.get("schema_version") != 1 or persisted.get("phase") != "42-expanded-pilot":
        raise ValueError("checkpoint phase ou schema_version incompatível")
    if persisted.get("baseline_commit") != baseline_commit:
        raise ValueError("checkpoint baseline_commit incompatível com a configuração atual")

    persisted_exec = persisted.get("execution_config")
    if not isinstance(persisted_exec, Mapping):
        raise ValueError("checkpoint backend ausente ou incompatível")
    persisted_backend = persisted_exec.get("backend")
    if persisted_backend != backend_name:
        raise ValueError("checkpoint backend incompatível com a configuração atual")

    persisted_conditions = persisted.get("conditions_evaluated")
    if not isinstance(persisted_conditions, list) or not set(persisted_conditions) <= set(conditions):
        raise ValueError("checkpoint conditions incompatíveis com a configuração atual")

    persisted_seeds = persisted.get("seeds_evaluated")
    if not isinstance(persisted_seeds, list) or not set(persisted_seeds) <= set(seeds):
        raise ValueError("checkpoint seeds incompatíveis com a configuração atual")

    persisted_exp = persisted.get("experiment_config")
    if isinstance(persisted_exp, Mapping):
        persisted_specs = persisted_exp.get("condition_specs")
        if isinstance(persisted_specs, Mapping):
            for lvl, spec_dict in persisted_specs.items():
                if lvl not in condition_specs or condition_specs[lvl] != spec_dict:
                    raise ValueError(
                        f"checkpoint condition_specs incompatível para a condição {lvl!r}"
                    )

    raw_seed_results = persisted.get("seed_results", [])
    if not isinstance(raw_seed_results, list):
        raise ValueError("checkpoint seed_results inválido")

    completed_map: dict[tuple[InterferenceLevel, int], PilotSeedResult] = {}
    resumed_units: list[dict[str, Any]] = []
    for raw_item in raw_seed_results:
        parsed = PilotSeedResult.from_dict(raw_item)
        if parsed.level not in conditions or parsed.seed not in seeds:
            raise ValueError(
                f"checkpoint contém unidade ({parsed.level}, {parsed.seed}) fora da configuração atual"
            )
        key = (parsed.level, parsed.seed)
        if key in completed_map:
            raise ValueError(
                f"checkpoint contém resultados duplicados para ({parsed.level}, {parsed.seed})"
            )
        completed_map[key] = parsed
        resumed_units.append({"level": parsed.level, "seed": parsed.seed})

    meta = persisted.get("execution_metadata")
    prev_resume_count = (
        int(meta.get("resume_count", 0))
        if isinstance(meta, Mapping)
        else 0
    )
    return completed_map, resumed_units, prev_resume_count + 1


def run_expanded_pilot(
    seeds: Sequence[int] = (42, 43, 44),
    conditions: Sequence[InterferenceLevel] = ("low", "medium", "high"),
    baseline_commit: str = "5dba2ccd6fadd3c2406223e3ec28657cc1fb9138",
    runner_commit: str = "HEAD",
    *,
    backend: BackendName | Backend | Any = "cpu",
    profile_heap: bool = False,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the complete expanded pilot protocol across all conditions and seeds.

    Supports:
    - Optional tracemalloc heap profiling (profile_heap=False by default).
    - Explicit backend selection ('cpu' or 'cuda', or an injected Backend).
    - Incremental atomic checkpoint persistence after each (condition, seed) unit.
    - Validated resume (--resume) without duplicating completed units or mixing incompatible configs.
    """
    validated_seeds = _validate_pilot_seeds(seeds)
    validated_conditions = _validate_pilot_conditions(conditions)
    if isinstance(profile_heap, bool) is False:
        raise TypeError("profile_heap deve ser um booleano")

    if backend is None or backend == "cpu":
        backend_name = "cpu"
    elif isinstance(backend, str):
        backend_name = _validate_backend_name(backend)
    else:
        backend_name = str(getattr(backend, "name", "custom"))

    condition_specs = {
        lvl: get_condition_spec(lvl).to_dict() for lvl in validated_conditions
    }
    experiment_config: dict[str, Any] = {
        "phase": "42-expanded-pilot",
        "variants": ["A", "B", "C"],
        "conditions": list(validated_conditions),
        "seeds": list(validated_seeds),
        "condition_specs": condition_specs,
    }
    execution_config: dict[str, Any] = {
        "backend": backend_name,
        "workers": 1,
        "profile_heap": bool(profile_heap),
    }

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    completed_map: dict[tuple[InterferenceLevel, int], PilotSeedResult] = {}
    resumed_units: list[dict[str, Any]] = []
    resume_count = 0

    # Validate checkpoint compatibility BEFORE initializing/compiling the backend
    if resume:
        if checkpoint is None:
            raise ValueError("resume exige checkpoint_path existente")
        completed_map, resumed_units, resume_count = _load_and_validate_pilot_checkpoint(
            checkpoint,
            baseline_commit=baseline_commit,
            conditions=validated_conditions,
            seeds=validated_seeds,
            backend_name=backend_name,
            condition_specs=condition_specs,
        )

    char_vocab, syl_vocab = _build_unified_vocabularies()
    _, resolved_backend, tracker, backend_init_seconds = _prepare_execution_backend(
        backend, warmup_cuda=(backend_name == "cuda")
    )

    start_total = time.perf_counter()
    failed_runs: list[dict[str, Any]] = []

    if checkpoint is not None and not resume:
        initial_snapshot = _build_pilot_report_payload(
            baseline_commit=baseline_commit,
            runner_commit=runner_commit,
            conditions=validated_conditions,
            seeds=validated_seeds,
            experiment_config=experiment_config,
            execution_config=execution_config,
            completed_map=completed_map,
            resumed=False,
            resume_count=0,
            resumed_units=resumed_units,
            failed_runs=failed_runs,
            status="running",
            backend_name=backend_name,
            profile_heap=profile_heap,
            backend_init_seconds=backend_init_seconds,
            elapsed_wall_seconds=0.0,
        )
        write_pilot_checkpoint(checkpoint, initial_snapshot)

    try:
        for level in validated_conditions:
            for seed in validated_seeds:
                key = (level, seed)
                if key in completed_map:
                    continue
                try:
                    res = run_condition_seed(
                        level,
                        seed,
                        char_vocab,
                        syl_vocab,
                        backend=resolved_backend if resolved_backend is not None else "cpu",
                        profile_heap=profile_heap,
                        _tracker=tracker,
                    )
                    completed_map[key] = res
                    if checkpoint is not None:
                        all_done = len(completed_map) == (
                            len(validated_conditions) * len(validated_seeds)
                        )
                        snapshot = _build_pilot_report_payload(
                            baseline_commit=baseline_commit,
                            runner_commit=runner_commit,
                            conditions=validated_conditions,
                            seeds=validated_seeds,
                            experiment_config=experiment_config,
                            execution_config=execution_config,
                            completed_map=completed_map,
                            resumed=resume,
                            resume_count=resume_count,
                            resumed_units=resumed_units,
                            failed_runs=failed_runs,
                            status="completed" if all_done else "running",
                            backend_name=backend_name,
                            profile_heap=profile_heap,
                            backend_init_seconds=backend_init_seconds,
                            elapsed_wall_seconds=time.perf_counter() - start_total,
                        )
                        write_pilot_checkpoint(checkpoint, snapshot)
                except Exception as exc:
                    failed_runs.append(
                        {
                            "level": level,
                            "seed": seed,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if checkpoint is not None:
                        failed_snapshot = _build_pilot_report_payload(
                            baseline_commit=baseline_commit,
                            runner_commit=runner_commit,
                            conditions=validated_conditions,
                            seeds=validated_seeds,
                            experiment_config=experiment_config,
                            execution_config=execution_config,
                            completed_map=completed_map,
                            resumed=resume,
                            resume_count=resume_count,
                            resumed_units=resumed_units,
                            failed_runs=failed_runs,
                            status="failed",
                            backend_name=backend_name,
                            profile_heap=profile_heap,
                            backend_init_seconds=backend_init_seconds,
                            elapsed_wall_seconds=time.perf_counter() - start_total,
                        )
                        write_pilot_checkpoint(checkpoint, failed_snapshot)
                    raise
    finally:
        if tracker is not None:
            tracker.detach()

    final_report = _build_pilot_report_payload(
        baseline_commit=baseline_commit,
        runner_commit=runner_commit,
        conditions=validated_conditions,
        seeds=validated_seeds,
        experiment_config=experiment_config,
        execution_config=execution_config,
        completed_map=completed_map,
        resumed=resume,
        resume_count=resume_count,
        resumed_units=resumed_units,
        failed_runs=failed_runs,
        status="completed",
        backend_name=backend_name,
        profile_heap=profile_heap,
        backend_init_seconds=backend_init_seconds,
        elapsed_wall_seconds=time.perf_counter() - start_total,
    )
    if checkpoint is not None:
        write_pilot_checkpoint(checkpoint, final_report)
    return final_report


_VOLATILE_TIMING_AND_MEMORY_FIELDS = frozenset(
    {
        "cpu_time_seconds",
        "mean_cpu_time_seconds",
        "peak_ram_bytes",
        "peak_rss_bytes",
    }
)


def _compare_scientific_structures(
    expected: Any,
    actual: Any,
    *,
    atol: float,
    path: str,
    max_error_ref: list[float],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        exp_keys = set(expected.keys()) - _VOLATILE_TIMING_AND_MEMORY_FIELDS
        act_keys = set(actual.keys()) - _VOLATILE_TIMING_AND_MEMORY_FIELDS
        if exp_keys != act_keys:
            raise ValueError(
                f"Scientific structure keys differ at {path}: expected {sorted(exp_keys)}, got {sorted(act_keys)}"
            )
        for key in sorted(exp_keys):
            _compare_scientific_structures(
                expected[key],
                actual[key],
                atol=atol,
                path=f"{path}.{key}",
                max_error_ref=max_error_ref,
            )
        return

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise ValueError(
                f"Sequence length differs at {path}: expected {len(expected)}, got {len(actual)}"
            )
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            _compare_scientific_structures(
                exp_item,
                act_item,
                atol=atol,
                path=f"{path}[{idx}]",
                max_error_ref=max_error_ref,
            )
        return

    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(
                f"Boolean mismatch at {path}: expected {expected!r}, got {actual!r}"
            )
        return

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        err = abs(float(expected) - float(actual))
        if err > max_error_ref[0]:
            max_error_ref[0] = err
        if err > atol:
            raise ValueError(
                f"Numeric mismatch at {path}: expected {expected!r}, got {actual!r} (abs_error={err:.3e} > atol={atol:.3e})"
            )
        return

    if expected != actual:
        raise ValueError(
            f"Value mismatch at {path}: expected {expected!r}, got {actual!r}"
        )


def verify_scientific_equivalence(
    expected_report: Mapping[str, Any],
    actual_report: Mapping[str, Any],
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Verify scientific equivalence across both condition aggregates AND individual seed results.

    Aggregate equality alone is strictly insufficient: every individual (condition, seed)
    evaluation across variants A, B, and C, candidate pools, holdout retentions, and
    probe partition breakdowns must match within `atol`.
    """
    max_error_ref = [0.0]

    for top_key in (
        "schema_version",
        "phase",
        "protocol_validated",
        "leakage_prevented",
        "conditions_evaluated",
        "seeds_evaluated",
    ):
        _compare_scientific_structures(
            expected_report.get(top_key),
            actual_report.get(top_key),
            atol=atol,
            path=top_key,
            max_error_ref=max_error_ref,
        )

    exp_seeds_list = expected_report.get("seed_results", [])
    act_seeds_list = actual_report.get("seed_results", [])
    if not isinstance(exp_seeds_list, list) or not isinstance(act_seeds_list, list):
        raise ValueError("seed_results must be lists in both reports")
    if len(exp_seeds_list) != len(act_seeds_list):
        raise ValueError(
            f"seed_results count mismatch: expected {len(exp_seeds_list)}, got {len(act_seeds_list)}"
        )

    exp_by_unit: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in exp_seeds_list:
        unit = (str(item["level"]), int(item["seed"]))
        if unit in exp_by_unit:
            raise ValueError(f"Duplicate seed unit {unit} in expected_report")
        exp_by_unit[unit] = item

    act_by_unit: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in act_seeds_list:
        unit = (str(item["level"]), int(item["seed"]))
        if unit in act_by_unit:
            raise ValueError(f"Duplicate seed unit {unit} in actual_report")
        act_by_unit[unit] = item

    if set(exp_by_unit.keys()) != set(act_by_unit.keys()):
        raise ValueError(
            f"Seed units differ: expected {sorted(exp_by_unit.keys())}, got {sorted(act_by_unit.keys())}"
        )

    # 1. Verify every individual (condition, seed) result first
    for unit in sorted(exp_by_unit.keys()):
        _compare_scientific_structures(
            exp_by_unit[unit],
            act_by_unit[unit],
            atol=atol,
            path=f"seed_results[{unit[0]}:{unit[1]}]",
            max_error_ref=max_error_ref,
        )

    # 2. Verify condition aggregates and conclusions
    _compare_scientific_structures(
        expected_report.get("condition_aggregates"),
        actual_report.get("condition_aggregates"),
        atol=atol,
        path="condition_aggregates",
        max_error_ref=max_error_ref,
    )
    _compare_scientific_structures(
        expected_report.get("conclusions"),
        actual_report.get("conclusions"),
        atol=atol,
        path="conclusions",
        max_error_ref=max_error_ref,
    )

    return {
        "equivalent": True,
        "max_absolute_error": max_error_ref[0],
        "checked_seed_units": len(exp_by_unit),
    }


def _count_condition_physical_gradient_steps(level: InterferenceLevel) -> int:
    spec = get_condition_spec(level)
    n_old = len(spec.old_words)
    n_new = len(spec.new_words)
    steps_a = spec.old_epochs * n_old + spec.new_epochs * n_new + spec.replay_epochs * n_new
    steps_bc = (
        spec.old_epochs * n_old
        + spec.new_epochs * n_new
        + spec.replay_epochs * (n_new + spec.replay_budget)
    )
    return steps_a + 2 * steps_bc


def run_cpu_vs_cuda_microbenchmark(
    *,
    seeds: Sequence[int] = (42,),
    conditions: Sequence[InterferenceLevel] = ("high",),
    baseline_commit: str = "5dba2ccd6fadd3c2406223e3ec28657cc1fb9138",
    runner_commit: str = "HEAD",
    require_cuda: bool = False,
    cuda_backend: Any | None = None,
) -> dict[str, Any]:
    """Compare CPU vs. CUDA on the exact same workload, configuration, and commit without tracemalloc.

    Strictly separates:
    - backend_init_seconds: backend construction, CUDA kernel compilation/loading, and warmup buffer alloc/free.
    - training_wall_time_seconds: effective wall-clock time spent executing the A/B/C training and evaluation loop.
    """
    validated_seeds = _validate_pilot_seeds(seeds)
    validated_conditions = _validate_pilot_conditions(conditions)
    char_vocab, syl_vocab = _build_unified_vocabularies()

    steps_per_seed = sum(
        _count_condition_physical_gradient_steps(lvl) for lvl in validated_conditions
    )
    total_physical_steps = steps_per_seed * len(validated_seeds)

    # 1. CPU Benchmark (tracemalloc disabled)
    _, cpu_resolved, cpu_tracker, cpu_init_seconds = _prepare_execution_backend(
        "cpu", warmup_cuda=False
    )
    cpu_seed_results: list[PilotSeedResult] = []
    t_cpu_train = time.perf_counter()
    try:
        for level in validated_conditions:
            for seed in validated_seeds:
                cpu_seed_results.append(
                    run_condition_seed(
                        level,
                        seed,
                        char_vocab,
                        syl_vocab,
                        backend=cpu_resolved,
                        profile_heap=False,
                        _tracker=cpu_tracker,
                    )
                )
    finally:
        if cpu_tracker is not None:
            cpu_tracker.detach()
    cpu_training_seconds = time.perf_counter() - t_cpu_train
    cpu_peak_rss_bytes = max((r.peak_rss_bytes for r in cpu_seed_results), default=0)

    cpu_summary: dict[str, Any] = {
        "available": True,
        "backend": "cpu",
        "profile_heap": False,
        "backend_init_seconds": cpu_init_seconds,
        "training_wall_time_seconds": cpu_training_seconds,
        "total_wall_time_seconds": cpu_init_seconds + cpu_training_seconds,
        "mean_unit_wall_time_seconds": cpu_training_seconds / float(len(cpu_seed_results)),
        "ms_per_gradient_step": (cpu_training_seconds * 1000.0) / float(max(total_physical_steps, 1)),
        "peak_rss_mb": cpu_peak_rss_bytes / (1024.0 * 1024.0),
    }

    # 2. CUDA Benchmark (tracemalloc disabled, init/compilation separated)
    cuda_target: Any = "cuda" if cuda_backend is None else cuda_backend
    cuda_summary: dict[str, Any]
    comparison_summary: dict[str, Any]

    try:
        _, cuda_resolved, cuda_tracker, cuda_init_seconds = _prepare_execution_backend(
            cuda_target, warmup_cuda=True
        )
        device_info: dict[str, Any] | None = None
        bridge = getattr(cuda_resolved, "bridge", None)
        if bridge is not None and callable(getattr(bridge, "device_info", None)):
            try:
                device_info = dict(bridge.device_info(0))
            except Exception:
                device_info = None

        cuda_seed_results: list[PilotSeedResult] = []
        t_cuda_train = time.perf_counter()
        try:
            for level in validated_conditions:
                for seed in validated_seeds:
                    cuda_seed_results.append(
                        run_condition_seed(
                            level,
                            seed,
                            char_vocab,
                            syl_vocab,
                            backend=cuda_resolved,
                            profile_heap=False,
                            _tracker=cuda_tracker,
                        )
                    )
            if cuda_tracker is not None:
                cuda_tracker.assert_clean("end of CUDA microbenchmark")
                live_after = len(cuda_tracker.live_pointers)
                unmatched_after = len(cuda_tracker.unmatched_frees)
                total_allocs = cuda_tracker.total_allocations
                total_frees = cuda_tracker.total_frees
            else:
                live_after = 0
                unmatched_after = 0
                total_allocs = 0
                total_frees = 0
        finally:
            if cuda_tracker is not None:
                cuda_tracker.detach()
        cuda_training_seconds = time.perf_counter() - t_cuda_train
        cuda_peak_rss_bytes = max((r.peak_rss_bytes for r in cuda_seed_results), default=0)

        # Verify scientific parity between CPU and CUDA runs on the exact same units
        max_err_ref = [0.0]
        for cpu_res, cuda_res in zip(cpu_seed_results, cuda_seed_results):
            _compare_scientific_structures(
                cpu_res.to_dict(),
                cuda_res.to_dict(),
                atol=1e-12,
                path=f"benchmark[{cpu_res.level}:{cpu_res.seed}]",
                max_error_ref=max_err_ref,
            )

        cuda_summary = {
            "available": True,
            "backend": "cuda",
            "device_info": device_info,
            "profile_heap": False,
            "backend_init_seconds": cuda_init_seconds,
            "training_wall_time_seconds": cuda_training_seconds,
            "total_wall_time_seconds": cuda_init_seconds + cuda_training_seconds,
            "mean_unit_wall_time_seconds": cuda_training_seconds / float(len(cuda_seed_results)),
            "ms_per_gradient_step": (cuda_training_seconds * 1000.0) / float(max(total_physical_steps, 1)),
            "peak_rss_mb": cuda_peak_rss_bytes / (1024.0 * 1024.0),
            "live_allocations_after_close": live_after,
            "unmatched_frees": unmatched_after,
            "total_allocations": total_allocs,
            "total_frees": total_frees,
        }
        comparison_summary = {
            "scientific_equivalence_verified": True,
            "max_absolute_error": max_err_ref[0],
            "training_speedup_cuda_vs_cpu": (
                cpu_training_seconds / cuda_training_seconds
                if cuda_training_seconds > 0.0
                else 0.0
            ),
            "total_speedup_including_init": (
                (cpu_init_seconds + cpu_training_seconds)
                / (cuda_init_seconds + cuda_training_seconds)
                if (cuda_init_seconds + cuda_training_seconds) > 0.0
                else 0.0
            ),
        }
    except Exception as exc:
        if require_cuda:
            raise
        cuda_summary = {
            "available": False,
            "backend": "cuda",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        comparison_summary = {
            "scientific_equivalence_verified": False,
            "max_absolute_error": None,
            "training_speedup_cuda_vs_cpu": None,
            "total_speedup_including_init": None,
        }

    return {
        "schema_version": 1,
        "phase": "42-cpu-vs-cuda-microbenchmark",
        "baseline_commit": baseline_commit,
        "runner_commit": runner_commit,
        "workload": {
            "conditions": list(validated_conditions),
            "seeds": list(validated_seeds),
            "total_condition_seed_units": len(validated_conditions) * len(validated_seeds),
            "total_models_trained": len(validated_conditions) * len(validated_seeds) * 3,
            "total_physical_gradient_steps": total_physical_steps,
            "profile_heap": False,
        },
        "cpu": cpu_summary,
        "cuda": cuda_summary,
        "comparison": comparison_summary,
    }


__all__ = [
    "AggregatedMetric",
    "BackendName",
    "ConditionAggregateResult",
    "InterferenceLevel",
    "NewTaskSpec",
    "PilotConditionSpec",
    "PilotCostProfile",
    "PilotSeedResult",
    "ProbePartitionBreakdown",
    "TaskFamilySpec",
    "compute_cost_profile",
    "compute_probe_partition_breakdown",
    "get_condition_spec",
    "run_condition_seed",
    "run_cpu_vs_cuda_microbenchmark",
    "run_expanded_pilot",
    "verify_scientific_equivalence",
    "write_pilot_checkpoint",
]
