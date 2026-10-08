"""One-shot, controlled replay primitives for experiments only.

Candidate snapshots, selectors, plans, and accounting live outside the trainer.
Selection never mutates a ``LearningMemory``; callers execute the resolved plan
through the existing trainer and keep the source memory frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from kinera.experiments.learning_memory import ExampleRef, LearningMemory
from kinera.experiments.learning_state import LearningState

ReplaySource = Literal["new", "replay"]


def _validate_ref_sequence(refs: Sequence[ExampleRef], name: str) -> tuple[ExampleRef, ...]:
    if isinstance(refs, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of ExampleRef values")
    result = tuple(refs)
    if any(not isinstance(ref, ExampleRef) for ref in result):
        raise TypeError(f"{name} must contain only ExampleRef values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} references must be unique")
    return result


def _validate_budget(budget: int, candidate_count: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer")
    if budget < 0 or budget > candidate_count:
        raise ValueError("budget must be between zero and the candidate count")


def _validate_candidate_pool(
    candidates: Sequence[ReplayCandidate],
) -> tuple[ReplayCandidate, ...]:
    pool = tuple(candidates)
    if any(not isinstance(candidate, ReplayCandidate) for candidate in pool):
        raise TypeError("candidate pool must contain only ReplayCandidate values")
    refs = tuple(candidate.example_ref for candidate in pool)
    if len(set(refs)) != len(refs):
        raise ValueError("candidate refs must be unique")
    return pool


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    """One eligible memory reference and its quantitative L3 priority."""

    example_ref: ExampleRef
    group_ids: tuple[str, ...]
    priority_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.example_ref, ExampleRef):
            raise TypeError("example_ref must be an ExampleRef")
        if not isinstance(self.group_ids, tuple) or not self.group_ids:
            raise ValueError("group_ids must be a non-empty tuple")
        if any(not isinstance(group, str) or not group.strip() for group in self.group_ids):
            raise ValueError("group_ids must contain non-empty strings")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("group_ids must be unique")
        if isinstance(self.priority_score, bool) or not isinstance(
            self.priority_score, (int, float)
        ):
            raise TypeError("priority_score must be a real scalar")
        if not math.isfinite(self.priority_score):
            raise ValueError("priority_score must be finite")
        if self.priority_score < 0.0:
            raise ValueError("priority_score must be non-negative")


def build_replay_candidates(
    memory: LearningMemory,
    learning_states: Mapping[str, LearningState],
) -> tuple[ReplayCandidate, ...]:
    """Freeze FIFO-ordered eligible memory entries into scored candidates.

    Every entry must have explicit event group associations and every such
    group must have a quantitative, finite, non-negative ``forgetting_score``.
    A candidate linked to multiple groups receives their maximum score.
    """
    if not isinstance(memory, LearningMemory):
        raise TypeError("memory must be a LearningMemory")
    if not isinstance(learning_states, Mapping):
        raise TypeError("learning_states must be a mapping by group_id")
    candidates: list[ReplayCandidate] = []
    for entry in memory.entries():
        group_ids = entry.latest_event.group_ids
        if not group_ids:
            raise ValueError(
                f"memory entry {entry.example_ref!r} has no explicit group association"
            )
        scores: list[float] = []
        for group_id in group_ids:
            if group_id not in learning_states:
                raise KeyError(f"missing LearningState for group {group_id!r}")
            state = learning_states[group_id]
            if not isinstance(state, LearningState):
                raise TypeError(f"state for group {group_id!r} must be a LearningState")
            if state.group_id != group_id:
                raise ValueError(f"LearningState key/group_id mismatch for {group_id!r}")
            score = state.forgetting_score
            if score is None:
                raise ValueError(f"forgetting_score is unavailable for group {group_id!r}")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"forgetting_score for group {group_id!r} must be numeric")
            if not math.isfinite(score) or score < 0.0:
                raise ValueError(
                    f"forgetting_score for group {group_id!r} must be finite and non-negative"
                )
            scores.append(float(score))
        candidates.append(
            ReplayCandidate(entry.example_ref, tuple(group_ids), max(scores))
        )
    return tuple(candidates)


def select_random_replay(
    candidates: Sequence[ReplayCandidate],
    budget: int,
    *,
    replay_seed: int,
) -> tuple[ReplayCandidate, ...]:
    """Select a seed-reproducible subset without replacement.

    A private ``random.Random`` is constructed only when this selector is
    called. Returned candidates retain original candidate-pool order, and
    priority scores do not participate in selection.
    """
    pool = _validate_candidate_pool(candidates)
    _validate_budget(budget, len(pool))
    if isinstance(replay_seed, bool) or not isinstance(replay_seed, int):
        raise TypeError("replay_seed must be an integer")
    import random

    rng = random.Random(replay_seed)
    chosen_indices = set(rng.sample(range(len(pool)), budget))
    return tuple(candidate for index, candidate in enumerate(pool) if index in chosen_indices)


def select_directed_replay(
    candidates: Sequence[ReplayCandidate],
    budget: int,
) -> tuple[ReplayCandidate, ...]:
    """Select highest forgetting scores; ties resolve by pool order, without RNG."""
    pool = _validate_candidate_pool(candidates)
    _validate_budget(budget, len(pool))
    ranked_indices = sorted(
        range(len(pool)), key=lambda index: (-pool[index].priority_score, index)
    )
    chosen_indices = set(ranked_indices[:budget])
    return tuple(candidate for index, candidate in enumerate(pool) if index in chosen_indices)


@dataclass(frozen=True, slots=True)
class ReplayPlanItem:
    """Auditable source/ref for one actual update in the interleaved plan."""

    source: ReplaySource
    example_ref: ExampleRef

    def __post_init__(self) -> None:
        if self.source not in ("new", "replay"):
            raise ValueError("source must be exactly 'new' or 'replay'")
        if not isinstance(self.example_ref, ExampleRef):
            raise TypeError("example_ref must be an ExampleRef")


def build_interleaved_plan(
    new_example_refs: Sequence[ExampleRef],
    replay_example_refs: Sequence[ExampleRef],
    *,
    replay_positions: Sequence[int],
) -> tuple[ReplayPlanItem, ...]:
    """Interleave refs at zero-based final-plan positions.

    New refs retain their supplied order and selected replay refs retain their
    candidate-pool order. ``replay_positions`` must name exactly the replay
    slots in the complete output plan; it is frozen before execution.
    """
    new_refs = _validate_ref_sequence(new_example_refs, "new_example_refs")
    replay_refs = _validate_ref_sequence(replay_example_refs, "replay_example_refs")
    positions = tuple(replay_positions)
    if len(positions) != len(replay_refs):
        raise ValueError("replay_positions count must equal the replay budget")
    if any(isinstance(position, bool) or not isinstance(position, int) for position in positions):
        raise TypeError("replay_positions must contain integers")
    if len(set(positions)) != len(positions):
        raise ValueError("replay_positions must be unique")
    total = len(new_refs) + len(replay_refs)
    if any(position < 0 or position >= total for position in positions):
        raise ValueError("replay position is outside the final plan")
    if positions != tuple(sorted(positions)):
        raise ValueError("replay_positions must be strictly increasing")
    replay_at = {position: ref for position, ref in zip(positions, replay_refs)}
    plan: list[ReplayPlanItem] = []
    new_index = 0
    for position in range(total):
        if position in replay_at:
            plan.append(ReplayPlanItem("replay", replay_at[position]))
        else:
            plan.append(ReplayPlanItem("new", new_refs[new_index]))
            new_index += 1
    return tuple(plan)


def resolve_replay_examples(
    plan: Sequence[ReplayPlanItem],
    aligned_refs: Sequence[ExampleRef],
    aligned_examples: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Resolve every plan ref exactly once and preserve plan order."""
    plan_items = tuple(plan)
    if any(not isinstance(item, ReplayPlanItem) for item in plan_items):
        raise TypeError("plan must contain only ReplayPlanItem values")
    plan_refs = tuple(item.example_ref for item in plan_items)
    if len(set(plan_refs)) != len(plan_refs):
        raise ValueError("plan example refs must be unique")
    refs = _validate_ref_sequence(aligned_refs, "aligned_refs")
    examples = tuple(aligned_examples)
    if len(refs) != len(examples):
        raise ValueError("aligned_refs and aligned_examples must have the same length")
    by_ref: dict[ExampleRef, dict[str, Any]] = {}
    for ref, example in zip(refs, examples):
        if not isinstance(example, dict):
            raise TypeError("each aligned example must be a dict")
        by_ref[ref] = example
    resolved: list[dict[str, Any]] = []
    for item in plan_items:
        try:
            resolved.append(by_ref[item.example_ref])
        except KeyError as exc:
            raise KeyError(f"missing aligned example for plan ref: {item.example_ref!r}") from exc
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class ReplayAccounting:
    """Update counts and selected refs kept separate from the L2 LearningState."""

    new_updates: int
    replay_updates: int
    total_updates: int
    selected_refs: tuple[ExampleRef, ...] = ()
    replay_seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("new_updates", "replay_updates", "total_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_updates != self.new_updates + self.replay_updates:
            raise ValueError("total_updates must equal new_updates + replay_updates")
        refs = _validate_ref_sequence(self.selected_refs, "selected_refs")
        if refs != self.selected_refs:
            raise TypeError("selected_refs must be a tuple")
        if len(refs) != self.replay_updates:
            raise ValueError("selected_refs count must equal replay_updates")
        if self.replay_seed is not None and (
            isinstance(self.replay_seed, bool) or not isinstance(self.replay_seed, int)
        ):
            raise TypeError("replay_seed must be an integer or None")


__all__ = [
    "ReplayAccounting",
    "ReplayCandidate",
    "ReplayPlanItem",
    "ReplaySource",
    "build_interleaved_plan",
    "build_replay_candidates",
    "resolve_replay_examples",
    "select_directed_replay",
    "select_random_replay",
]
