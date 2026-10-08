"""Threshold-free experimental learning state for L2.

Statuses are supplied by the experiment harness; this module does not infer
risk states or trigger replay. ``competence_delta`` is a direction-oriented
difference between two observations, not a time derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from kinera.experiments.retention import MetricDirection, ProbeObservation

LearningStatus = Literal[
    "NEW",
    "LEARNING",
    "STABLE",
    "AT_RISK",
    "FORGOTTEN",
    "UNRESOLVED",
]
_VALID_STATUSES = (
    "NEW",
    "LEARNING",
    "STABLE",
    "AT_RISK",
    "FORGOTTEN",
    "UNRESOLVED",
)
_VALID_DIRECTIONS = ("higher_is_better", "lower_is_better")


def _finite_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_direction(direction: object) -> None:
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            "direction must be 'higher_is_better' or 'lower_is_better'"
        )


def calculate_forgetting(
    reference_competence: float,
    current_competence: float,
    direction: MetricDirection,
) -> float:
    """Return a non-negative degradation signal, without causal classification."""
    reference = _finite_value(reference_competence, "reference_competence")
    current = _finite_value(current_competence, "current_competence")
    _validate_direction(direction)
    if direction == "higher_is_better":
        score = max(0.0, reference - current)
    else:
        score = max(0.0, current - reference)
    if not math.isfinite(score):
        raise ValueError("forgetting_score must be finite")
    return score


def oriented_competence_delta(
    previous_competence: float,
    current_competence: float,
    direction: MetricDirection,
) -> float:
    """Return an oriented difference; positive means competence improved."""
    previous = _finite_value(previous_competence, "previous_competence")
    current = _finite_value(current_competence, "current_competence")
    _validate_direction(direction)
    delta = (
        current - previous
        if direction == "higher_is_better"
        else previous - current
    )
    if not math.isfinite(delta):
        raise ValueError("competence_delta must be finite")
    return delta


@dataclass(frozen=True, slots=True)
class LearningState:
    """Harness-managed state; no threshold-based status transition is implied."""

    group_id: str
    reference_competence: float | None
    current_competence: float | None
    forgetting_score: float | None
    evidence_count: int
    replay_count: int
    last_practiced_step: int | None
    status: LearningStatus
    competence_delta: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str):
            raise TypeError("group_id must be a string")
        if not self.group_id.strip():
            raise ValueError("group_id must be non-empty")
        for name in (
            "reference_competence",
            "current_competence",
            "forgetting_score",
            "competence_delta",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_value(value, name)
        if self.forgetting_score is not None and self.forgetting_score < 0.0:
            raise ValueError("forgetting_score must be non-negative")
        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(self.evidence_count, int)
            or self.evidence_count < 0
        ):
            raise ValueError("evidence_count must be a non-negative integer")
        if isinstance(self.replay_count, bool) or not isinstance(self.replay_count, int):
            raise TypeError("replay_count must be an integer")
        if self.replay_count != 0:
            raise ValueError("replay_count must remain zero in L2")
        if self.last_practiced_step is not None and (
            isinstance(self.last_practiced_step, bool)
            or not isinstance(self.last_practiced_step, int)
            or self.last_practiced_step < 1
        ):
            raise ValueError("last_practiced_step must be a positive integer or None")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES!r}")


def build_learning_state(
    reference: ProbeObservation,
    current: ProbeObservation,
    *,
    status: LearningStatus,
    last_practiced_step: int | None,
) -> LearningState:
    """Build a two-evidence state; status is explicit and never thresholded."""
    if reference is current:
        raise ValueError("reference and current must be distinct observations")
    if not isinstance(reference, ProbeObservation) or not isinstance(
        current, ProbeObservation
    ):
        raise TypeError("reference and current must be ProbeObservation values")
    if reference.group_id != current.group_id:
        raise ValueError("reference/current group_id must match")
    if reference.example_refs != current.example_refs:
        raise ValueError("reference/current example_refs must match")
    if reference.primary_metric != current.primary_metric:
        raise ValueError("reference/current primary_metric must match")
    if reference.direction != current.direction:
        raise ValueError("reference/current direction must match")
    return LearningState(
        group_id=reference.group_id,
        reference_competence=reference.competence,
        current_competence=current.competence,
        forgetting_score=calculate_forgetting(
            reference.competence,
            current.competence,
            reference.direction,
        ),
        evidence_count=2,
        replay_count=0,
        last_practiced_step=last_practiced_step,
        status=status,
        competence_delta=oriented_competence_delta(
            reference.competence,
            current.competence,
            reference.direction,
        ),
    )


__all__ = [
    "LearningState",
    "LearningStatus",
    "build_learning_state",
    "calculate_forgetting",
    "oriented_competence_delta",
]
