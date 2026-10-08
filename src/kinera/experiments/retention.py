"""Evaluation-only retention probes for experimental learning state.

Probe examples are resolved from aligned references at call time. This module
never stores datasets and never admits probes to the training memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence, TYPE_CHECKING

from kinera.experiments.learning_memory import ExampleRef

if TYPE_CHECKING:
    from kinera.language.vocabulary import CharacterVocabulary
    from kinera.nn.losses.multitask import MultiTaskLoss
    from kinera.nn.models.hybrid import AutoregressiveModel


MetricDirection = Literal["higher_is_better", "lower_is_better"]
_VALID_DIRECTIONS = ("higher_is_better", "lower_is_better")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_direction(direction: object) -> None:
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            "direction must be 'higher_is_better' or 'lower_is_better'"
        )


@dataclass(frozen=True, slots=True)
class RetentionProbeGroup:
    """Explicit, immutable set of holdout refs and one evaluator metric."""

    group_id: str
    example_refs: tuple[ExampleRef, ...]
    primary_metric: str
    direction: MetricDirection

    def __post_init__(self) -> None:
        _non_empty_string(self.group_id, "group_id")
        _non_empty_string(self.primary_metric, "primary_metric")
        if not isinstance(self.example_refs, tuple):
            raise TypeError("example_refs must be a tuple")
        if not self.example_refs:
            raise ValueError("example_refs must not be empty")
        if any(not isinstance(ref, ExampleRef) for ref in self.example_refs):
            raise TypeError("example_refs must contain only ExampleRef values")
        if len(set(self.example_refs)) != len(self.example_refs):
            raise ValueError("example_refs must be unique")
        _validate_direction(self.direction)


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    """One scalar result tied to the exact immutable probe population."""

    group_id: str
    example_refs: tuple[ExampleRef, ...]
    primary_metric: str
    direction: MetricDirection
    competence: float
    support: int

    def __post_init__(self) -> None:
        _non_empty_string(self.group_id, "group_id")
        _non_empty_string(self.primary_metric, "primary_metric")
        _validate_direction(self.direction)
        if not isinstance(self.example_refs, tuple):
            raise TypeError("example_refs must be a tuple")
        if not self.example_refs:
            raise ValueError("example_refs must not be empty")
        if any(not isinstance(ref, ExampleRef) for ref in self.example_refs):
            raise TypeError("example_refs must contain only ExampleRef values")
        if len(set(self.example_refs)) != len(self.example_refs):
            raise ValueError("example_refs must be unique")
        if isinstance(self.competence, bool) or not isinstance(
            self.competence, (int, float)
        ):
            raise TypeError("competence must be a real scalar")
        if not math.isfinite(self.competence):
            raise ValueError("competence must be finite")
        if (
            isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support <= 0
        ):
            raise ValueError("support must be a positive integer")
        if self.support != len(self.example_refs):
            raise ValueError("support must equal the number of example_refs")


def resolve_probe_examples(
    group: RetentionProbeGroup,
    aligned_refs: Sequence[ExampleRef],
    aligned_examples: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Resolve exactly one example per ref, preserving group order.

    The aligned sequences may be a small probe subset; no resolved dataset is
    retained after this call. Duplicate alignment refs are rejected rather
    than silently overwritten as they would be in a dict conversion.
    """
    if not isinstance(group, RetentionProbeGroup):
        raise TypeError("group must be a RetentionProbeGroup")
    refs = tuple(aligned_refs)
    examples = tuple(aligned_examples)
    if len(refs) != len(examples):
        raise ValueError("aligned_refs and aligned_examples must have the same length")
    by_ref: dict[ExampleRef, dict[str, Any]] = {}
    for ref, example in zip(refs, examples):
        if not isinstance(ref, ExampleRef):
            raise TypeError("aligned_refs must contain only ExampleRef values")
        if ref in by_ref:
            raise ValueError(f"duplicate aligned ExampleRef: {ref!r}")
        if not isinstance(example, dict):
            raise TypeError("each aligned example must be a dict")
        by_ref[ref] = example
    resolved: list[dict[str, Any]] = []
    for ref in group.example_refs:
        try:
            resolved.append(by_ref[ref])
        except KeyError as exc:
            raise KeyError(f"missing aligned example for probe ref: {ref!r}") from exc
    return tuple(resolved)


def evaluate_retention_probe(
    model: AutoregressiveModel,
    group: RetentionProbeGroup,
    aligned_refs: Sequence[ExampleRef],
    aligned_examples: Sequence[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    loss_function: MultiTaskLoss | None = None,
) -> ProbeObservation:
    """Evaluate a probe through the existing evaluator, with no training path."""
    resolved = resolve_probe_examples(group, aligned_refs, aligned_examples)
    # Keep evaluator/backend imports out of the experimental module import path.
    from kinera.training.evaluator import evaluate_multitask_examples

    report = evaluate_multitask_examples(
        model,
        list(resolved),
        vocabulary,
        syllable_unknown_id=syllable_unknown_id,
        loss_function=loss_function,
    )
    if not isinstance(report, Mapping):
        raise TypeError("evaluator report must be a mapping")
    if group.primary_metric == "examples":
        raise ValueError("examples is support metadata, not a competence metric")
    if group.primary_metric not in report:
        raise KeyError(
            f"primary_metric {group.primary_metric!r} is absent from evaluator report"
        )
    competence = report[group.primary_metric]
    if isinstance(competence, bool) or not isinstance(competence, (int, float)):
        raise TypeError(
            f"primary_metric {group.primary_metric!r} must resolve to a scalar"
        )
    if not math.isfinite(competence):
        raise ValueError(
            f"primary_metric {group.primary_metric!r} must be finite"
        )
    support = report.get("examples")
    if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
        raise ValueError("evaluator report must provide positive integer examples support")
    if support != len(resolved):
        raise ValueError("evaluator support does not match resolved probe examples")
    return ProbeObservation(
        group_id=group.group_id,
        example_refs=group.example_refs,
        primary_metric=group.primary_metric,
        direction=group.direction,
        competence=float(competence),
        support=support,
    )


__all__ = [
    "MetricDirection",
    "ProbeObservation",
    "RetentionProbeGroup",
    "evaluate_retention_probe",
    "resolve_probe_examples",
]
