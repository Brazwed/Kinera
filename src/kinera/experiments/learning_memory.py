"""Passive, bounded learning-event recording for experiments.

This module deliberately owns no model, backend, dataset, clock, RNG, or replay
state. ``entries()`` and ``select()`` return entries oldest-first by first
insertion; ``select(limit)`` is deterministic retrieval only.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExampleRef:
    """Stable opaque identity for one example within an artifact split."""

    dataset_artifact_id: str
    split: str
    example_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_artifact_id, str) or not self.dataset_artifact_id.strip():
            raise ValueError("dataset_artifact_id must be a non-empty string")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        if (
            isinstance(self.example_index, bool)
            or not isinstance(self.example_index, int)
            or self.example_index < 0
        ):
            raise ValueError("example_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class LearningEvent:
    """Immutable scalar-only observation of one completed training update."""

    example_ref: ExampleRef
    epoch: int
    step: int
    observation_phase: str
    char_loss: float
    syllable_loss: float
    boundary_loss: float
    total_loss: float
    group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.example_ref, ExampleRef):
            raise TypeError("example_ref must be an ExampleRef")
        for name, value in (("epoch", self.epoch), ("step", self.step)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.observation_phase != "pre_update":
            raise ValueError("observation_phase must be exactly 'pre_update'")
        for name in ("char_loss", "syllable_loss", "boundary_loss", "total_loss"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.group_ids, tuple):
            raise TypeError("group_ids must be a tuple of non-empty strings")
        if any(not isinstance(group, str) or not group.strip() for group in self.group_ids):
            raise ValueError("group_ids must contain only non-empty strings")


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Bounded aggregate for one unique example reference."""

    example_ref: ExampleRef
    seen_count: int
    first_seen_step: int
    last_seen_step: int
    latest_event: LearningEvent


class LearningMemory:
    """Deterministic FIFO memory bounded by unique ``ExampleRef`` count.

    Re-observing a retained reference updates its latest event and exposure
    count without changing its insertion order. A new reference at capacity
    evicts the oldest first-inserted reference.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._entries: OrderedDict[ExampleRef, MemoryEntry] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, example_ref: ExampleRef) -> bool:
        if not isinstance(example_ref, ExampleRef):
            raise TypeError("example_ref must be an ExampleRef")
        return example_ref in self._entries

    def observe(self, event: LearningEvent) -> None:
        if not isinstance(event, LearningEvent):
            raise TypeError("event must be a LearningEvent")
        current = self._entries.get(event.example_ref)
        if current is not None:
            self._entries[event.example_ref] = MemoryEntry(
                example_ref=event.example_ref,
                seen_count=current.seen_count + 1,
                first_seen_step=current.first_seen_step,
                last_seen_step=event.step,
                latest_event=event,
            )
            return
        if len(self._entries) >= self._capacity:
            self._entries.popitem(last=False)
        self._entries[event.example_ref] = MemoryEntry(
            example_ref=event.example_ref,
            seen_count=1,
            first_seen_step=event.step,
            last_seen_step=event.step,
            latest_event=event,
        )

    def entries(self) -> tuple[MemoryEntry, ...]:
        """Return an immutable snapshot ordered oldest first-inserted first."""
        return tuple(self._entries.values())

    def select(self, limit: int) -> tuple[MemoryEntry, ...]:
        """Return up to ``limit`` oldest-first entries; this is not replay."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return tuple(self._entries.values())[:limit]


class LearningMemoryRecorder:
    """Map trainer positions to pre-aligned refs and store scalar reports."""

    def __init__(
        self,
        example_refs: Sequence[ExampleRef],
        memory: LearningMemory,
        *,
        group_ids: tuple[str, ...] = (),
    ) -> None:
        refs = tuple(example_refs)
        if any(not isinstance(ref, ExampleRef) for ref in refs):
            raise TypeError("example_refs must contain only ExampleRef values")
        if not isinstance(memory, LearningMemory):
            raise TypeError("memory must be a LearningMemory")
        if not isinstance(group_ids, tuple) or any(
            not isinstance(group, str) or not group.strip() for group in group_ids
        ):
            raise ValueError("group_ids must be a tuple of non-empty strings")
        self._example_refs = refs
        self._memory = memory
        self._group_ids = group_ids

    def __call__(
        self,
        epoch: int,
        example_index: int,
        global_step: int,
        scalar_report: Mapping[str, float],
    ) -> None:
        if isinstance(example_index, bool) or not isinstance(example_index, int):
            raise TypeError("example_index must be an integer")
        if example_index < 0:
            raise IndexError("trainer example_index must be non-negative")
        try:
            example_ref = self._example_refs[example_index]
        except IndexError as exc:
            raise IndexError("trainer example_index has no pre-aligned ExampleRef") from exc
        event = LearningEvent(
            example_ref=example_ref,
            epoch=epoch,
            step=global_step,
            observation_phase="pre_update",
            char_loss=scalar_report["char_loss"],
            syllable_loss=scalar_report["syllable_loss"],
            boundary_loss=scalar_report["boundary_loss"],
            total_loss=scalar_report["total_loss"],
            group_ids=self._group_ids,
        )
        self._memory.observe(event)


__all__ = ["ExampleRef", "LearningEvent", "LearningMemory", "LearningMemoryRecorder", "MemoryEntry"]
