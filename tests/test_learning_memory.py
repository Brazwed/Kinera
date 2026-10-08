from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from kinera.experiments.learning_memory import (
    ExampleRef,
    LearningEvent,
    LearningMemory,
    LearningMemoryRecorder,
)


class ExampleRefTests(unittest.TestCase):
    def test_example_ref_is_immutable_and_accepts_stable_artifact_identity(self) -> None:
        ref = ExampleRef("fixture:tiny", "train", 0)
        self.assertEqual((ref.dataset_artifact_id, ref.split, ref.example_index), ("fixture:tiny", "train", 0))
        with self.assertRaises((AttributeError, TypeError)):
            ref.example_index = 1  # type: ignore[misc]

    def test_example_ref_rejects_invalid_identity_and_index(self) -> None:
        invalid_values = (
            ("", "train", 0),
            ("fixture:x", "", 0),
            ("fixture:x", "train", -1),
            ("fixture:x", "train", True),
            (None, "train", 0),
            ("fixture:x", 1, 0),
        )
        for args in invalid_values:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                ExampleRef(*args)  # type: ignore[arg-type]


class LearningEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = ExampleRef("fixture:tiny", "train", 0)

    def test_event_is_immutable_and_contains_only_scalar_observations(self) -> None:
        event = LearningEvent(self.ref, 1, 1, "pre_update", 1.0, 2.0, 3.0, 6.0)
        self.assertEqual(event.group_ids, ())
        self.assertEqual(event.total_loss, 6.0)
        with self.assertRaises((AttributeError, TypeError)):
            event.step = 2  # type: ignore[misc]
        self.assertFalse(hasattr(event, "logits"))
        self.assertFalse(hasattr(event, "gradients"))
        self.assertFalse(hasattr(event, "model"))

    def test_event_rejects_invalid_phase_indices_losses_and_groups(self) -> None:
        valid = (self.ref, 1, 1, "pre_update", 1.0, 2.0, 3.0, 6.0)
        invalid = (
            (self.ref, 0, 1, "pre_update", 1.0, 2.0, 3.0, 6.0),
            (self.ref, 1, 0, "pre_update", 1.0, 2.0, 3.0, 6.0),
            (self.ref, 1, 1, "post_update", 1.0, 2.0, 3.0, 6.0),
            (self.ref, 1, 1, "pre_update", math.nan, 2.0, 3.0, 6.0),
            (self.ref, 1, 1, "pre_update", 1.0, math.inf, 3.0, 6.0),
            (self.ref, 1, 1, "pre_update", 1.0, 2.0, -math.inf, 6.0),
            (self.ref, 1, 1, "pre_update", 1.0, 2.0, 3.0, math.nan),
            (self.ref, 1, 1, "pre_update", True, 2.0, 3.0, 6.0),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                LearningEvent(*args)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            LearningEvent("not-a-ref", 1, 1, "pre_update", 1.0, 2.0, 3.0, 6.0)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            LearningEvent(*valid, group_ids=("valid", ""))
        with self.assertRaises((TypeError, ValueError)):
            LearningEvent(*valid, group_ids=["not-a-tuple"])  # type: ignore[arg-type]


class LearningMemoryTests(unittest.TestCase):
    def _event(self, index: int, step: int) -> LearningEvent:
        return LearningEvent(
            ExampleRef("fixture:tiny", "train", index),
            1,
            step,
            "pre_update",
            0.1,
            0.2,
            0.3,
            0.6,
        )

    def test_capacity_must_be_positive(self) -> None:
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises((TypeError, ValueError)):
                LearningMemory(capacity)  # type: ignore[arg-type]

    def test_repeated_reference_updates_entry_without_using_another_slot(self) -> None:
        memory = LearningMemory(capacity=2)
        first = self._event(0, 1)
        repeated = self._event(0, 4)
        memory.observe(first)
        memory.observe(repeated)

        entries = memory.entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].seen_count, 2)
        self.assertEqual(entries[0].first_seen_step, 1)
        self.assertEqual(entries[0].last_seen_step, 4)
        self.assertIs(entries[0].latest_event, repeated)
        self.assertTrue(memory.contains(first.example_ref))

    def test_full_memory_evicts_oldest_first_insertion_deterministically(self) -> None:
        memory = LearningMemory(capacity=2)
        events = [self._event(index, index + 1) for index in range(3)]
        for event in events:
            memory.observe(event)

        self.assertEqual(
            [entry.example_ref for entry in memory.entries()],
            [events[1].example_ref, events[2].example_ref],
        )
        self.assertFalse(memory.contains(events[0].example_ref))
        self.assertEqual(len(memory), 2)

    def test_entries_and_select_are_stable_oldest_first_and_bounded_by_limit(self) -> None:
        memory = LearningMemory(capacity=4)
        events = [self._event(index, index + 1) for index in range(3)]
        for event in events:
            memory.observe(event)

        expected = tuple(entry.example_ref for entry in memory.entries())
        self.assertEqual(expected, tuple(event.example_ref for event in events))
        self.assertEqual(tuple(entry.example_ref for entry in memory.select(0)), ())
        self.assertEqual(tuple(entry.example_ref for entry in memory.select(1)), expected[:1])
        self.assertEqual(tuple(entry.example_ref for entry in memory.select(3)), expected)
        self.assertEqual(tuple(entry.example_ref for entry in memory.select(9)), expected)
        self.assertEqual(memory.entries(), memory.entries())
        self.assertEqual(memory.select(2), memory.select(2))
        for invalid_limit in (-1, True, 1.5):
            with self.subTest(limit=invalid_limit), self.assertRaises(ValueError):
                memory.select(invalid_limit)  # type: ignore[arg-type]

    def test_memories_are_independent_and_observe_does_not_replay(self) -> None:
        left = LearningMemory(capacity=1)
        right = LearningMemory(capacity=1)
        event = self._event(0, 1)

        self.assertIsNone(left.observe(event))
        self.assertTrue(left.contains(event.example_ref))
        self.assertEqual(right.entries(), ())
        self.assertFalse(hasattr(left, "replay"))
        self.assertFalse(hasattr(left, "train"))

    def test_observer_maps_operational_position_to_pre_aligned_example_ref(self) -> None:
        refs = (ExampleRef("fixture:aligned", "train", 8),)
        memory = LearningMemory(capacity=2)
        observer = LearningMemoryRecorder(refs, memory)
        report = {
            "char_loss": 0.1,
            "syllable_loss": 0.2,
            "boundary_loss": 0.3,
            "total_loss": 0.6,
        }

        observer(1, 0, 1, report)

        event = memory.entries()[0].latest_event
        self.assertEqual(event.example_ref, refs[0])
        self.assertEqual(event.observation_phase, "pre_update")
        self.assertEqual(
            (event.char_loss, event.syllable_loss, event.boundary_loss, event.total_loss),
            tuple(report[name] for name in ("char_loss", "syllable_loss", "boundary_loss", "total_loss")),
        )
        with self.assertRaises(IndexError):
            observer(1, 1, 2, report)


class LearningMemoryImportBoundaryTests(unittest.TestCase):
    def test_import_is_cpu_only_and_has_no_runtime_side_effects(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
        script = r"""
import builtins, pathlib, sys, threading, time
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith(("kinera.backends", "kinera.monitor", "kinera.training", "kinera.nn")):
        raise AssertionError("forbidden import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
for name in ("time",):
    module = sys.modules[name]
    for clock in ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns"):
        setattr(module, clock, lambda *a, **k: (_ for _ in ()).throw(AssertionError("clock read")))
original_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if str(file).startswith("/proc/"):
        raise AssertionError("/proc read")
    return original_open(file, *args, **kwargs)
builtins.open = guarded_open
threading.Thread.__init__ = lambda *a, **k: (_ for _ in ()).throw(AssertionError("thread created"))
import kinera.experiments.learning_memory
assert not any(name.startswith("kinera.backends.cuda") for name in sys.modules)
assert not any(name.startswith("kinera.monitor.telemetry") for name in sys.modules)
assert not any(name.startswith("kinera.training") for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
