import unittest
from datetime import datetime, timedelta, timezone

from kinera.monitor.telemetry.timing import TimingSession, utc_now


class FakeClock:
    def __init__(self, wall_values: list[datetime], monotonic_values: list[int]) -> None:
        self._wall_values = iter(wall_values)
        self._monotonic_values = iter(monotonic_values)

    def wall(self) -> datetime:
        return next(self._wall_values)

    def monotonic_ns(self) -> int:
        return next(self._monotonic_values)


class TelemetryTimingTests(unittest.TestCase):
    def test_default_civil_clock_returns_aware_utc(self) -> None:
        timestamp = utc_now()
        self.assertEqual(timestamp.tzinfo, timezone.utc)
        self.assertIsNotNone(timestamp.utcoffset())

    def test_nested_stages_keep_utc_wall_time_and_monotonic_exclusive_duration(self) -> None:
        origin = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        clock = FakeClock(
            [origin, origin + timedelta(milliseconds=1),
             origin + timedelta(milliseconds=3), origin + timedelta(milliseconds=5)],
            [0, 1_000_000, 3_000_000, 5_000_000],
        )
        timing = TimingSession(wall_clock=clock.wall, monotonic_ns=clock.monotonic_ns)

        with timing.stage(stage_id="train", name="training"):
            with timing.stage(stage_id="validate", name="validation"):
                pass

        parent, child = timing.stages
        self.assertEqual(parent.started_at, origin.isoformat())
        self.assertEqual(parent.finished_at, (origin + timedelta(milliseconds=5)).isoformat())
        self.assertEqual(parent.duration_seconds, 0.005)
        self.assertEqual(parent.exclusive_duration_seconds, 0.003)
        self.assertEqual(child.duration_seconds, 0.002)
        self.assertEqual(child.exclusive_duration_seconds, 0.002)
        self.assertEqual(child.parent_stage_id, "train")
        self.assertEqual(parent.status, "completed")
        self.assertEqual(child.status, "completed")

    def test_injected_wall_clock_is_normalized_to_utc(self) -> None:
        offset_clock = datetime(
            2026, 10, 7, 8, 0, 0, tzinfo=timezone(timedelta(hours=5))
        )
        wall_values = iter([offset_clock, offset_clock])
        monotonic_values = iter([10, 20])
        timing = TimingSession(
            wall_clock=lambda: next(wall_values),
            monotonic_ns=lambda: next(monotonic_values),
        )

        with timing.stage(stage_id="utc", name="setup"):
            pass

        self.assertEqual(timing.stages[0].started_at, "2026-10-07T03:00:00+00:00")
        self.assertEqual(timing.stages[0].finished_at, "2026-10-07T03:00:00+00:00")

    def test_duration_uses_monotonic_clock_when_civil_clock_moves_backwards(self) -> None:
        start = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        clock = FakeClock(
            [start, start - timedelta(hours=1)],
            [2_000_000_000, 4_500_000_000],
        )
        timing = TimingSession(wall_clock=clock.wall, monotonic_ns=clock.monotonic_ns)

        with timing.stage(stage_id="stage-1", name="setup"):
            pass

        record = timing.stages[0]
        self.assertEqual(record.duration_seconds, 2.5)
        self.assertLess(record.finished_at, record.started_at)

    def test_context_manager_records_failure_but_propagates_original_exception(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        clock = FakeClock([instant, instant], [0, 100])
        timing = TimingSession(wall_clock=clock.wall, monotonic_ns=clock.monotonic_ns)

        with self.assertRaisesRegex(RuntimeError, "scientific failure"):
            with timing.stage(stage_id="step", name="sgd"):
                raise RuntimeError("scientific failure")

        record = timing.stages[0]
        self.assertEqual(record.status, "failed")
        self.assertIn("scientific failure", record.error)
        self.assertEqual(record.duration_seconds, 1e-07)

    def test_overflowing_monotonic_delta_does_not_mask_body_exception(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monotonic_values = iter([0, 10**400])
        timing = TimingSession(
            wall_clock=lambda: instant,
            monotonic_ns=lambda: next(monotonic_values),
        )
        original = ValueError("scientific failure")

        with self.assertRaises(ValueError) as caught:
            with timing.stage(stage_id="overflow", name="training"):
                raise original

        self.assertIs(caught.exception, original)
        self.assertEqual(timing.stages[0].status, "failed")
        self.assertIsNone(timing.stages[0].duration_seconds)
        self.assertIsNotNone(timing.stages[0].timing_error)
        assert timing.stages[0].timing_error is not None
        self.assertIn("OverflowError", timing.stages[0].timing_error)
        self.assertEqual(timing._stack, [])

    def test_unrepresentable_duration_is_diagnostic_without_failing_successful_body(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monotonic_values = iter([0, 10**400])
        timing = TimingSession(
            wall_clock=lambda: instant,
            monotonic_ns=lambda: next(monotonic_values),
        )

        scientific_status = "not_started"
        with timing.stage(stage_id="success-overflow", name="training"):
            scientific_status = "completed"

        self.assertEqual(scientific_status, "completed")
        self.assertEqual(timing.stages[0].status, "completed")
        self.assertIsNone(timing.stages[0].duration_seconds)
        self.assertIsNotNone(timing.stages[0].timing_error)
        self.assertEqual(timing._stack, [])

    def test_wall_clock_failure_does_not_mask_scientific_value_error(self) -> None:
        monotonic_values = iter([0, 100])

        def broken_wall_clock() -> datetime:
            raise OSError("civil clock unavailable")

        timing = TimingSession(
            wall_clock=broken_wall_clock,
            monotonic_ns=lambda: next(monotonic_values),
        )
        original = ValueError("scientific failure")

        with self.assertRaises(ValueError) as caught:
            with timing.stage(stage_id="wall-failure", name="training"):
                raise original

        self.assertIs(caught.exception, original)
        self.assertEqual(timing.stages[0].status, "failed")
        self.assertIsNotNone(timing.stages[0].timing_error)
        assert timing.stages[0].timing_error is not None
        self.assertIn("civil clock unavailable", timing.stages[0].timing_error)

    def test_monotonic_failure_after_success_keeps_success_status(self) -> None:
        calls = 0

        def monotonic_ns() -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0
            raise OSError("monotonic clock unavailable")

        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        timing = TimingSession(wall_clock=lambda: instant, monotonic_ns=monotonic_ns)
        scientific_result = "not_finished"
        with timing.stage(stage_id="mono-failure", name="training"):
            scientific_result = "completed"

        record = timing.stages[0]
        self.assertEqual(scientific_result, "completed")
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.duration_seconds)
        self.assertIsNotNone(record.timing_error)
        assert record.timing_error is not None
        self.assertIn("monotonic clock unavailable", record.timing_error)

    def test_record_serialization_failure_never_masks_body_exception(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monotonic_values = iter([0, 1])
        timing = TimingSession(
            wall_clock=lambda: instant,
            monotonic_ns=lambda: next(monotonic_values),
        )
        original = ValueError("scientific failure")

        def broken_serialization() -> dict:
            raise RuntimeError("record serialization failed")

        with self.assertRaises(ValueError) as caught:
            with timing.stage(stage_id="serialization", name="training"):
                timing.stages[0].to_dict = broken_serialization
                raise original

        self.assertIs(caught.exception, original)
        self.assertEqual(timing.stages[0].status, "failed")
        self.assertIsNotNone(timing.stages[0].timing_error)
        assert timing.stages[0].timing_error is not None
        self.assertIn("record serialization failed", timing.stages[0].timing_error)

    def test_record_serialization_failure_keeps_successful_body_status(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monotonic_values = iter([0, 1])
        timing = TimingSession(
            wall_clock=lambda: instant,
            monotonic_ns=lambda: next(monotonic_values),
        )

        def broken_serialization() -> dict:
            raise RuntimeError("record serialization failed")

        scientific_result = "not_finished"
        with timing.stage(stage_id="serialization-success", name="training"):
            timing.stages[0].to_dict = broken_serialization
            scientific_result = "completed"

        self.assertEqual(scientific_result, "completed")
        self.assertEqual(timing.stages[0].status, "completed")
        self.assertIsNotNone(timing.stages[0].timing_error)
        assert timing.stages[0].timing_error is not None
        self.assertIn("record serialization failed", timing.stages[0].timing_error)

    def test_clock_error_is_recorded_without_changing_body_result(self) -> None:
        def broken_wall_clock() -> datetime:
            raise OSError("civil clock unavailable")

        monotonic_values = iter([10, 1_000_000_010])
        timing = TimingSession(
            wall_clock=broken_wall_clock,
            monotonic_ns=lambda: next(monotonic_values),
        )
        scientific_status = "not_started"
        with timing.stage(stage_id="run", name="training"):
            scientific_status = "completed"

        record = timing.stages[0]
        self.assertEqual(scientific_status, "completed")
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.started_at)
        self.assertIsNone(record.finished_at)
        self.assertEqual(record.duration_seconds, 1.0)
        self.assertIn("civil clock unavailable", record.timing_error)

    def test_duplicate_stage_identity_is_rejected_without_generating_ids(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        clock = FakeClock([instant, instant], [0, 1])
        timing = TimingSession(wall_clock=clock.wall, monotonic_ns=clock.monotonic_ns)
        with timing.stage(stage_id="explicit-stage", name="one"):
            pass
        with self.assertRaisesRegex(ValueError, "stage_id"):
            timing.stage(stage_id="explicit-stage", name="duplicate")
        self.assertEqual([stage.stage_id for stage in timing.stages], ["explicit-stage"])


if __name__ == "__main__":
    unittest.main()
