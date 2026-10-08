import json
import unittest
from datetime import datetime, timezone

from kinera.experiments.config import RunRecord
from kinera.monitor.telemetry.records import (
    TELEMETRY_SCHEMA_VERSION,
    TelemetryAttemptRecord,
    TelemetryEnvelope,
    TelemetryRecordError,
    EnvironmentSnapshotRecord,
    ResourceSnapshotRecord,
    TelemetryRunRecord,
    TelemetrySegmentRecord,
    TelemetryStageRecord,
    _prepare_envelope_payload,
    merge_telemetry_run_records,
)
from kinera.monitor.telemetry.timing import TimingSession


UTC_START = "2026-01-02T03:04:05+00:00"
UTC_END = "2026-01-02T03:04:06+00:00"
ENVIRONMENT_FIELDS = (
    "python_version", "python_executable", "os_system", "os_release", "platform_version",
    "kernel_release", "architecture", "cpu_model", "logical_cpu_count", "cpu_affinity",
)
RESOURCE_FIELDS = (
    "process_rss_bytes", "process_ru_maxrss_kib", "cgroup_memory_current_bytes",
    "cgroup_memory_peak_bytes", "cgroup_memory_limit_bytes",
    "cgroup_memory_limit_unlimited", "cgroup_version",
)


def unavailable_fields(fields: tuple[str, ...], present: set[str]) -> dict[str, str]:
    return {name: "fixture intentionally omitted this field" for name in fields if name not in present}


def failed_attempt(attempt_id: str = "attempt-1") -> TelemetryAttemptRecord:
    return TelemetryAttemptRecord(
        attempt_id=attempt_id,
        process_id="logical-process-a",
        pid=4312,
        status="failed",
        started_at=UTC_START,
        finished_at=UTC_END,
        duration_seconds=1.0,
        error="worker error",
        cleanup_status="failed",
        cleanup_error="close failed",
        segments=[
            TelemetrySegmentRecord(
                segment_id="segment-1",
                status="failed",
                started_at=UTC_START,
                finished_at=UTC_END,
                duration_seconds=1.0,
                error="stage error",
                stages=[
                    TelemetryStageRecord(
                        stage_id="stage-1",
                        name="training",
                        started_at=UTC_START,
                        finished_at=UTC_END,
                        duration_seconds=1.0,
                        exclusive_duration_seconds=0.75,
                        status="failed",
                        error="training error",
                    )
                ],
            )
        ],
    )


def telemetry_run(*attempts: TelemetryAttemptRecord) -> TelemetryRunRecord:
    return TelemetryRunRecord(
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        run_id="run-7",
        attempts=list(attempts),
    )


class TelemetryRecordTests(unittest.TestCase):
    def test_run_and_envelope_round_trip_nested_records_as_json(self) -> None:
        run = telemetry_run(failed_attempt())
        envelope = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="session-a",
            started_at=UTC_START,
        )

        encoded_run = json.loads(json.dumps(run.to_dict()))
        encoded_envelope = json.loads(json.dumps(envelope.to_dict()))
        restored_run = TelemetryRunRecord.from_dict(encoded_run)
        restored_envelope = TelemetryEnvelope.from_dict(encoded_envelope)

        self.assertEqual(restored_run.to_dict(), run.to_dict())
        self.assertEqual(restored_envelope.to_dict(), envelope.to_dict())
        attempt = restored_run.attempts[0]
        self.assertEqual(attempt.process_id, "logical-process-a")
        self.assertEqual(attempt.pid, 4312)
        self.assertNotEqual(attempt.process_id, str(attempt.pid))
        self.assertEqual(attempt.cleanup_status, "failed")
        self.assertEqual(attempt.cleanup_error, "close failed")
        self.assertEqual(attempt.segments[0].stages[0].parent_stage_id, None)

    def test_envelope_round_trips_optional_t2_snapshots_without_bumping_t0_t1_schema(self) -> None:
        environment = EnvironmentSnapshotRecord(
            environment_schema_version=1,
            captured_at=UTC_START,
            python_version="3.14.7",
            python_executable="/usr/bin/python3",
            os_system="Linux",
            os_release="24.04",
            platform_version="build-1",
            kernel_release="6.12.0",
            architecture="x86_64",
            cpu_model="fixture CPU",
            logical_cpu_count=8,
            cpu_affinity=[0, 2],
            sources={
                "captured_at": "datetime.now(timezone.utc)",
                "python_version": "platform.python_version",
                "python_executable": "sys.executable",
                "os_system": "platform.system",
                "os_release": "platform.freedesktop_os_release:VERSION_ID",
                "platform_version": "platform.version",
                "kernel_release": "platform.release",
                "architecture": "platform.machine",
                "cpu_model": "/proc/cpuinfo",
                "logical_cpu_count": "os.cpu_count",
                "cpu_affinity": "os.sched_getaffinity(0)",
            },
        )
        resources = ResourceSnapshotRecord(
            resources_schema_version=1,
            captured_at=UTC_START,
            process_rss_bytes=2048,
            process_ru_maxrss_kib=1024,
            cgroup_memory_current_bytes=4096,
            cgroup_memory_peak_bytes=8192,
            cgroup_memory_limit_bytes=None,
            cgroup_memory_limit_unlimited=True,
            cgroup_version="v2",
            sources={
                "captured_at": "datetime.now(timezone.utc)",
                "process_rss_bytes": "/proc/self/status:VmRSS",
                "process_ru_maxrss_kib": "resource.getrusage(RUSAGE_SELF)",
                "cgroup_memory_current_bytes": "memory.current",
                "cgroup_memory_peak_bytes": "memory.peak",
                "cgroup_memory_limit_unlimited": "memory.max",
                "cgroup_version": "/proc/self/cgroup+mountinfo",
            },
            not_applicable={"cgroup_memory_limit_bytes": "memory.max is unlimited"},
        )
        envelope = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="session-t2",
            environment=environment,
            resources=resources,
        )

        encoded = json.loads(json.dumps(envelope.to_dict(), allow_nan=False))
        restored = TelemetryEnvelope.from_dict(encoded)
        old = TelemetryEnvelope.from_dict({
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": "old-session",
        }).to_dict()

        self.assertEqual(envelope.telemetry_schema_version, TELEMETRY_SCHEMA_VERSION)
        self.assertEqual(encoded["environment"]["environment_schema_version"], 1)
        self.assertEqual(encoded["resources"]["resources_schema_version"], 1)
        self.assertEqual(restored.to_dict(), encoded)
        self.assertNotIn("environment", old)
        self.assertNotIn("resources", old)

    def test_envelope_payload_normalizes_nested_snapshot_timestamps_to_utc(self) -> None:
        environment = EnvironmentSnapshotRecord(
            environment_schema_version=1,
            captured_at=UTC_START,
            sources={"captured_at": "explicit UTC clock"},
            unavailable=unavailable_fields(ENVIRONMENT_FIELDS, set()),
        ).to_dict()
        environment["captured_at"] = "2026-01-02T08:04:05+05:00"
        raw_envelope = {
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "session_id": "session-offset",
            "environment": environment,
        }

        prepared = _prepare_envelope_payload(raw_envelope, component="test")

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(
            prepared["environment"]["captured_at"],
            "2026-01-02T03:04:05+00:00",
        )

    def test_t2_snapshot_records_revalidate_numeric_values_and_utc_timestamp(self) -> None:
        environment = EnvironmentSnapshotRecord(
            environment_schema_version=1,
            captured_at="2026-01-02T08:04:05+05:00",
            logical_cpu_count=8,
            sources={
                "captured_at": "clock",
                "logical_cpu_count": "os.cpu_count",
            },
            unavailable=unavailable_fields(ENVIRONMENT_FIELDS, {"logical_cpu_count"}),
        )
        self.assertEqual(environment.to_dict()["captured_at"], UTC_START)
        environment.logical_cpu_count = -1
        with self.assertRaises(TelemetryRecordError):
            environment.to_dict()

        resources = ResourceSnapshotRecord(
            resources_schema_version=1,
            captured_at=UTC_START,
            process_rss_bytes=1024,
            sources={
                "captured_at": "clock",
                "process_rss_bytes": "/proc/self/status:VmRSS",
            },
            unavailable=unavailable_fields(RESOURCE_FIELDS, {"process_rss_bytes"}),
        )
        resources.process_rss_bytes = -1
        with self.assertRaises(TelemetryRecordError):
            resources.to_dict()

    def test_resource_snapshot_rejects_unhashable_cgroup_version(self) -> None:
        record = ResourceSnapshotRecord(
            resources_schema_version=1,
            captured_at=UTC_START,
            cgroup_version="v2",
            sources={"captured_at": "clock", "cgroup_version": "fixture"},
            unavailable=unavailable_fields(RESOURCE_FIELDS, {"cgroup_version"}),
        )
        setattr(record, "cgroup_version", {})
        with self.assertRaises(TelemetryRecordError):
            record.to_dict()

    def test_t2_snapshot_absent_fields_require_availability_reason(self) -> None:
        with self.assertRaises(TelemetryRecordError):
            EnvironmentSnapshotRecord(
                environment_schema_version=1,
                captured_at=UTC_START,
                sources={"captured_at": "clock"},
            )

    def test_old_run_record_round_trip_does_not_invent_telemetry(self) -> None:
        old = {"seed": 7, "status": "completed", "payload": {"loss": 1.25}}

        restored = RunRecord.from_dict(old)
        encoded = restored.to_dict()

        self.assertIsNone(restored.telemetry)
        self.assertNotIn("telemetry", encoded)
        self.assertEqual(encoded["seed"], 7)
        self.assertEqual(encoded["status"], "completed")
        self.assertEqual(encoded["payload"], {"loss": 1.25})
        with self.assertRaises(TelemetryRecordError):
            TelemetryEnvelope.from_dict({"session_id": "legacy-without-schema"})

    def test_run_record_round_trip_preserves_typed_telemetry_without_moving_payload(self) -> None:
        run = telemetry_run(failed_attempt())
        record = RunRecord(
            seed=7,
            status="failed",
            payload={"result": {"loss": 2.0}},
            error="scientific failure",
            telemetry=run,
        )

        encoded = record.to_dict()
        restored = RunRecord.from_dict(encoded)

        self.assertEqual(encoded["payload"], {"result": {"loss": 2.0}})
        self.assertEqual(restored.telemetry.to_dict(), run.to_dict())
        self.assertEqual(restored.status, "failed")
        self.assertEqual(restored.error, "scientific failure")

    def test_attempt_retry_merge_is_append_only_and_idempotent(self) -> None:
        old_failed = failed_attempt("attempt-1")
        new_attempt = TelemetryAttemptRecord(
            attempt_id="attempt-2",
            process_id="logical-process-b",
            pid=9021,
            status="completed",
            started_at=UTC_START,
            finished_at=UTC_END,
            duration_seconds=1.0,
            segments=[
                TelemetrySegmentRecord(
                    segment_id="segment-2",
                    status="completed",
                    started_at=UTC_START,
                    finished_at=UTC_END,
                    duration_seconds=1.0,
                )
            ],
        )
        previous = telemetry_run(old_failed)
        retry = telemetry_run(new_attempt)

        merged = merge_telemetry_run_records(previous, retry)
        merged_twice = merge_telemetry_run_records(merged, retry)

        self.assertEqual(
            [attempt.attempt_id for attempt in merged.attempts],
            ["attempt-1", "attempt-2"],
        )
        self.assertEqual(merged.attempts[0].status, "failed")
        self.assertEqual(merged.attempts[0].segments[0].segment_id, "segment-1")
        self.assertEqual(merged.attempts[1].status, "completed")
        self.assertEqual(merged_twice.to_dict(), merged.to_dict())

    def test_resume_does_not_regress_running_attempt_or_segment_to_pending(self) -> None:
        running = telemetry_run(
            TelemetryAttemptRecord(
                attempt_id="attempt-2",
                process_id="logical-process-b",
                status="running",
                segments=[TelemetrySegmentRecord(segment_id="segment-2", status="running")],
            )
        )
        stale_pending = telemetry_run(
            TelemetryAttemptRecord(
                attempt_id="attempt-2",
                process_id="logical-process-b",
                status="pending",
                segments=[TelemetrySegmentRecord(segment_id="segment-2", status="pending")],
            )
        )

        merged = merge_telemetry_run_records(running, stale_pending)
        self.assertIsNotNone(merged)
        assert merged is not None

        self.assertEqual(merged.attempts[0].status, "running")
        self.assertEqual(merged.attempts[0].segments[0].status, "running")

    def test_missing_replacement_telemetry_keeps_prior_history(self) -> None:
        previous = telemetry_run(failed_attempt())
        merged = merge_telemetry_run_records(previous, None)
        self.assertEqual(merged.to_dict(), previous.to_dict())

    def test_failed_attempt_is_not_retroactively_completed(self) -> None:
        previous = telemetry_run(failed_attempt("attempt-1"))
        retroactive = telemetry_run(
            TelemetryAttemptRecord(
                attempt_id="attempt-1",
                process_id="logical-process-a",
                pid=4312,
                status="completed",
                started_at=UTC_START,
                finished_at=UTC_END,
                duration_seconds=1.0,
            )
        )

        merged = merge_telemetry_run_records(previous, retroactive)

        self.assertEqual(merged.attempts[0].status, "failed")
        self.assertEqual(merged.attempts[0].error, "worker error")

    def test_cleanup_telemetry_does_not_change_scientific_run_status(self) -> None:
        run = telemetry_run(
            TelemetryAttemptRecord(
                attempt_id="attempt-cleanup",
                process_id="logical-process-cleanup",
                status="completed",
                cleanup_status="failed",
                cleanup_error="cleanup unavailable",
            )
        )
        record = RunRecord(seed=7, status="completed", telemetry=run)

        self.assertEqual(record.to_dict()["status"], "completed")
        self.assertEqual(
            record.to_dict()["telemetry"]["attempts"][0]["cleanup_status"],
            "failed",
        )

    def test_record_timestamps_are_normalized_to_utc(self) -> None:
        record = TelemetryStageRecord(
            stage_id="utc-stage",
            name="setup",
            started_at="2026-10-07T08:00:00+05:00",
        )
        self.assertEqual(record.to_dict()["started_at"], "2026-10-07T03:00:00+00:00")

    def test_to_dict_revalidates_mutated_records(self) -> None:
        stage_nan = TelemetryStageRecord(stage_id="nan", name="stage")
        stage_nan.duration_seconds = float("nan")
        segment_inf = TelemetrySegmentRecord(segment_id="infinity")
        segment_inf.duration_seconds = float("inf")
        attempt_negative = TelemetryAttemptRecord(
            attempt_id="negative", process_id="logical-negative"
        )
        attempt_negative.duration_seconds = -1.0
        attempt_huge = TelemetryAttemptRecord(
            attempt_id="huge", process_id="logical-huge"
        )
        attempt_huge.duration_seconds = 10**400
        envelope_naive = TelemetryEnvelope(
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            session_id="session-naive",
        )
        envelope_naive.started_at = "2026-10-07T03:04:05"
        attempt_bad_pid = TelemetryAttemptRecord(
            attempt_id="pid", process_id="logical-pid", pid=10
        )
        attempt_bad_pid.pid = -1
        run_bad_version = telemetry_run()
        run_bad_version.telemetry_schema_version = 0
        run_bad_id = telemetry_run()
        run_bad_id.run_id = ""
        duplicate_run = telemetry_run(failed_attempt("attempt-1"), failed_attempt("attempt-2"))
        duplicate_run.attempts[1].attempt_id = "attempt-1"
        stage_bad_status = TelemetryStageRecord(stage_id="status", name="stage")
        setattr(stage_bad_status, "status", "unknown")
        stage_bad_id = TelemetryStageRecord(stage_id="valid", name="stage")
        stage_bad_id.stage_id = ""

        for record in (
            stage_nan,
            segment_inf,
            attempt_negative,
            attempt_huge,
            envelope_naive,
            attempt_bad_pid,
            run_bad_version,
            run_bad_id,
            duplicate_run,
            stage_bad_status,
            stage_bad_id,
        ):
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(TelemetryRecordError):
                    record.to_dict()

    def test_parent_serialization_revalidates_nested_stage(self) -> None:
        stage = TelemetryStageRecord(stage_id="nested", name="fit")
        segment = TelemetrySegmentRecord(segment_id="segment", stages=[stage])
        attempt = TelemetryAttemptRecord(
            attempt_id="attempt", process_id="logical-process", segments=[segment]
        )
        run = telemetry_run(attempt)
        stage.duration_seconds = float("inf")

        with self.assertRaises(TelemetryRecordError):
            run.to_dict()

    def test_cleanup_stage_failure_is_recorded_without_changing_scientific_status(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        wall_values = iter([instant, instant])
        monotonic_values = iter([0, 1])
        timing = TimingSession(
            wall_clock=lambda: next(wall_values),
            monotonic_ns=lambda: next(monotonic_values),
        )
        attempt = TelemetryAttemptRecord(
            attempt_id="attempt-cleanup",
            process_id="logical-cleanup",
            status="completed",
        )

        class BrokenResource:
            def close(self) -> None:
                raise OSError("close failed")

        with self.assertRaisesRegex(OSError, "close failed"):
            with timing.stage(stage_id="cleanup-stage", name="cleanup"):
                BrokenResource().close()
        attempt.record_cleanup_stage(timing.stages[0])
        record = RunRecord(
            seed=7,
            status="completed",
            telemetry=TelemetryRunRecord(
                telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
                run_id="run-cleanup",
                attempts=[attempt],
            ),
        )

        self.assertEqual(attempt.cleanup_status, "failed")
        self.assertIsNotNone(attempt.cleanup_error)
        assert attempt.cleanup_error is not None
        self.assertIn("close failed", attempt.cleanup_error)
        self.assertEqual(record.status, "completed")

    def test_cleanup_failure_does_not_mask_prior_scientific_exception(self) -> None:
        instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        wall_values = iter([instant] * 4)
        monotonic_values = iter([0, 1, 2, 3])
        timing = TimingSession(
            wall_clock=lambda: next(wall_values),
            monotonic_ns=lambda: next(monotonic_values),
        )
        attempt = TelemetryAttemptRecord(
            attempt_id="attempt-cleanup",
            process_id="logical-cleanup",
            status="failed",
        )
        original = ValueError("scientific failure")

        def run_with_cleanup() -> None:
            try:
                with timing.stage(stage_id="train-stage", name="training"):
                    raise original
            except ValueError as scientific_error:
                try:
                    with timing.stage(stage_id="cleanup-stage", name="cleanup"):
                        raise OSError("close failed")
                except OSError:
                    attempt.record_cleanup_stage(timing.stages[1])
                raise scientific_error

        with self.assertRaises(ValueError) as caught:
            run_with_cleanup()

        self.assertIs(caught.exception, original)
        self.assertEqual(attempt.cleanup_status, "failed")
        self.assertIsNotNone(attempt.cleanup_error)
        assert attempt.cleanup_error is not None
        self.assertIn("close failed", attempt.cleanup_error)

    def test_records_reject_naive_timestamps_and_pid_as_process_identity(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryEnvelope(
                telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
                session_id="session-a",
                started_at="2026-01-02T03:04:05",
            )
        with self.assertRaises(TelemetryRecordError):
            TelemetryAttemptRecord(
                attempt_id="attempt-1",
                process_id=4312,
                pid=4312,
            )


if __name__ == "__main__":
    unittest.main()
