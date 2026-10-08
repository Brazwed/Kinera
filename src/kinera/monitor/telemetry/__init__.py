"""Opt-in execution telemetry contracts and monotonic timing helpers.

Importing this namespace performs no clock reads, hardware probes, collection,
thread/process creation, or CUDA initialization.
"""

from kinera.monitor.telemetry.records import (
    TELEMETRY_SCHEMA_VERSION,
    EnvironmentSnapshotRecord,
    ResourceSnapshotRecord,
    TelemetryAttemptRecord,
    TelemetryEnvelope,
    TelemetryRecordError,
    TelemetryRunRecord,
    TelemetrySegmentRecord,
    TelemetryStageRecord,
    merge_telemetry_run_records,
)
from kinera.monitor.telemetry.environment import collect_environment_snapshot
from kinera.monitor.telemetry.resources import collect_resource_snapshot
from kinera.monitor.telemetry.timing import TimingSession, utc_now

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "EnvironmentSnapshotRecord",
    "ResourceSnapshotRecord",
    "TelemetryAttemptRecord",
    "TelemetryEnvelope",
    "TelemetryRecordError",
    "TelemetryRunRecord",
    "TelemetrySegmentRecord",
    "TelemetryStageRecord",
    "TimingSession",
    "collect_environment_snapshot",
    "collect_resource_snapshot",
    "merge_telemetry_run_records",
    "utc_now",
]
