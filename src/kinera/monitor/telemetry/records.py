"""Typed, JSON-safe contracts for opt-in execution telemetry.

These records describe operational observations only. They do not import
experiment, training, model, backend, or hardware-probing modules.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TELEMETRY_SCHEMA_VERSION = 1
TelemetryStatus = Literal["pending", "running", "completed", "failed"]
CleanupStatus = Literal["completed", "failed", "not_run"]
_ALLOWED_STATUSES = {"pending", "running", "completed", "failed"}
_ALLOWED_CLEANUP_STATUSES = {"completed", "failed", "not_run"}


class TelemetryRecordError(ValueError):
    """Invalid telemetry data; never a scientific run status."""


_LOGGER = logging.getLogger("kinera.monitor.telemetry")


def _report_telemetry_diagnostic(component: str, error: BaseException) -> None:
    """Emit an operational warning without allowing logging to affect science."""
    try:
        try:
            detail = str(error)
        except BaseException:
            detail = "<unprintable exception>"
        _LOGGER.warning(
            "telemetry diagnostic component=%s error=%s: %s",
            component,
            type(error).__name__,
            detail,
        )
    except BaseException:
        pass


def _ensure_telemetry_json_safe(value: Any, component: str) -> None:
    """Validate only the telemetry subtree; scientific writers stay unchanged."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TelemetryRecordError(
            f"{component} não é JSON finito/serializável: {type(exc).__name__}"
        ) from exc


def _required_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TelemetryRecordError(f"{name} deve ser uma identidade lógica não vazia")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TelemetryRecordError(f"{name} deve ser texto ou None")
    return value


def _timestamp(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TelemetryRecordError(f"{name} deve ser timestamp ISO 8601 ou None")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TelemetryRecordError(f"{name} deve ser timestamp ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TelemetryRecordError(f"{name} deve incluir fuso horário")
    return parsed.astimezone(timezone.utc).isoformat()


def _duration(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryRecordError(f"{name} deve ser número finito não negativo")
    try:
        numeric = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise TelemetryRecordError(f"{name} deve ser número finito não negativo") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise TelemetryRecordError(f"{name} deve ser número finito não negativo")
    return numeric


def _status(value: object, name: str = "status") -> TelemetryStatus:
    if not isinstance(value, str) or value not in _ALLOWED_STATUSES:
        raise TelemetryRecordError(f"{name} de telemetry inválido")
    return value  # type: ignore[return-value]


def _schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TelemetryRecordError("telemetry_schema_version deve ser inteiro positivo")
    return value


def _pid(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TelemetryRecordError("pid deve ser inteiro positivo ou None")
    return value


def _require_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TelemetryRecordError(f"{name} deve ser um dicionário")
    return value


@dataclass
class TelemetryStageRecord:
    """One explicitly named stage.

    ``started_at``/``finished_at`` are timezone-aware civil timestamps;
    ``duration_seconds`` is the inclusive monotonic delta, while
    ``exclusive_duration_seconds`` subtracts the measured nested stages.
    """

    stage_id: str
    name: str
    parent_stage_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exclusive_duration_seconds: float | None = None
    status: TelemetryStatus = "pending"
    error: str | None = None
    timing_error: str | None = None

    def __post_init__(self) -> None:
        self.stage_id = _required_id(self.stage_id, "stage_id")
        self.name = _required_id(self.name, "stage name")
        if self.parent_stage_id is not None:
            self.parent_stage_id = _required_id(self.parent_stage_id, "parent_stage_id")
        self.started_at = _timestamp(self.started_at, "started_at")
        self.finished_at = _timestamp(self.finished_at, "finished_at")
        self.duration_seconds = _duration(self.duration_seconds, "duration_seconds")
        self.exclusive_duration_seconds = _duration(
            self.exclusive_duration_seconds, "exclusive_duration_seconds"
        )
        self.status = _status(self.status)
        self.error = _optional_text(self.error, "error")
        self.timing_error = _optional_text(self.timing_error, "timing_error")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "parent_stage_id": self.parent_stage_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "exclusive_duration_seconds": self.exclusive_duration_seconds,
            "status": self.status,
            "error": self.error,
            "timing_error": self.timing_error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetryStageRecord":
        record = _require_dict(value, "telemetry stage")
        return cls(
            stage_id=record["stage_id"],
            name=record["name"],
            parent_stage_id=record.get("parent_stage_id"),
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            duration_seconds=record.get("duration_seconds"),
            exclusive_duration_seconds=record.get("exclusive_duration_seconds"),
            status=record.get("status", "pending"),
            error=record.get("error"),
            timing_error=record.get("timing_error"),
        )


@dataclass
class TelemetrySegmentRecord:
    """A resumable segment within one logical attempt.

    Civil timestamps carry timezone offsets; ``duration_seconds`` is elapsed
    monotonic time for this segment, not CPU process time.
    """

    segment_id: str
    status: TelemetryStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    timing_error: str | None = None
    stages: list[TelemetryStageRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.segment_id = _required_id(self.segment_id, "segment_id")
        self.status = _status(self.status)
        self.started_at = _timestamp(self.started_at, "started_at")
        self.finished_at = _timestamp(self.finished_at, "finished_at")
        self.duration_seconds = _duration(self.duration_seconds, "duration_seconds")
        self.error = _optional_text(self.error, "error")
        self.timing_error = _optional_text(self.timing_error, "timing_error")
        if not isinstance(self.stages, list) or any(
            not isinstance(stage, TelemetryStageRecord) for stage in self.stages
        ):
            raise TelemetryRecordError("stages deve conter TelemetryStageRecord")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise TelemetryRecordError("stage_id duplicado no segment")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "segment_id": self.segment_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timing_error": self.timing_error,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetrySegmentRecord":
        record = _require_dict(value, "telemetry segment")
        stages = record.get("stages", [])
        if not isinstance(stages, list):
            raise TelemetryRecordError("stages deve ser uma lista")
        return cls(
            segment_id=record["segment_id"],
            status=record.get("status", "pending"),
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            duration_seconds=record.get("duration_seconds"),
            error=record.get("error"),
            timing_error=record.get("timing_error"),
            stages=[TelemetryStageRecord.from_dict(stage) for stage in stages],
        )


@dataclass
class TelemetryAttemptRecord:
    """One logical attempt; ``process_id`` is not the operating-system PID.

    ``pid`` is an observed OS value. Submitted/worker timestamps are civil
    times; queue and attempt durations, when present, are elapsed seconds from
    monotonic-clock deltas. Cleanup has its own status and error fields.
    """

    attempt_id: str
    process_id: str
    pid: int | None = None
    status: TelemetryStatus = "pending"
    submitted_at: str | None = None
    worker_started_at: str | None = None
    queue_duration_seconds: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    cleanup_status: CleanupStatus | None = None
    cleanup_error: str | None = None
    timing_error: str | None = None
    segments: list[TelemetrySegmentRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attempt_id = _required_id(self.attempt_id, "attempt_id")
        self.process_id = _required_id(self.process_id, "process_id")
        self.pid = _pid(self.pid)
        self.status = _status(self.status)
        self.submitted_at = _timestamp(self.submitted_at, "submitted_at")
        self.worker_started_at = _timestamp(self.worker_started_at, "worker_started_at")
        self.queue_duration_seconds = _duration(
            self.queue_duration_seconds, "queue_duration_seconds"
        )
        self.started_at = _timestamp(self.started_at, "started_at")
        self.finished_at = _timestamp(self.finished_at, "finished_at")
        self.duration_seconds = _duration(self.duration_seconds, "duration_seconds")
        self.error = _optional_text(self.error, "error")
        self.cleanup_error = _optional_text(self.cleanup_error, "cleanup_error")
        self.timing_error = _optional_text(self.timing_error, "timing_error")
        if self.cleanup_status is not None and (
            not isinstance(self.cleanup_status, str)
            or self.cleanup_status not in _ALLOWED_CLEANUP_STATUSES
        ):
            raise TelemetryRecordError("cleanup_status inválido")
        if not isinstance(self.segments, list) or any(
            not isinstance(segment, TelemetrySegmentRecord) for segment in self.segments
        ):
            raise TelemetryRecordError("segments deve conter TelemetrySegmentRecord")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise TelemetryRecordError("segment_id duplicado no attempt")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "attempt_id": self.attempt_id,
            "process_id": self.process_id,
            "pid": self.pid,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "worker_started_at": self.worker_started_at,
            "queue_duration_seconds": self.queue_duration_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "cleanup_status": self.cleanup_status,
            "cleanup_error": self.cleanup_error,
            "timing_error": self.timing_error,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetryAttemptRecord":
        record = _require_dict(value, "telemetry attempt")
        segments = record.get("segments", [])
        if not isinstance(segments, list):
            raise TelemetryRecordError("segments deve ser uma lista")
        return cls(
            attempt_id=record["attempt_id"],
            process_id=record["process_id"],
            pid=record.get("pid"),
            status=record.get("status", "pending"),
            submitted_at=record.get("submitted_at"),
            worker_started_at=record.get("worker_started_at"),
            queue_duration_seconds=record.get("queue_duration_seconds"),
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            duration_seconds=record.get("duration_seconds"),
            error=record.get("error"),
            cleanup_status=record.get("cleanup_status"),
            cleanup_error=record.get("cleanup_error"),
            timing_error=record.get("timing_error"),
            segments=[TelemetrySegmentRecord.from_dict(segment) for segment in segments],
        )

    def record_cleanup_stage(self, stage: TelemetryStageRecord) -> None:
        """Copy a caller-observed cleanup stage without changing attempt status."""
        self.__post_init__()
        if not isinstance(stage, TelemetryStageRecord):
            raise TelemetryRecordError("cleanup stage deve ser TelemetryStageRecord")
        stage.to_dict()
        if stage.status == "completed":
            self.cleanup_status = "completed"
            self.cleanup_error = None
        elif stage.status == "failed":
            self.cleanup_status = "failed"
            self.cleanup_error = stage.error
        else:
            raise TelemetryRecordError("cleanup stage precisa estar terminal")
        if stage.timing_error is not None:
            self.timing_error = (
                stage.timing_error
                if self.timing_error is None
                else f"{self.timing_error}; {stage.timing_error}"
            )


@dataclass
class TelemetryRunRecord:
    """Versioned telemetry history for one logical run; attempts stay ordered."""

    telemetry_schema_version: int
    run_id: str
    attempts: list[TelemetryAttemptRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.telemetry_schema_version = _schema_version(self.telemetry_schema_version)
        self.run_id = _required_id(self.run_id, "run_id")
        if not isinstance(self.attempts, list) or any(
            not isinstance(attempt, TelemetryAttemptRecord) for attempt in self.attempts
        ):
            raise TelemetryRecordError("attempts deve conter TelemetryAttemptRecord")
        attempt_ids = [attempt.attempt_id for attempt in self.attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise TelemetryRecordError("attempt_id duplicado")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "run_id": self.run_id,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetryRunRecord":
        record = _require_dict(value, "telemetry run")
        attempts = record.get("attempts", [])
        if not isinstance(attempts, list):
            raise TelemetryRecordError("attempts deve ser uma lista")
        return cls(
            telemetry_schema_version=_schema_version(record.get("telemetry_schema_version")),
            run_id=record["run_id"],
            attempts=[TelemetryAttemptRecord.from_dict(attempt) for attempt in attempts],
        )


@dataclass
class TelemetryEnvelope:
    """Optional session-level telemetry section for an experiment report."""

    telemetry_schema_version: int
    session_id: str
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        self.telemetry_schema_version = _schema_version(self.telemetry_schema_version)
        self.session_id = _required_id(self.session_id, "session_id")
        self.started_at = _timestamp(self.started_at, "started_at")
        self.finished_at = _timestamp(self.finished_at, "finished_at")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TelemetryEnvelope":
        record = _require_dict(value, "telemetry envelope")
        return cls(
            telemetry_schema_version=_schema_version(record.get("telemetry_schema_version")),
            session_id=record["session_id"],
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
        )


def _prepare_envelope_payload(
    value: TelemetryEnvelope | dict[str, Any] | None,
    *,
    component: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if isinstance(value, TelemetryEnvelope):
            payload = value.to_dict()
        elif isinstance(value, dict):
            validated = TelemetryEnvelope.from_dict(value)
            _ensure_telemetry_json_safe(value, component)
            payload = copy.deepcopy(value)
            for name in ("started_at", "finished_at"):
                if name in payload:
                    payload[name] = getattr(validated, name)
        else:
            raise TelemetryRecordError("envelope deve ser TelemetryEnvelope ou dicionário")
        _ensure_telemetry_json_safe(payload, component)
        return payload
    except Exception as error:
        _report_telemetry_diagnostic(component, error)
        return None


def merge_telemetry_run_records(
    previous: TelemetryRunRecord | dict[str, Any] | None,
    incoming: TelemetryRunRecord | dict[str, Any] | None,
) -> TelemetryRunRecord | None:
    """Merge a replacement without duplicating or rewriting terminal attempts."""
    prior = _coerce_run(previous)
    current = _coerce_run(incoming)
    if prior is None:
        return copy.deepcopy(current)
    if current is None:
        return copy.deepcopy(prior)
    if (
        prior.run_id != current.run_id
        or prior.telemetry_schema_version != current.telemetry_schema_version
    ):
        raise TelemetryRecordError("telemetry replacement pertence a outro run/schema")

    merged = copy.deepcopy(prior)
    by_attempt = {attempt.attempt_id: attempt for attempt in merged.attempts}
    for candidate in current.attempts:
        existing = by_attempt.get(candidate.attempt_id)
        if existing is None:
            appended = copy.deepcopy(candidate)
            merged.attempts.append(appended)
            by_attempt[appended.attempt_id] = appended
        elif existing.status not in {"completed", "failed"}:
            _merge_attempt(existing, candidate)
    return merged


def _coerce_run(
    value: TelemetryRunRecord | dict[str, Any] | None,
) -> TelemetryRunRecord | None:
    if value is None:
        return None
    if isinstance(value, TelemetryRunRecord):
        return TelemetryRunRecord.from_dict(value.to_dict())
    if isinstance(value, dict):
        _ensure_telemetry_json_safe(value, "telemetry_run")
        return TelemetryRunRecord.from_dict(value)
    raise TelemetryRecordError("telemetry de run deve ser record, dicionário ou None")


def _decode_run_payload(
    value: TelemetryRunRecord | dict[str, Any] | None,
    *,
    component: str,
) -> tuple[TelemetryRunRecord | None, BaseException | None]:
    if value is None:
        return None, None
    try:
        return _coerce_run(value), None
    except Exception as error:
        _report_telemetry_diagnostic(component, error)
        return None, error


def _merge_run_payloads(
    previous: TelemetryRunRecord | dict[str, Any] | None,
    incoming: TelemetryRunRecord | dict[str, Any] | None,
    *,
    component: str,
) -> dict[str, Any] | None:
    prior, _ = _decode_run_payload(previous, component=f"{component}.previous")
    current, _ = _decode_run_payload(incoming, component=f"{component}.incoming")
    if prior is None:
        if current is None:
            return None
        if isinstance(incoming, dict):
            return copy.deepcopy(incoming)
        return current.to_dict()
    if current is None:
        if isinstance(previous, dict):
            return copy.deepcopy(previous)
        return prior.to_dict()
    try:
        merged = merge_telemetry_run_records(prior, current)
        return None if merged is None else merged.to_dict()
    except Exception as error:
        _report_telemetry_diagnostic(component, error)
        if isinstance(previous, dict):
            return copy.deepcopy(previous)
        return prior.to_dict()


def _advance_status(previous: TelemetryStatus, incoming: TelemetryStatus) -> TelemetryStatus:
    if previous == "running" and incoming == "pending":
        return previous
    return incoming


def _merge_attempt(existing: TelemetryAttemptRecord, incoming: TelemetryAttemptRecord) -> None:
    if existing.process_id != incoming.process_id:
        raise TelemetryRecordError("process_id de um attempt não pode mudar")
    for name in (
        "pid", "submitted_at", "worker_started_at",
        "queue_duration_seconds", "started_at", "finished_at", "duration_seconds",
        "error", "cleanup_status", "cleanup_error", "timing_error",
    ):
        value = getattr(incoming, name)
        if value is not None:
            setattr(existing, name, value)
    existing.status = _advance_status(existing.status, incoming.status)
    by_segment = {segment.segment_id: segment for segment in existing.segments}
    for candidate in incoming.segments:
        current = by_segment.get(candidate.segment_id)
        if current is None:
            copied = copy.deepcopy(candidate)
            existing.segments.append(copied)
            by_segment[copied.segment_id] = copied
        elif current.status not in {"completed", "failed"}:
            _merge_segment(current, candidate)


def _merge_segment(existing: TelemetrySegmentRecord, incoming: TelemetrySegmentRecord) -> None:
    for name in (
        "started_at", "finished_at", "duration_seconds", "error", "timing_error"
    ):
        value = getattr(incoming, name)
        if value is not None:
            setattr(existing, name, value)
    existing.status = _advance_status(existing.status, incoming.status)
    by_stage = {stage.stage_id: stage for stage in existing.stages}
    for candidate in incoming.stages:
        current = by_stage.get(candidate.stage_id)
        if current is None:
            copied = copy.deepcopy(candidate)
            existing.stages.append(copied)
            by_stage[copied.stage_id] = copied
        elif current.status not in {"completed", "failed"}:
            for name in (
                "name", "parent_stage_id", "started_at", "finished_at",
                "duration_seconds", "exclusive_duration_seconds", "error", "timing_error",
            ):
                value = getattr(candidate, name)
                if value is not None:
                    setattr(current, name, value)
            current.status = _advance_status(current.status, candidate.status)
