"""Explicit, side-effect-free-until-used wall/monotonic stage timing."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from kinera.monitor.telemetry.records import TelemetryStageRecord

_MAX_FLOAT_SECONDS = int(sys.float_info.max)
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _seconds_from_ns(delta_ns: int) -> float:
    if isinstance(delta_ns, bool) or not isinstance(delta_ns, int) or delta_ns < 0:
        raise ValueError("monotonic delta deve ser inteiro não negativo")
    whole_seconds, remainder_ns = divmod(delta_ns, _NANOSECONDS_PER_SECOND)
    if whole_seconds > _MAX_FLOAT_SECONDS:
        raise OverflowError("monotonic delta excede o intervalo representável em segundos")
    seconds = float(whole_seconds) + remainder_ns / _NANOSECONDS_PER_SECOND
    if not math.isfinite(seconds) or seconds < 0.0:
        raise OverflowError("duração monotônica não finita ou negativa")
    return seconds


def utc_now() -> datetime:
    """Return a timezone-aware UTC civil timestamp when explicitly called."""
    return datetime.now(timezone.utc)


@dataclass
class _StageFrame:
    record: TelemetryStageRecord
    started_monotonic_ns: int | None
    child_duration_ns: int = 0
    has_unmeasured_child: bool = False


class TimingSession:
    """Collect explicitly scoped stages using injected civil and monotonic clocks.

    No clock is read when importing or constructing this object. Callers supply
    every stage ID; nested stages retain a parent reference and report both
    inclusive and exclusive monotonic durations. Clock failures are recorded in
    ``timing_error`` and never replace an exception from the measured body.
    """

    def __init__(
        self,
        *,
        wall_clock: Callable[[], datetime] = utc_now,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._wall_clock = wall_clock
        self._monotonic_ns = monotonic_ns
        self._stack: list[_StageFrame] = []
        self._stage_ids: set[str] = set()
        self.stages: list[TelemetryStageRecord] = []

    def stage(
        self,
        *,
        stage_id: str,
        name: str,
        parent_stage_id: str | None = None,
    ) -> "_StageContext":
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise ValueError("stage_id deve ser identidade explícita não vazia")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name deve ser não vazio")
        if stage_id in self._stage_ids:
            raise ValueError(f"stage_id duplicado: {stage_id}")
        active_parent = self._stack[-1].record.stage_id if self._stack else None
        if parent_stage_id is not None and active_parent is not None:
            if parent_stage_id != active_parent:
                raise ValueError("parent_stage_id deve corresponder ao stage ativo")
        if parent_stage_id is None:
            parent_stage_id = active_parent
        self._stage_ids.add(stage_id)
        return _StageContext(self, stage_id, name, parent_stage_id)

    def _read_wall(self) -> str:
        value = self._wall_clock()
        if not isinstance(value, datetime):
            raise TypeError("wall_clock deve retornar datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall_clock deve retornar datetime com fuso")
        return value.astimezone(timezone.utc).isoformat()

    def _read_monotonic(self) -> int:
        value = self._monotonic_ns()
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("monotonic_ns deve retornar inteiro")
        return value


class _StageContext:
    def __init__(
        self,
        session: TimingSession,
        stage_id: str,
        name: str,
        parent_stage_id: str | None,
    ) -> None:
        self._session = session
        self._record = TelemetryStageRecord(
            stage_id=stage_id,
            name=name,
            parent_stage_id=parent_stage_id,
            status="running",
        )
        self._frame: _StageFrame | None = None

    def __enter__(self) -> TelemetryStageRecord:
        self._session.stages.append(self._record)
        try:
            self._record.started_at = self._session._read_wall()
        except Exception as exc:
            self._add_timing_error(exc)
        try:
            started = self._session._read_monotonic()
        except Exception as exc:
            started = None
            self._add_timing_error(exc)
        self._frame = _StageFrame(self._record, started)
        self._session._stack.append(self._frame)
        return self._record

    def __exit__(self, exc_type, exc, traceback) -> bool:
        frame = self._frame
        if frame is None:
            return False
        elapsed_ns: int | None = None
        try:
            elapsed_ns = self._finish_measurement(frame)
        except BaseException as telemetry_error:
            self._add_timing_error(telemetry_error)
            self._record.duration_seconds = None
            self._record.exclusive_duration_seconds = None
            elapsed_ns = None
        try:
            self._record.status = "failed" if exc_type is not None else "completed"
            if exc is not None:
                self._record.error = _exception_text(exc)
        except BaseException as telemetry_error:
            self._add_timing_error(telemetry_error)
        try:
            self._record.to_dict()
        except BaseException as serialization_error:
            self._add_timing_error(serialization_error)
            self._record.duration_seconds = None
            self._record.exclusive_duration_seconds = None
            elapsed_ns = None
        self._close_frame_best_effort(frame, elapsed_ns)
        return False

    def _finish_measurement(self, frame: _StageFrame) -> int | None:
        try:
            self._record.finished_at = self._session._read_wall()
        except BaseException as clock_error:
            self._add_timing_error(clock_error)
        try:
            finished = self._session._read_monotonic()
        except BaseException as clock_error:
            finished = None
            self._add_timing_error(clock_error)

        if frame.started_monotonic_ns is None or finished is None:
            return None
        elapsed_ns = finished - frame.started_monotonic_ns
        if elapsed_ns < 0:
            self._add_timing_error(ValueError("monotonic clock moved backwards"))
            return None
        try:
            self._record.duration_seconds = _seconds_from_ns(elapsed_ns)
        except BaseException as duration_error:
            self._record.duration_seconds = None
            self._record.exclusive_duration_seconds = None
            self._add_timing_error(duration_error)
            return None

        if frame.has_unmeasured_child:
            self._record.exclusive_duration_seconds = None
            return elapsed_ns
        exclusive_ns = elapsed_ns - frame.child_duration_ns
        if exclusive_ns < 0:
            self._record.exclusive_duration_seconds = None
            self._add_timing_error(ValueError("child duration exceeds parent"))
            return elapsed_ns
        try:
            self._record.exclusive_duration_seconds = _seconds_from_ns(exclusive_ns)
        except BaseException as duration_error:
            self._record.exclusive_duration_seconds = None
            self._add_timing_error(duration_error)
        return elapsed_ns

    def _close_frame_best_effort(
        self, frame: _StageFrame, elapsed_ns: int | None
    ) -> None:
        try:
            if self._session._stack and self._session._stack[-1] is frame:
                self._session._stack.pop()
            else:
                self._add_timing_error(RuntimeError("stage stack mismatch"))
                if frame in self._session._stack:
                    self._session._stack.remove(frame)
            if self._session._stack:
                parent = self._session._stack[-1]
                if elapsed_ns is None:
                    parent.has_unmeasured_child = True
                else:
                    parent.child_duration_ns += elapsed_ns
        except BaseException as telemetry_error:
            self._add_timing_error(telemetry_error)

    def _add_timing_error(self, error: BaseException) -> None:
        try:
            message = _exception_text(error)
            current = self._record.timing_error
            self._record.timing_error = (
                message if current is None else f"{current}; {message}"
            )
        except BaseException:
            try:
                self._record.timing_error = "telemetry finalization error; details unavailable"
            except BaseException:
                pass


def _exception_text(error: BaseException) -> str:
    try:
        detail = str(error)
    except BaseException:
        detail = "<unprintable exception>"
    return f"{type(error).__name__}: {detail}"
