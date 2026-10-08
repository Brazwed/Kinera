"""Explicit stdlib/Linux environment snapshots; importing does no collection."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kinera.monitor.telemetry.records import (
    EnvironmentSnapshotRecord,
    _report_telemetry_diagnostic,
)
from kinera.monitor.telemetry.timing import utc_now

_CPU_MODEL_KEYS = {"model name", "hardware", "processor", "cpu model"}


def _parse_cpu_model(cpuinfo_text: str) -> tuple[str, str] | None:
    candidates: dict[str, str] = {}
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        normalized_key = key.strip().casefold()
        model = value.strip()
        if separator and normalized_key in _CPU_MODEL_KEYS and model:
            if normalized_key == "processor" and model.isdecimal():
                continue
            candidates.setdefault(normalized_key, model)
    for key in ("model name", "hardware", "cpu model", "processor"):
        if key in candidates:
            return candidates[key], key
    return None


def _safe_detail(error: BaseException) -> str:
    return type(error).__name__


def collect_environment_snapshot(
    *,
    captured_at: str | None = None,
    proc_root: str | Path = "/proc",
    read_text: Callable[[Path], str] | None = None,
) -> EnvironmentSnapshotRecord | None:
    """Collect one opt-in snapshot. Missing fields become availability metadata.

    ``read_text`` is injectable so tests can use fixture content and never depend
    on the host's procfs. No system call, clock read, or file read occurs at import.
    """
    if captured_at is None:
        try:
            captured_at = utc_now().isoformat()
        except Exception as error:
            _report_telemetry_diagnostic("environment.captured_at", error)
            return None

    sources: dict[str, str] = {"captured_at": "datetime.now(timezone.utc)"}
    unavailable: dict[str, str] = {}
    not_applicable: dict[str, str] = {}
    diagnostics: list[str] = []

    def unavailable_field(name: str, reason: str, error: BaseException | None = None) -> None:
        unavailable[name] = reason
        if error is not None:
            diagnostics.append(f"{name}:{_safe_detail(error)}")

    def read_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace") if read_text is None else read_text(path)

    def text_field(name: str, source: str, reader: Callable[[], Any]) -> str | None:
        try:
            value = reader()
        except Exception as error:
            unavailable_field(name, _safe_detail(error), error)
            return None
        if not isinstance(value, str) or not value.strip():
            unavailable_field(name, "field missing or empty")
            return None
        normalized = value.strip()
        sources[name] = source
        return normalized

    python_version = text_field("python_version", "platform.python_version", platform.python_version)
    python_executable = text_field("python_executable", "sys.executable", lambda: sys.executable)
    os_system = text_field("os_system", "platform.system", platform.system)
    platform_version = text_field("platform_version", "platform.version", platform.version)
    architecture = text_field("architecture", "platform.machine", platform.machine)
    kernel_release = text_field("kernel_release", "platform.release", platform.release)

    os_release: str | None = None
    if os_system is None:
        unavailable_field("os_release", "OS system unavailable; distribution release source cannot be selected")
    elif os_system == "Linux":
        os_release_reader = getattr(platform, "freedesktop_os_release", None)
        if not callable(os_release_reader):
            unavailable_field("os_release", "platform.freedesktop_os_release is unavailable")
        else:
            try:
                release_info = os_release_reader()
                if not isinstance(release_info, dict):
                    raise ValueError("freedesktop_os_release returned non-dictionary data")
                for release_key in ("VERSION_ID", "PRETTY_NAME"):
                    release_value = release_info.get(release_key)
                    if isinstance(release_value, str) and release_value.strip():
                        os_release = release_value.strip()
                        sources["os_release"] = f"platform.freedesktop_os_release:{release_key}"
                        break
                if os_release is None:
                    unavailable_field("os_release", "VERSION_ID/PRETTY_NAME not present")
            except Exception as error:
                unavailable_field("os_release", _safe_detail(error), error)
    else:
        not_applicable["os_release"] = (
            "Linux distribution release source platform.freedesktop_os_release is not applicable"
        )

    try:
        logical_cpu_count = os.cpu_count()
    except Exception as error:
        logical_cpu_count = None
        unavailable_field("logical_cpu_count", _safe_detail(error), error)
    if isinstance(logical_cpu_count, bool) or not isinstance(logical_cpu_count, int) or logical_cpu_count <= 0:
        logical_cpu_count = None
        unavailable.setdefault("logical_cpu_count", "os.cpu_count returned no positive count")
    else:
        sources["logical_cpu_count"] = "os.cpu_count"

    cpu_affinity: list[int] | None = None
    affinity_reader = getattr(os, "sched_getaffinity", None)
    if not callable(affinity_reader):
        not_applicable["cpu_affinity"] = "os.sched_getaffinity is unavailable on this platform"
    else:
        try:
            affinity = affinity_reader(0)
            if not isinstance(affinity, (set, frozenset, list, tuple)) or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                for cpu in affinity
            ):
                raise ValueError("affinity API returned invalid CPU identifiers")
            if len(affinity) != len(set(affinity)):
                raise ValueError("affinity API returned duplicate CPU identifiers")
            cpu_affinity = sorted(affinity)
            sources["cpu_affinity"] = "os.sched_getaffinity(0)"
        except Exception as error:
            unavailable_field("cpu_affinity", _safe_detail(error), error)

    cpu_model: str | None = None
    if os_system is None:
        unavailable_field("cpu_model", "OS system unavailable; CPU model source cannot be selected")
    elif os_system == "Linux":
        try:
            cpuinfo_text = read_text_file(Path(proc_root) / "cpuinfo")
            if not isinstance(cpuinfo_text, str):
                raise ValueError("cpuinfo reader returned non-text data")
            cpu_model_result = _parse_cpu_model(cpuinfo_text)
            if cpu_model_result is None:
                unavailable_field("cpu_model", "recognized model field not present in /proc/cpuinfo")
            else:
                cpu_model, cpu_model_key = cpu_model_result
                sources["cpu_model"] = f"/proc/cpuinfo:{cpu_model_key}"
        except Exception as error:
            unavailable_field("cpu_model", _safe_detail(error), error)
    else:
        not_applicable["cpu_model"] = "CPU model parsing uses the Linux /proc/cpuinfo format"

    try:
        return EnvironmentSnapshotRecord(
            environment_schema_version=1,
            captured_at=captured_at,
            python_version=python_version,
            python_executable=python_executable,
            os_system=os_system,
            os_release=os_release,
            platform_version=platform_version,
            kernel_release=kernel_release,
            architecture=architecture,
            cpu_model=cpu_model,
            logical_cpu_count=logical_cpu_count,
            cpu_affinity=cpu_affinity,
            sources=sources,
            unavailable=unavailable,
            not_applicable=not_applicable,
            diagnostics=diagnostics,
        )
    except Exception as error:
        _report_telemetry_diagnostic("environment.snapshot", error)
        return None
