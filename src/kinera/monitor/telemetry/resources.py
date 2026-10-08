"""Explicit passive process/Linux resource snapshots; no import-time probing."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

from kinera.monitor.telemetry.records import (
    ResourceSnapshotRecord,
    _report_telemetry_diagnostic,
)
from kinera.monitor.telemetry.timing import utc_now


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _exception_reason(error: BaseException) -> str:
    return type(error).__name__


def _parse_cgroup_membership(
    text: str,
) -> tuple[Literal["v1", "v2"] | None, str | None, bool, list[str]]:
    if not isinstance(text, str):
        raise ValueError("/proc/self/cgroup reader returned non-text data")
    v2_path: str | None = None
    v1_any_path: str | None = None
    v1_memory_path: str | None = None
    diagnostics: list[str] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            diagnostics.append("cgroup:malformed membership line")
            continue
        hierarchy, controllers, path = fields
        pure_path = PurePosixPath(path)
        if not pure_path.is_absolute() or ".." in pure_path.parts:
            diagnostics.append("cgroup:unsafe membership path")
            continue
        if hierarchy == "0" and controllers == "":
            v2_path = path
        else:
            v1_any_path = v1_any_path or path
            if "memory" in controllers.split(","):
                v1_memory_path = path
    if v2_path is not None:
        return "v2", v2_path, True, diagnostics
    if v1_any_path is not None:
        return "v1", v1_memory_path or v1_any_path, v1_memory_path is not None, diagnostics
    return None, None, False, diagnostics


def _unescape_mountinfo_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _relative_cgroup_parts(cgroup_path: str, mount_root: str) -> tuple[str, ...] | None:
    cgroup = PurePosixPath(cgroup_path)
    root = PurePosixPath(mount_root)
    if (
        not cgroup.is_absolute()
        or not root.is_absolute()
        or ".." in cgroup.parts
        or ".." in root.parts
        or cgroup.parts[: len(root.parts)] != root.parts
    ):
        return None
    return cgroup.parts[len(root.parts) :]


def _resolve_cgroup_directory(
    mountinfo_text: str,
    *,
    version: str,
    cgroup_path: str,
) -> tuple[Path | None, list[str]]:
    if not isinstance(mountinfo_text, str):
        raise ValueError("/proc/self/mountinfo reader returned non-text data")
    diagnostics: list[str] = []
    best_candidate: Path | None = None
    best_specificity = -1
    for line in mountinfo_text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            if separator < 6 or len(fields) <= separator + 3:
                raise ValueError("incomplete mountinfo line")
            mount_root = _unescape_mountinfo_path(fields[3])
            mount_point = _unescape_mountinfo_path(fields[4])
            mount_point_path = PurePosixPath(mount_point)
            if not mount_point_path.is_absolute() or ".." in mount_point_path.parts:
                diagnostics.append("mountinfo:unsafe mount point")
                continue
            filesystem = fields[separator + 1]
            source = fields[separator + 2]
            super_options = fields[separator + 3].split(",")
        except (ValueError, IndexError):
            diagnostics.append("mountinfo:malformed line")
            continue
        if version == "v2":
            if filesystem != "cgroup2":
                continue
        elif version == "v1":
            if filesystem != "cgroup" or not (
                "memory" in super_options or "memory" in source.split(",")
            ):
                continue
        else:
            return None, diagnostics
        relative_parts = _relative_cgroup_parts(cgroup_path, mount_root)
        if relative_parts is None:
            continue
        try:
            resolved_mount = Path(mount_point).resolve(strict=False)
            if not resolved_mount.is_dir():
                continue
            candidate = resolved_mount.joinpath(*relative_parts).resolve(strict=False)
            if not candidate.is_relative_to(resolved_mount) or not candidate.is_dir():
                diagnostics.append("cgroup:resolved path escaped or missing under discovered mount")
                continue
            specificity = len(PurePosixPath(mount_root).parts)
            if specificity > best_specificity:
                best_candidate = candidate
                best_specificity = specificity
        except (OSError, RuntimeError, ValueError) as error:
            diagnostics.append(f"cgroup:mount resolution failed:{_exception_reason(error)}")
    return best_candidate, diagnostics


def _safe_cgroup_file(directory: Path, filename: str) -> Path:
    if filename not in {"memory.current", "memory.peak", "memory.max"}:
        raise ValueError("unsupported cgroup memory filename")
    root = directory.resolve(strict=False)
    candidate = (root / filename).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError("cgroup metric path escaped the discovered mount")
    return candidate


def _parse_decimal(value: str, field: str) -> int:
    text = value.strip()
    if not text or not text.isdecimal():
        raise ValueError(f"{field} is not a non-negative decimal integer")
    return int(text)


def _parse_process_rss_bytes(status_text: str) -> int:
    if not isinstance(status_text, str):
        raise ValueError("/proc/self/status reader returned non-text data")
    for line in status_text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == "VmRSS":
            parts = value.split()
            if len(parts) != 2 or parts[1].casefold() != "kb":
                raise ValueError("VmRSS has an unsupported or missing unit")
            return _parse_decimal(parts[0], "VmRSS") * 1024
    raise ValueError("VmRSS field is missing")


def _read_ru_maxrss_kib() -> int:
    # resource is Unix-only; import only when a collector call explicitly asks.
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("ru_maxrss is not numeric")
    number = float(raw)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("ru_maxrss is not a non-negative integer")
    return int(number)


def collect_resource_snapshot(
    *,
    captured_at: str | None = None,
    proc_root: str | Path = "/proc",
    read_text: Callable[[Path], str] | None = None,
    get_ru_maxrss_kib: Callable[[], object] | None = None,
    system: str | None = None,
) -> ResourceSnapshotRecord | None:
    """Collect a single process/cgroup snapshot; no polling or writes are used.

    The path reader and ru_maxrss provider are injectable for deterministic
    fixture tests. The Linux procfs source is read only when this function is
    explicitly called.
    """
    if captured_at is None:
        try:
            captured_at = utc_now().isoformat()
        except Exception as error:
            _report_telemetry_diagnostic("resources.captured_at", error)
            return None
    proc_root = Path(proc_root)
    reader = _read_text if read_text is None else read_text
    system_name = system
    if system_name is None:
        try:
            import platform

            system_name = platform.system()
        except Exception:
            system_name = ""

    sources: dict[str, str] = {"captured_at": "datetime.now(timezone.utc)"}
    unavailable: dict[str, str] = {}
    not_applicable: dict[str, str] = {}
    diagnostics: list[str] = []

    process_rss_bytes: int | None = None
    if system_name == "Linux":
        try:
            process_rss_bytes = _parse_process_rss_bytes(reader(proc_root / "self" / "status"))
            sources["process_rss_bytes"] = "/proc/self/status:VmRSS"
        except Exception as error:
            unavailable["process_rss_bytes"] = _exception_reason(error)
            diagnostics.append(f"process_rss_bytes:{_exception_reason(error)}")
    elif system_name:
        not_applicable["process_rss_bytes"] = "process RSS source /proc/self/status is Linux-specific"
    else:
        unavailable["process_rss_bytes"] = "platform.system is unavailable; Linux procfs source was not probed"

    process_ru_maxrss_kib: int | None = None
    if system_name != "Linux":
        if system_name:
            not_applicable["process_ru_maxrss_kib"] = "Linux ru_maxrss unit contract is not used on this platform"
        else:
            unavailable["process_ru_maxrss_kib"] = "platform.system is unavailable; ru_maxrss was not queried"
    else:
        try:
            raw_ru_maxrss = (
                _read_ru_maxrss_kib()
                if get_ru_maxrss_kib is None
                else get_ru_maxrss_kib()
            )
            if isinstance(raw_ru_maxrss, bool) or not isinstance(raw_ru_maxrss, (int, float)):
                raise ValueError("ru_maxrss is not numeric")
            numeric_ru_maxrss = float(raw_ru_maxrss)
            if (
                not math.isfinite(numeric_ru_maxrss)
                or numeric_ru_maxrss < 0
                or not numeric_ru_maxrss.is_integer()
            ):
                raise ValueError("ru_maxrss is not a non-negative integer")
            process_ru_maxrss_kib = int(numeric_ru_maxrss)
            sources["process_ru_maxrss_kib"] = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
        except Exception as error:
            unavailable["process_ru_maxrss_kib"] = _exception_reason(error)
            diagnostics.append(f"process_ru_maxrss_kib:{_exception_reason(error)}")

    cgroup_version: Literal["v1", "v2"] | None = None
    cgroup_memory_current_bytes: int | None = None
    cgroup_memory_peak_bytes: int | None = None
    cgroup_memory_limit_bytes: int | None = None
    cgroup_memory_limit_unlimited: bool | None = None

    def unavailable_cgroup_field(
        name: str, reason: str, error: BaseException | None = None
    ) -> None:
        unavailable[name] = reason
        if error is not None:
            diagnostics.append(f"{name}:{_exception_reason(error)}")

    cgroup_memory_fields = (
        "cgroup_memory_current_bytes",
        "cgroup_memory_peak_bytes",
        "cgroup_memory_limit_bytes",
        "cgroup_memory_limit_unlimited",
    )
    if system_name == "Linux":
        try:
            membership_text = reader(proc_root / "self" / "cgroup")
            cgroup_version, membership_path, has_memory_controller, membership_diagnostics = (
                _parse_cgroup_membership(membership_text)
            )
            diagnostics.extend(membership_diagnostics)
        except Exception as error:
            cgroup_version = None
            membership_path = None
            has_memory_controller = False
            unavailable_cgroup_field("cgroup_version", _exception_reason(error), error)
    else:
        cgroup_version = None
        membership_path = None
        has_memory_controller = False

    if system_name != "Linux":
        reason = (
            "platform.system is unavailable; Linux cgroup source was not probed"
            if not system_name
            else "cgroup memory sources are Linux-specific"
        )
        for name in ("cgroup_version", *cgroup_memory_fields):
            if system_name:
                not_applicable[name] = reason
            else:
                unavailable.setdefault(name, reason)
    elif cgroup_version is None:
        cgroup_reason = unavailable.get("cgroup_version", "no cgroup membership detected")
        unavailable.setdefault("cgroup_version", cgroup_reason)
        for name in cgroup_memory_fields:
            unavailable_cgroup_field(name, cgroup_reason)
    else:
        sources["cgroup_version"] = "/proc/self/cgroup"
        if cgroup_version == "v1" and not has_memory_controller:
            for name in cgroup_memory_fields:
                not_applicable[name] = "cgroup v1 memory controller is not attached to this process"
        elif cgroup_version == "v1":
            for name in cgroup_memory_fields:
                unavailable_cgroup_field(name, "cgroup v1 memory collection is not implemented in T2a")
        else:
            mount_failure_reason: str | None = None
            try:
                mountinfo_text = reader(proc_root / "self" / "mountinfo")
                cgroup_directory, mount_diagnostics = _resolve_cgroup_directory(
                    mountinfo_text,
                    version=cgroup_version,
                    cgroup_path=membership_path or "",
                )
                diagnostics.extend(mount_diagnostics)
            except Exception as error:
                cgroup_directory = None
                mount_failure_reason = _exception_reason(error)
                unavailable_cgroup_field(
                    "cgroup_memory_current_bytes", mount_failure_reason, error
                )
            if cgroup_directory is None:
                reason = mount_failure_reason or "matching cgroup mount/path not found"
                for name in cgroup_memory_fields:
                    unavailable.setdefault(name, reason)
            else:
                metric_fields = {
                    "memory.current": "cgroup_memory_current_bytes",
                    "memory.peak": "cgroup_memory_peak_bytes",
                }
                for filename, field_name in metric_fields.items():
                    try:
                        metric_path = _safe_cgroup_file(cgroup_directory, filename)
                        amount = _parse_decimal(reader(metric_path), filename)
                        if field_name == "cgroup_memory_current_bytes":
                            cgroup_memory_current_bytes = amount
                        else:
                            cgroup_memory_peak_bytes = amount
                        sources[field_name] = f"cgroup-v2:{filename}"
                    except Exception as error:
                        unavailable_cgroup_field(field_name, _exception_reason(error), error)
                try:
                    limit_path = _safe_cgroup_file(cgroup_directory, "memory.max")
                    limit_text = reader(limit_path).strip()
                    if limit_text == "max":
                        cgroup_memory_limit_unlimited = True
                        sources["cgroup_memory_limit_unlimited"] = "cgroup-v2:memory.max"
                        not_applicable["cgroup_memory_limit_bytes"] = "memory.max is unlimited"
                    else:
                        cgroup_memory_limit_bytes = _parse_decimal(limit_text, "memory.max")
                        cgroup_memory_limit_unlimited = False
                        sources["cgroup_memory_limit_bytes"] = "cgroup-v2:memory.max"
                        sources["cgroup_memory_limit_unlimited"] = "cgroup-v2:memory.max"
                except Exception as error:
                    unavailable_cgroup_field("cgroup_memory_limit_bytes", _exception_reason(error), error)
                    unavailable_cgroup_field("cgroup_memory_limit_unlimited", _exception_reason(error), error)

    if cgroup_version is not None:
        sources["cgroup_version"] = "/proc/self/cgroup"

    try:
        return ResourceSnapshotRecord(
            resources_schema_version=1,
            captured_at=captured_at,
            process_rss_bytes=process_rss_bytes,
            process_ru_maxrss_kib=process_ru_maxrss_kib,
            cgroup_memory_current_bytes=cgroup_memory_current_bytes,
            cgroup_memory_peak_bytes=cgroup_memory_peak_bytes,
            cgroup_memory_limit_bytes=cgroup_memory_limit_bytes,
            cgroup_memory_limit_unlimited=cgroup_memory_limit_unlimited,
            cgroup_version=cgroup_version,
            sources=sources,
            unavailable=unavailable,
            not_applicable=not_applicable,
            diagnostics=diagnostics,
        )
    except Exception as error:
        _report_telemetry_diagnostic("resources.snapshot", error)
        return None
