"""P3b cost decomposition and independent-process stability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import benchmark_opencl_linear as baseline  # noqa: E402
from kinera.backends import get_backend  # noqa: E402
from kinera.backends.cpu.backend import CPUBackend  # noqa: E402
from kinera.nn.layers.linear import Linear  # noqa: E402

PROFILE: dict[str, Any] | None = None
PROFILE_PHASE = ""
PROFILE_STACK: list[dict[str, Any]] = []


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0}
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    mean = statistics.fmean(values)
    result = {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": mean,
        "cv_percent": (
            statistics.stdev(values) / mean * 100.0
            if len(values) > 1 and mean != 0.0
            else 0.0
        ),
        "range_percent_of_median": (
            (max(values) - min(values)) / abs(statistics.median(values)) * 100.0
            if statistics.median(values) != 0.0
            else 0.0
        ),
    }
    _validate_stats(result, values)
    return result


def _validate_stats(stats: dict[str, float | int], values: list[float]) -> None:
    """Check that a serialized stats row matches its source observations."""
    if not values:
        if stats != {"samples": 0}:
            raise AssertionError("empty statistics must have samples=0")
        return
    expected = {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": _quantile(values, 0.25),
        "q3_ms": _quantile(values, 0.75),
        "iqr_ms": _quantile(values, 0.75) - _quantile(values, 0.25),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "cv_percent": (
            statistics.stdev(values) / statistics.fmean(values) * 100.0
            if len(values) > 1 and statistics.fmean(values) != 0.0
            else 0.0
        ),
        "range_percent_of_median": (
            (max(values) - min(values)) / abs(statistics.median(values)) * 100.0
            if statistics.median(values) != 0.0
            else 0.0
        ),
    }
    if stats.get("samples") != expected["samples"]:
        raise AssertionError("statistics sample count does not match observations")
    for key, value in expected.items():
        if key == "samples":
            continue
        if not math.isclose(float(stats[key]), float(value), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"statistics field {key} is inconsistent")
    if (
        float(stats["iqr_ms"]) < 0
        or float(stats["min_ms"]) > float(stats["q1_ms"])
        or float(stats["q1_ms"]) > float(stats["median_ms"])
        or float(stats["median_ms"]) > float(stats["q3_ms"])
        or float(stats["q3_ms"]) > float(stats["max_ms"])
        or not float(stats["min_ms"]) <= float(stats["mean_ms"]) <= float(stats["max_ms"])
    ):
        raise AssertionError("statistics ordering is invalid")


def _stats_values(values: list[float]) -> dict[str, float | int]:
    stats = _stats_ms(values)
    if stats.get("samples", 0) == 0:
        return stats
    return {
        key.removesuffix("_ms"): value for key, value in stats.items()
    }


@contextmanager
def _event_timer(event: str) -> Iterator[None]:
    """Record inclusive and self time, subtracting direct nested timers once."""
    if PROFILE is None:
        yield
        return
    frame: dict[str, Any] = {
        "profile": PROFILE,
        "phase": PROFILE_PHASE,
        "event": event,
        "started_ns": time.perf_counter_ns(),
        "children_ns": 0,
    }
    PROFILE_STACK.append(frame)
    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - frame["started_ns"]
        if not PROFILE_STACK or PROFILE_STACK[-1] is not frame:
            raise AssertionError("profiler timer stack was corrupted")
        PROFILE_STACK.pop()
        exclusive_ns = elapsed_ns - frame["children_ns"]
        if exclusive_ns < 0:
            raise AssertionError(
                f"nested timers exceed inclusive time for {event}: "
                f"{frame['children_ns']} > {elapsed_ns} ns"
            )
        phases = frame["profile"].setdefault("events", {})
        phase = phases.setdefault(frame["phase"], {})
        phase.setdefault(event, []).append(
            {
                "inclusive_ns": elapsed_ns,
                "children_inclusive_ns": frame["children_ns"],
                "exclusive_ns": exclusive_ns,
            }
        )
        if PROFILE_STACK:
            PROFILE_STACK[-1]["children_ns"] += elapsed_ns


def _set_profile(profile: dict[str, Any] | None, phase: str = "") -> None:
    global PROFILE, PROFILE_PHASE
    if PROFILE_STACK:
        raise AssertionError("cannot switch profile while a timed call is active")
    PROFILE = profile
    PROFILE_PHASE = phase


def _sum_event(
    profile: dict[str, Any], phase: str, event: str, *, metric: str = "inclusive_ns"
) -> int:
    return sum(
        int(record[metric])
        for record in profile.get("events", {}).get(phase, {}).get(event, ())
    )


def _validate_profile_events(profile: dict[str, Any]) -> None:
    """Ensure every event has a valid inclusive/child/exclusive partition."""
    if PROFILE_STACK:
        raise AssertionError("profile validation found active timer scopes")
    for phase, events in profile.get("events", {}).items():
        for event, records in events.items():
            for record in records:
                inclusive = int(record["inclusive_ns"])
                children = int(record["children_inclusive_ns"])
                exclusive = int(record["exclusive_ns"])
                if min(inclusive, children, exclusive) < 0:
                    raise AssertionError(f"negative duration in {phase}.{event}")
                if children + exclusive != inclusive:
                    raise AssertionError(
                        f"inclusive/self times do not close for {phase}.{event}"
                    )


class ProfileLinear(Linear):
    """Linear subclass that times validation/preparation without source edits."""

    @classmethod
    def _validate_backend_array(
        cls, value: object, shape: tuple[int, ...], name: str
    ):
        if PROFILE is None:
            return super()._validate_backend_array(value, shape, name)
        with _event_timer("linear.validation"):
            return super()._validate_backend_array(value, shape, name)

    def _prepare_backend_array(
        self, value: object, shape: tuple[int, ...], name: str
    ):
        if PROFILE is None:
            return super()._prepare_backend_array(value, shape, name)
        with _event_timer("linear.prepare"):
            return super()._prepare_backend_array(value, shape, name)


def _patch_timed(owner: Any, name: str, event: str) -> Callable[[], None]:
    original = getattr(owner, name)

    def timed(*args, **kwargs):
        if PROFILE is None:
            return original(*args, **kwargs)
        with _event_timer(event):
            return original(*args, **kwargs)

    setattr(owner, name, timed)

    def restore() -> None:
        setattr(owner, name, original)

    return restore


def _install_tracing(cpu: CPUBackend, opencl: Any) -> list[Callable[[], None]]:
    restores: list[Callable[[], None]] = []
    for backend, prefix in ((cpu, "cpu"), (opencl, "opencl")):
        for name in ("buffer_from_host", "buffer_to_host"):
            restores.append(
                _patch_timed(backend, name, f"{prefix}.{name}")
            )
        for name in ("linear_forward", "linear_backward", "linear_step"):
            restores.append(
                _patch_timed(
                    backend,
                    name,
                    f"{prefix}.{name.removeprefix('linear_')}",
                )
            )

    for owner, name, event in (
        (opencl, "_require_buffer", "opencl.require_buffer"),
        (opencl, "_get_program", "opencl.program_lookup"),
        (opencl.runtime, "require_device", "opencl.device_check"),
        (opencl.memory, "allocate", "opencl.allocate"),
        (opencl.memory, "release", "opencl.release"),
        (opencl.runtime.api, "create_kernel", "opencl.create_kernel"),
        (opencl.runtime.api, "set_kernel_buffer", "opencl.set_kernel_arg"),
        (opencl.runtime.api, "set_kernel_int", "opencl.set_kernel_arg"),
        (opencl.runtime.api, "set_kernel_double", "opencl.set_kernel_arg"),
        (opencl.runtime.api, "enqueue_1d", "opencl.enqueue_and_clFinish"),
        (opencl.runtime.api, "release_kernel", "opencl.release_kernel"),
        (opencl.runtime.api, "write_buffer", "opencl.write_h2d"),
        (opencl.runtime.api, "read_buffer", "opencl.read_d2h"),
    ):
        restores.append(_patch_timed(owner, name, event))
    return restores


def _timed_phase(
    profile: dict[str, Any], phase: str, function: Callable[[], Any]
) -> tuple[int, Any]:
    _set_profile(profile, phase)
    started = time.perf_counter_ns()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter_ns() - started
        profile.setdefault("manual_ns", {})[phase] = elapsed
    return elapsed, result


def _timed_layer(
    profile: dict[str, Any], layer: ProfileLinear, method: str, *args: Any
) -> tuple[int, Any]:
    return _timed_phase(profile, method, lambda: getattr(layer, method)(*args))


def _resident_cycle(
    layer: ProfileLinear,
    inputs: Any,
    grad_output: Any,
    *,
    opencl: bool,
    profile: dict[str, Any] | None,
) -> int:
    started = time.perf_counter_ns()
    previous_output = layer.output
    if profile is not None:
        _, output = _timed_layer(profile, layer, "forward", inputs)
        if opencl and previous_output is not None:
            _timed_phase(profile, "close_previous_output", previous_output.close)
        _, grad_inputs = _timed_layer(profile, layer, "backward", grad_output)
        _timed_layer(profile, layer, "step", baseline.LEARNING_RATE)
        if opencl:
            _timed_phase(profile, "close_grad_inputs", grad_inputs.close)
            _timed_phase(
                profile,
                "cycle_local_cleanup",
                lambda: None,
            )
            del output, grad_inputs, previous_output
        else:
            _timed_phase(
                profile,
                "cycle_local_cleanup",
                lambda: None,
            )
            del output, grad_inputs, previous_output
    else:
        output = layer.forward(inputs)
        if opencl and previous_output is not None:
            previous_output.close()
        grad_inputs = layer.backward(grad_output)
        layer.step(baseline.LEARNING_RATE)
        if opencl:
            grad_inputs.close()
            del output
        else:
            del output, grad_inputs
        del previous_output
    return time.perf_counter_ns() - started


def _profile_resident_case(
    case: dict[str, Any],
    *,
    order: str,
    warmups: int,
    repetitions: int,
    include_samples: bool = False,
) -> dict[str, Any]:
    cpu = CPUBackend()
    opencl = get_backend("opencl")
    device = opencl.runtime.require_device(0, require_fp64=True)
    if device.pci_bdf != "0000:03:00.0" or "Radeon RX 550" not in device.name:
        opencl.close()
        raise RuntimeError(f"Expected RX 550; found {device.name} ({device.pci_bdf})")
    _ = opencl.memory.queue
    for _, filename in baseline.PROGRAMS:
        opencl._get_program(filename)

    layers: dict[str, ProfileLinear] = {}
    inputs_gpu = None
    grad_output_gpu = None
    try:
        layers["cpu"] = ProfileLinear(
            case["input_size"], case["output_size"], seed=case["seed"], backend=cpu
        )
        layers["opencl"] = ProfileLinear(
            case["input_size"], case["output_size"], seed=case["seed"], backend=opencl
        )
        inputs_gpu = opencl.buffer_from_host(
            case["inputs"], shape=(case["input_size"],)
        )
        grad_output_gpu = opencl.buffer_from_host(
            case["grad_output"], shape=(case["output_size"],)
        )

        restores = _install_tracing(cpu, opencl)
        try:
            _set_profile(None)
            for index in range(warmups):
                first = order if index % 2 == 0 else _other_order(order)
                backends = ("cpu", "opencl") if first == "cpu-first" else ("opencl", "cpu")
                for backend_name in backends:
                    _resident_cycle(
                        layers[backend_name],
                        case["inputs"] if backend_name == "cpu" else inputs_gpu,
                        case["grad_output"] if backend_name == "cpu" else grad_output_gpu,
                        opencl=backend_name == "opencl",
                        profile=None,
                    )

            samples: dict[str, list[dict[str, float]]] = {
                "cpu": [],
                "opencl": [],
            }
            for index in range(repetitions):
                first = order if index % 2 == 0 else _other_order(order)
                backends = ("cpu", "opencl") if first == "cpu-first" else ("opencl", "cpu")
                for backend_name in backends:
                    profile: dict[str, Any] = {"events": {}, "manual_ns": {}}
                    _set_profile(profile)
                    cycle_ns = _resident_cycle(
                        layers[backend_name],
                        case["inputs"] if backend_name == "cpu" else inputs_gpu,
                        case["grad_output"] if backend_name == "cpu" else grad_output_gpu,
                        opencl=backend_name == "opencl",
                        profile=profile,
                    )
                    samples[backend_name].append(
                    _resident_sample_metrics(profile, cycle_ns, backend_name)
                    )
            summaries = {
                backend_name: _summarize_records(records)
                for backend_name, records in samples.items()
            }
            if include_samples:
                return {
                    backend_name: {
                        "summary": summaries[backend_name],
                        "samples": records,
                    }
                    for backend_name, records in samples.items()
                }
            return summaries
        finally:
            _set_profile(None)
            for restore in reversed(restores):
                restore()
    finally:
        _set_profile(None)
        for layer in layers.values():
            layer.close()
        if grad_output_gpu is not None:
            grad_output_gpu.close()
        if inputs_gpu is not None:
            inputs_gpu.close()
        opencl.close()


def _other_order(order: str) -> str:
    return "opencl-first" if order == "cpu-first" else "cpu-first"


_OPENCL_COMPONENT_EVENTS = {
    "buffer_validation": "opencl.require_buffer",
    "device_check": "opencl.device_check",
    "allocation": "opencl.allocate",
    "program_lookup": "opencl.program_lookup",
    "memory_release": "opencl.release",
    "kernel_create": "opencl.create_kernel",
    "kernel_argument_setup": "opencl.set_kernel_arg",
    "enqueue_and_clFinish": "opencl.enqueue_and_clFinish",
    "kernel_release": "opencl.release_kernel",
}


def _opencl_components(
    phase_events: dict[str, list[dict[str, int]]], backend_ns: int
) -> tuple[dict[str, int], int]:
    """Return disjoint profiler self-time components and the backend residual."""
    components = {
        label: sum(int(row["exclusive_ns"]) for row in phase_events.get(event, ()))
        for label, event in _OPENCL_COMPONENT_EVENTS.items()
    }
    measured_ns = sum(components.values())
    residual_ns = backend_ns - measured_ns
    if residual_ns < 0:
        raise AssertionError(
            "exclusive OpenCL profiler components exceed backend method time: "
            f"components={measured_ns} ns, backend={backend_ns} ns"
        )
    if measured_ns + residual_ns != backend_ns:
        raise AssertionError("OpenCL component partition does not close")
    return components, residual_ns


def _resident_sample_metrics(
    profile: dict[str, Any], cycle_ns: int, backend_name: str
) -> dict[str, float]:
    _validate_profile_events(profile)
    metrics: dict[str, float] = {"cycle_total_ms": cycle_ns / 1e6}
    operation_wall = 0
    backend_wall = 0
    validation = 0
    host_prep = 0
    for operation in ("forward", "backward", "step"):
        phase_events = profile.get("events", {}).get(operation, {})
        wall_ns = profile.get("manual_ns", {}).get(operation, 0)
        backend_event = f"{backend_name}.{operation}"
        backend_ns = sum(
            int(record["inclusive_ns"])
            for record in phase_events.get(backend_event, ())
        )
        validation_ns = sum(
            int(record["inclusive_ns"])
            for record in phase_events.get("linear.validation", ())
        )
        prepare_ns = sum(
            int(record["inclusive_ns"])
            for record in phase_events.get("linear.prepare", ())
        )
        from_host_ns = sum(
            int(record["inclusive_ns"])
            for event in (
                "cpu.buffer_from_host",
                "opencl.buffer_from_host",
            )
            for record in phase_events.get(event, ())
        )
        operation_wall += wall_ns
        backend_wall += backend_ns
        validation += validation_ns
        host_prep += from_host_ns
        metrics[f"{operation}_layer_wall_ms"] = wall_ns / 1e6
        metrics[f"{operation}_backend_method_ms"] = backend_ns / 1e6
        metrics[f"{operation}_linear_validation_ms"] = validation_ns / 1e6
        metrics[f"{operation}_linear_prepare_inclusive_ms"] = prepare_ns / 1e6
        metrics[f"{operation}_buffer_from_host_ms"] = from_host_ns / 1e6

        if backend_name == "opencl":
            opencl_phases, residual_ns = _opencl_components(
                phase_events, backend_ns
            )
            for name, value in opencl_phases.items():
                metrics[f"{operation}_opencl_{name}_exclusive_ms"] = value / 1e6
            metrics[f"{operation}_opencl_backend_residual_ms"] = residual_ns / 1e6
            metrics[f"{operation}_opencl_accounted_backend_ms"] = (
                sum(opencl_phases.values()) + residual_ns
            ) / 1e6
            if not math.isclose(
                sum(opencl_phases.values()) + residual_ns,
                backend_ns,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise AssertionError("OpenCL components and residual do not equal backend time")

    housekeeping_ns = cycle_ns - operation_wall
    metrics["cycle_layer_methods_sum_ms"] = operation_wall / 1e6
    metrics["cycle_backend_methods_sum_ms"] = backend_wall / 1e6
    metrics["cycle_linear_validation_sum_ms"] = validation / 1e6
    metrics["cycle_buffer_from_host_sum_ms"] = host_prep / 1e6
    cycle_residual_ns = cycle_ns - backend_wall - validation - host_prep
    if cycle_residual_ns < 0:
        raise AssertionError("resident cycle components exceed the cycle wall")
    metrics["cycle_layer_residual_after_backend_validation_and_copy_ms"] = (
        cycle_residual_ns / 1e6
    )
    metrics["cycle_housekeeping_outside_layer_methods_ms"] = housekeeping_ns / 1e6
    if operation_wall > cycle_ns:
        raise AssertionError("layer method walls exceed the resident cycle wall")
    if backend_wall + validation + host_prep > cycle_ns:
        raise AssertionError(
            "sequential backend, validation, and host-copy times exceed cycle wall"
        )
    if backend_wall + validation + host_prep + cycle_residual_ns != cycle_ns:
        raise AssertionError("resident cycle partition does not close")
    return metrics


def _summarize_records(records: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({key for record in records for key in record})
    return {
        key: _stats_ms([record[key] for record in records if key in record])
        for key in keys
    }


def _profile_end_to_end_sample(
    case: dict[str, Any], *, backend_name: str, backend: Any
) -> dict[str, float]:
    profile: dict[str, Any] = {"events": {}, "manual_ns": {}}
    started = time.perf_counter_ns()
    layer: ProfileLinear | None = None
    inputs_gpu = None
    grad_output_gpu = None
    grad_inputs = None
    try:
        _, layer = _timed_phase(
            profile,
            "construction",
            lambda: ProfileLinear(
                case["input_size"],
                case["output_size"],
                seed=case["seed"],
                backend=backend,
            ),
        )
        if backend_name == "opencl":
            _, inputs_gpu = _timed_phase(
                profile,
                "input_preparation_input",
                lambda: backend.buffer_from_host(
                    case["inputs"], shape=(case["input_size"],), dtype="float64"
                ),
            )
            _, grad_output_gpu = _timed_phase(
                profile,
                "input_preparation_grad_output",
                lambda: backend.buffer_from_host(
                    case["grad_output"],
                    shape=(case["output_size"],),
                    dtype="float64",
                ),
            )

        forward_inputs = case["inputs"] if backend_name == "cpu" else inputs_gpu
        backward_grad = (
            case["grad_output"] if backend_name == "cpu" else grad_output_gpu
        )
        output = _timed_layer(profile, layer, "forward", forward_inputs)[1]
        grad_inputs = _timed_layer(profile, layer, "backward", backward_grad)[1]
        _timed_layer(profile, layer, "step", baseline.LEARNING_RATE)

        if backend_name == "opencl":
            _, materialized = _timed_phase(
                profile,
                "materialization",
                lambda: (
                    backend.buffer_to_host(output),
                    backend.buffer_to_host(layer.grad_weights),
                    backend.buffer_to_host(layer.grad_bias),
                    backend.buffer_to_host(grad_inputs),
                    backend.buffer_to_host(layer.weights),
                    backend.buffer_to_host(layer.bias),
                ),
            )
        else:
            materialized = (
                output,
                layer.grad_weights,
                layer.grad_bias,
                grad_inputs,
                layer.weights,
                layer.bias,
            )
            profile["manual_ns"]["materialization"] = 0

        _, checksum = _timed_phase(
            profile,
            "consumption",
            lambda: math.fsum(
                value
                for group in materialized
                for value in baseline._flatten(group)
            ),
        )
        del checksum
    finally:
        def cleanup() -> None:
            if layer is not None:
                layer.close()
            for buffer in (grad_inputs, grad_output_gpu, inputs_gpu):
                if buffer is not None and hasattr(buffer, "close"):
                    buffer.close()

        _timed_phase(profile, "cleanup", cleanup)
        profile.setdefault("manual_ns", {})["total"] = time.perf_counter_ns() - started

    _validate_profile_events(profile)
    metrics = {
        f"{phase}_wall_ms": elapsed / 1e6
        for phase, elapsed in profile["manual_ns"].items()
    }
    metrics["input_preparation_wall_ms"] = sum(
        profile["manual_ns"].get(phase, 0)
        for phase in (
            "input_preparation_input",
            "input_preparation_grad_output",
        )
    ) / 1e6
    for phase, events in profile.get("events", {}).items():
        for event, records in events.items():
            for timer_kind in ("inclusive_ns", "exclusive_ns"):
                metrics[f"{phase}_{event}_{timer_kind.removesuffix('_ns')}_ms"] = (
                    sum(int(record[timer_kind]) for record in records) / 1e6
                )

    materialization_wall = profile["manual_ns"].get("materialization", 0)
    d2h = _sum_event(profile, "materialization", "opencl.read_d2h")
    metrics["materialization_python_and_host_residual_ms"] = max(
        0.0, materialization_wall - d2h
    ) / 1e6
    for operation in ("forward", "backward", "step"):
        phase = profile.get("events", {}).get(operation, {})
        metrics[f"{operation}_backend_method_ms"] = sum(
            int(record["inclusive_ns"])
            for record in phase.get(f"{backend_name}.{operation}", ())
        ) / 1e6
    return metrics


def _profile_case(
    case: dict[str, Any],
    *,
    order: str,
    warmups: int,
    repetitions: int,
    include_samples: bool = False,
) -> dict[str, Any]:
    cpu = CPUBackend()
    opencl = get_backend("opencl")
    device = opencl.runtime.require_device(0, require_fp64=True)
    if device.pci_bdf != "0000:03:00.0" or "Radeon RX 550" not in device.name:
        opencl.close()
        raise RuntimeError(f"Expected RX 550; found {device.name} ({device.pci_bdf})")
    _ = opencl.memory.queue
    for _, filename in baseline.PROGRAMS:
        opencl._get_program(filename)
    restores = _install_tracing(cpu, opencl)
    try:
        samples = {"cpu": [], "opencl": []}
        for index in range(warmups):
            first = order if index % 2 == 0 else _other_order(order)
            backends = ("cpu", "opencl") if first == "cpu-first" else ("opencl", "cpu")
            for backend_name in backends:
                _set_profile(None)
                _profile_end_to_end_sample(
                    case,
                    backend_name=backend_name,
                    backend=cpu if backend_name == "cpu" else opencl,
                )
        for index in range(repetitions):
            first = order if index % 2 == 0 else _other_order(order)
            backends = ("cpu", "opencl") if first == "cpu-first" else ("opencl", "cpu")
            for backend_name in backends:
                _set_profile(None)
                sample = _profile_end_to_end_sample(
                    case,
                    backend_name=backend_name,
                    backend=cpu if backend_name == "cpu" else opencl,
                )
                samples[backend_name].append(sample)
        summaries = {
            backend_name: _summarize_records(records)
            for backend_name, records in samples.items()
        }
        if include_samples:
            return {
                backend_name: {
                    "summary": summaries[backend_name],
                    "samples": records,
                }
                for backend_name, records in samples.items()
            }
        return summaries
    finally:
        _set_profile(None)
        for restore in reversed(restores):
            restore()
        opencl.close()


def _profile_cost_decomposition(
    *, order: str, warmups: int, repetitions: int, include_samples: bool = False
) -> dict[str, Any]:
    cases = []
    for index, (label, output_size, input_size) in enumerate(baseline.CASES):
        case = baseline._case_data(output_size, input_size, seed=52052 + index)
        cases.append(
            {
                "label": label,
                "weights_shape": [output_size, input_size],
                "resident": _profile_resident_case(
                    case,
                    order=order,
                    warmups=warmups,
                    repetitions=repetitions,
                    include_samples=include_samples,
                ),
                "end_to_end": _profile_case(
                    case,
                    order=order,
                    warmups=warmups,
                    repetitions=repetitions,
                    include_samples=include_samples,
                ),
            }
        )
    return {
        "profile_repetitions_per_backend_per_shape": repetitions,
        "warmups_per_backend_per_shape": warmups,
        "within_process_order": "CPU/OpenCL pair order alternates; first pair follows process order",
        "cases": cases,
    }


def _ordered_pair_measure(
    cpu_call: Callable[[], Any],
    opencl_call: Callable[[], Any],
    *,
    warmups: int,
    repetitions: int,
    first_order: str,
    opencl_cleanup: Callable[[Any], None] | None = None,
    before_cpu: Callable[[], None] | None = None,
    before_opencl: Callable[[], None] | None = None,
) -> dict[str, Any]:
    def invoke(
        function: Callable[[], Any],
        prepare: Callable[[], None] | None,
        cleanup: Callable[[Any], None] | None,
    ) -> int:
        if prepare is not None:
            prepare()
        started = time.perf_counter_ns()
        result = function()
        elapsed = time.perf_counter_ns() - started
        if cleanup is not None:
            cleanup(result)
        return elapsed

    for index in range(warmups):
        first = first_order if index % 2 == 0 else _other_order(first_order)
        if first == "cpu-first":
            invoke(cpu_call, before_cpu, None)
            invoke(opencl_call, before_opencl, opencl_cleanup)
        else:
            invoke(opencl_call, before_opencl, opencl_cleanup)
            invoke(cpu_call, before_cpu, None)

    cpu_samples: list[int] = []
    opencl_samples: list[int] = []
    for index in range(repetitions):
        first = first_order if index % 2 == 0 else _other_order(first_order)
        if first == "cpu-first":
            cpu_samples.append(invoke(cpu_call, before_cpu, None))
            opencl_samples.append(invoke(opencl_call, before_opencl, opencl_cleanup))
        else:
            opencl_samples.append(invoke(opencl_call, before_opencl, opencl_cleanup))
            cpu_samples.append(invoke(cpu_call, before_cpu, None))
    ratios = [cpu / gpu for cpu, gpu in zip(cpu_samples, opencl_samples)]
    ratio_stats = _stats_values(ratios)
    return {
        "cpu": baseline._duration_stats(cpu_samples),
        "opencl": baseline._duration_stats(opencl_samples),
        "paired_cpu_over_opencl_speedup": ratio_stats,
        "speedup_definition": "CPU call / paired OpenCL call; >1 favors OpenCL",
        "first_order": first_order,
    }


def _child_main(args: argparse.Namespace) -> int:
    original_pair_measure = baseline._paired_measure
    baseline._paired_measure = lambda cpu_call, opencl_call, **kwargs: _ordered_pair_measure(
        cpu_call,
        opencl_call,
        first_order=args.order,
        **kwargs,
    )
    try:
        baseline_report = baseline.run_benchmark(
            warmups=args.warmups,
            repetitions=args.repetitions,
            setup_samples=args.setup_samples,
        )
    finally:
        baseline._paired_measure = original_pair_measure

    profile = _profile_cost_decomposition(
        order=args.order,
        warmups=args.profile_warmups,
        repetitions=args.profile_repetitions,
    )
    result = {
        "process": {
            "run_index": args.child_index,
            "first_order": args.order,
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "independent_process": True,
        },
        "baseline": baseline_report,
        "cost_decomposition": profile,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _get_operation_metrics(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "forward": case["resident_backend_operations"]["forward"],
        "backward": case["resident_backend_operations"]["backward"],
        "step": case["resident_backend_operations"]["step"],
        "linear_cycle_resident": case["integrated_linear_cycle_resident"],
        "linear_cycle_end_to_end": case["steady_state_end_to_end_linear_cycle"],
    }


def _between_process_stability(
    process_results: list[dict[str, Any]], baseline_report: dict[str, Any] | None
) -> dict[str, Any]:
    output = {}
    for case_index, process_case in enumerate(process_results[0]["baseline"]["cases"]):
        label = process_case["label"]
        per_case = {}
        for metric_name in _get_operation_metrics(process_case):
            rows = []
            for result in process_results:
                run_case = result["baseline"]["cases"][case_index]
                metric = _get_operation_metrics(run_case)[metric_name]
                rows.append(
                    {
                        "cpu_median_ms": metric["cpu"]["median_ms"],
                        "opencl_median_ms": metric["opencl"]["median_ms"],
                        "paired_speedup": metric[
                            "paired_cpu_over_opencl_speedup"
                        ]["median"],
                    }
                )
            row_summary = {}
            for key in rows[0]:
                row_summary[key] = (
                    _stats_values([row[key] for row in rows])
                    if key == "paired_speedup"
                    else _stats_ms([row[key] for row in rows])
                )
            if baseline_report is not None:
                base_case = baseline_report["cases"][case_index]
                base_metric = _get_operation_metrics(base_case)[metric_name]
                for backend_name in ("cpu", "opencl"):
                    baseline_ms = base_metric[backend_name]["median_ms"]
                    measured_ms = row_summary[f"{backend_name}_median_ms"]["median_ms"]
                    row_summary[f"{backend_name}_delta_vs_p3_percent"] = (
                        (measured_ms / baseline_ms - 1.0) * 100.0
                        if baseline_ms != 0.0
                        else None
                    )
            per_case[metric_name] = {
                "independent_process_medians": row_summary,
                "per_process": rows,
            }
        output[label] = per_case
    return output


def _between_process_costs(process_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case_index, process_case in enumerate(
        process_results[0]["cost_decomposition"]["cases"]
    ):
        label = process_case["label"]
        case_output: dict[str, Any] = {}
        for profile_name in ("resident", "end_to_end"):
            backend_output = {}
            for backend_name in ("cpu", "opencl"):
                sample = process_case[profile_name][backend_name]
                metrics = {}
                for metric_name in sample:
                    values = []
                    for result in process_results:
                        run_case = result["cost_decomposition"]["cases"][case_index]
                        values.append(
                            run_case[profile_name][backend_name][metric_name][
                                "median_ms"
                            ]
                        )
                    metrics[metric_name] = _stats_ms(values)
                backend_output[backend_name] = metrics
            case_output[profile_name] = backend_output
        output[label] = case_output
    return output


def _run_parent(args: argparse.Namespace) -> int:
    if args.runs < 4 or args.runs % 2:
        raise ValueError("--runs must be an even number >= 4 to balance start order")
    if args.warmups < 1 or args.repetitions < 5 or args.setup_samples < 3:
        raise ValueError("baseline requires warmups >=1, reps >=5, setup samples >=3")
    if args.profile_warmups < 1 or args.profile_repetitions < 3:
        raise ValueError("profile requires warmups >=1 and repetitions >=3")

    environment = os.environ.copy()
    environment.setdefault("RUSTICL_ENABLE", "radeonsi")
    environment.setdefault("RUSTICL_FEATURES", "fp64")
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_path, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    process_results = []
    log_paths = []
    for index in range(args.runs):
        order = "cpu-first" if index % 2 == 0 else "opencl-first"
        print(f"P3b process {index + 1}/{args.runs}: {order}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-index",
                str(index + 1),
                "--order",
                order,
                "--warmups",
                str(args.warmups),
                "--repetitions",
                str(args.repetitions),
                "--setup-samples",
                str(args.setup_samples),
                "--profile-warmups",
                str(args.profile_warmups),
                "--profile-repetitions",
                str(args.profile_repetitions),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path = Path(f"/tmp/kinera-issue52-p3b-run-{index + 1}.stderr")
        log_path.write_text(completed.stderr, encoding="utf-8")
        log_paths.append(str(log_path))
        if completed.returncode != 0:
            raise RuntimeError(
                f"child process {index + 1} failed ({completed.returncode}); "
                f"stderr saved in {log_path}: {completed.stderr[-3000:]}"
            )
        process_results.append(json.loads(completed.stdout))

    baseline_report = None
    if args.baseline_json is not None and args.baseline_json.exists():
        baseline_report = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    baseline_sha = hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest()
    output = {
        "schema_version": 1,
        "phase": "issue-52-p3b-cost-decomposition",
        "baseline_script_sha256": baseline_sha,
        "analysis_script_sha256": script_sha,
        "baseline_p3_json": str(args.baseline_json) if baseline_report else None,
        "protocol": {
            "independent_processes": args.runs,
            "process_first_order": [result["process"]["first_order"] for result in process_results],
            "baseline_warmups": args.warmups,
            "baseline_timed_repetitions_per_process": args.repetitions,
            "baseline_setup_compile_samples_per_process": args.setup_samples,
            "profile_warmups_per_backend_per_shape": args.profile_warmups,
            "profile_repetitions_per_backend_per_shape_per_process": args.profile_repetitions,
            "process_order": "Each fresh subprocess reverses which backend is measured first; within each measurement, pair order alternates.",
        },
        "process_results": process_results,
        "between_process_stability": _between_process_stability(
            process_results, baseline_report
        ),
        "between_process_cost_decomposition": _between_process_costs(process_results),
        "stderr_logs": log_paths,
        "limitations": [
            "Each process uses the same fixed fixtures and P3 semantics; process medians vary due to host/runtime state.",
            "Timing wrappers execute only in this analysis process and add instrumentation overhead; component sums are diagnostic, not an exact physical partition.",
            "OpenCL enqueue timing includes clFinish; raw device execution and enqueue are not separately observable without queue profiling.",
            "Driver cache, clocks, thermals, system activity, and GPU-global memory/utilization readings are not controlled.",
        ],
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"P3b complete: {args.runs} independent processes; JSON={args.output}")
    return 0


def _run_diagnostic(args: argparse.Namespace) -> int:
    if args.profile_warmups < 1 or args.profile_repetitions < 3:
        raise ValueError("profile requires warmups >=1 and repetitions >=3")
    os.environ.setdefault("RUSTICL_ENABLE", "radeonsi")
    os.environ.setdefault("RUSTICL_FEATURES", "fp64")
    profile = _profile_cost_decomposition(
        order=args.order or "cpu-first",
        warmups=args.profile_warmups,
        repetitions=args.profile_repetitions,
        include_samples=True,
    )
    output = {
        "schema_version": 1,
        "phase": "issue-52-p3b-accounting-correction-diagnostic",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "baseline_script_sha256": hashlib.sha256(
            Path(baseline.__file__).read_bytes()
        ).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": {
            "profile_warmups_per_backend_per_shape": args.profile_warmups,
            "profile_repetitions_per_backend_per_shape": args.profile_repetitions,
            "order": args.order or "cpu-first",
            "numeric_checks_rerun": False,
            "original_p3_and_p3b_timings_rerun": False,
            "timers": "Inclusive event time is retained; child event time is subtracted once to derive exclusive self time.",
        },
        "profile_cost_decomposition": profile,
        "automated_accounting_checks": {
            "nested_event_partitions": "PASS",
            "exclusive_opencl_components_plus_residual": "PASS",
            "resident_cycle_components_plus_residual": "PASS",
            "summary_statistics_consistency": "PASS",
        },
        "limitations": [
            "This diagnostic repeats only the instrumented P3b profile, not the P3 timing baseline or P3b four-process stability series.",
            "Instrumentation overhead remains, especially for small calls.",
            "enqueue timing includes clFinish; device execution time remains inseparable without queue profiling.",
            "No source in src/kinera, CPU/CUDA/OpenCL implementations, kernels, or scientific contracts was changed.",
        ],
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"P3b accounting diagnostic complete; JSON={args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--setup-samples", type=int, default=5)
    parser.add_argument("--profile-warmups", type=int, default=3)
    parser.add_argument("--profile-repetitions", type=int, default=5)
    parser.add_argument("--baseline-json", type=Path, default=Path("/tmp/kinera-issue52-p3-benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/kinera-issue52-p3b-cost-analysis.json"))
    parser.add_argument("--child-index", type=int)
    parser.add_argument("--order", choices=("cpu-first", "opencl-first"))
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    if args.diagnostic_only:
        if args.child_index is not None:
            parser.error("--diagnostic-only cannot be combined with --child-index")
        if args.output == Path("/tmp/kinera-issue52-p3b-cost-analysis.json"):
            args.output = ROOT / "experiments/results/opencl-rx550-p3b-accounting-diagnostic.json"
        return _run_diagnostic(args)
    if args.child_index is not None:
        if args.order is None:
            parser.error("--order is required for child mode")
        return _child_main(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
