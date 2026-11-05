"""Measure host conversion costs for CPU and OpenCL without changing production.

The worker mode runs one backend/shape in one fresh process. Timed components
are deliberately reported as separate probes when they require replaying a
transformation; those probes must not be added to public API wall times.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from benchmark_opencl_linear import CASES, _case_data  # noqa: E402
from kinera.backends.cpu.backend import CPUBackend  # noqa: E402
from kinera.nn.layers.linear import Linear  # noqa: E402

WARMUPS = 2
REPETITIONS = 8
PROCESS_REPETITIONS = 4
ATOL = 1e-12
RTOL = 1e-12


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats_ns(samples: list[int]) -> dict[str, float | int]:
    values = [sample / 1_000_000 for sample in samples]
    q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _stats_process(values: list[float]) -> dict[str, float | int]:
    q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
    return {
        "processes": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _stats_kib(values: list[float]) -> dict[str, float | int]:
    q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
    return {
        "processes": len(values),
        "median_kib": statistics.median(values),
        "q1_kib": q1,
        "q3_kib": q3,
        "iqr_kib": q3 - q1,
        "min_kib": min(values),
        "max_kib": max(values),
    }


class SpanRecorder:
    """Collect nested inclusive and exclusive spans without double counting."""

    def __init__(self) -> None:
        self.stack: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.stack.clear()
        self.events.clear()

    def call(self, name: str, function: Callable[..., Any], *args, **kwargs):
        span: dict[str, Any] = {"name": name, "children_ns": 0}
        self.stack.append(span)
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            inclusive = time.perf_counter_ns() - start
            self.stack.pop()
            span["inclusive_ns"] = inclusive
            if span["children_ns"] > inclusive:
                raise RuntimeError(
                    f"child spans ({span['children_ns']} ns) exceed inclusive "
                    f"span ({inclusive} ns) for {name}"
                )
            span["exclusive_ns"] = inclusive - span["children_ns"]
            self.events.append(span)
            if self.stack:
                self.stack[-1]["children_ns"] += inclusive

    def summary(self, events: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        selected_events = self.events if events is None else events
        names = sorted({event["name"] for event in selected_events})
        result: dict[str, dict[str, Any]] = {}
        for name in names:
            matching = [event for event in selected_events if event["name"] == name]
            result[name] = {
                "calls": len(matching),
                "inclusive_ms": _stats_ns([e["inclusive_ns"] for e in matching]),
                "exclusive_ms": _stats_ns([e["exclusive_ns"] for e in matching]),
            }
        return result


def _patch(recorder: SpanRecorder, target: object, attribute: str, name: str) -> None:
    original = getattr(target, attribute)

    def wrapped(*args, **kwargs):
        return recorder.call(name, original, *args, **kwargs)

    setattr(target, attribute, wrapped)


def _time_call(function: Callable[[], Any]) -> tuple[int, Any]:
    start = time.perf_counter_ns()
    value = function()
    return time.perf_counter_ns() - start, value


def _assert_same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label} changed values during conversion")


def _rss_kib() -> int:
    # Linux reports ru_maxrss in KiB. This is process high-water RSS, not GPU VRAM.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _make_host_probe(
    values: Any, shape: tuple[int, ...], repetitions: int
) -> dict[str, Any]:
    """Time production validation and isolated, non-additive host probes."""
    samples: dict[str, list[int]] = {
        "linear_list_validation": [],
        "diagnostic_shape_numeric_validation": [],
        "diagnostic_flatten_validated_tree": [],
        "opencl_flatten_validate": [],
        "ctypes_pack": [],
        "ctypes_host_array_allocation": [],
        "python_list_materialization": [],
        "reshape_to_python": [],
    }
    cpu_flat: list[float] | None = None

    def validate_tree(value: Any, expected_shape: tuple[int, ...]) -> None:
        if not expected_shape:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("probe values must be numeric, not bool")
            if not math.isfinite(float(value)):
                raise ValueError("probe values must be finite")
            return
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("probe shape must be a non-empty list/tuple tree")
        if len(value) != expected_shape[0]:
            raise ValueError("probe tree does not match requested shape")
        for child in value:
            validate_tree(child, expected_shape[1:])

    def flatten_validated(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)):
            return [number for child in value for number in flatten_validated(child)]
        return [float(value)]

    for _ in range(repetitions):
        elapsed, validated = _time_call(
            lambda: Linear._validate_backend_array(values, shape, "probe")
        )
        samples["linear_list_validation"].append(elapsed)
        _assert_same(validated, values, "Linear validation")

        elapsed, _ = _time_call(lambda: validate_tree(values, shape))
        samples["diagnostic_shape_numeric_validation"].append(elapsed)
        elapsed, diagnostic_flat = _time_call(lambda: flatten_validated(values))
        samples["diagnostic_flatten_validated_tree"].append(elapsed)

        # The production helper performs validation and flattening in one pass.
        # This timer is the complete helper, not an additive validation-only time.
        # Bound method is installed by the OpenCL worker after this function starts.
        flatten = getattr(_make_host_probe, "opencl_flatten", None)
        if flatten is not None:
            elapsed, flattened = _time_call(lambda: flatten(values))
            samples["opencl_flatten_validate"].append(elapsed)
            flat_values, observed_shape = flattened
            if observed_shape != shape:
                raise AssertionError(f"flattened shape {observed_shape} != {shape}")
            if cpu_flat is None:
                cpu_flat = list(flat_values)
            else:
                _assert_same(list(flat_values), cpu_flat, "OpenCL flatten")
            _assert_same(diagnostic_flat, list(flat_values), "diagnostic flatten")
        else:
            # CPU has no separate flatten operation; its recursive copy validates
            # and constructs the returned tree together (measured at the API).
            samples["opencl_flatten_validate"].append(0)
            cpu_flat = list(diagnostic_flat)

    if cpu_flat is not None:
        packed_samples: list[int] = []
        packed: Any = None
        for _ in range(repetitions):
            elapsed, packed = _time_call(
                lambda: (ctypes.c_double * len(cpu_flat))(*cpu_flat)
            )
            packed_samples.append(elapsed)
        samples["ctypes_pack"] = packed_samples
        if list(packed) != cpu_flat:
            raise AssertionError("ctypes packing changed float64 values")

        array_samples: list[int] = []
        for _ in range(repetitions):
            elapsed, _host_array = _time_call(
                lambda: (ctypes.c_double * len(cpu_flat))()
            )
            array_samples.append(elapsed)
        samples["ctypes_host_array_allocation"] = array_samples

        ctypes_source = (ctypes.c_double * len(cpu_flat))(*cpu_flat)
        materialize_samples: list[int] = []
        for _ in range(repetitions):
            elapsed, materialized = _time_call(lambda: list(ctypes_source))
            materialize_samples.append(elapsed)
            _assert_same(materialized, cpu_flat, "ctypes to Python list materialization")
        samples["python_list_materialization"] = materialize_samples

        from kinera.backends.opencl.memory import OpenCLMemoryManager

        reshape_samples: list[int] = []
        expected = OpenCLMemoryManager._reshape_host_values(materialized, shape)
        for _ in range(repetitions):
            elapsed, reshaped = _time_call(
                lambda: OpenCLMemoryManager._reshape_host_values(materialized, shape)
            )
            reshape_samples.append(elapsed)
            _assert_same(reshaped, expected, "reshape")
        samples["reshape_to_python"] = reshape_samples

    return {name: _stats_ns(values) if any(values) else None for name, values in samples.items()}


def _patch_opencl(recorder: SpanRecorder, backend: Any) -> None:
    memory = backend.memory
    api = backend.runtime.api
    for target, attribute, name in (
        (backend, "buffer_from_host", "opencl.backend.buffer_from_host"),
        (backend, "buffer_to_host", "opencl.backend.buffer_to_host"),
        (memory, "copy_to_device", "opencl.memory.copy_to_device"),
        (memory, "copy_to_host", "opencl.memory.copy_to_host"),
        (memory, "allocate", "opencl.allocate"),
        (memory, "_ensure_context", "opencl.ensure_context"),
        (backend.runtime, "require_device", "opencl.require_device"),
        (api, "create_context", "opencl.api.create_context"),
        (api, "create_command_queue", "opencl.api.create_command_queue"),
        (api, "create_buffer", "opencl.api.create_buffer"),
        (api, "write_buffer", "opencl.api.write_buffer"),
        (api, "read_buffer", "opencl.api.read_buffer"),
    ):
        _patch(recorder, target, attribute, name)

    original_flatten = memory._flatten_float_values

    def flatten(values):
        return recorder.call("opencl.flatten_validate", original_flatten, values)

    memory._flatten_float_values = flatten
    original_reshape = memory._reshape_host_values

    def reshape(values, shape):
        return recorder.call("opencl.reshape", original_reshape, values, shape)

    memory._reshape_host_values = reshape


def _worker(backend_name: str, case_index: int, process_index: int, warmups: int, repetitions: int) -> dict[str, Any]:
    label, output_size, input_size = CASES[case_index]
    seed = 52052 + case_index
    case = _case_data(output_size, input_size, seed)
    cpu_reference = CPUBackend()
    if backend_name == "cpu":
        backend: Any = cpu_reference
        device_info = None
        recorder = SpanRecorder()
        runtime_init_ms = None
    else:
        start = time.perf_counter_ns()
        from kinera.backends import get_backend

        backend = get_backend("opencl")
        runtime_init_ms = (time.perf_counter_ns() - start) / 1_000_000
        device_info = backend.runtime.require_device(backend.device_id, require_fp64=True)
        recorder = SpanRecorder()
        _patch_opencl(recorder, backend)

    original_linear_validate = Linear._validate_backend_array

    def timed_linear_validate(value: object, shape: tuple[int, ...], name: str):
        return recorder.call(
            "linear.validate_backend_array", original_linear_validate,
            value, shape, name,
        )

    Linear._validate_backend_array = staticmethod(timed_linear_validate)

    tensors = {
        "weights": (case["weights"], (output_size, input_size)),
        "bias": (case["bias"], (output_size,)),
        "inputs": (case["inputs"], (input_size,)),
    }
    process_baseline_rss_kib = _rss_kib()

    def upload(values: Any, shape: tuple[int, ...]):
        return backend.buffer_from_host(values, shape=shape, dtype="float64")

    # Warm the public API and allocate one source buffer for repeated D2H.
    resident: dict[str, Any] = {}
    for _ in range(warmups):
        for name, (values, shape) in tensors.items():
            copied = upload(values, shape)
            exported = backend.buffer_to_host(copied)
            _assert_same(exported, values, f"{backend_name} {name} warmup roundtrip")
            previous = resident.get(name)
            if backend_name == "opencl" and previous is not None:
                previous.close()
            resident[name] = copied
    # The last resident CPU value is an independent list, and for OpenCL the last
    # buffer is kept alive for D2H reads. Other OpenCL warmup buffers were closed.
    recorder.reset()

    api_results: dict[str, Any] = {}
    component_results: dict[str, Any] = {}
    for name, (values, shape) in tensors.items():
        upload_samples: list[int] = []
        upload_events: list[dict[str, Any]] = []
        recorder.reset()
        for _ in range(repetitions):
            elapsed, copied = _time_call(lambda: upload(values, shape))
            upload_samples.append(elapsed)
            upload_events.extend(recorder.events)
            recorder.reset()
            _assert_same(backend.buffer_to_host(copied), values, f"{name} prepared values")
            if backend_name == "opencl":
                copied.close()
            recorder.reset()
        upload_spans = recorder.summary(upload_events)

        source = resident[name]
        download_samples: list[int] = []
        recorder.reset()
        for _ in range(repetitions):
            elapsed, exported = _time_call(lambda: backend.buffer_to_host(source))
            download_samples.append(elapsed)
            _assert_same(exported, values, f"{name} materialized values")
        download_spans = recorder.summary()

        if backend_name == "cpu":
            # CPUBackend returns an independent Python tree on both directions.
            if resident[name] is values:
                raise AssertionError("CPU buffer_from_host returned its input by alias")
        api_results[name] = {
            "shape": list(shape),
            "elements": math.prod(shape),
            "nbytes_float64": math.prod(shape) * ctypes.sizeof(ctypes.c_double),
            "buffer_from_host": _stats_ns(upload_samples),
            "buffer_to_host": _stats_ns(download_samples),
            "roundtrip_exact": True,
        }
        if backend_name == "opencl":
            component_results[name] = {
                "buffer_from_host_nested_spans": upload_spans,
                "buffer_to_host_nested_spans": download_spans,
            }

    # Record actual Linear validation, host preparation, and constructor costs.
    linear_list_validation: list[int] = []
    prepare_samples: list[int] = []
    constructor_samples: list[int] = []
    linear_buffer_calls: dict[str, Any] = {}
    for _ in range(warmups):
        Linear._validate_backend_array(case["weights"], (output_size, input_size), "weights")
        probe_layer = Linear(input_size, output_size, seed=seed, backend=backend)
        prepared = probe_layer._prepare_backend_array(
            case["weights"], (output_size, input_size), "weights"
        )
        if backend_name == "opencl":
            prepared.close()
        probe_layer.close()
    recorder.reset()
    for _ in range(repetitions):
        elapsed, validated = _time_call(
            lambda: Linear._validate_backend_array(
                case["weights"], (output_size, input_size), "weights"
            )
        )
        linear_list_validation.append(elapsed)
        _assert_same(validated, case["weights"], "Linear host-list validation")

        layer = Linear(input_size, output_size, seed=seed, backend=backend)
        if backend_name == "opencl":
            _assert_same(backend.buffer_to_host(layer.weights), case["weights"], "Linear weights init")
            _assert_same(backend.buffer_to_host(layer.bias), case["bias"], "Linear bias init")
        else:
            _assert_same(layer.weights, case["weights"], "Linear weights init")
            _assert_same(layer.bias, case["bias"], "Linear bias init")
        start = time.perf_counter_ns()
        prepared = layer._prepare_backend_array(
            case["weights"], (output_size, input_size), "weights"
        )
        prepare_samples.append(time.perf_counter_ns() - start)
        if backend_name == "opencl":
            prepared.close()
        start = time.perf_counter_ns()
        constructed = Linear(input_size, output_size, seed=seed, backend=backend)
        constructor_samples.append(time.perf_counter_ns() - start)
        constructed.close()
        layer.close()
    if backend_name == "opencl":
        linear_buffer_calls = recorder.summary()

    host_probe = None
    if backend_name == "opencl":
        _make_host_probe.opencl_flatten = backend.memory._flatten_float_values
        host_probe = _make_host_probe(
            case["weights"], (output_size, input_size), repetitions
        )
        del _make_host_probe.opencl_flatten

        # Separately time a raw blocking H2D read/write to avoid Python conversion
        # and allocation. These probes are not components of public API totals.
        flat, shape = backend.memory._flatten_float_values(case["weights"])
        host_array = (ctypes.c_double * len(flat))(*flat)
        raw_buffer = backend.memory.allocate(shape, dtype="float64")
        write_samples: list[int] = []
        read_samples: list[int] = []
        host_read_array = (ctypes.c_double * len(flat))()
        try:
            for _ in range(warmups):
                backend.runtime.api.write_buffer(
                    backend.memory.queue, raw_buffer.handle,
                    ctypes.addressof(host_array), raw_buffer.nbytes,
                )
                backend.runtime.api.read_buffer(
                    backend.memory.queue, raw_buffer.handle,
                    ctypes.addressof(host_read_array), raw_buffer.nbytes,
                )
            for _ in range(repetitions):
                elapsed, _ = _time_call(lambda: backend.runtime.api.write_buffer(
                    backend.memory.queue, raw_buffer.handle,
                    ctypes.addressof(host_array), raw_buffer.nbytes,
                ))
                write_samples.append(elapsed)
                elapsed, _ = _time_call(lambda: backend.runtime.api.read_buffer(
                    backend.memory.queue, raw_buffer.handle,
                    ctypes.addressof(host_read_array), raw_buffer.nbytes,
                ))
                read_samples.append(elapsed)
            if list(host_read_array) != flat:
                raise AssertionError("raw D2H bytes differ from uploaded float64 values")
        finally:
            raw_buffer.close()
        raw_transfer = {
            "preallocated_blocking_h2d": _stats_ns(write_samples),
            "preallocated_blocking_d2h": _stats_ns(read_samples),
        }
    else:
        host_probe = {
            "linear_list_validation": _stats_ns(linear_list_validation),
            "cpu_tree_copy_validation_is_combined": True,
        }
        raw_transfer = None

    # Isolated allocate timing, after runtime/context warmup. Release is outside
    # the timed span; clCreateBuffer timing is not physical VRAM commitment time.
    allocation_samples: list[int] = []
    if backend_name == "opencl":
        alloc_warm = backend.memory.allocate((output_size, input_size), dtype="float64")
        alloc_warm.close()
        for _ in range(repetitions):
            elapsed, buffer = _time_call(
                lambda: backend.memory.allocate((output_size, input_size), dtype="float64")
            )
            allocation_samples.append(elapsed)
            buffer.close()

    # End-to-end constructor and per-layer prepare are distinct from direct API.
    layer_results = {
        "linear_list_validation": _stats_ns(linear_list_validation),
        "linear_prepare_host_weights": _stats_ns(prepare_samples),
        "linear_constructor": _stats_ns(constructor_samples),
        "linear_init_roundtrip_exact": True,
    }

    peak_rss = _rss_kib()
    result = {
        "backend": backend_name,
        "case": label,
        "shape": [output_size, input_size],
        "seed": seed,
        "process_index": process_index,
        "warmups": warmups,
        "repetitions": repetitions,
        "api": api_results,
        "layer": layer_results,
        "host_conversion_probes": host_probe,
        "opencl_nested_components": component_results,
        "opencl_raw_transfer": raw_transfer,
        "opencl_allocate": _stats_ns(allocation_samples) if allocation_samples else None,
        "rss": {
            "baseline_high_water_kib": process_baseline_rss_kib,
            "peak_high_water_kib": peak_rss,
            "observed_high_water_delta_kib": max(0, peak_rss - process_baseline_rss_kib),
            "meaning": "process ru_maxrss high-water; baseline after imports/runtime and input creation; not GPU VRAM",
        },
        "runtime_init_ms": runtime_init_ms,
        "device": None if device_info is None else {
            "name": device_info.name,
            "vendor": device_info.vendor,
            "platform": device_info.platform_name,
            "platform_vendor": device_info.platform_vendor,
            "version": device_info.version,
            "opencl_c_version": device_info.opencl_c_version,
            "driver_version": device_info.driver_version,
            "supports_fp64": device_info.supports_fp64,
            "global_memory_bytes": device_info.global_mem_size,
            "max_single_allocation_bytes": device_info.max_mem_alloc_size,
        },
        "cpu_only_opencl_api_loaded": "kinera.backends.opencl.api" in sys.modules,
    }

    for buffer in resident.values():
        if backend_name == "opencl" and not buffer.closed:
            buffer.close()
    if backend_name == "opencl":
        backend.close()
    return result


def _aggregate(worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for backend in ("cpu", "opencl"):
        for case in (entry[0] for entry in CASES):
            group = [r for r in worker_results if r["backend"] == backend and r["case"] == case]
            if not group:
                continue
            paths: dict[str, Any] = {}
            for tensor in ("weights", "bias", "inputs"):
                for phase in ("buffer_from_host", "buffer_to_host"):
                    key = f"{tensor}.{phase}"
                    paths[key] = _stats_process([
                        item["api"][tensor][phase]["median_ms"] for item in group
                    ])
            for phase in (
                "linear_list_validation", "linear_prepare_host_weights", "linear_constructor"
            ):
                paths[f"layer.{phase}"] = _stats_process([
                    item["layer"][phase]["median_ms"] for item in group
                ])
            if backend == "opencl":
                paths["opencl.allocate"] = _stats_process([
                    item["opencl_allocate"]["median_ms"] for item in group
                ])
                paths["opencl.raw_h2d"] = _stats_process([
                    item["opencl_raw_transfer"]["preallocated_blocking_h2d"]["median_ms"] for item in group
                ])
                paths["opencl.raw_d2h"] = _stats_process([
                    item["opencl_raw_transfer"]["preallocated_blocking_d2h"]["median_ms"] for item in group
                ])
            aggregate[f"{backend}.{case}"] = {
                "independent_processes": len(group),
                "paths": paths,
                "rss_delta_kib": _stats_kib([
                    float(item["rss"]["observed_high_water_delta_kib"]) for item in group
                ]),
                "peak_rss_kib": _stats_kib([
                    float(item["rss"]["peak_high_water_kib"]) for item in group
                ]),
            }
    return aggregate


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    worker_results: list[dict[str, Any]] = []
    stderr_notes: list[str] = []
    execution_order: list[dict[str, Any]] = []
    for process_index in range(args.process_repetitions):
        for case_index, (label, _out, _in) in enumerate(CASES):
            order = ("cpu", "opencl") if process_index % 2 == 0 else ("opencl", "cpu")
            for backend_name in order:
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--backend", backend_name, "--case-index", str(case_index),
                    "--process-index", str(process_index), "--warmups", str(args.warmups),
                    "--repetitions", str(args.repetitions),
                ]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "experiments")
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                execution_order.append({
                    "process_index": process_index,
                    "case": label,
                    "backend": backend_name,
                    "exit_code": completed.returncode,
                })
                if completed.stderr.strip():
                    stderr_notes.extend(
                        line for line in completed.stderr.splitlines()
                        if "Patched Mesa libclc not detected" in line
                    )
                if completed.returncode:
                    raise RuntimeError(
                        f"worker {backend_name}/{label}/{process_index} failed:\n"
                        f"stdout: {completed.stdout[-3000:]}\nstderr: {completed.stderr[-3000:]}"
                    )
                worker_results.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    return {
        "schema_version": 1,
        "experiment": "P3c Python/OpenCL host conversion costs",
        "measurement_clock": "time.perf_counter_ns",
        "statistics": "within-process median/IQR; aggregate median/IQR across independent process medians",
        "warmups": args.warmups,
        "repetitions_per_process": args.repetitions,
        "independent_process_pairs": args.process_repetitions,
        "order": "CPU then OpenCL on even process index; OpenCL then CPU on odd process index; per shape",
        "tolerances": {"atol": ATOL, "rtol": RTOL, "dtype": "float64"},
        "cases": [
            {"label": label, "weights_shape": [out, inn], "seed": 52052 + index}
            for index, (label, out, inn) in enumerate(CASES)
        ],
        "machine": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_processors": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "rusticl_enable": os.environ.get("RUSTICL_ENABLE"),
            "rusticl_features": os.environ.get("RUSTICL_FEATURES"),
        },
        "runtime_warning": "Patched Mesa libclc not detected" if stderr_notes else None,
        "runtime_warning_observations": len(stderr_notes),
        "execution_order": execution_order,
        "aggregate": _aggregate(worker_results),
        "workers": worker_results,
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "opencl"))
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--process-repetitions", type=int, default=PROCESS_REPETITIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.backend is None or args.case_index is None:
            parser.error("worker requires --backend and --case-index")
        print(json.dumps(_worker(
            args.backend, args.case_index, args.process_index,
            args.warmups, args.repetitions,
        ), separators=(",", ":")))
        return
    if args.warmups < 1 or args.repetitions < 2 or args.process_repetitions < 2:
        parser.error("need at least 1 warmup, 2 repetitions, and 2 independent processes")
    result = _parent(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
