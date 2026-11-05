"""Reproducible CPUBackend vs. OpenCL Linear benchmark (no training pilots)."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinera.backends import get_backend  # noqa: E402
from kinera.backends.cpu.backend import CPUBackend  # noqa: E402
from kinera.nn.layers.linear import Linear  # noqa: E402

ATOL = 1e-12
RTOL = 1e-12
LEARNING_RATE = 0.03125
PROGRAMS = (
    ("forward", "linear_forward.cl"),
    ("backward", "linear_backward.cl"),
    ("step", "linear_step.cl"),
)

# These correspond to the existing scalar gate and Kinera head shapes. The
# transposed 26x3932 case holds parameter count constant while reversing the
# serial reduction lengths in forward and backward.
CASES = (
    ("1x1", 1, 1),
    ("51x26", 51, 26),
    ("3932x26", 3932, 26),
    ("26x3932", 26, 3932),
)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list) and values and isinstance(values[0], list):
        return [value for row in values for value in row]
    return list(values)


def _max_abs_error(actual: Any, expected: Any, label: str) -> float:
    left = _flatten(actual)
    right = _flatten(expected)
    if len(left) != len(right):
        raise AssertionError(f"{label}: element count {len(left)} != {len(right)}")
    maximum = 0.0
    for index, (got, want) in enumerate(zip(left, right)):
        error = abs(got - want)
        maximum = max(maximum, error)
        if error > ATOL + RTOL * abs(want):
            raise AssertionError(
                f"{label}[{index}] differs: OpenCL={got!r}, CPU={want!r}, "
                f"abs_error={error!r}"
            )
    return maximum


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


def _duration_stats(samples_ns: list[int]) -> dict[str, float | int]:
    values_ms = [sample / 1_000_000 for sample in samples_ns]
    q1 = _quantile(values_ms, 0.25)
    q3 = _quantile(values_ms, 0.75)
    return {
        "samples": len(samples_ns),
        "median_ms": statistics.median(values_ms),
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def _ratio_stats(values: list[float]) -> dict[str, float | int]:
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(values),
        "max": max(values),
    }


def _time_call(function: Callable[[], Any]) -> tuple[int, Any]:
    started = time.perf_counter_ns()
    result = function()
    return time.perf_counter_ns() - started, result


def _paired_measure(
    cpu_call: Callable[[], Any],
    opencl_call: Callable[[], Any],
    *,
    warmups: int,
    repetitions: int,
    opencl_cleanup: Callable[[Any], None] | None = None,
    before_cpu: Callable[[], None] | None = None,
    before_opencl: Callable[[], None] | None = None,
) -> dict[str, Any]:
    def invoke(
        function: Callable[[], Any],
        cleanup: Callable[[Any], None] | None,
        prepare: Callable[[], None] | None,
    ) -> int:
        if prepare is not None:
            prepare()
        duration, result = _time_call(function)
        if cleanup is not None:
            cleanup(result)
        return duration

    for _ in range(warmups):
        invoke(cpu_call, None, before_cpu)
        invoke(opencl_call, opencl_cleanup, before_opencl)

    cpu_samples: list[int] = []
    opencl_samples: list[int] = []
    for index in range(repetitions):
        if index % 2 == 0:
            cpu_samples.append(invoke(cpu_call, None, before_cpu))
            opencl_samples.append(
                invoke(opencl_call, opencl_cleanup, before_opencl)
            )
        else:
            opencl_samples.append(
                invoke(opencl_call, opencl_cleanup, before_opencl)
            )
            cpu_samples.append(invoke(cpu_call, None, before_cpu))
    speedups = [cpu / gpu for cpu, gpu in zip(cpu_samples, opencl_samples)]
    return {
        "cpu": _duration_stats(cpu_samples),
        "opencl": _duration_stats(opencl_samples),
        "paired_cpu_over_opencl_speedup": _ratio_stats(speedups),
        "speedup_definition": "CPU median sample / paired OpenCL sample; >1 favors OpenCL",
    }


def _single_measure(
    function: Callable[[], Any],
    *,
    warmups: int,
    repetitions: int,
    cleanup: Callable[[Any], None] | None = None,
) -> list[int]:
    for _ in range(warmups):
        value = function()
        if cleanup is not None:
            cleanup(value)
    samples = []
    for _ in range(repetitions):
        duration, value = _time_call(function)
        samples.append(duration)
        if cleanup is not None:
            cleanup(value)
    return samples


def _git_text(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fingerprint() -> str:
    relative_paths = [
        "src/kinera/backends/base.py",
        "src/kinera/backends/__init__.py",
        "src/kinera/backends/cpu/backend.py",
        "src/kinera/backends/cpu/ops/linear.py",
        "src/kinera/nn/layers/linear.py",
        "src/kinera/backends/opencl/backend.py",
        "src/kinera/backends/opencl/api.py",
        "src/kinera/backends/opencl/memory.py",
        "src/kinera/backends/opencl/runtime.py",
        "src/kinera/backends/opencl/kernels/linear_forward.cl",
        "src/kinera/backends/opencl/kernels/linear_backward.cl",
        "src/kinera/backends/opencl/kernels/linear_step.cl",
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_model() -> str | None:
    cpu_info = _read_optional(Path("/proc/cpuinfo"))
    if cpu_info:
        for line in cpu_info.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _system_snapshot(pci_bdf: str | None) -> dict[str, Any]:
    cpu_frequency = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    gpu_values: dict[str, str | None] = {}
    if pci_bdf:
        gpu_root = Path("/sys/bus/pci/devices") / pci_bdf
        for filename in (
            "mem_info_vram_used",
            "mem_info_vram_total",
            "gpu_busy_percent",
            "power_dpm_force_performance_level",
        ):
            gpu_values[filename] = _read_optional(gpu_root / filename)
        clock_values = _read_optional(gpu_root / "pp_dpm_sclk")
        gpu_values["pp_dpm_sclk"] = clock_values
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return {
        "load_average_1_5_15": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu0_scaling_governor": _read_optional(cpu_frequency / "scaling_governor"),
        "cpu0_scaling_cur_freq_khz": _read_optional(cpu_frequency / "scaling_cur_freq"),
        "cpu0_scaling_max_freq_khz": _read_optional(cpu_frequency / "scaling_max_freq"),
        "process_cpu_affinity": affinity,
        "gpu_sysfs": gpu_values,
    }


def _case_data(output_size: int, input_size: int, seed: int) -> dict[str, Any]:
    cpu = CPUBackend()
    seeded = Linear(input_size, output_size, seed=seed, backend=cpu)
    weights = deepcopy(seeded.weights)
    bias = deepcopy(seeded.bias)
    seeded.close()

    rng = random.Random(seed + 0x5A52)
    inputs = [rng.uniform(-0.5, 0.5) for _ in range(input_size)]
    grad_output = [rng.uniform(-0.5, 0.5) for _ in range(output_size)]
    return {
        "weights": weights,
        "bias": bias,
        "inputs": inputs,
        "grad_output": grad_output,
        "seed": seed,
        "input_size": input_size,
        "output_size": output_size,
    }


def _validate_integrated(case: dict[str, Any]) -> dict[str, Any]:
    cpu = CPUBackend()
    output_size = case["output_size"]
    input_size = case["input_size"]
    seed = case["seed"]
    cpu_layer = Linear(input_size, output_size, seed=seed, backend=cpu)
    opencl = get_backend("opencl")
    gpu_layer = Linear(input_size, output_size, seed=seed, backend=opencl)
    gpu_inputs = opencl.buffer_from_host(
        case["inputs"], shape=(input_size,), dtype="float64"
    )
    gpu_grad_output = opencl.buffer_from_host(
        case["grad_output"], shape=(output_size,), dtype="float64"
    )
    errors: dict[str, float] = {}
    try:
        errors["initial_weights"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.weights), cpu_layer.weights, "initial weights"
        )
        errors["initial_bias"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.bias), cpu_layer.bias, "initial bias"
        )

        cpu_output = cpu_layer.forward(case["inputs"])
        gpu_output = gpu_layer.forward(gpu_inputs)
        errors["forward"] = _max_abs_error(
            opencl.buffer_to_host(gpu_output), cpu_output, "forward output"
        )

        cpu_grad_inputs = cpu_layer.backward(case["grad_output"])
        gpu_grad_inputs = gpu_layer.backward(gpu_grad_output)
        errors["grad_weights"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.grad_weights),
            cpu_layer.grad_weights,
            "grad_weights",
        )
        errors["grad_bias"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.grad_bias),
            cpu_layer.grad_bias,
            "grad_bias",
        )
        errors["grad_inputs"] = _max_abs_error(
            opencl.buffer_to_host(gpu_grad_inputs),
            cpu_grad_inputs,
            "grad_inputs",
        )

        cpu_layer.step(LEARNING_RATE)
        gpu_layer.step(LEARNING_RATE)
        errors["final_weights"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.weights), cpu_layer.weights, "final weights"
        )
        errors["final_bias"] = _max_abs_error(
            opencl.buffer_to_host(gpu_layer.bias), cpu_layer.bias, "final bias"
        )
        return {
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "max_absolute_error_by_result": errors,
            "overall_max_absolute_error": max(errors.values(), default=0.0),
            "pass": True,
        }
    finally:
        gpu_grad_inputs = locals().get("gpu_grad_inputs")
        if gpu_grad_inputs is not None:
            gpu_grad_inputs.close()
        gpu_layer.close()
        gpu_inputs.close()
        gpu_grad_output.close()
        opencl.close()
        cpu_layer.close()


def _resident_primitives(
    case: dict[str, Any], cpu: CPUBackend, opencl: Any, *, warmups: int, repetitions: int
) -> dict[str, Any]:
    output_size = case["output_size"]
    input_size = case["input_size"]
    weights = deepcopy(case["weights"])
    bias = deepcopy(case["bias"])
    inputs = case["inputs"]
    grad_output = case["grad_output"]
    grad_weights, grad_bias, _ = cpu.linear_backward(weights, inputs, grad_output)
    gpu_weights = opencl.buffer_from_host(weights, shape=(output_size, input_size))
    gpu_bias = opencl.buffer_from_host(bias, shape=(output_size,))
    gpu_inputs = opencl.buffer_from_host(inputs, shape=(input_size,))
    gpu_grad_output = opencl.buffer_from_host(grad_output, shape=(output_size,))
    gpu_grad_weights = opencl.buffer_from_host(
        grad_weights, shape=(output_size, input_size)
    )
    gpu_grad_bias = opencl.buffer_from_host(grad_bias, shape=(output_size,))
    weight_host = (ctypes.c_double * (output_size * input_size))(
        *_flatten(weights)
    )
    bias_host = (ctypes.c_double * output_size)(*bias)

    def reset_opencl_parameters() -> None:
        opencl.runtime.api.write_buffer(
            opencl.memory.queue,
            gpu_weights.handle,
            ctypes.addressof(weight_host),
            gpu_weights.nbytes,
        )
        opencl.runtime.api.write_buffer(
            opencl.memory.queue,
            gpu_bias.handle,
            ctypes.addressof(bias_host),
            gpu_bias.nbytes,
        )

    result: dict[str, Any] = {}
    try:
        result["forward"] = _paired_measure(
            lambda: cpu.linear_forward(weights, bias, inputs),
            lambda: opencl.linear_forward(gpu_weights, gpu_bias, gpu_inputs),
            warmups=warmups,
            repetitions=repetitions,
            opencl_cleanup=lambda output: output.close(),
        )
        result["backward"] = _paired_measure(
            lambda: cpu.linear_backward(weights, inputs, grad_output),
            lambda: opencl.linear_backward(gpu_weights, gpu_inputs, gpu_grad_output),
            warmups=warmups,
            repetitions=repetitions,
            opencl_cleanup=lambda outputs: [buffer.close() for buffer in outputs],
        )

        def cpu_step() -> tuple[Any, Any]:
            return cpu.linear_step(
                weights,
                bias,
                grad_weights,
                grad_bias,
                LEARNING_RATE,
            )

        result["step"] = _paired_measure(
            cpu_step,
            lambda: opencl.linear_step(
                gpu_weights,
                gpu_bias,
                gpu_grad_weights,
                gpu_grad_bias,
                LEARNING_RATE,
            ),
            warmups=warmups,
            repetitions=repetitions,
            before_opencl=reset_opencl_parameters,
        )
        cpu_step_weights, cpu_step_bias = cpu_step()
        result["step_end_state_max_abs_error"] = {
            "weights": _max_abs_error(
                opencl.buffer_to_host(gpu_weights), cpu_step_weights, "step weights"
            ),
            "bias": _max_abs_error(
                opencl.buffer_to_host(gpu_bias), cpu_step_bias, "step bias"
            ),
        }
        return result
    finally:
        for buffer in (
            gpu_grad_bias,
            gpu_grad_weights,
            gpu_grad_output,
            gpu_inputs,
            gpu_bias,
            gpu_weights,
        ):
            buffer.close()


def _integrated_resident(
    case: dict[str, Any], cpu: CPUBackend, opencl: Any, *, warmups: int, repetitions: int
) -> dict[str, Any]:
    input_size = case["input_size"]
    output_size = case["output_size"]
    cpu_layer = Linear(input_size, output_size, seed=case["seed"], backend=cpu)
    gpu_layer = Linear(input_size, output_size, seed=case["seed"], backend=opencl)
    gpu_inputs = opencl.buffer_from_host(case["inputs"], shape=(input_size,))
    gpu_grad_output = opencl.buffer_from_host(
        case["grad_output"], shape=(output_size,)
    )

    def cpu_cycle() -> None:
        previous_output = cpu_layer.output
        output = cpu_layer.forward(case["inputs"])
        grad_inputs = cpu_layer.backward(case["grad_output"])
        cpu_layer.step(LEARNING_RATE)
        del previous_output, output, grad_inputs

    def opencl_cycle() -> None:
        previous_output = gpu_layer.output
        output = gpu_layer.forward(gpu_inputs)
        if previous_output is not None:
            previous_output.close()
        grad_inputs = gpu_layer.backward(gpu_grad_output)
        gpu_layer.step(LEARNING_RATE)
        grad_inputs.close()
        del output

    try:
        return _paired_measure(
            cpu_cycle,
            opencl_cycle,
            warmups=warmups,
            repetitions=repetitions,
        )
    finally:
        cpu_layer.close()
        gpu_layer.close()
        gpu_inputs.close()
        gpu_grad_output.close()


def _end_to_end_cycle(
    case: dict[str, Any], cpu: CPUBackend, opencl: Any, *, warmups: int, repetitions: int
) -> dict[str, Any]:
    input_size = case["input_size"]
    output_size = case["output_size"]

    def cpu_cycle() -> float:
        layer = Linear(input_size, output_size, seed=case["seed"], backend=cpu)
        try:
            output = layer.forward(case["inputs"])
            grad_inputs = layer.backward(case["grad_output"])
            layer.step(LEARNING_RATE)
            values = (
                output,
                layer.grad_weights,
                layer.grad_bias,
                grad_inputs,
                layer.weights,
                layer.bias,
            )
            return math.fsum(value for group in values for value in _flatten(group))
        finally:
            layer.close()

    def opencl_cycle() -> float:
        layer = Linear(input_size, output_size, seed=case["seed"], backend=opencl)
        gpu_inputs = None
        gpu_grad_output = None
        grad_inputs = None
        try:
            gpu_inputs = opencl.buffer_from_host(
                case["inputs"], shape=(input_size,), dtype="float64"
            )
            gpu_grad_output = opencl.buffer_from_host(
                case["grad_output"], shape=(output_size,), dtype="float64"
            )
            output = layer.forward(gpu_inputs)
            grad_inputs = layer.backward(gpu_grad_output)
            layer.step(LEARNING_RATE)
            consumed = (
                opencl.buffer_to_host(output),
                opencl.buffer_to_host(layer.grad_weights),
                opencl.buffer_to_host(layer.grad_bias),
                opencl.buffer_to_host(grad_inputs),
                opencl.buffer_to_host(layer.weights),
                opencl.buffer_to_host(layer.bias),
            )
            return math.fsum(
                value for group in consumed for value in _flatten(group)
            )
        finally:
            layer.close()
            for buffer in (grad_inputs, gpu_grad_output, gpu_inputs):
                if buffer is not None:
                    buffer.close()

    return _paired_measure(
        cpu_cycle,
        opencl_cycle,
        warmups=warmups,
        repetitions=repetitions,
    )


def _transfer_costs(
    case: dict[str, Any], opencl: Any, *, warmups: int, repetitions: int
) -> dict[str, Any]:
    tensors = (
        ("weights", case["weights"], (case["output_size"], case["input_size"])),
        ("bias", case["bias"], (case["output_size"],)),
        ("inputs", case["inputs"], (case["input_size"],)),
        ("grad_output", case["grad_output"], (case["output_size"],)),
    )
    results = {}
    for name, values, shape in tensors:
        flat = _flatten(values)
        host = (ctypes.c_double * len(flat))(*flat)
        device_buffer = opencl.memory.allocate(shape, dtype="float64")
        try:
            # Seed the preallocated destination so D2H never reads undefined data.
            opencl.runtime.api.write_buffer(
                opencl.memory.queue,
                device_buffer.handle,
                ctypes.addressof(host),
                device_buffer.nbytes,
            )
            results[name] = {
                "bytes": len(flat) * ctypes.sizeof(ctypes.c_double),
                "raw_blocking_h2d": _duration_stats(
                    _single_measure(
                        lambda: opencl.runtime.api.write_buffer(
                            opencl.memory.queue,
                            device_buffer.handle,
                            ctypes.addressof(host),
                            device_buffer.nbytes,
                        ),
                        warmups=warmups,
                        repetitions=repetitions,
                    )
                ),
                "raw_blocking_d2h": _duration_stats(
                    _single_measure(
                        lambda: opencl.runtime.api.read_buffer(
                            opencl.memory.queue,
                            device_buffer.handle,
                            ctypes.addressof(host),
                            device_buffer.nbytes,
                        ),
                        warmups=warmups,
                        repetitions=repetitions,
                    )
                ),
                "public_buffer_from_host_h2d_and_allocation": _duration_stats(
                    _single_measure(
                        lambda: opencl.buffer_from_host(values, shape=shape),
                        warmups=warmups,
                        repetitions=repetitions,
                        cleanup=lambda buffer: buffer.close(),
                    )
                ),
                "public_buffer_to_host_d2h_and_materialization": _duration_stats(
                    _single_measure(
                        lambda: opencl.buffer_to_host(device_buffer),
                        warmups=warmups,
                        repetitions=repetitions,
                    )
                ),
            }
        finally:
            device_buffer.close()
    return results


def _measure_startup_and_compilation(samples: int) -> dict[str, Any]:
    startup: list[int] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        backend = get_backend("opencl")
        backend.runtime.require_device(0, require_fp64=True)
        startup.append(time.perf_counter_ns() - started)
        backend.close()

    context_queue: list[int] = []
    for _ in range(samples):
        backend = get_backend("opencl")
        backend.runtime.require_device(0, require_fp64=True)
        duration, _ = _time_call(lambda: backend.memory.queue)
        context_queue.append(duration)
        backend.close()

    builds: dict[str, Any] = {}
    for operation, filename in PROGRAMS:
        timings = []
        for _ in range(samples):
            backend = get_backend("opencl")
            backend.runtime.require_device(0, require_fp64=True)
            _ = backend.memory.queue
            duration, _ = _time_call(lambda: backend._get_program(filename))
            timings.append(duration)
            backend.close()
        builds[operation] = {
            "program_file": filename,
            "source_read_cache_insert_and_clBuildProgram": _duration_stats(timings),
            "ordered_samples_ms": [value / 1_000_000 for value in timings],
        }
    return {
        "backend_runtime_discovery_and_device_selection": _duration_stats(startup),
        "context_and_command_queue_creation": _duration_stats(context_queue),
        "per_program_build": builds,
        "build_note": "New OpenCL program/context each sample; Mesa driver cache was not flushed and may serve cached binaries.",
    }


def _device_record(device: Any) -> dict[str, Any]:
    return {
        "name": device.name,
        "vendor": device.vendor,
        "vendor_id": device.vendor_id,
        "pci_bdf": device.pci_bdf,
        "platform": device.platform_name,
        "platform_vendor": device.platform_vendor,
        "opencl_version": device.version,
        "opencl_c_version": device.opencl_c_version,
        "driver_version": device.driver_version,
        "extensions": list(device.extensions),
        "global_memory_bytes": device.global_mem_size,
        "max_single_allocation_bytes": device.max_mem_alloc_size,
        "local_memory_bytes": device.local_mem_size,
        "max_work_group_size": device.max_work_group_size,
        "double_fp_config": device.double_fp_config,
    }


def run_benchmark(*, warmups: int, repetitions: int, setup_samples: int) -> dict[str, Any]:
    initial_backend = get_backend("opencl")
    device = initial_backend.runtime.require_device(0, require_fp64=True)
    if device.pci_bdf != "0000:03:00.0" or "Radeon RX 550" not in device.name:
        initial_backend.close()
        raise RuntimeError(
            f"Expected RX 550 at 0000:03:00.0, got {device.name!r} ({device.pci_bdf!r})"
        )
    initial_backend.close()

    startup_and_compilation = _measure_startup_and_compilation(setup_samples)
    opencl = get_backend("opencl")
    device = opencl.runtime.require_device(0, require_fp64=True)
    _ = opencl.memory.queue
    for _, filename in PROGRAMS:
        opencl._get_program(filename)

    cpu = CPUBackend()
    system_before = _system_snapshot(device.pci_bdf)
    case_reports = []
    try:
        for index, (label, output_size, input_size) in enumerate(CASES):
            case = _case_data(output_size, input_size, seed=52052 + index)
            validation = _validate_integrated(case)
            primitives = _resident_primitives(
                case, cpu, opencl, warmups=warmups, repetitions=repetitions
            )
            cycle_resident = _integrated_resident(
                case, cpu, opencl, warmups=warmups, repetitions=repetitions
            )
            end_to_end = _end_to_end_cycle(
                case, cpu, opencl, warmups=warmups, repetitions=repetitions
            )
            transfers = _transfer_costs(
                case, opencl, warmups=warmups, repetitions=repetitions
            )
            weights_count = output_size * input_size
            h2d_bytes = (weights_count + 2 * output_size + input_size) * 8
            d2h_bytes = (2 * weights_count + 3 * output_size + input_size) * 8
            case_reports.append(
                {
                    "label": label,
                    "weights_shape": [output_size, input_size],
                    "seed": case["seed"],
                    "float_dtype": "float64",
                    "resident_backend_operations": primitives,
                    "integrated_linear_cycle_resident": cycle_resident,
                    "steady_state_end_to_end_linear_cycle": end_to_end,
                    "end_to_end_transfer_bytes": {
                        "h2d": h2d_bytes,
                        "d2h": d2h_bytes,
                        "definition": "H2D weights/bias/input/grad_output; D2H output, all three gradients, and final weights/bias",
                    },
                    "transfer_measurements": transfers,
                    "numerical_equivalence": validation,
                }
            )
    finally:
        opencl.close()

    system_after = _system_snapshot(device.pci_bdf)
    return {
        "schema_version": 1,
        "phase": "issue-52-p3-linear-cpu-vs-opencl",
        "code": {
            "base_commit": _git_text("rev-parse", "HEAD"),
            "working_tree_status": _git_text("status", "--short"),
            "benchmark_fingerprint_sha256": _fingerprint(),
        },
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "kernel": platform.release(),
        "cpu_model": _cpu_model(),
            "cpu_logical_processors": os.cpu_count(),
            "opencl_device": _device_record(device),
            "rusticl_enable": os.environ.get("RUSTICL_ENABLE"),
            "rusticl_features": os.environ.get("RUSTICL_FEATURES"),
            "system_before": system_before,
            "system_after": system_after,
            "runtime_warning": "Patched Mesa libclc not detected",
        },
        "protocol": {
            "warmups_per_measurement": warmups,
            "timed_repetitions": repetitions,
            "setup_compile_samples": setup_samples,
            "timer": "time.perf_counter_ns (monotonic high-resolution wall clock)",
            "sample_order": "CPU/OpenCL pairs alternate first position by repetition",
            "arithmetic_dtype": "float64",
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "learning_rate": LEARNING_RATE,
            "random_seed_rule": "52052 + case index for weights; seed + 0x5A52 for inputs and grad_output",
            "resident_operation_timing": "CPUBackend/OpenCLBackend call; OpenCL includes validation, temporary allocation, enqueue, clFinish synchronization, and kernel release; result cleanup/host transfers are outside this metric",
            "step_sample_state": "Each resident step timing starts from the same deterministic weights and bias; OpenCL parameter reset uses blocking H2D writes before the timer, CPU linear_step is functional and receives the same host values; reset transfers are excluded from step latency and separately measured in transfer_costs",
            "end_to_end_timing": "Linear construction, input preparation/H2D, forward-backward-step, result consumption/D2H, and explicit object/buffer close; runtime discovery and program compilation are prewarmed and reported separately",
            "result_consumption": "End-to-end CPU and OpenCL paths compute math.fsum across output, grad_weights, grad_bias, grad_inputs, and updated weights/bias; OpenCL first materializes those buffers to host",
            "transfer_timing": "Raw preallocated blocking API transfers and public buffer_from_host/buffer_to_host paths are both measured",
            "kernel_device_timestamps": "Not available: current command queue does not enable OpenCL profiling; resident wall latency is not raw kernel time",
            "external_interference_control": "One sequential process; no other benchmark or training process launched; CPU/GPU affinity, load, frequency, VRAM and utilization snapshots recorded when exposed; clocks/governors were not changed",
        },
        "startup_and_compilation": startup_and_compilation,
        "cases": case_reports,
        "limitations": [
            "Measurements characterize this RX 550, Rusticl/Mesa build and runtime state only.",
            "Mesa driver compiler cache was not cleared; program-build samples may include cached binaries.",
            "CPU turbo/frequency and GPU clocks or thermal state were not locked; before/after telemetry is observational.",
            "OpenCL operation wall times include host dispatch, per-call temporary allocations, and clFinish; no claim is made about isolated kernel time.",
            "End-to-end results represent one Linear forward-backward-step with explicit final materialization, not a training run or full Kinera model.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--setup-samples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 5 or args.setup_samples < 3:
        parser.error("require warmups >= 1, repetitions >= 5, setup-samples >= 3")
    report = run_benchmark(
        warmups=args.warmups,
        repetitions=args.repetitions,
        setup_samples=args.setup_samples,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
