#!/usr/bin/env python3
"""Experimental FP64 exp/log/sum/cross-entropy probe for Mesa Rusticl.

No production kernel is imported or modified. Run with:

  RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
    python experiments/investigate_opencl_fp64_cross_entropy_math.py \
    --output experiments/results/opencl-rx550-f2-3a-math-audit.json
"""
from __future__ import annotations

import argparse
import ctypes
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
import platform
import subprocess
import struct
import tempfile
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from kinera.backends.cpu import CPUBackend
from kinera.backends.opencl.runtime import OpenCLRuntime


SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

__kernel void probe_exp(__global const double *input, int count,
                        __global double *output, __global ulong *echo) {
  const int i = (int)get_global_id(0);
  if (i < count) {
    echo[i] = as_ulong(input[i]);
    output[i] = exp(input[i]);
  }
}

__kernel void probe_log(__global const double *input, int count,
                        __global double *output, __global ulong *echo) {
  const int i = (int)get_global_id(0);
  if (i < count) {
    echo[i] = as_ulong(input[i]);
    output[i] = log(input[i]);
  }
}

__kernel void probe_sum(__global const double *input, int count, int reverse,
                        __global double *output, __global ulong *echo) {
  if (get_global_id(0) != 0) return;
  double total = 0.0;
  for (int step = 0; step < count; ++step) {
    const int i = reverse ? count - 1 - step : step;
    echo[i] = as_ulong(input[i]);
    total = total + input[i];
  }
  output[0] = total;
}

__kernel void probe_cross_entropy(__global const double *logits, int count,
                                  int target, double scalar_argument,
                                  __global double *trace,
                                  __global ulong *input_echo,
                                  __global ulong *scalar_echo,
                                  __global int *target_echo) {
  if (get_global_id(0) != 0) return;
  double maximum = logits[0];
  for (int i = 0; i < count; ++i) {
    input_echo[i] = as_ulong(logits[i]);
    if (logits[i] > maximum) maximum = logits[i];
  }
  trace[0] = maximum;
  double total = 0.0;
  for (int i = 0; i < count; ++i) {
    const double shifted = logits[i] - maximum;
    const double exponential = exp(shifted);
    trace[1 + i] = shifted;
    trace[1 + count + i] = exponential;
    total = total + exponential;
    trace[1 + 2 * count + i] = total;
  }
  const int log_index = 1 + 3 * count;
  trace[log_index] = log(total);
  trace[log_index + 1] = maximum + trace[log_index];
  trace[log_index + 2] = trace[log_index + 1] - logits[target];
  scalar_echo[0] = as_ulong(scalar_argument);
  target_echo[0] = target;
}
"""

OPTIONS = ("", "-cl-opt-disable", "-cl-fast-relaxed-math")
TRIALS = 3
ATOL = 1.0e-12
RTOL = 1.0e-12


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def ordered(value: float) -> int:
    raw = int(bits(value), 16)
    return (~raw & ((1 << 64) - 1)) if raw >> 63 else raw | (1 << 63)


def compare(expected: float, actual: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected": repr(expected), "expected_bits": bits(expected),
        "actual": repr(actual), "actual_bits": bits(actual),
    }
    if not math.isfinite(expected) or not math.isfinite(actual):
        same = (math.isnan(expected) and math.isnan(actual)) or expected == actual
        result.update(absolute_error=None, relative_error=None, ulps=None,
                      status="SPECIAL_MATCH" if same else "FAIL", category="non_finite")
        return result
    error = abs(actual - expected)
    limit = max(ATOL, RTOL * max(abs(expected), abs(actual)))
    relative = error / abs(expected) if expected != 0.0 else None
    error_json: float | str = error if math.isfinite(error) else repr(error)
    relative_json: float | str | None = (
        relative if relative is None or math.isfinite(relative) else repr(relative)
    )
    result.update(
        absolute_error=error_json,
        relative_error=relative_json,
        ulps=abs(ordered(actual) - ordered(expected)),
        tolerance_limit=limit,
        status="PASS" if error <= limit else "FAIL",
        category="finite",
    )
    if expected == 0.0 and actual == 0.0:
        result["zero_sign_matches"] = bits(expected) == bits(actual)
    return result


def decimal_exp(value: float) -> float:
    with localcontext() as context:
        context.prec = 100
        return float(Decimal.from_float(value).exp())


def decimal_log(value: float) -> float:
    with localcontext() as context:
        context.prec = 100
        return float(Decimal.from_float(value).ln())


def decimal_cross_entropy(logits: list[float], target: int) -> float:
    with localcontext() as context:
        context.prec = 120
        values = [Decimal.from_float(value) for value in logits]
        maximum = max(values)
        total = sum(((value - maximum).exp() for value in values), Decimal(0))
        # Cancel the shared maximum before adding the small logarithm. This
        # computes the high-precision mathematical reference without repeating
        # the cancellation present in the CPU expression under investigation.
        result = maximum - values[target] + total.ln()
        try:
            return float(result)
        except OverflowError:
            return -math.inf if result.is_signed() else math.inf


def _float_array(values: list[float]) -> Any:
    return (ctypes.c_double * len(values))(*values)


def _read_doubles(api: Any, queue: int, buffer: int, count: int) -> list[float]:
    host = (ctypes.c_double * count)()
    api.read_buffer(queue, buffer, ctypes.addressof(host), ctypes.sizeof(host))
    return [float(value) for value in host]


def _read_ulongs(api: Any, queue: int, buffer: int, count: int) -> list[int]:
    host = (ctypes.c_ulonglong * count)()
    api.read_buffer(queue, buffer, ctypes.addressof(host), ctypes.sizeof(host))
    return [int(value) for value in host]


def _write_doubles(api: Any, queue: int, buffer: int, values: list[float]) -> None:
    host = _float_array(values)
    api.write_buffer(queue, buffer, ctypes.addressof(host), ctypes.sizeof(host))


def _comparison_vector(expected: list[float], actual: list[float]) -> list[dict[str, Any]]:
    return [compare(left, right) for left, right in zip(expected, actual, strict=True)]


def exp_cases() -> list[tuple[str, float]]:
    return [
        ("positive_zero", 0.0), ("negative_zero", -0.0),
        ("negative_epsilon", -math.ulp(1.0)),
        ("negative_half_epsilon", -math.ulp(1.0) / 2.0),
        ("minus_one", -1.0), ("minus_ten", -10.0),
        ("minus_hundred", -100.0), ("minus_700", -700.0),
        ("near_normal_underflow", math.log(sys.float_info.min)),
        ("near_subnormal", -708.0), ("near_underflow_1", -740.0),
        ("near_underflow_2", -744.0), ("subnormal_rounding_edge", -745.0),
        ("near_zero_rounding_edge", -745.1332191019411),
        ("underflow_zero", -746.0), ("very_negative", -1000.0),
        ("extreme_negative", -1.0e6),
    ]


def log_cases() -> list[tuple[str, float]]:
    rows = [
        ("next_below_one", math.nextafter(1.0, 0.0)),
        ("one", 1.0), ("next_above_one", math.nextafter(1.0, math.inf)),
        ("half", 0.5), ("two", 2.0), ("small_positive", 2.0**-52),
        ("min_normal", sys.float_info.min), ("min_subnormal", math.ulp(0.0)),
        ("one_e_minus_300", 1.0e-300), ("four", 4.0),
        ("sum_3", 3.0), ("sum_100", 100.0),
        ("large_finite", 1.0e300),
    ]
    for label, logits, _target in ce_cases():
        maximum = max(logits)
        denominator = sum(math.exp(value - maximum) for value in logits)
        rows.append((f"ce_denominator_{label}", denominator))
    return rows


def sum_cases() -> list[tuple[str, list[float]]]:
    return [
        ("singleton", [1.0]),
        ("halfway_tie_short", [1.0, 2.0**-53, 2.0**-53]),
        ("small_terms_after_large", [1.0] + [2.0**-53] * 100),
        ("small_terms_before_large", [2.0**-53] * 100 + [1.0]),
        ("exp_like_short", [math.exp(-index) for index in range(32)]),
        ("exp_like_long", [math.exp(-index / 8.0) for index in range(256)]),
        ("many_equal_positive", [1.0] * 1000),
        ("mixed_positive_magnitudes", [1.0, 1.0e-16, 1.0e-16, 1.0e-16] * 64),
    ]


def ce_cases() -> list[tuple[str, list[float], int]]:
    close = math.nextafter(1.0, 0.0)
    near_large = math.nextafter(1.0e300, 0.0)
    cases = [
        ("vocab_one", [0.0], 0),
        ("two_equal_target_first", [0.0, 0.0], 0),
        ("two_equal_target_last", [0.0, 0.0], 1),
        ("signed_zero_tie", [-0.0, 0.0], 0),
        ("multiple_equal_middle", [1.0] * 7, 3),
        ("target_first", [1.0, -1.0, -2.0, -3.0], 0),
        ("target_middle", [1.0, -1.0, -2.0, -3.0], 2),
        ("target_last", [1.0, -1.0, -2.0, -3.0], 3),
        ("small_logit_differences", [0.0, -math.ulp(1.0), -2 * math.ulp(1.0)], 0),
        ("close_large_logits", [1.0, close, math.nextafter(close, 0.0)], 2),
        ("positive_large_logits", [1000.0, 999.0, 998.0], 1),
        ("negative_large_logits", [-1000.0, -1001.0, -1002.0], 2),
        ("underflow_shifted_exponentials", [0.0, -745.0, -746.0, -1000.0], 3),
        ("subtraction_cancellation_1e16", [1.0e16, 1.0e16], 0),
        ("subtraction_cancellation_1e300", [1.0e300, 1.0e300], 0),
        ("wide_finite_logits_loss_overflow", [1.0e308, -1.0e308], 1),
        ("near_largest_close_logits", [1.0e300, near_large], 1),
        ("long_vocabulary", [0.0] * 256, 255),
    ]
    return cases


def _make_resources(api: Any, device: Any, options: str) -> tuple[Any, ...]:
    stack = ExitStack()
    context = api.create_context(device.handle)
    stack.callback(api.release_context, context)
    queue = api.create_command_queue(context, device.handle)
    stack.callback(api.release_command_queue, queue)
    program = api.build_program(context, device.handle, SOURCE, options)
    stack.callback(api.release_program, program)
    kernels = {}
    for name in ("probe_exp", "probe_log", "probe_sum", "probe_cross_entropy"):
        kernel = api.create_kernel(program, name)
        stack.callback(api.release_kernel, kernel)
        kernels[name] = kernel
    return stack, context, queue, kernels


def _run_unary(api: Any, queue: int, context: int, kernels: dict[str, int],
               name: str, items: list[tuple[str, float]]) -> list[dict[str, Any]]:
    values = [value for _, value in items]
    input_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double) * len(values))
    output_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double) * len(values))
    echo_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_ulonglong) * len(values))
    try:
        _write_doubles(api, queue, input_buffer, values)
        kernel = kernels[name]
        api.set_kernel_buffer(kernel, 0, input_buffer)
        api.set_kernel_int(kernel, 1, len(values))
        api.set_kernel_buffer(kernel, 2, output_buffer)
        api.set_kernel_buffer(kernel, 3, echo_buffer)
        api.enqueue_1d(queue, kernel, len(values))
        outputs = _read_doubles(api, queue, output_buffer, len(values))
        echoes = _read_ulongs(api, queue, echo_buffer, len(values))
    finally:
        api.release_mem(echo_buffer)
        api.release_mem(output_buffer)
        api.release_mem(input_buffer)
    rows = []
    for (label, value), output, echoed in zip(items, outputs, echoes, strict=True):
        oracle = math.exp(value) if name == "probe_exp" else math.log(value)
        high_precision = decimal_exp(value) if name == "probe_exp" else decimal_log(value)
        rows.append({
            "case": label, "input": repr(value), "input_bits": bits(value),
            "ce_denominator_domain": name == "probe_log" and value >= 1.0,
            "output": repr(output), "output_bits": bits(output),
            "argument_echo_bits": f"{echoed:016x}",
            "echo_bits_match": echoed == int(bits(value), 16),
            "vs_cpu": compare(oracle, output),
            "vs_decimal_rounded_to_fp64": compare(high_precision, output),
        })
    return rows


def _run_sum(api: Any, queue: int, context: int, kernels: dict[str, int],
             label: str, values: list[float]) -> dict[str, Any]:
    input_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double) * len(values))
    output_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double))
    echo_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_ulonglong) * len(values))
    try:
        _write_doubles(api, queue, input_buffer, values)
        kernel = kernels["probe_sum"]
        api.set_kernel_buffer(kernel, 0, input_buffer)
        api.set_kernel_int(kernel, 1, len(values))
        api.set_kernel_int(kernel, 2, 0)
        api.set_kernel_buffer(kernel, 3, output_buffer)
        api.set_kernel_buffer(kernel, 4, echo_buffer)
        api.enqueue_1d(queue, kernel, 1)
        forward = _read_doubles(api, queue, output_buffer, 1)[0]
        echoes = _read_ulongs(api, queue, echo_buffer, len(values))
        api.set_kernel_int(kernel, 2, 1)
        api.enqueue_1d(queue, kernel, 1)
        reverse = _read_doubles(api, queue, output_buffer, 1)[0]
    finally:
        api.release_mem(echo_buffer)
        api.release_mem(output_buffer)
        api.release_mem(input_buffer)
    cpu_sum = sum(value for value in values)
    fsum = math.fsum(values)
    with localcontext() as context_decimal:
        context_decimal.prec = 100
        exact = float(sum((Decimal.from_float(value) for value in values), Decimal(0)))
    return {
        "case": label, "count": len(values),
        "input_bits": [bits(value) for value in values],
        "input_echo_bits_match": echoes == [int(bits(value), 16) for value in values],
        "cpu_builtin_sum": repr(cpu_sum), "cpu_builtin_sum_bits": bits(cpu_sum),
        "cpu_fsum": repr(fsum), "cpu_fsum_bits": bits(fsum),
        "decimal_exact_sum_rounded_to_fp64": repr(exact),
        "decimal_exact_sum_bits": bits(exact),
        "opencl_forward_sum": repr(forward), "opencl_forward_sum_bits": bits(forward),
        "opencl_reverse_sum": repr(reverse), "opencl_reverse_sum_bits": bits(reverse),
        "forward_vs_cpu": compare(cpu_sum, forward),
        "reverse_vs_cpu": compare(cpu_sum, reverse),
        "forward_vs_fsum": compare(fsum, forward),
        "order_changes_result": bits(forward) != bits(reverse),
    }


def _run_ce(api: Any, queue: int, context: int, kernels: dict[str, int],
            label: str, logits: list[float], target: int) -> dict[str, Any]:
    count = len(logits)
    trace_count = 3 * count + 4
    input_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double) * count)
    trace_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_double) * trace_count)
    input_echo_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_ulonglong) * count)
    scalar_echo_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_ulonglong))
    target_echo_buffer = api.create_buffer(context, ctypes.sizeof(ctypes.c_int))
    scalar_argument = float(target) + 0.25
    try:
        _write_doubles(api, queue, input_buffer, logits)
        kernel = kernels["probe_cross_entropy"]
        api.set_kernel_buffer(kernel, 0, input_buffer)
        api.set_kernel_int(kernel, 1, count)
        api.set_kernel_int(kernel, 2, target)
        api.set_kernel_double(kernel, 3, scalar_argument)
        api.set_kernel_buffer(kernel, 4, trace_buffer)
        api.set_kernel_buffer(kernel, 5, input_echo_buffer)
        api.set_kernel_buffer(kernel, 6, scalar_echo_buffer)
        api.set_kernel_buffer(kernel, 7, target_echo_buffer)
        api.enqueue_1d(queue, kernel, 1)
        trace = _read_doubles(api, queue, trace_buffer, trace_count)
        input_echoes = _read_ulongs(api, queue, input_echo_buffer, count)
        scalar_echo = _read_ulongs(api, queue, scalar_echo_buffer, 1)[0]
        target_echo = ctypes.c_int()
        api.read_buffer(queue, target_echo_buffer, ctypes.addressof(target_echo), ctypes.sizeof(target_echo))
    finally:
        api.release_mem(target_echo_buffer)
        api.release_mem(scalar_echo_buffer)
        api.release_mem(input_echo_buffer)
        api.release_mem(trace_buffer)
        api.release_mem(input_buffer)

    maximum = max(logits)
    shifted = [value - maximum for value in logits]
    exponentials = [math.exp(value) for value in shifted]
    cpu_sum = sum(exponentials)
    cpu_log = math.log(cpu_sum)
    cpu_lse = maximum + cpu_log
    cpu_loss = CPUBackend().cross_entropy(logits.copy(), target)
    decimal_loss = decimal_cross_entropy(logits, target)
    gpu_maximum = trace[0]
    gpu_shifted = trace[1:1 + count]
    gpu_exponentials = trace[1 + count:1 + 2 * count]
    gpu_partial_sums = trace[1 + 2 * count:1 + 3 * count]
    gpu_sum = gpu_partial_sums[-1]
    log_index = 1 + 3 * count
    gpu_log, gpu_lse, gpu_loss = trace[log_index:log_index + 3]
    return {
        "case": label, "logits": [repr(value) for value in logits],
        "logits_bits": [bits(value) for value in logits], "target": target,
        "target_echo": target_echo.value,
        "target_echo_matches": target_echo.value == target,
        "scalar_argument": repr(scalar_argument),
        "scalar_argument_bits": bits(scalar_argument),
        "scalar_argument_echo_bits": f"{scalar_echo:016x}",
        "scalar_argument_echo_matches": scalar_echo == int(bits(scalar_argument), 16),
        "input_echo_bits": [f"{value:016x}" for value in input_echoes],
        "input_echo_matches": input_echoes == [int(bits(value), 16) for value in logits],
        "cpu": {
            "maximum": repr(maximum), "maximum_bits": bits(maximum),
            "shifted": [repr(value) for value in shifted],
            "shifted_bits": [bits(value) for value in shifted],
            "exp": [repr(value) for value in exponentials],
            "exp_bits": [bits(value) for value in exponentials],
            "sum_builtin": repr(cpu_sum), "sum_builtin_bits": bits(cpu_sum),
            "log": repr(cpu_log), "log_bits": bits(cpu_log),
            "log_sum_exp": repr(cpu_lse), "log_sum_exp_bits": bits(cpu_lse),
            "loss": repr(cpu_loss), "loss_bits": bits(cpu_loss),
        },
        "opencl": {
            "maximum": repr(gpu_maximum), "maximum_bits": bits(gpu_maximum),
            "shifted": [repr(value) for value in gpu_shifted],
            "shifted_bits": [bits(value) for value in gpu_shifted],
            "exp": [repr(value) for value in gpu_exponentials],
            "exp_bits": [bits(value) for value in gpu_exponentials],
            "sum_sequential": repr(gpu_sum), "sum_sequential_bits": bits(gpu_sum),
            "log": repr(gpu_log), "log_bits": bits(gpu_log),
            "log_sum_exp": repr(gpu_lse), "log_sum_exp_bits": bits(gpu_lse),
            "loss": repr(gpu_loss), "loss_bits": bits(gpu_loss),
        },
        "stage_comparisons": {
            "maximum": compare(maximum, gpu_maximum),
            "shifted": _comparison_vector(shifted, gpu_shifted),
            "exp": _comparison_vector(exponentials, gpu_exponentials),
            "sum": compare(cpu_sum, gpu_sum), "log": compare(cpu_log, gpu_log),
            "log_sum_exp": compare(cpu_lse, gpu_lse),
            "loss_vs_cpu_backend": compare(cpu_loss, gpu_loss),
            "loss_vs_decimal_high_precision": compare(decimal_loss, gpu_loss),
        },
        "decimal_high_precision_loss_rounded_to_fp64": repr(decimal_loss),
        "decimal_high_precision_loss_bits": bits(decimal_loss),
    }


def _summary(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group, rows in results.items():
        for row in rows:
            for name, result in row.get("comparisons", {}).items():
                if not isinstance(result, dict) or "status" not in result:
                    continue
                key = f"{group}.{name}"
                item = summary.setdefault(key, {"pass": 0, "fail": 0, "special_match": 0,
                                                "max_absolute_error": 0.0,
                                                "max_relative_error": None, "max_ulps": 0})
                item[result["status"].lower()] += 1
                if isinstance(result["absolute_error"], (int, float)):
                    item["max_absolute_error"] = max(item["max_absolute_error"], result["absolute_error"])
                if isinstance(result["relative_error"], (int, float)):
                    current = item["max_relative_error"]
                    item["max_relative_error"] = result["relative_error"] if current is None else max(current, result["relative_error"])
                if result["ulps"] is not None:
                    item["max_ulps"] = max(item["max_ulps"], result["ulps"])
    return summary


def _single_comparison_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return _summary({"comparison": [{"comparisons": {"value": row}} for row in comparisons]})[
        "comparison.value"
    ]


def _signature(trial: dict[str, Any], group: str) -> Any:
    if group in {"exp", "log"}:
        return [row["output_bits"] for row in trial[group]]
    if group == "sum":
        return [[row["opencl_forward_sum_bits"], row["opencl_reverse_sum_bits"]]
                for row in trial[group]]
    return [{
        "maximum": row["opencl"]["maximum_bits"],
        "shifted": row["opencl"]["shifted_bits"],
        "exp": row["opencl"]["exp_bits"],
        "sum": row["opencl"]["sum_sequential_bits"],
        "log": row["opencl"]["log_bits"],
        "log_sum_exp": row["opencl"]["log_sum_exp_bits"],
        "loss": row["opencl"]["loss_bits"],
    } for row in trial[group]]


def _input_signature(trial: dict[str, Any], group: str) -> Any:
    if group in {"exp", "log"}:
        return [row["input_bits"] for row in trial[group]]
    if group == "sum":
        return [row["input_bits"] for row in trial[group]]
    return [{"logits": row["logits_bits"], "target": row["target"]}
            for row in trial[group]]


def _output_digest(trial: dict[str, Any], group: str) -> str:
    serialized = json.dumps(_signature(trial, group), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _input_digest(trial: dict[str, Any], group: str) -> str:
    serialized = json.dumps(_input_signature(trial, group), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _echo_integrity(trial: dict[str, Any]) -> bool:
    return (
        all(row["echo_bits_match"] for row in trial["exp"])
        and all(row["echo_bits_match"] for row in trial["log"])
        and all(row["input_echo_bits_match"] for row in trial["sum"])
        and all(
            row["input_echo_matches"] and row["scalar_argument_echo_matches"]
            and row["target_echo_matches"]
            for row in trial["cross_entropy"]
        )
    )


def _trial_verification(trials: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = next(trial for trial in trials
                     if trial["options"] == "<default>" and trial["trial"] == min(
                         item["trial"] for item in trials if item["options"] == "<default>"
                     ))
    groups = ("exp", "log", "sum", "cross_entropy")
    canonical_digests = {group: _output_digest(canonical, group) for group in groups}
    checks = []
    for trial in trials:
        checks.append({
            "options": trial["options"], "process_trial": trial["trial"],
            "cases": {group: len(trial[group]) for group in groups},
            "input_sha256": {group: _input_digest(trial, group) for group in groups},
            "output_sha256": {group: _output_digest(trial, group) for group in groups},
            "matches_canonical_output_bits": {
                group: _output_digest(trial, group) == canonical_digests[group]
                for group in groups
            },
            "all_argument_echo_bits_match": _echo_integrity(trial),
        })
    return canonical, _repeatability_from_checks(checks)


def _repeatability_from_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    by_option: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        by_option.setdefault(check["options"], []).append(check)
    result = {}
    for option, rows in by_option.items():
        result[option] = {}
        for group in ("exp", "log", "sum", "cross_entropy"):
            result[option][group] = {
                "fresh_context_and_program_builds": len(rows),
                "bitwise_stable_across_trials": all(
                    row["matches_canonical_output_bits"][group] for row in rows
                ),
            }
    result["across_compilation_options"] = {
        group: all(
            row["matches_canonical_output_bits"][group]
            for row in checks
        )
        for group in ("exp", "log", "sum", "cross_entropy")
    }
    return result


def run(trial_numbers: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    runtime = OpenCLRuntime()
    device = runtime.require_device(0, require_fp64=True)
    api = runtime.api
    all_trials: list[dict[str, Any]] = []
    for option in OPTIONS:
        for trial_number in trial_numbers:
            stack, context, queue, kernels = _make_resources(api, device, option)
            try:
                exp_rows = _run_unary(api, queue, context, kernels, "probe_exp", exp_cases())
                log_rows = _run_unary(api, queue, context, kernels, "probe_log", log_cases())
                sum_rows = [_run_sum(api, queue, context, kernels, label, values)
                            for label, values in sum_cases()]
                ce_rows = [_run_ce(api, queue, context, kernels, label, logits, target)
                           for label, logits, target in ce_cases()]
            finally:
                stack.close()
            # Normalize rows to a common shape so summary accounting is deterministic.
            all_trials.append({"options": option or "<default>", "trial": trial_number,
                               "exp": exp_rows, "log": log_rows, "sum": sum_rows,
                               "cross_entropy": ce_rows})

    aggregates: dict[str, Any] = {}
    ce_stage_aggregates: dict[str, Any] = {}
    for option in OPTIONS:
        selected = [trial for trial in all_trials if trial["options"] == (option or "<default>")]
        groups = {
            "exp_vs_cpu": [{"comparisons": {"cpu": row["vs_cpu"]}}
                           for trial in selected for row in trial["exp"]],
            "exp_vs_decimal": [{"comparisons": {"decimal": row["vs_decimal_rounded_to_fp64"]}}
                               for trial in selected for row in trial["exp"]],
            "log_vs_cpu": [{"comparisons": {"cpu": row["vs_cpu"]}}
                           for trial in selected for row in trial["log"]],
            "log_vs_decimal": [{"comparisons": {"decimal": row["vs_decimal_rounded_to_fp64"]}}
                               for trial in selected for row in trial["log"]],
            "sum_forward_vs_cpu": [{"comparisons": {"cpu": row["forward_vs_cpu"]}}
                                   for trial in selected for row in trial["sum"]],
            "sum_reverse_vs_cpu": [{"comparisons": {"cpu": row["reverse_vs_cpu"]}}
                                   for trial in selected for row in trial["sum"]],
            "ce_vs_cpu": [{"comparisons": {"cpu": row["stage_comparisons"]["loss_vs_cpu_backend"]}}
                          for trial in selected for row in trial["cross_entropy"]],
            "ce_vs_decimal": [{"comparisons": {"decimal": row["stage_comparisons"]["loss_vs_decimal_high_precision"]}}
                              for trial in selected for row in trial["cross_entropy"]],
        }
        aggregates[option or "<default>"] = _summary(groups)
        stage_pairs: dict[str, list[dict[str, Any]]] = {
            name: [] for name in (
                "maximum", "shifted", "exp", "sum", "log", "log_sum_exp",
                "loss_vs_cpu_backend", "loss_vs_decimal_high_precision",
            )
        }
        for trial in selected:
            for row in trial["cross_entropy"]:
                for stage, comparison_result in row["stage_comparisons"].items():
                    if isinstance(comparison_result, list):
                        stage_pairs[stage].extend(comparison_result)
                    else:
                        stage_pairs[stage].append(comparison_result)
        ce_stage_aggregates[option or "<default>"] = {
            stage: _single_comparison_summary(comparisons)
            for stage, comparisons in stage_pairs.items()
        }

    version_text = Path("/proc/version").read_text().strip() if Path("/proc/version").exists() else ""
    source_path = Path(__file__).resolve()
    package_query = subprocess.run(
        ["rpm", "-q", "mesa-dri-drivers", "mesa-libOpenCL", "llvm", "llvm-libs",
         "libclc22", "libclc22-spirv"],
        capture_output=True, text=True, check=False,
    )
    canonical, repeatability = _trial_verification(all_trials)
    return {
        "schema": 1,
        "purpose": "experimental F2.3a-P0 only; no production kernel or backend changed",
        "baseline": "7bff9617a340faab342f5c844464a94d0acec6d0",
        "python": sys.version,
        "platform": platform.platform(),
        "kernel": version_text,
        "runtime_packages": {
            "query_command": "rpm -q mesa-dri-drivers mesa-libOpenCL llvm llvm-libs libclc22 libclc22-spirv",
            "stdout": package_query.stdout.strip().splitlines(),
            "stderr": package_query.stderr.strip().splitlines(),
            "returncode": package_query.returncode,
        },
        "environment": {key: os.environ.get(key) for key in ("RUSTICL_ENABLE", "RUSTICL_FEATURES")},
        "device": {
            "name": device.name, "vendor": device.vendor, "vendor_id": f"0x{device.vendor_id:04x}",
            "pci_bdf": device.pci_bdf, "driver": device.driver_version,
            "opencl": device.version, "opencl_c": device.opencl_c_version,
            "platform": device.platform_name, "platform_vendor": device.platform_vendor,
            "fp64_config": device.double_fp_config,
        },
        "implementation": {
            "cpu_oracle": "kinera.backends.cpu.CPUBackend.cross_entropy",
            "high_precision": "decimal.Decimal, exact float inputs via Decimal.from_float; 100/120 digit contexts",
            "opencl_source_sha256": hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
            "diagnostic_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "compilation_options": list(OPTIONS), "trials_per_option": len(trial_numbers),
            "atol": ATOL, "rtol": RTOL,
            "comparison_rule": "abs(actual-expected) <= max(atol, rtol*max(abs(actual),abs(expected)))",
            "cpu_sum": "built-in sum(generator) on the recorded Python interpreter",
            "python_float_sum_compensated_probe": sum((1.0e16, 1.0, -1.0e16)) == 1.0,
            "opencl_sum": "explicit sequential left-to-right FP64 accumulation; reverse-order probe is separate",
        },
        "matrices": {
            "E1_exp": ["zero and signed zero", "negative near zero", "-1/-10/-100", "underflow boundaries", "extreme negatives"],
            "E2_log": ["neighbors of one", "small positive and subnormal", "sums/classes magnitudes"],
            "E3_sum": ["order and cancellation", "short/long positive sequences", "Python sum vs fsum/Decimal", "sequential/reverse device order"],
            "E4_cross_entropy": ["vocab 1/2/many", "equal/near/extreme logits", "target first/middle/last", "underflow and final cancellation", "full intermediate trace"],
            "E5_data_integrity": ["buffer input bits echoed from device", "scalar double and integer target echoed"],
            "E6_compilation": ["default", "-cl-opt-disable", "-cl-fast-relaxed-math (diagnostic only; not a production recommendation)"],
            "E7_future_dependencies": ["softmax/division/CE gradient and BCE are explicitly out of scope"],
        },
        "summary_by_compilation": aggregates,
        "cross_entropy_stage_summary_by_compilation": ce_stage_aggregates,
        "repeatability": repeatability,
        "canonical_results": canonical,
        "trial_verification": [
            {
                "options": trial["options"], "process_trial": trial["trial"],
                "cases": {group: len(trial[group]) for group in ("exp", "log", "sum", "cross_entropy")},
                "input_sha256": {group: _input_digest(trial, group)
                                 for group in ("exp", "log", "sum", "cross_entropy")},
                "output_sha256": {group: _output_digest(trial, group)
                                  for group in ("exp", "log", "sum", "cross_entropy")},
                "matches_canonical_output_bits": {
                    group: _output_digest(trial, group) == _output_digest(canonical, group)
                    for group in ("exp", "log", "sum", "cross_entropy")
                },
                "all_argument_echo_bits_match": _echo_integrity(trial),
            }
            for trial in all_trials
        ],
    }


def _merge_summaries(target: dict[str, Any], source: dict[str, Any]) -> None:
    for option, metrics in source.items():
        target_metrics = target[option]
        for name, values in metrics.items():
            current = target_metrics[name]
            current["pass"] += values["pass"]
            current["fail"] += values["fail"]
            current["special_match"] += values["special_match"]
            current["max_absolute_error"] = max(
                current["max_absolute_error"], values["max_absolute_error"]
            )
            left, right = current["max_relative_error"], values["max_relative_error"]
            if right is not None:
                current["max_relative_error"] = right if left is None else max(left, right)
            current["max_ulps"] = max(current["max_ulps"], values["max_ulps"])


def _combine_process_results(reports: list[dict[str, Any]]) -> dict[str, Any]:
    combined = reports[0]
    checks = [check for report in reports for check in report["trial_verification"]]
    reference_outputs = reports[0]["trial_verification"][0]["output_sha256"]
    if any(
        check["output_sha256"] != reference_outputs
        for report in reports for check in report["trial_verification"]
    ):
        raise RuntimeError("independent process outputs are not bitwise identical")
    if any(
        report["canonical_results"]["options"] != "<default>"
        for report in reports
    ):
        raise RuntimeError("missing default-options canonical trace")
    combined["trial_verification"] = checks
    combined["implementation"]["trials_per_option"] = len(reports)
    for report in reports[1:]:
        _merge_summaries(combined["summary_by_compilation"], report["summary_by_compilation"])
        _merge_summaries(combined["cross_entropy_stage_summary_by_compilation"],
                         report["cross_entropy_stage_summary_by_compilation"])
    combined["repeatability"] = _repeatability_from_checks(checks)
    combined["independent_process_repetitions"] = len(reports)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--single-trial", type=int, choices=range(1, TRIALS + 1),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_trial is not None:
        result = run((args.single_trial,))
    else:
        reports = []
        captured_stderr = []
        with tempfile.TemporaryDirectory(prefix="kinera-f23a-p0-") as temp_dir:
            for trial_number in range(1, TRIALS + 1):
                temp_output = Path(temp_dir) / f"trial-{trial_number}.json"
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--output", str(temp_output),
                     "--single-trial", str(trial_number)],
                    check=False, capture_output=True, text=True,
                )
                captured_stderr.append(child.stderr)
                if child.returncode != 0:
                    raise RuntimeError(
                        f"independent diagnostic process {trial_number} failed: {child.stderr}"
                    )
                reports.append(json.loads(temp_output.read_text()))
        for warning in captured_stderr:
            if warning:
                print(warning, file=sys.stderr, end="")
        result = _combine_process_results(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(result['trial_verification'])} independently verified option/process runs to {args.output}")


if __name__ == "__main__":
    main()
