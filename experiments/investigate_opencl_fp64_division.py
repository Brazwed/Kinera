#!/usr/bin/env python3
"""Deterministic FP64 division diagnostic. Requires a real OpenCL FP64 GPU.

Run: RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src \
  python experiments/investigate_opencl_fp64_division.py --output results.json
"""
from __future__ import annotations

import argparse
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
import platform
import struct
import sys
from contextlib import ExitStack
from pathlib import Path

from kinera.backends.opencl.runtime import OpenCLRuntime

SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void probe(__global const double *inp, double scalar,
                    double host_inverse, __global double *out) {
    const double n = inp[0], d = inp[1];
    out[0] = scalar;
    out[1] = d;
    out[2] = n / 3.0;
    out[3] = n / scalar;
    out[4] = n / d;
    out[5] = n * host_inverse;
    out[6] = n * (1.0 / d);
    out[7] = n;
}
"""
CASES = (
    ("one_third", 1., 3.), ("one_seventh", 1., 7.),
    ("two_thirds", 2., 3.), ("negative_n", -1., 3.),
    ("negative_d", 1., -3.), ("fractional", 1., 0.1),
    ("power_two", 1., 8.), ("fractional_exact", 2.5, 0.5),
    ("small", 1e-200, 3.), ("large", 1e200, 7.),
    ("ratio_small", 1e-300, 1e100), ("ratio_large", 1e300, 1e-100),
    ("min_normal", sys.float_info.min, 3.),
    ("min_subnormal", math.ulp(0.0), 1.),
    ("near_max", sys.float_info.max, 2.),
    ("large_divisor", 1., 1e300), ("tiny_divisor", 1., 1e-300),
    ("zero_positive", 1., 0.), ("zero_negative", 1., -0.),
    ("zero_zero", 0., 0.), ("infinity_denominator", 1., math.inf),
    ("negative_zero_result", -1., math.inf),
    ("infinity_numerator", math.inf, 3.), ("nan_numerator", math.nan, 3.),
)
OPTIONS = ("", "-cl-opt-disable", "-cl-fast-relaxed-math")
VARIANTS = ("literal_3", "scalar", "buffer", "host_reciprocal", "device_reciprocal")


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def ordered(value: float) -> int:
    raw = int(bits(value), 16)
    return (~raw & ((1 << 64) - 1)) if raw >> 63 else raw | (1 << 63)


def reference(n: float, d: float) -> float:
    if math.isnan(n) or math.isnan(d):
        return math.nan
    if d == 0.0:
        if n == 0.0:
            return math.nan
        return math.copysign(math.inf, math.copysign(1., n) * math.copysign(1., d))
    return n / d


def compare(expected: float, actual: float) -> dict:
    result = {"expected": repr(expected), "expected_bits": bits(expected),
              "actual": repr(actual), "actual_bits": bits(actual)}
    if not math.isfinite(expected) or not math.isfinite(actual):
        equivalent = (math.isnan(expected) and math.isnan(actual)) or expected == actual
        result.update(absolute_error=None, relative_error=None, ulps=None,
                      status="PASS" if equivalent else "FAIL", category="special")
    else:
        error = abs(actual - expected)
        relative = error / abs(expected) if expected != 0. else None
        # math.isclose follows the Kinera gate's absolute OR relative rule.
        result.update(absolute_error=error if math.isfinite(error) else repr(error),
                      relative_error=relative if relative is None or math.isfinite(relative) else repr(relative),
                      ulps=abs(ordered(actual) - ordered(expected)),
                      status="PASS" if math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12) else "FAIL",
                      category="finite")
        if expected == 0.0 and actual == 0.0:
            result["zero_sign_matches"] = bits(expected) == bits(actual)
    return result


def rational_reference(n: float, d: float) -> float | None:
    """Correctly rounded quotient of the exact binary inputs, if finite and nonzero."""
    if not math.isfinite(n) or not math.isfinite(d) or d == 0.0:
        return None
    try:
        return float(Fraction.from_float(n) / Fraction.from_float(d))
    except OverflowError:
        return math.copysign(math.inf, n * math.copysign(1.0, d))


def run() -> dict:
    runtime = OpenCLRuntime()
    device = runtime.require_device(0, require_fp64=True)
    api = runtime.api
    results = []
    with ExitStack() as stack:
        context = api.create_context(device.handle)
        stack.callback(api.release_context, context)
        queue = api.create_command_queue(context, device.handle)
        stack.callback(api.release_command_queue, queue)
        inp = api.create_buffer(context, 16)
        stack.callback(api.release_mem, inp)
        out = api.create_buffer(context, 64)
        stack.callback(api.release_mem, out)
        for options in OPTIONS:
            program = api.build_program(context, device.handle, SOURCE, options)
            with ExitStack() as program_stack:
                program_stack.callback(api.release_program, program)
                kernel = api.create_kernel(program, "probe")
                program_stack.callback(api.release_kernel, kernel)
                api.set_kernel_buffer(kernel, 0, inp)
                api.set_kernel_buffer(kernel, 3, out)
                for label, n, d in CASES:
                    values = (ctypes.c_double * 2)(n, d)
                    api.write_buffer(queue, inp, ctypes.addressof(values), ctypes.sizeof(values))
                    api.set_kernel_double(kernel, 1, d)
                    api.set_kernel_double(kernel, 2, reference(1., d))
                    repetitions = []
                    for _ in range(3):
                        api.enqueue_1d(queue, kernel, 1)
                        answer = (ctypes.c_double * 8)()
                        api.read_buffer(queue, out, ctypes.addressof(answer), ctypes.sizeof(answer))
                        repetitions.append([float(x) for x in answer])
                    observed = repetitions[0]
                    # Compare exact bits, including special values and signs.
                    stable = all([bits(v) for v in rep] == [bits(v) for v in observed] for rep in repetitions[1:])
                    row = {"case": label, "numerator": repr(n), "numerator_bits": bits(n),
                           "denominator": repr(d), "denominator_bits": bits(d),
                           "options": options, "repetitions": len(repetitions), "stable_bits": stable,
                           "scalar_echo_bits": bits(observed[0]), "buffer_echo_bits": bits(observed[1]),
                           "numerator_echo_bits": bits(observed[7]),
                           "rational_reference_bits": None if rational_reference(n, d) is None else bits(rational_reference(n, d)),
                           "variants": {}}
                    for index, name in enumerate(VARIANTS, start=2):
                        if name == "literal_3" and d != 3.:
                            continue
                        row["variants"][name] = compare(reference(n, d), observed[index])
                    results.append(row)
    return {"schema": 1, "baseline": "35882d8c4ffb4ba46720267e500118d2a1ce624d",
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "python": sys.version, "platform": platform.platform(),
            "environment": {k: os.environ.get(k) for k in ("RUSTICL_ENABLE", "RUSTICL_FEATURES")},
            "device": {"name": device.name, "pci": device.pci_bdf, "driver": device.driver_version,
                       "runtime": device.version, "opencl_c": device.opencl_c_version,
                       "fp64_config": device.double_fp_config},
            "options": OPTIONS, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(result['results'])} cases to {args.output}")


if __name__ == "__main__":
    main()
