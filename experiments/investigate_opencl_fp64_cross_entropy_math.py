#!/usr/bin/env python3
"""Experimental scalar math and CE composition; never a production kernel."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import struct
import sys
from contextlib import ExitStack
from decimal import Decimal, localcontext
from pathlib import Path

from kinera.backends.cpu.backend import CPUBackend
from kinera.backends.opencl.runtime import OpenCLRuntime

BASELINE = "7bff9617a340faab342f5c844464a94d0acec6d0"
SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void unary(__global const double *x, double scalar, int op, __global double *o) {
  o[0] = x[0]; o[1] = scalar;
  o[2] = op == 0 ? exp(x[0]) : log(x[0]);
  o[3] = op == 0 ? exp(scalar) : log(scalar);
}
__kernel void reduction(__global const double *x, int n, int compensated, __global double *o) {
  double hi=0.0, lo=0.0;
  for(int i=0; i<n; ++i) {
    double v=x[i], next=hi+v;
    if (compensated) { if (fabs(hi)>=fabs(v)) lo+=(hi-next)+v; else lo+=(v-next)+hi; }
    hi=next;
  }
  if(compensated && lo != 0.0 && isfinite(lo)) hi+=lo;
  o[0]=hi;
}
__kernel void compose(__global const double *x, int n, int target, int compensated, __global double *o) {
  double m=x[0];
  for(int i=1;i<n;i++) if(x[i]>m) m=x[i];
  double hi=0.0,lo=0.0;
  for(int i=0;i<n;i++) {
    double d=x[i]-m, v=exp(d), next=hi+v;
    o[5+i]=d; o[5+n+i]=v; o[5+2*n+i]=x[i];
    if(compensated) { if(fabs(hi)>=fabs(v)) lo+=(hi-next)+v; else lo+=(v-next)+hi; }
    hi=next;
  }
  if(compensated && lo != 0.0 && isfinite(lo)) hi+=lo;
  double l=log(hi), lse=m+l;
  o[0]=m; o[1]=hi; o[2]=l; o[3]=lse; o[4]=lse-x[target];
}
"""
OPTIONS = ("", "-cl-opt-disable")
EXPS = [0., -0., -2**-52, -1e-15, -1e-8, -0.1, -0.5, -math.log(2.), -1., -2., -5., -10., -20., -50., -100., -700., -708., -709., -744., -745., -746., -1000., -math.inf]
LOGS = [1., math.nextafter(1., 2.), 1.+2**-50, 1.00000001, 1.5, 2., 3., 7., 26., 51., 257., 3932., sum(math.exp(v-3.) for v in [3., -2., 0., 1., -10.]), sum(math.exp(v-1e16) for v in [1e16, 1e16, 1e16-2.])]
SUMS = {"unit": [1.], "ties": [1., 2**-53, 2**-53], "cancel": [1e16, 1., -1e16],
        "alternating": [1e16, 1., -1e16, -1.] * 32,
        "positive_long": [1.] + [2**-53] * 256, "positive_exp": [math.exp(-i/16) for i in range(257)]}
CE_CASES = {
 "unit": ([2.5], [0]), "equal2": ([0., 0.], [0,1]),
 "small_gap": ([0., -1e-15, -2e-15], [0,1,2]),
 "mixed": ([3., -2., 0., 1., -10.], [0,2,4]),
 "extreme": ([1000., 0., -1000.], [0,1,2]),
 "finite_overflow_difference": ([1e308, -1e308], [0,1]),
 "underflow": ([0., -700., -744., -745., -746., -1000.], [0,3,5]),
 "cancel_final": ([1e16, 1e16, 1e16 - 2.], [0,2]),
 "long_equal": ([0.] * 257, [0,128,256]),
 "head_equal_26": ([0.] * 26, [0,13,25]),
 "long_mixed": ([-i/16 for i in range(257)], [0,128,256]),
 "near_tie": ([1., math.nextafter(1., 2.), math.nextafter(1., 0.)], [0,1,2]),
}


def bits(x: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', x))[0]:016x}"


def value(hex_bits: str) -> float:
    return struct.unpack('>d', bytes.fromhex(hex_bits))[0]


def ordered(x: float) -> int:
    b = int(bits(x), 16)
    return (~b & ((1 << 64)-1)) if b >> 63 else b | (1 << 63)


def compare(expected: float, actual: float) -> dict:
    row = {"expected":repr(expected), "expected_bits":bits(expected),
           "actual":repr(actual), "actual_bits":bits(actual)}
    if not math.isfinite(expected) or not math.isfinite(actual):
        ok = (math.isnan(expected) and math.isnan(actual)) or expected == actual
        row.update(abs_error=None, rel_error=None, ulps=None, status="PASS" if ok else "FAIL")
    else:
        err = abs(expected-actual)
        rel = err/abs(expected) if expected else None
        row.update(abs_error=err if math.isfinite(err) else repr(err),
                   rel_error=rel if rel is None or math.isfinite(rel) else repr(rel),
                   ulps=abs(ordered(expected)-ordered(actual)),
                   status="PASS" if math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12) else "FAIL")
    return row


def decimal_unary(op: str, x: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 400
        a = Decimal.from_float(x)
        return float(a.exp() if op == "exp" else a.ln())


def decimal_ce(logits: list[float], target: int) -> float:
    with localcontext() as ctx:
        ctx.prec = 400
        xs = [Decimal.from_float(x) for x in logits]
        m = max(xs)
        return float(m + sum((v-m).exp() for v in xs).ln() - xs[target])


def run() -> dict:
    runtime = OpenCLRuntime(); device = runtime.require_device(0, require_fp64=True); api = runtime.api
    compensated = sum((1e16, 1., -1e16)) == 1.
    records = []
    with ExitStack() as cleanup:
        ctx = api.create_context(device.handle); cleanup.callback(api.release_context, ctx)
        queue = api.create_command_queue(ctx, device.handle); cleanup.callback(api.release_command_queue, queue)
        for options in OPTIONS:
            with ExitStack() as prog_clean:
                program = api.build_program(ctx, device.handle, SOURCE, options); prog_clean.callback(api.release_program, program)
                kernels = {}
                for name in ("unary", "reduction", "compose"):
                    kernels[name] = api.create_kernel(program, name); prog_clean.callback(api.release_kernel, kernels[name])
                def invoke(kind: str, inputs: list[float], args: list[tuple[str, float]], count: int):
                    with ExitStack() as mem:
                        i = api.create_buffer(ctx, len(inputs)*8); mem.callback(api.release_mem, i)
                        o = api.create_buffer(ctx, count*8); mem.callback(api.release_mem, o)
                        arr = (ctypes.c_double*len(inputs))(*inputs)
                        api.write_buffer(queue, i, ctypes.addressof(arr), ctypes.sizeof(arr))
                        k = kernels[kind]; api.set_kernel_buffer(k, 0, i)
                        for index, (typ, v) in enumerate(args, 1):
                            (api.set_kernel_double if typ == "double" else api.set_kernel_int)(k, index, v)
                        api.set_kernel_buffer(k, len(args)+1, o)
                        runs=[]
                        for _ in range(3):
                            api.enqueue_1d(queue, k, 1); out=(ctypes.c_double*count)()
                            api.read_buffer(queue, o, ctypes.addressof(out), ctypes.sizeof(out))
                            runs.append([float(v) for v in out])
                        return runs[0], all([bits(v) for v in run] == [bits(v) for v in runs[0]] for run in runs[1:])
                for op, samples in (("exp", EXPS), ("log", LOGS)):
                    for x in samples:
                        vals, stable = invoke("unary", [x], [("double",x),("int",0 if op=="exp" else 1)], 4)
                        expected = (math.exp if op == "exp" else math.log)(x)
                        records.append({"group":op, "options":options, "input":repr(x), "input_bits":bits(x),
                          "buffer_echo_bits":bits(vals[0]), "scalar_echo_bits":bits(vals[1]), "stable":stable,
                          "decimal_bits":bits(decimal_unary(op,x)), "buffer":compare(expected,vals[2]), "scalar":compare(expected,vals[3])})
                for name, xs in SUMS.items():
                    modes={}; stability=[]
                    for mode in (0,1):
                        vals, stable=invoke("reduction",xs,[("int",len(xs)),("int",mode)],1)
                        modes["compensated" if mode else "naive"]=compare(sum(v for v in xs),vals[0]); stability.append(stable)
                    records.append({"group":"sum", "options":options,"name":name, "inputs_bits":[bits(x) for x in xs],
                      "selected":"compensated" if compensated else "naive", "fsum_bits":bits(math.fsum(xs)), "modes":modes, "stable":all(stability)})
                cpu = CPUBackend()
                for name,(xs,targets) in CE_CASES.items():
                    for target in targets:
                        vals, stable=invoke("compose",xs,[("int",len(xs)),("int",target),("int",int(compensated))],5+3*len(xs))
                        maximum=max(xs); shifts=[v-maximum for v in xs]; es=[math.exp(v) for v in shifts]
                        total=sum(v for v in es); logarithm=math.log(total); lse=maximum+logarithm
                        cpu_loss=cpu.cross_entropy(xs,target)
                        assert bits(cpu_loss)==bits(lse-xs[target])
                        pairs={"maximum":compare(maximum,vals[0]),"sum":compare(total,vals[1]),"log":compare(logarithm,vals[2]),"log_sum_exp":compare(lse,vals[3]),"loss":compare(cpu_loss,vals[4]),
                               "shift":[compare(x,y) for x,y in zip(shifts,vals[5:5+len(xs)])],
                               "exp":[compare(x,y) for x,y in zip(es,vals[5+len(xs):5+2*len(xs)])]}
                        records.append({"group":"ce", "options":options,"name":name,"target":target,"inputs_bits":[bits(x) for x in xs],
                          "echo_bits":[bits(x) for x in vals[5+2*len(xs):]], "decimal_loss_bits":bits(decimal_ce(xs,target)), "stable":stable,"stages":pairs})
    return {"schema":1,"baseline":BASELINE,"source_sha256":hashlib.sha256(SOURCE.encode()).hexdigest(),
            "python":sys.version,"platform":platform.platform(),"sum_compensated":compensated,
            "environment":{k:os.environ.get(k) for k in ("RUSTICL_ENABLE","RUSTICL_FEATURES")},
            "device":{"name":device.name,"pci":device.pci_bdf,"driver":device.driver_version,"runtime":device.version,"opencl_c":device.opencl_c_version,"fp64_config":device.double_fp_config},
            "repetitions_per_case":3,"records":records}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    data=run(); args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); print('records:',len(data['records']))


if __name__ == '__main__': main()
