# P3e — OpenCL device profiling on the RX 550

Date: 2026-11-05

Repository baseline: `main` at `1759bd196dee268e4a9e0e80f6820e0a37db36ed`

Worktree branch: `feature/issue-52-p3e-device-profiling`

## Goal and result

P3e separates host-observed `Linear` latency from OpenCL command timestamps on the GPU. On the tested Mesa Rusticl RX 550, the command queue supplied valid profiling timestamps for every measured kernel and transfer. The event interval `END - START` therefore provides a direct measurement of the command's device-side execution interval in this environment. It does not include Python preparation, host enqueue work, queue delay, or the host's wait in `clFinish`.

This is evidence about these kernels, shapes, driver, and run conditions. It is not a claim that the backend is faster than CPU, and the instrumented host times are not equivalent to uninstrumented production timings.

## Static audit and profiling design

The production `OpenCLApi` creates a command queue with properties `0`. Its buffer transfers are blocking (`CL_TRUE`); `enqueue_1d` submits one NDRange kernel and calls `clFinish`. `Linear.forward`, `Linear.backward`, and `Linear.step` each invoke one kernel. Backward's one kernel produces `grad_weights`, `grad_bias`, and `grad_inputs`. An integrated resident cycle invokes forward, backward, and step serially; each existing backend call finishes before the next begins.

Production does not return kernel events or expose `clGetEventProfilingInfo` or `clReleaseEvent`. The experiment therefore adds no production hooks. `experiments/profile_opencl_device.py` uses `ctypes` and the existing API as a base, binds the missing event functions in an experiment-only subclass, creates a separate queue with `CL_QUEUE_PROFILING_ENABLE`, and verifies the resulting queue property using `clGetCommandQueueInfo`. It requests one event for each kernel enqueue and blocking transfer. The original `clFinish` and blocking-transfer semantics remain in place.

After command completion, the harness queries `CL_PROFILING_COMMAND_QUEUED`, `SUBMIT`, `START`, and `END`, then releases each event once. Missing or invalid profiling data is represented explicitly as unavailable. Release and error paths are covered by deterministic mock tests. The measured run reported no unavailable timestamps or event release failures.

The directly measured intervals are:

- Host operation and API component durations from `time.perf_counter_ns()`.
- Device event durations `END - START`, plus device-clock queue intervals `SUBMIT - QUEUED` and `START - SUBMIT`.
- Blocking H2D/D2H host call durations and their OpenCL event intervals.
- Setup and program build durations, measured separately from timed operations.

Host and device timestamps have separate clock origins. Only durations within one clock domain are compared; no host timestamp is subtracted from a device timestamp. The host `clFinish` component includes waiting and runtime/driver behavior, so it is not treated as kernel execution. Inclusive operation timings are not added to their component timings. For the integrated cycle, the reported device total is the per-sample sum of the three serial event durations; it is not a sum of host times.

## Protocol and environment

The harness reused the deterministic cases and seeds from `experiments/benchmark_opencl_linear.py`, with `float64`, `atol=rtol=1e-12`, and learning rate `0.03125`.

| Weights shape | Seed | Scenario |
|---|---:|---|
| `1×1` | 52052 | Minimum operation |
| `51×26` | 52053 | Head-like intermediate dimensions |
| `3932×26` | 52054 | Long batch dimension |
| `26×3932` | 52055 | Long reduction dimension |

Each case used five warmups and 30 timed repetitions in each of three independent worker processes. Workers ran sequentially. In each process, profiled-queue and normal-queue controls used paired data and alternated execution order. No P3d, training, historical pilot, or other benchmark was launched concurrently. Every operation remained resident between stages; transfers were measured separately. The cycle collected one event for each of its three kernels.

| Component | Observed value |
|---|---|
| GPU | AMD Radeon RX 550 / 550 Series, PCI `0000:03:00.0`, 4 GiB VRAM |
| CPU | Intel Xeon E5-2680 v4, 28 logical processors |
| OS/kernel | Linux, kernel `7.2.11-300.fc45.x86_64`, glibc 2.44 |
| Python | CPython 3.15.0 |
| OpenCL platform/runtime | Mesa/X.org Rusticl, driver 26.2.6, OpenCL 3.1, OpenCL C 1.2 |
| FP64 | `cl_khr_fp64`; device double FP config `63`; harness enabled `RUSTICL_FEATURES=fp64` |
| Profiling timer resolution | 40 ns reported by `clinfo` |
| Queue | `CL_QUEUE_PROFILING_ENABLE` verified; property value `2` |
| Driver warning | `Patched Mesa libclc not detected` |

CPU governor was `performance`; observed CPU frequency was about 2.395 GHz. Load averages were 1.37–1.92 on the 28-processor host. GPU busy snapshots were 0% before workers and 0–19% after them. VRAM usage snapshots varied from about 858–882 MiB before to 894–930 MiB after. These snapshots cannot identify which process or external client caused the change, so exclusive GPU ownership is not claimed. GPU clock was observed at 734 MHz in the initial snapshots; the harness records the available before/after clock and utilization fields per worker.

## Device and host results

The table reports the median device execution interval with the pooled first and third quartiles from 90 samples per shape and operation. For `resident_cycle`, the device column is the per-sample sum of its forward, backward, and step event durations. The host column is the public `Linear` operation median in milliseconds on the profiled queue and its paired normal-queue control. These host values include the layer, API, and synchronization costs and are not device-only measurements.

| Shape | Operation | Device event median [Q1, Q3] (µs) | Host profiled / normal queue (ms) |
|---|---|---:|---:|
| `1×1` | forward | 9.12 [8.00, 10.40] | 0.3126 / 0.2067 |
| `1×1` | backward | 29.92 [28.96, 36.16] | 0.3463 / 0.2303 |
| `1×1` | step | 7.44 [7.20, 9.60] | 0.2099 / 0.1355 |
| `1×1` | resident cycle, sum of events | 45.28 [40.04, 56.48] | 0.8423 / 0.5587 |
| `51×26` | forward | 37.60 [33.76, 57.52] | 0.3407 / 0.2408 |
| `51×26` | backward | 88.48 [86.08, 110.36] | 0.3947 / 0.2852 |
| `51×26` | step | 8.00 [7.84, 8.16] | 0.2038 / 0.1318 |
| `51×26` | resident cycle, sum of events | 134.00 [129.36, 171.96] | 0.9295 / 0.6577 |
| `3932×26` | forward | 312.80 [311.40, 320.92] | 0.8098 / 0.6830 |
| `3932×26` | backward | 4,423.36 [4,113.40, 4,433.36] | 5.1435 / 4.8936 |
| `3932×26` | step | 164.64 [80.72, 165.12] | 0.5284 / 0.4638 |
| `3932×26` | resident cycle, sum of events | 4,902.80 [4,611.44, 5,020.72] | 6.5369 / 5.9807 |
| `26×3932` | forward | 4,579.12 [3,156.32, 4,583.64] | 5.0657 / 4.9758 |
| `26×3932` | backward | 334.72 [325.96, 338.96] | 1.0632 / 0.7847 |
| `26×3932` | step | 158.08 [76.64, 158.56] | 0.4939 / 0.4063 |
| `26×3932` | resident cycle, sum of events | 4,847.68 [3,086.36, 5,080.92] | 6.3119 / 5.5534 |

The `26×3932` forward still has high dispersion: its pooled event IQR is about 1.427 ms, and the three process-level medians ranged from about 2.685 to 4.582 ms. Its process-median IQR is also wide. This is consistent with the variability previously observed in P3, but P3 and P3e use different harnesses and are not pooled. The `3932×26` step also has a broad event distribution. These results warrant repeating under better-controlled GPU occupancy before making a tighter performance claim.

For scale, `1×1` forward's host `clFinish` median was about 137 µs while its device event median was 9.12 µs. For `26×3932` forward, the host wait median was about 4.853 ms and device execution median was 4.579 ms. The differences are not computed by subtracting device time from host time: these measurements use distinct clocks and the host wait includes driver scheduling and synchronization work.

The event queue intervals are independently available. For `26×3932` forward, median `SUBMIT - QUEUED` was 48.88 µs and `START - SUBMIT` was 112.16 µs, followed by 4.579 ms for `END - START`. For `3932×26` forward the corresponding intervals were 25.60 µs, 104.72 µs, and 312.80 µs. These are device-clock durations; the first two should not be conflated with Python enqueue-call latency.

For illustration, the profiler also captures an exclusive host-side accounting for each sample. The following are separately aggregated medians for forward; because a median of components need not equal the median of their sum, these columns are not an additive latency equation. The JSON preserves per-sample components and the harness checks that exclusive components plus validation and residual account for the measured operation wall time.

| Forward host component median | `1×1` (µs) | `26×3932` (µs) |
|---|---:|---:|
| Public operation wall time | 312.61 | 5,065.68 |
| `Linear` backend-array validation | 8.40 | 11.38 |
| Remaining Python/layer work | 71.16 | 87.76 |
| OpenCL buffer creation API | 9.12 | 12.83 |
| Kernel argument API calls | 18.25 | 19.82 |
| Native kernel enqueue call | 13.86 | 16.59 |
| Host `clFinish` wait | 137.13 | 4,853.42 |
| Event profiling query | 23.23 | 31.11 |
| Event release | 2.48 | 3.28 |

Kernel creation/release calls and the remaining enqueue-wrapper work are also recorded in the structured artifact. They are not repeated in the compact table above.

The paired profiled-minus-control host median was shape-dependent: about 96–152 µs for forward, 113–277 µs for backward, 74–102 µs for step, and 270–653 µs for a resident cycle. For `26×3932` forward, the paired delta IQR spans negative and positive values (about -55 to 299 µs). Profiling queue behavior, event creation/query/release, and run noise are therefore not represented by a single correction factor. These paired host deltas describe instrumentation effects; they are not kernel-time adjustments.

### Transfers and setup

Every transfer event was available. Median device event interval and blocking host call time were:

| Shape | H2D event (µs) | H2D host call (ms) | D2H event (µs) | D2H host call (ms) |
|---|---:|---:|---:|---:|
| `1×1` | 0.80 | 0.1149 | 58.72 | 0.1837 |
| `51×26` | 7.68 | 0.1224 | 56.48 | 0.1725 |
| `3932×26` | 165.12 | 0.4342 | 281.12 | 0.4522 |
| `26×3932` | 164.96 | 0.4577 | 284.64 | 0.4914 |

These transfer event intervals are distinct from host call latency. They do not isolate all host marshaling or allocation costs discussed in P3c.

Initialization and build were recorded apart from operation timings. With only three process-level setup samples, the profiled/control medians were: API/runtime discovery 116.48/3.81 ms, queue/context creation 0.431/0.378 ms, and layer/parameter/input setup 2.919/0.330 ms. Program build medians were 8.274/1.826 ms for forward, 3.456/3.045 ms for backward, and 1.552/1.505 ms for step. These setup figures are environment- and process-order-sensitive; they should not be interpreted as a stable per-operation profiling tax. The event query itself had a median host cost around 23–31 µs for a single-kernel operation, plus about 2–3 µs for event release.

## Numerical checks and resource integrity

All profiled operations were compared with the CPU reference using the original deterministic inputs and `atol=rtol=1e-12`. The maximum absolute numerical error reported for forward output, backward gradients, step parameters, and the integrated cycle was `0.0`. The transfer roundtrips also matched the prepared host values. No CPU, CUDA, OpenCL production source, kernel, mathematical contract, or historical P3/P3b/P3c artifact was modified for this experiment.

The mock tests cover event release on successful queries, unavailable timestamps, invalid timestamp order, enqueue/finish failures, and a failing event-release call. The real run released every captured event successfully and reported all timestamps as valid. Event completion is forced by the existing synchronous `clFinish` or blocking transfer before query; the experiment adds no second device-wide finish. Asynchronous failure behavior beyond these completed event paths was not tested.

The warning `Patched Mesa libclc not detected` appeared during the Rusticl run. It did not prevent event profiling or the exact CPU equivalence observed here, but it remains a driver reliability caveat; these measurements do not establish correctness for untested inputs or kernels.

## Context with P3/P3b/P3c

P3/P3b primarily measured host wall time around direct backend calls and could not distinguish kernel execution from enqueue and wait time. P3c showed the cost of Python flattening, contiguous host preparation, and result materialization, particularly for the large shapes. P3e's operation measurements use resident buffers through the public `Linear` layer and report device event intervals separately; the separately measured transfer calls retain their blocking semantics. The P3 numbers remain useful only as historical context because they differ in layer coverage and instrumentation. The host/device separation here confirms that an OpenCL operation's wall latency must not be reported as GPU kernel time.

## Reproduction and artifact integrity

From the repository root on a machine exposing the RX 550 through Mesa Rusticl:

```bash
RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64 PYTHONPATH=src:experiments \
  python experiments/profile_opencl_device.py \
  --process-runs 3 --warmups 5 --repetitions 30 \
  --output experiments/results/opencl-rx550-p3e-device-profiling.json
```

Recorded base commit: `1759bd196dee268e4a9e0e80f6820e0a37db36ed`.

Source fingerprint (production contracts/kernels, P3 baseline harness, and P3e harness): `a50ee80ae65123504e607ccb0fc615c9fd76a38a6a40bdc86fb036972bc4b868`.

For publication, the JSON was reserialized with compact whitespace after GitHub's authenticated upload endpoint rejected the pretty-printed 7.8 MB file with HTTP 413. Parsing the pre- and post-serialization files produced identical JSON values; no experiment was rerun and no measurements or metadata values were changed.

SHA-256 of the compact structured result: `3abcf58ab4fa4cf9e468665f623789d18b367e1d3a1c654ab4a9c6941a9af5ff`.

Raw per-process samples, timestamps, queue intervals, host components, environment snapshots, event lifecycle status, and aggregate quartiles are in `experiments/results/opencl-rx550-p3e-device-profiling.json`. The published P3/P3b/P3c artifacts and fingerprints were left intact.

## Conclusion and limitations

Rusticl on this RX 550 provides usable native OpenCL event timestamps, so P3e directly measured the duration between device `START` and `END` for the tested kernels. These are OpenCL command execution intervals, not hardware instruction-cycle counters or a measure of active ALU occupancy; device scheduling or preemption may still affect elapsed command time. The measurements show microsecond-scale device intervals for small operations and millisecond-scale execution for the two large, compute-heavy cases. Host-observed latency remains materially larger for small operations and includes queueing, Python/API work, and synchronization.

These are instrumented, driver-specific observations from one GPU with some un-attributed utilization/VRAM variation. Event timestamps are valid and numerically checked here, but the unpatched-libclc warning, high long-reduction dispersion, one physical device, and lack of exclusive-GPU guarantees limit generalization. No speedup, scheduler threshold, production optimization, or conclusion about other AMD GPUs is claimed. A later repeat under controlled GPU occupancy would strengthen the `26×3932` forward estimate.
