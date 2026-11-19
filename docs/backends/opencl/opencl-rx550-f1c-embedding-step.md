# OpenCL F1c: `Embedding.step` on the RX 550

## Scope and provenance

This report records the F1c gate for the OpenCL `Embedding.step` operation. Work
started from the requested `main` baseline
`685fa68ea11b1cd500c489b64f7df900920b7f44` (merge of PR #65). After `main`
advanced with the unrelated Issue #26 merge, the F1c branch was rebased onto
`ed31cf90df257a18bfd02536a99470c46717d025` before publication. The work was
performed on the isolated branch
`feature/issue-61-f1c-opencl-embedding-step`.

F1c adds only the individual embedding parameter update. It does not implement
F1d, model training, benchmark workloads, scheduler behavior, or runner
selection. The CPU and CUDA implementations and the historical P3/P3b/P3c data
were not changed.

## Implementation

`OpenCLBackend.embedding_step()` validates both resident buffers before any
kernel submission: buffer type, open state, float64 dtype, rank two, identical
shape, row-major layout, exact byte count, selected device, manager/context
ownership, and non-aliasing. Dimensions are checked against the kernel's signed
32-bit arguments and the host `size_t` dispatch range. The learning rate must
be an `int` or `float`, but not `bool`; it must convert to a finite,
non-negative `float`.

The method returns the original weights buffer. It does not allocate another
device buffer or transfer either tensor to the host. A zero learning rate
returns the same buffer after validation without building or enqueueing a
kernel.

The new `embedding_step.cl` assigns one work item per table element. It computes
`update = learning_rate * gradient` and then subtracts `update` from the weight.
`#pragma OPENCL FP_CONTRACT OFF` prevents contraction into an FMA, preserving
the CPU operation's separate multiplication and subtraction rounding. The
kernel is added to the existing `kernels/*.cl` package-data rule.

Failures before enqueue leave the weights unchanged in the fake API tests.
Created kernel objects are released once; if cleanup also fails, the original
operation error remains primary. An error reported after command submission
does not have a demonstrated rollback guarantee: the in-place update may have
completed even when the caller receives an exception. A kernel release error
after successful enqueue is reported, and the kernel release is not retried.
At the layer API, an exception leaves the backward-ready flag set; callers
should treat the parameter state as indeterminate and avoid retrying the same
update blindly.

## RX 550 environment

| Item | Observed value |
| --- | --- |
| OS | Fedora Linux 45 Workstation |
| Python | 3.15.0 |
| Device | AMD Radeon RX 550 / 550 Series (`radeonsi`, `polaris12`, ACO) |
| PCI address | `0000:03:00.0` |
| Driver | `26.2.6` |
| OpenCL runtime | OpenCL 3.1 |
| OpenCL C | OpenCL C 1.2 |
| FP64 | Supported and required by the gate |
| Rusticl warning | `Patched Mesa libclc not detected` |

The warning was emitted during the run and is retained as a limitation. The
gate used `RUSTICL_ENABLE=radeonsi RUSTICL_FEATURES=fp64` and
`KINERA_RUN_OPENCL_TESTS=1`.

## Numerical cases and results

The RX 550 opt-in gate exercised:

- A signed `2×3` table with positive, negative, and zero gradients, including
  values near `1e308`.
- A cancellation case selected to distinguish separate multiply/subtract
  rounding from fused multiply-add.
- Three consecutive updates with rates `0.1`, `0.025`, and `0.1`.
- A zero-rate update, verifying no enqueue and no parameter change.
- Invalid rates: boolean, string, negative, NaN, positive/negative infinity,
  and an integer too large to convert to float.
- The `Embedding.step` layer path, checking that the parameter buffer stays
  resident and identical and that the layer clears its backward-ready flag.

Every GPU result was explicitly copied to the host after the operation for
comparison with `CPUBackend.embedding_step()`. No comparison read or write was
performed inside the operation. The comparison used `atol=rtol=1e-12` across
all parameters.

| Gate | Result |
| --- | --- |
| F1c RX 550 tests | 4 passed |
| Maximum absolute CPU/OpenCL error | `0.0` |
| GPU parameter updates | Same resident weights buffer returned |
| Implicit H2D/D2H during update | None observed |
| Zero-rate kernel enqueue | None |

The FMA-sensitive value produced the CPU result `0.0`; the fused expression
would produce `2**-54`. This checks that the kernel's contraction setting is
effective for the tested device/runtime.

## Failure, ownership, and lifecycle checks

CPU-only fake OpenCL API tests cover invalid shape/rank/dtype/layout, closed and
foreign buffers, manager/device mismatch, buffer-size mismatch, aliasing,
invalid learning rates, and an FP64 capability query failure. They also inject
program build, kernel creation, argument binding, enqueue, and kernel release
failures.

The failure checks confirm that pre-enqueue validation/setup errors do not
mutate the weights, any created kernel is released once, and an enqueue error
remains the primary exception if release also fails. A release failure after a
successful enqueue is surfaced without a second release attempt; the fake
records that the in-place update may already have happened. Successful calls
leave the caller-owned gradient buffer open and return the same weights object.
The layer-level test confirms the existing `Embedding.step` state transition.

## Verification

The local validation set includes:

- CPU-only fake API F1c tests: 10 passed.
- Opt-in RX 550/Rusticl F1c tests: 4 passed, maximum absolute error `0.0`.
- Existing F1a/F1b Embedding gates and Linear OpenCL gates on the RX 550: 26
  passed, including 439,937 scalar subtests, in 87.72 seconds. This run emitted
  the Rusticl libclc warning above.
- Repository CI test selection (locally on Python 3.15): 734 passed, 52
  skipped, and the workflow's 5 explicitly deselected historical
  pilot/benchmark tests; 534 subtests passed.
  The skipped cases include hardware-gated checks. The only warning was an
  upstream pytest/AnyIO deprecation notice.
- `py_compile`, wheel content inspection, P3c JSON/source fingerprint integrity,
  and `git diff --check` passed.

The wheel is expected to contain
`kinera/backends/opencl/kernels/embedding_step.cl` via the existing kernel
package-data glob. Historical P3c JSON and report fingerprints remain
unchanged. The P3c integrity test's pin for the *current* OpenCL backend source
was advanced for F1c; the P3c, F1a, and F1b source fingerprints remain recorded
as historical constants.

The implementation source SHA-256 fingerprints are:

| File | SHA-256 |
| --- | --- |
| `src/kinera/backends/opencl/backend.py` | `33ec898ad316e0083713f35bd31a3fdbe31eb7fe0aeb4b6d387a0ea55cf98525` |
| `src/kinera/backends/opencl/kernels/embedding_step.cl` | `22367b259cab6793eeceefe1f71dee8f66c31a3a226a509085c78f0a6f513808` |

The published commit and GitHub Actions status are reported with the PR metadata.

## Limitations and status

The result establishes numerical and lifecycle behavior for this individual
operation on this RX 550/Rusticl FP64 setup. It is not a performance result and
does not establish support for other GPUs, other OpenCL implementations,
Embedding backward/forward integration into training, or complete model
training. The Rusticl libclc warning remains applicable. F1d is outside this
gate and remains pending review.
