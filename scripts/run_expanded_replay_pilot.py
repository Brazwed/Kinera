#!/usr/bin/env python3
"""Run Phase 42 Expanded Controlled Replay Pilot.

Executes controlled multi-seed A/B/C replay experiments across low, medium,
and high interference levels with incremental checkpointing, optional resume,
explicit CPU/CUDA backend selection, and optional heap profiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from kinera.experiments.expanded_pilot import (
    run_cpu_vs_cuda_microbenchmark,
    run_expanded_pilot,
    verify_scientific_equivalence,
    write_pilot_checkpoint,
)

# Baseline commit approved on main (merge commit of PR #41)
APPROVED_BASELINE_COMMIT = "5dba2ccd6fadd3c2406223e3ec28657cc1fb9138"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 42 Expanded Controlled Replay Pilot"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds to evaluate across variants",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["low", "medium", "high"],
        default=["low", "medium", "high"],
        help="Interference conditions to evaluate",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution backend ('cpu' default or 'cuda')",
    )
    parser.add_argument(
        "--profile-heap",
        action="store_true",
        default=False,
        help="Enable tracemalloc Python heap profiling (disabled by default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume execution from an existing incremental checkpoint file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit path for incremental checkpointing (defaults to --output)",
    )
    parser.add_argument(
        "--runner-commit",
        default=None,
        help="Explicit runner commit SHA (default: git rev-parse HEAD)",
    )
    parser.add_argument(
        "--baseline-commit",
        default=APPROVED_BASELINE_COMMIT,
        help="Baseline commit SHA",
    )
    parser.add_argument(
        "--output",
        default="experiments/phase42_pilot_results.json",
        help="Output path for the experimental JSON report",
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        help="Optional reference JSON report path to verify scientific equivalence per seed and aggregate",
    )
    parser.add_argument(
        "--benchmark-backends",
        action="store_true",
        default=False,
        help="Run CPU vs. CUDA microbenchmark instead of the full pilot report",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        default=False,
        help="Fail closed if CUDA is unavailable during --benchmark-backends",
    )
    return parser


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    output_path = _resolve_path(project_root, args.output)
    checkpoint_path = (
        _resolve_path(project_root, args.checkpoint)
        if args.checkpoint is not None
        else output_path
    )

    if args.runner_commit:
        runner_commit = args.runner_commit
    else:
        try:
            runner_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
            ).strip()
        except Exception:
            runner_commit = "unknown"

    if args.benchmark_backends:
        print("=" * 60)
        print("Kinera Phase 42: CPU vs. CUDA Microbenchmark")
        print(f"Seeds: {args.seeds}")
        print(f"Conditions: {args.conditions}")
        print(f"Require CUDA: {args.require_cuda}")
        print(f"Runner Commit: {runner_commit[:10]}")
        print("=" * 60)
        bench = run_cpu_vs_cuda_microbenchmark(
            seeds=tuple(args.seeds),
            conditions=tuple(args.conditions),  # type: ignore[arg-type]
            baseline_commit=args.baseline_commit,
            runner_commit=runner_commit,
            require_cuda=args.require_cuda,
        )
        write_pilot_checkpoint(output_path, bench)
        cpu_b = bench["cpu"]
        cuda_b = bench["cuda"]
        comp_b = bench["comparison"]
        print(
            f"CPU  | init={cpu_b['backend_init_seconds']:.4f}s | "
            f"train={cpu_b['training_wall_time_seconds']:.4f}s | "
            f"{cpu_b['ms_per_gradient_step']:.3f} ms/step"
        )
        if cuda_b.get("available"):
            print(
                f"CUDA | init={cuda_b['backend_init_seconds']:.4f}s | "
                f"train={cuda_b['training_wall_time_seconds']:.4f}s | "
                f"{cuda_b['ms_per_gradient_step']:.3f} ms/step | "
                f"live_allocs_after_close={cuda_b['live_allocations_after_close']}"
            )
            print(
                f"Speedup (train only): {comp_b['training_speedup_cuda_vs_cpu']:.2f}x | "
                f"Speedup (with init): {comp_b['total_speedup_including_init']:.2f}x | "
                f"Max metric error: {comp_b['max_absolute_error']:.3e}"
            )
        else:
            print(f"CUDA | unavailable ({cuda_b.get('reason')})")
        print(f"\nBenchmark report written to {output_path}")
        return 0

    print("=" * 60)
    print("Kinera Phase 42: Expanded Controlled Replay Pilot")
    print(f"Seeds:           {args.seeds}")
    print(f"Conditions:      {args.conditions}")
    print(f"Backend:         {args.backend}")
    print(f"Profile Heap:    {args.profile_heap}")
    print(f"Resume:          {args.resume}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Baseline Commit: {args.baseline_commit[:10]}")
    print(f"Runner Commit:   {runner_commit[:10]}")
    print("=" * 60)

    report = run_expanded_pilot(
        seeds=tuple(args.seeds),
        conditions=tuple(args.conditions),  # type: ignore[arg-type]
        baseline_commit=args.baseline_commit,
        runner_commit=runner_commit,
        backend=args.backend,
        profile_heap=args.profile_heap,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    if output_path != checkpoint_path:
        write_pilot_checkpoint(output_path, report)

    cost = report["cost_profile"]
    concl = report["conclusions"]
    meta = report["execution_metadata"]

    print("\n" + "=" * 60)
    print("Execution Summary:")
    print(f"Backend:          {meta['backend']} (init: {meta['backend_init_seconds']:.4f} s)")
    print(f"Resumed Run:      {meta['resumed']} (resume_count={meta['resume_count']})")
    print(f"Completed Units:  {len(meta['completed_units'])}")
    print(f"Total Train Time: {cost['total_cpu_time_seconds']:.2f} s")
    print(
        f"Peak Heap (TM):   {cost['peak_ram_mb']:.2f} MB ({cost['peak_ram_bytes']} bytes, "
        f"enabled={cost['profile_heap_enabled']})"
    )
    print(f"Peak RSS (OS):    {cost.get('peak_rss_mb', 0.0):.2f} MB ({cost.get('peak_rss_bytes', 0)} bytes)")
    print(f"Scaling Forecast: {cost['projected_scaled_hours_cpu']:.2f} hours CPU")
    print(f"Recommendation:   {cost['scaling_recommendation'].upper()}")
    print(f"Protocol Valid:   {concl['protocol_validated']}")
    print(f"Budget Matched:   {concl['fairness_confirmed_all_seeds']}")
    print(f"High Rec Adv:     {concl['high_interference_mean_recovery_advantage']:+.4f}")
    print(f"High Ret Adv:     {concl['high_interference_mean_retention_advantage']:+.4f}")
    print(
        f"Zero-Ref Acq B/C: {concl['high_interference_zero_ref_acquisition_b']:+.4f} / "
        f"{concl['high_interference_zero_ref_acquisition_c']:+.4f}"
    )
    print(f"High Sup Frac:    {concl['high_interference_superiority_fraction'] * 100:.1f}%")
    print("-" * 60)
    print(f"Claim: {concl['summary_claim']}")
    print("=" * 60)

    if args.verify_against:
        ref_path = _resolve_path(project_root, args.verify_against)
        ref_report = json.loads(ref_path.read_text(encoding="utf-8"))
        eq = verify_scientific_equivalence(ref_report, report)
        print(
            f"Scientific Equivalence Verified against {ref_path}: "
            f"checked_units={eq['checked_seed_units']}, max_abs_error={eq['max_absolute_error']:.3e}"
        )

    print(f"\nReport successfully written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
