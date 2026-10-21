#!/usr/bin/env python3
"""Run Phase 42 Expanded Controlled Replay Pilot.

Executes controlled multi-seed A/B/C replay experiments across low, medium,
and high interference levels and writes:
  experiments/phase42_pilot_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kinera.experiments.expanded_pilot import run_expanded_pilot

# Baseline commit approved on main (merge commit of PR #41)
APPROVED_BASELINE_COMMIT = "5dba2ccd6fadd3c2406223e3ec28657cc1fb9138"


def main() -> None:
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output

    if args.runner_commit:
        runner_commit = args.runner_commit
    else:
        try:
            runner_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
            ).strip()
        except Exception:
            runner_commit = "unknown"

    print("=" * 60)
    print("Kinera Phase 42: Expanded Controlled Replay Pilot")
    print(f"Seeds: {args.seeds}")
    print(f"Conditions: {args.conditions}")
    print(f"Baseline Commit: {args.baseline_commit[:10]}")
    print(f"Runner Commit:   {runner_commit[:10]}")
    print("=" * 60)

    report = run_expanded_pilot(
        seeds=tuple(args.seeds),
        conditions=tuple(args.conditions),  # type: ignore[arg-type]
        baseline_commit=args.baseline_commit,
        runner_commit=runner_commit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cost = report["cost_profile"]
    concl = report["conclusions"]

    print("\n" + "=" * 60)
    print("Execution Summary:")
    print(f"Total CPU Time:   {cost['total_cpu_time_seconds']:.2f} s")
    print(f"Peak Heap (TM):   {cost['peak_ram_mb']:.2f} MB ({cost['peak_ram_bytes']} bytes)")
    print(f"Peak RSS (OS):    {cost.get('peak_rss_mb', 0.0):.2f} MB ({cost.get('peak_rss_bytes', 0)} bytes)")
    print(f"Scaling Forecast: {cost['projected_scaled_hours_cpu']:.1f} hours CPU")
    print(f"Recommendation:   {cost['scaling_recommendation'].upper()}")
    print(f"Protocol Valid:   {concl['protocol_validated']}")
    print(f"Budget Matched:   {concl['fairness_confirmed_all_seeds']}")
    print(f"High Rec Adv:     {concl['high_interference_mean_recovery_advantage']:+.4f}")
    print(f"High Sup Frac:    {concl['high_interference_superiority_fraction'] * 100:.1f}%")
    print("-" * 60)
    print(f"Claim: {concl['summary_claim']}")
    print("=" * 60)
    print(f"\nReport successfully written to {output_path}")


if __name__ == "__main__":
    main()
