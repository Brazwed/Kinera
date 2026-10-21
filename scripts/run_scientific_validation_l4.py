#!/usr/bin/env python3
"""Run Phase 34 (Tranche L4) scientific validation protocol.

Executes both synthetic and natural fixtures across variants A, B, and C,
computes empirical continuous learning metrics, and writes:
  experiments/phase34_l4_scientific_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from kinera.experiments.scientific_runner import (
    build_validation_report,
    run_scientific_protocol,
)

# Approved main baseline commit for Phase 34 Tranche L4
L4_APPROVED_BASELINE_COMMIT = "b4c12df395c930ec93cb4aaf2eab55dd49329988"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 34 L4 scientific validation protocol")
    parser.add_argument("--runner-commit", default=None, help="Explicit runner commit SHA")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "experiments" / "phase34_l4_scientific_validation.json"

    # Current execution/code revision
    if args.runner_commit:
        runner_commit = args.runner_commit
    else:
        try:
            runner_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
            ).strip()
        except Exception:
            runner_commit = "unknown"

    synthetic_results = run_scientific_protocol("synthetic", replay_seed=0)
    natural_results = run_scientific_protocol("natural", replay_seed=1)

    report = build_validation_report(
        synthetic_results=synthetic_results,
        natural_results=natural_results,
        baseline_commit=L4_APPROVED_BASELINE_COMMIT,
        runner_commit=runner_commit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"L4 scientific validation report written to {output_path}")


if __name__ == "__main__":
    main()
