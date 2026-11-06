"""CPU-only integrity tests for the experimental F2.3a-P0 diagnostic."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/investigate_opencl_fp64_cross_entropy_math.py"
RESULTS = ROOT / "experiments/results/opencl-rx550-f2-3a-math-audit.json"
SPEC = importlib.util.spec_from_file_location("f23a_math_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def _reject_json_constant(value: str) -> None:
    raise AssertionError(f"non-standard JSON constant: {value}")


class CrossEntropyMathAuditTests(unittest.TestCase):
    def test_bit_and_ulp_helpers_cover_both_signs(self) -> None:
        one_up = math.nextafter(1.0, math.inf)
        minus_one_down = math.nextafter(-1.0, -math.inf)
        self.assertEqual(DIAGNOSTIC.bits(1.0), "3ff0000000000000")
        self.assertEqual(abs(DIAGNOSTIC.ordered(one_up) - DIAGNOSTIC.ordered(1.0)), 1)
        self.assertEqual(
            abs(DIAGNOSTIC.ordered(minus_one_down) - DIAGNOSTIC.ordered(-1.0)), 1
        )

    def test_tolerance_and_zero_relative_error_are_explicit(self) -> None:
        result = DIAGNOSTIC.compare(0.0, 1.0e-13)
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["relative_error"])
        self.assertEqual(result["tolerance_limit"], DIAGNOSTIC.ATOL)
        failed = DIAGNOSTIC.compare(1.0, 1.0 + 2.0e-12)
        self.assertEqual(failed["status"], "FAIL")

    def test_decimal_composition_avoids_copying_cpu_cancellation(self) -> None:
        self.assertEqual(DIAGNOSTIC.decimal_cross_entropy([1.0, 1.0], 0), math.log(2.0))
        expected = DIAGNOSTIC.decimal_cross_entropy([1.0e300, 1.0e300], 0)
        self.assertEqual(expected, math.log(2.0))

    def test_non_finite_comparison_remains_json_serializable(self) -> None:
        self.assertEqual(DIAGNOSTIC.compare(math.inf, math.inf)["status"], "SPECIAL_MATCH")
        self.assertEqual(DIAGNOSTIC.compare(math.nan, math.nan)["status"], "SPECIAL_MATCH")
        json.dumps(DIAGNOSTIC.compare(math.inf, math.inf), allow_nan=False)

    def test_published_artifact_provenance_accounting_and_echo_integrity(self) -> None:
        raw = RESULTS.read_text()
        data = json.loads(raw, parse_constant=_reject_json_constant)
        self.assertEqual(data["baseline"], "7bff9617a340faab342f5c844464a94d0acec6d0")
        self.assertEqual(data["independent_process_repetitions"], DIAGNOSTIC.TRIALS)
        self.assertEqual(data["implementation"]["diagnostic_sha256"],
                         hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(data["implementation"]["opencl_source_sha256"],
                         hashlib.sha256(DIAGNOSTIC.SOURCE.encode()).hexdigest())
        self.assertEqual(len(data["trial_verification"]),
                         len(DIAGNOSTIC.OPTIONS) * DIAGNOSTIC.TRIALS)
        canonical = data["canonical_results"]
        self.assertEqual(canonical["options"], "<default>")
        self.assertEqual(canonical["trial"], 1)
        self.assertTrue(all(check["all_argument_echo_bits_match"]
                            for check in data["trial_verification"]))
        self.assertTrue(all(all(check["matches_canonical_output_bits"].values())
                            for check in data["trial_verification"]))
        groups = ("exp", "log", "sum", "cross_entropy")
        expected_input_hashes = {group: DIAGNOSTIC._input_digest(canonical, group)
                                 for group in groups}
        expected_output_hashes = {group: DIAGNOSTIC._output_digest(canonical, group)
                                  for group in groups}
        for check in data["trial_verification"]:
            self.assertEqual(check["input_sha256"], expected_input_hashes)
            self.assertEqual(check["output_sha256"], expected_output_hashes)

        for row in canonical["exp"]:
            self.assertTrue(row["echo_bits_match"])
        for row in canonical["log"]:
            self.assertTrue(row["echo_bits_match"])
        for row in canonical["sum"]:
            self.assertTrue(row["input_echo_bits_match"])
        for row in canonical["cross_entropy"]:
            self.assertTrue(row["input_echo_matches"], row["case"])
            self.assertTrue(row["scalar_argument_echo_matches"], row["case"])
            self.assertTrue(row["target_echo_matches"], row["case"])
            self.assertEqual(len(row["opencl"]["exp_bits"]), len(row["logits"]))

        for options_summary in data["summary_by_compilation"].values():
            expected_counts = {
                "exp_vs_cpu.cpu": len(DIAGNOSTIC.exp_cases()) * DIAGNOSTIC.TRIALS,
                "exp_vs_decimal.decimal": len(DIAGNOSTIC.exp_cases()) * DIAGNOSTIC.TRIALS,
                "log_vs_cpu.cpu": len(DIAGNOSTIC.log_cases()) * DIAGNOSTIC.TRIALS,
                "log_vs_decimal.decimal": len(DIAGNOSTIC.log_cases()) * DIAGNOSTIC.TRIALS,
                "sum_forward_vs_cpu.cpu": len(DIAGNOSTIC.sum_cases()) * DIAGNOSTIC.TRIALS,
                "sum_reverse_vs_cpu.cpu": len(DIAGNOSTIC.sum_cases()) * DIAGNOSTIC.TRIALS,
                "ce_vs_cpu.cpu": len(DIAGNOSTIC.ce_cases()) * DIAGNOSTIC.TRIALS,
                "ce_vs_decimal.decimal": len(DIAGNOSTIC.ce_cases()) * DIAGNOSTIC.TRIALS,
            }
            self.assertEqual(set(options_summary), set(expected_counts))
            for category, expected_count in expected_counts.items():
                counts = options_summary[category]
                self.assertEqual(counts["pass"] + counts["fail"] + counts["special_match"], expected_count)

        for option, groups in data["repeatability"].items():
            if option == "across_compilation_options":
                self.assertTrue(all(groups.values()))
            else:
                self.assertTrue(all(value["bitwise_stable_across_trials"]
                                    for value in groups.values()))
                self.assertEqual(all(value["fresh_context_and_program_builds"] == DIAGNOSTIC.TRIALS
                                     for value in groups.values()), True)

        selectors = {
            "exp_vs_cpu.cpu": ("exp", lambda row: row["vs_cpu"]),
            "exp_vs_decimal.decimal": ("exp", lambda row: row["vs_decimal_rounded_to_fp64"]),
            "log_vs_cpu.cpu": ("log", lambda row: row["vs_cpu"]),
            "log_vs_decimal.decimal": ("log", lambda row: row["vs_decimal_rounded_to_fp64"]),
            "sum_forward_vs_cpu.cpu": ("sum", lambda row: row["forward_vs_cpu"]),
            "sum_reverse_vs_cpu.cpu": ("sum", lambda row: row["reverse_vs_cpu"]),
            "ce_vs_cpu.cpu": ("cross_entropy", lambda row: row["stage_comparisons"]["loss_vs_cpu_backend"]),
            "ce_vs_decimal.decimal": ("cross_entropy", lambda row: row["stage_comparisons"]["loss_vs_decimal_high_precision"]),
        }
        for option, options_summary in data["summary_by_compilation"].items():
            for key, (group, selector) in selectors.items():
                observed_statuses = [selector(row)["status"] for row in canonical[group]] * DIAGNOSTIC.TRIALS
                self.assertEqual(options_summary[key]["pass"], observed_statuses.count("PASS"), key)
                self.assertEqual(options_summary[key]["fail"], observed_statuses.count("FAIL"), key)
                self.assertEqual(options_summary[key]["special_match"],
                                 observed_statuses.count("SPECIAL_MATCH"), key)

        for option, stages in data["cross_entropy_stage_summary_by_compilation"].items():
            self.assertEqual(stages["loss_vs_cpu_backend"]["pass"],
                             data["summary_by_compilation"][option]["ce_vs_cpu.cpu"]["pass"])
            self.assertEqual(stages["loss_vs_decimal_high_precision"]["fail"],
                             data["summary_by_compilation"][option]["ce_vs_decimal.decimal"]["fail"])

        # Independently recompute the contract outcomes for every CE trial.
        for row in canonical["cross_entropy"]:
            observed = DIAGNOSTIC.compare(
                float(row["cpu"]["loss"]), float(row["opencl"]["loss"])
            )
            self.assertEqual(observed, row["stage_comparisons"]["loss_vs_cpu_backend"])


if __name__ == "__main__":
    unittest.main()
