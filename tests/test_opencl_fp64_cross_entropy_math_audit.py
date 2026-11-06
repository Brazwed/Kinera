"""CPU-only integrity checks for the consolidated F2.3a-P0 audit."""
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
HISTORICAL_RESULTS = ROOT / "experiments/results/opencl-rx550-f2-3a-math-audit.json"
RESULTS = ROOT / "experiments/results/opencl-rx550-f2-3a-math-audit-consolidated.json"
HISTORICAL_RESULTS_SHA256 = "82940bdeb55a4c4e221bcba1a1ad76de43ed102808363254e3c9216d15245f80"
CONSOLIDATED_RESULTS_SHA256 = "1cffbd4cd625b41d5c87d8704316374f16ca4e76b27c969d034850d9d394a614"
HISTORICAL_SCRIPT_SHA256 = "c928921c35e692e811ecde5d559ddc80a443f544d712a1a65b04edf4ea54c2a0"
SPEC = importlib.util.spec_from_file_location("f23a_math_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def _reject_json_constant(value: str) -> None:
    raise AssertionError(f"non-standard JSON constant: {value}")


def _value_from_bits(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _comparison_results(value):
    if isinstance(value, dict):
        if {"expected_bits", "actual_bits", "status"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from _comparison_results(child)
    elif isinstance(value, list):
        for child in value:
            yield from _comparison_results(child)


class CrossEntropyMathAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = RESULTS.read_text()
        cls.data = json.loads(raw, parse_constant=_reject_json_constant)

    def test_bit_ulp_and_tolerance_helpers(self):
        one_up = math.nextafter(1.0, math.inf)
        self.assertEqual(DIAGNOSTIC.bits(1.0), "3ff0000000000000")
        self.assertEqual(abs(DIAGNOSTIC.ordered(one_up) - DIAGNOSTIC.ordered(1.0)), 1)
        within_atol = DIAGNOSTIC.compare(0.0, 1.0e-13)
        self.assertEqual(within_atol["status"], "PASS")
        self.assertIsNone(within_atol["relative_error"])
        self.assertEqual(within_atol["tolerance_limit"], 1.0e-12)
        self.assertEqual(DIAGNOSTIC.compare(math.inf, math.inf)["status"], "SPECIAL_MATCH")
        self.assertEqual(DIAGNOSTIC.compare(math.nan, math.nan)["status"], "SPECIAL_MATCH")

    def test_decimal_reference_and_python_315_sum_probe(self):
        self.assertEqual(DIAGNOSTIC.decimal_exp(-math.inf), 0.0)
        self.assertEqual(DIAGNOSTIC.decimal_log(1.0), 0.0)
        self.assertEqual(DIAGNOSTIC.decimal_cross_entropy([0.0, 0.0], 0), math.log(2.0))
        self.assertEqual(sum((1.0e16, 1.0, -1.0e16)), 1.0)

    def test_historical_pr84_collection_is_byte_preserved(self):
        self.assertEqual(hashlib.sha256(HISTORICAL_RESULTS.read_bytes()).hexdigest(),
                         HISTORICAL_RESULTS_SHA256)
        legacy = json.loads(HISTORICAL_RESULTS.read_text(), parse_constant=_reject_json_constant)
        self.assertEqual(legacy["baseline"], "7bff9617a340faab342f5c844464a94d0acec6d0")
        self.assertEqual(legacy["implementation"]["diagnostic_sha256"], HISTORICAL_SCRIPT_SHA256)

    def test_consolidated_provenance_fixture_origins_and_process_counts(self):
        data = self.data
        self.assertEqual(hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
                         CONSOLIDATED_RESULTS_SHA256)
        self.assertEqual(data["baseline"], "7bff9617a340faab342f5c844464a94d0acec6d0")
        self.assertEqual(data["implementation"]["diagnostic_sha256"],
                         hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(data["implementation"]["opencl_source_sha256"],
                         hashlib.sha256(DIAGNOSTIC.SOURCE.encode("utf-8")).hexdigest())
        self.assertEqual(data["implementation"]["consolidated_from"]["pr83_commit"],
                         "f6fa46438c6b6ba68101c1cffca79fb9516d3d7b")
        self.assertEqual(data["implementation"]["consolidated_from"]["pr84_commit"],
                         "66f0c20457bf1f5d9a71bac7fa3016b535ff80dd")
        self.assertEqual(data["implementation"]["consolidated_from"]["pr83_results_sha256"],
                         "68a5478aedd08bfc45089ada1cead367c78ccc55313ffbe15e5d9f6c385cf014")
        self.assertEqual(data["independent_process_repetitions"], DIAGNOSTIC.TRIALS)
        self.assertEqual(len(data["trial_verification"]), len(DIAGNOSTIC.OPTIONS) * DIAGNOSTIC.TRIALS)

        canonical = data["canonical_results"]
        self.assertEqual(canonical["options"], "<default>")
        self.assertEqual(canonical["trial"], 1)
        expected_cases = {
            "exp": len(DIAGNOSTIC.exp_cases()), "log": len(DIAGNOSTIC.log_cases()),
            "sum": len(DIAGNOSTIC.sum_cases()), "cross_entropy": len(DIAGNOSTIC.ce_cases()),
        }
        origins = {group: {row["fixture_origin"] for row in rows}
                   for group, rows in canonical.items() if group in expected_cases}
        self.assertEqual(origins["exp"], {"PR #83 exclusive", "PR #84 base"})
        self.assertEqual(origins["log"], {"PR #83 exclusive", "PR #83-derived CE denominator", "PR #84 base"})
        self.assertEqual(origins["sum"], {"PR #83 exclusive", "PR #84 base"})
        self.assertEqual(origins["cross_entropy"], {"PR #83 exclusive", "PR #84 base"})
        for row in data["trial_verification"]:
            self.assertEqual(row["cases"], expected_cases)
            self.assertTrue(row["all_argument_echo_bits_match"])
        for option in DIAGNOSTIC.OPTIONS:
            option_name = option or "<default>"
            checks = [row for row in data["trial_verification"] if row["options"] == option_name]
            self.assertEqual(len(checks), DIAGNOSTIC.TRIALS)

        self.assertIn("log_257_from_pr83", {row["case"] for row in canonical["log"]})
        self.assertIn("pr83_equal_logits_257", {row["case"] for row in canonical["cross_entropy"]})
        self.assertIn("pr83_cancel_1e16", {row["case"] for row in canonical["sum"]})

    def test_outputs_comparisons_and_summary_accounting_are_consistent(self):
        data = self.data
        canonical = data["canonical_results"]
        output_by_option = {}
        for check in data["trial_verification"]:
            if check["process_trial"] == 1:
                output_by_option[check["options"]] = check["output_sha256"]
        for group in ("exp", "log", "sum", "cross_entropy"):
            expected_input = DIAGNOSTIC._input_digest(canonical, group)
            for check in data["trial_verification"]:
                self.assertEqual(check["input_sha256"][group], expected_input)
                self.assertEqual(check["output_sha256"][group], output_by_option[check["options"]][group])

        for row in canonical["sum"]:
            self.assertTrue(row["input_echo_bits_match"], row["case"])
            for field in ("forward_vs_cpu", "reverse_vs_cpu", "compensated_forward_vs_cpu",
                          "compensated_reverse_vs_cpu", "forward_vs_fsum",
                          "compensated_forward_vs_fsum", "forward_vs_decimal",
                          "reverse_vs_decimal", "compensated_forward_vs_decimal",
                          "compensated_reverse_vs_decimal"):
                result = row[field]
                expected = _value_from_bits(result["expected_bits"])
                actual = _value_from_bits(result["actual_bits"])
                self.assertEqual(DIAGNOSTIC.compare(expected, actual), result)

        for row in canonical["cross_entropy"]:
            self.assertTrue(row["input_echo_matches"], row["case"])
            self.assertTrue(row["scalar_argument_echo_matches"], row["case"])
            self.assertTrue(row["target_echo_matches"], row["case"])
            for result in _comparison_results(row["stage_comparisons"]):
                expected = _value_from_bits(result["expected_bits"])
                actual = _value_from_bits(result["actual_bits"])
                self.assertEqual(DIAGNOSTIC.compare(expected, actual), result)

        for option, summary in data["summary_by_compilation"].items():
            for key, field, group in (
                ("ce_vs_cpu.cpu", "loss_vs_cpu_backend", "cross_entropy"),
                ("ce_compensated_vs_cpu.cpu", "compensated_loss_vs_cpu_backend", "cross_entropy"),
                ("ce_vs_decimal.decimal", "loss_vs_decimal_high_precision", "cross_entropy"),
                ("ce_compensated_vs_decimal.decimal", "compensated_loss_vs_decimal_high_precision", "cross_entropy"),
            ):
                statuses = [row["stage_comparisons"][field]["status"] for row in canonical[group]] * DIAGNOSTIC.TRIALS
                counts = summary[key]
                self.assertEqual(counts["pass"], statuses.count("PASS"), (option, key))
                self.assertEqual(counts["fail"], statuses.count("FAIL"), (option, key))
                self.assertEqual(counts["special_match"], statuses.count("SPECIAL_MATCH"), (option, key))

        for option, groups in data["repeatability"].items():
            if option != "across_compilation_options":
                self.assertTrue(all(item["bitwise_stable_across_trials"] for item in groups.values()))


if __name__ == "__main__":
    unittest.main()
