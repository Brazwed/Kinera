"""Pure-Python integrity checks for the experimental P3b cost profiler."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import analyze_opencl_linear_costs as analysis  # noqa: E402


class P3CostAccountingTests(unittest.TestCase):
    def tearDown(self) -> None:
        analysis._set_profile(None)

    def test_nested_event_self_time_is_subtracted_once(self) -> None:
        profile = {"events": {}}
        analysis._set_profile(profile, "forward")
        with analysis._event_timer("opencl.linear_forward"):
            with analysis._event_timer("opencl.allocate"):
                with analysis._event_timer("opencl.device_check"):
                    time.sleep(0.001)

        analysis._validate_profile_events(profile)
        events = profile["events"]["forward"]
        allocate = events["opencl.allocate"][0]
        device_check = events["opencl.device_check"][0]
        self.assertEqual(
            allocate["children_inclusive_ns"] + allocate["exclusive_ns"],
            allocate["inclusive_ns"],
        )
        self.assertGreater(allocate["children_inclusive_ns"], 0)
        self.assertGreater(device_check["exclusive_ns"], 0)

    def test_opencl_components_and_residual_form_disjoint_partition(self) -> None:
        events = {
            "opencl.allocate": [
                {
                    "inclusive_ns": 300,
                    "children_inclusive_ns": 50,
                    "exclusive_ns": 250,
                }
            ],
            "opencl.device_check": [
                {
                    "inclusive_ns": 50,
                    "children_inclusive_ns": 0,
                    "exclusive_ns": 50,
                }
            ],
            "opencl.require_buffer": [
                {
                    "inclusive_ns": 20,
                    "children_inclusive_ns": 0,
                    "exclusive_ns": 20,
                }
            ],
        }
        components, residual = analysis._opencl_components(events, backend_ns=1000)
        self.assertEqual(components["allocation"], 250)
        self.assertEqual(components["device_check"], 50)
        self.assertEqual(components["buffer_validation"], 20)
        self.assertEqual(sum(components.values()) + residual, 1000)
        self.assertEqual(residual, 680)

    def test_opencl_partition_rejects_components_larger_than_backend(self) -> None:
        events = {
            "opencl.enqueue_and_clFinish": [
                {
                    "inclusive_ns": 1001,
                    "children_inclusive_ns": 0,
                    "exclusive_ns": 1001,
                }
            ]
        }
        with self.assertRaises(AssertionError):
            analysis._opencl_components(events, backend_ns=1000)

    def test_statistics_are_recomputed_and_checked(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        stats = analysis._stats_ms(values)
        analysis._validate_stats(stats, values)
        self.assertEqual(stats["samples"], 4)
        self.assertEqual(stats["iqr_ms"], stats["q3_ms"] - stats["q1_ms"])

        corrupted = dict(stats, median_ms=99.0)
        with self.assertRaises(AssertionError):
            analysis._validate_stats(corrupted, values)

    def test_published_artifacts_match_scripts_and_redact_local_state(self) -> None:
        results = ROOT / "experiments" / "results"
        p3_path = results / "opencl-rx550-p3-benchmark.json"
        p3b_path = results / "opencl-rx550-p3b-original-results.json"
        diagnostic_path = results / "opencl-rx550-p3b-accounting-diagnostic.json"
        p3 = json.loads(p3_path.read_text(encoding="utf-8"))
        p3b = json.loads(p3b_path.read_text(encoding="utf-8"))
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

        benchmark_sha = hashlib.sha256(
            (ROOT / "experiments" / "benchmark_opencl_linear.py").read_bytes()
        ).hexdigest()
        analysis_sha = hashlib.sha256(
            (ROOT / "experiments" / "analyze_opencl_linear_costs.py").read_bytes()
        ).hexdigest()
        self.assertEqual(p3["code"]["benchmark_script_sha256"], benchmark_sha)
        self.assertEqual(diagnostic["baseline_script_sha256"], benchmark_sha)
        self.assertEqual(diagnostic["analysis_script_sha256"], analysis_sha)
        self.assertEqual(
            p3["code"]["benchmark_fingerprint_sha256"],
            analysis.baseline._fingerprint(),
        )
        self.assertEqual(
            p3["source_json_sha256"],
            "049c5b57eede9adc052f2fe10b56230c1575bbcbd54e53ec5f6d0395ff35cc21",
        )
        self.assertEqual(
            p3b["source_json_sha256"],
            "21c139cc3cdee636ee0ffb1e4e0e16d43b73939e000fe47a7c13b217e7021ff5",
        )
        self.assertEqual(
            p3b["original_analysis_script_sha256"],
            "df29feff492e4e8fa831b7e8775a5894b2dd5fd3396bb02a0dcaf85b9e4389ab",
        )

        forbidden_keys = {
            "pid",
            "python_executable",
            "stderr_logs",
            "system_before",
            "system_after",
        }
        forbidden_values = ("/tmp/kinera-issue52", "/home/brazwed/", "process_cpu_affinity")
        for artifact in (p3, p3b, diagnostic):
            rendered = json.dumps(artifact, ensure_ascii=False)
            self.assertFalse(any(value in rendered for value in forbidden_values))

            def check_keys(value):
                if isinstance(value, dict):
                    self.assertFalse(forbidden_keys.intersection(value))
                    for child in value.values():
                        check_keys(child)
                elif isinstance(value, list):
                    for child in value:
                        check_keys(child)
            check_keys(artifact)

    def test_corrected_diagnostic_samples_and_statistics_are_consistent(self) -> None:
        diagnostic_path = (
            ROOT
            / "experiments"
            / "results"
            / "opencl-rx550-p3b-accounting-diagnostic.json"
        )
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        checks = diagnostic["automated_accounting_checks"]
        self.assertTrue(checks)
        self.assertTrue(all(status == "PASS" for status in checks.values()))
        expected_components = (
            "buffer_validation",
            "device_check",
            "allocation",
            "program_lookup",
            "memory_release",
            "kernel_create",
            "kernel_argument_setup",
            "enqueue_and_clFinish",
            "kernel_release",
        )

        for case in diagnostic["profile_cost_decomposition"]["cases"]:
            for backend_name in ("cpu", "opencl"):
                for profile_name in ("resident", "end_to_end"):
                    record = case[profile_name][backend_name]
                    summary = record["summary"]
                    samples = record["samples"]
                    self.assertEqual(len(samples), 5)
                    for metric_name, stats in summary.items():
                        self.assertEqual(stats["samples"], len(samples))
                        analysis._validate_stats(
                            stats, [sample[metric_name] for sample in samples]
                        )

            resident_samples = case["resident"]["opencl"]["samples"]
            for sample in resident_samples:
                for operation in ("forward", "backward", "step"):
                    component_total = sum(
                        sample[f"{operation}_opencl_{component}_exclusive_ms"]
                        for component in expected_components
                    )
                    component_total += sample[
                        f"{operation}_opencl_backend_residual_ms"
                    ]
                    self.assertTrue(
                        math.isclose(
                            component_total,
                            sample[f"{operation}_backend_method_ms"],
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    )
                    self.assertEqual(
                        sample[f"{operation}_opencl_accounted_backend_ms"],
                        sample[f"{operation}_backend_method_ms"],
                    )
                cycle_total = sum(
                    sample[key]
                    for key in (
                        "cycle_backend_methods_sum_ms",
                        "cycle_linear_validation_sum_ms",
                        "cycle_buffer_from_host_sum_ms",
                        "cycle_layer_residual_after_backend_validation_and_copy_ms",
                    )
                )
                self.assertTrue(
                    math.isclose(cycle_total, sample["cycle_total_ms"], abs_tol=1e-9)
                )


if __name__ == "__main__":
    unittest.main()
