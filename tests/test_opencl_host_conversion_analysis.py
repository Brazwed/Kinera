"""stdlib-only integrity tests for the P3c measurement harness and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import analyze_opencl_host_conversions as profiler  # noqa: E402
from kinera.backends.cpu.backend import CPUBackend  # noqa: E402


RESULT_PATH = ROOT / "experiments/results/opencl-rx550-p3c-host-conversions.json"
REPORT_PATH = ROOT / "docs/opencl-rx550-p3c-host-conversion-analysis.md"

# The JSON is the original P3c collection. The harness fingerprint is updated
# after the fail-closed nested-span check was added; the measurements were not
# regenerated for that accounting guard.
EXPECTED_SHA256 = {
    "experiments/analyze_opencl_host_conversions.py":
        "bf11fef67ee39b1f8432bef90bfc4619a990b5fab9d3871ca0d14ae479fe2856",
    "experiments/results/opencl-rx550-p3c-host-conversions.json":
        "d8d0b5932037320a27f585ebd6c74d7cbfd22ab81c67ac42ee8abbeb93e4adc8",
    "src/kinera/backends/opencl/memory.py":
        "19d67e70eb685c82bcb7311c2448e03ff43e9005900eb0453e1e1a3a873ec5d9",
    "src/kinera/backends/opencl/backend.py":
        "cae03486353643ea4ee3d4cab68d085a8c89544359fa29024e280852f6772ff7",
    "src/kinera/backends/cpu/backend.py":
        "990b9422f425e0136a720bad880dbe5ebc045c275ac762751a0357c8ec9c2ffb",
    "src/kinera/nn/layers/linear.py":
        "41da794a9dd3a34d08b3548cef83da90e217c3a94edc3890e895c103e686ea81",
    "docs/opencl-rx550-p3-benchmark.md":
        "e8deffe2a433f1d1e79d4f1d36283e01234ca5759c3c968c95e0fb89fa10705b",
    "docs/opencl-rx550-p3-cost-analysis.md":
        "36b21a2576159344e6426781645f4e8a2a0f7399be71aed53e9c2e3c87dbef24",
    "docs/opencl-rx550-p3b-cost-accounting-correction.md":
        "eb250d9d428aac67ec717fb7d761791ad3423d812725a7831795ead86c364b8c",
    "experiments/benchmark_opencl_linear.py":
        "abffdeb154743a03b4937b4b557e3d1ce05d1c38cc69c7882d6cbdfd6fc04cbc",
    "experiments/analyze_opencl_linear_costs.py":
        "93f7e896f067e20150031926b9b04ca1bc29727a4f63d66b3be45d352c15868b",
    "experiments/results/opencl-rx550-p3-benchmark.json":
        "6fa50759b3146d9455ccf05cb26ea29a683438f3d7df94636dc46d41b6079da0",
    "experiments/results/opencl-rx550-p3b-original-results.json":
        "2057cdec6a561e3dee3c733778e079c2b708143229e6c8f4cdb0d8af587e53fd",
    "experiments/results/opencl-rx550-p3b-accounting-diagnostic.json":
        "c61235b04424dd5f0fe51a91062c1dffda6b644e2d562c1b37056079e7614255",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SpanRecorderTests(unittest.TestCase):
    def test_nested_inclusive_and_exclusive_durations_are_exact(self) -> None:
        recorder = profiler.SpanRecorder()
        clock = iter((0, 2, 5, 10))
        with patch.object(profiler.time, "perf_counter_ns", side_effect=lambda: next(clock)):
            result = recorder.call(
                "parent", lambda: recorder.call("child", lambda: "ok")
            )

        self.assertEqual(result, "ok")
        spans = {span["name"]: span for span in recorder.events}
        self.assertEqual(spans["child"]["inclusive_ns"], 3)
        self.assertEqual(spans["child"]["exclusive_ns"], 3)
        self.assertEqual(spans["parent"]["inclusive_ns"], 10)
        self.assertEqual(spans["parent"]["exclusive_ns"], 7)
        self.assertEqual(recorder.stack, [])

    def test_child_time_greater_than_parent_is_rejected(self) -> None:
        recorder = profiler.SpanRecorder()
        # Faulty timing source simulates impossible nested accounting. This must
        # fail closed instead of clipping the parent's exclusive time to zero.
        clock = iter((0, 1, 8, 5))
        with patch.object(profiler.time, "perf_counter_ns", side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(RuntimeError, "child spans .* exceed inclusive"):
                recorder.call(
                    "parent", lambda: recorder.call("child", lambda: None)
                )
        self.assertEqual(recorder.stack, [])


class StatisticsTests(unittest.TestCase):
    def test_duration_and_process_statistics_are_deterministic(self) -> None:
        durations = profiler._stats_ns([1_000_000, 2_000_000, 3_000_000, 4_000_000])
        self.assertEqual(durations["n"], 4)
        self.assertAlmostEqual(durations["median_ms"], 2.5)
        self.assertAlmostEqual(durations["q1_ms"], 1.75)
        self.assertAlmostEqual(durations["q3_ms"], 3.25)
        self.assertAlmostEqual(durations["iqr_ms"], 1.5)

        process_stats = profiler._stats_process([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(process_stats["processes"], 4)
        self.assertAlmostEqual(process_stats["median_ms"], 2.5)
        self.assertAlmostEqual(process_stats["q1_ms"], 1.75)
        self.assertAlmostEqual(process_stats["q3_ms"], 3.25)
        self.assertAlmostEqual(process_stats["iqr_ms"], 1.5)

    def test_rss_statistics_keep_kib_units(self) -> None:
        stats = profiler._stats_kib([100.0, 200.0, 300.0, 400.0])
        self.assertEqual(stats["processes"], 4)
        self.assertAlmostEqual(stats["median_kib"], 250.0)
        self.assertAlmostEqual(stats["iqr_kib"], 150.0)
        self.assertNotIn("median_ms", stats)


class ResultArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_structured_result_covers_all_processes_shapes_and_roundtrips(self) -> None:
        result = self.result
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["warmups"], 2)
        self.assertEqual(result["repetitions_per_process"], 8)
        self.assertEqual(result["independent_process_pairs"], 4)
        self.assertEqual(len(result["workers"]), 32)
        self.assertEqual(len(result["execution_order"]), 32)
        self.assertTrue(all(row["exit_code"] == 0 for row in result["execution_order"]))
        self.assertEqual(
            {row["case"] for row in result["workers"]},
            {"1x1", "51x26", "3932x26", "26x3932"},
        )
        self.assertEqual(sum(row["backend"] == "cpu" for row in result["workers"]), 16)
        self.assertEqual(sum(row["backend"] == "opencl" for row in result["workers"]), 16)
        self.assertEqual(result["runtime_warning_observations"], 16)
        self.assertTrue(
            all(not row["cpu_only_opencl_api_loaded"] for row in result["workers"]
                if row["backend"] == "cpu")
        )
        for row in result["workers"]:
            self.assertTrue(row["layer"]["linear_init_roundtrip_exact"])
            for tensor in ("weights", "bias", "inputs"):
                self.assertTrue(row["api"][tensor]["roundtrip_exact"])
                self.assertEqual(row["api"][tensor]["buffer_from_host"]["n"], 8)
                self.assertEqual(row["api"][tensor]["buffer_to_host"]["n"], 8)

    def test_nested_spans_and_aggregate_statistics_are_valid(self) -> None:
        for row in self.result["workers"]:
            if row["backend"] != "opencl":
                continue
            stages = row["opencl_nested_components"]["weights"]
            for phase in ("buffer_from_host_nested_spans", "buffer_to_host_nested_spans"):
                for metrics in stages[phase].values():
                    for kind in ("inclusive_ms", "exclusive_ms"):
                        stats = metrics[kind]
                        self.assertGreater(stats["n"], 0)
                        self.assertGreater(stats["min_ms"], 0)
                        self.assertLessEqual(stats["min_ms"], stats["q1_ms"])
                        self.assertLessEqual(stats["q1_ms"], stats["median_ms"])
                        self.assertLessEqual(stats["median_ms"], stats["q3_ms"])
                        self.assertLessEqual(stats["q3_ms"], stats["max_ms"])
                        self.assertAlmostEqual(
                            stats["iqr_ms"], stats["q3_ms"] - stats["q1_ms"], places=9
                        )
                    self.assertLessEqual(
                        metrics["exclusive_ms"]["median_ms"],
                        metrics["inclusive_ms"]["median_ms"],
                    )

        aggregate = self.result["aggregate"]
        self.assertEqual(len(aggregate), 8)
        for group in aggregate.values():
            self.assertEqual(group["independent_processes"], 4)
            self.assertEqual(group["paths"]["weights.buffer_from_host"]["processes"], 4)
            rss = group["rss_delta_kib"]
            self.assertIn("median_kib", rss)
            self.assertNotIn("median_ms", rss)

    def test_cpu_backend_roundtrips_reference_shapes_without_opencl(self) -> None:
        backend = CPUBackend()
        for rows, columns in ((1, 1), (51, 26), (3932, 26), (26, 3932)):
            values = [
                [(-1.0 if (row + column) % 2 else 1.0) * (row * columns + column) / 17.0
                 for column in range(columns)]
                for row in range(rows)
            ]
            copied = backend.buffer_from_host(
                values, shape=(rows, columns), dtype="float64"
            )
            exported = backend.buffer_to_host(copied)
            self.assertEqual(exported, values)
            self.assertIsNot(copied, values)
            self.assertIsNot(copied[0], values[0])
            self.assertIsNot(exported[0], copied[0])

        # Use a fresh interpreter so unrelated tests that import OpenCL cannot
        # make this CPU-only lazy-import check order-dependent.
        cpu_only_code = (
            "import sys; from kinera.backends.cpu.backend import CPUBackend; "
            "b=CPUBackend(); assert b.buffer_to_host(b.buffer_from_host([1.0], "
            "shape=(1,), dtype='float64')) == [1.0]; "
            "assert 'kinera.backends.opencl.api' not in sys.modules"
        )
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run(
            [sys.executable, "-c", cpu_only_code],
            cwd=ROOT,
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_report_and_source_fingerprints_match_preserved_artifacts(self) -> None:
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        for relative_path, expected in EXPECTED_SHA256.items():
            with self.subTest(path=relative_path):
                actual = _sha256(ROOT / relative_path)
                self.assertEqual(actual, expected)
                if relative_path in {
                    "experiments/analyze_opencl_host_conversions.py",
                    "experiments/results/opencl-rx550-p3c-host-conversions.json",
                    "src/kinera/backends/opencl/memory.py",
                    "src/kinera/backends/cpu/backend.py",
                    "src/kinera/nn/layers/linear.py",
                }:
                    self.assertIn(expected, report_text)


if __name__ == "__main__":
    unittest.main()
