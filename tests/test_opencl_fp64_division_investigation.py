"""CPU-only checks for the FP64 diagnostic's references and accounting."""
import importlib.util
import json
import math
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fp64_probe", ROOT / "experiments/investigate_opencl_fp64_division.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class DivisionProbeTests(unittest.TestCase):
    def test_ulp_and_absolute_tolerance(self):
        self.assertEqual(probe.compare(1.0, math.nextafter(1.0, math.inf))["ulps"], 1)
        self.assertEqual(probe.compare(-1.0, math.nextafter(-1.0, -math.inf))["ulps"], 1)
        self.assertEqual(probe.compare(1 / 3, 0.3333333320915699)["status"], "FAIL")
        self.assertEqual(probe.compare(1e-300, 0.0)["status"], "PASS")

    def test_special_reference_and_zero_sign(self):
        self.assertTrue(math.isnan(probe.reference(0.0, 0.0)))
        self.assertEqual(probe.reference(1.0, -0.0), -math.inf)
        self.assertFalse(probe.compare(-0.0, 0.0)["zero_sign_matches"])
        self.assertEqual(probe.compare(math.nan, math.nan)["status"], "PASS")
        self.assertEqual(probe.compare(math.inf, -math.inf)["status"], "FAIL")

    def test_rational_reference(self):
        self.assertEqual(probe.bits(probe.rational_reference(1.0, 3.0)), probe.bits(1.0 / 3.0))
        self.assertEqual(probe.rational_reference(sys.float_info.max, 0.5), math.inf)
        self.assertIsNone(probe.rational_reference(1.0, 0.0))

    def test_published_result_structure(self):
        data = json.loads((ROOT / "experiments/results/opencl-rx550-fp64-division.json").read_text())
        self.assertEqual(len(data["results"]), len(probe.CASES) * len(probe.OPTIONS))
        self.assertEqual(data["source_sha256"], __import__("hashlib").sha256(probe.SOURCE.encode()).hexdigest())
        for row in data["results"]:
            self.assertEqual(row["repetitions"], 3)
            for name, result in row["variants"].items():
                with self.subTest(case=row["case"], name=name, options=row["options"]):
                    expected = struct.unpack(">d", bytes.fromhex(result["expected_bits"]))[0]
                    actual = struct.unpack(">d", bytes.fromhex(result["actual_bits"]))[0]
                    self.assertEqual(probe.bits(actual), result["actual_bits"])
                    self.assertEqual(probe.compare(expected, actual), result)


if __name__ == "__main__":
    unittest.main()
