"""CPU-only integrity checks for the independent F2.3a math audit."""
import hashlib
import importlib.util
import json
import math
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ce_math_probe', ROOT / 'experiments/investigate_opencl_fp64_cross_entropy_math.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class CrossEntropyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'experiments/results/opencl-rx550-f2-3a-math-audit.json').read_text())

    def test_references_and_special_values(self):
        self.assertEqual(probe.decimal_unary('exp', 0.0), 1.0)
        self.assertEqual(probe.decimal_unary('exp', -math.inf), 0.0)
        self.assertEqual(probe.decimal_unary('log', 1.0), 0.0)
        self.assertTrue(math.isclose(probe.decimal_ce([0.0, 0.0], 0), math.log(2), rel_tol=1e-15))
        self.assertEqual(probe.compare(float('nan'), float('nan'))['status'], 'PASS')
        self.assertEqual(probe.compare(math.inf, -math.inf)['status'], 'FAIL')
        self.assertEqual(probe.compare(1., math.nextafter(1., math.inf))['ulps'], 1)
        self.assertEqual(probe.compare(5.54907608489522, 5.549076084930974)['status'], 'FAIL')
        self.assertEqual(probe.compare(1e-300, 0.)['status'], 'PASS')

    def test_provenance_and_record_counts(self):
        data = self.data
        self.assertEqual(data['baseline'], probe.BASELINE)
        self.assertEqual(data['source_sha256'], hashlib.sha256(probe.SOURCE.encode()).hexdigest())
        self.assertEqual(data['repetitions_per_case'], 3)
        self.assertEqual(len(data['records']), 148)
        for opt in probe.OPTIONS:
            counts = Counter(r['group'] for r in data['records'] if r['options'] == opt)
            self.assertEqual(counts, {'exp': 23, 'log': 14, 'sum': 6, 'ce': 31})

    def test_records_comparisons_echo_and_outcomes(self):
        for row in self.data['records']:
            self.assertTrue(row['stable'])
            if row['group'] in ('exp', 'log'):
                self.assertEqual(row['input_bits'], row['buffer_echo_bits'])
                self.assertEqual(row['input_bits'], row['scalar_echo_bits'])
                comparisons = [row['buffer'], row['scalar']]
            elif row['group'] == 'sum':
                comparisons = list(row['modes'].values())
            else:
                self.assertEqual(row['inputs_bits'], row['echo_bits'])
                comparisons = [v for item in row['stages'].values() for v in (item if isinstance(item, list) else [item])]
            for result in comparisons:
                expected, actual = probe.value(result['expected_bits']), probe.value(result['actual_bits'])
                self.assertEqual(probe.compare(expected, actual), result)
        for opt in probe.OPTIONS:
            rows = [r for r in self.data['records'] if r['options'] == opt]
            self.assertEqual(sum(r['buffer']['status'] == 'FAIL' for r in rows if r['group'] == 'exp'), 0)
            self.assertEqual(sum(r['buffer']['status'] == 'FAIL' for r in rows if r['group'] == 'log'), 3)
            self.assertEqual(sum(r['stages']['loss']['status'] == 'FAIL' for r in rows if r['group'] == 'ce'), 6)


if __name__ == '__main__':
    unittest.main()
