"""Regression checks for detecting changed or incomplete reproduction results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from verify_results import compare_result_dirs


class ReproductionVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.reference = Path(self.tmp.name) / 'reference'
        self.candidate = Path(self.tmp.name) / 'candidate'
        self.reference.mkdir()
        self.candidate.mkdir()
        for folder in (self.reference, self.candidate):
            (folder / 'allocation_a.csv').write_text('workers,static_time\n1,1.0\n4,1.25\n')
            (folder / 'allocation_summary.json').write_text(json.dumps({'seed': 20261018, 'a': [{'workers': 1, 'speedup_pct': 0.0}]}))
            np.savez_compressed(folder / 'allocation_a_paired_samples.npz', static_time=np.array([[1., 1.25]]), theory_time=np.array([[1., 1.01]]))

    def test_identical_values_pass(self):
        self.assertEqual(compare_result_dirs(self.reference, self.candidate), [])

    def test_changed_csv_is_reported(self):
        (self.candidate / 'allocation_a.csv').write_text('workers,static_time\n1,1.0\n4,1.30\n')
        errors = compare_result_dirs(self.reference, self.candidate)
        self.assertTrue(any('allocation_a.csv' in x and 'static_time' in x for x in errors), errors)

    def test_missing_output_is_reported(self):
        (self.candidate / 'allocation_summary.json').unlink()
        self.assertTrue(any('allocation_summary.json' in x for x in compare_result_dirs(self.reference, self.candidate)))

    def test_seed_change_is_reported(self):
        (self.candidate / 'allocation_summary.json').write_text(json.dumps({'seed': 1, 'a': [{'workers': 1, 'speedup_pct': 0.0}]}))
        self.assertTrue(any('seed' in x for x in compare_result_dirs(self.reference, self.candidate)))

    def test_changed_paired_sample_is_reported(self):
        np.savez_compressed(self.candidate / 'allocation_a_paired_samples.npz', static_time=np.array([[1., 1.25]]), theory_time=np.array([[1., 1.05]]))
        self.assertTrue(any('theory_time' in x for x in compare_result_dirs(self.reference, self.candidate)))

    def test_nonfinite_candidate_does_not_pass(self):
        (self.candidate / 'allocation_a.csv').write_text('workers,static_time\n1,1.0\n4,nan\n')
        self.assertTrue(compare_result_dirs(self.reference, self.candidate))

    def test_empty_reference_is_rejected(self):
        empty = Path(self.tmp.name) / 'empty'
        empty.mkdir()
        self.assertTrue(compare_result_dirs(empty, self.candidate))

    def test_runtime_version_changes_do_not_mask_numeric_comparison(self):
        for folder, version in ((self.reference, '3.12.14'), (self.candidate, '3.13.0')):
            (folder / 'summary.json').write_text(json.dumps({'metadata': {'python': version, 'numpy': '2.5.4', 'seed': 20261014}, 'minimum_rates': [{'mean_minimum': 0.8}]}))
        self.assertEqual(compare_result_dirs(self.reference, self.candidate), [])


if __name__ == '__main__':
    unittest.main()
