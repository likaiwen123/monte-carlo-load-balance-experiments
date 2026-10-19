"""Checks of experimental outputs against analytic and accounting identities."""
import unittest

from run_study import study_minimum, study_histories, study_drift, study_combined


class ExperimentTests(unittest.TestCase):
    def test_seeded_minimum_export_and_ordering(self):
        a = study_minimum(42, repeats=100, workers=[1, 64, 4096])
        self.assertEqual(a, study_minimum(42, repeats=100, workers=[1, 64, 4096]))
        for epsilon in (.01, .05, .1, .2):
            rows = [r for r in a if r['parent_std'] == epsilon]
            self.assertTrue(all(x['mean_minimum'] >= y['mean_minimum'] for x, y in zip(rows, rows[1:])))

    def test_history_export_matches_theoretical_variability(self):
        for row in study_histories(42, repeats=25000, counts=[4, 64]):
            self.assertLess(abs(row['empirical_time_cv']/row['analytic_time_cv']-1), .05)

    def test_drift_summary_has_hand_calculated_step(self):
        rows = study_drift()
        row = next(r for r in rows if r['profile'] == 'step' and r['drop'] == .4)
        self.assertAlmostEqual(row['completion_time'], .3+.7/.6)

    def test_complete_pilot_error_zero_in_every_combined_case(self):
        rows = study_combined(42, repeats=4, workers=8, histories=16, counts=[1, 4, 16])
        full = [r for r in rows if r['pilot_histories'] == 16]
        self.assertEqual(len(full), 4)
        for row in full:
            self.assertEqual(row['mean_abs_error_pct'], 0.)
        self.assertTrue(all(row['pilot_fraction'] <= 1 for row in rows))


if __name__ == '__main__':
    unittest.main()
