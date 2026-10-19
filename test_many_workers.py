"""Physical and statistical limits for the finite-history multi-worker study."""
import unittest
from pathlib import Path

class ManyWorkerTests(unittest.TestCase):
    def study(self, **kwargs):
        self.assertTrue((Path(__file__).parent/'many_workers_study.py').exists(),
                        'The multi-worker finite-history experiment is not implemented')
        from many_workers_study import study_many_workers
        return study_many_workers(**kwargs)

    def test_constant_histories_have_no_spread_at_any_worker_count(self):
        rows = self.study(seed=42, repeats=32, workers=[1, 8, 64], counts=[1, 16], models=['constant'])
        for row in rows:
            for key in ['mean_maximum','p025','median','p975']:
                self.assertEqual(row[key], 1.)

    def test_single_worker_mean_and_maximum_growth(self):
        args=dict(seed=42, repeats=6000, workers=[1, 16, 256], counts=[16,256], models=['mixed_poisson'])
        rows=self.study(**args)
        self.assertEqual(rows,self.study(**args))
        for count in (16,256):
            group=[r for r in rows if r['histories']==count]
            self.assertLess(abs(group[0]['mean_maximum']-1), .03)
            self.assertTrue(all(a['mean_maximum'] < b['mean_maximum'] for a,b in zip(group,group[1:])))
        self.assertGreater(rows[2]['mean_maximum'],rows[5]['mean_maximum'])

if __name__=='__main__':unittest.main()
