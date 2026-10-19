import unittest
import tempfile
from pathlib import Path
import numpy as np
from allocation_comparison import aggregate_time, allocate_counts, allocation_times, study_a, study_b, study_c, study_d
from models import capacity_at


class AllocationComparisonTests(unittest.TestCase):
    def test_case_a_retains_paired_samples_for_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'paired.npz'
            rows=study_a(20261018,repeats=3,sample_path=path)
            with np.load(path) as samples:
                static=samples['static_time'];theory=samples['theory_time']
                self.assertEqual(static.shape,(4,21,3))
                np.testing.assert_array_equal(static[:,0,:],theory[:,0,:])
                self.assertTrue(np.all(static>=theory))
                self.assertEqual(int(samples['seed']),20261018)
                self.assertEqual(int(samples['workers'][-1]),1048576)
                for j,std in enumerate(samples['parent_std']):
                    row=next(r for r in rows if r['parent_std']==std and r['workers']==1)
                    self.assertEqual(row['static_rate_p025'],row['theory_rate_p025'])
                    self.assertEqual(row['static_rate_p975'],row['theory_rate_p975'])
                    self.assertEqual(row['speedup_pct_p975'],0.)
                    self.assertEqual(row['static_rate_p025'],float(np.quantile(1/static[j,0],.025)))

    def test_single_worker_uses_identical_realized_rates_in_case_a(self):
        for row in study_a(20261018,repeats=3):
            if row['workers']==1:
                self.assertEqual(row['static_rate'],row['theory_rate'])
                self.assertEqual(row['static_time'],row['theory_time'])
                self.assertEqual(row['speedup_pct'],0.)

    def test_single_worker_uses_identical_work_and_rates_in_case_b(self):
        for row in study_b(20261018,repeats=3):
            if row['workers']==1:
                self.assertEqual(row['static_rate'],row['theory_rate'])
                self.assertEqual(row['static_time'],row['theory_time'])
                self.assertEqual(row['speedup_pct'],0.)

    def test_half_workers_sphere_ratio_analytic_comparison(self):
        rows=[r for r in study_c() if r['drop']==1-1/3.343 and r['affected_fraction']==.5]
        self.assertEqual(len(rows),2)
        for r in rows:
            if r['profile']=='step':
                self.assertAlmostEqual(r['nominal_time'],.3+.7*3.343)
                self.assertAlmostEqual(r['theory_time'],(.85*3.343+.15)/(.5*3.343+.5))
            else:
                self.assertAlmostEqual(r['nominal_time'],.425+.575*3.343)
                self.assertAlmostEqual(r['theory_time'],(.7875*3.343+.2125)/(.5*3.343+.5))

    def test_exact_half_workers_in_combined_study(self):
        rows=study_d(20261018,repeats=4,affected_fraction=.5,drop=.4,exact_fraction=True)
        drift=next(r for r in rows if r['case']=='drift_only' and r['pilot_histories']==16)
        self.assertAlmostEqual(drift['nominal_time'],83/60)
        self.assertAlmostEqual(drift['pilot_time'],83/60)
        self.assertAlmostEqual(drift['theory_time'],183/160)

    def test_actual_work_and_actual_rates(self):
        rates=np.array([.6, 1.2, .9])
        self.assertAlmostEqual(aggregate_time(81.,rates,np.zeros(3),10,4),30.)

    def test_time_varying_capacity_is_conserved(self):
        rates=np.array([.6, 1.2, .9]);drop=np.array([.4,0,.6])
        t=aggregate_time(81.,rates,drop,3,2)
        self.assertAlmostEqual(capacity_at(t,base_rate=rates,drop=drop,onset=3,ramp=2).sum(),81.)

    def test_identical_constant_work_has_no_pilot_penalty(self):
        w=np.full((8,32),25.)
        out=allocation_times(w,np.ones(8),np.zeros(8),[1,8,16,32])
        self.assertAlmostEqual(out['theory'],800.)
        for t in out['pilot'].values():self.assertAlmostEqual(t,800.)

    def test_whole_run_pilot_equals_nominal(self):
        rng=np.random.default_rng(5);w=5+rng.poisson(20,(8,32))
        out=allocation_times(w,np.linspace(.6,1.4,8),np.array([.4,0]*4),[1,16,32])
        self.assertAlmostEqual(out['pilot'][32],out['nominal'])
        self.assertTrue(all(t>=out['theory']-1e-8 for t in out['pilot'].values()))
        self.assertGreaterEqual(out['nominal'],out['theory'])

    def test_count_allocation_conserves_histories(self):
        for n in (0,1,7,1001):
            x=allocate_counts(n,np.array([1.,3.,.2]))
            self.assertEqual(x.sum(),n)
            self.assertTrue(np.all(x>=0))
            self.assertTrue(np.all(np.abs(x-n*np.array([1.,3.,.2])/4.2)<1.0000001))

    def test_work_is_reused_without_loss(self):
        w=np.arange(1,97,dtype=float).reshape(3,32)
        out=allocation_times(w,np.array([1.,1.5,.7]),np.array([0.,.4,0.]),[1,8,16,32])
        self.assertEqual(out['work'],w.sum())
        for x in out['assigned_work'].values():self.assertAlmostEqual(x.sum(),w.sum())

if __name__=='__main__': unittest.main()
