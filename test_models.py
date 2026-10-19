"""Independent analytic checks for the supplementary mechanism study."""
import unittest
import numpy as np

from models import (workload_moments, sample_work_sum, prefix_workloads,
                    minimum_rate_samples, rate_at, capacity_at,
                    completion_time, prediction_error)


class WorkloadTests(unittest.TestCase):
    def test_poisson_analytic_moments(self):
        self.assertEqual(workload_moments('poisson'), (25.0, 20.0))

    def test_mixture_includes_between_intensity_variance(self):
        mean, variance = workload_moments('mixed_poisson')
        self.assertAlmostEqual(mean, 25.0)
        self.assertAlmostEqual(variance, 20 + .9 * (4-20)**2 + .1 * (164-20)**2)

    def test_sum_samples_match_analytic_moments(self):
        for model in ('poisson', 'mixed_poisson'):
            mean, variance = workload_moments(model)
            draws = sample_work_sum(np.random.default_rng(12), 8, 100000, model)
            self.assertLess(abs(draws.mean() / (8*mean) - 1), .006)
            self.assertLess(abs(draws.var() / (8*variance) - 1), .025)

    def test_prefixes_include_pilot_and_repeat_seed(self):
        a = prefix_workloads(np.random.default_rng(4), [1, 4, 16], 100, 'mixed_poisson')
        b = prefix_workloads(np.random.default_rng(4), [1, 4, 16], 100, 'mixed_poisson')
        for n in a:
            np.testing.assert_array_equal(a[n], b[n])
        self.assertTrue(np.all(a[16] >= a[4]))
        self.assertTrue(np.all(a[4] >= a[1]))
        np.testing.assert_array_equal(sample_work_sum(np.random.default_rng(0), 0, 3, 'poisson'), 0)

    def test_invalid_workload_inputs(self):
        with self.assertRaises(ValueError):
            workload_moments('unknown')
        with self.assertRaises(ValueError):
            sample_work_sum(np.random.default_rng(0), -1, 3, 'poisson')


class MinimumTests(unittest.TestCase):
    def test_one_worker_symmetric_mean(self):
        draws = minimum_rate_samples(np.random.default_rng(1), 1, .2, 20000)
        self.assertLess(abs(draws.mean()-1), .005)
        self.assertTrue(np.all((draws >= .2) & (draws <= 1.8)))

    def test_coupled_minima_are_nonincreasing(self):
        one = minimum_rate_samples(np.random.default_rng(2), 1, .1, 1000)
        many = minimum_rate_samples(np.random.default_rng(2), 1024, .1, 1000)
        self.assertTrue(np.all(many <= one))

    def test_large_worker_count_stays_positive(self):
        draws = minimum_rate_samples(np.random.default_rng(3), 2**20, .2, 1000)
        self.assertTrue(np.all(np.isfinite(draws)))
        self.assertGreaterEqual(draws.min(), .2)

    def test_invalid_rate_arguments(self):
        with self.assertRaises(ValueError):
            minimum_rate_samples(np.random.default_rng(0), 0, .1, 3)


class CapacityTests(unittest.TestCase):
    def test_constant_rate(self):
        np.testing.assert_allclose(completion_time([0, 2, 5], base_rate=2), [0, 1, 2.5])

    def test_step_hand_calculation(self):
        self.assertAlmostEqual(float(completion_time(1, drop=.4, onset=.3, mode='step')), .3+.7/.6)

    def test_ramp_hand_calculation(self):
        self.assertAlmostEqual(float(capacity_at(.55, drop=.4, onset=.3, ramp=.25)), .5)
        self.assertAlmostEqual(float(completion_time(1, drop=.4, onset=.3, ramp=.25)), .55+.5/.6)

    def test_inverse_over_profiles_and_vector_rates(self):
        t = np.linspace(0, 2, 101)
        for mode in ('step', 'ramp'):
            work = capacity_at(t, base_rate=1.7, drop=.6, onset=.3, ramp=.25, mode=mode)
            np.testing.assert_allclose(completion_time(work, base_rate=1.7, drop=.6, onset=.3, ramp=.25, mode=mode), t, atol=1e-12)
        work = [1, 1]
        np.testing.assert_allclose(completion_time(work, base_rate=[1, 2]), [1, .5])

    def test_ramp_is_continuous_and_rate_is_derivative(self):
        for t in (.1, .4, .8):
            derivative = (capacity_at(t+1e-6, drop=.4)-capacity_at(t-1e-6, drop=.4))/2e-6
            self.assertAlmostEqual(float(derivative), float(rate_at(t, drop=.4)), places=6)

    def test_invalid_capacity_arguments(self):
        for kwargs in ({'base_rate':0}, {'drop':1}, {'drop':-.1}, {'mode':'bad'}):
            with self.assertRaises(ValueError):
                completion_time(1, **kwargs)

    def test_full_pilot_has_zero_prediction_error(self):
        predicted, actual, error = prediction_error([200, 400], [200, 400], 16, 16,
            base_rate=[1, 2], drop=.4, onset=20, ramp=10)
        np.testing.assert_array_equal(predicted, actual)
        np.testing.assert_array_equal(error, 0)

    def test_prediction_uses_actual_pilot_time(self):
        predicted, actual, error = prediction_error(100, 50, 4, 2, drop=.5, onset=10, mode='step')
        self.assertAlmostEqual(float(predicted), 2*(10+40/.5))
        self.assertAlmostEqual(float(actual), 10+90/.5)
        self.assertAlmostEqual(float(error), float(predicted/actual-1))


if __name__ == '__main__':
    unittest.main()
