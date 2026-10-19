"""Workload and effective-rate models for mechanism experiments.

Time and capacity are normalized. The functions do not execute or emulate an
MPI scheduler. A workload includes the complete pilot subset used to predict it.
"""
from statistics import NormalDist
import numpy as np


def workload_moments(model, lam=20.0, overhead=5.0):
    """Return single-history work mean and variance (in cycle-like units)."""
    if not np.isfinite(lam) or lam < 0 or not np.isfinite(overhead) or overhead <= 0:
        raise ValueError('lam must be nonnegative and overhead positive')
    if model == 'constant':
        return overhead + lam, 0.0
    if model == 'poisson':
        return overhead + lam, lam
    if model == 'mixed_poisson':
        intensities = np.array([.2, 8.2]) * lam
        between = np.dot([.9, .1], (intensities-lam)**2)
        return overhead + lam, float(lam + between)
    raise ValueError(f'Unknown workload model: {model}')


def sample_work_sum(rng, n, size, model='poisson', lam=20.0, overhead=5.0):
    """Sample the sum of n IID histories without materializing all histories."""
    workload_moments(model, lam, overhead)
    if int(n) != n or n < 0:
        raise ValueError('n must be a nonnegative integer')
    n = int(n)
    if model == 'constant':
        return np.full(size, n*(overhead+lam), dtype=float)
    if model == 'poisson':
        counts = rng.poisson(n*lam, size=size)
    else:
        long_count = rng.binomial(n, .1, size=size)
        intensity = ((n-long_count)*.2 + long_count*8.2)*lam
        counts = rng.poisson(intensity)
    return counts.astype(float) + n*overhead


def prefix_workloads(rng, counts, size, model='poisson', **kwargs):
    """Generate nested prefixes so that the pilot remains part of the total."""
    counts = list(counts)
    if not counts or any(int(n) != n or n <= 0 for n in counts):
        raise ValueError('prefix counts must be positive integers')
    if counts != sorted(set(counts)):
        raise ValueError('prefix counts must be unique and increasing')
    previous = 0
    cumulative = np.zeros(size, dtype=float)
    result = {}
    for count in counts:
        cumulative = cumulative + sample_work_sum(rng, count-previous, size, model, **kwargs)
        result[count] = cumulative.copy()
        previous = count
    return result


def minimum_rate_samples(rng, workers, cv, size, lower=.2, upper=1.8):
    """Exact inverse-CDF samples of a truncated-normal order statistic.

    The parent normal has mean 1 and standard deviation cv. Only one uniform
    draw per repetition is needed even when workers is in the millions.
    """
    if int(workers) != workers or workers < 1 or not np.isfinite(cv) or cv <= 0:
        raise ValueError('workers and cv must be positive')
    if not 0 < lower < 1 < upper:
        raise ValueError('positive bounds must straddle 1')
    normal = NormalDist()
    lo, hi = normal.cdf((lower-1)/cv), normal.cdf((upper-1)/cv)
    uniform = rng.random(size)
    minimum_cdf = -np.expm1(np.log1p(-uniform)/workers)
    probability = lo + (hi-lo)*minimum_cdf
    probability = np.clip(probability, np.nextafter(0., 1.), np.nextafter(1., 0.))
    z = np.fromiter((normal.inv_cdf(float(p)) for p in probability.flat),
                    dtype=float, count=probability.size).reshape(probability.shape)
    return np.clip(1+cv*z, lower, upper)


def _rate_arguments(base_rate, drop, onset, ramp, mode):
    base_rate, drop = np.asarray(base_rate, float), np.asarray(drop, float)
    if np.any(~np.isfinite(base_rate)) or np.any(base_rate <= 0):
        raise ValueError('base_rate must be finite and positive')
    if np.any(~np.isfinite(drop)) or np.any(drop < 0) or np.any(drop >= 1):
        raise ValueError('drop must lie in [0, 1)')
    if not np.isfinite(onset) or onset < 0 or not np.isfinite(ramp) or ramp < 0:
        raise ValueError('onset and ramp must be finite and nonnegative')
    if mode not in ('step', 'ramp'):
        raise ValueError('mode must be step or ramp')
    return base_rate, drop


def rate_at(t, base_rate=1.0, drop=0.0, onset=.3, ramp=.25, mode='ramp'):
    """Instantaneous capacity under a step or a linear thermal transition."""
    base_rate, drop = _rate_arguments(base_rate, drop, onset, ramp, mode)
    t = np.asarray(t, float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError('time must be finite and nonnegative')
    fraction = (t >= onset).astype(float) if mode == 'step' or ramp == 0 else np.clip((t-onset)/ramp, 0, 1)
    return base_rate*(1-drop*fraction)


def capacity_at(t, base_rate=1.0, drop=0.0, onset=.3, ramp=.25, mode='ramp'):
    """Integral of rate_at from zero through t, with exact ramp integration."""
    base_rate, drop = _rate_arguments(base_rate, drop, onset, ramp, mode)
    t = np.asarray(t, float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError('time must be finite and nonnegative')
    elapsed = np.maximum(t-onset, 0.)
    if mode == 'step' or ramp == 0:
        loss = drop*elapsed
    else:
        loss = drop*(np.minimum(elapsed, ramp)**2/(2*ramp) + np.maximum(elapsed-ramp, 0.))
    return base_rate*(t-loss)


def completion_time(work, base_rate=1.0, drop=0.0, onset=.3, ramp=.25, mode='ramp'):
    """Invert integrated rate, including speed changes during the workload."""
    base_rate, drop = _rate_arguments(base_rate, drop, onset, ramp, mode)
    work = np.asarray(work, float)
    if np.any(~np.isfinite(work)) or np.any(work < 0):
        raise ValueError('work must be finite and nonnegative')
    demand = work/base_rate
    after_start = np.maximum(demand-onset, 0.)
    if mode == 'step' or ramp == 0:
        later = onset+after_start/(1-drop)
    else:
        ramp_capacity = ramp*(1-drop/2)
        # Stable form of the smaller quadratic root; no cancellation at drop=0.
        root = np.sqrt(np.maximum(1-2*drop*after_start/ramp, 0.))
        during = 2*after_start/(1+root)
        beyond = ramp+(after_start-ramp_capacity)/(1-drop)
        later = onset+np.where(after_start <= ramp_capacity, during, beyond)
    return np.where(demand <= onset, demand, later)


def prediction_error(total_work, pilot_work, total_histories, pilot_histories, **profile):
    """Return pilot-predicted time, realized total time, and relative error."""
    if not 0 < pilot_histories <= total_histories:
        raise ValueError('pilot must be a nonempty subset of all histories')
    if np.any(np.asarray(pilot_work) > np.asarray(total_work)):
        raise ValueError('pilot work cannot exceed total work')
    pilot_time = completion_time(pilot_work, **profile)
    actual_time = completion_time(total_work, **profile)
    if np.any(actual_time == 0):
        raise ValueError('total time must be positive')
    predicted = total_histories/pilot_histories*pilot_time
    return predicted, actual_time, predicted/actual_time-1
