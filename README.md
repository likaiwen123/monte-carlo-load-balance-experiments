# Monte Carlo load balance experiments

This package reproduces the supplementary experiments for **A Time-based Dynamic Load Balance Method for Monte Carlo Shielding Simulations**. It contains all simulation scripts, numerical tests, reference results, ten figures, and an editable plotting notebook.

The experiments compare static allocations with the proposed method using completion times evaluated from the realized workload and CPU rates. They cover differences between CPU rates, random particle-history workloads, changes in CPU rates during execution, and combinations of these effects.

## Quick start

Clone this repository and use Python 3.12. The pinned dependencies match the environment used to generate the supplied reference results.

```sh
git clone https://github.com/likaiwen123/monte-carlo-load-balance-experiments.git
cd monte-carlo-load-balance-experiments
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the numerical tests:

```sh
python -m unittest discover -s . -p "test_*.py" -v
```

Redraw all ten figures from the supplied data, without resampling:

```sh
python reproduce.py plot --input reference_results --output replotted_figures
```

Rerun all experiments with the published parameters and random seeds:

```sh
python reproduce.py run --output reproduced_results
python verify_results.py --reference reference_results --candidate reproduced_results
```

The full run includes **6,000 paired repetitions** for comparison (a), with up to **1,048,576 workers** in each realization. This is the most computationally demanding part. Progress is printed every 500 repetitions. Each experiment runs in a single Python process and simulates the workers numerically; a computing cluster is not required. `reproduced_results/run_manifest.json` records elapsed times, runtime versions, and source-file checksums.

Install Times New Roman on the machine to reproduce the figure typography. The font itself is not distributed in this package. Numerical results do not depend on the font.

## Running selected groups

The three groups can be run or plotted independently:

```sh
python reproduce.py run --group mechanisms --output reproduced_results
python reproduce.py run --group many-workers --output reproduced_results
python reproduce.py run --group comparisons --output reproduced_results
python reproduce.py plot --group comparisons --input reference_results --output replotted_figures
```

| Group | Script | Default seed | Contents |
| --- | --- | --- | --- |
| `mechanisms` | `run_study.py` | 20261014 | Minimum-rate statistics, finite-history variability, rate changes, and prediction errors |
| `many-workers` | `many_workers_study.py` | 20261016 | Finite-history completion times on identical workers |
| `comparisons` | `allocation_comparison.py` | 20261018 | Paired comparisons (a)–(d), including the two exactly-50%-affected cases |

The original scripts also remain directly executable with `--output`. `run_study.py` and `allocation_comparison.py` accept `--seed`; `many_workers_study.py` uses the default seed above. Experimental parameters are specified in their `study_*` functions. The unified entry point calls these scripts without changing their defaults.

## Models and comparison protocol

A history has workload `W = 5 + K`. The simple model uses `K ~ Poisson(20)`. The mixed model selects `Poisson(4)` with probability 0.9 or `Poisson(164)` with probability 0.1. Both have mean workload 25; their workload variances are 20 and 2,324. Fixed-work controls use `W = 25`.

CPU rates are either one or sampled from a normal distribution with mean one, truncated by rejection sampling to `[0.2, 1.8]`. The symbol σ specifies the parent normal standard deviation. A rate change starts at normalized time 0.30; a gradual change develops linearly over a further 0.25. `models.py` provides analytical rate integrals and their inverses.

For constant rates, the proposed method's completion time is `T_D = P / sum(u_i)`, where `P` is the realized total workload. For changing rates, it solves `sum(integral_0^T_D u_i(t) dt) = P`. The effective rate per worker is `P / (M T)`. Reported speedups average the paired quantities `100 (T_static / T_D - 1)`. Curves show sample means unless otherwise stated; shaded 95% ranges are the 2.5th–97.5th percentiles of realized results.

Within each comparison, the methods share the same realized histories and CPU profiles. A retained trial processes the first `k` histories on every worker. Once these trials finish, remaining histories are assigned in proportion to `k / measured_trial_time`, using largest remainders for integer counts. Trial work and elapsed time are included. Rate changes use the same absolute time axis for all methods.

## Experiment settings

| Experiment | Settings | Repetitions |
| --- | --- | --- |
| Original minimum-rate analysis | σ = 0.01, 0.05, 0.10, 0.20; worker counts `2^0` through `2^20`; inverse-CDF sampling of the minimum | 6,000 |
| Single-worker history variability | Simple and mixed Poisson; `2^0` through `2^10` histories | 12,000 |
| Rate-change illustration | Abrupt and gradual changes; rate reductions 20%, 40%, 60% | Analytical |
| Prediction-error controls | 256 workers × 256 histories; trial sizes 1, 4, 16, 64, 128, 256; four controls | 128 |
| Identical-worker history study | Simple and mixed Poisson; 16, 64, 256 histories; 1, 4, 16, 64, 256, 1,024, 4,096 workers | 2,000 |
| Comparison (a) | Same rate parameters and worker counts as the original minimum study; complete CPU-rate realizations shared by the methods | 6,000 |
| Comparison (b) | Simple and mixed Poisson; 16, 64, 256 histories; 1–4,096 workers as above; σ = 0 or 0.10 | 2,000 |
| Comparison (c) | Affected fractions 0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1; reductions 0.20, 0.40, 0.60, `1−1/3.343`; abrupt and gradual changes | Analytical |
| Comparison (d) | 256 workers × 256 histories; four controls; trial sizes as above; each worker independently has probability 0.25 of a 40% rate reduction | 128 |
| Exactly-half comparisons | Exactly 128 of 256 workers affected; reductions 40% and `1−1/3.343 ≈ 70.09%`; paired inputs differing only in reduction magnitude | 128 |

The four controls are node differences only, history variability only, rate change only, and all three combined. Node differences use σ = 0.10; history variability uses the mixed-Poisson model; rate changes are gradual in the combined experiments.

`allocation_a_paired_samples.npz` retains the static and proposed completion times for each σ, worker count, and repetition: shape `(4, 21, 6000)` for each method. It also stores the seed, σ values, and worker counts. These paired times allow rates, speedups, and quantiles to be recomputed without generating the CPU-rate arrays again.

## Figure map

All figures are supplied as PNG and PDF in `reference_results`. The first eight rows are used in the appendix; the final two are additional mechanism plots.

| Appendix figure | File stem | Meaning |
| --- | --- | --- |
| A.1 | `fig06_mean_minimum_comparison` | Effective rates and paired speedups in comparison (a) |
| A.2 | `fig02_history_variability` | Finite-history variability and trial-rate estimates |
| A.3 | `fig07_history_rate_comparison` | History variability, CPU-rate variability, and their combination |
| A.4 | `fig03_rate_drift` | Rate profiles and cumulative capacity |
| A.5 | `fig08_drift_allocation_comparison` | Changing rates and the fraction of affected workers |
| A.6 | `fig04_combined` | Nominal-rate and trial-based time-prediction errors |
| A.7 | `fig09_static_theory_comparison` | Method comparisons in the original combined case |
| A.8 | `fig10_half_workers_paired` | Two reduction magnitudes with exactly half the workers affected |
| Additional | `fig01_minimum_rate` | Original minimum-rate analysis |
| Additional | `fig05_many_workers_histories` | Completion-time variability on identical workers |

## Editing figures in the notebook

`appendix_all_figures.ipynb` includes editable code, embedded numerical data, and all ten executed figure outputs. It can be copied and used independently. Install the optional notebook environment with:

```sh
python -m pip install -r requirements-notebook.txt
jupyter lab appendix_all_figures.ipynb
```

Run the common-settings, data, and helper cells before individual figure cells. Settings include `FONT_FAMILY`, `FONT_DIR`, `FONT_SCALE`, `COLORS`, `FIG_SIZES`, and `SHOW_95_RANGE`. The settings cell discovers and registers installed fonts if Matplotlib has a stale font cache. To use font files from another location, set `FONT_DIR` to that folder and rerun the settings cell.

Set `EXPORT_FIGURES = True` to write PNG/PDF files to `notebook_figures`. The notebook embeds the supplied snapshot. To read newly generated data, set `USE_LOCAL_DATA = True` and point `DATA_DIR` to `reproduced_results` (or `reference_results` for the supplied files).

## Verification and files

`verify_results.py` checks recognized numerical reference files, including CSV rows, JSON statistics and parameters, and paired NPZ arrays. It reports missing files and changed or non-finite values, returns a nonzero exit status on failure, and uses default relative/absolute tolerances of `1e-12`. Runtime-version fields are recorded but excluded from numerical comparison. PNG/PDF bytes are not part of this numerical check.

Tests cover analytical moments, capacity integration and inversion, paired single-worker equality, conservation of histories and workload, exactly-half controls, and detection of altered or incomplete results. `VALIDATION.md` records verification performed for this distribution. `MANIFEST.sha256` contains file checksums; with GNU coreutils, run `sha256sum -c MANIFEST.sha256` from this directory to check them.

| File or directory | Purpose |
| --- | --- |
| `models.py` | Workload distributions, rate profiles, and analytical times |
| `run_study.py`, `many_workers_study.py`, `allocation_comparison.py` | Complete experiment implementations |
| `reproduce.py` | Unified simulation and plotting entry point |
| `verify_results.py` | Numerical comparison with the reference data |
| `test_*.py` | Numerical and reproducibility tests |
| `requirements.txt`, `environment.json` | Pinned dependencies and reference runtime |
| `requirements-notebook.txt` | Optional Jupyter environment |
| `reference_results/` | Numerical outputs, paired times, and figures |
| `appendix_all_figures.ipynb` | Editable plotting notebook with saved outputs |

The scripts and supplied data run independently of the manuscript repository and do not require Word, LaTeX, Typst, or Git.

## Relationship to the manuscript

This repository provides the numerical experiments discussed in Appendix A of **A Time-based Dynamic Load Balance Method for Monte Carlo Shielding Simulations**. It contains the standalone materials previously distributed as `appendix_experiments.zip`. The C++ implementation of the dynamic load balance method is maintained separately at https://github.com/likaiwen123/dynamic-load-balance.

Use the files in `reference_results` to reproduce the published figures without resampling. The repository contains no manuscript drafts or reviewer correspondence.
