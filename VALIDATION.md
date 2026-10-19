# Validation

The distribution was extracted into a separate directory and run independently of the manuscript repository.

- All 42 numerical and reproducibility tests passed.
- All three experiment groups were rerun with their full default settings, including 6,000 paired repetitions for each sigma in comparison (a).
- All 15 numerical reference files matched the reproduced outputs. The comparison also passed with both relative and absolute tolerances set to zero, including the paired NPZ time arrays.
- Both the full simulation route and the saved-data plotting route generated all ten PNG and ten PDF figures.
- All exported figure PDFs use Times New Roman fonts.
- The plotting notebook was previously executed independently with all ten figure outputs saved; its scientific contents are unchanged in this distribution.
- The original supplementary distribution contained 54 files. The current repository manifest covers all distributed files except the manifest itself and Git metadata.

Runtime: Python 3.12.14, NumPy 2.5.4, Matplotlib 3.11.2. The pinned NumPy and Matplotlib packages were also successfully downloaded from the Python package index.

Observed wall times on the validation machine are shown below; the calculation uses one Python process per experiment group.

| Group | Wall time (seconds) |
| --- | ---: |
| mechanisms | 1.7 |
| many-workers | 3.0 |
| comparisons | 314.9 |

Commands used:

```sh
python -m unittest discover -s . -p "test_*.py" -v
python reproduce.py run --output reproduced_results
python verify_results.py --reference reference_results --candidate reproduced_results --rtol 0 --atol 0
python reproduce.py plot --input reference_results --output replotted_figures
```

The numerical models and default parameters are unchanged from those used for the reference data. The unified entry point, numerical verification utility, distribution metadata, and reproducibility documentation accompany the original experiments. Plot labels use "Proposed method", and the figure map identifies Figures A.1–A.8. These presentation changes reuse the same numerical data.

The final distribution uses the neutral archive name `appendix_experiments.zip` and notebook name `appendix_all_figures.ipynb`. These file-name and documentation changes do not alter the numerical experiments.

The subplot titles in A.3 and A.5–A.8 describe the quantities displayed in each panel. A.3 and A.5 show completion times and speedups; A.6 shows prediction errors by scenario and versus trial size; A.7 and A.8 show completion times and effective work rates. The updated notebook was executed without resampling. Under the reference environment with the same Times New Roman font files selected, the notebook and script exports of these figures are pixel-identical. All 15 numerical files are unchanged by these title updates.

Font registration was also checked with Python 3.10.13 and Matplotlib 3.8.0 in a kernel with a stale font cache. A.3 exported successfully after automatically registering the installed Times New Roman fonts. The supplied figure PDFs embed Times New Roman. If several versions of this font are installed, select the same font files when comparing raster output pixel by pixel.

## Repository distribution

The experiment code, numerical reference data, figures, and executed notebook are unchanged in the repository distribution. Repository documentation and citation metadata have been added, and MANIFEST.sha256 has been refreshed.
