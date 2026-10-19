"""Compare reproduced numerical results with the supplied reference results."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

NUMERICAL_FILES = (
    'minimum_rates.csv', 'history_variability.csv', 'rate_drift.csv',
    'combined.csv', 'summary.json', 'many_workers_histories.csv',
    'many_workers_metadata.json', 'allocation_a.csv', 'allocation_b.csv',
    'allocation_c.csv', 'allocation_d.csv', 'allocation_d_half_40.csv',
    'allocation_d_half_sphere.csv', 'allocation_summary.json',
    'allocation_a_paired_samples.npz',
)


def _csv_values(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return rows


def _compare(expected, actual, location, errors, rtol, atol):
    if len(errors) >= 50:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        # Runtime versions are provenance, not numerical experiment inputs.
        ignored = {'python', 'numpy', 'matplotlib'}
        keys = set(expected) - ignored
        if keys != set(actual) - ignored:
            errors.append(f'{location}: different field names')
        for key in sorted(keys & set(actual)):
            _compare(expected[key], actual[key], f'{location}.{key}', errors, rtol, atol)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f'{location}: expected {len(expected)} rows/items, got {len(actual)}')
        for i, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f'{location}[{i}]', errors, rtol, atol)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol)):
            errors.append(f'{location}: expected {expected!r}, got {actual!r}')
    elif expected != actual:
        errors.append(f'{location}: expected {expected!r}, got {actual!r}')


def compare_result_dirs(reference, candidate, *, rtol=1e-12, atol=1e-12):
    """Check every recognized reference file; return human-readable differences."""
    reference, candidate = Path(reference), Path(candidate)
    files = [reference / name for name in NUMERICAL_FILES if (reference / name).is_file()]
    if not files:
        return [f'{reference}: no reference numerical files found']
    errors = []
    for expected_path in files:
        actual_path = candidate / expected_path.name
        if not actual_path.is_file():
            errors.append(f'{expected_path.name}: missing candidate file')
            continue
        try:
            if expected_path.suffix == '.npz':
                with np.load(expected_path, allow_pickle=False) as left, np.load(actual_path, allow_pickle=False) as right:
                    if set(left.files) != set(right.files):
                        errors.append(f'{expected_path.name}: different array names')
                    for key in sorted(set(left.files) & set(right.files)):
                        a, b = left[key], right[key]
                        if a.shape != b.shape:
                            errors.append(f'{expected_path.name}.{key}: shape {a.shape} != {b.shape}')
                        elif not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                                  and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)):
                            errors.append(f'{expected_path.name}.{key}: array values differ')
            else:
                loader = _csv_values if expected_path.suffix == '.csv' else lambda p: json.loads(p.read_text(encoding='utf-8'))
                _compare(loader(expected_path), loader(actual_path), expected_path.name, errors, rtol, atol)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'{expected_path.name}: cannot compare ({exc})')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_reference = Path(__file__).resolve().parent / 'reference_results'
    if not default_reference.is_dir():
        default_reference = Path(__file__).resolve().parent / 'results'
    parser.add_argument('--reference', type=Path, default=default_reference)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--rtol', type=float, default=1e-12)
    parser.add_argument('--atol', type=float, default=1e-12)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
        parser.error('tolerances must be finite and nonnegative')
    errors = compare_result_dirs(args.reference, args.candidate, rtol=args.rtol, atol=args.atol)
    if errors:
        print('FAIL: numerical results differ')
        for error in errors:
            print('  ' + error)
        raise SystemExit(1)
    count = sum((args.reference / name).is_file() for name in NUMERICAL_FILES)
    print(f'PASS: {count} numerical files match (rtol={args.rtol:g}, atol={args.atol:g}).')


if __name__ == '__main__':
    main()
