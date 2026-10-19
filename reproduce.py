"""Run the published experiment settings, or redraw figures from saved results.

Examples:
    python reproduce.py run --output reproduced_results
    python reproduce.py run --group many-workers --output reproduced_results
    python reproduce.py plot --input reference_results --output replotted_figures
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
GROUPS = {
    'mechanisms': 'run_study.py',
    'many-workers': 'many_workers_study.py',
    'comparisons': 'allocation_comparison.py',
}


def run_experiments(output, groups):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    elapsed = {}
    for group in groups:
        script = SCRIPT_DIR / GROUPS[group]
        print(f'\n=== Running {group}: {script.name} ===', flush=True)
        before = time.perf_counter()
        subprocess.run([sys.executable, str(script), '--output', str(output)], check=True)
        elapsed[group] = time.perf_counter() - before
    import numpy
    import matplotlib
    source_files = ['models.py', *GROUPS.values(), 'reproduce.py']
    manifest = {
        'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'matplotlib': matplotlib.__version__,
        'groups': list(groups),
        'elapsed_seconds': elapsed,
        'source_sha256': {name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest() for name in source_files},
    }
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'\nAll selected experiments completed. Results: {output}', flush=True)


def plot_results(source, output, groups):
    import csv
    from run_study import make_figures
    from many_workers_study import make_figure
    from allocation_comparison import figures, combined_figure

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if 'mechanisms' in groups:
        data = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
        make_figures(output, data['minimum_rates'], data['history_variability'], data['rate_drift'], data['combined'])
    if 'many-workers' in groups:
        with (source / 'many_workers_histories.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in row:
                if key != 'model':
                    row[key] = float(row[key])
        make_figure(output, rows)
    if 'comparisons' in groups:
        data = json.loads((source / 'allocation_summary.json').read_text(encoding='utf-8'))
        figures(output, data['a'], data['b'], data['c'], data['d'])
        combined_figure(output, data['d_half_40'], 'fig10_half_workers_paired',
                        'Completion times', data['d_half_sphere'])
    print(f'Figures saved to {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run', help='run all simulations with the published parameters and seeds')
    run.add_argument('--output', type=Path, default=Path('reproduced_results'))
    plot = subparsers.add_parser('plot', help='redraw saved data without resampling')
    source = SCRIPT_DIR / 'reference_results'
    plot.add_argument('--input', type=Path, default=source if source.is_dir() else SCRIPT_DIR / 'results')
    plot.add_argument('--output', type=Path, default=Path('replotted_figures'))
    for subparser in (run, plot):
        subparser.add_argument('--group', choices=['all', *GROUPS], default='all')
    args = parser.parse_args()
    groups = tuple(GROUPS) if args.group == 'all' else (args.group,)
    if args.command == 'run':
        run_experiments(args.output, groups)
    else:
        plot_results(args.input, args.output, groups)


if __name__ == '__main__':
    main()
