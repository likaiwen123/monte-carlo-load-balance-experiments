"""Run four reproducible mechanism experiments and export publication figures."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'cpc-matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Apply Times New Roman to ordinary text and logarithmic tick math alike.
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'custom',
    'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
    'mathtext.bf':'Times New Roman:bold','mathtext.bfit':'Times New Roman:bold:italic',
    'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman',
    'mathtext.cal':'Times New Roman','mathtext.fallback':None})

from models import (workload_moments, sample_work_sum, prefix_workloads,
                    minimum_rate_samples, rate_at, capacity_at,
                    completion_time, prediction_error)

COLORS = ['#246a9d', '#d07b27', '#31816b', '#8c558c']
CASE_NAMES = {'node_only':'Node differences', 'history_only':'History variability',
              'drift_only':'Rate drift', 'combined':'Combined'}


def study_minimum(seed, repeats=6000, workers=None):
    workers = list(workers if workers is not None else 2**np.arange(21))
    rows = []
    for index, cv in enumerate((.01, .05, .1, .2)):
        for count in workers:
            # Reset the stream across worker counts for coupled quantile samples.
            values = minimum_rate_samples(np.random.default_rng(np.random.SeedSequence([seed, 1, index])),
                                          int(count), cv, repeats)
            rows.append(dict(parent_std=cv, workers=int(count), repeats=repeats,
                mean_minimum=float(values.mean()), p025=float(np.quantile(values, .025)),
                p975=float(np.quantile(values, .975)),
                standard_deviation=float(values.std(ddof=1))))
    return rows


def study_histories(seed, repeats=12000, counts=None):
    counts = list(counts if counts is not None else 2**np.arange(11))
    rows = []
    for index, model in enumerate(('poisson', 'mixed_poisson')):
        mean, variance = workload_moments(model)
        for count in counts:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2, index, int(count)]))
            total = sample_work_sum(rng, int(count), repeats, model)
            estimate = count*mean/total
            rows.append(dict(model=model, histories=int(count), repeats=repeats,
                mean_work=mean, work_variance=variance,
                empirical_time_cv=float(total.std(ddof=1)/total.mean()),
                analytic_time_cv=float(np.sqrt(variance)/(mean*np.sqrt(count))),
                speed_median=float(np.median(estimate)),
                speed_p025=float(np.quantile(estimate, .025)),
                speed_p975=float(np.quantile(estimate, .975))))
    return rows


def study_drift():
    rows = []
    for mode in ('step', 'ramp'):
        for drop in (.2, .4, .6):
            finish = float(completion_time(1., drop=drop, onset=.3, ramp=.25, mode=mode))
            rows.append(dict(profile=mode, drop=drop, onset=.3, transition=.25 if mode=='ramp' else 0.,
                work=1., predicted_time=1., completion_time=finish,
                time_increase_pct=100*(finish-1)))
    return rows


def study_combined(seed, repeats=128, workers=256, histories=256, counts=None):
    counts = list(counts if counts is not None else [1, 4, 16, 64, 128, histories])
    counts = sorted(set(counts))
    if not counts or counts[0] < 1 or counts[-1] != histories:
        raise ValueError('counts must end at the complete history count')
    shape = (repeats, workers)
    mean, _ = workload_moments('poisson')
    reference_time = histories*mean
    rows = []
    for case_index, case in enumerate(CASE_NAMES):
        node_variation = case in ('node_only', 'combined')
        history_variation = case in ('history_only', 'combined')
        thermal = case in ('drift_only', 'combined')
        base = (minimum_rate_samples(np.random.default_rng(np.random.SeedSequence([seed, 3])),
                                     1, .1, shape) if node_variation else np.ones(shape))
        # The same seeded affected-node masks are used for both thermal cases.
        affected = np.random.default_rng(np.random.SeedSequence([seed, 4])).random(shape) < .25
        drops = .4*affected if thermal else np.zeros(shape)
        model = 'mixed_poisson' if history_variation else 'constant'
        work = prefix_workloads(np.random.default_rng(np.random.SeedSequence([seed, 5])),
                                counts, shape, model)
        profile = dict(base_rate=base, drop=drops, onset=.3*reference_time,
                       ramp=.25*reference_time, mode='ramp')
        true_time = completion_time(work[histories], **profile)
        nominal_error = np.abs(reference_time/true_time-1)
        for count in counts:
            _, _, error = prediction_error(work[histories], work[count], histories, count, **profile)
            absolute = np.abs(error)
            rows.append(dict(case=case, workers=workers, repeats=repeats,
                total_histories=histories, pilot_histories=count, pilot_fraction=count/histories,
                mean_abs_error_pct=float(100*absolute.mean()),
                p90_abs_error_pct=float(100*np.quantile(absolute, .9)),
                median_signed_error_pct=float(100*np.median(error)),
                nominal_mean_abs_error_pct=float(100*nominal_error.mean()),
                mean_affected_abs_error_pct=float(100*absolute[affected].mean()) if thermal else None))
    return rows


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, folder, name):
    fig.savefig(folder/f'{name}.png', dpi=220, facecolor='white')
    fig.savefig(folder/f'{name}.pdf', facecolor='white', metadata={'CreationDate':None, 'ModDate':None})
    plt.close(fig)


def make_figures(folder, minimum, histories, drift, combined):
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.labelsize':10, 'legend.fontsize':9, 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.3, 4.3), layout='constrained')
    for color, cv in zip(COLORS, (.01, .05, .1, .2)):
        rows = [r for r in minimum if r['parent_std']==cv]
        x = [r['workers'] for r in rows]
        ax.plot(x, [r['mean_minimum'] for r in rows], color=color, label=rf'Parent $\sigma/\mu$ = {cv:g}')
        ax.fill_between(x, [r['p025'] for r in rows], [r['p975'] for r in rows], color=color, alpha=.10)
    ax.axhline(1, color='#555555', ls='--', lw=1, label='Identical nominal frequency')
    ax.set(xscale='log', xlabel='Number of CPU workers', ylabel='Minimum effective rate / nominal rate', ylim=(.15,1.43))
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=.15)
    _save(fig, folder, 'fig01_minimum_rate')

    fig, axes = plt.subplots(1, 2, figsize=(10.2,4.0), layout='constrained')
    for color, model, label in zip(COLORS, ('poisson','mixed_poisson'), ('Poisson','Mixed Poisson')):
        rows = [r for r in histories if r['model']==model];x=[r['histories'] for r in rows]
        axes[0].loglog(x, [r['analytic_time_cv'] for r in rows],color=color,label=label)
        axes[0].scatter(x, [r['empirical_time_cv'] for r in rows],color=color,s=18)
        axes[1].plot(x, [r['speed_median'] for r in rows],color=color,label=label)
        axes[1].fill_between(x,[r['speed_p025'] for r in rows],[r['speed_p975'] for r in rows],color=color,alpha=.16)
    axes[0].set(xlabel='Histories per worker',ylabel='Relative standard deviation of total time',title='(a) Workload variability')
    axes[0].text(.03,.04,'Lines: analytic   Points: simulation',transform=axes[0].transAxes,fontsize=8)
    axes[1].set(xscale='log',xlabel='Pilot histories',ylabel='Estimated rate / actual rate',title='(b) Pilot rate estimate (median and 95% range)')
    axes[1].axhline(1,color='#555555',ls='--',lw=1)
    for ax in axes:
        ax.legend(frameon=False);ax.grid(alpha=.15)
    _save(fig,folder,'fig02_history_variability')

    fig, axes = plt.subplots(1,2,figsize=(10.2,4.0),layout='constrained')
    time = np.linspace(0,1.8,1001)
    for color, mode, label in zip(COLORS, ('step','ramp'), ('Step decrease','Gradual decrease')):
        profile=dict(drop=.4,onset=.3,ramp=.25,mode=mode)
        axes[0].plot(time,rate_at(time,**profile),color=color,label=label)
        axes[1].plot(time,capacity_at(time,**profile),color=color,label=label)
        finish=float(completion_time(1,**profile));axes[1].plot([finish],[1],marker='o',color=color)
    axes[0].set(xlabel='Time / initial predicted completion time',ylabel='Effective rate / initial rate',title='(a) 40% rate decrease',ylim=(.5,1.08))
    axes[1].plot(time,time,color='#555555',ls='--',label='Initial-rate forecast')
    axes[1].axhline(1,color='#888888',ls=':',lw=1)
    axes[1].set(xlabel='Time / initial predicted completion time',ylabel='Cumulative completed work / target',title='(b) Accumulated processing capacity')
    for ax in axes:
        ax.legend(frameon=False);ax.grid(alpha=.15)
    _save(fig,folder,'fig03_rate_drift')

    fig,axes=plt.subplots(1,2,figsize=(10.2,4.2),layout='constrained')
    labels=list(CASE_NAMES.values());x=np.arange(len(labels));barwidth=.36
    selected=[next(r for r in combined if r['case']==case and r['pilot_histories']==16) for case in CASE_NAMES]
    axes[0].bar(x-barwidth/2,[r['nominal_mean_abs_error_pct'] for r in selected],barwidth,color=COLORS[0],label='Nominal-rate forecast')
    axes[0].bar(x+barwidth/2,[r['mean_abs_error_pct'] for r in selected],barwidth,color=COLORS[1],label='16-history pilot forecast')
    axes[0].set_xticks(x,labels,rotation=18,ha='right')
    axes[0].set(ylabel='Mean absolute completion-time prediction error (%)',title='(a) Prediction errors by scenario')
    axes[0].set_ylim(0,1.30*max(r['mean_abs_error_pct'] for r in selected))
    axes[0].legend(frameon=False)
    for color,case in zip(COLORS,CASE_NAMES):
        rows=[r for r in combined if r['case']==case]
        axes[1].plot([100*r['pilot_fraction'] for r in rows],[r['mean_abs_error_pct'] for r in rows],marker='o',ms=4,color=color,label=CASE_NAMES[case])
    axes[1].set(xscale='log',xlabel='Pilot histories / total histories (%)',ylabel='Mean absolute prediction error (%)',title='(b) Prediction errors versus trial size')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    _save(fig,folder,'fig04_combined')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    minimum=study_minimum(args.seed);histories=study_histories(args.seed)
    drift=study_drift();combined=study_combined(args.seed)
    data={'minimum_rates':minimum,'history_variability':histories,'rate_drift':drift,'combined':combined}
    for name,rows in data.items():save_csv(args.output/f'{name}.csv',rows)
    metadata={'seed':args.seed,'numpy':np.__version__,'matplotlib':matplotlib.__version__,
        'python':platform.python_version(),'minimum_repeats':6000,'history_repeats':12000,
        'combined_repeats':128,'combined_workers':256,'histories_per_worker':256,
        'poisson_mean':20,'overhead_work':5,'mixture_probabilities':[.9,.1],
        'mixture_intensities':[4,164],'rate_bounds':[.2,1.8],
        'combined_node_parent_std':.1,'combined_affected_probability':.25,
        'combined_rate_drop':.4,'normalized_thermal_onset':.3,'normalized_ramp_duration':.25}
    (args.output/'summary.json').write_text(json.dumps({'metadata':metadata,**data},indent=2)+'\n')
    make_figures(args.output,minimum,histories,drift,combined)
    print(f'Wrote {sum(len(rows) for rows in data.values())} summary rows and four figures to {args.output}')


if __name__=='__main__':
    main()
