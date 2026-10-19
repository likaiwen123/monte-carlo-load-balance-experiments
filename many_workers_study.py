"""Finite-history completion times on identical, constant-speed CPU workers.

Each worker receives H IID histories. Record max_i T_i / (H E[W]) at
unit speed; no scheduler or dynamic-method speedup is simulated here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from models import sample_work_sum, workload_moments
from run_study import save_csv, _save, COLORS, plt


def study_many_workers(seed=20261016, repeats=2000, workers=None, counts=None, models=None):
    workers=list(workers if workers is not None else [1,4,16,64,256,1024,4096])
    counts=list(counts if counts is not None else [16,64,256])
    models=list(models if models is not None else ['poisson','mixed_poisson'])
    rows=[]
    for model in models:
        model_id={'constant':0,'poisson':1,'mixed_poisson':2}[model]
        mean,_=workload_moments(model)
        for count in counts:
            rng=np.random.default_rng(np.random.SeedSequence([seed,6,model_id,count]))
            maxima=np.empty((repeats,len(workers)))
            # Batches limit memory; worker prefixes couple the M values.
            for start in range(0,repeats,128):
                size=min(128,repeats-start)
                total=sample_work_sum(rng,count,(size,max(workers)),model)
                cumulative_max=np.maximum.accumulate(total,axis=1)/(count*mean)
                maxima[start:start+size]=cumulative_max[:,np.array(workers)-1]
            for j,number in enumerate(workers):
                values=maxima[:,j]
                rows.append(dict(model=model,histories=count,workers=number,repeats=repeats,
                                 mean_maximum=float(values.mean()),
                                 p025=float(np.quantile(values,.025)),
                                 median=float(np.median(values)),
                                 p975=float(np.quantile(values,.975))))
    return rows


def make_figure(folder,rows):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,
                        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
    for ax,model,title in zip(axes,['poisson','mixed_poisson'],['(a) Poisson histories','(b) Mixed-Poisson histories']):
        for color,count in zip(COLORS,[16,64,256]):
            group=[r for r in rows if r['model']==model and r['histories']==count]
            x=[r['workers'] for r in group]
            ax.plot(x,[r['mean_maximum'] for r in group],color=color,label=f'{count} histories / worker')
            ax.fill_between(x,[r['p025'] for r in group],[r['p975'] for r in group],color=color,alpha=.1)
        ax.axhline(1,color='#666666',ls='--',lw=1)
        ax.set(xscale='log',xlabel='Number of identical CPU workers',
               ylabel='Maximum completion time / mean single-worker time',title=title)
        ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    _save(fig,folder,'fig05_many_workers_histories')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=study_many_workers()
    save_csv(args.output/'many_workers_histories.csv',rows)
    (args.output/'many_workers_metadata.json').write_text(json.dumps({
        'seed':20261016,'repeats':2000,'rate':1,'histories_per_worker':[16,64,256],
        'workers':[1,4,16,64,256,1024,4096],
        'normalization':'max_i T_i / (H E[W]); E[W] = 25',
        'history_models':'Same models as models.py; W = 5 + K',
        'numpy':np.__version__},indent=2)+'\n')
    make_figure(args.output,rows)
    print(f'Wrote {len(rows)} rows and figure 5 to {args.output}')

if __name__=='__main__':main()
