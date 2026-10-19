"""Compare static allocations with the paper's aggregate-capacity prediction.

The proposed method's curve uses realized work and realized CPU rates.
No MPI implementation is emulated. All policies reuse the same history pool.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from models import capacity_at, completion_time, sample_work_sum
from run_study import plt, COLORS, CASE_NAMES, save_csv, _save

SPHERE_RATE_RATIO=3.343
SPHERE_DROP=1-1/SPHERE_RATE_RATIO


def aggregate_time(work, rates, drops, onset, ramp, mode='ramp'):
    rates=np.asarray(rates); drops=np.asarray(drops)
    # Common onset and transition: summing capacities gives the same profile
    # with aggregate base rate and rate-weighted fractional reduction.
    return float(completion_time(work,base_rate=rates.sum(),
        drop=np.dot(rates,drops)/rates.sum(),onset=onset,ramp=ramp,mode=mode))


def allocate_counts(total, weights):
    shares=total*np.asarray(weights)/np.sum(weights)
    counts=np.floor(shares).astype(int)
    remainder=int(total-counts.sum())
    counts[np.argsort(-(shares-counts),kind='stable')[:remainder]]+=1
    return counts


def allocation_times(work, rates, drops, pilots):
    """One uniform allocation, one retained-pilot static allocation per k.

    Each worker first tracks k histories. After all pilots finish, remaining
    histories are assigned in proportion to k / measured pilot time. Pool
    ordering is independent of unseen history costs; integer shares use
    largest remainders. All histories are completed once for each policy.
    """
    workers,histories=work.shape; reference=histories*25.
    profile=dict(base_rate=rates,drop=drops,onset=.3*reference,ramp=.25*reference)
    total=float(work.sum()); prefix=np.cumsum(work,axis=1)
    result=dict(work=total,reference=reference,
        theory=aggregate_time(total,rates,drops,.3*reference,.25*reference),
        nominal=float(completion_time(work.sum(axis=1),**profile).max()),
        pilot={},assigned_work={})
    for k in pilots:
        pilot_work=prefix[:,k-1]
        times=completion_time(pilot_work,**profile)
        start=float(times.max())
        remaining=work[:,k:].ravel()
        counts=allocate_counts(len(remaining),k/times)
        edges=np.r_[0,np.cumsum(counts)]
        cumulative=np.r_[0.,np.cumsum(remaining)]
        assigned=np.diff(cumulative[edges])
        result['assigned_work'][k]=assigned+pilot_work
        finishes=completion_time(capacity_at(start,**profile)+assigned,**profile)
        result['pilot'][k]=float(finishes.max())
    return result


def truncated_rates(rng, size, std):
    if std==0:return np.ones(size)
    rates=rng.normal(1,std,size)
    mask=(rates<.2)|(rates>1.8)
    while np.any(mask):
        rates[mask]=rng.normal(1,std,int(mask.sum()))
        mask=(rates<.2)|(rates>1.8)
    return rates


def summarize(samples, *, intervals=False, **labels):
    """Average paired time, actual throughput, and speedup across realizations."""
    a=np.asarray(samples) # nominal time, proposed time, actual work per worker
    static, theory, work=a.T
    result=dict(**labels,repeats=len(a),static_time=float(static.mean()),
        theory_time=float(theory.mean()),static_rate=float(np.mean(work/static)),
        theory_rate=float(np.mean(work/theory)),
        speedup_pct=float(100*np.mean(static/theory-1)),
        time_saved_pct=float(100*np.mean(1-theory/static)))
    if intervals:
        metrics=dict(static_time=static,theory_time=theory,
            static_rate=work/static,theory_rate=work/theory,
            speedup_pct=100*(static/theory-1))
        for key,values in metrics.items():
            result[key+'_p025'],result[key+'_p975']=map(float,np.quantile(values,[.025,.975]))
    return result


def study_a(seed, repeats=6000, *, sample_path=None, progress=False):
    counts=2**np.arange(21); rows=[]; paired=[]; stds=(.01,.05,.1,.2)
    for j,std in enumerate(stds):
        rng=np.random.default_rng(np.random.SeedSequence([seed,1,j])); values={int(m):[] for m in counts}
        for repeat in range(repeats):
            rates=truncated_rates(rng,int(counts[-1]),std)
            minima=np.minimum.accumulate(rates)[counts-1]
            means=np.cumsum(rates)[counts-1]/counts
            for m,low,mean in zip(counts,minima,means):values[int(m)].append((1/low,1/mean,1.))
            if progress and (repeat+1)%500==0:print(f'Case a: SD={std:g}, {repeat+1}/{repeats} paired realizations',flush=True)
        rows.extend(summarize(values[int(m)],intervals=True,parent_std=std,workers=int(m)) for m in counts)
        if sample_path is not None:paired.append(np.asarray([values[int(m)] for m in counts]))
    if sample_path is not None:
        samples=np.asarray(paired)
        np.savez_compressed(sample_path,seed=seed,workers=counts,parent_std=stds,
            static_time=samples[:,:,:,0],theory_time=samples[:,:,:,1])
    return rows


def study_b(seed, repeats=2000):
    counts=np.array([1,4,16,64,256,1024,4096]);rows=[]
    for model in ('poisson','mixed_poisson'):
      for h in (16,64,256):
        for std in (0.,.1):
            rng=np.random.default_rng(np.random.SeedSequence([seed,2,h,model=='mixed_poisson']))
            rng_rate=np.random.default_rng(np.random.SeedSequence([seed,3,h]))
            values={int(m):[] for m in counts}
            for start in range(0,repeats,64):
                size=(min(64,repeats-start),int(counts[-1]))
                work=sample_work_sum(rng,h,size,model)
                rates=truncated_rates(rng_rate,size,std)
                maxtime=np.maximum.accumulate(work/rates,axis=1)[:,counts-1]
                sums=np.cumsum(work,axis=1)[:,counts-1]
                totalrates=np.cumsum(rates,axis=1)[:,counts-1]
                for j,m in enumerate(counts):
                    # Normalize time by H*25; rates remain in work/time units.
                    values[int(m)].extend(zip(maxtime[:,j]/(25*h),sums[:,j]/totalrates[:,j]/(25*h),sums[:,j]/m/(25*h)))
            rows.extend(summarize(values[int(m)],intervals=True,model=model,histories=h,parent_std=std,workers=int(m)) for m in counts)
    return rows


def study_c():
    rows=[]
    for mode in ('step','ramp'):
      for drop in (.2,.4,.6,SPHERE_DROP):
        for fraction in (0.,.01,.05,.1,.25,.5,.75,1.):
            static=float(completion_time(1,drop=drop if fraction else 0,onset=.3,ramp=.25,mode=mode))
            theory=float(completion_time(1,drop=fraction*drop,onset=.3,ramp=.25,mode=mode))
            rows.append(dict(profile=mode,drop=drop,affected_fraction=fraction,
                nominal_time=static,pilot_time=static,theory_time=theory,
                nominal_rate=1/static,pilot_rate=1/static,theory_rate=1/theory,
                speedup_pct=100*(static/theory-1)))
    return rows


def study_d(seed, repeats=128, workers=256, histories=256, *, affected_fraction=.25, drop=.4, exact_fraction=False):
    if exact_fraction and not float(workers*affected_fraction).is_integer():
        raise ValueError('The exact affected fraction must give a whole number of workers.')
    pilots=[1,4,16,64,128,256];rows=[]
    for case in CASE_NAMES:
        rng_w=np.random.default_rng(np.random.SeedSequence([seed,4]));rng_r=np.random.default_rng(np.random.SeedSequence([seed,5]));rng_d=np.random.default_rng(np.random.SeedSequence([seed,6]))
        collected={k:[] for k in pilots};nominal=[]
        for _ in range(repeats):
            work=sample_work_sum(rng_w,1,(workers,histories),'mixed_poisson' if case in ('history_only','combined') else 'constant')
            rates=truncated_rates(rng_r,workers,.1 if case in ('node_only','combined') else 0.)
            drops=np.zeros(workers)
            if case in ('drift_only','combined'):
                if exact_fraction:
                    drops[rng_d.choice(workers,int(workers*affected_fraction),replace=False)]=drop
                else:
                    drops=drop*(rng_d.random(workers)<affected_fraction)
            result=allocation_times(work,rates,drops,pilots)
            ref=result['reference'];pw=result['work']/workers/ref
            nominal.append((result['nominal']/ref,result['theory']/ref,pw))
            for k in pilots:collected[k].append((result['pilot'][k]/ref,result['theory']/ref,pw))
        nr=summarize(nominal)
        for k in pilots:
            row=summarize(collected[k],case=case,workers=workers,histories=histories,pilot_histories=k,pilot_fraction=k/histories)
            row['pilot_time']=row.pop('static_time');row['pilot_rate']=row.pop('static_rate')
            row['nominal_time']=nr['static_time'];row['nominal_rate']=nr['static_rate'];row['speedup_over_nominal_pct']=nr['speedup_pct']
            rows.append(row)
    return rows


def figures(folder,a,b,c,d):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':14,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':12,'pdf.fonttype':42})
    def save(fig,axes,name,ncol):
        handles,labels=axes[0].get_legend_handles_labels()
        for ax in axes:
            legend=ax.get_legend()
            if legend is not None:legend.remove()
        fig.set_size_inches(10.2,5.0)
        fig.legend(handles,labels,loc='outside lower center',ncol=ncol,frameon=False,fontsize=12)
        _save(fig,folder,name)
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.0),layout='constrained')
    for col,std in zip(COLORS,(.01,.05,.1,.2)):
        rr=[r for r in a if r['parent_std']==std];x=[r['workers'] for r in rr]
        axes[0].plot(x,[r['static_rate'] for r in rr],color=col,label=rf'$\sigma$ = {std:g}')
        axes[0].plot(x,[r['theory_rate'] for r in rr],color=col,ls='--',lw=1.5)
        for method in ('static','theory'):
            axes[0].fill_between(x,[r[method+'_rate_p025'] for r in rr],[r[method+'_rate_p975'] for r in rr],color=col,alpha=.08)
        axes[1].plot(x,[r['speedup_pct'] for r in rr],color=col,label=rf'$\sigma$ = {std:g}')
        axes[1].fill_between(x,[r['speedup_pct_p025'] for r in rr],[r['speedup_pct_p975'] for r in rr],color=col,alpha=.10)
    axes[0].plot([],[],color='#555',ls='-',label='Nominal-frequency static')
    axes[0].plot([],[],color='#555',ls='--',label='Proposed method')
    axes[0].set(ylabel='Effective work rate per worker',title='(a) Effective processing rates')
    axes[1].set(ylabel='Speedup over static allocation (%)',title='(b) Speedup of proposed method')
    for ax in axes:ax.set(xscale='log',xlabel='Number of CPU workers');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,axes,'fig06_mean_minimum_comparison',3)

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    for col,h in zip(COLORS,(16,64,256)):
      for std,ls in ((0.,'-'),(.1,'--')):
        rr=[r for r in b if r['model']=='mixed_poisson' and r['histories']==h and r['parent_std']==std];x=[r['workers'] for r in rr]
        label=f'H = {h}, CPU '+rf'$\sigma$ = {std:g}'
        axes[0].plot(x,[r['static_time'] for r in rr],color=col,ls=ls,label=label)
        axes[1].plot(x,[r['speedup_pct'] for r in rr],color=col,ls=ls,label=label)
        axes[0].plot(x,[r['theory_time'] for r in rr],color=col,ls=':',lw=1)
        for method in ('static','theory'):
            axes[0].fill_between(x,[r[method+'_time_p025'] for r in rr],[r[method+'_time_p975'] for r in rr],color=col,alpha=.06)
        axes[1].fill_between(x,[r['speedup_pct_p025'] for r in rr],[r['speedup_pct_p975'] for r in rr],color=col,alpha=.06)
    axes[0].plot([],[],color='#555',ls=':',label='Proposed method')
    axes[0].set(ylabel='Completion time / (25 H)',title='(a) Completion times')
    axes[1].set(ylabel='Proposed speedup over static (%)',title='(b) Speedups')
    for ax in axes:ax.set(xscale='log',xlabel='Number of CPU workers');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,axes,'fig07_history_rate_comparison',2)

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    for col,mode in zip(COLORS,('step','ramp')):
        rr=[r for r in c if r['profile']==mode and r['drop']==.4];x=[100*r['affected_fraction'] for r in rr]
        axes[0].plot(x,[r['nominal_time'] for r in rr],color=col,ls='--',label=f'{mode}: nominal / early pilot')
        axes[0].plot(x,[r['theory_time'] for r in rr],color=col,label=f'{mode}: proposed method')
        axes[1].plot(x,[r['speedup_pct'] for r in rr],color=col,label=mode)
    axes[0].set(ylabel='Normalized completion time',title='(a) Completion times')
    axes[1].set(ylabel='Proposed speedup over static (%)',title='(b) Speedups')
    for ax in axes:ax.set(xlabel='Affected CPU workers (%)');ax.legend(frameon=False);ax.grid(alpha=.15)
    save(fig,axes,'fig08_drift_allocation_comparison',2)

    combined_figure(folder,d,'fig09_static_theory_comparison','Completion times')


def combined_figure(folder, rows, filename, title, comparison=None):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':14,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.0),layout='constrained')
    rr=[r for r in rows if r['case']=='combined'];x=[100*r['pilot_fraction'] for r in rr]
    for key,col,label in [('nominal',COLORS[0],'Nominal-frequency static'),('pilot',COLORS[1],'Retained-pilot static'),('theory',COLORS[2],'Proposed method')]:
        axes[0].plot(x,[r[key+'_time'] for r in rr],color=col,marker='o',ms=4,label=label)
        axes[1].plot(x,[r[key+'_rate'] for r in rr],color=col,marker='o',ms=4,label=label)
        if comparison is not None:
            other=[r for r in comparison if r['case']=='combined']
            axes[0].plot(x,[r[key+'_time'] for r in other],color=col,ls='--',marker='s',ms=4)
            axes[1].plot(x,[r[key+'_rate'] for r in other],color=col,ls='--',marker='s',ms=4)
    axes[0].set(ylabel='Completion time / (256 × 25)',title='(a) '+title)
    axes[1].set(ylabel='Effective work rate per worker',title='(b) Effective work rates')
    for ax in axes:ax.set(xscale='log',xlabel='Pilot histories / total histories (%)');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    if comparison is not None:
        from matplotlib.lines import Line2D
        handles += [Line2D([],[],color='#555',ls='-',marker='o'),Line2D([],[],color='#555',ls='--',marker='s')]
        labels += ['40% reduction','70.09% reduction (3.343:1)']
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=11 if comparison is not None else 12)
    _save(fig,folder,filename)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results');parser.add_argument('--seed',type=int,default=20261018);args=parser.parse_args()
    folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    a=study_a(args.seed,sample_path=folder/'allocation_a_paired_samples.npz',progress=True);print('Case a complete',flush=True)
    b=study_b(args.seed);print('Case b complete',flush=True)
    c=study_c();d=study_d(args.seed)
    d_half_40=study_d(args.seed,affected_fraction=.5,drop=.4,exact_fraction=True)
    d_half_sphere=study_d(args.seed,affected_fraction=.5,drop=SPHERE_DROP,exact_fraction=True)
    print('Cases c and d complete, including paired reductions on exactly half the workers',flush=True)
    data=dict(a=a,b=b,c=c,d=d,d_half_40=d_half_40,d_half_sphere=d_half_sphere)
    for key,rows in data.items():save_csv(folder/f'allocation_{key}.csv',rows)
    (folder/'allocation_summary.json').write_text(json.dumps({'seed':args.seed,'definitions':{'theory':'integrated aggregate capacity equals realized total workload','rate':'realized total workload / (worker count * completion time)','pilot':'retained, globally synchronized; remaining counts weighted by k / pilot time','speedup':'mean of paired static/theory time ratios minus one','paired_half':'exactly the same 128 of 256 workers selected without replacement in each pair; reductions of 0.4 and 1-1/3.343; same realized histories, initial rates, onset and ramp duration'},**data},indent=2)+'\n')
    figures(folder,a,b,c,d)
    combined_figure(folder,d_half_40,'fig10_half_workers_paired','Completion times',d_half_sphere)
    for key,rows in data.items():
      for r in rows:
        if (key=='a' and r['parent_std'] in (.01,.05) and r['workers'] in (128,1048576)) or (key=='b' and r['model']=='mixed_poisson' and r['histories']==256 and r['workers'] in (256,4096)) or (key=='c' and r['drop']==.4 and r['affected_fraction']==.25) or (key=='d' and r['pilot_histories'] in (16,64,128)):
            print(key,json.dumps(r),flush=True)

if __name__=='__main__':main()
