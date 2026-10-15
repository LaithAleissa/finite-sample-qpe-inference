"""Post-review plug-in multinomial adequacy diagnostic with full refitting."""
from studies import *
from fast_profile import batch_maximum

def deviance(counts,maximum):
    c=np.atleast_2d(counts);saturated=np.sum(xlogy(c,c/c.sum(axis=1)[:,None]),axis=1)
    return np.maximum(0,2*(saturated-np.asarray(maximum)))

def analyze(task):
    index,r=task;start=time.perf_counter();ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
    c=np.array(r['batch_a'])+r['batch_b'];fit=ProfileLikelihood(ex,r['batch_a'],r['batch_b'])
    observed=float(deviance(c,fit.maximum)[0]);q=(1-fit.eta)*ex.probabilities(fit.estimate)+fit.eta/2**r['n']
    rng=np.random.default_rng(np.random.SeedSequence([2026100506,index]));counts=rng.multinomial(int(c.sum()),q,size=999)
    statistics=[]
    for j in range(0,999,32):
        cc=counts[j:j+32];statistics.extend(deviance(cc,batch_maximum(ex,cc)).tolist())
    ge=sum(x>=observed for x in statistics);p=(1+ge)/1000
    return dict(case=r['case'],trial=r['trial'],fitted_energy=float(fit.estimate),fitted_eta=float(fit.eta),observed_deviance=observed,statistics=statistics,p_value=p,rejected=p<=.05,ge_count=ge,seed_index=index,seconds=time.perf_counter()-start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    raw=[r for r in read(ROOT/'results/circuit/raw_counts.json') if r['trial']<25];assert len(raw)==75
    out=ROOT/'results/adequacy';save(out/'raw_counts.json',raw)
    seal=dict(plan=digest(ROOT/'postreview_analysis_plan.md'),code=digest(Path(__file__)),profile=digest(SRC/'continuous_inversion/code/profile_lr.py'),batch_search=digest(ROOT/'code/fast_profile.py'))
    if (out/'seal.json').exists():assert read(out/'seal.json')==seal
    else:save(out/'seal.json',seal)
    cp=out/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
    tasks=[(i,r) for i,r in enumerate(raw) if not (cp/f'{i:03d}.json').exists()]
    save(out/'status.json',dict(status='running',records=75,resamples=74925))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (i,r),result in zip(tasks,pool.map(analyze,tasks,chunksize=1)):
            save(cp/f'{i:03d}.json',result);print('Adequacy',i+1,'/75 seconds',round(result['seconds']),flush=True)
    rr=[read(cp/f'{i:03d}.json') for i in range(75)];save(out/'analysis.json',rr);rows=[]
    for case in ('low','medium','high'):
        xx=[x for x in rr if x['case']==case];k=sum(x['rejected'] for x in xx);lo,hi=binomial_interval(k,25,.05/3)
        rows.append(dict(case=case,trials=25,rejected=k,mc_lower=lo,mc_upper=hi,median_p=float(np.median([x['p_value'] for x in xx])),median_deviance=float(np.median([x['observed_deviance'] for x in xx]))))
    csvsave(out/'summary.csv',rows);save(out/'status.json',dict(status='complete',records=75,resamples=74925))
    print(rows,flush=True)
if __name__=='__main__':main()
