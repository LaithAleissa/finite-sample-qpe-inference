"""Candidate-wise bootstrap diagnostic: full refitting, no interval inversion."""
from studies import *
from fast_profile import batch_profile,batch_maximum

def bootstrap_record(task):
    index,r=task;ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
    c=np.array(r['batch_a'])+r['batch_b'];candidate=r['energy']
    t=time.perf_counter();obs=ProfileLikelihood(ex,r['batch_a'],r['batch_b'])
    null,_,eta=profile_bounds(ex.probabilities(candidate),c);stat=max(0.,2*(obs.maximum-null[0]))
    q=(1-eta[0])*ex.probabilities(candidate)+eta[0]/2**r['n']
    rng=np.random.default_rng(np.random.SeedSequence([2026100501,index]));counts=rng.multinomial(int(sum(c)),q,size=999)
    statistics=[]
    # The pooled PLR depends only on the sum; preserve nonempty integer batches.
    for start in range(0,999,32):
        cc=counts[start:start+32]
        maximum=batch_maximum(ex,cc)
        constrained=batch_profile(np.tile(ex.probabilities(candidate),(len(cc),1)),cc)[0]
        statistics.extend(np.maximum(0,2*(maximum-constrained)).tolist())
    ge=int(sum(x>=stat for x in statistics));p=(1+ge)/1000
    return dict(case=r['case'],trial=r['trial'],candidate=candidate,eta_generating=r['eta'],shots_per_batch=r['shots_per_batch'],null_eta=float(eta[0]),observed_statistic=float(stat),bootstrap_replicates=999,ge_count=ge,p_value=p,accepted=p>.05,conventional_accepted=bool(stat<=obs.cutoff),seconds=time.perf_counter()-t,seed_index=index,statistics=statistics)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    records=[r for r in read(SRC/'continuous_inversion/results/raw_counts.json') if r['group']=='transition' and r['eta']>=.9 and r['trial']<25]
    if args.benchmark:
        r=records[0];ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));start=time.perf_counter()
        for i in range(10):ProfileLikelihood(ex,r['batch_a'],r['batch_b'])
        print('seconds per fit',(time.perf_counter()-start)/10);return
    out=ROOT/'results/bootstrap';save(out/'raw_counts.json',records)
    seal=dict(protocol=digest(ROOT/'analysis_amendment_2026-10-05.md'),bootstrap=digest(Path(__file__)),profile=digest(SRC/'continuous_inversion/code/profile_lr.py'),batch_search=digest(ROOT/'code/fast_profile.py'))
    if (out/'seal.json').exists():
        oldseal=read(out/'seal.json')
        if oldseal!=seal:
            approved=read(ROOT/'qa/bootstrap_initial_seal.json')
            assert oldseal==approved and {k:v for k,v in oldseal.items() if k not in ('batch_search','bootstrap')}=={k:v for k,v in seal.items() if k not in ('batch_search','bootstrap')},'Unapproved source change'
            assert read(ROOT/'qa/bootstrap_acceleration_check.json')['status']=='passed'
            save(out/'seal_pre_acceleration.json',oldseal);save(out/'seal.json',seal)
    else:save(out/'seal.json',seal)
    cp=out/'checkpoints';cp.mkdir(parents=True,exist_ok=True);tasks=[(i,r) for i,r in enumerate(records) if not (cp/f'{i:03d}.json').exists()]
    save(out/'status.json',dict(status='running',records=len(records),resamples=199800))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (i,r),result in zip(tasks,pool.map(bootstrap_record,tasks,chunksize=1)):
            save(cp/f'{i:03d}.json',result);print('bootstrap',i+1,'/',len(records),'seconds',round(result['seconds']),flush=True)
    result=[read(cp/f'{i:03d}.json') for i in range(len(records))];save(out/'analysis.json',result)
    rows=[]
    for case in dict.fromkeys(r['case'] for r in records):
        rr=[r for r in result if r['case']==case];n=len(rr)
        for name,key in [('bootstrap','accepted'),('conventional','conventional_accepted')]:
            count=sum(r[key] for r in rr);lo,hi=binomial_interval(count,n,.05/16)
            rows.append(dict(case=case,method=name,trials=n,accepted=count,mc_lower=lo,mc_upper=hi,eta=rr[0]['eta_generating'],shots_per_batch=rr[0]['shots_per_batch'],median_seconds=float(np.median([r['seconds'] for r in rr]))))
    csvsave(out/'summary.csv',rows);save(out/'status.json',dict(status='complete',records=len(records),resamples=199800,scope='Candidate-wise diagnostic only; no bootstrap interval lengths'))
if __name__=='__main__':main()
