"""Matched primary unknown-noise DIR comparison, separately sealed."""
from valuable_followup import *

def run(record):
    ex=Experiment(record['n'],record.get('time',.8377580409572781),tuple(record.get('bounds',[-3.75,3.75])))
    start=time.perf_counter();obj=UnknownDirichlet(ex,record['batch_a'],record['batch_b'],.05);cs=obj.confidence_set();point,reject=obj.membership(record['energy']);assert point or reject
    inside=any(lo<=record['energy']<=hi for lo,hi in cs['intervals']);assert not point or inside
    return dict(record=record,set=cs,point_covered=point,set_covered=inside,log_predictive=obj.log_predictive,seconds=time.perf_counter()-start)

def main():
    out=ROOT/'results/primary_dirichlet'
    seal={p.relative_to(ROOT).as_posix():digest(p) for p in [ROOT/'primary_dirichlet_plan.md',Path(__file__),ROOT/'code/valuable_followup.py',ROOT/'code/methods.py',ROOT/'results/mixture/raw_counts.json']}
    if (out/'seal.json').exists():assert read(out/'seal.json')==seal
    else:save(out/'seal.json',seal)
    rows=[r for r in read(ROOT/'results/mixture/raw_counts.json') if r['group']=='paired'];cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    tasks=[(i,r) for i,r in enumerate(rows) if not (cp/f'{i:04d}.json').exists()]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(run,r):i for i,r in tasks}
        for j,f in enumerate(as_completed(fs),1):
            save(cp/f'{fs[f]:04d}.json',f.result())
            if j%100==0:print('Primary DIR',j,'/',len(tasks),flush=True)
    states=[read(cp/f'{i:04d}.json') for i in range(len(rows))];save(out/'states.json',states)
    qpm={(s['record']['case'],s['record']['trial']):s for s in read(ROOT/'results/mixture/analysis.json')}
    cases=list(dict.fromkeys(r['case'] for r in rows));summary=[]
    for case in cases:
        ss=[s for s in states if s['record']['case']==case];rr=[qpm[(case,s['record']['trial'])] for s in ss]
        lengths=np.array([sum(hi-lo for lo,hi in s['set']['intervals']) for s in ss]);regret=[r['fits']['qpe_mixture']['log_predictive']-s['log_predictive'] for s,r in zip(ss,rr)]
        k=sum(s['point_covered'] for s in ss);lo,hi=binomial_interval(k,len(ss),.05/len(cases))
        summary.append(dict(case=case,n=ss[0]['record']['n'],eta=ss[0]['record']['eta'],trials=len(ss),point_covered=k,set_covered=sum(s['set_covered'] for s in ss),mc_lower=lo,mc_upper=hi,median_length=float(np.median(lengths)),qpm_median_length=float(np.median([r['metrics']['qpe_mixture']['length'] for r in rr])),median_qpm_minus_dir_log_predictive=float(np.median(regret)),full=sum(s['set']['intervals']==[[-3.75,3.75]] for s in ss),empty=sum(not s['set']['intervals'] for s in ss),disconnected=sum(len(s['set']['intervals'])>1 for s in ss),median_seconds=float(np.median([s['seconds'] for s in ss]))))
    save(out/'summary.json',summary);save(out/'status.json',dict(status='complete',state_datasets=len(states),designs=len(cases)))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
