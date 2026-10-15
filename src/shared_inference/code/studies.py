"""Checkpointed frozen studies. Raw count files precede inference."""
from methods import *
import json,csv,time,hashlib,argparse,os
from concurrent.futures import ProcessPoolExecutor

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def csvsave(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sealed_sources():
    return {str(p.relative_to(SRC)):digest(p) for folder in ('shared_inference/code','continuous_inversion/code','predictive_inference/code','known_noise','likelihood_baseline/code') for p in (SRC/folder).glob('*.py')}

def evaluate(task):
    r,study=task;ex=Experiment(r['n'],r.get('time',.8377580409572781),tuple(r.get('bounds',[-3.75,3.75])),r.get('assumed_uniform_noise',0))
    truth=r.get('energy',r.get('truth_for_validation_only'))
    result=dict(record=r,metrics={},sets={},fits={})
    classes=[('qpe_mixture',QPEMixture)] if study=='mixture' else ([('crossfit',Inference),('lr',LikelihoodRatio),('dirichlet',MixtureInference)] if study=='known' else [('crossfit',RefinedJoint),('qpe_mixture',QPEMixture),('profile',RefinedProfile)])
    if study=='circuit':classes=classes[:2]
    for name,cls in classes:
        t=time.perf_counter();obj=cls(ex,r['batch_a'],r['batch_b'])
        cs=known_invert(obj) if study=='known' else obj.confidence_set()
        if name=='crossfit' and study!='known':
            lo,hi=obj.projection_membership_bounds(truth);point=bool(hi[0]<=-np.log(.05));reject=bool(lo[0]>-np.log(.05))
            fit=dict(fit_a=list(obj.fit_a),fit_b=list(obj.fit_b),numerator_log_a=obj.na,numerator_log_b=obj.nb)
        elif name in ('profile','qpe_mixture'):
            point,reject=obj.membership(truth)
            fit=dict(log_predictive=obj.log_predictive,threshold=obj.threshold) if name=='qpe_mixture' else dict(energy=obj.estimate,eta=obj.eta,maximum=obj.maximum,threshold=obj.threshold)
        else:point=obj.contains(truth);reject=not point;fit={}
        if not(point or reject):raise RuntimeError(f'Ambiguous truth decision {r["case"]}/{r["trial"]}/{name}')
        inside=any(a<=truth<=b for a,b in cs['intervals']);assert not point or inside
        result['metrics'][name]=dict(point_covered=point,set_covered=inside,length=length(cs),components=len(cs['intervals']),full_branch=cs['intervals']==[list(ex.bounds)],empty=not cs['intervals'],seconds=time.perf_counter()-t,cells=cs['visited_cells'])
        result['sets'][name]=cs;result['fits'][name]=fit
    return result

def fresh_records():
    cases=[('interior',1.013,.1,512),('seam',-3.7499,.1,512),('near_uniform',1.013,.98,128),('uniform',1.013,1.,128)]
    rows=[];ex=Experiment(4,.8377580409572781,(-3.75,3.75))
    for i,(name,e,eta,shots) in enumerate(cases):
        rng=np.random.default_rng(np.random.SeedSequence([2026100503,i]));q=(1-eta)*ex.probabilities(e)+eta/16
        for trial in range(500):
            a,b=rng.multinomial(shots,q,size=2)
            rows.append(dict(case=name,group='replication',trial=trial,n=4,energy=e,eta=eta,shots_per_batch=shots,mechanism='uniform',batch_a=a.tolist(),batch_b=b.tolist()))
    return rows

def inputs(study):
    if study=='mixture':return [r for r in read(SRC/'continuous_inversion/results/raw_counts.json') if r['group']!='register']
    if study=='known':return [json.loads(s) for s in (SRC/'known_noise/results/raw_counts.jsonl').read_text().splitlines() if json.loads(s)['stage']=='coverage']
    if study=='replication':return fresh_records()
    if study=='circuit':return read(ROOT/'results/circuit/raw_counts.json')

def summarize(items,study):
    summaries=[];rng=np.random.default_rng(2026100504)
    for case in dict.fromkeys(x['record']['case'] for x in items):
        rr=[x for x in items if x['record']['case']==case];r=rr[0]['record'];group=r.get('group',study)
        methods=list(rr[0]['metrics']);family=len(methods)*len({x['record']['case'] for x in items if x['record'].get('group',study)==group})
        for name in methods:
            mm=[x['metrics'][name] for x in rr];v=np.array([x['length'] for x in mm]);point=sum(x['point_covered'] for x in mm);inside=sum(x['set_covered'] for x in mm)
            lo,hi=binomial_interval(point,len(mm),.05/family)
            s=dict(case=case,group=group,method=name,n=r['n'],energy=r.get('energy',r.get('truth_for_validation_only')),eta=r.get('eta',r.get('assumed_uniform_noise')),shots_per_batch=r.get('shots_per_batch',sum(r['batch_a'])),trials=len(mm),point_covered=point,set_covered=inside,mc_lower=lo,mc_upper=hi,median_length=float(np.median(v)),mean_length=float(v.mean()),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),full_branch=sum(x['full_branch'] for x in mm),empty=sum(x['empty'] for x in mm),disconnected=sum(x['components']>1 for x in mm),median_components=float(np.median([x['components'] for x in mm])),median_seconds=float(np.median([x['seconds'] for x in mm])),median_cells=float(np.median([x['cells'] for x in mm])))
            summaries.append(s)
    return summaries

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=['mixture','known','replication','circuit']);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=ROOT/'results'/args.study;raw=inputs(args.study)
    if (out/'raw_counts.json').exists():assert read(out/'raw_counts.json')==raw
    else:save(out/'raw_counts.json',raw)
    # Only numerical dependencies seal checkpoints, not unrelated authoring scripts.
    seal={str(p):digest(p) for p in [ROOT/'analysis_amendment_2026-10-05.md',ROOT/'code/methods.py',ROOT/'code/studies.py',SRC/'continuous_inversion/code/refined_inversion.py',SRC/'continuous_inversion/code/profile_lr.py',SRC/'predictive_inference/code/predictive.py',SRC/'known_noise/inference.py',SRC/'likelihood_baseline/code/baseline.py']}
    if (out/'seal.json').exists():assert read(out/'seal.json')==seal,'Source changed; use explicit correction/rerun'
    else:save(out/'seal.json',seal)
    checkpoint=out/'checkpoints';checkpoint.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for i,r in enumerate(raw):
        path=checkpoint/f'{i:05d}.json'
        if path.exists():assert read(path)['record']==r
        else:tasks.append((i,r))
    save(out/'status.json',dict(status='running',records=len(raw),remaining=len(tasks),raw_sha256=digest(out/'raw_counts.json')))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (i,r),result in zip(tasks,pool.map(evaluate,[(r,args.study) for i,r in tasks],chunksize=4)):
            save(checkpoint/f'{i:05d}.json',result)
            if (i+1)%100==0:print(args.study,i+1,'/',len(raw),'elapsed',round(time.perf_counter()-start),flush=True)
    items=[read(checkpoint/f'{i:05d}.json') for i in range(len(raw))]
    save(out/'analysis.json',items);csvsave(out/'summary.csv',summarize(items,args.study))
    save(out/'status.json',dict(status='complete',records=len(raw),source_seal=digest(out/'seal.json'),raw_sha256=digest(out/'raw_counts.json'),seconds_this_invocation=time.perf_counter()-start))
    print(args.study,'COMPLETE',len(items),flush=True)
if __name__=='__main__':main()
