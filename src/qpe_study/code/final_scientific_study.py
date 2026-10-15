"""Frozen review-driven studies; checkpoints and every outcome are retained."""
from final_scientific_methods import *
from pathlib import Path
import json,time,argparse,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/final_scientific'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def freeze():
    history=ROOT/'history/before_final_scientific';history.mkdir(exist_ok=True)
    for name in ['manuscript/body.tex','manuscript/abstract.txt','manuscript/submission_metadata.json',
                 'code/build_manuscripts.py','code/final_release.py','manuscript/main.pdf',
                 'ieee_access/main.pdf','ieee_access/submission.pdf','supplement/main.pdf']:
        target=history/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(ROOT/name,target)
    if not (OUT/'protected.json').exists():
        save(OUT/'protected.json',{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'results').rglob('*') if p.is_file() and OUT not in p.parents})
    paths=[ROOT/'scientific_revision_plan.md',Path(__file__),ROOT/'code/final_scientific_methods.py']
    seal={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    if (OUT/'seal.json').exists():assert read(OUT/'seal.json')==seal,'Frozen source changed'
    else:save(OUT/'seal.json',seal)

def fresh(kind):
    records=[]
    if kind=='scaling':
        designs=[(4,e,.1,.1,n) for e in [1.013,-3.7499] for n in [256,1024,4096,16384,65536]];trials=20;seed=2026101003
    else:
        designs=[(n,e,a,b,1024) for n in [4,6] for e in [1.013,-3.7499]
                 for a,b in [(.1,.1),(.1,.2),(.2,.4),(.4,.8),(.95,.98)]];trials=30;seed=2026101004
    for design,(n,e,a,b,total) in enumerate(designs):
        ex=Experiment(n,.8377580409572781,(-3.75,3.75));p=ex.probabilities(e)
        rng=np.random.default_rng(np.random.SeedSequence([seed,design]))
        for trial in range(trials):
            aa=rng.multinomial(total//2,(1-a)*p+a/2**n);bb=rng.multinomial(total//2,(1-b)*p+b/2**n)
            records.append(dict(case=f'{kind}_{design}',trial=trial,n=n,energy=e,eta_a=a,eta_b=b,
                                N=total,batch_a=aa.tolist(),batch_b=bb.tolist()))
    path=OUT/f'{kind}_counts.json'
    if path.exists():assert read(path)==records
    else:save(path,records)
    return records

def evaluate(task):
    kind,i,r=task;ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));out=dict(record=r,methods={})
    if kind=='cp':objects=[('cp',lambda:SimultaneousCP(ex,r['batch_a'],r['batch_b']))]
    elif kind=='scaling':objects=[('crossfit',lambda:RefinedJoint(ex,r['batch_a'],r['batch_b'])),
                                ('qpm_fixed',lambda:QPEMixture(ex,r['batch_a'],r['batch_b'])),
                                ('qpm_growing',lambda:GrowingQPM(ex,r['batch_a'],r['batch_b']))]
    else:objects=[('crossfit',lambda:RefinedJoint(ex,r['batch_a'],r['batch_b']))]
    shared=None
    for name,construct in objects:
        start=time.perf_counter();obj=construct();cs=obj.confidence_set()
        if name=='crossfit':shared=obj
        if name=='crossfit':
            lo,hi=obj.projection_membership_bounds(r['energy']);point=bool(hi[0]<=-np.log(.05));assert point or lo[0]>-np.log(.05)
        else:point,rejected=obj.membership(r['energy']);assert point or rejected
        inside=any(a<=r['energy']<=b for a,b in cs['intervals']);assert not point or inside
        fit=(dict(fit_a=list(obj.fit_a),fit_b=list(obj.fit_b),na=obj.na,nb=obj.nb) if name=='crossfit'
             else dict(lower=obj.lower.tolist(),upper=obj.upper.tolist()) if name=='cp'
             else dict(log_predictive=obj.log_predictive,s=getattr(obj,'s',1)))
        out['methods'][name]=dict(set=cs,point_covered=point,set_covered=inside,fit=fit,seconds=time.perf_counter()-start)
    if kind=='drift':
        start=time.perf_counter();obj=DriftCF(ex,r['batch_a'],r['batch_b'],shared=shared);cs=obj.confidence_set()
        lo,hi=obj.projection_membership_bounds(r['energy']);point=bool(hi[0]<=-np.log(.05));assert point or lo[0]>-np.log(.05)
        inside=any(a<=r['energy']<=b for a,b in cs['intervals']);assert not point or inside
        out['methods']['drift_cf']=dict(set=cs,point_covered=point,set_covered=inside,seconds_incremental=time.perf_counter()-start)
    return kind,i,out

def aggregate(kind,states):
    rows=[]
    for case in dict.fromkeys(s['record']['case'] for s in states):
        group=[s for s in states if s['record']['case']==case];r=group[0]['record']
        for name in group[0]['methods']:
            mm=[s['methods'][name] for s in group];lens=[sum(b-a for a,b in v['set']['intervals']) for v in mm]
            k=sum(v['point_covered'] for v in mm);ci=binomtest(k,len(mm)).proportion_ci(method='exact')
            rows.append(dict(case=case,n=r['n'],energy=r['energy'],N=r.get('N',2*r.get('shots_per_batch',0)),
                             eta_a=r.get('eta_a',r.get('eta')),eta_b=r.get('eta_b',r.get('eta')),method=name,
                             trials=len(mm),point_covered=k,set_covered=sum(v['set_covered'] for v in mm),
                             mc_lower=ci.low,mc_upper=ci.high,median_length=float(np.median(lens)),
                             q25=float(np.quantile(lens,.25)),q75=float(np.quantile(lens,.75)),
                             full=sum(v['set']['intervals']==[[-3.75,3.75]] for v in mm),
                             empty=sum(not v['set']['intervals'] for v in mm),
                             disconnected=sum(len(v['set']['intervals'])>1 for v in mm),
                             median_seconds=float(np.median([v.get('seconds',v.get('seconds_incremental')) for v in mm]))))
    save(OUT/f'{kind}_summary.json',rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cp','scaling','drift','freeze']);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    freeze()
    if a.mode=='freeze':return
    records=([r for r in read(ROOT/'results/mixture/raw_counts.json') if r['group']=='paired'] if a.mode=='cp' else fresh(a.mode))
    cp=OUT/'checkpoints';cp.mkdir(exist_ok=True)
    tasks=[(a.mode,i,r) for i,r in enumerate(records) if not (cp/f'{a.mode}_{i:04d}.json').exists()]
    save(OUT/f'{a.mode}_status.json',dict(status='running',datasets=len(records)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(evaluate,t) for t in tasks]
        for j,f in enumerate(as_completed(futures),1):
            kind,i,out=f.result();save(cp/f'{kind}_{i:04d}.json',out)
            if j%20==0:print(kind,j,'/',len(tasks),flush=True)
    states=[read(cp/f'{a.mode}_{i:04d}.json') for i in range(len(records))]
    save(OUT/f'{a.mode}_states.json',states);aggregate(a.mode,states)
    save(OUT/f'{a.mode}_status.json',dict(status='complete',datasets=len(records)))
    print(a.mode,'complete',flush=True)
if __name__=='__main__':main()
