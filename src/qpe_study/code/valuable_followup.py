"""Review-driven matched predictor and query-budget studies; old sources immutable."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import gammaln
from methods import QPEMixture,Experiment,RefinedJoint,RefinedProfile,counts_array,binomial_interval
from physical_bridge import save,read,digest,af_intersection,response_image,SCALE
from inference import difference_set
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/valuable_followup'
ALPHA=.025

def dirichlet_numerator(counts):
    c=np.asarray(counts);m=len(c)
    return float(gammaln(m/2)-gammaln(c.sum()+m/2)+np.sum(gammaln(c+.5)-gammaln(.5)))

class UnknownDirichlet(QPEMixture):
    def __init__(self,experiment,a,b,alpha=.05):
        if experiment.uniform_noise!=0 or not 0<alpha<1:raise ValueError('Invalid unknown-noise experiment/alpha')
        self.experiment=experiment;self.alpha=alpha
        self.counts=counts_array(a,experiment.n)+counts_array(b,experiment.n)
        self.log_predictive=dirichlet_numerator(self.counts)
        self.threshold=self.log_predictive+np.log(alpha)

METHODS={'crossfit':RefinedJoint,'qpe_mixture':QPEMixture,'dirichlet':UnknownDirichlet,'profile':RefinedProfile}

def evaluate(task):
    kind,index,r=task;ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));out=dict(record=r,methods={})
    for name in (['dirichlet'] if kind=='paired' else METHODS):
        start=time.perf_counter();obj=METHODS[name](ex,r['batch_a'],r['batch_b'],ALPHA);cs=obj.confidence_set()
        e=r['energy_for_scoring_only']
        if name=='crossfit':
            lo,hi=obj.projection_membership_bounds(e);accepted=bool(hi[0]<=-np.log(ALPHA));rejected=bool(lo[0]>-np.log(ALPHA))
            fit=dict(fit_a=list(obj.fit_a),fit_b=list(obj.fit_b),na=obj.na,nb=obj.nb)
        else:
            accepted,rejected=obj.membership(e)
            fit=(dict(log_predictive=obj.log_predictive,threshold=obj.threshold) if name!='profile'
                 else dict(energy=obj.estimate,eta=obj.eta,maximum=obj.maximum,threshold=obj.threshold))
        assert accepted or rejected
        included=any(lo<=e<=hi for lo,hi in cs['intervals']);assert not accepted or included
        out['methods'][name]=dict(set=cs,point_covered=accepted,set_covered=included,fit=fit,seconds=time.perf_counter()-start)
    return kind,index,out

def freeze():
    paths=[ROOT/'followup_plan.md',Path(__file__),ROOT/'code/methods.py',ROOT/'code/physical_bridge.py',ROOT.parent/'continuous_inversion/code/refined_inversion.py',ROOT.parent/'continuous_inversion/code/profile_lr.py']
    seal=dict(files={p.relative_to(ROOT.parent).as_posix():digest(p) for p in paths},original_counts_sha256=digest(ROOT/'results/physical_bridge/raw_counts.json'),externally_preregistered=False,review_driven=True)
    if (OUT/'seal.json').exists():assert read(OUT/'seal.json')==seal
    else:save(OUT/'seal.json',seal)
    lam=1-.9**(1/15);cells=[(4,.1),(6,.1),(4,.95),(6,.95),(6,1-(1-lam)**63)]
    rows=[]
    for design,(n,eta) in enumerate(cells):
        ex=Experiment(n,.8377580409572781,(-3.75,3.75));shots=2142 if n==4 else 510
        assert 4*shots*(2**n-1)==128520
        for si,(state,e) in enumerate([('S',-3.),('T0',1.)]):
            rng=np.random.default_rng(np.random.SeedSequence([2026100805,design,si]));q=(1-eta)*ex.probabilities(e)+eta/2**n
            for trial in range(100):
                a,b=rng.multinomial(shots,q,size=2)
                rows.append(dict(design=design,case=f'budget_{design}',n=n,eta_for_scoring_only=eta,state=state,energy_for_scoring_only=e,trial=trial,shots_per_batch=shots,base_queries_per_gap=128520,batch_a=a.tolist(),batch_b=b.tolist()))
    if (OUT/'raw_counts.json').exists():assert read(OUT/'raw_counts.json')==rows
    else:save(OUT/'raw_counts.json',rows)
    save(OUT/'raw_seal.json',dict(sha256=digest(OUT/'raw_counts.json'),state_datasets=len(rows),lambda_per_query=lam))
    return rows

def aggregate(states,kind):
    grouped={}
    for s in states:grouped.setdefault((s['record']['design'],s['record']['trial']),{})[s['record']['state']]=s
    trials=[]
    for (design,trial),pair in sorted(grouped.items()):
        rec=pair['S']['record'];item=dict(design=design,trial=trial,n=rec['n'],eta=rec['eta_for_scoring_only'],methods={})
        for name in pair['S']['methods']:
            raw=difference_set(pair['T0']['methods'][name]['set']['intervals'],pair['S']['methods'][name]['set']['intervals']);af=af_intersection(raw);resp=response_image(af)
            item['methods'][name]=dict(gap_intervals=raw,af_gap_intervals=af,gap_covered=any(lo<=4<=hi for lo,hi in af),length_af=sum(hi-lo for lo,hi in af),components=len(af),empty=not af,full_af=af==[[0.,7.5]],response=resp)
        trials.append(item)
    summary=[]
    for design in sorted(set(t['design'] for t in trials)):
        rr=[t for t in trials if t['design']==design]
        for name in rr[0]['methods']:
            mm=[t['methods'][name] for t in rr];k=sum(t['gap_covered'] for t in mm);lo,hi=binomial_interval(k,len(mm),.05/24)
            ss=[s['methods'][name] for s in states if s['record']['design']==design]
            summary.append(dict(design=design,n=rr[0]['n'],eta=rr[0]['eta'],method=name,trials=len(mm),covered=k,mc_lower=lo,mc_upper=hi,median_length_K=float(np.median([t['length_af'] for t in mm]))*SCALE,median_length_af=float(np.median([t['length_af'] for t in mm])),full=sum(t['full_af'] for t in mm),empty=sum(t['empty'] for t in mm),disconnected=sum(t['components']>1 for t in mm),median_seconds=float(np.median([t['seconds'] for t in ss]))))
    save(OUT/f'{kind}_gap_trials.json',trials);save(OUT/f'{kind}_summary.json',summary)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    budget=freeze();paired=read(ROOT/'results/physical_bridge/raw_counts.json');cp=OUT/'checkpoints';cp.mkdir(exist_ok=True)
    tasks=[(kind,i,r) for kind,rows in [('paired',paired),('budget',budget)] for i,r in enumerate(rows) if not (cp/f'{kind}_{i:04d}.json').exists()]
    save(OUT/'status.json',dict(status='running',remaining=len(tasks)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(evaluate,t) for t in tasks]
        for j,f in enumerate(as_completed(fs),1):
            kind,i,s=f.result();save(cp/f'{kind}_{i:04d}.json',s)
            if j%40==0:print(j,'/',len(tasks),flush=True)
    for kind,rows in [('paired',paired),('budget',budget)]:
        states=[read(cp/f'{kind}_{i:04d}.json') for i in range(len(rows))];save(OUT/f'{kind}_states.json',states);aggregate(states,kind)
    save(OUT/'status.json',dict(status='complete',paired_state_datasets=1600,budget_state_datasets=1000,budget_gap_trials=500,base_queries_per_gap=128520))
    print('Follow-up complete',flush=True)
if __name__=='__main__':main()
