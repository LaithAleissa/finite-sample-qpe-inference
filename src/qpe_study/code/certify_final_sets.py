"""Directed interval enclosures of selected complete CF/QPM energy sets.

Only cells with certified e>1/alpha throughout are rejected. All other cells
are retained or subdivided. This does not certify historical production sets.
The model evaluates exact finite Fourier probabilities at stored binary-float
parameters; reference points and fitted predictor parameters are held fixed.
"""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_scientific_study import ROOT,OUT,read,save,sha
from final_scientific_methods import Experiment,RefinedJoint,QPEMixture,profile_bounds,_merge
import numpy as np
from mpmath import iv
iv.dps=40

def lower(x):return iv.mpf(x.a)
def upper(x):return iv.mpf(x.b)
def exact(x):
    if isinstance(x,(int,np.integer)):return iv.mpf(int(x))
    n,d=float(x).as_integer_ratio();return iv.mpf(n)/d
def bound_string(x):return str(x)

def point_prob(n,t,e):
    m=2**n;t=exact(t);e=exact(e);p=[];der=[]
    for k in range(m):
        phase=-t*e-2*iv.pi*k/m
        value=iv.mpf(m);slope=iv.mpf(0)
        for r in range(1,m):
            value+=2*(m-r)*iv.cos(r*phase)
            slope+=2*(m-r)*r*t*iv.sin(r*phase)
        p.append(value/(m*m));der.append(slope/(m*m))
    return p,der

def clipped_upper(x):return min(iv.mpf(1),upper(x))
def clipped_lower(x):return max(iv.mpf(0),lower(x))

def envelope(ex,a,b):
    mid=(a+b)/2;radius=exact((b-a)/2);p,d=point_prob(ex.n,ex.time,mid)
    t=exact(ex.time);m=2**ex.n;curvature=t*t*(m*m-1)/6
    lipschitz=abs(t)*(m*m-1)/(3*m)
    lo=[];hi=[]
    for v,g in zip(p,d):
        pad=min(upper(lipschitz*radius),upper(abs(g)*radius+curvature*radius*radius/2))
        lo.append(clipped_lower(v-pad));hi.append(clipped_upper(v+pad))
    return lo,hi

def likelihood(counts,p):
    value=iv.mpf(0)
    for c,q in zip(counts,p):
        if c:value+=int(c)*iv.ln(q)
    return value

def candidate_upper(counts,p,eta):
    """Concavity tangent upper bound over full eta range; no optimized-fit claim."""
    # Any interior eta gives a globally valid tangent; approximate fitting
    # chooses a useful tangent but cannot invalidate it.
    eta=exact(min(1-1e-12,max(1e-12,float(eta))));m=len(p)
    q=[(1-eta)*v+eta/m for v in p]
    val=likelihood(counts,q)
    grad=sum((int(c)*(iv.mpf(1)/m-v)/qq for c,v,qq in zip(counts,p,q) if c),iv.mpf(0))
    return upper(val+max(upper(grad*(0-eta)),upper(grad*(1-eta))))

def numerator_cf(obj):
    result=[]
    for c,fit in [(obj.a,obj.fit_b),(obj.b,obj.fit_a)]:
        p,_=point_prob(obj.experiment.n,obj.experiment.time,fit[0]);eta=exact(fit[1]);m=len(p)
        q=[(1-eta)*v+eta/m for v in p]
        if any(lower(v)<=0 for v,cnt in zip(q,c) if cnt):
            raise RuntimeError('Predictor support needs separate interval treatment')
        result.append(likelihood(c,q))
    return result

REFERENCE={}
def reference_logs(ex):
    if ex.n in REFERENCE:return REFERENCE[ex.n]
    m=2**ex.n;logs=[]
    for e in np.linspace(*ex.bounds,32*m,endpoint=False):
        p,_=point_prob(ex.n,ex.time,e)
        for eta_float in np.linspace(0,1,17):
            eta=exact(eta_float);q=[(1-eta)*v+eta/m for v in p]
            # At eta=0 true QPE zeros may straddle zero numerically. Taking
            # [0,upper] is safe and allows a -infinite log lower bound.
            row=[]
            for v in q:
                row.append(iv.ln(v) if lower(v)>0
                           else iv.mpf(['-inf',upper(iv.ln(clipped_upper(v)))]))
            logs.append(row)
    REFERENCE[ex.n]=logs;return logs

def numerator_qpm(ex,counts):
    values=[]
    for row in reference_logs(ex):values.append(sum((int(c)*v for c,v in zip(counts,row) if c),iv.mpf(0)))
    shift=max(upper(v) for v in values)
    total=sum((iv.exp(v-shift) for v in values),iv.mpf(0))
    return shift+iv.ln(total)-iv.ln(len(values))

def certificate(ex,counts_a,counts_b,method,numerators,tolerance):
    pending=[tuple(ex.bounds)];keep=[];reject=[];visited=0;cutoff=iv.ln(20)
    while pending:
        a,b=pending.pop();visited+=1
        if visited>100000:raise RuntimeError('Certificate cell budget exceeded')
        lo,hi=envelope(ex,a,b)
        if method=='qpm':
            _,_,eta=profile_bounds(np.array([float(v) for v in hi]),counts_a+counts_b)
            den=candidate_upper(counts_a+counts_b,hi,eta[0])
            certified=lower(numerators-den)>upper(cutoff)
        else:
            den=[]
            for c in [counts_a,counts_b]:
                _,_,eta=profile_bounds(np.array([float(v) for v in hi]),c)
                den.append(candidate_upper(c,hi,eta[0]))
            # Separate denominator maxima give a lower bound even when the
            # actual candidate nuisance is shared across the two terms.
            value=(iv.exp(numerators[0]-den[0])+iv.exp(numerators[1]-den[1]))/2
            certified=lower(value)>iv.mpf(20)
        if certified:reject.append([a,b]);continue
        if b-a<=tolerance:keep.append([a,b]);continue
        # An accepted witness across a whole cell can stop refinement.
        if method=='qpm':
            p=np.array([float(v) for v in lo]);_,_,eta=profile_bounds(p,counts_a+counts_b)
            ee=exact(eta[0]);q=[(1-ee)*v+ee/(2**ex.n) for v in lo]
            if all(v>0 for v,c in zip(q,counts_a+counts_b) if c) and upper(numerators-likelihood(counts_a+counts_b,q))<=lower(cutoff):
                keep.append([a,b]);continue
        else:
            # Shared eta witness preserves the original CF model.
            # Cheap finite witness grid; this is only an acceptance shortcut.
            for eta_float in [.0,.1,.25,.5,.75,.95,1.]:
                ee=exact(eta_float);q=[(1-ee)*v+ee/(2**ex.n) for v in lo]
                if not all(v>0 for v,c in zip(q,counts_a+counts_b) if c):continue
                value=(iv.exp(numerators[0]-likelihood(counts_a,q))+iv.exp(numerators[1]-likelihood(counts_b,q)))/2
                if upper(value)<=iv.mpf(20):break
            else:value=None
            if value is not None and upper(value)<=iv.mpf(20):keep.append([a,b]);continue
        mid=(a+b)/2;pending.extend([(a,mid),(mid,b)])
    return dict(intervals=_merge(keep),rejected_cells=reject,visited_cells=visited,tolerance=tolerance,
                arithmetic='mpmath directed interval arithmetic, 40 decimal digits',
                scope='certified outer enclosure of exact region for these fixed predictors/reference points')

def main():
    raw=read(ROOT/'results/mixture/raw_counts.json')
    names=['grid_noise_015','offgrid_noise_010','halfbin_lowshots','branch_noise_010','positive_noise_040']
    selected=[next(r for r in raw if r['case']==case and r['trial']==0) for case in names]
    selected.append(next(r for r in raw if r['group']=='transition' and r['eta']==.95 and r['trial']==0))
    results=[];dest=OUT/'certification';dest.mkdir(exist_ok=True)
    save(dest/'seal.json',dict(code=sha(Path(__file__)),plan=sha(ROOT/'scientific_revision_plan.md'),precision=40))
    for i,r in enumerate(selected):
        ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));aa=np.array(r['batch_a']);bb=np.array(r['batch_b'])
        for name,cls in [('cf',RefinedJoint),('qpm',QPEMixture)]:
            target=dest/f'{i}_{name}.json'
            if target.exists():result=read(target)
            else:
                start=time.perf_counter();obj=cls(ex,aa,bb)
                num=numerator_cf(obj) if name=='cf' else numerator_qpm(ex,aa+bb)
                cert=certificate(ex,aa,bb,name,num,ex.bin_width/2000)
                production=obj.confidence_set()
                result=dict(record=r,method=name,certificate=cert,production=production,
                            numerator_intervals=[bound_string(v) for v in num] if name=='cf' else bound_string(num),seconds=time.perf_counter()-start)
                save(target,result)
            results.append(result);print('Certified',i,name,result['certificate']['visited_cells'],flush=True)
    save(OUT/'certification_summary.json',results)
    save(OUT/'certification_status.json',dict(status='complete',datasets=len(selected),complete_certified_sets=len(results),precision=40,
                                             all_historical_sets_certified=False))
if __name__=='__main__':main()
