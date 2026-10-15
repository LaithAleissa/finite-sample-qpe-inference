"""Independent replay using a sinc kernel, scalar nuisance searches and union algebra.

Checks saved alternative fits, not their global optimality. Double precision is
deliberate; inherited 50-digit checks apply only to their original datasets.
"""
import os
for _n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_n]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,xlogy
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/physical_bridge'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def kernel(e,n):
    """Closed-form squared Dirichlet kernel, independent of production FFT."""
    m=2**n
    x=(np.asarray(e)[...,None]*.8377580409572781+2*np.pi*np.arange(m)/m)/(2*np.pi)
    x=(x+.5)%1-.5
    return (np.sinc(m*x)/np.sinc(x))**2

def merge(intervals):
    out=[]
    for lo,hi in sorted(intervals):
        if out and lo<=out[-1][1]+1e-12:out[-1][1]=max(hi,out[-1][1])
        else:out.append([lo,hi])
    return out

def profile(counts,p):
    def ll(eta):return float(np.sum(xlogy(counts,(1-eta)*p+eta/len(p))))
    opt=minimize_scalar(lambda eta:-ll(eta),bounds=(0,1),method='bounded',options={'xatol':1e-14})
    return max(ll(0),ll(1),ll(opt.x))

def main():
    assert read(OUT/'status.json')['status']=='complete'
    raw=read(OUT/'raw_counts.json');states=read(OUT/'state_analysis.json');trials=read(OUT/'gap_trials.json')
    assert len(raw)==len(states)==1600 and len(trials)==800
    assert hashlib.sha256((OUT/'raw_counts.json').read_bytes()).hexdigest()==read(OUT/'raw_seal.json')['sha256']
    ref={};max_num=0.;max_fit=0.;decisions=0
    # Matrix-evaluate all independent mixture numerators in register groups.
    mixture={}
    for n in (4,6):
        m=2**n;energies=np.linspace(-3.75,3.75,32*m,endpoint=False);etas=np.linspace(0,1,17)
        p=kernel(energies,n);q=((1-etas[None,:,None])*p[:,None,:]+etas[None,:,None]/m).reshape(-1,m)
        logs=np.log(np.maximum(q,1e-300))
        indexes=[i for i,r in enumerate(raw) if r['n']==n]
        for j in range(0,len(indexes),40):
            ix=indexes[j:j+40];counts=np.array([np.array(raw[i]['batch_a'])+raw[i]['batch_b'] for i in ix])
            vals=logsumexp(logs@counts.T,axis=0)-np.log(len(logs))
            mixture.update(zip(ix,vals))
    for i,(r,s) in enumerate(zip(raw,states)):
        assert s['record']==r and sum(r['batch_a'])==sum(r['batch_b'])==512
        a=np.array(r['batch_a']);b=np.array(r['batch_b']);c=a+b;p=kernel(r['energy_for_scoring_only'],r['n'])
        for name,item in s['methods'].items():
            fit=item['fit']
            if name=='crossfit':
                def log_at(counts,theta):
                    pe=kernel(theta[0],r['n']);return float(np.sum(xlogy(counts,(1-theta[1])*pe+theta[1]/len(pe))))
                na=log_at(a,fit['fit_b']);nb=log_at(b,fit['fit_a'])
                max_num=max(max_num,abs(na-fit['na']),abs(nb-fit['nb']))
                def le(eta):
                    q=(1-eta)*p+eta/len(p)
                    return float(logsumexp([na-np.sum(xlogy(a,q)),nb-np.sum(xlogy(b,q))])-np.log(2))
                opt=minimize_scalar(le,bounds=(0,1),method='bounded',options={'xatol':1e-14})
                accepted=min(le(0),le(1),le(opt.x))<=-np.log(.025)
            elif name=='qpe_mixture':
                max_num=max(max_num,abs(mixture[i]-fit['log_predictive']))
                accepted=profile(c,p)>=mixture[i]+np.log(.025)
            else:
                saved=float(np.sum(xlogy(c,(1-fit['eta'])*kernel(fit['energy'],r['n'])+fit['eta']/len(p))))
                max_fit=max(max_fit,abs(saved-fit['maximum']))
                accepted=2*(saved-profile(c,p))<=chi2.ppf(.975,1)
            assert bool(accepted)==item['point_covered'],(i,name,'point disagreement')
            decisions+=1
    # Independently reconstruct full difference unions and classical images.
    indexed={(x['record']['design'],x['record']['trial'],x['record']['state']):x for x in states}
    from scipy.constants import N_A, h, c, k, physical_constants
    scale=315*h*c*100/k/4
    curie=N_A*(physical_constants['Bohr magneton'][0]*1e3)**2/(k*1e7)
    g2=(2.42**2+2*2.08**2)/3;temperatures=np.array([50.,100.,200.,283.,300.,500.])
    curve_errors=[]
    for r in trials:
        for name,item in r['methods'].items():
            ss=indexed[r['design'],r['trial'],'S']['methods'][name]['set']['intervals']
            tt=indexed[r['design'],r['trial'],'T0']['methods'][name]['set']['intervals']
            rawgap=merge([[tl-sh,th-sl] for tl,th in tt for sl,sh in ss])
            assert rawgap==item['gap_intervals']
            af=[[max(0,lo),hi] for lo,hi in rawgap if hi>=0]
            assert af==item['af_gap_intervals']
            assert any(lo<=4<=hi for lo,hi in af)==item['gap_covered']
            assert item['response_covered']==item['gap_covered']
            assert not item['response_covered'] or item['response_hull_covered']
            if af:
                lower_gap=min(x[0] for x in af)*scale;upper_gap=max(x[1] for x in af)*scale
                def curve(gap):
                    q=np.exp(-gap/temperatures);return curie*g2/temperatures*q/(1+3*q)
                curve_errors.extend(abs(curve(upper_gap)-item['response']['lower']))
                curve_errors.extend(abs(curve(lower_gap)-item['response']['upper']))
    dense_checked=0
    for ex in read(OUT/'examples.json'):
        trial=next(x for x in trials if x['case']==ex['case'] and x['trial']==0)
        temps=np.array(ex['temperatures'])
        def dense_curve(gap):
            q=np.exp(-gap*scale/temps);return curie*g2/temps*q/(1+3*q)
        curve_errors.extend(abs(dense_curve(4)-ex['reference']));dense_checked+=len(temps)
        for name,image in ex['methods'].items():
            gaps=trial['methods'][name]['af_gap_intervals']
            if gaps:
                curve_errors.extend(abs(dense_curve(max(hi for lo,hi in gaps))-image['lower']))
                curve_errors.extend(abs(dense_curve(min(lo for lo,hi in gaps))-image['upper']))
                dense_checked+=2*len(temps)
    assert max_num<1e-7 and max_fit<1e-7 and max(curve_errors)<1e-15
    report=dict(status='passed',state_datasets=1600,independently_recomputed_point_decisions=decisions,
        gap_trials=800,method_gap_unions=2400,response_values_checked=len(curve_errors),dense_example_values_checked=dense_checked,
        maximum_log_numerator_discrepancy=max_num,maximum_saved_profile_fit_discrepancy=max_fit,
        maximum_response_discrepancy=max(curve_errors),
        precision='independent double precision; inherited high-precision audit has separate scope',
        global_alternative_maxima_certified=False,all_endpoints_certified=False)
    (ROOT/'qa/bridge_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
