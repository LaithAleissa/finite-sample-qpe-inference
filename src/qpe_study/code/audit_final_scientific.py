"""Independent scalar checks and complete outcome/provenance validation."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from scipy.stats import binomtest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/final_scientific'
sys.path.insert(0,str(ROOT/'code'))
from final_scientific_methods import Experiment,SimultaneousCP,DriftCF,RefinedJoint,streamed_mixture,QPEMixture

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_expected(p, expected):
    actual=sha(p)
    if actual==expected:return
    # Historical execution seals remain intact. Relocated-source checks use
    # the recorded original and current hashes, including text normalization.
    repo=ROOT.parents[1]
    record=read(repo/'source_relocation.json')['files'].get(p.relative_to(repo).as_posix())
    assert record is not None and record['original_sha256']==expected and record['relocated_sha256']==actual, str(p)

def sinc_kernel(n,time,energy):
    m=2**n;phase=-time*energy/(2*np.pi);z=phase-np.arange(m)/m
    p=(np.sinc(m*z)/np.sinc(z))**2
    return p/p.sum()
def cp_feasible(p,low,high):
    lo,hi=0.,1.
    for v,a,b in zip(p,low,high):
        slope=1/len(p)-v
        if abs(slope)<1e-15:
            if not a-1e-12<=v<=b+1e-12:return False
        else:
            u,w=sorted(((a-v)/slope,(b-v)/slope));lo=max(lo,u);hi=min(hi,w)
    return lo<=hi
def length(x):return sum(b-a for a,b in x)
def outside(a,b):
    result=[]
    for lo,hi in a:
        pieces=[[lo,hi]]
        for u,v in b:
            fresh=[]
            for x,y in pieces:
                if v<=x or u>=y:fresh.append([x,y]);continue
                if x<u:fresh.append([x,min(y,u)])
                if v<y:fresh.append([max(x,v),y])
            pieces=fresh
        result.extend(pieces)
    return result
def main():
    checks={}
    for name,h in read(OUT/'seal.json').items():verify_expected(ROOT/name,h)
    seal=read(OUT/'certification/seal.json')
    verify_expected(ROOT/'code/certify_final_sets.py',seal['code'])
    verify_expected(ROOT/'scientific_revision_plan.md',seal['plan'])
    protected=read(OUT/'protected.json')
    omitted=[]
    for name,h in protected.items():
        target=ROOT/name
        if not target.exists() and {'checkpoints','truth_audit'}&set(Path(name).parts):
            omitted.append(name);continue
        verify_expected(target,h)
    checks['historical_scientific_files_verified_after_relocation']=len(protected)-len(omitted)
    checks['public_package_excluded_intermediates']=len(omitted)
    cp=read(OUT/'cp_states.json');assert len(cp)==1600
    for s in cp:
        r=s['record'];v=s['methods']['cp'];counts=np.array(r['batch_a'])+r['batch_b'];m=len(counts);n=int(counts.sum())
        lows=[];highs=[]
        for c in counts:
            ci=binomtest(int(c),n).proportion_ci(confidence_level=1-.05/m,method='exact');lows.append(ci.low);highs.append(ci.high)
        assert np.max(np.abs(np.array(lows)-v['fit']['lower']))<1e-10
        assert np.max(np.abs(np.array(highs)-v['fit']['upper']))<1e-10
        point=cp_feasible(sinc_kernel(r['n'],.8377580409572781,r['energy']),lows,highs)
        assert point==v['point_covered']
    checks['independent_cp_truth_decisions']=len(cp)
    drift=read(OUT/'drift_states.json');assert len(drift)==600;nesting_excess=[];max_delta=0.
    for s in drift:
        r=s['record'];v=s['methods']['crossfit'];fit=v['fit'];p=sinc_kernel(r['n'],.8377580409572781,r['energy']);m=len(p)
        numer=[]
        for c,theta in [(r['batch_a'],fit['fit_b']),(r['batch_b'],fit['fit_a'])]:
            pred=(1-theta[1])*sinc_kernel(r['n'],.8377580409572781,theta[0])+theta[1]/m
            numer.append(float(np.sum(xlogy(c,pred))))
        den=[]
        for c in [r['batch_a'],r['batch_b']]:
            fn=lambda eta:-float(np.sum(xlogy(c,(1-eta)*p+eta/m)))
            opt=minimize_scalar(fn,bounds=(0,1),method='bounded',options={'xatol':1e-13})
            den.append(-min(fn(0),fn(1),opt.fun))
        loge=np.logaddexp(numer[0]-den[0],numer[1]-den[1])-np.log(2)
        assert (loge<=np.log(20))==s['methods']['drift_cf']['point_covered']
        aa=np.array(r['batch_a']);bb=np.array(r['batch_b'])
        def shared_fn(eta):
            q=(1-eta)*p+eta/m
            return float(np.logaddexp(numer[0]-np.sum(xlogy(aa,q)),numer[1]-np.sum(xlogy(bb,q)))-np.log(2))
        shared_opt=minimize_scalar(shared_fn,bounds=(0,1),method='bounded',options={'xatol':1e-13})
        assert (min(shared_fn(0),shared_fn(1),shared_opt.fun)<=np.log(20))==v['point_covered']
        nesting_excess.append(length(outside(v['set']['intervals'],s['methods']['drift_cf']['set']['intervals'])))
    checks['independent_drift_truth_decisions']=len(drift)
    checks['maximum_shared_production_length_outside_drift']=max(nesting_excess)
    # Outer approximations need not inherit exact nesting; retain any discrepancy.
    scaling=read(OUT/'scaling_states.json');assert len(scaling)==200
    for s in scaling:
        r=s['record'];assert sum(r['batch_a'])+sum(r['batch_b'])==r['N']
        counts=np.array(r['batch_a'])+r['batch_b']
        ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));p=sinc_kernel(r['n'],ex.time,r['energy'])
        fn=lambda eta:-float(np.sum(xlogy(counts,(1-eta)*p+eta/len(p))))
        opt=minimize_scalar(fn,bounds=(0,1),method='bounded',options={'xatol':1e-13});maximum=-min(fn(0),fn(1),opt.fun)
        for name in ['qpm_fixed','qpm_growing']:
            v=s['methods'][name];assert (maximum>=v['fit']['log_predictive']+np.log(.05))==v['point_covered']
        fit=s['methods']['crossfit']['fit'];nums=[]
        for c,theta in [(r['batch_a'],fit['fit_b']),(r['batch_b'],fit['fit_a'])]:
            pred=(1-theta[1])*sinc_kernel(r['n'],ex.time,theta[0])+theta[1]/len(p)
            nums.append(float(np.sum(xlogy(c,pred))))
        def cf_fn(eta):
            q=(1-eta)*p+eta/len(p)
            return float(np.logaddexp(nums[0]-np.sum(xlogy(r['batch_a'],q)),nums[1]-np.sum(xlogy(r['batch_b'],q)))-np.log(2))
        opt=minimize_scalar(cf_fn,bounds=(0,1),method='bounded',options={'xatol':1e-13})
        assert (min(cf_fn(0),cf_fn(1),opt.fun)<=np.log(20))==s['methods']['crossfit']['point_covered']
    checks['independent_scaling_qpm_truth_decisions']=400
    checks['independent_fresh_shared_cf_truth_decisions']=800
    # Independent full finite-grid numerator evaluation on first smallest cell.
    r=scaling[0]['record'];ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
    standard=QPEMixture(ex,r['batch_a'],r['batch_b']).log_predictive
    checks['streamed_base_log_numerator_difference']=abs(standard-scaling[0]['methods']['qpm_growing']['fit']['log_predictive'])
    assert checks['streamed_base_log_numerator_difference']<1e-9
    cert=read(OUT/'certification_summary.json');assert len(cert)==12
    enclosure=[]
    for s in cert:
        c=s['certificate'];assert abs(length(c['intervals'])+length(c['rejected_cells'])-7.5)<1e-12
        cells=sorted(c['intervals']+c['rejected_cells']);assert cells[0][0]==-3.75 and cells[-1][1]==3.75
        assert all(a[1]==b[0] for a,b in zip(cells,cells[1:]))
        enclosure.append(dict(case=s['record']['case'],method=s['method'],
                              certified_length=length(c['intervals']),production_length=length(s['production']['intervals']),
                              certified_extra_length=length(outside(c['intervals'],s['production']['intervals'])),
                              production_extra_length=length(outside(s['production']['intervals'],c['intervals']))))
    checks['complete_interval_certificates']=12;checks['certificate_comparisons']=enclosure
    report=dict(status='passed',checks=checks,independent_human_verification=False,
                certification_scope='12 selected newly generated outer sets only; historical unions not globally certified')
    if '--read-only' not in sys.argv:
        (ROOT/'qa/final_scientific_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
