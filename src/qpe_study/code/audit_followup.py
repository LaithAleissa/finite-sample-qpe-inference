"""Independent scalar/60-digit predictor replay and targeted envelope checks."""
from valuable_followup import *
from audit_bridge import kernel,profile,merge
from methods import probability_envelope,profile_bounds
from scipy.special import xlogy,logsumexp
from scipy.optimize import minimize_scalar
import mpmath as mp
from functools import lru_cache
mp.mp.dps=60

@lru_cache(None)
def lg(x):return mp.loggamma(mp.mpf(x))
def precise_dir(c):
    m=len(c)
    return lg(m/2)-lg(sum(c)+m/2)+sum(lg(int(v)+.5)-lg(.5) for v in c)

def main():
    for rel,sha in read(OUT/'seal.json')['files'].items():assert digest(ROOT.parent/rel)==sha
    for rel,sha in read(ROOT/'results/primary_dirichlet/seal.json').items():assert digest(ROOT/rel)==sha
    assert digest(ROOT/'results/physical_bridge/raw_counts.json')==read(OUT/'seal.json')['original_counts_sha256']
    assert read(OUT/'status.json')['status']=='complete'
    assert read(ROOT/'results/primary_dirichlet/status.json')['status']=='complete'
    paired=read(OUT/'paired_states.json');budget=read(OUT/'budget_states.json');primary=read(ROOT/'results/primary_dirichlet/states.json')
    # Independently replay new multinomial observations using the sinc kernel.
    raw=read(OUT/'raw_counts.json');assert [s['record'] for s in budget]==raw
    for design in range(5):
        for si,state in enumerate(('S','T0')):
            rr=[r for r in raw if r['design']==design and r['state']==state];first=rr[0]
            rng=np.random.default_rng(np.random.SeedSequence([2026100805,design,si]))
            eta=first['eta_for_scoring_only'];p=kernel(first['energy_for_scoring_only'],first['n']);q=(1-eta)*p+eta/len(p)
            for r in rr:
                a,b=rng.multinomial(r['shots_per_batch'],q,size=2)
                assert a.tolist()==r['batch_a'] and b.tolist()==r['batch_b']
    gamma_error=0.;decisions=0;max_qpm=0.;profiles=0
    for states in (paired,budget):
        for s in states:
            r=s['record'];c=np.array(r['batch_a'])+r['batch_b'];p=kernel(r['energy_for_scoring_only'],r['n'])
            item=s['methods']['dirichlet'];v=float(precise_dir(c));gamma_error=max(gamma_error,abs(v-item['fit']['log_predictive']))
            accepted=profile(c,p)>=v+np.log(.025)
            assert accepted==item['point_covered'];decisions+=1
    for s in primary:
        r=s['record'];c=np.array(r['batch_a'])+r['batch_b'];v=float(precise_dir(c));gamma_error=max(gamma_error,abs(v-s['log_predictive']))
        assert (profile(c,kernel(r['energy'],r['n']))>=v+np.log(.05))==s['point_covered'];decisions+=1
    # Independently reconstruct every resource QPM numerator with the sinc law.
    numerators={}
    for n in (4,6):
        m=2**n;ee=np.linspace(-3.75,3.75,32*m,endpoint=False);etas=np.linspace(0,1,17)
        p=kernel(ee,n);q=((1-etas[None,:,None])*p[:,None,:]+etas[None,:,None]/m).reshape(-1,m)
        with np.errstate(divide='ignore'):logs=np.log(q)
        ix=[i for i,s in enumerate(budget) if s['record']['n']==n]
        for j in range(0,len(ix),25):
            ids=ix[j:j+25];cc=np.array([np.array(budget[i]['record']['batch_a'])+budget[i]['record']['batch_b'] for i in ids])
            # Avoid 0*(-inf), retaining exact support semantics.
            for i,c in zip(ids,cc):numerators[i]=float(logsumexp(logs[:,c>0]@c[c>0])-np.log(len(logs)))
    for i,s in enumerate(budget):
        r=s['record'];a=np.array(r['batch_a']);b=np.array(r['batch_b']);c=a+b;p=kernel(r['energy_for_scoring_only'],r['n'])
        for name in ('crossfit','qpe_mixture','profile'):
            z=s['methods'][name];fit=z['fit']
            if name=='crossfit':
                def ll(c,theta):return float(np.sum(xlogy(c,(1-theta[1])*kernel(theta[0],r['n'])+theta[1]/len(p))))
                na=ll(a,fit['fit_b']);nb=ll(b,fit['fit_a'])
                def le(eta):
                    q=(1-eta)*p+eta/len(p)
                    return float(np.logaddexp(na-np.sum(xlogy(a,q)),nb-np.sum(xlogy(b,q)))-np.log(2))
                opt=minimize_scalar(le,bounds=(0,1),method='bounded',options={'xatol':1e-14})
                accepted=min(le(0),le(1),le(opt.x))<=-np.log(.025)
            else:
                threshold=fit['threshold']
                if name=='qpe_mixture':max_qpm=max(max_qpm,abs(numerators[i]-fit['log_predictive']))
                else:
                    alt=kernel(fit['energy'],r['n']);maximum=float(np.sum(xlogy(c,(1-fit['eta'])*alt+fit['eta']/len(p))))
                    assert abs(maximum-fit['maximum'])<1e-6
                accepted=profile(c,p)>=threshold
            assert accepted==z['point_covered'];profiles+=1
    from scipy.constants import N_A,h,c as lightspeed,k as boltzmann,physical_constants
    scale=315*h*lightspeed*100/boltzmann/4
    curie=N_A*(physical_constants['Bohr magneton'][0]*1e3)**2/(boltzmann*1e7)
    temps=np.array([50.,100.,200.,283.,300.,500.]);g2=(2.42**2+2*2.08**2)/3
    def curve(gap):
        q=np.exp(-gap*scale/temps);return curie*g2/temps*q/(1+3*q)
    unions=0;response_values=0;response_error=0.
    for kind,states in [('paired',paired),('budget',budget)]:
        lookup={(s['record']['design'],s['record']['trial'],s['record']['state']):s for s in states}
        for t in read(OUT/f'{kind}_gap_trials.json'):
            for name,z in t['methods'].items():
                S=lookup[t['design'],t['trial'],'S']['methods'][name]['set']['intervals'];T=lookup[t['design'],t['trial'],'T0']['methods'][name]['set']['intervals']
                expected=merge([[a-d,b-c] for a,b in T for c,d in S]);assert np.allclose(expected,z['gap_intervals'],rtol=0,atol=1e-12)
                af=[[max(0,a),b] for a,b in expected if b>=0];assert af==z['af_gap_intervals']
                assert z['gap_covered']==any(a<=4<=b for a,b in af);unions+=1
                image=z['response'];assert len(image['interval_images'])==len(af)
                for (a,b),im in zip(af,image['interval_images']):
                    response_error=max(response_error,float(np.max(abs(curve(b)-im['lower']))),float(np.max(abs(curve(a)-im['upper']))));response_values+=12
                if af:
                    response_error=max(response_error,float(np.max(abs(curve(max(b for a,b in af))-image['lower']))),float(np.max(abs(curve(min(a for a,b in af))-image['upper']))));response_values+=12
                else:assert image['lower']==image['upper']==[]
    # Trial-zero endpoints plus explicit seams/grid/near-uniform candidates.
    cells=[]
    for s in budget:
        if s['record']['trial']!=0:continue
        n=s['record']['n'];ex=Experiment(n,.8377580409572781,(-3.75,3.75))
        points=[-3.75,3.75,0.,-3.,1.]
        for z in s['methods'].values():points += [v for interval in z['set']['intervals'] for v in interval]
        for e in sorted(set(points)):
            radius=ex.bin_width/10000
            counts=np.array(s['record']['batch_a'])+s['record']['batch_b']
            cells.append((n,e,radius,counts))
    maximum_kernel_error=0.;envelope_checks=0;min_margin=1.;profile_envelope_checks=0
    for n,e,radius,counts in cells:
        ex=Experiment(n,.8377580409572781,(-3.75,3.75));lo,hi=probability_envelope(ex,np.array([e]),np.array([radius]));m=2**n
        _,profile_upper,_=profile_bounds(hi,counts)
        for fraction in (-1,-.5,0,.5,1):
            E=mp.mpf(float(e))+mp.mpf(float(radius))*fraction;t=mp.mpf(ex.time)
            probs=[]
            for k in range(m):
                x=E*t+2*mp.pi*k/m
                # Finite geometric sum in closed form, with removable-zero limit.
                den=mp.sin(x/2)
                val=mp.mpf(1) if abs(den)<mp.mpf('1e-55') else (mp.sin(m*x/2)/(m*den))**2
                probs.append(float(val))
            p=np.array(probs);margin=min(float(np.min(p-lo[0])),float(np.min(hi[0]-p)));min_margin=min(min_margin,margin)
            assert margin>=-1e-14
            assert profile(counts,p)<=profile_upper[0]+1e-8
            profile_envelope_checks+=1
            production=np.asarray(ex.probabilities(float(E))).reshape(-1)
            assert production.shape==p.shape
            maximum_kernel_error=max(maximum_kernel_error,float(np.max(np.abs(p-production))));envelope_checks+=m
    assert gamma_error<1e-7 and max_qpm<1e-7 and maximum_kernel_error<1e-10
    assert response_error<1e-15
    report=dict(status='passed',dirichlet_60digit_numerators_and_scalar_truth_decisions=decisions,budget_other_scalar_truth_decisions=profiles,reconstructed_gap_unions=unions,targeted_cells=len(cells),targeted_component_envelope_checks=envelope_checks,minimum_sampled_envelope_margin=min_margin,maximum_60digit_kernel_discrepancy=maximum_kernel_error,maximum_60digit_dirichlet_log_discrepancy=gamma_error,maximum_independent_qpm_log_discrepancy=max_qpm,limitations='Targeted sampled envelope diagnostics only; no directed rounding, exhaustive rejection-cell certification or global PLR maximum certificate.')
    report.update(independently_recreated_budget_state_counts=len(raw),response_values_checked=response_values,maximum_response_discrepancy=response_error)
    report['targeted_scalar_profile_upper_bound_checks']=profile_envelope_checks
    save(ROOT/'qa/valuable_followup_audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
