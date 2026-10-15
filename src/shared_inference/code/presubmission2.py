"""Sealed, outcome-independent reference sensitivity and synthetic phase sweep."""
from evidence import *
from circuit_noise import noisy_distribution,best_approximation,Model,transpile
from adequacy import deviance,batch_maximum
from dimer import prepare_state,controlled_evolution,qpe
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate

OUT=ROOT/'results/presubmission2'
REFERENCES=('base','dense','adaptive')
ENERGIES=(1.,.234375,-3.7499)
RATES=(('low',.0001,.001,.002),('medium',.001,.01,.01),('high',.005,.03,.03))

def dimensions(n,N,name):
    s=1 if name=='base' else (2 if name=='dense' else max(1,int(np.ceil(np.sqrt(N/256)))))
    if name not in REFERENCES:raise ValueError(name)
    return 32*2**n*s,1+16*s

@lru_cache(maxsize=12)
def grid_logs(n,time,bounds,ne,nh):
    ex=Experiment(n,time,bounds);e=np.linspace(*bounds,ne,endpoint=False);eta=np.linspace(0,1,nh)
    p=ex.probabilities(e)
    q=((1-eta[None,:,None])*p[:,None,:]+eta[None,:,None]/2**n).reshape(-1,2**n)
    with np.errstate(divide='ignore'):return np.log(q)

def predictive(ex,c,name):
    ne,nh=dimensions(ex.n,int(sum(c)),name)
    logs=grid_logs(ex.n,ex.time,tuple(ex.bounds),ne,nh);used=c>0
    return float(logsumexp(logs[:,used]@c[used])-np.log(len(logs)))

class ReferenceMixture(QPEMixture):
    def __init__(self,ex,a,b,name):
        super().__init__(ex,a,b)
        self.log_predictive=predictive(ex,self.counts,name)
        self.threshold=self.log_predictive+np.log(self.alpha)

def seal(folder):
    deps=[ROOT/'presubmission2_analysis_plan.md',Path(__file__),ROOT/'code/methods.py',ROOT/'code/circuit_noise.py',ROOT/'code/adequacy.py',ROOT/'code/fast_profile.py',SRC/'quantum_model/dimer.py',SRC/'continuous_inversion/code/refined_inversion.py',SRC/'continuous_inversion/code/profile_lr.py']
    data={str(p.relative_to(SRC)):digest(p) for p in deps}
    path=folder/'seal.json'
    if path.exists():assert read(path)==data,'Numerical source changed after freeze'
    else:save(path,data)

def qpm_task(x):
    r=x['record'];ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75));c=np.array(r['batch_a'])+r['batch_b']
    result=dict(record=r,metrics={},sets={},fits={},crossfit=x['metrics']['crossfit'],crossfit_set=x['sets']['crossfit'])
    maximum=x['fits']['profile']['maximum'];cache={}
    for name in REFERENCES:
        dims=dimensions(r['n'],sum(c),name)
        if dims in cache:
            other=cache[dims]
            for field in ('metrics','sets','fits'):result[field][name]=result[field][other]
            continue
        cache[dims]=name;start=time.perf_counter()
        if name=='base':
            cs=x['sets']['qpe_mixture'];metric=dict(x['metrics']['qpe_mixture']);lp=x['fits']['qpe_mixture']['log_predictive']
        else:
            obj=ReferenceMixture(ex,r['batch_a'],r['batch_b'],name);lp=obj.log_predictive;cs=obj.confidence_set()
            point,reject=obj.membership(r['energy']);assert point or reject
            inside=any(a<=r['energy']<=b for a,b in cs['intervals']);assert not point or inside
            metric=dict(point_covered=point,set_covered=inside,length=length(cs),components=len(cs['intervals']),full_branch=cs['intervals']==[list(ex.bounds)],empty=not cs['intervals'],seconds=time.perf_counter()-start,cells=cs['visited_cells'])
        result['sets'][name]=cs;result['metrics'][name]=metric
        result['fits'][name]=dict(log_predictive=lp,D=float(maximum-lp),cutoff=float(2*(maximum-lp-np.log(.05))),energies=int(dims[0]),eta_points=int(dims[1]))
    return result

def qpm_summary(items):
    rows=[];rng=np.random.default_rng(2026100511)
    for case in dict.fromkeys(x['record']['case'] for x in items):
        rr=[x for x in items if x['record']['case']==case];r=rr[0]['record'];ix=rng.integers(0,len(rr),size=(4000,len(rr)))
        cf=np.array([x['crossfit']['length'] for x in rr]);cfboot=np.median(cf[ix],axis=1);assert min(cfboot)>0
        for name in REFERENCES:
            m=[x['metrics'][name] for x in rr];v=np.array([z['length'] for z in m]);boot=np.median(v[ix],axis=1)/cfboot
            lo,hi=binomial_interval(sum(z['point_covered'] for z in m),len(m),.05/(3*len(set(x['record']['case'] for x in items))))
            rows.append(dict(case=case,group=r['group'],reference=name,n=r['n'],eta=r['eta'],N=sum(r['batch_a'])+sum(r['batch_b']),trials=len(m),point_covered=sum(z['point_covered'] for z in m),set_covered=sum(z['set_covered'] for z in m),mc_lower=lo,mc_upper=hi,median_length=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),full_branch=sum(z['full_branch'] for z in m),disconnected=sum(z['components']>1 for z in m),empty=sum(z['empty'] for z in m),median_D=float(np.median([x['fits'][name]['D'] for x in rr])),median_cutoff=float(np.median([x['fits'][name]['cutoff'] for x in rr])),ratio_to_CF=float(np.median(v)/np.median(cf)),ratio_lower=float(np.quantile(boot,.025)),ratio_upper=float(np.quantile(boot,.975))))
    return rows

def run_qpm(workers):
    folder=OUT/'qpm';seal(folder)
    items=[x for x in merged() if x['record']['group']=='paired' or (x['record']['group']=='transition' and x['record']['eta'] in (.95,.98))];assert len(items)==2400
    save(folder/'raw_counts.json',[x['record'] for x in items]);cp=folder/'checkpoints';cp.mkdir(exist_ok=True)
    tasks=[(i,x) for i,x in enumerate(items) if not (cp/f'{i:04d}.json').exists()]
    save(folder/'status.json',dict(status='running',records=2400,remaining=len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (i,x),res in zip(tasks,pool.map(qpm_task,[x for i,x in tasks],chunksize=4)):
            save(cp/f'{i:04d}.json',res)
            if (i+1)%200==0:print('QPM sensitivity',i+1,'/2400',flush=True)
    rr=[read(cp/f'{i:04d}.json') for i in range(2400)];save(folder/'analysis.json',rr);csvsave(folder/'summary.csv',qpm_summary(rr))
    save(folder/'status.json',dict(status='complete',records=2400,references=3));print('QPM sensitivity COMPLETE',flush=True)

def shifted_circuit(e):
    # Only the known prepared phase must lie on the inference branch. Other
    # eigenvalues may alias after shifting; they have zero ideal preparation mass.
    model=Model(offset=e-1);time0=.8377580409572781;n=4
    qc=QuantumCircuit(n+2);prepare_state(qc,(n,n+1),'triplet_plus')
    for k in range(n):
        qc.h(k);qc.compose(controlled_evolution(model,time0,2**k),[k,n,n+1],inplace=True)
    qc.append(QFTGate(n).inverse(),range(n))
    return transpile(qc,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=731)

def circuit_inputs():
    folder=OUT/'circuit';seal(folder)
    if (folder/'raw_counts.json').exists():return read(folder/'raw_counts.json')
    original=transpile(qpe(Model(),'triplet_plus',4,.8377580409572781),basis_gates=['u','cx'],optimization_level=1,seed_transpiler=731)
    assert shifted_circuit(1)==original
    ex=Experiment(4,.8377580409572781,(-3.75,3.75));rows=[];diagnostics=[];validation=[]
    oldraw=read(ROOT/'results/circuit/raw_counts.json');olddiag=read(ROOT/'results/circuit/distribution_diagnostics.json')
    for ei,e in enumerate(ENERGIES):
        circuit=shifted_circuit(e);ideal=noisy_distribution(circuit,0,0,0);err=float(max(abs(ideal-ex.probabilities(e))));assert err<1e-10
        validation.append(dict(energy=e,offset=e-1,ideal_max_error=err,other_eigenvalues=Model(offset=e-1).energies(),depth=circuit.depth(),operations=dict(circuit.count_ops())))
        for ni,(name,p1,p2,flip) in enumerate(RATES):
            case=f'e{ei}_{name}'
            if ei==0:
                d=dict(next(z for z in olddiag if z['case']==name));assert (d['p1'],d['p2'],d['readout'])==(p1,p2,flip)
                for old in oldraw:
                    if old['case']==name:rows.append(dict(old,case=case,level=name,group='phase_scope',energy_index=ei))
            else:
                dist=noisy_distribution(circuit,p1,p2,flip);d=best_approximation(dist,ex)
                rng=np.random.default_rng(np.random.SeedSequence([2026100512,ei,ni]))
                for trial in range(200):
                    a,b=rng.multinomial(512,dist,size=2)
                    rows.append(dict(case=case,level=name,group='phase_scope',trial=trial,n=4,energy=e,eta=None,shots_per_batch=512,mechanism='gate_readout_noise',batch_a=a.tolist(),batch_b=b.tolist(),energy_index=ei))
            d.update(case=case,level=name,target_energy=e,p1=p1,p2=p2,readout=flip,circuit_depth=circuit.depth(),operations=dict(circuit.count_ops()));diagnostics.append(d)
    save(folder/'raw_counts.json',rows);save(folder/'distribution_diagnostics.json',diagnostics);save(folder/'validation.json',dict(original_circuit_identical=True,cells=validation,old_E1_reused=True,synthetic_not_hardware=True))
    return rows

def run_circuit(workers):
    folder=OUT/'circuit';raw=circuit_inputs();assert len(raw)==1800
    cp=folder/'checkpoints';cp.mkdir(exist_ok=True)
    old={(x['record']['case'],x['record']['trial']):x for x in read(ROOT/'results/circuit/analysis.json')}
    for i,r in enumerate(raw):
        p=cp/f'{i:04d}.json'
        if r['energy_index']==0 and not p.exists():save(p,dict(old[r['level'],r['trial']],record=r))
    tasks=[(i,r) for i,r in enumerate(raw) if not (cp/f'{i:04d}.json').exists()]
    save(folder/'status.json',dict(status='running',records=1800,remaining=len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (i,r),res in zip(tasks,pool.map(evaluate,[(r,'circuit') for i,r in tasks],chunksize=4)):
            save(cp/f'{i:04d}.json',res)
            if (i+1)%100==0:print('Phase scope',i+1,'/1800',flush=True)
    rr=[read(cp/f'{i:04d}.json') for i in range(1800)];save(folder/'analysis.json',rr);csvsave(folder/'summary.csv',summarize(rr,'circuit'))
    save(folder/'status.json',dict(status='complete',records=1800,new_pairs=1200,reused_pairs=600));print('Phase scope COMPLETE',flush=True)

def adequacy_task(task):
    i,r=task;start=time.perf_counter();ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
    c=np.array(r['batch_a'])+r['batch_b'];fit=ProfileLikelihood(ex,r['batch_a'],r['batch_b'])
    observed=float(deviance(c,fit.maximum)[0]);q=(1-fit.eta)*ex.probabilities(fit.estimate)+fit.eta/2**r['n']
    rng=np.random.default_rng(np.random.SeedSequence([2026100513,i]));counts=rng.multinomial(int(c.sum()),q,size=999);statistics=[]
    for j in range(0,999,32):
        cc=counts[j:j+32];statistics.extend(deviance(cc,batch_maximum(ex,cc)).tolist())
    ge=sum(x>=observed for x in statistics);p=(1+ge)/1000
    return dict(case=r['case'],trial=r['trial'],fitted_energy=float(fit.estimate),fitted_eta=float(fit.eta),observed_deviance=observed,statistics=statistics,p_value=p,rejected=p<=.05,ge_count=ge,seed_namespace=[2026100513,i],seconds=time.perf_counter()-start)

def run_adequacy(workers):
    folder=OUT/'adequacy';seal(folder);raw=[r for r in circuit_inputs() if r['trial']<25];assert len(raw)==225
    save(folder/'raw_counts.json',raw);cp=folder/'checkpoints';cp.mkdir(exist_ok=True)
    old={(x['case'],x['trial']):x for x in read(ROOT/'results/adequacy/analysis.json')}
    for i,r in enumerate(raw):
        p=cp/f'{i:03d}.json'
        if r['energy_index']==0 and not p.exists():save(p,dict(old[r['level'],r['trial']],case=r['case'],reused=True))
    tasks=[(i,r) for i,r in enumerate(raw) if not (cp/f'{i:03d}.json').exists()]
    save(folder/'status.json',dict(status='running',records=225,new_resamples=149850,remaining=len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (i,r),res in zip(tasks,pool.map(adequacy_task,tasks,chunksize=1)):
            save(cp/f'{i:03d}.json',res);print('Phase adequacy',i+1,'/225',flush=True)
    rr=[read(cp/f'{i:03d}.json') for i in range(225)];save(folder/'analysis.json',rr);rows=[]
    for case in dict.fromkeys(r['case'] for r in raw):
        xx=[x for x in rr if x['case']==case];k=sum(x['rejected'] for x in xx);lo,hi=binomial_interval(k,25,.05/9)
        rows.append(dict(case=case,trials=25,rejected=k,mc_lower=lo,mc_upper=hi,median_p=float(np.median([x['p_value'] for x in xx])),median_deviance=float(np.median([x['observed_deviance'] for x in xx]))))
    csvsave(folder/'summary.csv',rows);save(folder/'status.json',dict(status='complete',records=225,new_records=150,resamples=224775,new_resamples=149850));print('Phase adequacy COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=['qpm','circuit','adequacy']);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    {'qpm':run_qpm,'circuit':run_circuit,'adequacy':run_adequacy}[args.study](args.workers)
if __name__=='__main__':main()
