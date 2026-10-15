"""Post-review supplemental diagnostics; never retune the frozen comparison."""
from studies import *
from scipy.stats import binomtest
from inference import difference_set

def paired_table(items):
    tab=[[0,0],[0,0]]
    for r in items:tab[int(r['conventional_accepted'])][int(r['accepted'])]+=1
    plus,minus=tab[0][1],tab[1][0]
    return dict(table_PLR_rows_bootstrap_columns=tab,bootstrap_only=plus,PLR_only=minus,
                exact_two_sided_p=float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.)

def descriptive():
    old={(x['case'],x['trial']):x for x in read(SRC/'continuous_inversion/results/sets.json')}
    groups={}
    for x in read(ROOT/'results/mixture/analysis.json'):
        r=x['record']
        if r['group']!='paired':continue
        ex=Experiment(r['n'],r.get('time',.8377580409572781),(-3.75,3.75))
        c=np.array(r['batch_a'])+r['batch_b'];_,_,logs=reference(ex.n,ex.time,ex.bounds)
        ll=logs[:,c>0]@c[c>0];maxref=float(ll.max())
        maximum=old[r['case'],r['trial']]['fits']['profile']['maximum']
        D=maximum-x['fits']['qpe_mixture']['log_predictive']
        identity=np.log(len(ll))+(maximum-maxref)-logsumexp(ll-maxref)
        assert abs(D-identity)<2e-10 and D>=-1e-8
        groups.setdefault(r['case'],[]).append((D,2*(D+np.log(20)),maximum-maxref))
    rows=[]
    raw=read(ROOT/'results/mixture/raw_counts.json')
    for case,v in groups.items():
        r=next(r for r in raw if r['case']==case);ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
        q=(1-r['eta'])*ex.probabilities(r['energy'])+r['eta']/2**r['n'];_,_,logs=reference(ex.n,ex.time,ex.bounds)
        entropy=float(np.sum(xlogy(q,q)));grid_expected=logs[:,q>0]@q[q>0]
        delta=max(0.,entropy-float(np.max(grid_expected)))
        a=np.array(v)
        rows.append(dict(case=case,J=len(logs),trials=len(v),median_D=float(np.median(a[:,0])),median_cutoff=float(np.median(a[:,1])),minimum_cutoff=float(a[:,1].min()),maximum_cutoff=float(a[:,1].max()),median_grid_fit_loss=float(np.median(a[:,2])),population_grid_KL=delta))
    csvsave(ROOT/'results/postreview/qpm_regret.csv',rows)
    boot=read(ROOT/'results/bootstrap/analysis.json');result=paired_table(boot)
    result['per_design']=[dict(case=case,**paired_table([x for x in boot if x['case']==case])) for case in dict.fromkeys(x['case'] for x in boot)]
    save(ROOT/'results/postreview/paired_bootstrap.json',result)
    print('Predictive regret and paired bootstrap diagnostics complete.',flush=True)

def spectral_record(task):
    i,r=task;ex=Experiment(r['n'],r['time'],tuple(r['bounds']),r['assumed_uniform_noise'])
    obj=Inference(ex,r['batch_a'],r['batch_b'],.025 if r['stage']=='gap' else .05/3)
    cs=known_invert(obj)
    return dict(record=r,set=cs)

def spectral(workers):
    raw=[json.loads(s) for s in (SRC/'known_noise/results/raw_counts.jsonl').read_text().splitlines() if json.loads(s)['stage'] in ('gap','resolution')]
    out=ROOT/'results/spectral_refined';save(out/'raw_counts.json',raw)
    seal=dict(plan=digest(ROOT/'postreview_analysis_plan.md'),code=digest(Path(__file__)),methods=digest(ROOT/'code/methods.py'),inference=digest(SRC/'known_noise/inference.py'))
    if (out/'seal.json').exists():assert read(out/'seal.json')==seal
    else:save(out/'seal.json',seal)
    cp=out/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
    tasks=[(i,r) for i,r in enumerate(raw) if not (cp/f'{i:05d}.json').exists()]
    save(out/'status.json',dict(status='running',state_datasets=len(raw)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (i,r),x in zip(tasks,pool.map(spectral_record,tasks,chunksize=8)):
            save(cp/f'{i:05d}.json',x)
            if i%200==0:print('Spectral state',i+1,'/',len(raw),flush=True)
    results=[read(cp/f'{i:05d}.json') for i in range(len(raw))];save(out/'analysis.json',results)
    gaps=[];ordering=[];groups={}
    for x in results:
        r=x['record']
        if r['stage']=='gap':key=('gap',r['trial'])
        else:key=('ordering',r['n'],r['assumed_uniform_noise'],r['case'].split(';h=')[1].split(';')[0],r['trial'])
        groups.setdefault(key,[]).append(x)
    for key,xx in groups.items():
        xx.sort(key=lambda x:x['record']['truth_for_validation_only']);unions=[x['set']['intervals'] for x in xx]
        ds=[difference_set(unions[i+1],unions[i]) for i in range(len(unions)-1)]
        if key[0]=='gap':
            assert len(xx)==2
            gaps.append(dict(trial=key[1],intervals=ds[0],covered=any(lo<=4<=hi for lo,hi in ds[0]),length=sum(hi-lo for lo,hi in ds[0])))
        else:
            assert len(xx)==3
            separate=all(unions) and all(d and all(lo>0 for lo,hi in d) for d in ds)
            ordering.append(dict(n=key[1],noise=key[2],h=float(key[3]),trial=key[4],separated=bool(separate),state_sets=unions,gap_sets=ds))
    assert len(gaps)==200 and len(ordering)==2000
    save(out/'gap_trials.json',gaps);save(out/'ordering_trials.json',ordering)
    rows=[]
    for key in dict.fromkeys((x['n'],x['noise'],x['h']) for x in ordering):
        rr=[x for x in ordering if (x['n'],x['noise'],x['h'])==key];k=sum(x['separated'] for x in rr);lo,hi=binomial_interval(k,len(rr),.05/20)
        rows.append(dict(n=key[0],noise=key[1],h=key[2],trials=len(rr),separated=k,mc_lower=lo,mc_upper=hi))
    csvsave(out/'ordering_summary.csv',rows)
    zero=[x for x in ordering if x['h']==0]
    save(out/'summary.json',dict(gap_trials=200,gap_covered=sum(x['covered'] for x in gaps),gap_median_length=float(np.median([x['length'] for x in gaps])),gap_mean_length=float(np.mean([x['length'] for x in gaps])),ordering_trials=2000,zero_field_trials=len(zero),zero_field_false_separations=sum(x['separated'] for x in zero),ordering_cells=20))
    save(out/'status.json',dict(status='complete',state_datasets=len(raw),gap_trials=200,ordering_trials=2000))
    print(read(out/'summary.json'),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['descriptive','spectral']);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if args.mode=='descriptive':descriptive()
    else:spectral(args.workers)
