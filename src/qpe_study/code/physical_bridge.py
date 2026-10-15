"""Frozen fresh unknown-noise spectroscopy-to-response experiment.

Run with -B. All truth fields are confined to sampling and scoring; fits receive
only counts, register, time, branch and alpha. Never discard unresolved sets.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import sys, json, hashlib, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.dont_write_bytecode = True
from methods import Experiment, RefinedJoint, RefinedProfile, QPEMixture, length, binomial_interval
from inference import difference_set
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'material_model'))
from physics import bleaney_bowers, CM1_TO_K, G

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/physical_bridge'
METHODS = {'crossfit': RefinedJoint, 'qpe_mixture': QPEMixture, 'profile': RefinedProfile}
ALPHA = .025
SCALE = 315 * CM1_TO_K / 4
TEMPERATURES = np.array([50., 100., 200., 283., 300., 500.])
GPOW = float(np.sqrt((G[0]**2 + 2*G[1]**2)/3))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def af_intersection(intervals):
    """Declared positive-coupling prior; zero included as limiting response."""
    return [[max(0., lo), hi] for lo, hi in intervals if hi >= 0.]

def response_image(intervals, temperatures=TEMPERATURES):
    if not intervals:
        return dict(lower=[], upper=[], interval_images=[])
    images = [dict(lower=bleaney_bowers(temperatures, hi*SCALE, GPOW).tolist(),
                   upper=bleaney_bowers(temperatures, lo*SCALE, GPOW).tolist())
              for lo, hi in intervals]
    return dict(lower=np.min([x['lower'] for x in images], axis=0).tolist(),
                upper=np.max([x['upper'] for x in images], axis=0).tolist(),
                interval_images=images)

def freeze():
    plan = ROOT / 'analysis_plan.md'
    assert plan.exists()
    sources = [Path(__file__), ROOT/'code/methods.py',
               ROOT.parent/'continuous_inversion/code/refined_inversion.py',
               ROOT.parent/'continuous_inversion/code/profile_lr.py',
               ROOT.parent/'predictive_inference/code/predictive.py',
               ROOT.parent/'known_noise/inference.py', ROOT.parent/'material_model/physics.py']
    seal = dict(plan_sha256=digest(plan), sources={p.relative_to(ROOT.parent).as_posix():digest(p) for p in sources},
                externally_preregistered=False, seed=2026100804)
    if (OUT/'seal.json').exists():
        original=read(OUT/'seal.json')
        if original!=seal:
            correction=read(ROOT/'scoring_correction.json')
            assert original['sources']['qpe_study/code/physical_bridge.py']==correction['original_source_sha256']
            adjusted=json.loads(json.dumps(original));adjusted['sources']['qpe_study/code/physical_bridge.py']=correction['corrected_source_sha256']
            assert adjusted==seal, 'Unrecorded frozen source change; do not mix runs'
    else:
        save(OUT/'seal.json', seal)
    if (OUT/'raw_counts.json').exists():
        rows = read(OUT/'raw_counts.json')
    else:
        rows=[]
        for design, (n, eta) in enumerate([(4,.1),(6,.1),(4,.95),(6,.95)]):
            ex=Experiment(n,.8377580409572781,(-3.75,3.75))
            for state_index, (state, energy) in enumerate([('S',-3.),('T0',1.)]):
                rng=np.random.default_rng(np.random.SeedSequence([2026100804,design,state_index]))
                q=(1-eta)*ex.probabilities(energy)+eta/(2**n)
                for trial in range(200):
                    a,b=rng.multinomial(512,q,size=2)
                    rows.append(dict(design=design,case=f'n{n}_eta{eta:g}',n=n,eta_for_scoring_only=eta,
                                     state=state,energy_for_scoring_only=energy,trial=trial,
                                     batch_a=a.tolist(),batch_b=b.tolist()))
        save(OUT/'raw_counts.json',rows)
    raw_hash=digest(OUT/'raw_counts.json')
    if (OUT/'raw_seal.json').exists(): assert read(OUT/'raw_seal.json')['sha256']==raw_hash
    else: save(OUT/'raw_seal.json',dict(sha256=raw_hash,state_datasets=len(rows)))
    assert len(rows)==1600
    return rows

def evaluate(record):
    ex=Experiment(record['n'],.8377580409572781,(-3.75,3.75))
    result=dict(record=record, methods={})
    for name,cls in METHODS.items():
        start=time.perf_counter()
        obj=cls(ex,record['batch_a'],record['batch_b'],ALPHA)
        cs=obj.confidence_set()
        truth=record['energy_for_scoring_only']
        if name=='crossfit':
            lo,hi=obj.projection_membership_bounds(truth)
            accepted=bool(hi[0]<=-np.log(ALPHA)); rejected=bool(lo[0]>-np.log(ALPHA))
            fit=dict(fit_a=list(obj.fit_a),fit_b=list(obj.fit_b),na=obj.na,nb=obj.nb)
        else:
            accepted,rejected=obj.membership(truth)
            fit=(dict(log_predictive=obj.log_predictive,threshold=obj.threshold) if name=='qpe_mixture'
                 else dict(energy=obj.estimate,eta=obj.eta,maximum=obj.maximum,threshold=obj.threshold))
        if not (accepted or rejected): raise RuntimeError('Ambiguous truth decision')
        included=any(lo<=truth<=hi for lo,hi in cs['intervals'])
        assert not accepted or included
        result['methods'][name]=dict(set=cs,point_covered=bool(accepted),set_covered=bool(included),
                                    fit=fit,seconds=time.perf_counter()-start)
    return result

def aggregate(states):
    grouped={}
    for x in states: grouped.setdefault((x['record']['design'],x['record']['trial']),{})[x['record']['state']]=x
    trials=[];examples=[];temps=np.arange(50.,501.)
    true_response=bleaney_bowers(TEMPERATURES,4*SCALE,GPOW)
    for (design,trial),pair in sorted(grouped.items()):
        assert set(pair)=={'S','T0'}
        rec=pair['S']['record'];r=dict(design=design,case=rec['case'],n=rec['n'],eta=rec['eta_for_scoring_only'],trial=trial,methods={})
        for name in METHODS:
            raw=difference_set(pair['T0']['methods'][name]['set']['intervals'],pair['S']['methods'][name]['set']['intervals'])
            af=af_intersection(raw);response=response_image(af)
            gap_in=any(lo<=4<=hi for lo,hi in af)
            response_in=bool(af) and all(any(im['lower'][j]<=value<=im['upper'][j] for im in response['interval_images']) for j,value in enumerate(true_response))
            hull_in=bool(af) and bool(np.all(np.asarray(response['lower'])<=true_response)&np.all(true_response<=np.asarray(response['upper'])))
            assert response_in==gap_in
            r['methods'][name]=dict(gap_intervals=raw,af_gap_intervals=af,
                gap_covered=bool(gap_in),length_raw=sum(hi-lo for lo,hi in raw),length_af=sum(hi-lo for lo,hi in af),
                components=len(af),empty=not af,full_af=af==[[0.,7.5]],
                response=response,response_covered=response_in,response_hull_covered=hull_in,
                response_relative_hull_width_at_283=(response['upper'][3]-response['lower'][3])/true_response[3] if af else 0.)
        trials.append(r)
        if trial==0:
            examples.append(dict(case=rec['case'],trial=0,temperatures=temps.tolist(),
                reference=bleaney_bowers(temps,4*SCALE,GPOW).tolist(),
                methods={m:response_image(r['methods'][m]['af_gap_intervals'],temps) for m in METHODS}))
    summary=[]
    for design in range(4):
        rr=[x for x in trials if x['design']==design]
        for m in METHODS:
            mm=[r['methods'][m] for r in rr];k=sum(x['gap_covered'] for x in mm)
            lo,hi=binomial_interval(k,len(mm),.05/12)
            summary.append(dict(case=rr[0]['case'],n=rr[0]['n'],eta=rr[0]['eta'],method=m,trials=len(mm),
                gap_covered=k,response_covered=sum(x['response_covered'] for x in mm),
                mc_lower=lo,mc_upper=hi,median_length_raw=float(np.median([x['length_raw'] for x in mm])),
                median_length_af=float(np.median([x['length_af'] for x in mm])),
                q25_length_af=float(np.quantile([x['length_af'] for x in mm],.25)),
                q75_length_af=float(np.quantile([x['length_af'] for x in mm],.75)),
                median_length_K=float(np.median([x['length_af'] for x in mm]))*SCALE,
                empty=sum(x['empty'] for x in mm),full_af=sum(x['full_af'] for x in mm),
                disconnected=sum(x['components']>1 for x in mm),
                median_relative_response_width_283=float(np.median([x['response_relative_hull_width_at_283'] for x in mm]))))
    save(OUT/'gap_trials.json',trials);save(OUT/'examples.json',examples);save(OUT/'summary.json',summary)
    save(OUT/'status.json',dict(status='complete',state_datasets=1600,gap_trials=800,methods=list(METHODS),
        alpha_per_state=ALPHA,total_shots_per_gap=2048,scale_K_per_unit=SCALE,temperatures=TEMPERATURES.tolist(),
        curve_coverage='Fixed response image gives simultaneous containment over all positive temperatures; empirical grid replay is diagnostic.'))
    print(json.dumps(summary,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:
        aggregate(read(OUT/'state_analysis.json'));return
    rows=freeze();cp=OUT/'checkpoints';cp.mkdir(exist_ok=True)
    tasks=[(i,r) for i,r in enumerate(rows) if not (cp/f'{i:04d}.json').exists()]
    save(OUT/'status.json',dict(status='running',state_datasets=1600,remaining=len(tasks)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate,r):i for i,r in tasks}
        for j,f in enumerate(as_completed(futures),1):
            save(cp/f'{futures[f]:04d}.json',f.result())
            if j%40==0: print('State analyses completed:',1600-len(tasks)+j,'/1600',flush=True)
    states=[read(cp/f'{i:04d}.json') for i in range(1600)]
    save(OUT/'state_analysis.json',states);aggregate(states)

if __name__=='__main__': main()
