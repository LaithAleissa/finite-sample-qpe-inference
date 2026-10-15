"""Independent 50-digit truth-point audit, not endpoint/global certification."""
from studies import *
sys.path.insert(0,str(SRC/'continuous_inversion/vendor'))
import mpmath as mp
from functools import lru_cache
mp.mp.dps=50
TIME=.8377580409572781
SCALE=10**45

@lru_cache(maxsize=12000)
def kernel(e,n):
    m=2**n;e=mp.mpf(float(e));t=mp.mpf(TIME);out=[]
    for k in range(m):
        z=mp.exp(-1j*(e*t+2*mp.pi*k/m));power=mp.mpc(1);terms=[]
        for j in range(m):terms.append(power);power*=z
        out.append(abs(mp.fsum(terms)/m)**2)
    return tuple(out)

def ll(c,p,eta):
    eta=mp.mpf(eta);m=len(p);terms=[]
    for x,v in zip(c,p):
        if x:
            q=(1-eta)*v+eta/m
            if q==0:return mp.ninf
            terms.append(int(x)*mp.log(q))
    return mp.fsum(terms)

def optimize(p,a,b,na=None,nb=None):
    m=len(p);c=[x+y for x,y in zip(a,b)]
    def value(eta):
        if na is None:return ll(c,p,eta)
        ra=na-ll(a,p,eta) if na!=mp.ninf else mp.ninf
        rb=nb-ll(b,p,eta) if nb!=mp.ninf else mp.ninf
        if ra==rb==mp.ninf:return mp.ninf
        mx=max(ra,rb);return mx+mp.log((mp.exp(ra-mx)+mp.exp(rb-mx))/2)
    def gradient(eta):
        slope=[(mp.mpf(1)/m-v)/((1-eta)*v+eta/m) for v in p]
        if na is None:return mp.fsum(int(x)*s for x,s in zip(c,slope))
        ra=na-ll(a,p,eta) if na!=mp.ninf else mp.ninf
        rb=nb-ll(b,p,eta) if nb!=mp.ninf else mp.ninf
        if ra==rb==mp.ninf:return mp.mpf(0)
        mx=max(ra,rb);wa,wb=mp.exp(ra-mx),mp.exp(rb-mx)
        da=-mp.fsum(int(x)*s for x,s in zip(a,slope));db=-mp.fsum(int(x)*s for x,s in zip(b,slope))
        return (wa*da+wb*db)/(wa+wb)
    lo,hi=mp.mpf(0),mp.mpf(1)
    for _ in range(110):
        mid=(lo+hi)/2;g=gradient(mid)
        if g>0 if na is None else g<0:lo=mid
        else:hi=mid
    vals=[value((lo+hi)/2),value(mp.mpf(0)),value(mp.mpf(1))]
    return max(vals) if na is None else min(vals)

@lru_cache(maxsize=2)
def reference_logs(n):
    energies=np.linspace(-3.75,3.75,32*2**n,endpoint=False);etas=np.linspace(0,1,17)
    logs=[]
    for e in energies:
        p=kernel(float(e),n)
        for eta in etas[:-1]:
            q=[(1-mp.mpf(float(eta)))*v+mp.mpf(float(eta))/len(p) for v in p]
            logs.append(tuple(int(mp.floor(mp.log(v)*SCALE)) if v>0 else None for v in q))
    # eta=1 has one identical law for each energy; collapse multiplicity exactly.
    return logs,len(energies)

def independent_mixture(c,n):
    logs,ne=reference_logs(n);used=[(i,int(x)) for i,x in enumerate(c) if x]
    logits=[]
    for row in logs:
        if any(row[i] is None for i,x in used):continue
        logits.append(sum(row[i]*x for i,x in used))
    total_count=int(sum(c));uniform=-total_count*mp.log(2**n)
    anchor=max(uniform,max(mp.mpf(x)/SCALE for x in logits));values=[];skipped=0
    for x in logits:
        z=mp.mpf(x)/SCALE-anchor
        if z<-120:skipped+=1
        else:values.append(mp.exp(z))
    total=mp.fsum(values)+ne*mp.exp(uniform-anchor)
    lower=anchor+mp.log(total)-mp.log(ne*17)
    upper=anchor+mp.log(total+skipped*mp.exp(-120))-mp.log(ne*17)+mp.mpf(total_count)/SCALE
    return lower,upper

def audit_task(task):
    index,r,s,mix=task;a,b=r['batch_a'],r['batch_b'];c=np.array(a)+b;n=r['n'];truth=r['energy'];p=kernel(truth,n)
    cf=s['fits']['crossfit'];pa=kernel(cf['fit_a'][0],n);pb=kernel(cf['fit_b'][0],n)
    na=ll(a,pb,cf['fit_b'][1]);nb=ll(b,pa,cf['fit_a'][1]);log_e=optimize(p,a,b,na,nb)
    null=optimize(p,a,b);fit=s['fits']['profile'];alternative=ll(c,kernel(fit['energy'],n),fit['eta']);stat=2*(alternative-null)
    ml,mh=independent_mixture(c,n);mix_stat_hi=mh-null;mix_stat_lo=ml-null;cutoff=-mp.log(mp.mpf('.05'))
    expected={method:bool(v) for method,v in [('crossfit',log_e<=cutoff),('profile',stat<=mp.mpf(float(2*(fit['maximum']-fit['threshold'])))),('qpe_mixture',mix_stat_hi<=cutoff)]}
    if (mix_stat_lo<=cutoff)!=(mix_stat_hi<=cutoff):raise RuntimeError('High-precision mixture classification ambiguous')
    production={'crossfit':s['point_cf'],'profile':s['point_plr'],'qpe_mixture':mix['metrics']['qpe_mixture']['point_covered']}
    if production!=expected:raise RuntimeError(f'Truth audit mismatch {r["case"]}/{r["trial"]}: {production}/{expected}')
    discrepancy=abs(float(ml)-mix['fits']['qpe_mixture']['log_predictive'])
    if discrepancy>1e-7:raise RuntimeError(f'Mixture numerator discrepancy {discrepancy}')
    return dict(case=r['case'],trial=r['trial'],index=index,decisions=expected,log_e=str(log_e),profile_statistic=str(stat),mixture_log_e_lower=str(mix_stat_lo),mixture_log_e_upper=str(mix_stat_hi),mixture_log_enclosure_width=str(mh-ml),mixture_double_discrepancy=discrepancy,crossfit_numerator_max_discrepancy=max(abs(float(na)-cf['numerator_log_a']),abs(float(nb)-cf['numerator_log_b'])),profile_alternative_discrepancy=abs(float(alternative)-fit['maximum']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    old=read(SRC/'continuous_inversion/results/raw_counts.json');sets=read(SRC/'continuous_inversion/results/sets.json');rows=list(csv.DictReader((SRC/'continuous_inversion/results/trials.csv').open()))
    mixtures={(x['record']['case'],x['record']['trial']):x for x in read(ROOT/'results/mixture/analysis.json')}
    tasks=[]
    for r,s,row in zip(old,sets,rows):
        if r['group'] not in ('paired','transition'):continue
        s=dict(s,point_cf=row['crossfit_point_covered']=='True',point_plr=row['profile_point_covered']=='True');tasks.append((len(tasks),r,s,mixtures[(r['case'],r['trial'])]))
    if args.benchmark:
        start=time.perf_counter();result=audit_task(tasks[0]);print('first audit seconds',time.perf_counter()-start);print(result);return
    out=ROOT/'qa/truth_audit';out.mkdir(parents=True,exist_ok=True)
    seal=dict(protocol=digest(ROOT/'analysis_amendment_2026-10-05.md'),audit=digest(Path(__file__)),counts=digest(SRC/'continuous_inversion/results/raw_counts.json'),sets=digest(SRC/'continuous_inversion/results/sets.json'),mixture=digest(ROOT/'results/mixture/analysis.json'))
    if (out/'seal.json').exists():assert read(out/'seal.json')==seal
    else:save(out/'seal.json',seal)
    remaining=[t for t in tasks if not (out/f'{t[0]:05d}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task,result in zip(remaining,pool.map(audit_task,remaining,chunksize=10)):
            save(out/f'{task[0]:05d}.json',result)
            if (task[0]+1)%100==0:print('truth audit',task[0]+1,'/',len(tasks),flush=True)
    rr=[read(out/f'{i:05d}.json') for i in range(len(tasks))]
    save(ROOT/'qa/truth_audit.json',dict(status='passed',datasets=len(rr),method_decisions=3*len(rr),precision_digits=50,mixture_fixed_log_digits=45,maximum_mixture_double_discrepancy=max(x['mixture_double_discrepancy'] for x in rr),maximum_cf_numerator_discrepancy=max(x['crossfit_numerator_max_discrepancy'] for x in rr),maximum_profile_alternative_discrepancy=max(x['profile_alternative_discrepancy'] for x in rr),scope='Independent truth rules and saved-fit likelihoods. Mixture uses 50-digit kernels/logs with explicit 45-decimal fixed-log and exp-tail enclosure. No endpoint or global-MLE certification.',reports=rr))
    print('TRUTH AUDIT COMPLETE',len(rr),flush=True)
if __name__=='__main__':main()
