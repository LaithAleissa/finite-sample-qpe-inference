"""One checked evidence bridge for authoring and aggregate reproduction."""
from studies import *

def inherited_items():
    raw=read(SRC/'continuous_inversion/results/raw_counts.json');sets=read(SRC/'continuous_inversion/results/sets.json')
    rows=list(csv.DictReader((SRC/'continuous_inversion/results/trials.csv').open()))
    output=[]
    for r,s,t in zip(raw,sets,rows):
        assert (r['case'],r['trial'])==(s['case'],s['trial'])==(t['case'],int(t['trial']))
        x=dict(record=r,metrics={},sets={},fits=s['fits'])
        for m in ('crossfit','profile'):
            x['sets'][m]=s[m]
            x['metrics'][m]={k:(t[m+'_'+src]=='True' if kind=='bool' else kind(t[m+'_'+src])) for k,src,kind in [('point_covered','point_covered','bool'),('set_covered','covered','bool'),('length','length',float),('components','components',int),('full_branch','full_branch','bool'),('empty','empty','bool'),('seconds','seconds',float),('cells','visited_cells',int)]}
        output.append(x)
    return output

def merged():
    mix={(x['record']['case'],x['record']['trial']):x for x in read(ROOT/'results/mixture/analysis.json')};output=[]
    for x in inherited_items():
        key=x['record']['case'],x['record']['trial']
        if key in mix:
            for name in ('metrics','sets','fits'):x[name].update(mix[key][name])
        output.append(x)
    return output

def ratios(items):
    rng=np.random.default_rng(2026100504);rows=[]
    for case in dict.fromkeys(x['record']['case'] for x in items):
        rr=[x for x in items if x['record']['case']==case]
        if 'qpe_mixture' not in rr[0]['metrics']:continue
        ix=rng.integers(0,len(rr),size=(5000,len(rr)))
        for numerator,denominator in [('crossfit','profile'),('qpe_mixture','profile'),('qpe_mixture','crossfit')]:
            a=np.array([x['metrics'][numerator]['length'] for x in rr]);b=np.array([x['metrics'][denominator]['length'] for x in rr]);den=np.median(b[ix],axis=1)
            if np.any(den<=0):raise RuntimeError('Zero ratio denominator; explicit report required')
            boot=np.median(a[ix],axis=1)/den
            rows.append(dict(case=case,numerator=numerator,denominator=denominator,ratio=float(np.median(a)/np.median(b)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975))))
    return rows

def main():
    items=merged();summ=summarize(items,'unknown')
    csvsave(ROOT/'results/combined_summary.csv',summ);csvsave(ROOT/'results/paired_width_ratios.csv',ratios(items))
    headline={}
    for group in ('paired','transition','register','misspecification'):
        rr=[x for x in items if x['record']['group']==group];headline[group]={}
        for method in rr[0]['metrics']:
            headline[group][method]={key:sum(x['metrics'][method][key] for x in rr) for key in ('point_covered','set_covered','full_branch','empty')}
    primary=[x for x in items if x['record']['group']=='paired' and x['record']['eta']<1]
    headline['primary_nonuniform']={m:sum(x['metrics'][m]['point_covered'] for x in primary) for m in primary[0]['metrics']}
    save(ROOT/'results/headline.json',headline)
    print(json.dumps(headline,indent=2))
if __name__=='__main__':main()
