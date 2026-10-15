"""Post-analysis numerical diagnostic; does not replace frozen estimates."""
from studies import *
def main():
    raw=read(ROOT/'results/mixture/analysis.json');cases=list(dict.fromkeys(x['record']['case'] for x in raw if x['record']['group'] in ('paired','transition')))
    reports=[]
    for case in cases:
        x=next(v for v in raw if v['record']['case']==case and v['record']['trial']==0);r=x['record'];ex=Experiment(r['n'],.8377580409572781,(-3.75,3.75))
        obj=QPEMixture(ex,r['batch_a'],r['batch_b']);fine=obj.confidence_set(ex.bin_width/100000)
        coarse=x['sets']['qpe_mixture'];cl,fl=length(coarse),length(fine)
        assert fl<=cl+1e-10
        for lo,hi in fine['intervals']:assert any(a<=lo+1e-12 and hi<=b+1e-12 for a,b in coarse['intervals'])
        assert not obj.membership(r['energy'])[0] or any(lo<=r['energy']<=hi for lo,hi in fine['intervals'])
        reports.append(dict(case=case,coarse_length=cl,fine_length=fl,relative_length_change=0 if cl==0 else (cl-fl)/cl,coarse_components=len(coarse['intervals']),fine_components=len(fine['intervals'])))
    save(ROOT/'qa/mixture_resolution.json',dict(status='passed',selection='trial zero of each primary and transition design; post-analysis diagnostic only',datasets=len(reports),maximum_relative_length_change=max(x['relative_length_change'] for x in reports),reports=reports))
    print('Mixture tenfold resolution diagnostic passed',len(reports))
if __name__=='__main__':main()
