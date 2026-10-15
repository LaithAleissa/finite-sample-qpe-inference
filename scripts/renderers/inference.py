"""Publication figures from sealed observations and complete numerical unions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/qpe_study/code'))
from evidence import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'crossfit':'#087f8c','qpe_mixture':'#7356a3','profile':'#bd5c34','lr':'#bd5c34','dirichlet':'#7356a3'}
MARKERS={'crossfit':'o','qpe_mixture':'s','profile':'^','lr':'^','dirichlet':'s'}
STYLES={'crossfit':'-','qpe_mixture':'--','profile':':'}
NAMES={'crossfit':'Cross-fit','qpe_mixture':'QPE mixture','profile':'Profile LR','lr':'Known-noise LR','dirichlet':'Dirichlet ratio'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def savefig(fig,name):
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',dpi=210)
    plt.close(fig)
def draw_comparison(rows,methods,name,title):
    primary=name=='03_primary'
    fig,ax=plt.subplots(1,3 if primary else 2,figsize=(8,3.05),layout='constrained');cases=list(dict.fromkeys(x['case'] for x in rows))
    for i,method in enumerate(methods):
        rr=[next(r for r in rows if r['case']==case and r['method']==method) for case in cases];x=np.arange(1,len(cases)+1)+(i-1)*.18
        y=np.array([int(r['point_covered'])/int(r['trials']) for r in rr]);lo=np.array([float(r['mc_lower']) for r in rr]);hi=np.array([float(r['mc_upper']) for r in rr])
        ax[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt=MARKERS[method],ms=3,capsize=2,color=COLORS[method],label=NAMES[method])
        y=np.array([float(r['median_length']) for r in rr]);lo=np.array([float(r['q25']) for r in rr]);hi=np.array([float(r['q75']) for r in rr])
        ax[1].errorbar(x,y,yerr=[y-lo,hi-y],fmt=MARKERS[method],ms=3,capsize=2,color=COLORS[method])
    ax[0].axhline(.95,ls='--',color='.45');ax[0].set(ylabel='Truth-point inclusion',ylim=(.5,1.03),title='(a) Point rule and MC intervals');ax[0].legend(fontsize=7,loc='lower left')
    ax[1].axhline(7.5,ls=':',color='.5');ax[1].set(yscale='log',ylabel='Total union length (model units)',title='(b) Median and IQR')
    if primary:
        rr=list(csv.DictReader((ROOT/'results/postreview/qpm_regret.csv').open()))
        assert [r['case'] for r in rr]==cases
        y=np.array([float(r['median_cutoff']) for r in rr]);lo=np.array([float(r['minimum_cutoff']) for r in rr]);hi=np.array([float(r['maximum_cutoff']) for r in rr])
        ax[2].errorbar(range(1,len(cases)+1),y,yerr=[y-lo,hi-y],fmt='o',ms=3,capsize=2,color=COLORS['qpe_mixture'],label='QPM median/range')
        ax[2].axhline(3.841458820694124,ls='--',color=COLORS['profile'],label='PLR: 3.8415')
        ax[2].axhline(2*np.log(20),ls=':',color='.5',label='QPM lower bound')
        ax[2].set(ylabel='Profile deviance cutoff',title='(c) Predictive regret',ylim=(0,35));ax[2].legend(fontsize=6,loc='lower left')
    for a in ax:a.set(xlabel=title,xticks=range(1,len(cases)+1));a.tick_params(axis='x',labelsize=7)
    savefig(fig,name)

def draw_transition(summary):
    fig,ax=plt.subplots(2,2,figsize=(8,5.5),layout='constrained')
    etas=[.4,.6,.8,.9,.95,.98,1]
    for row,shots in enumerate((128,512)):
        for i,m in enumerate(('crossfit','qpe_mixture','profile')):
            rr=[r for r in summary if r['group']=='transition' and int(r['shots_per_batch'])==shots and r['method']==m]
            y=np.array([int(r['point_covered'])/200 for r in rr]);lo=np.array([float(r['mc_lower']) for r in rr]);hi=np.array([float(r['mc_upper']) for r in rr]);x=np.arange(7)+(i-1)*.12
            ax[row,0].errorbar(x,y,yerr=[y-lo,hi-y],fmt=MARKERS[m],ms=3,capsize=2,color=COLORS[m],label=NAMES[m])
            ax[row,1].plot(x,[float(r['median_length']) for r in rr],marker=MARKERS[m],linestyle='none',ms=3,color=COLORS[m],label=NAMES[m])
        ax[row,0].axhline(.95,ls='--',color='.5');ax[row,0].set(ylabel='Truth-point inclusion',ylim=(.45,1.025),title=f'({chr(97+row*2)}) {2*shots} total shots')
        ax[row,1].axhline(7.5,ls=':',color='.5');ax[row,1].set(yscale='log',ylabel='Median union length',title=f'({chr(98+row*2)}) Identification and length')
        for a in ax[row]:a.set(xticks=range(7),xticklabels=etas,xlabel='Replacement fraction (design positions)')
    ax[0,0].legend(fontsize=7,loc='lower left');savefig(fig,'04_transition')

def main():
    summary=list(csv.DictReader((ROOT/'results/combined_summary.csv').open()))
    known=list(csv.DictReader((ROOT/'results/known/summary.csv').open()))
    draw_comparison(known,['crossfit','lr','dirichlet'],'02_known','Known-channel design')
    draw_comparison([r for r in summary if r['group']=='paired'],['crossfit','qpe_mixture','profile'],'03_primary','Primary design (8: uniform)')
    draw_transition(summary)
    items=merged();example=next(x for x in items if x['record']['case']=='branch_noise_010' and x['record']['trial']==0)
    fig,ax=plt.subplots(1,2,figsize=(8,2.5),layout='constrained')
    for j,(lo,hi) in enumerate(((-3.75,-3.55),(3.55,3.75))):
        for i,m in enumerate(('crossfit','qpe_mixture','profile')):
            for a,b in example['sets'][m]['intervals']:
                if b>=lo and a<=hi:ax[j].plot([max(a,lo),min(b,hi)],[i,i],lw=7,color=COLORS[m],solid_capstyle='butt')
        ax[j].set(xlim=(lo,hi),ylim=(-.5,2.5),yticks=range(3),yticklabels=[NAMES[m] for m in ('crossfit','qpe_mixture','profile')],xlabel='Energy (model units)',title='(a) Left seam' if j==0 else '(b) Right seam')
        if j==0:ax[j].axvline(-3.7499,color='black',ls='--',lw=1)
    savefig(fig,'05_alias')
    diagnostics=read(ROOT/'results/circuit/distribution_diagnostics.json');circuit=list(csv.DictReader((ROOT/'results/circuit/summary.csv').open()))
    fig,ax=plt.subplots(1,3,figsize=(8,2.7),layout='constrained')
    for a,d in zip(ax,diagnostics):
        x=np.arange(16);a.bar(x-.17,d['observed'],width=.34,color='#306b87',label='Noisy circuit');a.bar(x+.17,d['fitted'],width=.34,color='#d59a5b',label='Replacement fit')
        a.set(title=f"{d['case'].capitalize()}: TV={d['total_variation']:.3f}",xlabel='Counting output bin',ylabel='Probability',xticks=[0,4,8,12])
    ax[0].legend(fontsize=7);savefig(fig,'06_circuit')
    if (ROOT/'results/bootstrap/summary.csv').exists():
        rows=list(csv.DictReader((ROOT/'results/bootstrap/summary.csv').open()));fig,ax=plt.subplots(1,2,figsize=(8,3),layout='constrained')
        for j,shots in enumerate((128,512)):
            for i,(method,color) in enumerate((('bootstrap','#7356a3'),('conventional','#bd5c34'))):
                rr=[r for r in rows if int(r['shots_per_batch'])==shots and r['method']==method];y=np.array([int(r['accepted'])/int(r['trials']) for r in rr]);lo=np.array([float(r['mc_lower']) for r in rr]);hi=np.array([float(r['mc_upper']) for r in rr]);x=np.arange(4)+(i-.5)*.12
                ax[j].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',color=color,capsize=2,label='Plug-in bootstrap' if i==0 else 'Conventional PLR')
            ax[j].axhline(.95,ls='--',color='.5');ax[j].set(xticks=range(4),xticklabels=[.9,.95,.98,1],xlabel='Replacement fraction',ylabel='Candidate truth accepted',ylim=(.1,1.04),title=f'({chr(97+j)}) {2*shots} total shots; 25 trials/design')
        ax[0].legend(fontsize=7,loc='lower left');savefig(fig,'07_bootstrap')
    if (ROOT/'results/replication/summary.csv').exists():
        rows=list(csv.DictReader((ROOT/'results/replication/summary.csv').open()));draw_comparison(rows,['crossfit','qpe_mixture','profile'],'08_replication','Fresh design (interior / seam / near-uniform / uniform)')
    sys.path.insert(0,str(SRC/'material_model'))
    from physics import bleaney_bowers,G
    material=read(ROOT/'results/material_refined.json');gap_lo,gap_hi=material['gap_confidence_set_K'][0]
    fig,ax=plt.subplots(figsize=(7.3,2.8),layout='constrained')
    for y,(lo,hi),color in [(2,(gap_lo,gap_hi),'#306b87'),(1,(412.38,493.27),'#d55a3a'),(0,(397.08,427.51),'#448b49')]:
        ax.plot([lo,hi],[y,y],color=color,lw=2);ax.plot([lo,lo],[y-.06,y+.06],color=color);ax.plot([hi,hi],[y-.06,y+.06],color=color)
    ax.axvline(412.38,color='.6',ls=':',lw=1);ax.set(yticks=[2,1,0],yticklabels=['Conditional QPE gap set\n95%, fixed scale and channel','Fit choices\nrange, not a confidence interval','Leave-one-out fits\nrange, not a confidence interval'],xlabel='Gap / K',title='Different uncertainty questions on the same energy scale',ylim=(-.35,2.35));ax.grid(axis='x',alpha=.2);savefig(fig,'14_uncertainty_scopes')
    t=np.linspace(50,350,301);gpow=np.sqrt((G[0]**2+2*G[1]**2)/3)
    base=bleaney_bowers(t,material['anchor_K'],gpow);middle=bleaney_bowers(t,material['estimated_gap_K'],gpow);low=bleaney_bowers(t,gap_hi,gpow);high=bleaney_bowers(t,gap_lo,gpow)
    fig,ax=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    ax[0].fill_between(t,low*1000,high*1000,alpha=.4,color='#448b77',label='Conditional 95% envelope');ax[0].plot(t,base*1000,label='Fixed literature gap');ax[0].plot(t,middle*1000,'--',label='One QPE gap trial');ax[0].set(xlabel='Temperature (K)',ylabel='Susceptibility ($10^{-3}$ cm$^3$/mol Cu)',title='(a) Classical spin susceptibility');ax[0].legend(fontsize=7)
    ax[1].fill_between(t,100*(low/base-1),100*(high/base-1),alpha=.4,color='#448b77');ax[1].plot(t,100*(middle/base-1),'--');ax[1].axhline(0,color='.5',lw=.8);ax[1].set(xlabel='Temperature (K)',ylabel='Change from fixed-gap curve (%)',title='(b) Conditional gap effect')
    savefig(fig,'08_propagation')
    print('Evidence figures rebuilt')
if __name__=='__main__':main()
