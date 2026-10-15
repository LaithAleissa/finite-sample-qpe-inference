"""Render clarified article figures from unchanged, saved scientific evidence."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'src/qpe_study'
TARGET=Path(sys.argv[1]).resolve()
TARGET.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(TARGET/f'{name}.{ext}',bbox_inches='tight',dpi=210)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    names={'crossfit':'CF','qpe_mixture':'QPM','dirichlet':'DIR','cp':'CP','profile':'PLR'}
    colors={'crossfit':'#087f8c','qpe_mixture':'#7356a3','dirichlet':'#687569','cp':'#c49212','profile':'#bd5c34'}
    markers={'crossfit':'o','qpe_mixture':'s','dirichlet':'D','cp':'x','profile':'^'}
    combined=read(ROOT/'results/final_scientific/matched_comparator_summary.json')
    fig,ax=plt.subplots(1,3,figsize=(8,3),layout='constrained')
    for i,m in enumerate(names):
        rr=[x for x in combined if x['method']==m];xx=np.arange(1,9)+(i-2)*.14
        y=np.array([x['point_covered']/200 for x in rr]);lo=np.array([x['mc_lower'] for x in rr]);hi=np.array([x['mc_upper'] for x in rr])
        ax[0].errorbar(xx,y,yerr=[y-lo,hi-y],fmt=markers[m],ms=3,capsize=1.5,color=colors[m],label=names[m])
        y=np.array([x['median_length'] for x in rr]);lo=np.array([x['q25'] for x in rr]);hi=np.array([x['q75'] for x in rr])
        ax[1].errorbar(xx,y,yerr=[y-lo,hi-y],fmt=markers[m],ms=3,capsize=1.5,color=colors[m])
    ax[0].axhline(.95,color='.5',ls='--',lw=.8)
    ax[0].set(ylabel='Truth-point inclusion',ylim=(.5,1.04),title='(a) Inclusion and calibration');ax[0].legend(ncol=2,fontsize=6,loc='lower left')
    ax[1].set(yscale='log',ylabel='Union length (model units)',title='(b) Localization: median / IQR')
    regret=list(csv.DictReader((ROOT/'results/postreview/qpm_regret.csv').open()))
    y=np.array([float(x['median_cutoff']) for x in regret]);lo=np.array([float(x['minimum_cutoff']) for x in regret]);hi=np.array([float(x['maximum_cutoff']) for x in regret])
    ax[2].errorbar(range(1,9),y,yerr=[y-lo,hi-y],fmt='s',ms=3,capsize=2,color=colors['qpe_mixture'],label='QPM median / range')
    ax[2].axhline(3.841458820694124,color=colors['profile'],ls='--',label='PLR cutoff')
    ax[2].axhline(2*np.log(20),color='.5',ls=':',label='QPM lower bound')
    ax[2].set(ylabel='Profile deviance cutoff',ylim=(0,35),title='(c) QPM cutoff from regret');ax[2].legend(fontsize=6,loc='lower left')
    for a in ax:a.set(xticks=range(1,9),xlabel='Primary design (8: uniform)')
    save(fig,'03_primary')

    rows=read(ROOT/'results/physical_bridge/summary.json')
    examples=read(ROOT/'results/physical_bridge/examples.json')
    fig,axs=plt.subplots(2,2,figsize=(8,5.6),layout='constrained')
    physical_names={m:names[m] for m in ('crossfit','qpe_mixture','profile')}
    for i,m in enumerate(physical_names):
        rr=[r for r in rows if r['method']==m];x=np.arange(4)+(i-1)*.18;y=np.array([r['gap_covered']/200 for r in rr])
        axs[0,0].errorbar(x,y,yerr=[y-np.array([r['mc_lower'] for r in rr]),np.array([r['mc_upper'] for r in rr])-y],fmt='o',ms=3,capsize=2,color=colors[m],label=names[m])
        axs[0,1].plot(x,[r['median_length_K'] for r in rr],'o',color=colors[m],ms=4)
    axs[0,0].axhline(.95,ls='--',color='.5');axs[0,0].set(ylabel='Gap inclusion proportion',ylim=(.8,1.02),title='(a) Ensemble: gap inclusion');axs[0,0].legend(fontsize=7)
    axs[0,1].set(ylabel='Median AF gap-set length (K)',yscale='log',title='(b) Ensemble: localization')
    for a in axs[0]:a.set(xticks=np.arange(4),xticklabels=['4 / .10','6 / .10','4 / .95','6 / .95'],xlabel='n / generating replacement fraction')
    for j,(ax,ex) in enumerate(zip(axs[1],[examples[1],examples[2]])):
        t=np.array(ex['temperatures']);ref=np.array(ex['reference'])
        for m in physical_names:
            image=ex['methods'][m]
            if image['lower']:
                lo=100*(np.array(image['lower'])/ref-1);hi=100*(np.array(image['upper'])/ref-1)
                ax.fill_between(t,lo,hi,color=colors[m],alpha=.12)
                ax.plot(t,lo,color=colors[m],lw=.8);ax.plot(t,hi,color=colors[m],lw=.8,label=names[m])
        ax.axhline(0,color='.4',ls=':');ax.set_yscale('symlog',linthresh=1)
        nlabel,etalabel=ex['case'].split('_')
        ax.set(xlabel='Temperature (K)',ylabel='Response change from reference (%)',title=rf'({chr(99+j)}) Single trial 0: $n={nlabel[1:]},\eta={etalabel[3:]}$')
        ax.legend(fontsize=6,loc='lower right')
    save(fig,'18_unknown_gap_response')

    sys.path.insert(0,str(ROOT.parent/'material_model'))
    from physics import bleaney_bowers,G
    material=read(ROOT/'results/material_refined.json');lo_gap,hi_gap=material['gap_confidence_set_K'][0]
    fig,ax=plt.subplots(figsize=(7.3,2.8),layout='constrained')
    ranges=[(2,(lo_gap,hi_gap),'#306b87'),(1,(412.38,493.27),'#d55a3a'),(0,(397.08,427.51),'#448b49')]
    for y,(lo,hi),color in ranges:
        ax.axhspan(y-.30,y+.30,color=color,alpha=.05,zorder=-1)
        ax.plot([lo,hi],[y,y],color=color,lw=2)
        for v in (lo,hi):ax.plot([v,v],[y-.06,y+.06],color=color)
    ax.axvline(412.38,color='.6',ls=':',lw=1)
    ax.set(yticks=[2,1,0],yticklabels=['Statistical confidence set\n95%; fixed QPE/material inputs','Model-choice sensitivity range','Deletion sensitivity range'],xlabel='Gap (K)',title='Different uncertainty questions',ylim=(-.35,2.35))
    ax.grid(axis='x',alpha=.2);save(fig,'14_uncertainty_scopes')
    t=np.linspace(50,350,301);gpow=np.sqrt((G[0]**2+2*G[1]**2)/3)
    base=bleaney_bowers(t,material['anchor_K'],gpow);middle=bleaney_bowers(t,material['estimated_gap_K'],gpow)
    low=bleaney_bowers(t,hi_gap,gpow);high=bleaney_bowers(t,lo_gap,gpow)
    fig,ax=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    ax[0].fill_between(t,low*1000,high*1000,alpha=.4,color='#448b77',label='95% envelope; fixed inputs')
    ax[0].plot(t,base*1000,label='Literature-calibrated reference');ax[0].plot(t,middle*1000,'--',label='Single prespecified trial 0')
    ax[0].set(xlabel='Temperature (K)',ylabel='Susceptibility ($10^{-3}$ cm$^3$/mol Cu)',title='(a) Single-trial response illustration');ax[0].legend(fontsize=7)
    ax[1].fill_between(t,100*(low/base-1),100*(high/base-1),alpha=.4,color='#448b77')
    ax[1].plot(t,100*(middle/base-1),'--');ax[1].axhline(0,color='.5',lw=.8)
    ax[1].set(xlabel='Temperature (K)',ylabel='Change from fixed-gap curve (%)',title='(b) Same trial; fixed material inputs')
    save(fig,'08_propagation')
    print('Rendered clarified comparison and response figures from saved evidence.')

if __name__=='__main__':main()
