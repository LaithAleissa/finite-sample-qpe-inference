from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'src/qpe_study'
OUT=ROOT/'results'
TARGET=Path(sys.argv[1]).resolve()
TARGET.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def savefig(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(TARGET/f'{name}.{ext}',bbox_inches='tight',dpi=210)
    plt.close(fig)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

combined=read(OUT/'final_scientific/matched_comparator_summary.json')
scaling=read(OUT/'final_scientific/scaling_summary.json')
drift=read(OUT/'final_scientific/drift_summary.json')
names={'crossfit':'CF','qpe_mixture':'QPM','dirichlet':'DIR','cp':'CP','profile':'PLR'}
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors={'crossfit':'#087f8c','qpe_mixture':'#7356a3','dirichlet':'#687569','cp':'#c49212','profile':'#bd5c34'}
markers={'crossfit':'o','qpe_mixture':'s','dirichlet':'D','cp':'x','profile':'^'}
fig,ax=plt.subplots(1,3,figsize=(8,3),layout='constrained')
for i,m in enumerate(names):
    rr=[x for x in combined if x['method']==m];xx=np.arange(1,9)+(i-2)*.14
    y=np.array([x['point_covered']/200 for x in rr]);lo=np.array([x['mc_lower'] for x in rr]);hi=np.array([x['mc_upper'] for x in rr])
    ax[0].errorbar(xx,y,yerr=[y-lo,hi-y],fmt=markers[m],ms=3,capsize=1.5,color=colors[m],label=names[m])
    y=np.array([x['median_length'] for x in rr]);lo=np.array([x['q25'] for x in rr]);hi=np.array([x['q75'] for x in rr])
    ax[1].errorbar(xx,y,yerr=[y-lo,hi-y],fmt=markers[m],ms=3,capsize=1.5,color=colors[m])
ax[0].axhline(.95,color='.5',ls='--',lw=.8);ax[0].set(ylabel='Truth-point inclusion',ylim=(.5,1.04),title='(a) Valid and approximate rules');ax[0].legend(ncol=2,fontsize=6,loc='lower left')
ax[1].set(yscale='log',ylabel='Complete union length',title='(b) Median and IQR')
regret=list(csv.DictReader((ROOT/'results/postreview/qpm_regret.csv').open()));y=np.array([float(x['median_cutoff']) for x in regret]);lo=np.array([float(x['minimum_cutoff']) for x in regret]);hi=np.array([float(x['maximum_cutoff']) for x in regret])
ax[2].errorbar(range(1,9),y,yerr=[y-lo,hi-y],fmt='s',ms=3,capsize=2,color=colors['qpe_mixture'],label='QPM median/range')
ax[2].axhline(3.841458820694124,color=colors['profile'],ls='--',label='PLR cutoff');ax[2].axhline(2*np.log(20),color='.5',ls=':',label='QPM lower bound')
ax[2].set(ylabel='Profile deviance cutoff',ylim=(0,35),title='(c) Predictive regret');ax[2].legend(fontsize=6,loc='lower left')
for a in ax:a.set(xticks=range(1,9),xlabel='Primary design (8: uniform)')
for ext in ['pdf','png','svg']:fig.savefig(TARGET/f'03_primary.{ext}',dpi=210)
plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(8,5.2),layout='constrained')
for j,e in enumerate([1.013,-3.7499]):
    for m,color,marker,label in [('crossfit','#087f8c','o','CF'),('qpm_fixed','#7356a3','s','Fixed QPM'),('qpm_growing','#c49212','^','Growing QPM')]:
        rr=[x for x in scaling if x['energy']==e and x['method']==m];y=np.array([x['median_length'] for x in rr]);lo=np.array([x['q25'] for x in rr]);hi=np.array([x['q75'] for x in rr])
        ax[0,j].errorbar([x['N'] for x in rr],y,yerr=[y-lo,hi-y],marker=marker,color=color,ms=3,lw=1,capsize=2,label=label)
    ax[0,j].set(xscale='log',yscale='log',xlabel='Total shots',ylabel='Median union length',title=f'({chr(97+j)}) E = {e}');ax[0,j].legend(fontsize=7)
pairs=['.10/.10','.10/.20','.20/.40','.40/.80','.95/.98']
for j,n in enumerate([4,6]):
    for e,color,marker in [(1.013,'#087f8c','o'),(-3.7499,'#7356a3','s')]:
        aa=[x for x in drift if x['n']==n and x['energy']==e and x['method']=='crossfit'];bb=[x for x in drift if x['n']==n and x['energy']==e and x['method']=='drift_cf']
        ax[1,j].plot(range(5),[b['median_length']/a['median_length'] for a,b in zip(aa,bb)],marker=marker,color=color,ms=4,lw=1,label=f'E = {e}')
    ax[1,j].axhline(1,color='.5',ls=':',lw=.8);ax[1,j].set(xticks=range(5),xticklabels=pairs,ylim=(.995,1.10),ylabel='Drift/shared median length',xlabel='Batch fractions',title=f'({chr(99+j)}) {n} counting qubits');ax[1,j].legend(fontsize=7)
for ext in ['pdf','png','svg']:fig.savefig(TARGET/f'20_scaling_drift.{ext}',dpi=210)
plt.close(fig)
