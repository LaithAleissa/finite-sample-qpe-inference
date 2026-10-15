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

OUT=ROOT/'results/presubmission2'
q=list(csv.DictReader((OUT/'qpm/summary.csv').open()))
c=list(csv.DictReader((OUT/'circuit/summary.csv').open()))
d=read(OUT/'circuit/distribution_diagnostics.json')
cases=list(dict.fromkeys(x['case'] for x in q))
labels={case:('P'+str(i+1) if i<8 else 'T'+str(i-7)) for i,case in enumerate(cases)}
sys.path.insert(0,str(ROOT/'code'))
from presubmission2 import REFERENCES,ENERGIES,RATES
COLORS={'crossfit':'#087f8c','qpe_mixture':'#7356a3'}
fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
colors=['#7356a3','#087f8c','#c47d16']
for ax,sel,title in zip(axes,[cases[:7],cases[8:]],['(a) Nonuniform primary cells','(b) Weak-identification cells']):
    for j,name in enumerate(REFERENCES):
        xx=[next(x for x in q if x['case']==case and x['reference']==name) for case in sel];v=np.array([float(x['ratio_to_CF']) for x in xx]);lo=np.array([float(x['ratio_lower']) for x in xx]);hi=np.array([float(x['ratio_upper']) for x in xx])
        ax.errorbar(np.arange(len(sel))+(j-1)*.16,v,yerr=[v-lo,hi-v],fmt='o',ms=3,capsize=2,color=colors[j],label=['Base','Dense','N-adaptive'][j])
    ax.axhline(1,color='.45',ls='--');ax.set(title=title,xticks=range(len(sel)),xticklabels=[labels[z] for z in sel],ylabel='Ratio of median lengths: QPM / CF',ylim=(.95,2.5) if len(sel)==7 else (.98,1.16));ax.legend(fontsize=7)
savefig(fig,'13_qpm_sensitivity')
# Nine exact distributions; no outcome-selected illustrative histogram.
fig,axes=plt.subplots(3,3,figsize=(8,5.3),layout='constrained')
for ei,e in enumerate(ENERGIES):
    for ni,(level,*_) in enumerate(RATES):
        z=next(x for x in d if x['case']==f'e{ei}_{level}');ax=axes[ei,ni]
        ax.bar(range(16),z['observed'],color='#aacbd0',label='Exact noisy circuit');ax.plot(range(16),z['fitted'],'o-',color='#bd5c34',ms=2,lw=1,label='Replacement fit')
        ax.set(title=f'E={e:g}; {level}; TV={z["total_variation"]:.3f}',xticks=[0,4,8,12,15]);ax.tick_params(labelsize=7)
        if ni==0:ax.set_ylabel('Probability')
        if ei==2:ax.set_xlabel('Counting bin')
axes[0,0].legend(fontsize=6);savefig(fig,'14_circuit_phase_distributions')
fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
for ei,e in enumerate(ENERGIES):
    xx=[next(x for x in d if x['case']==f'e{ei}_{level}') for level,*_ in RATES]
    axes[0].plot(range(3),[z['total_variation'] for z in xx],'o-',label=f'E={e:g}')
axes[0].set(xticks=range(3),xticklabels=['Low','Medium','High'],ylabel='TV: exact law vs replacement fit',title='(a) Observation-model discrepancy');axes[0].legend(fontsize=7)
labels2=[f'{e:g}\n{level}' for e in ENERGIES for level,*_ in RATES]
for j,m in enumerate(('crossfit','qpe_mixture')):
    xx=[next(x for x in c if x['case']==z['case'] and x['method']==m) for z in d];v=np.array([int(x['point_covered'])/200 for x in xx]);lo=np.array([float(x['mc_lower']) for x in xx]);hi=np.array([float(x['mc_upper']) for x in xx])
    axes[1].errorbar(np.arange(9)+(j-.5)*.17,v,yerr=[v-lo,hi-v],fmt='o',ms=3,capsize=2,color=COLORS[m],label='CF' if j==0 else 'QPM')
axes[1].set(xticks=range(9),xticklabels=['L','M','H']*3,ylabel='Programmed-target containment',title='(b) 200 pairs / energy and level',ylim=(0,1.04));axes[1].legend(fontsize=7,loc='lower left')
for center,e in zip([1,4,7],ENERGIES):axes[1].text(center,-.16,f'E={e:g}',ha='center',transform=axes[1].get_xaxis_transform(),fontsize=7)
for bound in [2.5,5.5]:axes[1].axvline(bound,color='.8',lw=.8)
savefig(fig,'15_circuit_phase_summary')
