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

budget=read(OUT/'valuable_followup/budget_summary.json')
NAMES={'crossfit':'CF','qpe_mixture':'QPM','dirichlet':'DIR','profile':'PLR'}
fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained');colors=['#087f8c','#7356a3','#4c783e','#bd5c34']
for i,(m,col) in enumerate(zip(NAMES,colors)):
    rr=[r for r in budget if r['method']==m];x=np.arange(5)+(i-1.5)*.14;y=np.array([r['covered']/100 for r in rr])
    axes[0].plot(x,[r['median_length_K'] for r in rr],'o',ms=4,color=col,label=NAMES[m]);axes[1].errorbar(x,y,yerr=[y-[r['mc_lower'] for r in rr],[r['mc_upper'] for r in rr]-y],fmt='o',ms=3,capsize=2,color=col)
for ax in axes:ax.set(xticks=range(5),xticklabels=['4/.10','6/.10','4/.95','6/.95','6/.358'],xlabel='Counting qubits / generating replacement');ax.spines[['top','right']].set_visible(False)
axes[0].set(ylabel='Median AF gap-set length (K)',yscale='log');axes[0].legend(fontsize=7);axes[1].set(ylabel='Gap inclusion proportion',ylim=(.8,1.015));axes[1].axhline(.95,ls='--',color='.5')
for ext in ('pdf','svg','png'):fig.savefig(TARGET/f'20_equal_query_followup.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
