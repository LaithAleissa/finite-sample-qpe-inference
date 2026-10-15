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
    for ext in ('pdf','svg','png'):fig.savefig(TARGET/f'{name}.{ext}',bbox_inches='tight',dpi=200)
    plt.close(fig)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

COLORS={'crossfit':'#087f8c','qpe_mixture':'#7356a3','profile':'#bd5c34'}
NAMES={'crossfit':'CF','qpe_mixture':'QPM','profile':'PLR'}
def restored():
    fig,ax=plt.subplots(figsize=(8,3.4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    boxes=[(.1,2.5,2.1,.85,'Dimer model\nPrepared eigenstate'),(2.65,2.5,1.9,.85,'Controlled QPE\ncounts'),
           (5,2.5,2.05,.85,'CF / QPM / DIR / CP\nPLR (approximate)\nEnergy sets'),(7.5,2.5,2.2,.85,'Complete differences\nand gap sets'),
           (7.5,.55,2.2,.85,'Classical response\nimages'),(3.7,.55,2.9,.85,'External scale, g factors\nand response assumptions'),
           (.1,.55,2.8,.85,'Independent historical fit\nand model sensitivity')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor='#edf5f5',edgecolor='#087f8c'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    for a,b in [((2.25,2.93),(2.58,2.93)),((4.6,2.93),(4.93,2.93)),((7.1,2.93),(7.43,2.93)),
                ((8.6,2.4),(8.6,1.5)),((6.7,.98),(7.4,.98))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#344955',lw=1.5))
    ax.text(5,3.68,'Quantum experiment and model-conditional inference',ha='center',weight='bold',fontsize=10)
    ax.text(8.65,1.95,'fixed response map',ha='center',fontsize=7)
    ax.annotate('',xy=(7.4,1.6),xytext=(2.95,1.6),arrowprops=dict(arrowstyle='<->',ls='--',color='#344955'))
    ax.text(5.1,1.72,'Compare uncertainty scopes',ha='center',fontsize=7)
    ax.text(5,.05,'Physical interpretation remains conditional on the supplied calibration and response model.',ha='center',fontsize=8)
    savefig(fig,'16_physical_workflow')
    fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    rows=list(csv.DictReader((ROOT.parent/'quantum_model/results/resources.csv').open()))
    for method,style,color in [('scaled','o-','#087f8c'),('repeated','s--','#bd5c34')]:
        rr=[r for r in rows if r['model']=='isotropic' and r['method']==method]
        axs[0].plot([int(r['n']) for r in rr],[int(r['cx']) for r in rr],style,color=color,label=method)
    axs[0].set(xlabel='Counting qubits n',ylabel='Logical compiled CX count',yscale='log',title='(a) Exact controlled-power synthesis');axs[0].legend(fontsize=8)
    rows=list(csv.DictReader((ROOT/'results/spectral_refined/ordering_summary.csv').open()))
    for i,(n,eta) in enumerate([(4,0),(4,.1),(6,0),(6,.1)]):
        rr=[r for r in rows if int(r['n'])==n and float(r['noise'])==eta]
        y=np.array([int(r['separated'])/100 for r in rr]);lo=np.array([float(r['mc_lower']) for r in rr]);hi=np.array([float(r['mc_upper']) for r in rr])
        axs[1].errorbar(np.arange(5)+(i-1.5)*.13,y,yerr=[y-lo,hi-y],fmt='os^D'[i],capsize=2,ms=3,label=f'n={n}, eta={eta:g}')
    axs[1].set(xticks=np.arange(5),xticklabels=['0','.005','.015','.04','.10'],xlabel='Tested Zeeman coefficient h',ylabel='Triplet separation frequency',ylim=(-.03,1.05),title='(b) Known-eta CF ordering');axs[1].legend(fontsize=6,loc='center right')
    savefig(fig,'17_resources_ordering')

def report():
    out=ROOT/'results/physical_bridge';assert read(out/'status.json')['status']=='complete'
    rows=read(out/'summary.json');examples=read(out/'examples.json');status=read(out/'status.json')
    fig,axs=plt.subplots(2,2,figsize=(8,5.6),layout='constrained');labels=['4 / .10','6 / .10','4 / .95','6 / .95']
    for i,m in enumerate(NAMES):
        rr=[r for r in rows if r['method']==m];x=np.arange(4)+(i-1)*.18;y=np.array([r['gap_covered']/200 for r in rr])
        axs[0,0].errorbar(x,y,yerr=[y-np.array([r['mc_lower'] for r in rr]),np.array([r['mc_upper'] for r in rr])-y],fmt='o',ms=3,capsize=2,color=COLORS[m],label=NAMES[m])
        axs[0,1].plot(x,[r['median_length_K'] for r in rr],'o',color=COLORS[m],ms=4,label=NAMES[m])
    axs[0,0].axhline(.95,ls='--',color='.5');axs[0,0].set(ylabel='Gap inclusion proportion',ylim=(.8,1.02),title='(a) Unknown-noise gap sets');axs[0,0].legend(fontsize=7)
    axs[0,1].set(ylabel='Median AF gap-set length (K)',yscale='log',title='(b) Localization under fixed scale')
    for a in axs[0]:a.set(xticks=np.arange(4),xticklabels=labels,xlabel='n / replacement fraction (sampling only)')
    for ax,ex in zip(axs[1],[examples[1],examples[2]]):
        t=np.array(ex['temperatures']);ref=np.array(ex['reference'])
        for m in NAMES:
            image=ex['methods'][m]
            if image['lower']:
                lo=100*(np.array(image['lower'])/ref-1);hi=100*(np.array(image['upper'])/ref-1)
                ax.fill_between(t,lo,hi,color=COLORS[m],alpha=.12)
                ax.plot(t,lo,color=COLORS[m],lw=.8);ax.plot(t,hi,color=COLORS[m],lw=.8,label=NAMES[m])
        ax.axhline(0,color='.4',ls=':');ax.set_yscale('symlog',linthresh=1);ax.set(xlabel='Temperature (K)',ylabel='Response change from reference (%)',title=ex['case'].replace('_',', ')+'; trial 0');ax.legend(fontsize=6)
    savefig(fig,'18_unknown_gap_response')
    fig,axs=plt.subplots(2,2,figsize=(8,5.5),layout='constrained')
    for ax,ex in zip(axs.ravel(),examples):
        t=np.array(ex['temperatures']);ax.plot(t,1e3*np.array(ex['reference']),'k--',lw=1,label='Fixed reference')
        for m in NAMES:
            image=ex['methods'][m]
            if image['lower']:
                ax.fill_between(t,1e3*np.array(image['lower']),1e3*np.array(image['upper']),color=COLORS[m],alpha=.12)
                ax.plot(t,1e3*np.array(image['lower']),color=COLORS[m],lw=.8);ax.plot(t,1e3*np.array(image['upper']),color=COLORS[m],lw=.8,label=NAMES[m])
        ax.set(title=ex['case'].replace('_',', ')+'; trial 0',xlabel='Temperature (K)',ylabel='Spin susceptibility (10^-3 cm^3/mol Cu)');ax.legend(fontsize=6)
    savefig(fig,'19_all_response_examples')

restored()
report()
