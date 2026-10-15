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
def figures():
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    folder=TARGET
    fig,ax=plt.subplots(figsize=(3.45,2.85))
    fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
    ax.set(xlim=(0,3.45),ylim=(0,2.85));ax.axis('off')
    boxes=[(.05,2.27,1.5,.49,'Dimer model and\nprepared eigenstates'),
           (1.89,2.27,1.5,.49,'Controlled QPE\ncounts'),
           (1.89,1.45,1.5,.49,'CF / QPM / DIR / CP\nPLR (approximate)\nEnergy sets'),
           (.05,1.45,1.5,.49,'Complete differences\nand gap sets'),
           (.05,.63,1.5,.49,'Classical response\nimages'),
           (1.89,.63,1.5,.49,'External scale, g and\nresponse assumptions')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor='#edf5f5',edgecolor='#087f8c',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.4)
    for a,b in [((1.6,2.515),(1.84,2.515)),((2.64,2.21),(2.64,2.0)),
                ((1.84,1.695),(1.60,1.695)),((.8,1.39),(.8,1.18)),
                ((1.84,.875),(1.60,.875))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=.9,color='#344955'))
    ax.plot([.8,.8],[.56,.39],ls='--',lw=.8,color='#344955')
    ax.text(1.72,.20,'Compare with independent historical fit\nand model sensitivity (separate uncertainty)',ha='center',va='center',fontsize=7.2)
    for ext in ('pdf','svg','png'):fig.savefig(folder/f'16_physical_workflow_column.{ext}',dpi=200)
    plt.close(fig)
    material=json.loads((ROOT/'results/material_refined.json').read_text())
    lo,hi=material['gap_confidence_set_K'][0]
    fig,ax=plt.subplots(figsize=(3.45,1.85),layout='constrained')
    for y,(a,b),c in [(2,(lo,hi),'#306b87'),(1,(412.38,493.27),'#d55a3a'),(0,(397.08,427.51),'#448b49')]:
        ax.plot([a,b],[y,y],color=c,lw=1.5)
        ax.plot([a,a],[y-.07,y+.07],color=c,lw=1)
        ax.plot([b,b],[y-.07,y+.07],color=c,lw=1)
    ax.axvline(412.38,color='.55',ls=':',lw=.8)
    ax.set(yticks=[2,1,0],yticklabels=['95% QPE set\nfixed inputs','Model-choice\nsensitivity','Deletion\nsensitivity'],xlabel='Gap (K)',xlim=(390,500),ylim=(-.4,2.4),xticks=[400,430,460,490])
    for y,c in [(2,'#306b87'),(1,'#d55a3a'),(0,'#448b49')]:
        ax.axhspan(y-.30,y+.30,color=c,alpha=.05,zorder=-1)
    ax.tick_params(labelsize=7.5);ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    for ext in ('pdf','svg','png'):fig.savefig(folder/f'14_uncertainty_scopes_column.{ext}',dpi=200)
    plt.close(fig)
figures()
