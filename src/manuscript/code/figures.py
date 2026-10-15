"""All manuscript figures, from verified tables and the common physics core."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

DATA=ROOT/'data'; OUT=ROOT/'figs'
COLORS=['#17628a','#b34328','#3b805a','#8063a2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.prop_cycle':plt.cycler(color=COLORS)})
INDEX=[]

def save(fig,name,source):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None})
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=240)
    plt.close(fig);INDEX.append(dict(figure=name,source=source))

def nums(rows,key):return np.array([float(r[key]) for r in rows])

def schematic():
    fig,ax=plt.subplots(figsize=(7.0,2.9));ax.set(xlim=(0,12),ylim=(-.1,4.7));ax.axis('off')
    ys=[3.8,2.9,2.0]
    def box(x,y,w,h,label):
        ax.add_patch(Rectangle((x,y),w,h,fc='#eef4f7',ec=COLORS[0],lw=1,zorder=3))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9,zorder=4)
    for k,y in enumerate(ys):
        ax.plot([1.3,11.1],[y,y],color='0.35',lw=.9)
        ax.text(.05,y,rf'$q_{k}: |0\rangle$',va='center')
        box(1.5,y-.23,.6,.46,'H')
    ax.plot([1.3,11.1],[.75,.75],color='0.35',lw=2)
    ax.text(.05,.75,r'$|\psi_j\rangle$',va='center')
    for k,(x,y) in enumerate(zip([3,4.6,6.2],ys)):
        ax.plot([x,x],[1.1,y],color='0.2');ax.plot(x,y,'o',color='0.2',ms=4)
        box(x-.57,.4,1.14,.7,rf'$U({2**k}t)$')
    box(7.5,1.7,1.6,2.4,r'$\mathrm{QFT}^{-1}$')
    for y in ys:box(9.7,y-.23,.7,.46,'M')
    ax.text(10.7,2.9,'integer\nbitstring\ncounts',va='center',ha='left',fontsize=8,bbox=dict(fc='white',ec='none',pad=2),zorder=4)
    ax.text(4.6,0.05,r'$U(2^kt)=\exp[-iH(2^kt)]$: one angle-scaled block',ha='center',fontsize=8)
    ax.text(0,4.4,'QPE on two target spins (three counting qubits illustrated)',weight='bold')
    save(fig,'01_qpe_circuit','Schematic of quantum_model/dimer.py; full inverse QFT includes swaps')
    fig,ax=plt.subplots(figsize=(7,1.6));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,2))
    labels=['Supplied spin\nHamiltonian','QPE counts\n(two batches)','Energy and\ngap sets','Classical\n'+r'$\chi(T;\Delta_K)$']
    for i,label in enumerate(labels):
        x=i*2.5;ax.add_patch(Rectangle((x,.65),2,.8,fc='#eef4f7',ec=COLORS[0]))
        ax.text(x+1,1.05,label,ha='center',va='center')
        if i<3:ax.add_patch(FancyArrowPatch((x+2,1.05),(x+2.45,1.05),arrowstyle='->',mutation_scale=12))
    ax.text(8.5,.25,'Fixed literature scale, g and model',ha='center',fontsize=8)
    ax.text(0,1.85,'Computational chain and external calibration',weight='bold')
    save(fig,'02_workflow','Logical chain implemented by code/analyze.py')

def resources():
    rows=read_csv(DATA/'resources.csv');fig,axes=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    for ax,model in zip(axes,['isotropic','xxz']):
        for method,marker in [('scaled','o'),('repeated','s')]:
            sub=[r for r in rows if r['model']==model and r['method']==method]
            ax.plot(nums(sub,'n'),nums(sub,'cx'),marker+'-',label=method.capitalize())
        ax.set(yscale='log',xlabel='Counting qubits n',ylabel='Compiled CX count',title='Isotropic' if model=='isotropic' else r'XXZ: $J_z/J_{xy}=2.1$')
        ax.set_xticks([2,3,4,5,6,8]);ax.legend();ax.grid(alpha=.2)
    save(fig,'03_resources','data/resources.csv; independently recompiled')

def spectra():
    fig,axes=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
    rows=read_csv(DATA/'quantum_model/xxz_sweep.csv')
    for state,label in [('singlet','S'),('triplet_plus',r'$T_0$'),('triplet_00',r'$T_\pm$')]:
        sub=[r for r in rows if r['state']==state]
        line,=axes[0].plot(nums(sub,'ratio'),nums(sub,'exact_energy'),label=label)
        axes[0].scatter(nums(sub,'ratio'),nums(sub,'modal_energy'),s=23,facecolors='none',edgecolors=line.get_color())
    axes[0].set(xlabel=r'$J_z/J_{xy}$',ylabel=r'Energy / $J_{xy}$',title='(a) XXZ spectrum, zero field')
    rows=read_csv(DATA/'quantum_model/zeeman_sweep.csv')
    for state,label in [('triplet_11',r'$T_-$'),('triplet_plus',r'$T_0$'),('triplet_00',r'$T_+$')]:
        sub=[r for r in rows if r['state']==state and r['axis']=='z']
        line,=axes[1].plot(nums(sub,'field_parameter')*2.42,nums(sub,'exact_energy'),label=label)
        axes[1].scatter(nums(sub,'field_parameter')*2.42,nums(sub,'modal_energy'),s=23,facecolors='none',edgecolors=line.get_color())
    axes[1].set(xlabel=r'Zeeman coefficient $h/J$',ylabel=r'Energy / $J$',title='(b) Isotropic triplet, z field')
    for ax in axes:ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    save(fig,'04_spectra','quantum_model/xxz_sweep.csv and zeeman_sweep.csv; ideal modal diagnostics, n=6')

def confidence():
    exs=read_json(DATA/'known_noise/example_inferences.json')
    fig,axes=plt.subplots(1,3,figsize=(7,2.9),layout='constrained')
    names=['two_bit_boundary_mode','on_grid','near_negative_branch_end']
    for ax,name,title in zip(axes,names,['(a) Coarse off-grid','(b) Exact grid point','(c) Branch ambiguity']):
        r=next(r for r in exs if r['case']==name)
        for lo,hi in r['intervals']:
            ax.plot([lo,hi],[1,1],lw=4,color=COLORS[2],solid_capstyle='butt')
            ax.plot([lo,hi],[1,1],ls='none',marker='|',ms=8,color=COLORS[2])
        ax.scatter([r['truth']],[1.16],marker='|',s=150,color='black',label='Generating energy')
        ax.scatter([r['estimate']],[.84],marker='o',s=22,label='Likelihood estimate')
        ax.scatter([r['modal']],[.66],marker='x',s=30,color=COLORS[1],label='Modal energy')
        ax.set(xlim=(-3.85,3.85),ylim=(.5,1.38),yticks=[],xlabel=r'Energy / $J$',title=title)
        ax.text(.5,.97,'95% confidence set (green)',transform=ax.transAxes,ha='center',va='top',fontsize=7)
        ax.grid(axis='x',alpha=.2)
    axes[0].legend(loc='lower center',fontsize=6.8)
    save(fig,'05_confidence_sets','known_noise/example_inferences.json; first trial in each named design')
    rows=read_csv(DATA/'known_noise/coverage.csv');x=np.arange(1,len(rows)+1)
    fig,ax=plt.subplots(figsize=(7,2.8),layout='constrained')
    y=nums(rows,'coverage');ax.errorbar(x,y,yerr=[y-nums(rows,'simultaneous_mc_lo'),nums(rows,'simultaneous_mc_hi')-y],fmt='o',capsize=3)
    ax.axhline(.95,color='black',ls='--',lw=1,label='Nominal coverage lower bound')
    ax.set(xlabel='Fixed design case (Appendix table)',ylabel='Coverage',xticks=x,ylim=(.94,1.005))
    ax.legend(loc='lower right');ax.grid(alpha=.2)
    save(fig,'06_coverage','known_noise/coverage.csv; 200 repetitions/case; simultaneous MC intervals over 12 cases')

def resolution():
    rows=read_csv(DATA/'known_noise/resolution_power.csv')
    fig,axes=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
    fields=[0,.005,.015,.04,.1]
    for ax,noise in zip(axes,[0,.1]):
        for n,shift,marker in [(4,-.09,'o'),(6,.09,'s')]:
            sub=[r for r in rows if int(r['n'])==n and float(r['noise'])==noise];y=nums(sub,'probability')
            ax.errorbar(np.arange(5)+shift,y,yerr=[y-nums(sub,'simultaneous_mc_lo'),nums(sub,'simultaneous_mc_hi')-y],fmt=marker,capsize=3,label=f'n = {n}')
        ax.axhline(.8,color='gray',ls='--',lw=1)
        ax.set(xticks=np.arange(5),xticklabels=[str(v) for v in fields],xlabel=r'Tested $h/J$ (categorical spacing)',ylabel='Resolution probability',ylim=(-.03,1.03),title=rf'Known replacement fraction $\eta={noise}$')
        ax.legend(loc='lower right');ax.grid(alpha=.2)
    save(fig,'07_resolution','known_noise/resolution_power.csv; all 20 tested cells, no interpolation')

def propagation():
    rows=read_csv(DATA/'material_model/qpe_classical_propagation.csv');t=nums(rows,'temperature_K');base=nums(rows,'anchor_spin')
    estimate=nums(rows,'qpe_gap_spin');lo=nums(rows,'conditional_lower');hi=nums(rows,'conditional_upper')
    fig,axes=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
    axes[0].fill_between(t,lo*1e4,hi*1e4,color=COLORS[2],alpha=.35,label='Conditional 95% envelope')
    axes[0].plot(t,base*1e4,color='black',lw=1,label='Fixed literature gap')
    axes[0].plot(t,estimate*1e4,'--',label='Trial 0 gap estimate')
    axes[0].set(xlabel='Temperature (K)',ylabel=r'$\chi$ ($10^{-4}$ cm$^3$ mol$^{-1}$ Cu)',title='(a) Classical spin susceptibility')
    axes[1].fill_between(t,100*(lo/base-1),100*(hi/base-1),color=COLORS[2],alpha=.35)
    axes[1].plot(t,100*(estimate/base-1),'--');axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xlabel='Temperature (K)',ylabel='Change from fixed-gap curve (%)',title='(b) Conditional gap effect')
    axes[0].legend(fontsize=7);[ax.grid(alpha=.2) for ax in axes]
    save(fig,'08_propagation','material_model/qpe_classical_propagation.csv; recomputed from replayed known-noise trial 0')

def historical():
    rows=read_csv(DATA/'fit_residuals.csv');p=read_json(DATA/'material_model/primary_fit.json');t=np.linspace(75,320,400)
    fig,axes=plt.subplots(2,2,figsize=(7,4.3),sharex='col',gridspec_kw={'height_ratios':[2.4,1]},layout='constrained')
    for i,o in enumerate(['parallel','perpendicular']):
        sub=[r for r in rows if r['orientation']==o];g=p['g_parallel'] if i==0 else p['g_perpendicular']
        axes[0,i].scatter(nums(sub,'temperature_K'),nums(sub,'chi_observed')*1e4,s=24,color='black',label='Inherited data')
        axes[0,i].plot(t,susceptibility(t,p['gap_K'],g,p['monomer_fraction'],TIP[i])*1e4,label='Dimer + monomer fit')
        axes[0,i].plot(t,susceptibility(t,315*CM1_TO_K,G[i],0,TIP[i])*1e4,'--',label='Historical-parameter model')
        axes[0,i].set(title=o.capitalize(),ylabel=r'$\chi$ ($10^{-4}$ cm$^3$ mol$^{-1}$ Cu)')
        axes[1,i].axhline(0,color='gray',lw=.8);axes[1,i].scatter(nums(sub,'temperature_K'),100*nums(sub,'relative_residual'),s=24)
        axes[1,i].set(xlabel='Temperature (K)',ylabel='Residual (%)')
        axes[0,i].legend(fontsize=6.9)
    [ax.grid(alpha=.2) for ax in axes.flat]
    save(fig,'09_historical_fit','data/fit_residuals.csv; recomputed primary fit; no experimental error bars supplied')

def sensitivity():
    rows=read_csv(DATA/'material_model/fit_sensitivity.csv');fig,axes=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
    for ax,model in zip(axes,['dimer_only','dimer_plus_monomer']):
        for w,label,mark in [('relative_chi',r'Relative $\chi$','o'),('equal_chi',r'Equal $\chi$','s'),('equal_p2',r'Equal $p^2$','^')]:
            sub=[r for r in rows if r['model']==model and r['weighting']==w]
            ax.plot(nums(sub,'cut'),nums(sub,'gap_K'),mark+'-',label=label)
        ax.set(xlabel='Points removed per orientation',ylabel='Fitted gap (K)',xticks=range(4),ylim=(310,510),title='Dimer only' if model=='dimer_only' else 'Dimer + monomer')
        ax.legend();ax.grid(alpha=.2)
    save(fig,'10_fit_sensitivity','material_model/fit_sensitivity.csv; all 24 fits independently recomputed')
    profile=read_csv(DATA/'material_model/gap_profile.csv');p=read_json(DATA/'material_model/primary_fit.json')
    fig,axes=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
    axes[0].plot(nums(profile,'gap_K'),nums(profile,'cost')-p['cost'])
    axes[0].set(xlabel='Fixed gap (K)',ylabel='Increase in residual sum of squares',title='(a) Profile objective (diagnostic)');axes[0].grid(alpha=.2)
    im=axes[1].imshow(p['local_correlation'],vmin=-1,vmax=1,cmap='coolwarm')
    labels=[r'$g_\parallel$',r'$g_\perp$',r'$\Delta_K$','f']
    axes[1].set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,title='(b) Local parameter correlation')
    for i in range(4):
        for j in range(4):axes[1].text(j,i,f"{p['local_correlation'][i][j]:.2f}",ha='center',va='center',fontsize=8,color='white' if p['local_correlation'][i][j]>.7 else 'black')
    fig.colorbar(im,ax=axes[1],shrink=.8)
    save(fig,'11_fit_diagnostics','material_model/gap_profile.csv and primary_fit.json; diagnostic, not inferential confidence regions')

def components():
    rows=read_csv(DATA/'material_model/susceptibility_ablation.csv');t=nums(rows,'temperature_K')
    fig,axes=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
    for key,label in [('TIP_component','TIP'),('rounding_component','Rounded coefficients'),('D_E_positive_change',r'$D=+0.34$, $E=0.01$')]:
        axes[0].plot(t,nums(rows,key)*1e6,label=label)
    for key,label in [('D_positive_change',r'$D=+0.34$, $E=0$'),('D_negative_change',r'$D=-0.34$, $E=0$'),('D_E_positive_change',r'$D=+0.34$, $E=0.01$')]:
        axes[1].plot(t,nums(rows,key)*1e6,label=label,ls='--' if key=='D_E_positive_change' else '-')
    for ax,title in zip(axes,['(a) Common-baseline components','(b) Anisotropy on an expanded scale']):
        ax.set(xlabel='Temperature (K)',ylabel=r'$\delta\chi$ ($10^{-6}$ cm$^3$ mol$^{-1}$ Cu)',title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,'12_components','material_model/susceptibility_ablation.csv; D/E in cm^-1; explicit total-spin extension')

def main():
    assert read_json(ROOT/'qa/status.json')['status'] in ('analysis_complete','complete')
    OUT.mkdir(exist_ok=True)
    for fn in (schematic,resources,spectra,confidence,resolution,propagation,historical,sensitivity,components):fn()
    write_json(OUT/'index.json',INDEX)
    print(f'{len(INDEX)} figures generated in PDF, SVG and PNG.')

if __name__=='__main__':main()
