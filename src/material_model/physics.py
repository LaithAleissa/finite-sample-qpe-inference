"""Classical susceptibility, explicitly per mole of Cu ions in cgs cm^3/mol."""
import numpy as np
from scipy.constants import h, c, k, N_A, physical_constants
from scipy.optimize import least_squares

VERSION='material_model-v1.0.0'
CM1_TO_K=h*c*100/k
MU_B=physical_constants['Bohr magneton'][0]
CURIE=N_A*(MU_B*1e3)**2/(k*1e7)
G=np.array([2.42,2.08])
TIP=np.array([132e-6,25e-6])
I=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
SPIN=[(np.kron(p,I)+np.kron(I,p))/2 for p in (X,Y,Z)]


def _temperatures(t):
    t=np.asarray(t,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<=0):raise ValueError('Temperatures must be finite and positive')
    return t


def bleaney_bowers(t,gap_k,g):
    t=_temperatures(t)
    if not np.isfinite(gap_k) or gap_k<0 or np.any(~np.isfinite(g)) or np.any(np.asarray(g)<=0):
        raise ValueError('AF gap must be nonnegative and g positive')
    q=np.exp(-gap_k/t)
    return CURIE*np.asarray(g)**2/t*q/(1+3*q)


def susceptibility(t,gap_k,g,fraction=0.0,tip=0.0,impurity_g=None):
    """Cu-ion fraction of free S=1/2 monomers, independent Curie term."""
    t=_temperatures(t)
    if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('fraction must be in [0,1]')
    if not np.all(np.isfinite(tip)):raise ValueError('TIP must be finite')
    gi=g if impurity_g is None else impurity_g
    if np.any(np.asarray(gi)<=0) or np.any(~np.isfinite(gi)):raise ValueError('impurity g must be positive')
    return (1-fraction)*bleaney_bowers(t,gap_k,g)+fraction*CURIE*np.asarray(gi)**2/(4*t)+tip


def spin_hamiltonian(j_cm1,d_cm1=0.0,e_cm1=0.0,offset_cm1=0.0):
    return CM1_TO_K*(j_cm1*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z))
        +d_cm1*(SPIN[2]@SPIN[2])+e_cm1*(SPIN[0]@SPIN[0]-SPIN[1]@SPIN[1])+offset_cm1*np.eye(4))


def kubo_susceptibility(hamiltonian_k,t,g=2.0,axis=2):
    """Zero-field static Kubo response; handles noncommuting H and moment."""
    t=_temperatures(t)
    if axis not in (0,1,2):raise ValueError('axis must be 0,1,2')
    matrix=np.asarray(hamiltonian_k,dtype=complex)
    if matrix.shape!=(4,4) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix,matrix.conj().T):
        raise ValueError('Hamiltonian must be finite Hermitian 4x4')
    if not np.isfinite(g) or g<=0:raise ValueError('g must be positive')
    energy,v=np.linalg.eigh(matrix);energy-=energy.min()
    moment=v.conj().T@(g*SPIN[axis])@v
    result=[]
    for temperature in t.ravel():
        p=np.exp(-energy/temperature);p/=p.sum()
        kernel=np.empty((4,4))
        for a in range(4):
            for b in range(4):
                diff=energy[b]-energy[a]
                if abs(diff)<1e-10:kernel[a,b]=p[a]/temperature
                else:kernel[a,b]=(p[a]-p[b])/diff
        mean=float(np.dot(p,np.diag(moment).real))
        result.append(CURIE/2*(np.sum(kernel*abs(moment)**2)-mean**2/temperature))
    return np.array(result).reshape(t.shape)


def data_arrays(rows):
    t=np.array([float(r['temperature_K']) for r in rows])
    orient=np.array([0 if r['orientation']=='parallel' else 1 for r in rows])
    p2=np.array([float(r['p2']) for r in rows])
    # Preserve the original p_eff^2 = 8 chi T conversion for legacy regression.
    return t,orient,p2/(8*t)


def fit_data(rows,impurity=True,weighting='relative_chi',cut=0,fixed_g=False,
             fixed_gap=None,tip_scale=1.0,multistart=True):
    selected=[r for r in rows if int(r['rank_in_orientation'])>=cut]
    t,o,observed=data_arrays(selected)
    if weighting=='relative_chi':scale=observed
    elif weighting=='equal_chi':scale=np.full_like(t,1e-3)
    elif weighting=='equal_p2':scale=1/(8*t)
    else:raise ValueError('Unknown residual weighting')
    names=[];lo=[];hi=[]
    if not fixed_g:names+=['g_parallel','g_perpendicular'];lo +=[1.5,1.5];hi +=[3.,3.]
    if fixed_gap is None:names+=['gap_K'];lo +=[100.];hi +=[800.]
    if impurity:names+=['monomer_fraction'];lo +=[0.];hi +=[.2]
    if len(t)<=len(names):raise ValueError('Insufficient points for fit')
    def unpack(x):
        params=dict(zip(names,x))
        gs=G if fixed_g else np.array([params['g_parallel'],params['g_perpendicular']])
        gap=fixed_gap if fixed_gap is not None else params['gap_K']
        fraction=params.get('monomer_fraction',0.)
        return gs,gap,fraction
    def residual(x):
        gs,gap,fraction=unpack(x)
        return (susceptibility(t,gap,gs[o],fraction,TIP[o]*tip_scale)-observed)/scale
    fits=[]
    for initial_gap in ([300.,450.,650.] if multistart else [450.]):
        initial={'g_parallel':2.42,'g_perpendicular':2.08,'gap_K':initial_gap,'monomer_fraction':.03}
        fit=least_squares(residual,[initial[name] for name in names],bounds=(lo,hi),
            x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=3000)
        fits.append(fit)
    fit=min(fits,key=lambda f:np.sum(f.fun**2))
    if not fit.success:raise RuntimeError(fit.message)
    gs,gap,fraction=unpack(fit.x)
    predicted=susceptibility(t,gap,gs[o],fraction,TIP[o]*tip_scale)
    norms=np.linalg.norm(fit.jac,axis=0)
    scaled=fit.jac/np.maximum(norms,1e-30)
    singular=np.linalg.svd(scaled,compute_uv=False)
    information=fit.jac.T@fit.jac
    covariance=np.linalg.pinv(information)*np.dot(fit.fun,fit.fun)/(len(t)-len(names))
    sd=np.sqrt(np.maximum(np.diag(covariance),0))
    corr=covariance/np.maximum(np.outer(sd,sd),1e-300)
    return dict(g_parallel=float(gs[0]),g_perpendicular=float(gs[1]),gap_K=float(gap),
        monomer_fraction=float(fraction),cost=float(np.dot(fit.fun,fit.fun)),n_points=len(t),
        rms_relative=float(np.sqrt(np.mean(((predicted-observed)/observed)**2))),
        scaled_condition=float(singular[0]/singular[-1]),parameter_names=names,
        local_correlation=corr.tolist(),at_bound=bool(np.any(np.isclose(fit.x,lo,atol=1e-5)) or np.any(np.isclose(fit.x,hi,atol=1e-5))),
        residuals=fit.fun.tolist(),predicted=predicted.tolist())
