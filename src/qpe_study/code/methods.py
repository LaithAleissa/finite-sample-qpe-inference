"""shared_inference methods. Fixed mixture is a sequence law, not an adaptive prior."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(SRC/'continuous_inversion/code'))
from refined_inversion import RefinedJoint,RefinedProfile,probability_envelope,invert
from profile_lr import Experiment,profile_bounds,counts_array,ProfileLikelihood,length
from inference import Inference,_merge,binomial_interval
from baseline import LikelihoodRatio
from predictive import MixtureInference
import numpy as np
from scipy.special import xlogy,logsumexp
from functools import lru_cache

@lru_cache(maxsize=8)
def reference(n,time,bounds):
    ex=Experiment(n,time,bounds)
    energies=np.linspace(*bounds,32*2**n,endpoint=False)
    eta=np.linspace(0,1,17)
    p=ex.probabilities(energies)
    q=((1-eta[None,:,None])*p[:,None,:]+eta[None,:,None]/2**n).reshape(-1,2**n)
    with np.errstate(divide='ignore'): logs=np.log(q)
    return energies,eta,logs

def mixture_numerator(ex,counts):
    _,_,logs=reference(ex.n,ex.time,tuple(ex.bounds))
    used=counts>0
    vals=logs[:,used]@counts[used]
    return float(logsumexp(vals)-np.log(len(vals)))

class QPEMixture:
    def __init__(self,experiment,a,b,alpha=.05):
        if experiment.uniform_noise!=0:raise ValueError('Unknown-noise mixture needs ideal kernel')
        if not np.isfinite(alpha) or not 0<alpha<1:raise ValueError('Invalid alpha')
        self.experiment=experiment;self.alpha=alpha
        self.counts=counts_array(a,experiment.n)+counts_array(b,experiment.n)
        self.log_predictive=mixture_numerator(experiment,self.counts)
        self.threshold=self.log_predictive+np.log(alpha)
    def profile(self,e):return profile_bounds(self.experiment.probabilities(e),self.counts)
    def membership(self,e):
        lo,hi,_=self.profile(e)
        return bool(lo[0]>=self.threshold),bool(hi[0]<self.threshold)
    def confidence_set(self,tolerance=None,max_cells=300000):
        return invert(self,True,tolerance,max_cells)

def known_invert(obj,tolerance=None,max_cells=300000):
    ex=obj.experiment;tol=ex.bin_width/10000 if tolerance is None else tolerance
    if not np.isfinite(tol) or tol<=0:raise ValueError('Invalid tolerance')
    if not isinstance(max_cells,int) or max_cells<1:raise ValueError('Invalid cell budget')
    cells=np.array([ex.bounds]);retained=[];visited=0
    # Apply the identical IDEAL envelope then the known affine channel.
    ideal=Experiment(ex.n,ex.time,ex.bounds)
    while len(cells):
        visited+=len(cells)
        if visited>max_cells:raise RuntimeError('Refined known-channel cell budget exceeded')
        mid=cells.mean(axis=1);radius=np.diff(cells,axis=1)[:,0]/2
        minus,plus=probability_envelope(ideal,mid,radius)
        minus=(1-ex.uniform_noise)*minus+ex.uniform_noise/2**ex.n
        plus=(1-ex.uniform_noise)*plus+ex.uniform_noise/2**ex.n
        if isinstance(obj,Inference):
            def ratio(p):
                la=np.sum(xlogy(obj.a,p),axis=1);lb=np.sum(xlogy(obj.b,p),axis=1)
                ra=obj.numerator_a-la if np.isfinite(obj.numerator_a) else np.full_like(la,-np.inf)
                rb=obj.numerator_b-lb if np.isfinite(obj.numerator_b) else np.full_like(lb,-np.inf)
                return np.logaddexp(ra,rb)-np.log(2)
            lower,upper=ratio(plus),ratio(minus);cutoff=-np.log(obj.alpha)
            reject=lower>cutoff+1e-8;accept=(~reject)&((upper<=cutoff-1e-8)|(2*radius<=tol))
        else:
            upper=np.sum(xlogy(obj.counts,plus),axis=1);lower=np.sum(xlogy(obj.counts,minus),axis=1)
            reject=upper<obj.threshold-1e-8;accept=(~reject)&((lower>=obj.threshold+1e-8)|(2*radius<=tol))
        retained.extend(cells[accept].tolist());u=cells[~reject&~accept];halves=u.mean(axis=1)
        cells=np.concatenate((np.column_stack((u[:,0],halves)),np.column_stack((halves,u[:,1]))))
    return dict(intervals=_merge(retained),tolerance=tol,visited_cells=visited)
