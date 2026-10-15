"""Unknown-noise pooled profile LR, compared on the entire declared branch.

The chi-square threshold is asymptotic; no finite-sample guarantee is claimed.
The null nuisance maximum is bracketed by concavity. The global alternative
maximum uses all coarse local maxima, not a truth-centered search window.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import xlogy
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(SRC/'predictive_inference/code'))
from predictive import Experiment,JointInference,counts_array,length
from inference import _merge,integer,binomial_interval


def profile_bounds(p,counts,steps=42):
    """Bounds for max_eta sum c log((1-eta)p+eta/M).

    Evaluated likelihood is a lower bound; a concave supporting tangent on
    the derivative bracket is an upper bound. Envelopes p need not normalize.
    Floating-point margins are safeguards, not interval arithmetic.
    """
    p=np.atleast_2d(p);m=p.shape[-1];lo=np.zeros(len(p));hi=np.ones(len(p))
    for _ in range(steps):
        eta=(lo+hi)/2;q=(1-eta[:,None])*p+eta[:,None]/m
        grad=np.sum(counts*(1/m-p)/q,axis=1)
        lo=np.where(grad>0,eta,lo);hi=np.where(grad<=0,eta,hi)
    eta=(lo+hi)/2;q=(1-eta[:,None])*p+eta[:,None]/m
    value=np.sum(xlogy(counts,q),axis=1)
    grad=np.sum(counts*(1/m-p)/q,axis=1)
    upper=value+np.maximum(grad*(lo-eta),grad*(hi-eta))+1e-9
    left=np.sum(xlogy(counts,p),axis=1);right=-float(np.sum(counts))*np.log(m)
    eta=np.where(left>value,0.,eta);value=np.maximum(value,left)
    eta=np.where(right>value,1.,eta);value=np.maximum(value,right)
    return value,np.maximum(upper,value),eta


class ProfileLikelihood:
    def __init__(self,experiment,batch_a,batch_b,alpha=.05,grid_per_bin=32):
        if experiment.uniform_noise!=0:raise ValueError('Supply ideal kernel; eta is unknown')
        if not np.isfinite(alpha) or not 0<alpha<1:raise ValueError('Invalid alpha')
        integer(grid_per_bin,'grid_per_bin')
        self.experiment=experiment
        self.counts=counts_array(batch_a,experiment.n)+counts_array(batch_b,experiment.n)
        self.alpha=alpha;self.cutoff=float(chi2.ppf(1-alpha,1))
        m=2**experiment.n
        grid=np.linspace(*experiment.bounds,max(32,grid_per_bin*m)+1)
        values,_,etas=profile_bounds(experiment.probabilities(grid),self.counts)
        peaks=np.flatnonzero((values[1:-1]>=values[:-2])&(values[1:-1]>=values[2:]))+1
        # Avoid redundant scalar searches when the entire neighborhood has
        # exactly the uniform likelihood; uniform itself is always a witness.
        peaks=[i for i in peaks if values[i]>-sum(self.counts)*np.log(m)+1e-10]
        idx=int(np.argmax(values));best=(float(values[idx]),float(grid[idx]),float(etas[idx]))
        brackets=[(grid[0],grid[1]),(grid[-2],grid[-1])]+[(grid[i-1],grid[i+1]) for i in peaks]
        for left,right in brackets:
            opt=minimize_scalar(lambda e:-float(self.profile(e)[0][0]),bounds=(left,right),method='bounded',options={'xatol':1e-12})
            v,_,eta=self.profile(opt.x)
            if v[0]>best[0]:best=(float(v[0]),float(opt.x),float(eta[0]))
        self.maximum,self.estimate,self.eta=best
        self.threshold=self.maximum-self.cutoff/2
        self.coarse_peaks_refined=len(brackets)

    def profile(self,energy):return profile_bounds(self.experiment.probabilities(energy),self.counts)

    def membership(self,energy):
        lower,upper,_=self.profile(energy)
        return bool(lower[0]>=self.threshold),bool(upper[0]<self.threshold)

    def confidence_set(self,tolerance=None,max_cells=300000):
        ex=self.experiment;tol=ex.bin_width/10000 if tolerance is None else tolerance
        if not np.isfinite(tol) or tol<=0:raise ValueError('Invalid tolerance')
        integer(max_cells,'max_cells');uniform=-sum(self.counts)*np.log(2**ex.n)
        if uniform>=self.threshold:
            return dict(intervals=[list(ex.bounds)],tolerance=tol,visited_cells=0,uniform_channel_accepted=True)
        cells=np.array([ex.bounds]);retained=[];visited=0
        while len(cells):
            visited+=len(cells)
            if visited>max_cells:raise RuntimeError('Profile LR cell budget exceeded; no partial set')
            mid=cells.mean(axis=1);radius=np.diff(cells,axis=1)/2
            p=ex.probabilities(mid);pad=ex.derivative_bound*radius+1e-12
            _,upper,eta=profile_bounds(np.minimum(1,p+pad),self.counts)
            reject=upper<self.threshold-1e-8
            q=(1-eta[:,None])*np.maximum(0,p-pad)+eta[:,None]/(2**ex.n)
            lower=np.sum(xlogy(self.counts,q),axis=1)
            accept=(~reject)&((lower>=self.threshold+1e-8)|(2*radius[:,0]<=tol))
            retained.extend(cells[accept].tolist());uncertain=cells[~reject&~accept];halves=uncertain.mean(axis=1)
            cells=np.concatenate((np.column_stack((uncertain[:,0],halves)),np.column_stack((halves,uncertain[:,1]))))
        return dict(intervals=_merge(retained),tolerance=tol,visited_cells=visited,uniform_channel_accepted=False)
