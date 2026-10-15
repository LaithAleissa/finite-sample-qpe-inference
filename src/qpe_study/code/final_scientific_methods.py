"""Independent CP construction and batch-fraction extension; no truth inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
from scipy.stats import beta
from scipy.special import logsumexp
from methods import (Experiment, RefinedJoint, QPEMixture, counts_array,
                     profile_bounds, probability_envelope, _merge)


def affine_feasible(offset, slope, target, greater):
    """Feasibility of affine inequalities in ONE common eta in [0,1]."""
    offset=np.atleast_2d(offset);slope=np.broadcast_to(slope,offset.shape)
    rhs=np.asarray(target)-offset
    if not greater:slope=-slope;rhs=-rhs
    positive=slope>0;negative=slope<0
    ratio=np.divide(rhs,slope,out=np.zeros_like(offset),where=slope!=0)
    lower=np.maximum(0,np.max(np.where(positive,ratio,-np.inf),axis=1))
    upper=np.minimum(1,np.min(np.where(negative,ratio,np.inf),axis=1))
    possible=(lower<=upper)&np.all((slope!=0)|(rhs<=0),axis=1)
    return lower,upper,possible


class SimultaneousCP:
    def __init__(self,experiment,a,b,alpha=.05):
        if experiment.uniform_noise!=0 or not 0<alpha<1:raise ValueError('Invalid experiment/alpha')
        self.experiment=experiment;self.alpha=alpha
        self.counts=counts_array(a,experiment.n)+counts_array(b,experiment.n)
        n=int(self.counts.sum());m=len(self.counts);tail=alpha/(2*m)
        self.lower=np.where(self.counts==0,0,beta.ppf(tail,self.counts,n-self.counts+1))
        self.upper=np.where(self.counts==n,1,beta.ppf(1-tail,self.counts+1,n-self.counts))

    def feasible(self,minus,plus):
        # q_plus >= CP lower and q_minus <= CP upper are necessary.
        a,b,c=affine_feasible(plus,1/len(self.counts)-plus,self.lower,True)
        d,e,f=affine_feasible(minus,1/len(self.counts)-minus,self.upper,False)
        return c&f&(np.maximum(a,d)<=np.minimum(b,e))

    def membership(self,energy):
        p=self.experiment.probabilities(energy)
        result=bool(self.feasible(p,p)[0]);return result,not result

    def confidence_set(self,tolerance=None,max_cells=300000):
        ex=self.experiment;tol=ex.bin_width/10000 if tolerance is None else tolerance
        if not np.isfinite(tol) or tol<=0:raise ValueError('Invalid tolerance')
        u=1/len(self.counts)
        if np.all(self.lower<=u)&np.all(u<=self.upper):
            return dict(intervals=[list(ex.bounds)],tolerance=tol,visited_cells=0)
        cells=np.array([ex.bounds]);retained=[];visited=0
        while len(cells):
            visited+=len(cells)
            if visited>max_cells:raise RuntimeError('CP inversion budget exceeded; no partial set')
            mid=cells.mean(axis=1);radius=np.diff(cells,axis=1)[:,0]/2
            minus,plus=probability_envelope(ex,mid,radius)
            possible=self.feasible(minus,plus)
            # A common eta making the ENTIRE probability envelope lie inside
            # the bands certifies acceptance (floating safeguards only).
            accepted=self.feasible(plus,minus)
            keep=possible&(accepted|(2*radius<=tol))
            retained.extend(cells[keep].tolist());uncertain=cells[possible&~keep]
            halves=uncertain.mean(axis=1)
            cells=np.concatenate((np.column_stack((uncertain[:,0],halves)),
                                  np.column_stack((halves,uncertain[:,1]))))
        return dict(intervals=_merge(retained),tolerance=tol,visited_cells=visited)


class DriftCF:
    def __init__(self,experiment,a,b,alpha=.05,shared=None):
        base=shared if shared is not None else RefinedJoint(experiment,a,b,alpha)
        if base.experiment is not experiment or base.alpha!=alpha:raise ValueError('Mismatched predictor')
        self.__dict__.update(base.__dict__)

    def projection_membership_bounds(self,energy):
        return self.bounds(self.experiment.probabilities(energy))

    def bounds(self,p):
        la,ua,_=profile_bounds(p,self.a);lb,ub,_=profile_bounds(p,self.b)
        return (np.logaddexp(self.na-ua,self.nb-ub)-np.log(2),
                np.logaddexp(self.na-la,self.nb-lb)-np.log(2))

    def confidence_set(self,tolerance=None,max_cells=300000):
        ex=self.experiment;tol=ex.bin_width/10000 if tolerance is None else tolerance
        if not np.isfinite(tol) or tol<=0:raise ValueError('Invalid tolerance')
        cutoff=-np.log(self.alpha);m=2**ex.n
        uniform=np.logaddexp(self.na+self.a.sum()*np.log(m),self.nb+self.b.sum()*np.log(m))-np.log(2)
        if uniform<=cutoff:return dict(intervals=[list(ex.bounds)],tolerance=tol,visited_cells=0)
        cells=np.array([ex.bounds]);retained=[];visited=0
        while len(cells):
            visited+=len(cells)
            if visited>max_cells:raise RuntimeError('Drift inversion budget exceeded; no partial set')
            mid=cells.mean(axis=1);radius=np.diff(cells,axis=1)[:,0]/2
            minus,plus=probability_envelope(ex,mid,radius)
            lower,_=self.bounds(plus);_,upper=self.bounds(minus)
            reject=lower>cutoff+1e-8
            keep=(~reject)&((upper<=cutoff-1e-8)|(2*radius<=tol))
            retained.extend(cells[keep].tolist());uncertain=cells[~reject&~keep];halves=uncertain.mean(axis=1)
            cells=np.concatenate((np.column_stack((uncertain[:,0],halves)),
                                  np.column_stack((halves,uncertain[:,1]))))
        return dict(intervals=_merge(retained),tolerance=tol,visited_cells=visited)


def streamed_mixture(ex,counts,s):
    """Exactly the specified finite grid, chunked; no likelihood pruning."""
    m=2**ex.n;energies=np.linspace(*ex.bounds,32*m*s,endpoint=False)
    etas=np.linspace(0,1,1+16*s);total=-np.inf;used=counts>0
    for start in range(0,len(energies),128):
        p=ex.probabilities(energies[start:start+128])
        q=((1-etas[None,:,None])*p[:,None,:]+etas[None,:,None]/m).reshape(-1,m)
        with np.errstate(divide='ignore'):vals=np.log(q[:,used])@counts[used]
        total=np.logaddexp(total,logsumexp(vals))
    return float(total-np.log(len(energies)*len(etas)))


class GrowingQPM(QPEMixture):
    def __init__(self,experiment,a,b,alpha=.05):
        if experiment.uniform_noise!=0 or not 0<alpha<1:raise ValueError('Invalid experiment/alpha')
        self.experiment=experiment;self.alpha=alpha
        self.counts=counts_array(a,experiment.n)+counts_array(b,experiment.n)
        self.s=max(1,int(np.ceil(np.sqrt(self.counts.sum()/256))))
        self.log_predictive=streamed_mixture(experiment,self.counts,self.s)
        self.threshold=self.log_predictive+np.log(alpha)
