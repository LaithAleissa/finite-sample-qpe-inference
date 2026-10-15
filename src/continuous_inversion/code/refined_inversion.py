"""Intersect global Lipschitz and local Taylor probability envelopes.

The likelihood rules, training fits and nuisance optimizations are unchanged.
For the ideal QPE kernel, |p''_k| <= t^2 (M^2-1)/6 follows from its finite
Fourier expansion. Floating margins do not constitute directed rounding.
"""
from profile_lr import *
from profile_lr import _merge

def probability_envelope(ex,mid,radius):
    m=2**ex.n;r=np.arange(m)
    waves=np.exp(-1j*np.asarray(mid)[:,None]*ex.time*r)
    amplitude=np.fft.fft(waves,axis=1)/m
    derivative_amplitude=np.fft.fft(waves*(-1j*ex.time*r),axis=1)/m
    derivative=2*np.real(np.conj(amplitude)*derivative_amplitude)
    curvature=ex.time**2*(m*m-1)/6
    radius=np.asarray(radius).reshape(-1,1)
    pad=np.minimum(ex.derivative_bound*radius,
                   (np.abs(derivative)+1e-8)*radius+.5*curvature*radius**2)+1e-12
    p=ex.probabilities(mid)
    return np.maximum(0,p-pad),np.minimum(1,p+pad)

def invert(obj,profile,tolerance=None,max_cells=300000):
    ex=obj.experiment;tol=ex.bin_width/10000 if tolerance is None else tolerance
    if not np.isfinite(tol) or tol<=0:raise ValueError('Invalid tolerance')
    integer(max_cells,'max_cells')
    cutoff=-np.log(obj.alpha)
    uniform_accepted=(-sum(obj.counts)*np.log(2**ex.n)>=obj.threshold) if profile else (obj.log_e(np.mean(ex.bounds),1.)<=cutoff)
    if uniform_accepted:return dict(intervals=[list(ex.bounds)],tolerance=tol,visited_cells=0,uniform_channel_accepted=True)
    cells=np.array([ex.bounds]);retained=[];visited=0
    while len(cells):
        visited+=len(cells)
        if visited>max_cells:raise RuntimeError('Taylor-envelope cell budget exceeded; no partial set')
        mid=cells.mean(axis=1);radius=np.diff(cells,axis=1)/2
        minus,plus=probability_envelope(ex,mid,radius)
        if profile:
            _,upper,eta=profile_bounds(plus,obj.counts)
            reject=upper<obj.threshold-1e-8
            q=(1-eta[:,None])*minus+eta[:,None]/(2**ex.n)
            lower=np.sum(xlogy(obj.counts,q),axis=1)
            accept=(~reject)&((lower>=obj.threshold+1e-8)|(2*radius[:,0]<=tol))
        else:
            lower,_,eta=obj._profile_bounds(plus)
            upper=obj._value(minus,eta)
            reject=lower>cutoff+1e-8
            accept=(~reject)&((upper<=cutoff-1e-8)|(2*radius[:,0]<=tol))
        retained.extend(cells[accept].tolist());uncertain=cells[~reject&~accept];halves=uncertain.mean(axis=1)
        cells=np.concatenate((np.column_stack((uncertain[:,0],halves)),np.column_stack((halves,uncertain[:,1]))))
    return dict(intervals=_merge(retained),tolerance=tol,visited_cells=visited,uniform_channel_accepted=False)

class RefinedJoint(JointInference):
    def confidence_set(self,tolerance=None,max_cells=300000):return invert(self,False,tolerance,max_cells)

class RefinedProfile(ProfileLikelihood):
    def confidence_set(self,tolerance=None,max_cells=300000):return invert(self,True,tolerance,max_cells)
