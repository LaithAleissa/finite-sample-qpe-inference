"""Synthetic finite-register QPE inference with an unknown replacement fraction.

All training uses observations only. Truth is never an inference argument.
The exact statistical theorem and floating-point enclosure are distinguished.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import xlogy, gammaln

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent
sys.path.insert(0, str(SRC / 'known_noise'))
sys.path.insert(0, str(SRC / 'likelihood_baseline' / 'code'))
from inference import Experiment, Inference, counts_array, _merge, integer
from baseline import LikelihoodRatio, length


def _ll(counts, p):
    return np.sum(xlogy(counts, p), axis=-1)


def _ratio(numerator, denominator):
    return numerator-denominator if np.isfinite(numerator) else np.full_like(denominator, -np.inf)


def fit_joint(ex, counts):
    """Multistart energy fit with concave noise profile, including boundaries.

    Only numerator efficiency needs this fit; coverage needs no MLE theorem.
    """
    counts = counts_array(counts, ex.n)
    m = 2**ex.n
    def at_energy(e):
        p = ex.probabilities(e)
        def objective(eta):
            return -float(_ll(counts, (1-eta)*p+eta/m))
        opt = minimize_scalar(objective, bounds=(0., 1.), method='bounded',
                              options={'xatol': 1e-10})
        return min((objective(0.), float(e), 0.),
                   (objective(1.), float(e), 1.),
                   (float(opt.fun), float(e), float(opt.x)))
    grid = np.linspace(*ex.bounds, max(32, 16*m)+1)
    # Profile every coarse energy; refine eight best local maxima.
    pg = ex.probabilities(grid)
    lo, hi = np.zeros(len(grid)), np.ones(len(grid))
    # Concave likelihood in eta: batched derivative bisection replaces
    # hundreds of independent scalar optimizer calls on the coarse grid.
    for _ in range(38):
        eta = (lo+hi)/2
        q = (1-eta[:,None])*pg+eta[:,None]/m
        slope = np.sum(counts*(1/m-pg)/q,axis=1)
        lo = np.where(slope > 0,eta,lo)
        hi = np.where(slope <= 0,eta,hi)
    etas = (lo+hi)/2
    obj = -_ll(counts,(1-etas[:,None])*pg+etas[:,None]/m)
    left, right = -_ll(counts,pg), -float(_ll(counts,np.full(m,1/m)))
    etas = np.where(left<obj,0.,etas); obj=np.minimum(left,obj)
    etas = np.where(right<obj,1.,etas); obj=np.minimum(right,obj)
    vals = list(zip(obj,grid,etas))
    peaks = np.flatnonzero((obj[1:-1] <= obj[:-2]) & (obj[1:-1] <= obj[2:]))+1
    candidates = list(vals)
    for i in sorted(peaks, key=lambda i: obj[i])[:8]:
        opt = minimize_scalar(lambda e: at_energy(e)[0], bounds=(grid[i-1], grid[i+1]),
                              method='bounded', options={'xatol': 1e-11})
        candidates.append(at_energy(opt.x))
    _, energy, eta = min(candidates)
    return energy, eta


class JointInference:
    """Joint cross-fit region on branch x [0,1], projected conservatively.

    eta is shared between independent iid batches. No plug-in noise estimate
    is substituted into the candidate denominator.
    """
    def __init__(self, experiment, batch_a, batch_b, alpha=.05):
        if experiment.uniform_noise != 0:
            raise ValueError('Supply the ideal kernel; noise is unknown on [0,1]')
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must be in (0,1)')
        self.experiment = experiment
        self.a = counts_array(batch_a, experiment.n)
        self.b = counts_array(batch_b, experiment.n)
        self.alpha = alpha
        self.fit_a = fit_joint(experiment, self.a)
        self.fit_b = fit_joint(experiment, self.b)
        self.na = float(_ll(self.a, self.probabilities(*self.fit_b)))
        self.nb = float(_ll(self.b, self.probabilities(*self.fit_a)))

    def probabilities(self, energy, eta):
        eta = np.asarray(eta)
        if np.any(~np.isfinite(eta)) or np.any((eta < 0) | (eta > 1)):
            raise ValueError('eta must lie in [0,1]')
        return (1-eta[..., None])*self.experiment.probabilities(energy)+eta[..., None]/(2**self.experiment.n)

    def _value(self, p, eta, derivative=False):
        m = 2**self.experiment.n
        q = (1-eta[..., None])*p+eta[..., None]/m
        ra = _ratio(self.na, _ll(self.a, q))
        rb = _ratio(self.nb, _ll(self.b, q))
        val = np.logaddexp(ra, rb)-np.log(2.)
        if not derivative:
            return val
        # This routine evaluates derivatives strictly inside (0,1), q > 0.
        slope = (1/m-p)/q
        da, db = -np.sum(self.a*slope, axis=-1), -np.sum(self.b*slope, axis=-1)
        both_zero = ~np.isfinite(self.na) and ~np.isfinite(self.nb)
        if both_zero:
            return val, np.zeros_like(val)
        w = np.exp(ra-np.logaddexp(ra, rb))
        return val, w*da+(1-w)*db

    def log_e(self, energy, eta):
        self.probabilities(energy, eta)  # validate noise
        return self._value(self.experiment.probabilities(energy), np.asarray(eta))

    def _profile_bounds(self, p):
        """Bound min_eta log e with convexity, not an unchecked optimizer.

        Negative multinomial log likelihoods are convex in eta, as is their
        log-sum-exp. Derivative bisection brackets a minimizer (boundary
        minimizers included). The supporting tangent gives a lower bound.
        The evaluated value is an upper bound. Also works for probability
        upper envelopes that need not sum to one.
        """
        p = np.atleast_2d(p)
        lo, hi = np.zeros(len(p)), np.ones(len(p))
        for _ in range(38):
            mid = (lo+hi)/2
            _, grad = self._value(p, mid, True)
            hi = np.where(grad >= 0, mid, hi)
            lo = np.where(grad < 0, mid, lo)
        eta = (lo+hi)/2
        value, grad = self._value(p, eta, True)
        lower = value+np.minimum(grad*(lo-eta), grad*(hi-eta))-1e-9
        # Include exact boundaries as witnesses; cannot raise the lower bound.
        left, right = self._value(p, np.zeros(len(p))), self._value(p, np.ones(len(p)))
        eta = np.where(left < value, 0., eta)
        value = np.minimum(left, value)
        eta = np.where(right < value, 1., eta)
        value = np.minimum(right, value)
        lower = np.minimum(lower, value)
        return lower, value, eta

    def projection_membership_bounds(self, energy):
        return self._profile_bounds(self.experiment.probabilities(energy))[:2]

    def confidence_set(self, tolerance=None, max_cells=300000):
        ex = self.experiment
        tol = ex.bin_width/10000 if tolerance is None else tolerance
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError('tolerance must be positive')
        integer(max_cells, 'max_cells')
        cells = np.array([ex.bounds], float)
        retained, visited = [], 0
        cutoff = -np.log(self.alpha)
        # eta=1 is independent of E: if accepted, projection is the full branch.
        if self.log_e(np.mean(ex.bounds), 1.) <= cutoff:
            return dict(intervals=[list(ex.bounds)], tolerance=tol, visited_cells=0,
                        uniform_channel_accepted=True)
        while len(cells):
            visited += len(cells)
            if visited > max_cells:
                raise RuntimeError('Joint projection exceeded cell budget; no partial set returned')
            mid = cells.mean(axis=1)
            radius = np.diff(cells, axis=1)/2
            p = ex.probabilities(mid)
            pad = ex.derivative_bound*radius+1e-12
            lower_e, _, eta = self._profile_bounds(np.minimum(1., p+pad))
            # A single eta witness accepted over a whole cell proves projection
            # acceptance; otherwise retain unresolved terminal cells.
            upper_e = self._value(np.maximum(0., p-pad), eta)
            reject = lower_e > cutoff+1e-8
            accept = (~reject) & ((upper_e <= cutoff-1e-8) | (2*radius[:, 0] <= tol))
            retained.extend(cells[accept].tolist())
            uncertain = cells[~reject & ~accept]
            halves = uncertain.mean(axis=1)
            cells = np.concatenate((np.column_stack((uncertain[:,0], halves)),
                                    np.column_stack((halves, uncertain[:,1]))))
        return dict(intervals=_merge(retained), tolerance=tol, visited_cells=visited,
                    uniform_channel_accepted=False)


class MixtureInference(LikelihoodRatio):
    """Pooled Dirichlet(1/2,...,1/2) predictive likelihood / candidate likelihood.

    This is a normalized sequence distribution, not a Bayesian credible set.
    Inherited inversion works with its likelihood threshold; no MLE is used.
    """
    def __init__(self, experiment, batch_a, batch_b, alpha=.05):
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must be in (0,1)')
        self.experiment = experiment
        self.counts = counts_array(batch_a, experiment.n)+counts_array(batch_b, experiment.n)
        self.alpha = alpha
        m, total = len(self.counts), int(self.counts.sum())
        self.log_predictive = float(gammaln(m/2)-gammaln(total+m/2)
                                    +np.sum(gammaln(self.counts+.5)-gammaln(.5)))
        self.threshold = self.log_predictive+np.log(alpha)

    def log_e(self, energy):
        return self.log_predictive-_ll(self.counts, self.experiment.probabilities(energy))
