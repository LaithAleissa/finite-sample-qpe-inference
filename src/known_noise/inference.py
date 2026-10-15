"""Finite-QPE likelihood and finite-sample cross-fit confidence sets.

No chi-square approximation, posterior grid, modal bootstrap, or truth-based
energy window is used. Two batches must be independent and identically sampled.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, xlogy
from scipy.stats import beta

VERSION = 'known_noise-v1.0.0'


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def counts_array(counts, n):
    """Accept integer vectors or strict, MSB-first n-bit Qiskit count keys."""
    size = 2**integer(n, 'n')
    if isinstance(counts, dict):
        result = np.zeros(size, dtype=np.int64)
        for key, value in counts.items():
            if not isinstance(key, str) or len(key) != n or any(c not in '01' for c in key):
                raise ValueError('Counts require exactly n binary digits per key; no extra registers')
            result[int(key, 2)] = integer(value, 'count', 0)
    else:
        raw = np.asarray(counts)
        if raw.shape != (size,) or raw.dtype.kind not in 'iu' or np.any(raw < 0):
            raise ValueError('Counts must be a nonnegative integer vector of length 2**n')
        if np.any(raw > np.iinfo(np.int64).max):
            raise ValueError('Counts exceed int64')
        result = raw.astype(np.int64)
    if sum(int(x) for x in result) < 1 or sum(int(x) for x in result) > 10**9:
        raise ValueError('Batch size must be between 1 and 1e9')
    return result


@dataclass(frozen=True)
class Experiment:
    n: int
    time: float
    bounds: tuple
    uniform_noise: float = 0.0

    def __post_init__(self):
        integer(self.n, 'n')
        if self.n > 10:
            raise ValueError('This validated implementation supports n <= 10')
        if not math.isfinite(self.time) or self.time <= 0:
            raise ValueError('time must be positive and finite')
        if len(self.bounds) != 2 or not all(math.isfinite(x) for x in self.bounds):
            raise ValueError('bounds must contain two finite energies')
        lo, hi = self.bounds
        if lo >= hi or hi-lo > 2*np.pi/self.time*(1+1e-14):
            raise ValueError('Declare an ordered energy branch no wider than one phase period')
        if max(abs(lo*self.time), abs(hi*self.time)) > 100*np.pi:
            raise ValueError('Energy branch is too far from the numerical origin')
        if not math.isfinite(self.uniform_noise) or not 0 <= self.uniform_noise <= 1:
            raise ValueError('uniform_noise must be a known fraction in [0, 1]')

    @property
    def bin_width(self):
        return 2*np.pi/(self.time*2**self.n)

    @property
    def derivative_bound(self):
        # Fourier series: p(delta)=1/M+2/M^2 sum_(r=1)^(M-1)(M-r)cos(2pi*r*delta).
        # The triangle inequality gives |dp/dE| <= t*(M^2-1)/(3*M).
        m = 2**self.n
        return (1-self.uniform_noise)*self.time*(m*m-1)/(3*m)

    def probabilities(self, energies):
        energy = np.asarray(energies, dtype=float)
        if not np.all(np.isfinite(energy)):
            raise ValueError('Energy must be finite')
        m = 2**self.n
        delta = (-energy[..., None]*self.time/(2*np.pi)-np.arange(m)/m+.5) % 1-.5
        p = (np.sinc(m*delta)/np.sinc(delta))**2
        # Explicitly enforce analytical zeros and peaks at representable grid points.
        grid = np.rint(m*delta)
        at_grid = np.abs(m*delta-grid) < 8*np.finfo(float).eps*m
        p = np.where(at_grid, np.where(grid == 0, 1.0, 0.0), p)
        return (1-self.uniform_noise)*p+self.uniform_noise/m


def log_likelihood(experiment, counts, energy):
    return np.sum(xlogy(counts, experiment.probabilities(energy)), axis=-1)


def fit(experiment, counts, exhaustive=False):
    """Multi-start likelihood estimate; best eight coarse maxima refined.

    Fit accuracy affects efficiency/point estimates, not coverage: any estimate
    trained on the opposite batch yields a valid held-out likelihood ratio.
    """
    counts = counts_array(counts, experiment.n)
    lo, hi = experiment.bounds
    if experiment.uniform_noise == 1:
        return (lo+hi)/2
    pieces = max(32, int(np.ceil((hi-lo)/experiment.bin_width))*16)
    grid = np.linspace(lo, hi, pieces+1)
    values = log_likelihood(experiment, counts, grid)
    candidates = [(float(values[i]), float(grid[i])) for i in (0, int(np.argmax(values)), pieces)]
    peaks = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))+1
    if not exhaustive:
        peaks = sorted(peaks, key=lambda i: values[i], reverse=True)[:8]
    for i in peaks:
        if not np.isfinite(values[i]):
            continue
        opt = minimize_scalar(lambda e: -float(log_likelihood(experiment, counts, e)),
                              bounds=(grid[i-1], grid[i+1]), method='bounded',
                              options={'xatol': 1e-12*max(1, hi-lo)})
        candidates.append((-float(opt.fun), float(opt.x)))
    return max(candidates)[1]


def _merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]+1e-14:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([float(lo), float(hi)])
    return result


class Inference:
    def __init__(self, experiment, batch_a, batch_b, alpha=.05):
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        self.experiment = experiment
        self.a = counts_array(batch_a, experiment.n)
        self.b = counts_array(batch_b, experiment.n)
        self.alpha = alpha
        self.fit_a = fit(experiment, self.a)
        self.fit_b = fit(experiment, self.b)
        # Each numerator uses an estimate fitted ONLY on the other batch.
        self.numerator_a = float(log_likelihood(experiment, self.a, self.fit_b))
        self.numerator_b = float(log_likelihood(experiment, self.b, self.fit_a))

    def log_e(self, energy):
        la = log_likelihood(self.experiment, self.a, energy)
        lb = log_likelihood(self.experiment, self.b, energy)
        terms = np.stack((self.numerator_a-la if np.isfinite(self.numerator_a) else np.full_like(la,-np.inf),
                          self.numerator_b-lb if np.isfinite(self.numerator_b) else np.full_like(lb,-np.inf)), axis=0)
        # A zero numerator contributes zero, including at a zero denominator.
        if not np.isfinite(self.numerator_a): terms[0] = -np.inf
        if not np.isfinite(self.numerator_b): terms[1] = -np.inf
        return logsumexp(terms, axis=0)-np.log(2)

    def contains(self, energy):
        lo, hi = self.experiment.bounds
        return bool(lo <= energy <= hi and self.log_e(energy) <= -np.log(self.alpha))

    def confidence_set(self, tolerance=None, max_cells=300000):
        """Outer enclosure of {E: e(E)<=1/alpha}; never discard an unclassified cell.

        Bound p on each interval using a global analytic derivative bound.
        Upper likelihood bounds imply lower e-values: only cells proven rejected
        throughout are dropped. Retained boundary cells are subdivided to tolerance.
        This is conservative continuous inversion, not evaluating a posterior grid.
        """
        ex = self.experiment
        tolerance = ex.bin_width/10000 if tolerance is None else tolerance
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        integer(max_cells, 'max_cells')
        cells = np.array([ex.bounds], dtype=float)
        retained = []
        visited = 0
        cutoff = -np.log(self.alpha)
        while len(cells):
            visited += len(cells)
            if visited > max_cells:
                raise RuntimeError('Confidence inversion exceeded cell budget; no truncated set returned')
            mid = cells.mean(axis=1)
            radius = (cells[:,1]-cells[:,0])/2
            p = ex.probabilities(mid)
            pad = ex.derivative_bound*radius[:,None]+1e-12
            upper_p = np.minimum(1, p+pad)
            lower_p = np.maximum(0, p-pad)
            upper_la = np.sum(xlogy(self.a, upper_p),axis=1)
            upper_lb = np.sum(xlogy(self.b, upper_p),axis=1)
            lower_la = np.sum(xlogy(self.a, lower_p),axis=1)
            lower_lb = np.sum(xlogy(self.b, lower_p),axis=1)
            def ratio_bound(la, lb):
                a = self.numerator_a-la if np.isfinite(self.numerator_a) else np.full_like(la,-np.inf)
                b = self.numerator_b-lb if np.isfinite(self.numerator_b) else np.full_like(lb,-np.inf)
                return np.logaddexp(a,b)-np.log(2)
            lower_e = ratio_bound(upper_la,upper_lb)
            upper_e = ratio_bound(lower_la,lower_lb)
            reject = lower_e > cutoff+1e-10
            accept = (~reject) & ((upper_e <= cutoff-1e-10) | (2*radius <= tolerance))
            retained.extend(cells[accept].tolist())
            uncertain = cells[~reject & ~accept]
            halves = uncertain.mean(axis=1)
            cells = np.concatenate((np.column_stack((uncertain[:,0],halves)),
                                    np.column_stack((halves,uncertain[:,1]))))
        return {'intervals':_merge(retained), 'tolerance':float(tolerance), 'visited_cells':visited}

    def summary(self, tolerance=None):
        result = self.confidence_set(tolerance)
        result.update(estimate=fit(self.experiment,self.a+self.b), alpha=self.alpha,
                      energy_bin_width=self.experiment.bin_width,
                      shots_a=int(self.a.sum()), shots_b=int(self.b.sum()))
        result['status'] = 'nonempty' if result['intervals'] else 'model_incompatible'
        return result


def difference_set(upper, lower):
    """Interval subtraction; input sets must have a jointly controlled error rate."""
    return _merge([(a-d,b-c) for a,b in upper for c,d in lower])


def classify_ordered(results, minimum_gap=0.0, family_alpha=.05):
    """Declare all ordered energies separated only if every possible adjacent gap > margin."""
    if not math.isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError('minimum_gap must be nonnegative')
    if not math.isfinite(family_alpha) or not 0 < family_alpha < 1:
        raise ValueError('family_alpha must be between 0 and 1')
    if any(not math.isfinite(r['alpha']) or not 0 < r['alpha'] < 1 for r in results):
        raise ValueError('Each state must have a valid alpha')
    if len(results) < 2 or sum(r['alpha'] for r in results) > family_alpha+1e-14:
        raise ValueError('Allocate per-state alpha so their sum <= family_alpha')
    if any(not r['intervals'] for r in results):
        return {'status':'model_incompatible', 'gap_sets':[]}
    gaps = [difference_set(b['intervals'],a['intervals']) for a,b in zip(results,results[1:])]
    resolved = all(g[0][0] > minimum_gap for g in gaps)
    return {'status':'resolved' if resolved else 'not_established', 'gap_sets':gaps}


def binomial_interval(successes, trials, alpha=.05):
    trials = integer(trials, 'trials')
    successes = integer(successes, 'successes', 0)
    if successes > trials or not 0 < alpha < 1:
        raise ValueError('Invalid successes or alpha')
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha/2,successes,trials-successes+1))
    hi = 1.0 if successes == trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes))
    return [lo,hi]


def nearest_bins(energy, time, n):
    m=2**integer(n,'n')
    position=(-energy*time/(2*np.pi)*m) % m
    lower=int(np.floor(position))
    if abs(position-lower-.5) < 1e-10:
        return sorted({lower % m,(lower+1) % m})
    return [int(np.floor(position+.5)) % m]


def ideal_modal_regions(n, time, j=1.0, field_max=.3):
    """Exact half-bin transition enumeration for E=(J-2h,J,J+2h).

    A diagnostic of the OLD nearest-modal-bin definition, not an inference limit.
    All boundary ties and every open interval are returned; no monotonicity assumed.
    """
    Experiment(n,time,(-np.pi/time,np.pi/time))
    if not math.isfinite(j) or not math.isfinite(field_max) or field_max <= 0:
        raise ValueError('Invalid j or field_max')
    m=2**n
    points=[0.0,float(field_max)]
    for slope in (-2,2):
        endpoints=[-(j+slope*h)*time*m/(2*np.pi) for h in (0,field_max)]
        for k in range(math.floor(min(endpoints))-1,math.ceil(max(endpoints))+1):
            h=(-(k+.5)*2*np.pi/(time*m)-j)/slope
            if 0 < h < field_max: points.append(h)
    points=sorted(set(points))
    def distinct(h):
        bins=[nearest_bins(j+s*h,time,n) for s in (-2,0,2)]
        return all(set(bins[a]).isdisjoint(bins[b]) for a in range(3) for b in range(a+1,3))
    return {'boundaries':[{'h':h,'all_ties_distinct':distinct(h)} for h in points],
            'open_regions':[{'lo':a,'hi':b,'distinct':distinct((a+b)/2)} for a,b in zip(points,points[1:])]}
