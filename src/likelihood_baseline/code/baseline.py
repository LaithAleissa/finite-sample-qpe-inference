"""Pooled likelihood-ratio benchmark, retaining the entire energy branch.

The chi-square cutoff is an asymptotic reference, not a finite-sample theorem.
No generating energy is accepted by the estimator or set inversion.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent
sys.path.insert(0, str(SRC/'known_noise'))
from inference import Experiment, Inference, counts_array, fit, log_likelihood, _merge


class LikelihoodRatio:
    def __init__(self, experiment, batch_a, batch_b, alpha=.05):
        if not 0 < alpha < 1: raise ValueError('alpha must be in (0,1)')
        self.experiment = experiment
        self.counts = counts_array(batch_a, experiment.n)+counts_array(batch_b, experiment.n)
        self.alpha = alpha
        self.estimate = fit(experiment, self.counts)
        self.maximum = float(log_likelihood(experiment, self.counts, self.estimate))
        if not np.isfinite(self.maximum): raise RuntimeError('No finite pooled likelihood found')
        self.cutoff = float(chi2.ppf(1-alpha, 1))
        self.threshold = self.maximum-self.cutoff/2

    def contains(self, energy):
        lo, hi = self.experiment.bounds
        return bool(lo <= energy <= hi and log_likelihood(self.experiment, self.counts, energy) >= self.threshold)

    def confidence_set(self, tolerance=None, max_cells=300000):
        ex = self.experiment
        tolerance = ex.bin_width/10000 if tolerance is None else tolerance
        if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError('Invalid tolerance')
        if not isinstance(max_cells,int) or max_cells < 1: raise ValueError('Invalid cell budget')
        cells = np.array([ex.bounds],float); retained=[]; visited=0
        while len(cells):
            visited += len(cells)
            if visited > max_cells: raise RuntimeError('LR inversion exceeded cell budget')
            mid=cells.mean(axis=1); radius=np.diff(cells,axis=1)/2
            p=ex.probabilities(mid); pad=ex.derivative_bound*radius+1e-12
            upper=np.sum(xlogy(self.counts,np.minimum(1,p+pad)),axis=1)
            lower=np.sum(xlogy(self.counts,np.maximum(0,p-pad)),axis=1)
            reject=upper < self.threshold-1e-10
            accept=(~reject)&((lower >= self.threshold+1e-10)|(2*radius[:,0] <= tolerance))
            retained.extend(cells[accept].tolist())
            uncertain=cells[~reject & ~accept]; halves=uncertain.mean(axis=1)
            cells=np.concatenate((np.column_stack((uncertain[:,0],halves)),np.column_stack((halves,uncertain[:,1]))))
        return {'intervals':_merge(retained),'tolerance':float(tolerance),'visited_cells':visited}


def length(result):
    return float(sum(b-a for a,b in result['intervals']))
