import unittest
import numpy as np
from scipy.stats import binomtest
from final_scientific_methods import *

class ScientificRevisionTests(unittest.TestCase):
    def setUp(self):self.ex=Experiment(2,.8377580409572781,(-3.75,3.75))
    def test_cp_marginals(self):
        obj=SimultaneousCP(self.ex,[0,2,5,3],[0,3,4,3])
        for k,c in enumerate(obj.counts):
            ci=binomtest(int(c),20).proportion_ci(confidence_level=1-.05/4,method='exact')
            self.assertAlmostEqual(obj.lower[k],ci.low,12);self.assertAlmostEqual(obj.upper[k],ci.high,12)
    def test_shared_eta_constraints(self):
        lo,hi,ok=affine_feasible([[0.,1.]],[[1.,-1.]],[.8,.8],True)
        self.assertFalse(ok[0])
    def test_zero_slope(self):
        self.assertFalse(affine_feasible([[.25]],[[0.]],[.3],True)[2][0])
        self.assertTrue(affine_feasible([[.25]],[[0.]],[.2],True)[2][0])
    def test_cp_uniform_full(self):
        obj=SimultaneousCP(self.ex,[25]*4,[25]*4)
        self.assertEqual(obj.confidence_set()['intervals'],[[-3.75,3.75]])
    def test_streaming_same_grid(self):
        a=[30,2,5,3];b=[25,3,4,8];base=QPEMixture(self.ex,a,b)
        self.assertAlmostEqual(streamed_mixture(self.ex,np.array(a)+b,1),base.log_predictive,11)
    def test_drift_nested_point(self):
        shared=RefinedJoint(self.ex,[30,2,5,3],[25,3,4,8]);drift=DriftCF(self.ex,shared.a,shared.b,shared=shared)
        grid=np.linspace(-3.75,3.75,101)
        _,s=shared.projection_membership_bounds(grid);d,_=drift.projection_membership_bounds(grid)
        self.assertTrue(np.all(d<=s+1e-8))
    def test_drift_separable_minimum(self):
        shared=RefinedJoint(self.ex,[30,2,5,3],[25,3,4,8]);obj=DriftCF(self.ex,shared.a,shared.b,shared=shared)
        e=.3;p=self.ex.probabilities(e);eta=np.linspace(0,1,2001)
        q=(1-eta[:,None])*p+eta[:,None]/4
        from scipy.special import xlogy
        exact=np.logaddexp(shared.na-np.max(np.sum(xlogy(shared.a,q),axis=1)),shared.nb-np.max(np.sum(xlogy(shared.b,q),axis=1)))-np.log(2)
        lo,hi=obj.projection_membership_bounds(e)
        self.assertLessEqual(lo[0],exact+1e-8);self.assertLessEqual(hi[0],exact+1e-6)
    def test_no_partial_budget(self):
        obj=SimultaneousCP(self.ex,[100,0,0,0],[100,0,0,0])
        with self.assertRaises(RuntimeError):obj.confidence_set(max_cells=1)
    def test_exact_small_sample_cp_coverage(self):
        from scipy.stats import multinomial
        for eta in [0.,.2,1.]:
            p=(1-eta)*self.ex.probabilities(.3)+eta/4;coverage=0.
            for a in range(5):
                for b in range(5-a):
                    for c in range(5-a-b):
                        counts=[a,b,c,4-a-b-c];left=counts.copy();right=[0]*4
                        index=next(i for i,x in enumerate(left) if x);left[index]-=1;right[index]=1
                        obj=SimultaneousCP(self.ex,left,right)
                        if obj.membership(.3)[0]:coverage+=multinomial.pmf(counts,4,p)
            self.assertGreaterEqual(coverage,.95-1e-12)

if __name__=='__main__':unittest.main()
