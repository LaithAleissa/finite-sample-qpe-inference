"""Independent limiting cases and numerical failure checks for the new baseline."""
import unittest
from baseline import *

class BaselineTests(unittest.TestCase):
    def test_analytic_one_bit(self):
        # p0(E)=cos(E/2)^2 when n=1,t=1. All N outcomes are zero.
        ex=Experiment(1,1.,(-np.pi,np.pi)); lr=LikelihoodRatio(ex,[10,0],[10,0])
        half=2*np.arccos(np.exp(-lr.cutoff/(4*20)))
        s=lr.confidence_set(); self.assertEqual(len(s['intervals']),1)
        a,b=s['intervals'][0]; self.assertLessEqual(a,-half); self.assertGreaterEqual(b,half)
        self.assertLess(abs(a+half),2*s['tolerance']); self.assertLess(abs(b-half),2*s['tolerance'])

    def test_uninformative(self):
        ex=Experiment(2,1.,(-3.,3.),1.)
        self.assertEqual(LikelihoodRatio(ex,[1,2,3,4],[4,3,2,1]).confidence_set()['intervals'],[[-3.,3.]])

    def test_branch_union(self):
        ex=Experiment(2,1.,(-np.pi,np.pi)); lr=LikelihoodRatio(ex,[0,0,20,0],[0,0,20,0])
        result=lr.confidence_set(); self.assertEqual(len(result['intervals']),2)
        self.assertTrue(lr.contains(-np.pi)); self.assertTrue(lr.contains(np.pi))
        self.assertLess(length(result),1.)

    def test_dense_and_refinement(self):
        ex=Experiment(2,1.,(-3.,3.),.1); lr=LikelihoodRatio(ex,[3,8,20,2],[5,9,18,1])
        coarse=lr.confidence_set(ex.bin_width/1000); fine=lr.confidence_set()
        for a,b in fine['intervals']: self.assertTrue(any(c<=a and b<=d for c,d in coarse['intervals']))
        for e in np.linspace(-3,3,10001):
            if lr.contains(e): self.assertTrue(any(a<=e<=b for a,b in fine['intervals']))
        best=fit(ex,lr.counts,exhaustive=True)
        self.assertAlmostEqual(lr.maximum,float(log_likelihood(ex,lr.counts,best)),places=8)

    def test_failure(self):
        ex=Experiment(2,1.,(-3.,3.)); lr=LikelihoodRatio(ex,[3,8,20,2],[5,9,18,1])
        with self.assertRaises(RuntimeError): lr.confidence_set(max_cells=1)
        with self.assertRaises(ValueError): lr.confidence_set(tolerance=0)
        with self.assertRaises(ValueError): LikelihoodRatio(ex,[1.5,2,3,4],[1,2,3,4])

if __name__=='__main__': unittest.main(verbosity=2)
