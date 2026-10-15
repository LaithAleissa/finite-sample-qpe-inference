import unittest,itertools
from methods import *
class MethodsTests(unittest.TestCase):
    def test_mixture_normalization_and_e_expectation(self):
        # Enumerate every TWO-shot sequence; support zeros included.
        ex=Experiment(1,.8377580409572781,(-3.75,3.75))
        for e,eta in [(0,0),(.731,.2),(3.75,1)]:
            q=(1-eta)*ex.probabilities(e)+eta/2;mean=0.;mass=0.
            for seq in itertools.product(range(2),repeat=2):
                a=np.bincount(seq[:1],minlength=2);b=np.bincount(seq[1:],minlength=2);o=QPEMixture(ex,a,b)
                den=np.prod(q[list(seq)]);num=np.exp(o.log_predictive);mass+=num
                if den>0:mean+=den*num/den
            self.assertAlmostEqual(mass,1,places=12);self.assertLessEqual(mean,1+1e-12)
    def test_reference_half_open(self):
        e,eta,logs=reference(4,.8377580409572781,(-3.75,3.75))
        self.assertEqual(len(e),512);self.assertEqual(len(eta),17)
        self.assertEqual(e[0],-3.75);self.assertLess(e[-1],3.75)
        self.assertTrue(np.allclose(np.exp(logs).sum(axis=1),1))
    def test_zero_support_and_alias(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));a=np.array([10,0,0,0]);b=a.copy();obj=QPEMixture(ex,a,b)
        self.assertTrue(np.isfinite(obj.log_predictive));self.assertEqual(obj.membership(-3.75),obj.membership(3.75))
    def test_dense_points_inside_sets(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));a=np.array([8,13,7,4]);b=np.array([9,11,6,6]);o=QPEMixture(ex,a,b);s=o.confidence_set()
        grid=np.linspace(*ex.bounds,2001);v=profile_bounds(ex.probabilities(grid),o.counts)[0]>=o.threshold
        self.assertTrue(all(any(lo<=e<=hi for lo,hi in s['intervals']) for e in grid[v]))
    def test_uniform_witness(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));o=QPEMixture(ex,np.array([16]*4),np.array([16]*4))
        self.assertEqual(o.confidence_set()['intervals'],[[-3.75,3.75]])
    def test_budget_fail_closed(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));o=QPEMixture(ex,np.array([60,2,1,1]),np.array([60,1,2,1]))
        with self.assertRaises(RuntimeError):o.confidence_set(max_cells=1)
    def test_known_refinement(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75),.1);a=np.array([30,3,1,2]);b=np.array([29,2,2,3])
        for cls in (Inference,LikelihoodRatio,MixtureInference):
            o=cls(ex,a,b);s=known_invert(o);old=o.confidence_set()
            self.assertLessEqual(length(s),length(old)+1e-9)
            for e in np.linspace(*ex.bounds,2001):
                if o.contains(e):self.assertTrue(any(lo<=e<=hi for lo,hi in s['intervals']))
if __name__=='__main__':unittest.main()
