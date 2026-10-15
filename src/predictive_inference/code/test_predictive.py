import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binom
from predictive import *


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.ex = Experiment(2, .8*np.pi/3, (-3.75,3.75))
        rng = np.random.default_rng(719)
        p = .8*self.ex.probabilities(1.013)+.2/4
        self.a, self.b = rng.multinomial(64,p), rng.multinomial(64,p)

    def test_convex_profile_bounds(self):
        j = JointInference(self.ex,self.a,self.b)
        for energy in np.linspace(-3.75,3.75,31):
            p = self.ex.probabilities(energy)
            lower, upper, eta = j._profile_bounds(p)
            opt = minimize_scalar(lambda v: float(j.log_e(energy,v)), bounds=(0,1), method='bounded')
            reference = min(opt.fun, float(j.log_e(energy,0)),float(j.log_e(energy,1)))
            self.assertLessEqual(lower[0],reference+1e-7)
            self.assertLessEqual(upper[0], reference+1e-5)
            self.assertLessEqual(upper[0]-lower[0],1e-6)

    def test_projection_contains_dense_joint_points_and_refines(self):
        j = JointInference(self.ex,self.a,self.b)
        coarse = j.confidence_set(tolerance=.01)['intervals']
        fine = j.confidence_set(tolerance=.002)['intervals']
        for a,b in fine:
            self.assertTrue(any(c<=a and b<=d for c,d in coarse))
        for energy in np.linspace(-3.75,3.75,301):
            if np.min(j.log_e(energy,np.linspace(0,1,101))) <= -np.log(.05):
                self.assertTrue(any(a<=energy<=b for a,b in fine))

    def test_probability_bounds_on_rectangles(self):
        rng=np.random.default_rng(31)
        for _ in range(50):
            e=rng.uniform(-3,3); r=rng.uniform(0,.2); eta=rng.uniform()
            p=self.ex.probabilities(e)
            actual=(1-eta)*self.ex.probabilities(e+rng.uniform(-r,r))+eta/4
            bound=(1-eta)*np.minimum(1,p+self.ex.derivative_bound*r+1e-12)+eta/4
            self.assertTrue(np.all(actual<=bound+1e-14))

    def test_uniform_limit(self):
        j=JointInference(self.ex,[16]*4,[16]*4)
        self.assertEqual(j.confidence_set()['intervals'],[[-3.75,3.75]])

    def test_alias_endpoints(self):
        j=JointInference(self.ex,[0,0,64,0],[0,0,64,0])
        cs=j.confidence_set(tolerance=.002)['intervals']
        self.assertEqual(cs[0][0],-3.75)
        self.assertEqual(cs[-1][1],3.75)
        self.assertGreaterEqual(len(cs),2)

    def test_exact_enumeration_joint_e_expectation(self):
        ex=Experiment(1,.8*np.pi/3,(-3.75,3.75))
        # Cache all data-dependent fits before evaluating multiple truths.
        experiments=[(a,b,JointInference(ex,[a,3-a],[b,3-b])) for a in range(4) for b in range(4)]
        for energy,eta in [(-3.,.2),(0.,0.),(1.013,.7),(3.75,.1)]:
            q=(1-eta)*ex.probabilities(energy)+eta/2
            expectation=0.; coverage=0.
            for a,b,j in experiments:
                prob=binom.pmf(a,3,q[0])*binom.pmf(b,3,q[0])
                if prob==0: continue
                value=float(j.log_e(energy,eta))
                expectation+=prob*np.exp(value)
                coverage+=prob*(value<=-np.log(.05))
            self.assertLessEqual(expectation,1+1e-10)
            self.assertGreaterEqual(coverage,.95-1e-10)

    def test_exact_enumeration_mixture(self):
        ex=Experiment(1,.8*np.pi/3,(-3.75,3.75),.2)
        for energy in (-3.,0.,1.013,3.75):
            p=ex.probabilities(energy)[0]
            total=0.
            for k in range(7):
                a=min(3,k); b=k-a
                obj=MixtureInference(ex,[a,3-a],[b,3-b])
                total+=binom.pmf(k,6,p)*np.exp(obj.log_e(energy))
            self.assertAlmostEqual(total,1.,places=10)

    def test_invalid_inputs_and_budget(self):
        with self.assertRaises(ValueError): JointInference(self.ex,self.a,self.b,alpha=np.nan)
        with self.assertRaises(ValueError): JointInference(self.ex,[.5]*4,self.b)
        j=JointInference(self.ex,self.a,self.b)
        with self.assertRaises(ValueError): j.log_e(0,1.1)
        with self.assertRaises(ValueError): j.confidence_set(tolerance=0)
        with self.assertRaises(RuntimeError): j.confidence_set(max_cells=1)


if __name__=='__main__': unittest.main(verbosity=2)
