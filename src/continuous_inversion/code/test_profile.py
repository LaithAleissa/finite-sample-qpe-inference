import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from profile_lr import *
class Tests(unittest.TestCase):
 def setUp(self):
  self.ex=Experiment(4,.8*np.pi/3,(-3.75,3.75));rng=np.random.default_rng(444)
  self.a,self.b=rng.multinomial(128,.8*self.ex.probabilities(1.013)+.2/16,size=2)
 def test_profile_bounds(self):
  for e in np.linspace(-3.75,3.75,53):
   p=self.ex.probabilities(e);c=self.a+self.b;l,u,eta=profile_bounds(p,c)
   opt=minimize_scalar(lambda x:-sum(xlogy(c,(1-x)*p+x/16)),bounds=(0,1),method='bounded')
   ref=max(-opt.fun,sum(xlogy(c,p)),-sum(c)*np.log(16))
   self.assertLessEqual(ref,u[0]+1e-8);self.assertLessEqual(l[0],ref+1e-5);self.assertLess(u[0]-l[0],1e-6)
 def test_grid_refinement_and_dense_containment(self):
  obj=ProfileLikelihood(self.ex,self.a,self.b);dense=ProfileLikelihood(self.ex,self.a,self.b,grid_per_bin=64)
  self.assertAlmostEqual(obj.maximum,dense.maximum,places=7)
  cs=obj.confidence_set(tolerance=.005)
  for e in np.linspace(-3.75,3.75,1001):
   if obj.membership(e)[0]:self.assertTrue(any(a<=e<=b for a,b in cs['intervals']))
  fine=obj.confidence_set(tolerance=.001)
  for a,b in fine['intervals']:self.assertTrue(any(c<=a and b<=d for c,d in cs['intervals']))
 def test_boundary_adjacent_maximum(self):
  rng=np.random.default_rng(923);p=.9*self.ex.probabilities(-3.7499)+.1/16;a,b=rng.multinomial(512,p,size=2)
  obj=ProfileLikelihood(self.ex,a,b);dense=ProfileLikelihood(self.ex,a,b,grid_per_bin=64)
  self.assertAlmostEqual(obj.maximum,dense.maximum,places=7)
  for bounds in ((-3.75,-3.7),(3.7,3.75)):
   opt=minimize_scalar(lambda e:-obj.profile(e)[0][0],bounds=bounds,method='bounded')
   self.assertGreaterEqual(obj.maximum,-opt.fun-1e-7)
 def test_uniform_full_branch(self):
  obj=ProfileLikelihood(self.ex,[8]*16,[8]*16);self.assertEqual(obj.confidence_set()['intervals'],[[-3.75,3.75]])
 def test_grid_and_alias(self):
  c=np.zeros(16,dtype=int);c[8]=64;obj=ProfileLikelihood(self.ex,c,c);cs=obj.confidence_set()
  self.assertEqual(cs['intervals'][0][0],-3.75);self.assertEqual(cs['intervals'][-1][1],3.75)
  self.assertAlmostEqual(obj.maximum,0.)
 def test_invalid_inputs_and_budget(self):
  with self.assertRaises(ValueError):ProfileLikelihood(self.ex,self.a,self.b,alpha=np.nan)
  obj=ProfileLikelihood(self.ex,self.a,self.b)
  with self.assertRaises(ValueError):obj.confidence_set(tolerance=0)
  with self.assertRaises(RuntimeError):obj.confidence_set(max_cells=1)
if __name__=='__main__':unittest.main(verbosity=2)
