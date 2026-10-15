import unittest
from refined_inversion import *

class Tests(unittest.TestCase):
 def test_one_bit_derivative_envelope(self):
  ex=Experiment(1,1.,(-np.pi,np.pi))
  for e in np.linspace(-3,3,21):
   lo,hi=probability_envelope(ex,np.array([e]),np.array([1e-5]))
   for v in np.linspace(e-1e-5,e+1e-5,11):
    p=np.array([np.cos(v/2)**2,np.sin(v/2)**2]);self.assertTrue(np.all(lo[0]<=p+1e-14));self.assertTrue(np.all(p<=hi[0]+1e-14))

 def test_multiregister_envelopes(self):
  rng=np.random.default_rng(8231)
  for n in (2,4,6,8):
   ex=Experiment(n,.8*np.pi/3,(-3.75,3.75));mid=np.r_[rng.uniform(-3.5,3.5,25),0.,ex.bin_width,-3.75,3.75];radius=10.**rng.uniform(-8,-1,len(mid))
   lo,hi=probability_envelope(ex,mid,radius)
   for fraction in np.linspace(-1,1,31):
    p=ex.probabilities(mid+fraction*radius);self.assertTrue(np.all(lo<=p+1e-13));self.assertTrue(np.all(p<=hi+1e-13))
   p=ex.probabilities(mid);pad=ex.derivative_bound*radius[:,None]+1e-12
   self.assertTrue(np.all(lo>=np.maximum(0,p-pad)));self.assertTrue(np.all(hi<=np.minimum(1,p+pad)))

 def test_sets_and_refinement(self):
  ex=Experiment(4,.8*np.pi/3,(-3.75,3.75));rng=np.random.default_rng(9983);a,b=rng.multinomial(128,.6*ex.probabilities(1.013)+.4/16,size=2)
  for cls,old in ((RefinedJoint,JointInference),(RefinedProfile,ProfileLikelihood)):
   obj=cls(ex,a,b);s=obj.confidence_set();fine=obj.confidence_set(tolerance=ex.bin_width/100000);coarse=old(ex,a,b).confidence_set()
   for x,y in s['intervals']:self.assertTrue(any(c<=x and y<=d for c,d in coarse['intervals']))
   for x,y in fine['intervals']:self.assertTrue(any(c<=x and y<=d for c,d in s['intervals']))
   for e in np.linspace(-3.75,3.75,501):
    accepted=obj.membership(e)[0] if cls is RefinedProfile else obj.projection_membership_bounds(e)[1][0]<=-np.log(.05)
    if accepted:self.assertTrue(any(x<=e<=y for x,y in s['intervals']))

 def test_limits_and_failure(self):
  ex=Experiment(4,.8*np.pi/3,(-3.75,3.75));a=np.full(16,8)
  for cls in (RefinedJoint,RefinedProfile):
   obj=cls(ex,a,a);self.assertEqual(obj.confidence_set()['intervals'],[[-3.75,3.75]])
   with self.assertRaises(ValueError):obj.confidence_set(tolerance=0)
  a=np.zeros(16,dtype=int);a[0]=128
  for cls in (RefinedJoint,RefinedProfile):
   with self.assertRaises(RuntimeError):cls(ex,a,a).confidence_set(max_cells=1)
if __name__=='__main__':unittest.main()
