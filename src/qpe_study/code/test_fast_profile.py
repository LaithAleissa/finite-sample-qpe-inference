import unittest,time
from fast_profile import *
class FastProfileTests(unittest.TestCase):
    def test_profile_equivalence(self):
        ex=Experiment(4,.8377580409572781,(-3.75,3.75));rng=np.random.default_rng(9987)
        counts=rng.multinomial(256,np.full(16,1/16),size=10);energies=np.linspace(-3.75,3.75,10)
        v,e=batch_profile(ex.probabilities(energies),counts)
        for i in range(10):self.assertAlmostEqual(v[i],profile_bounds(ex.probabilities(energies[i]),counts[i])[0][0],places=9)
    def test_full_search_equivalence(self):
        ex=Experiment(4,.8377580409572781,(-3.75,3.75));rng=np.random.default_rng(7721);allc=[]
        for energy,eta in [(1.013,.9),(1.013,.98),(1.013,1),(-3.7499,.1),(0,0)]:
            q=(1-eta)*ex.probabilities(energy)+eta/16;allc.extend(rng.multinomial(256,q,size=4))
        counts=np.array(allc);start=time.perf_counter();maximum=batch_maximum(ex,counts);print('batch seconds per fit',(time.perf_counter()-start)/len(counts))
        for i,c in enumerate(counts):
            aa=c//2;bb=c-aa;o=ProfileLikelihood(ex,aa,bb)
            self.assertLess(abs(maximum[i]-o.maximum),1e-6)
if __name__=='__main__':unittest.main()
