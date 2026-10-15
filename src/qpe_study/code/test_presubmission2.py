import unittest
from presubmission2 import *
from qiskit.quantum_info import Statevector
from scipy.special import gammaln
import itertools

class FollowupTests(unittest.TestCase):
    def test_aliasing_of_unprepared_states_does_not_change_target_kernel(self):
        ex=Experiment(4,.8377580409572781,(-3.75,3.75))
        for e in ENERGIES:
            actual=Statevector.from_instruction(shifted_circuit(e)).probabilities(list(range(4)))
            self.assertLess(max(abs(actual-ex.probabilities(e))),1e-12)

    def test_reference_law_normalizes_over_all_small_count_outcomes(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));N=3
        for name in REFERENCES:
            mass=0.
            for c in itertools.product(range(N+1),repeat=4):
                if sum(c)!=N:continue
                cc=np.array(c);logcoef=gammaln(N+1)-sum(gammaln(cc+1))
                mass+=np.exp(predictive(ex,cc,name)+logcoef)
            self.assertAlmostEqual(mass,1.,places=12)

    def test_base_matches_frozen_mixture_with_zero_count_bins(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));c=np.array([0,2,0,6])
        self.assertAlmostEqual(predictive(ex,c,'base'),mixture_numerator(ex,c),places=12)

    def test_duplicate_grid_definitions_and_half_open_seam(self):
        self.assertEqual(dimensions(4,1024,'dense'),dimensions(4,1024,'adaptive'))
        self.assertEqual(dimensions(4,256,'base'),dimensions(4,256,'adaptive'))
        self.assertGreater(dimensions(4,4096,'adaptive')[0],dimensions(4,1024,'adaptive')[0])
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));ne,nh=dimensions(2,256,'base')
        logs=grid_logs(2,ex.time,tuple(ex.bounds),ne,nh)
        self.assertTrue(np.allclose(np.exp(logs).sum(axis=1),1,atol=1e-13))
        self.assertEqual(logs.shape,(ne*nh,4))
if __name__=='__main__':unittest.main()
