import unittest
from circuit_noise import *
class CircuitTests(unittest.TestCase):
    def test_depolarizing_convention(self):
        for n in (1,2):
            rho=DensityMatrix.from_label('0'*n);p=.23
            observed=rho.evolve(depolarizing_channel(p,n)).data
            self.assertTrue(np.allclose(observed,(1-p)*rho.data+p*np.eye(2**n)/2**n,atol=1e-13))
    def test_zero_noise_qpe_and_full_readout(self):
        c=transpile(qpe(Model(),'triplet_plus',4,.8377580409572781),basis_gates=['u','cx'],optimization_level=1,seed_transpiler=731)
        ex=Experiment(4,.8377580409572781,(-3.75,3.75))
        self.assertLess(max(abs(noisy_distribution(c,0,0,0)-ex.probabilities(1))),1e-12)
        self.assertTrue(np.allclose(noisy_distribution(c,0,0,.5),np.full(16,1/16),atol=1e-12))
if __name__=='__main__':unittest.main()
