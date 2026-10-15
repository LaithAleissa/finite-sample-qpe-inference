import unittest,itertools
import numpy as np
from valuable_followup import dirichlet_numerator,UnknownDirichlet,Experiment

class PredictiveLawTests(unittest.TestCase):
    def test_sequence_law_normalizes(self):
        # Exhaust all length-four binary sequences, including multiplicity.
        total=sum(np.exp(dirichlet_numerator(np.bincount(seq,minlength=2))) for seq in itertools.product(range(2),repeat=4))
        self.assertAlmostEqual(total,1.,places=13)

    def test_predictor_permutation_and_zero_count_categories(self):
        c=np.array([4,0,2,1]);self.assertEqual(dirichlet_numerator(c),dirichlet_numerator(c[::-1]))
        self.assertLess(dirichlet_numerator(c),0)
        self.assertAlmostEqual(dirichlet_numerator(np.zeros(4,dtype=int)),0)

    def test_queries_and_accumulation(self):
        self.assertEqual(4*2142*15,4*510*63)
        lam=1-.9**(1/15)
        self.assertAlmostEqual(1-(1-lam)**15,.1)
        self.assertGreater(1-(1-lam)**63,.1)

    def test_unknown_channel_and_alpha_validation(self):
        with self.assertRaises(ValueError):UnknownDirichlet(Experiment(2,1,(-np.pi,np.pi),.1),[1,0,0,0],[1,0,0,0])
        with self.assertRaises(ValueError):UnknownDirichlet(Experiment(2,1,(-np.pi,np.pi)),[1,0,0,0],[1,0,0,0],0)
