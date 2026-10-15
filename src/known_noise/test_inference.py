import itertools
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.special import gammaln

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantum_model'))
from dimer import Model, probabilities as circuit_probabilities, ideal_probabilities
from inference import (Experiment, Inference, fit, counts_array, difference_set,
                       classify_ordered, binomial_interval, nearest_bins, ideal_modal_regions)

TIME=.8*np.pi/3
BOUNDS=(-3.75,3.75)


class InferenceTests(unittest.TestCase):
    def test_cli_schema_and_output(self):
        from infer_counts import infer_document
        data=dict(n=2,time=TIME,energy_bounds=list(BOUNDS),known_uniform_noise=0,
                  batch_a=[0,0,100,0],batch_b=[0,0,100,0])
        result=infer_document(data)
        self.assertEqual(result['status'],'nonempty')
        self.assertEqual(result['shots_a'],100)
        with self.assertRaises(ValueError): infer_document(dict(data,true_energy=-3))

    def test_zero_numerators_without_numerical_warnings(self):
        import warnings
        from unittest.mock import patch
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            # Isolate exact zero-numerator handling from optimizer rounding.
            with patch('inference.fit',side_effect=[0.0,3.75]):
                infer=Inference(Experiment(1,TIME,BOUNDS),[10,0],[0,10])
            self.assertTrue(np.isneginf(infer.log_e(0)))
            self.assertEqual(infer.confidence_set()['intervals'],[list(BOUNDS)])

    def test_invalid_family_alpha(self):
        for alpha in (0,-.1,1,float('nan')):
            with self.assertRaises(ValueError):
                classify_ordered([{'alpha':.01,'intervals':[[0,1]]}]*2,family_alpha=alpha)

    def test_probability_oracles(self):
        rng=np.random.default_rng(4739)
        for n in (1,2,4,6,8):
            ex=Experiment(n,TIME,BOUNDS)
            for energy in rng.uniform(-3.74,3.74,4):
                np.testing.assert_allclose(ex.probabilities(energy),ideal_probabilities(energy,TIME,n),atol=2e-13)
            for model in (Model(),Model(1,1,.21,.07),Model(1,1,.2,axis='x')):
                if n > 6: continue
                for state,e in model.energies().items():
                    np.testing.assert_allclose(ex.probabilities(e),circuit_probabilities(model,state,n,TIME),atol=2e-12)
            for k in (0,1,2**n-1):
                energy=-k*ex.bin_width
                expected=np.zeros(2**n); expected[k]=1
                np.testing.assert_allclose(ex.probabilities(energy),expected,atol=1e-13)

    def test_probability_normalization_noise_and_derivative(self):
        for n in (1,4,8):
            ex=Experiment(n,TIME,BOUNDS,.13)
            grid=np.linspace(-3.75,3.75,173)
            p=ex.probabilities(grid)
            np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
            self.assertGreaterEqual(p.min(),.13/(2**n)-1e-15)
            derivative=(ex.probabilities(grid+1e-7)-ex.probabilities(grid-1e-7))/2e-7
            self.assertLessEqual(abs(derivative).max(),ex.derivative_bound*(1+1e-8))

    def test_validation_and_bit_order(self):
        np.testing.assert_array_equal(counts_array({'001':4,'110':2},3),[0,4,0,0,0,0,2,0])
        for counts in ([1.0,2,3,4], [1,-1,2,3], [0,0,0,0], {'0 1':2}, {'01':True}):
            with self.assertRaises(ValueError): counts_array(counts,2)
        for kwargs in ({'time':0},{'bounds':(-5,5)},{'uniform_noise':-.1},{'n':True}):
            args=dict(n=2,time=TIME,bounds=BOUNDS);args.update(kwargs)
            with self.assertRaises(ValueError): Experiment(**args)
        with self.assertRaises(ValueError): binomial_interval(3,2)

    def test_likelihood_uses_tails_and_zero_uncertainty_is_impossible(self):
        ex=Experiment(4,TIME,BOUNDS)
        counts=np.zeros(16,dtype=int);counts[4]=100
        result=Inference(ex,counts,counts).summary()
        self.assertTrue(any(a < -1.875 < b for a,b in result['intervals']))
        self.assertGreater(sum(b-a for a,b in result['intervals']),0)
        rng=np.random.default_rng(789)
        p=ex.probabilities(1.023)
        a=rng.multinomial(3000,p);b=rng.multinomial(3000,p)
        estimate=fit(ex,a+b)
        modal=(-int(np.argmax(a+b))*ex.bin_width+3.75)%7.5-3.75
        self.assertLess(abs(estimate-1.023),abs(modal-1.023))

    def test_multistart_fit_against_all_coarse_maxima(self):
        from inference import log_likelihood
        rng=np.random.default_rng(471)
        for n in (2,4,6,8):
            ex=Experiment(n,TIME,BOUNDS,.1)
            for energy in (-3.74,-1.875,0,1.013):
                c=rng.multinomial(1024,ex.probabilities(energy))
                a=fit(ex,c); b=fit(ex,c,exhaustive=True)
                self.assertGreaterEqual(log_likelihood(ex,c,a),log_likelihood(ex,c,b)-1e-7)

    def test_continuous_enclosure_including_hidden_modes(self):
        rng=np.random.default_rng(22)
        for n,energy,noise in [(1,.6,0),(2,-3.749,0),(4,-1.8749,0),(4,.217,.1)]:
            ex=Experiment(n,TIME,BOUNDS,noise)
            p=ex.probabilities(energy)
            infer=Inference(ex,rng.multinomial(120,p),rng.multinomial(120,p))
            coarse=infer.confidence_set(ex.bin_width/500)
            fine=infer.confidence_set(ex.bin_width/4000)
            grid=np.unique(np.r_[np.linspace(*BOUNDS,20001),energy])
            accepted=grid[infer.log_e(grid)<=-np.log(infer.alpha)]
            for e in accepted:
                self.assertTrue(any(a-1e-12 <= e <= b+1e-12 for a,b in fine['intervals']))
            for a,b in fine['intervals']:
                self.assertTrue(any(c-1e-12 <= a and b <= d+1e-12 for c,d in coarse['intervals']))
        # A one-bit QPE experiment cannot distinguish reflected phases.
        self.assertGreaterEqual(len(Inference(Experiment(1,TIME,BOUNDS),[95,5],[95,5]).summary()['intervals']),2)

    def test_crossfit_finite_sample_guarantee_by_exact_enumeration(self):
        # Exhaust all two independent three-shot batches in a 2-category experiment.
        ex=Experiment(1,TIME,BOUNDS,.1)
        batches=[np.array([i,3-i]) for i in range(4)]
        for energy in (-3.5,-.3,0,2.4):
            p=ex.probabilities(energy)
            expected_e=0.0; rejected=0.0
            for a,b in itertools.product(batches,repeat=2):
                weight=1.0
                for counts in (a,b):
                    weight*=np.exp(gammaln(4)-gammaln(counts+1).sum()+np.dot(counts,np.log(p)))
                infer=Inference(ex,a,b,alpha=.2)
                expected_e+=weight*np.exp(infer.log_e(energy))
                rejected+=weight*(not infer.contains(energy))
            self.assertLessEqual(expected_e,1+1e-10)
            self.assertLessEqual(rejected,.2+1e-10)

    def test_resolution_gap_and_family_error_guards(self):
        np.testing.assert_allclose(difference_set([[2,3],[5,6]],[[0,1]]),[[1,3],[4,6]])
        def r(a,b):return {'intervals':[[a,b]],'alpha':.05/3}
        self.assertEqual(classify_ordered([r(0,.2),r(.3,.5),r(.6,.8)])['status'],'resolved')
        self.assertEqual(classify_ordered([r(0,.3),r(.3,.5),r(.6,.8)])['status'],'not_established')
        self.assertEqual(classify_ordered([r(0,.2),{'intervals':[],'alpha':.01}])['status'],'model_incompatible')
        with self.assertRaises(ValueError): classify_ordered([{'intervals':[[0,1]],'alpha':.1}]*3)
        self.assertEqual(binomial_interval(0,10)[0],0)
        self.assertEqual(binomial_interval(10,10)[1],1)

    def test_analytic_modal_regions_and_ties(self):
        for n in (2,4,6,8):
            regions=ideal_modal_regions(n,TIME)
            for region in regions['open_regions']:
                for fraction in (.1,.5,.9):
                    h=region['lo']+(region['hi']-region['lo'])*fraction
                    modes=[int(np.argmax(Experiment(n,TIME,BOUNDS).probabilities(1+s*h))) for s in (-2,0,2)]
                    self.assertEqual(region['distinct'],len(set(modes))==3)
        self.assertEqual(nearest_bins(-.9375,TIME,2),[0,1])

    def test_no_information_and_fail_closed_budget(self):
        ex=Experiment(2,TIME,BOUNDS,1)
        r=Inference(ex,[2,3,4,5],[5,4,3,2]).summary()
        self.assertEqual(r['intervals'],[list(BOUNDS)])
        with self.assertRaises(RuntimeError):
            Inference(Experiment(2,TIME,BOUNDS),[2,3,4,5],[5,4,3,2]).confidence_set(max_cells=1)


if __name__=='__main__':unittest.main(verbosity=2)
