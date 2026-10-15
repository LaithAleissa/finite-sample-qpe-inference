import unittest
from postreview import paired_table
from adequacy import deviance
from fast_profile import batch_maximum
from methods import *

class PostreviewTests(unittest.TestCase):
    def test_paired_exact_discordances(self):
        items=[dict(conventional_accepted=False,accepted=True)]*27+[dict(conventional_accepted=True,accepted=True)]*167+[dict(conventional_accepted=False,accepted=False)]*6
        r=paired_table(items)
        self.assertEqual(r['table_PLR_rows_bootstrap_columns'],[[6,27],[0,167]])
        self.assertAlmostEqual(r['exact_two_sided_p'],2**-26)
        items.append(dict(conventional_accepted=True,accepted=False))
        self.assertEqual(paired_table(items)['PLR_only'],1)

    def test_zero_counts_saturated_deviance(self):
        c=np.array([0,8,0,0]);self.assertEqual(float(deviance(c,0)[0]),0)
        self.assertAlmostEqual(float(deviance(c,-3)[0]),6)

    def test_regret_representation_and_nesting(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));a=np.array([21,3,4,4]);b=np.array([22,2,3,5])
        q=QPEMixture(ex,a,b);p=ProfileLikelihood(ex,a,b);D=p.maximum-q.log_predictive
        self.assertGreaterEqual(D,-1e-8)
        grid=np.linspace(*ex.bounds,501);ll=profile_bounds(ex.probabilities(grid),a+b)[0]
        direct=ll>=q.threshold;rewritten=2*(p.maximum-ll)<=2*(np.log(20)+D)
        self.assertTrue(np.array_equal(direct,rewritten))
        self.assertTrue(np.all(direct[2*(p.maximum-ll)<=p.cutoff]))

    def test_adequacy_batch_refits_match_scalar_maxima(self):
        ex=Experiment(2,.8377580409572781,(-3.75,3.75));cc=np.array([[10,0,0,0],[8,8,8,8],[1,4,13,2],[0,0,0,12]])
        batch=batch_maximum(ex,cc)
        scalar=np.array([ProfileLikelihood(ex,c//2,c-c//2).maximum for c in cc])
        self.assertTrue(np.allclose(batch,scalar,rtol=0,atol=1e-7))
        self.assertTrue(np.allclose(deviance(cc,batch),deviance(cc,scalar),rtol=0,atol=2e-7))

if __name__=='__main__':unittest.main()
