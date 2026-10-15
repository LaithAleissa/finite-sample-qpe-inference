import unittest
import numpy as np
from physical_bridge import af_intersection, response_image, SCALE, GPOW, TEMPERATURES
from physics import bleaney_bowers
from inference import difference_set

class PhysicalBridgeTests(unittest.TestCase):
    def test_disconnected_difference_and_declared_prior(self):
        gaps=difference_set([[.9,1.1]], [[-3.1,-2.9],[3.4,3.6]])
        self.assertEqual(len(gaps),2)
        af=af_intersection(gaps)
        self.assertEqual(len(af),1)
        self.assertAlmostEqual(af[0][0],3.8)
        self.assertAlmostEqual(af[0][1],4.2)

    def test_full_af_response_and_empty_outcomes(self):
        self.assertEqual(response_image([])['lower'],[])
        image=response_image([[0,7.5]])
        for gap in (0.,.01,4.,7.5):
            curve=bleaney_bowers(TEMPERATURES,gap*SCALE,GPOW)
            self.assertTrue(np.all(curve>=image['lower']))
            self.assertTrue(np.all(curve<=image['upper']))

    def test_response_union_keeps_components(self):
        image=response_image([[1.,2.],[4.,5.]])
        self.assertEqual(len(image['interval_images']),2)
        curve=bleaney_bowers(TEMPERATURES,4.5*SCALE,GPOW)
        self.assertTrue(np.all(curve>=image['lower']))
        self.assertTrue(np.all(curve<=image['upper']))

    def test_plot_hull_does_not_fill_union_for_scoring(self):
        image=response_image([[1.,2.],[5.,6.]])
        curve=bleaney_bowers(TEMPERATURES,4*SCALE,GPOW)
        self.assertTrue(np.all(curve>=image['lower']))
        self.assertTrue(np.all(curve<=image['upper']))
        for j,value in enumerate(curve):
            self.assertFalse(any(im['lower'][j]<=value<=im['upper'][j] for im in image['interval_images']))

if __name__=='__main__': unittest.main()
