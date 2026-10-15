import unittest
import numpy as np
from physics import (bleaney_bowers,susceptibility,kubo_susceptibility,spin_hamiltonian,
                     CURIE,CM1_TO_K,SPIN,G,fit_data)


class PhysicsTests(unittest.TestCase):
    def test_high_low_temperature_and_molar_normalization(self):
        self.assertAlmostEqual(float(bleaney_bowers(1e8,400,2))*1e8/CURIE,1,places=5)
        self.assertEqual(float(bleaney_bowers(.01,400,2)),0)
        self.assertAlmostEqual(float(susceptibility(10,400,2,1)),CURIE/10)
        self.assertAlmostEqual(CM1_TO_K,1.4387768775039338,places=12)

    def test_isotropic_kubo_matches_closed_form(self):
        for j in (0,1,78.75):
            for g in (2.,2.42):
                t=np.array([1.,40.,100.,300.,1000.])
                expected=bleaney_bowers(t,4*j*CM1_TO_K,g)
                for axis in (0,1,2):
                    np.testing.assert_allclose(kubo_susceptibility(spin_hamiltonian(j),t,g,axis),expected,atol=1e-13,rtol=1e-9)

    def test_noncommuting_kubo_against_magnetization_derivative(self):
        h=spin_hamiltonian(2,.7,.12)
        for axis in (0,1,2):
            moment=2.1*SPIN[axis]
            for t in (3.,50.):
                def magnetization(b):
                    e,v=np.linalg.eigh(h-b*moment);p=np.exp(-(e-e.min())/t);p/=p.sum()
                    return np.dot(p,np.diag(v.conj().T@moment@v).real)
                step=1e-4
                finite=CURIE/2*(magnetization(step)-magnetization(-step))/(2*step)
                self.assertAlmostEqual(float(kubo_susceptibility(h,t,2.1,axis)),finite,places=9)

    def test_energy_origin_and_powder_average(self):
        t=np.array([30.,100.,300.]);h=spin_hamiltonian(78.75,.34,.01)
        np.testing.assert_allclose(kubo_susceptibility(h,t),kubo_susceptibility(h+123*np.eye(4),t),atol=1e-14)
        average=(bleaney_bowers(t,450,G[0])+2*bleaney_bowers(t,450,G[1]))/3
        np.testing.assert_allclose(average,bleaney_bowers(t,450,np.sqrt((G[0]**2+2*G[1]**2)/3)),rtol=1e-14)

    def test_gap_monotonicity_and_impurity_limit(self):
        t=np.array([10.,100.,300.])
        self.assertTrue(np.all(bleaney_bowers(t,450,2.2)<bleaney_bowers(t,400,2.2)))
        np.testing.assert_allclose(susceptibility(t,400,2.1,.04,1e-5),.96*bleaney_bowers(t,400,2.1)+.04*CURIE*2.1**2/(4*t)+1e-5)

    def test_synthetic_parameter_recovery(self):
        from physics import TIP
        rows=[]
        for orientation,g,tip in zip(('parallel','perpendicular'),(2.4,2.1),TIP):
            for rank,t in enumerate(np.linspace(40,330,14)):
                chi=susceptibility(t,430,g,.045,tip)
                rows.append(dict(orientation=orientation,temperature_K=t,p2=8*t*chi,rank_in_orientation=rank))
        for weighting in ('relative_chi','equal_chi','equal_p2'):
            result=fit_data(rows,weighting=weighting)
            np.testing.assert_allclose([result[k] for k in ('g_parallel','g_perpendicular','gap_K','monomer_fraction')],[2.4,2.1,430,.045],rtol=1e-7)

    def test_invalid_inputs(self):
        for t in (0,-1,float('nan')):
            with self.assertRaises(ValueError):bleaney_bowers(t,400,2)
        with self.assertRaises(ValueError):susceptibility(100,400,2,1.1)


if __name__=='__main__':unittest.main(verbosity=2)
