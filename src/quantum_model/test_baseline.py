"""Deterministic independent numerical oracles; no phase-insensitive equivalence."""
import unittest
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from dimer import Model, STATES, controlled_evolution, prepare_state, probabilities, ideal_probabilities, qpe, literature_anchors

class BaselineTests(unittest.TestCase):
    def test_random_controlled_unitaries(self):
        rng = np.random.default_rng(20260909)
        worst = 0.0
        for case in range(48):
            j, jz, h, c, t = rng.uniform(-3, 3, 5)
            axis = 'x' if case % 3 == 0 else 'z'
            model = Model(j, j if axis == 'x' else jz, h, c, axis)
            power = [0, 1, 2, 7, 16, 32][case % 6]
            expected = np.eye(8, dtype=complex)
            expected[1::2, 1::2] = expm(-1j*model.matrix()*t*power)
            for method in ('scaled', 'repeated'):
                actual = Operator(controlled_evolution(model, t, power, method)).data
                worst = max(worst, float(np.max(abs(actual-expected))))
                np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
        print(f'48 random cases x 2 constructions: max element error {worst:.3e}')

    def test_prepared_states_and_spectra(self):
        for model in (Model(1.7, 1.7, .31, .2, 'x'), Model(-.8, 2.1, -.4, -.3)):
            np.testing.assert_allclose(np.sort(list(model.energies().values())), np.linalg.eigvalsh(model.matrix()), atol=1e-12)
            for state, energy in model.energies().items():
                qc = QuantumCircuit(2)
                prepare_state(qc, (0, 1), state, model.axis)
                v = Statevector.from_instruction(qc).data
                np.testing.assert_allclose(model.matrix()@v, energy*v, atol=1e-12)

    def test_full_qpe_distributions(self):
        for model in (Model(), Model(1.3, 1.3, -.2, .27, 'x'), Model(-.7, 1.8, .3, -.41)):
            for n in (2, 4, 6):
                for state, energy in model.energies().items():
                    actual = probabilities(model, state, n)
                    expected = ideal_probabilities(energy, model.safe_time(), n)
                    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)

    def test_transpilation_and_power_agreement(self):
        model = Model(1.2, 2.1, .21, .17)
        qc = qpe(model, 'singlet', 3)
        compiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=20260909)
        np.testing.assert_allclose(Operator(qc).data, Operator(compiled).data, atol=2e-12)
        np.testing.assert_allclose(probabilities(model, 'triplet_00', 4), probabilities(model, 'triplet_00', 4, method='repeated'), atol=2e-12)

    def test_identity_shift_is_observable(self):
        from dimer import Z, I
        model = literature_anchors()['bleaney_bowers_axial']
        sz = (np.kron(Z, I)+np.kron(I, Z))/2
        exact = Model(78.75, 78.75).matrix()+.34*(sz@sz)
        np.testing.assert_allclose(model.matrix(), exact, atol=1e-12)
        self.assertGreater(np.max(abs(probabilities(Model(offset=.4), 'singlet', 4, .5)-probabilities(Model(), 'singlet', 4, .5))), .1)
        self.assertAlmostEqual(model.jz/model.jxy, 1.00215873015873)
        self.assertAlmostEqual(literature_anchors()['bofill_axial'].jz/literature_anchors()['bofill_axial'].jxy, 1.0022836187297, places=10)

    def test_guards_and_zero_model(self):
        from dimer import energy_from_bin
        for kwargs in ({'axis':'y'}, {'jxy':1, 'jz':2, 'axis':'x'}, {'h':float('nan')}):
            with self.assertRaises(ValueError): Model(**kwargs)
        with self.assertRaises(ValueError): qpe(Model(), 'singlet', 4, 2)
        with self.assertRaises(ValueError): controlled_evolution(Model(), 1, -1)
        np.testing.assert_allclose(probabilities(Model(0, 0), 'singlet', 3), [1,0,0,0,0,0,0,0], atol=1e-12)
        # Even a non-aliased true phase can round onto the signed branch endpoint.
        self.assertEqual(int(np.argmax(probabilities(Model(), 'singlet', 2))), 2)
        self.assertGreater(energy_from_bin(2, 2, Model().safe_time()), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
