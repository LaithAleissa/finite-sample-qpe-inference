"""Exact dimer circuits. Energies use Pauli coefficients and hbar=1."""
from dataclasses import dataclass
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

VERSION = 'quantum_model-v1.0.0'
STATES = ('singlet', 'triplet_plus', 'triplet_00', 'triplet_11')
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])

@dataclass(frozen=True)
class Model:
    jxy: float = 1.0
    jz: float = 1.0
    h: float = 0.0
    offset: float = 0.0
    axis: str = 'z'

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.jxy, self.jz, self.h, self.offset)):
            raise ValueError('All coefficients must be finite')
        if self.axis not in ('x', 'z'):
            raise ValueError('axis must be x or z')
        if self.axis == 'x' and self.jxy != self.jz:
            raise ValueError('Transverse field requires isotropic exchange in this exact factorization')

    def matrix(self):
        p = X if self.axis == 'x' else Z
        return (self.jxy * (np.kron(X, X) + np.kron(Y, Y))
                + self.jz * np.kron(Z, Z) + self.h * (np.kron(I, p) + np.kron(p, I))
                + self.offset * np.eye(4))

    def energies(self):
        return dict(zip(STATES, (self.offset-self.jz-2*self.jxy,
                    self.offset-self.jz+2*self.jxy,
                    self.offset+self.jz+2*self.h, self.offset+self.jz-2*self.h)))

    def safe_time(self, safety=0.8):
        if not 0 < safety < 1:
            raise ValueError('safety must be strictly between 0 and 1')
        bound = max(abs(e) for e in self.energies().values())
        return safety*np.pi/bound if bound else 1.0

def _step(qc, control, targets, model, time):
    a, b = targets
    # Parity gadgets are identity on the control-zero branch.
    for axis, coefficient in [('z', model.jz), ('x', model.jxy), ('y', model.jxy)]:
        if axis == 'x':
            qc.h(a); qc.h(b)
        elif axis == 'y':
            qc.rx(np.pi/2, a); qc.rx(np.pi/2, b)
        qc.cx(a, b); qc.crz(2*coefficient*time, control, b); qc.cx(a, b)
        if axis == 'x':
            qc.h(a); qc.h(b)
        elif axis == 'y':
            qc.rx(-np.pi/2, a); qc.rx(-np.pi/2, b)
    if model.axis == 'x':
        qc.h(a); qc.h(b)
    qc.crz(2*model.h*time, control, a)
    qc.crz(2*model.h*time, control, b)
    if model.axis == 'x':
        qc.h(a); qc.h(b)
    # An identity energy shift becomes a RELATIVE phase on the control.
    qc.p(-model.offset*time, control)

def controlled_evolution(model, time, power=1, method='scaled'):
    if not isinstance(power, int) or isinstance(power, bool) or power < 0:
        raise ValueError('power must be a nonnegative integer')
    if not math.isfinite(time):
        raise ValueError('time must be finite')
    if method not in ('scaled', 'repeated'):
        raise ValueError('method must be scaled or repeated')
    qc = QuantumCircuit(3)
    if power:
        for _ in range(power if method == 'repeated' else 1):
            _step(qc, 0, (1, 2), model, time if method == 'repeated' else time*power)
    return qc

def prepare_state(qc, targets, state, axis='z'):
    a, b = targets
    if state == 'singlet':
        qc.x(a); qc.h(a); qc.cx(a, b); qc.x(b)
    elif state == 'triplet_plus':
        qc.h(a); qc.cx(a, b); qc.x(b)
    elif state == 'triplet_11':
        qc.x(a); qc.x(b)
    elif state != 'triplet_00':
        raise ValueError('Unknown eigenstate')
    if axis == 'x':
        qc.h(a); qc.h(b)

def qpe(model, state, n, time=None, method='scaled', measure=False):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('n must be a positive integer')
    time = model.safe_time() if time is None else time
    if not math.isfinite(time) or time <= 0:
        raise ValueError('QPE time must be positive and finite')
    if max(abs(e*time) for e in model.energies().values()) >= np.pi:
        raise ValueError('Spectrum aliases outside the signed phase branch')
    qc = QuantumCircuit(n+2, n if measure else 0)
    prepare_state(qc, (n, n+1), state, model.axis)
    for k in range(n):
        qc.h(k)
        qc.compose(controlled_evolution(model, time, 2**k, method), [k, n, n+1], inplace=True)
    qc.append(QFTGate(n).inverse(), range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc

def probabilities(model, state, n, time=None, method='scaled'):
    return Statevector.from_instruction(qpe(model, state, n, time, method)).probabilities(list(range(n)))

def ideal_probabilities(energy, time, n):
    """Independent finite geometric-sum QPE distribution (no circuit calls)."""
    size = 2**n
    delta = -energy*time/(2*np.pi) - np.arange(size)/size
    amplitudes = np.exp(2j*np.pi*np.outer(delta, np.arange(size))).mean(axis=1)
    return abs(amplitudes)**2

def energy_from_bin(index, n, time):
    phase = index/2**n
    signed = phase if phase < 0.5 else phase-1
    return -2*np.pi*signed/time

def literature_anchors():
    """Positive-D, axial approximation to the manuscript's D Sz^2 convention."""
    base, d = 315.0/4, 0.34
    return {
        'bleaney_bowers_axial': Model(base, base+d/2, offset=d/2),
        'bofill_axial': Model((297.793+297.753)/8, 298.453/4),
    }
