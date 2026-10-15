"""Exact density-matrix simulation with generic frozen gate/readout noise."""
from studies import *
sys.path.insert(0,str(SRC/'quantum_model'))
from dimer import Model,qpe
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix,Kraus,Operator,Pauli
import itertools
from scipy.special import rel_entr
from scipy.optimize import minimize_scalar

def depolarizing_channel(p,k):
    mats=[Pauli(''.join(s)).to_matrix() for s in itertools.product('IXYZ',repeat=k)]
    return Kraus([np.sqrt(1-p+p/4**k)*mats[0]]+[np.sqrt(p/4**k)*a for a in mats[1:]])

def noisy_distribution(circuit,p1,p2,readout):
    rho=DensityMatrix.from_label('0'*circuit.num_qubits)
    channels={1:depolarizing_channel(p1,1),2:depolarizing_channel(p2,2)}
    for inst in circuit.data:
        qubits=[circuit.find_bit(q).index for q in inst.qubits]
        rho=rho.evolve(inst.operation,qargs=qubits)
        rho=rho.evolve(channels[len(qubits)],qargs=qubits)
    p=rho.probabilities(list(range(4)))
    m=16;transition=np.array([(1-readout)**(4-(i^j).bit_count())*readout**((i^j).bit_count()) for i in range(m) for j in range(m)]).reshape(m,m)
    observed=p@transition
    assert abs(sum(observed)-1)<1e-10 and min(observed)>-1e-12
    return observed

def best_approximation(observed,ex):
    # Production pooled fit accepts integer counts; use the equivalent continuous
    # expected log likelihood directly here, with full branch multistart.
    grid=np.linspace(*ex.bounds,32*2**ex.n+1)
    values,_,etas=profile_bounds(ex.probabilities(grid),observed)
    peaks=np.flatnonzero((values[1:-1]>=values[:-2])&(values[1:-1]>=values[2:]))+1
    best=(float(values.max()),float(grid[values.argmax()]),float(etas[values.argmax()]))
    for lo,hi in [(grid[0],grid[1]),(grid[-2],grid[-1])]+[(grid[i-1],grid[i+1]) for i in peaks]:
        opt=minimize_scalar(lambda e:-profile_bounds(ex.probabilities(e),observed)[0][0],bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
        v,_,eta=profile_bounds(ex.probabilities(opt.x),observed)
        if v[0]>best[0]:best=(float(v[0]),float(opt.x),float(eta[0]))
    e,eta=best[1:];q=(1-eta)*ex.probabilities(e)+eta/16
    return dict(energy=e,eta=eta,total_variation=float(abs(observed-q).sum()/2),kl=float(rel_entr(observed,q).sum()),observed=observed.tolist(),fitted=q.tolist())

def main():
    out=ROOT/'results/circuit';out.mkdir(parents=True,exist_ok=True)
    circuit=transpile(qpe(Model(),'triplet_plus',4,.8377580409572781),basis_gates=['u','cx'],optimization_level=1,seed_transpiler=731)
    ex=Experiment(4,.8377580409572781,(-3.75,3.75));ideal=noisy_distribution(circuit,0,0,0)
    assert max(abs(ideal-ex.probabilities(1)))<1e-12
    rates=[('low',.0001,.001,.002),('medium',.001,.01,.01),('high',.005,.03,.03)]
    records=[];diagnostics=[]
    for index,(name,p1,p2,flip) in enumerate(rates):
        dist=noisy_distribution(circuit,p1,p2,flip);best=best_approximation(dist,ex)
        best.update(case=name,p1=p1,p2=p2,readout=flip,circuit_depth=circuit.depth(),operations=dict(circuit.count_ops()));diagnostics.append(best)
        rng=np.random.default_rng(np.random.SeedSequence([2026100502,index]))
        for trial in range(200):
            a,b=rng.multinomial(512,dist,size=2)
            records.append(dict(case=name,group='circuit',trial=trial,n=4,energy=1.,eta=None,shots_per_batch=512,mechanism='gate_readout_noise',batch_a=a.tolist(),batch_b=b.tolist()))
    save(out/'raw_counts.json',records);save(out/'distribution_diagnostics.json',diagnostics)
    save(out/'circuit_validation.json',dict(ideal_max_error=float(max(abs(ideal-ex.probabilities(1)))),density_matrix_exact=True,rates='Synthetic generic; no backend calibration'))
    print('circuit distributions generated',[(d['case'],d['total_variation']) for d in diagnostics],flush=True)
if __name__=='__main__':main()
