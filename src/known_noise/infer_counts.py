"""Infer one eigenenergy from two independent raw-count batches.

Usage: python infer_counts.py input.json output.json
Input must declare n, time, energy_bounds, known_uniform_noise, batch_a, batch_b.
"""
import argparse
import json
from pathlib import Path
from inference import Experiment, Inference, VERSION


def infer_document(data):
    required={'n','time','energy_bounds','known_uniform_noise','batch_a','batch_b'}
    allowed=required|{'alpha','label'}
    if not isinstance(data,dict) or set(data)-allowed or not required <= set(data):
        raise ValueError(f'Expected fields {sorted(required)}; optional alpha, label')
    ex=Experiment(data['n'],data['time'],tuple(data['energy_bounds']),data['known_uniform_noise'])
    result=Inference(ex,data['batch_a'],data['batch_b'],data.get('alpha',.05)).summary()
    result.update(release=VERSION,label=data.get('label',''),n=ex.n,time=ex.time,
        energy_bounds=list(ex.bounds),known_uniform_noise=ex.uniform_noise,
        confidence_interpretation='Frequentist outer confidence set under the declared iid fixed-channel model; intervals are not posterior probabilities.',
        assumptions='Two independent fixed-size batches, one eigenphase, known channel, correct energy branch. Hardware drift and calibration uncertainty are not covered.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=infer_document(json.loads(args.input.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{result["status"]}: {result["intervals"]}')
