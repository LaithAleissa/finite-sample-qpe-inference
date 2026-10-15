"""Single import bridge to the preserved, tested quantum, inference and material implementations."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent
PROJECT = SRC.parent
for phase in ('quantum_model', 'known_noise', 'material_model'):
    sys.path.insert(0, str(SRC / phase))

from dimer import Model, qpe, probabilities, literature_anchors
from inference import Experiment, Inference, difference_set, classify_ordered, counts_array, binomial_interval
from physics import (CM1_TO_K, CURIE, G, TIP, bleaney_bowers, susceptibility,
                     spin_hamiltonian, kubo_susceptibility, fit_data, data_arrays)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
