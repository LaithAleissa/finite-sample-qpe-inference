"""Integrity check, scientific unit tests, or independent final-study audit."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,os
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['hashes','tests','audit']);a=ap.parse_args()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
if a.mode=='hashes':
    files=json.loads((root/'manifest.json').read_text())['files']
    for name,expected in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    print('Verified',len(files),'file hashes.')
elif a.mode=='tests':
    for folder in ['quantum_model','known_noise','material_model','likelihood_baseline/code','predictive_inference/code','continuous_inversion/code','qpe_study/code']:
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(root/'src'/folder),'-p','test*.py'],env=env,check=True)
else:
    subprocess.run([sys.executable,'-B',str(root/'src/qpe_study/code/audit_final_scientific.py'),'--read-only'],env=env,check=True)
