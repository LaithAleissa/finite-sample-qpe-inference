"""Run the final CP/scaling/drift computations in a separate evidence tree."""
from pathlib import Path
import argparse,shutil,subprocess,sys,os
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('study',choices=['cp','scaling','drift']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
out=a.output.resolve()
if out.exists():raise SystemExit('Output must be a new directory.')
if out==root or out in root.parents or (root/'src')==out or (root/'src') in out.parents or (root/'figures')==out or (root/'figures') in out.parents:raise SystemExit('Unsafe output directory.')
shutil.copytree(root/'src',out/'src')
program="import sys;sys.path.insert(0,sys.argv.pop(1));import final_scientific_study as s;s.freeze=lambda:None;s.main()"
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
subprocess.run([sys.executable,'-B','-c',program,str(out/'src/qpe_study/code'),a.study,'--workers',str(a.workers)],env=env,check=True)
