"""Rebuild every article/supplement figure without editing supplied evidence."""
from pathlib import Path
import argparse,json,subprocess,sys,shutil,os
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=root/'reproductions/figures');a=ap.parse_args()
out=a.output.resolve()
if out.exists():raise SystemExit('Choose a new output directory to preserve previous renders.')
if out==root or out in root.parents or root/'src'==out or (root/'src') in out.parents or (root/'figures')==out or (root/'figures') in out.parents:raise SystemExit('Choose an output outside the supplied code/data/figure folders.')
assets=out/'assets';assets.mkdir(parents=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name in ['inference','physical','noise_reference','comparators','budget','material','compact','clarity']:
    subprocess.run([sys.executable,'-B',str(root/f'scripts/renderers/{name}.py'),str(assets)],env=env,check=True)
index=json.loads((root/'figures/index.json').read_text())
for row in index:
    for ext in ['pdf','svg','png']:
        source=assets/(row['asset']+'.'+ext);assert source.is_file(),source
        target=out/row['section']/(row['figure']+'.'+ext);target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
print('Rendered',len(index),'numbered figures in PDF, SVG and PNG:',out)
