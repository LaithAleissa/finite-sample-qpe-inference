from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'src/qpe_study'
OUT=ROOT/'results'
TARGET=Path(sys.argv[1]).resolve()
TARGET.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def savefig(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(TARGET/f'{name}.{ext}',bbox_inches='tight',dpi=210)
    plt.close(fig)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

sys.path.insert(0,str(REPO/'src/manuscript/code'))
import figures as material
material.OUT=TARGET
for fn in [material.schematic,material.resources,material.historical,material.sensitivity,material.components]:fn()
