import numpy as np
import trimesh
from pathlib import Path
from numpy import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatepolygons import add_noise_and_craters
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for h in range(10):
    root = DATASET_DIR/f"batch{33+h}"
    root.mkdir(exist_ok=True)
    for g in range(1000):
        root1 = DATASET_DIR/f"batch{33+h}/sample{g}"
        root1.mkdir(exist_ok=True)
        root2=DATASET_DIR/f"batch{23+h}/sample{g}"
        files = list(root2.rglob("asteroid*.stl"))
        m = trimesh.load(root2/f"{os.path.basename(files[0])}")
        m2 = add_noise_and_craters(m, noise_amp=0.03, noise_scale=0.1, n_craters=500, crater_max_radius=0.25, crater_max_depth=0.15)
        m2.export(root1/f"{os.path.basename(files[0])}")
        print("folder:"+str(h)+"sample:"+str(g))