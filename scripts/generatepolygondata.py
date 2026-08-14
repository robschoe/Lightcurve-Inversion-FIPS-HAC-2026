import numpy as np
import trimesh
from pathlib import Path
from numpy import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatepolygons import (sample_points_in_cylinder,points_to_convex_stl)
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for h in range(10):
    root = DATASET_DIR/f"batch{23+h}"
    root.mkdir(exist_ok=True)
    for g in range (1000):
        root = DATASET_DIR/f"batch{23+h}/sample{g}"
        root.mkdir(exist_ok=True)

        R = random.rand()*4.9 +0.1
        pts = sample_points_in_cylinder(2+(g//10), R)

        hull = points_to_convex_stl(pts, root / f"asteroid{g}radius{R}.stl")
        print("STL gespeichert:", (root / f"asteroid{g}radius{R}.stl").absolute())
        print("Hull bounds z min/max:", hull.bounds[:,2])