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

#Generate batches of random convex object meshes.
for h in range(10):
    batch_dir = DATASET_DIR/f"batch{23+h}"
    batch_dir.mkdir(exist_ok=True)

    for g in range (1000):
        sample_dir = batch_dir / f"sample{g}"

        sample_dir.mkdir(parents=True,exist_ok=True)

        #Sample a random radius in the interval [0.1, 6.0).
        R = random.rand() * 5.9 + 0.1

        #With increasing samples, point amount of point cloud increases as well.
        pts = sample_points_in_cylinder(2+(g//10), R)

        stl_path = sample_dir / (f"asteroid_radius{R:.12f}.stl")

        hull = points_to_convex_stl(pts, stl_path)
        
        print("STL saved:", stl_path.absolute())

        print("Hull z bounds:", hull.bounds[:, 2])