from pathlib import Path

import numpy as np
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatedata import generate_shape_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

#Sample 100 random spheres
generate_shape_batch(
    dataset_dir=DATASET_DIR,
    batch_id=1,
    n_samples=100,
    shape_type="sphere",
)

#Sample 100 random ellipsoids
generate_shape_batch(
    dataset_dir=DATASET_DIR,
    batch_id=2,
    n_samples=100,
    shape_type="ellipsoid",
)