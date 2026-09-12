import numpy as np
import trimesh
from pathlib import Path
from numpy import random
from pathlib import Path

import numpy as np
import trimesh

from scipy.spatial.transform import Rotation
from skimage import measure
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatepolygons import add_noise_and_craters
from lightcurve_fips.data.generatedata import cylinder_radius_about_z, rotate_and_normalize_height
import os

"""Simple Script to add craters to already existing data."""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng()

#Original batches are modified and then saved as the new batch.
ORIGINAL_BATCH_NR=1
NEW_BATCH_NR=100

for batch in range(10):
    root = DATASET_DIR/f"batch{ORIGINAL_BATCH_NR+batch}"
    root.mkdir(exist_ok=True)

    for g in range(1000):
        new_sample_path = DATASET_DIR/f"batch{NEW_BATCH_NR+batch}/sample{g}"
        new_sample_path.mkdir(parents=True,exist_ok=True)

        #Locate original sample directory.
        original_sample_path=DATASET_DIR/f"batch{ORIGINAL_BATCH_NR+batch}/sample{g}"

        #Find and load the original stl object.
        files = list(original_sample_path.rglob("asteroid*.stl"))
        if not files:
            print(
                f"Warning: No asteroid STL found in "
                f"{original_sample_path}. Skipping."
            )
            continue
        m = trimesh.load(files[0], force="mesh")

        #Add craters and noise.
        mesh = add_noise_and_craters(m, noise_amp=0.03, noise_scale=0.1, n_craters=500, crater_max_radius=0.25, crater_max_depth=0.15)

        mesh = rotate_and_normalize_height(mesh, rng)

        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        radius = cylinder_radius_about_z(mesh)

        #Check for correct z values.
        if not np.isclose(z_min, -1.0, atol=1e-5):
            raise RuntimeError(f"Incorrect z_min: {z_min}")

        if not np.isclose(z_max, 1.0, atol=1e-5):
            raise RuntimeError(f"Incorrect z_max: {z_max}")

        if not mesh.is_watertight:
            print(f"Warning: sample{g} is not watertight.")

        stl_path = new_sample_path / (f"asteroid_radius{radius:.12f}.stl")

        mesh.export(stl_path)

        print(
            f"{new_sample_path.name} | "
            f"radius={radius:.8f} | "
            f"faces={len(mesh.faces)} | "
            f"watertight={mesh.is_watertight}"
        )