from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import parse_radius_from_stl
from lightcurve_fips.data.generatedata import (sample_multiple_sdf_sets_from_loaded_mesh, load_normalized_mesh)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for x in range(60):
    root = DATASET_DIR/f"batch{43+x}"

    samples = sorted([
            p for p in root.rglob("sample*")
            if list(p.glob("asteroid*.stl"))
        ])

    for folder in tqdm(samples):
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]
        radius = parse_radius_from_stl(stl_path)
        mesh = trimesh.load(stl_path, force="mesh")
        
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                list(mesh.geometry.values())
            )

        R = np.max(np.sqrt(mesh.vertices[:,0]**2 + mesh.vertices[:,1]**2))
        if R!=radius:
            print(f"Falscher Radius bei: {folder}")