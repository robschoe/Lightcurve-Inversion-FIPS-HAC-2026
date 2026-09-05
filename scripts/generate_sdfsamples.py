from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import parse_radius_from_stl
from lightcurve_fips.data.generatedata import (sample_multiple_sdf_sets_from_loaded_mesh, load_normalized_mesh)


n_points=8192
n_sets=4
tau=0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for x in range(12):
    root = DATASET_DIR/f"batch{102+x}"

    samples = sorted([
        p for p in root.rglob("sample*")
        if list(p.glob("asteroid*.stl"))
    ])

    print("Found samples:", len(samples))

    for folder in tqdm(samples):
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]

        missing_sets = []

        for k in range(n_sets):
            points_path = folder / f"points_{n_points}_{k}.npy"
            sdf_path = folder / f"sdf_{n_points}_{k}.npy"

            if not (points_path.exists() and sdf_path.exists()):
                missing_sets.append(k)

        if not missing_sets:
            continue

        radius = parse_radius_from_stl(stl_path)

        mesh = load_normalized_mesh(stl_path, radius)

        TARGET_SDF_FACES = 2000

        if len(mesh.faces) > TARGET_SDF_FACES:
            mesh = mesh.simplify_quadric_decimation(
                face_count=TARGET_SDF_FACES
            )

            trimesh.repair.fix_normals(
                mesh,
                multibody=True
            )

        query = trimesh.proximity.ProximityQuery(mesh)

        all_points, all_sdf = sample_multiple_sdf_sets_from_loaded_mesh(
            mesh,
            n_sets=len(missing_sets),
            n_points=n_points,
            tau=tau,
            query=query
        )

        for local_index, k in enumerate(missing_sets):
            points_path = folder / f"points_{n_points}_{k}.npy"
            sdf_path = folder / f"sdf_{n_points}_{k}.npy"

            np.save(points_path, all_points[local_index])
            np.save(sdf_path, all_sdf[local_index])