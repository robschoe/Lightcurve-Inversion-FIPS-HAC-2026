from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import parse_radius_from_stl
from lightcurve_fips.data.generatedata import sample_sdf_from_mesh


n_points=8192
n_sets=4
tau=0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for x in range(10):
    root = DATASET_DIR/f"batch{33+x}"

    samples = sorted([
        p for p in root.rglob("sample*")
        if list(p.glob("asteroid*.stl"))
    ])

    print("Found samples:", len(samples))

    for folder in tqdm(samples):
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]
        radius = parse_radius_from_stl(stl_path)

        for k in range(n_sets):
            points_path = folder / f"points_{n_points}_{k}.npy"
            sdf_path = folder / f"sdf_{n_points}_{k}.npy"

            if points_path.exists() and sdf_path.exists():
                continue

            points, sdf = sample_sdf_from_mesh(
                stl_path,
                radius,
                n_points=n_points,
                tau=tau
            )

            np.save(points_path, points)
            np.save(sdf_path, sdf)