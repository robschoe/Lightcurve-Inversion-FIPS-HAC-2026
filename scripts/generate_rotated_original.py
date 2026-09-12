from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatedata import cylinder_radius_about_z, rotate_and_normalize_height
from lightcurve_fips.training.utils import load_as_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

BATCH_ID = 128
N_SAMPLES = 100
SEED = 42

def main():
    """Generate randomly rotated simplified variants of one asteroid mesh."""
    
    rng = np.random.default_rng(SEED)

    batch_dir = DATASET_DIR / f"batch{BATCH_ID}"
    batch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = load_as_mesh(PROJECT_ROOT / "asteroid3_scaled_radius0.8782467278262304.stl")

    print(f"Source faces: {len(mesh.faces)}")

    target_faces = 5000

    mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

    print(f"Simplified faces: {len(mesh.faces)}")

    if not mesh.is_watertight:
        print("Warning: The simplified source mesh is not watertight.")

    for sample_index in range(N_SAMPLES):
        sample_dir = batch_dir / f"sample{sample_index}"
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )        

        mesh = rotate_and_normalize_height(
            mesh,
            rng,
        )

        #Measure final radius.
        radius = cylinder_radius_about_z(mesh)

        #Check z values.
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        assert np.isclose(
            z_min,
            -1.0,
            atol=1e-6,
        ), f"z_min is not -1, but {z_min}"

        assert np.isclose(
            z_max,
            1.0,
            atol=1e-6,
        ), f"z_max is not +1, but {z_max}"

        if not mesh.is_watertight:
            print(
                f"Warning: Mesh in {sample_dir} is not watertight."
            )

        stl_name = f"asteroid_radius{radius:.12f}.stl"
        stl_path = sample_dir / stl_name

        mesh.export(stl_path)

        print(
            f"{sample_dir.name}: "
            f"radius={radius:.12f}, "
            f"z=[{z_min:.6f}, {z_max:.6f}], "
            f"watertight={mesh.is_watertight}"
        )

    print(
        f"\nFinished: generated {N_SAMPLES} asteroid variants "
        f"in {batch_dir}"
    )


if __name__ == "__main__":
    main()