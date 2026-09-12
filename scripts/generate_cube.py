from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatedata import cylinder_radius_about_z, create_random_rotated_cube


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

BATCH_ID = 1
N_SAMPLES = 1000

def main():
    """Generate a batch of randomly rotated and height-normalized cubes."""

    rng = np.random.default_rng()

    batch_dir = DATASET_DIR / f"batch{BATCH_ID}"
    batch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for sample_index in range(N_SAMPLES):
        sample_dir = batch_dir / f"sample{sample_index}"
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mesh = create_random_rotated_cube(rng)

        #Measure the radius of the final rotated and normalized cube.
        radius = cylinder_radius_about_z(mesh)

        #Verify the expected coordinate normalization.
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

        assert mesh.is_watertight, (
            f"Cube in {sample_dir} is not watertight."
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
        f"\nFinished: generated {N_SAMPLES} cubes in {batch_dir}"
    )


if __name__ == "__main__":
    main()