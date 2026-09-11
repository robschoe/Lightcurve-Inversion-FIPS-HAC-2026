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
        
        mesh = load_as_mesh(PROJECT_ROOT / "asteroid3_scaled_radius0.8782467278262304.stl")

        print(f"Faces vorher: {len(mesh.faces)}")

        target_faces = 5000
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

        print(f"Faces nachher: {len(mesh.faces)}")

        mesh = rotate_and_normalize_height(
            mesh,
            rng,
        )

        # Radius des final rotierten UND normierten Würfels.
        radius = cylinder_radius_about_z(mesh)

        # Kontrolle der gewünschten Koordinatenkonvention.
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        assert np.isclose(
            z_min,
            -1.0,
            atol=1e-6,
        ), f"z_min ist nicht -1, sondern {z_min}"

        assert np.isclose(
            z_max,
            1.0,
            atol=1e-6,
        ), f"z_max ist nicht +1, sondern {z_max}"

        assert mesh.is_watertight, (
            f"Würfel in {sample_dir} ist nicht watertight."
        )

        # Keine wissenschaftliche Schreibweise verwenden, falls dein
        # Radius-Regex nur normale Dezimalzahlen unterstützt.
        #
        # 12 Nachkommastellen sind für den Dateinamen ausreichend genau.
        stl_name = f"asteroid_radius{radius:.12f}.stl"
        stl_path = sample_dir / stl_name

        mesh.export(stl_path)

        print(
            f"{sample_dir.name}: "
            f"radius={radius:.12f}, "
            f"z=[{z_min:.6f}, {z_max:.6f}], "
            f"watertight={mesh.is_watertight}"
        )

    print(f"\nFertig: {N_SAMPLES} Würfel erzeugt in {batch_dir}")


if __name__ == "__main__":
    main()