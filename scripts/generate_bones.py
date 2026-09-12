from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.generatedata import save_bone_asteroid_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Generate batches with 1000 samples of bone-shaped objects.
for x in range (10):
    save_bone_asteroid_batch(
        dataset_dir=PROJECT_ROOT / "data" / "dataset",
        batch_id=1+x,
        n_samples=1000,
    )