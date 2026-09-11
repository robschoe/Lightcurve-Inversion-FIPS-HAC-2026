import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import parse_radius_from_stl
from lightcurve_fips.data.generatedata import (sample_multiple_sdf_sets_from_loaded_mesh2, load_normalized_mesh)

def generate_sdf_for_sample(task):
    folder, missing_sets, n_points, tau = task

    try:
        repaired_stls = sorted(
            folder.glob("asteroid*_repaired.stl")
        )

        original_stls = sorted([
            path
            for path in folder.glob("asteroid*.stl")
            if not path.stem.endswith("_repaired")
        ])

        if repaired_stls:
            stl_path = repaired_stls[0]
        elif original_stls:
            stl_path = original_stls[0]
        else:
            return {"status": "skipped", "folder": str(folder), "reason": "Keine STL-Datei gefunden"}

        radius = parse_radius_from_stl(stl_path)

        # Erwartete kanonische Normierung:
        # x_norm = x / radius
        # y_norm = y / radius
        # z_norm = z
        mesh = load_normalized_mesh(stl_path, radius)

        if mesh.is_empty:
            return {
                "status": "skipped",
                "folder": str(folder),
                "reason": "Mesh ist leer",
            }

        if not mesh.is_watertight:
            return {
                "status": "skipped",
                "folder": str(folder),
                "reason": "Mesh ist nicht watertight",
            }

        query = trimesh.proximity.ProximityQuery(mesh)

        all_points, all_sdf = sample_multiple_sdf_sets_from_loaded_mesh2(
            mesh=mesh,
            n_sets=len(missing_sets),
            n_points=n_points,
            tau=tau,
            query=query,
        )

        for local_index, k in enumerate(missing_sets):
            points_path = folder / f"points_{n_points}_{k}.npy"
            sdf_path = folder / f"sdf_{n_points}_{k}.npy"

            # Temporäre Namen verhindern halbfertige Dateien,
            # falls ein Prozess während des Schreibens abstürzt.
            points_tmp = folder / f".points_{n_points}_{k}.tmp.npy"
            sdf_tmp = folder / f".sdf_{n_points}_{k}.tmp.npy"

            np.save(points_tmp, all_points[local_index])
            np.save(sdf_tmp, all_sdf[local_index])

            points_tmp.replace(points_path)
            sdf_tmp.replace(sdf_path)

        return {
            "status": "success",
            "folder": str(folder),
            "n_sets": len(missing_sets),
        }

    except Exception as exc:
        return {
            "status": "error",
            "folder": str(folder),
            "reason": repr(exc),
        }

n_points=8192*4
n_sets=4
tau=0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

N_WORKERS = 56

def collect_tasks():
    tasks = []
    for x in range(155):
        root = DATASET_DIR/f"batch{1+x}"

        if not root.exists():
            print(f"Batch existiert nicht: {root}")
            continue
        samples = sorted([
            p for p in root.rglob("sample*")
            if p.is_dir() and list(p.glob("asteroid*.stl"))
        ])

        print("Found samples:", len(samples))

        for folder in samples:
            missing_sets = []

            for k in range(n_sets):
                points_path = folder / f"points_{n_points}_{k}.npy"
                sdf_path = folder / f"sdf_{n_points}_{k}.npy"

                if not (points_path.exists() and sdf_path.exists()):
                    missing_sets.append(k)

            if missing_sets:
                tasks.append(
                    (
                        folder,
                        missing_sets,
                        n_points,
                        tau,
                    )
                )
    return tasks

if __name__ == "__main__":
    tasks = collect_tasks()

    print(f"Samples mit fehlenden SDF-Sets: {len(tasks)}")
    print(f"Verwendete Worker: {N_WORKERS}")

    n_success = 0
    n_skipped = 0
    n_errors = 0

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
    ) as executor:
        try:
            results = executor.map(
                generate_sdf_for_sample,
                tasks,
                chunksize=1,
            )

            for result in tqdm(
                results,
                total=len(tasks),
                desc="Generating SDF samples",
            ):
                status = result["status"]

                if status == "success":
                    n_success += 1

                elif status == "skipped":
                    n_skipped += 1
                    print(
                        f"\nÜbersprungen: {result['folder']} | "
                        f"{result['reason']}"
                    )

                else:
                    n_errors += 1
                    print(
                        f"\nFehler: {result['folder']} | "
                        f"{result['reason']}"
                    )

        except KeyboardInterrupt:
            print("\nAbbruch erkannt – Worker werden beendet ...")

            # Achtung: _processes ist ein internes Attribut von Python.
            # Für kontrollierte Skripte ist dies dennoch oft praktikabel.
            for process in executor._processes.values():
                process.terminate()

            for process in executor._processes.values():
                process.join(timeout=5)

                # Falls terminate() nicht genügt:
                if process.is_alive():
                    print(f"Worker {process.pid} reagiert nicht, wird gekillt.")
                    process.kill()
                    process.join()

            raise

        finally:
            # Wartende, noch nicht gestartete Aufgaben verwerfen
            executor.shutdown(wait=False, cancel_futures=True)

    print("\nFertig.")
    print(f"Erfolgreich: {n_success}")
    print(f"Übersprungen: {n_skipped}")
    print(f"Fehler:      {n_errors}")