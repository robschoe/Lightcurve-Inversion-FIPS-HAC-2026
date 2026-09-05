import numpy as np
from scipy.spatial import cKDTree
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import sdf_to_stl,parse_radius_from_stl
import re
from lightcurve_fips.evaluation.evaluate import (make_watertight_with_pymeshfix,compare_stl_files)
from lightcurve_fips.data.generatedata import sample_multiple_sdf_sets_from_loaded_mesh,load_normalized_mesh
import shutil
import trimesh
import time
from concurrent.futures import ProcessPoolExecutor
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def sdf_samples_to_grid(
    points,
    sdf_values,
    resolution=64,
    k_neighbors=8,
    chunk_size=100_000
):
    points = np.asarray(points, dtype=np.float32)
    sdf_values = np.asarray(sdf_values, dtype=np.float32).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points muss Form [N, 3] haben, erhalten: {points.shape}"
        )

    if len(points) != len(sdf_values):
        raise ValueError(
            f"points und sdf_values haben unterschiedliche Länge: "
            f"{len(points)} vs. {len(sdf_values)}"
        )

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(sdf_values)
    )

    points = points[valid]
    sdf_values = sdf_values[valid]

    if len(points) == 0:
        raise ValueError("Keine gültigen SDF-Samples vorhanden.")

    tree = cKDTree(points)

    coordinates = np.linspace(
        -1.0,
        1.0,
        resolution,
        dtype=np.float32
    )

    x, y, z = np.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing="ij"
    )

    grid_points = np.column_stack([
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1)
    ])

    interpolated_sdf = np.empty(
        len(grid_points),
        dtype=np.float32
    )

    k_neighbors = min(k_neighbors, len(points))

    for start in range(0, len(grid_points), chunk_size):
        end = min(start + chunk_size, len(grid_points))

        query_points = grid_points[start:end]

        distances, indices = tree.query(
            query_points,
            k=k_neighbors,
            workers=1
        )

        # Sonderfall: Nur ein Nachbar
        if k_neighbors == 1:
            interpolated_sdf[start:end] = sdf_values[indices]
            continue

        # Falls ein Gitterpunkt exakt einem Sample entspricht:
        exact_match = distances[:, 0] < 1e-10

        weights = 1.0 / np.maximum(
            distances,
            1e-8
        ) ** 2

        neighbor_sdf = sdf_values[indices]

        values = np.sum(
            weights * neighbor_sdf,
            axis=1
        ) / np.sum(
            weights,
            axis=1
        )

        values[exact_match] = neighbor_sdf[
            exact_match,
            0
        ]

        interpolated_sdf[start:end] = values

    sdf_grid = interpolated_sdf.reshape(
        resolution,
        resolution,
        resolution
    )

    return sdf_grid

def number_from_filename(path):
    """
    Extrahiert die letzte Zahl vor .npy.

    sdf_8192_12.npy -> 12
    """
    match = re.search(r"_(\d+)\.npy$", path.name)

    if match is None:
        raise ValueError(
            f"Keine Set-Nummer im Dateinamen gefunden: {path.name}"
        )

    return int(match.group(1))


def load_all_sdf_sets(sample_folder, n_points=8192):
    sample_folder = Path(sample_folder)

    sdf_paths = sorted(
        sample_folder.glob(f"sdf_{n_points}_*.npy"),
        key=number_from_filename
    )

    if not sdf_paths:
        raise FileNotFoundError(
            f"Keine SDF-Dateien gefunden in: {sample_folder}"
        )

    points_list = []
    sdf_list = []

    for sdf_path in sdf_paths:
        set_index = number_from_filename(sdf_path)

        points_path = sample_folder / (
            f"points_{n_points}_{set_index}.npy"
        )

        if not points_path.is_file():
            raise FileNotFoundError(
                f"Passende Punktdatei fehlt:\n{points_path}"
            )

        points = load_npy_checked(points_path)
        sdf = load_npy_checked(sdf_path)

        # Erwartet: points [8192, 3], sdf [8192]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Unerwartete Punkteform in {points_path}: "
                f"{points.shape}"
            )

        if sdf.ndim != 1:
            raise ValueError(
                f"Unerwartete SDF-Form in {sdf_path}: "
                f"{sdf.shape}"
            )

        if len(points) != len(sdf):
            raise ValueError(
                f"Punkte und SDF haben unterschiedliche Länge "
                f"für Set {set_index}: "
                f"{len(points)} vs. {len(sdf)}"
            )

        points_list.append(points.astype(np.float32))
        sdf_list.append(sdf.astype(np.float32))

    # [Anzahl_Sets, 8192, 3] -> [Anzahl_Sets * 8192, 3]
    all_points = np.concatenate(points_list, axis=0)

    # [Anzahl_Sets, 8192] -> [Anzahl_Sets * 8192]
    all_sdf = np.concatenate(sdf_list, axis=0)

    return all_points, all_sdf

def improve_sdf_samples_until_good(
    sample_folder,
    n_points=8192,
    n_sets=4,
    tau=0.1,
    voxel_goal=0.98,
    max_attempts=10,
    screen_resolution=32,
    final_resolution=64
):
    sample_folder = Path(sample_folder)
    voxel_score = float("-inf")
    side_score = float("nan")

    stl_path = sorted(
        sample_folder.glob("asteroid*.stl")
    )[0]

    if not stl_path:
        print(f"Keine STL gefunden: {sample_folder}")
        return False, voxel_score

    radius = parse_radius_from_stl(stl_path)

    mesh = load_normalized_mesh(
        stl_path,
        radius
    )

    query = trimesh.proximity.ProximityQuery(mesh)

    try:
        all_points, all_sdf = load_all_sdf_sets(
            sample_folder,
            n_points=n_points
        )

    except (FileNotFoundError, ValueError) as exc:
        print(
            f"Defekte/fehlende SDF-Samples bei {sample_folder}: {exc}. "
            "Erzeuge neue Samples.",
            flush=True
        )

        all_points, all_sdf = sample_multiple_sdf_sets_from_loaded_mesh(
            mesh=mesh,
            n_sets=n_sets,
            n_points=n_points,
            tau=tau,
            query=query
        )

    all_points = all_points.reshape(-1,n_points,3)

    all_sdf = all_sdf.reshape(-1,n_points)

    all_points = all_points[:n_sets]
    all_sdf = all_sdf[:n_sets]

    temp_stl = sample_folder / "_temporary_reconstruction.stl"

    for attempt in range(max_attempts):
        points_flat = all_points.reshape(-1, 3)
        sdf_flat = all_sdf.reshape(-1)

        sdf_grid_small = sdf_samples_to_grid(
            points=points_flat,
            sdf_values=sdf_flat,
            resolution=screen_resolution,
            k_neighbors=4
        )

        mesh_small = sdf_to_stl(
            sdf=sdf_grid_small,
            out_path=temp_stl,
            radius=radius
        )

        if mesh_small is None:
            rough_score = 0.0
        else:
            vox_small, _ = compare_stl_files(
                stl_path,
                temp_stl,
                voxel_resolution=32,
                projection_resolution=64,
                per_axis_normalization=False
            )

            rough_score = vox_small.get(
                "voxel_score",
                0.0
            )
        print(
            f"{sample_folder.name} | "
            f"{sample_folder.parent.name} | "
            f"Versuch {attempt + 1}/{max_attempts} | "
            f"grober Score: {rough_score:.4f}"
        )

        if rough_score < 0.90:
            all_points, all_sdf = (
                sample_multiple_sdf_sets_from_loaded_mesh(
                    mesh=mesh,
                    n_sets=n_sets,
                    n_points=n_points,
                    tau=tau,
                    query=query
                )
            )

            continue

        sdf_grid_final = sdf_samples_to_grid(
            points=points_flat,
            sdf_values=sdf_flat,
            resolution=final_resolution,
            k_neighbors=8
        )

        mesh_final = sdf_to_stl(
            sdf=sdf_grid_final,
            out_path=temp_stl,
            radius=radius
        )

        if mesh_final is None:
            voxel_score = 0.0
            side_score = 0.0
        else:
            vox, side = compare_stl_files(
                stl_path,
                temp_stl,
                voxel_resolution=64,
                projection_resolution=128,
                per_axis_normalization=False
            )

            voxel_score = vox.get(
                "voxel_score",
                0.0
            )

            side_score = side.get(
                "mean_boundary_similarity",
                0.0
            )

        # print(
        #     f"  Final: voxel={voxel_score:.4f}, "
        #     f"side={side_score:.4f}"
        # )

        if voxel_score >= voxel_goal:
            for k in range(n_sets):
                np.save(
                    sample_folder / f"points_{n_points}_{k}.npy",
                    all_points[k]
                )

                np.save(
                    sample_folder / f"sdf_{n_points}_{k}.npy",
                    all_sdf[k]
                )

            if temp_stl.exists():
                temp_stl.unlink()

            return True, voxel_score

        all_points, all_sdf = (
            sample_multiple_sdf_sets_from_loaded_mesh(
                mesh=mesh,
                n_sets=n_sets,
                n_points=n_points,
                tau=tau,
                query=query
            )
        )

    if temp_stl.exists():
        temp_stl.unlink()

    print(
        f"Kein ausreichender Score nach {max_attempts} Versuchen. "
        f"Lösche Ordner: {sample_folder}"
    )

    shutil.rmtree(sample_folder)

    return False, voxel_score

def process_sample_folder(folder):
    try:
        return improve_sdf_samples_until_good(
            folder,
            n_points=8192,
            n_sets=4,
            tau=0.1,
            voxel_goal=0.98,
            max_attempts=10,
            screen_resolution=32,
            final_resolution=64
        )

    except Exception as exc:
        print(
            f"Fehler bei {folder}: {type(exc).__name__}: {exc}",
            flush=True
        )

        return False, 0.0

def load_npy_checked(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Leere NPY-Datei: {path}")

    try:
        return np.load(path)

    except (EOFError, ValueError, OSError) as exc:
        raise ValueError(
            f"Defekte oder unvollständige NPY-Datei: {path}"
        ) from exc

def get_batch_number(sample_folder):
    match = re.search(
        r"batch(\d+)",
        sample_folder.parent.name
    )

    if match is None:
        raise ValueError(
            f"Keine Batchnummer gefunden: {sample_folder}"
        )

    return int(match.group(1))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

folders = sorted(
    [
        folder
        for folder in PROJECT_ROOT.glob(
            "data/dataset/batch*/sample*"
        )
        if get_batch_number(folder) < 103 and get_batch_number(folder) > 94
    ],
    key=lambda folder: (
        get_batch_number(folder),
        int(folder.name.replace("sample", ""))
    )
)

workers = min(
    32,
    os.cpu_count() or 1
)

with ProcessPoolExecutor(
    max_workers=workers
) as executor:
    for folder, result in zip(
        folders,
        executor.map(process_sample_folder, folders)
    ):
        success, score = result
        #print(folder, success, score)


# found=0
# start = time.time()
# voxel_goal=0.98
# for batch in range(102):
#     for sample in range(1000):
#         end=time.time()
#         print(end-start)
#         start = time.time()
#         sample_folder=PROJECT_ROOT / f"data/dataset/batch{batch+1}/sample{sample}"
#         n_points=8192
#         n_sets=4
#         tau=0.1
#         voxel_score=0
#         count=0
#         all_points, all_sdf = load_all_sdf_sets(
#             sample_folder,
#             n_points=n_points
#         )
#         all_points = all_points.reshape(-1, n_points, 3)
#         all_sdf = all_sdf.reshape(-1, n_points)
#         while voxel_score<=voxel_goal:
#             if sample_folder.exists()==False:
#                 break

#             stl_path = sorted(
#                 sample_folder.glob("asteroid*.stl")
#             )[0]

#             radius=parse_radius_from_stl(stl_path)


#             sdf_grid = sdf_samples_to_grid(
#                 points=all_points.reshape(-1, 3),
#                 sdf_values=all_sdf.reshape(-1),
#                 resolution=64,
#                 k_neighbors=8
#             )
            
#             recon_stl=PROJECT_ROOT/"sdf_samples_reconstruction.stl"

#             mesh = sdf_to_stl(
#                 sdf=sdf_grid,
#                 out_path=recon_stl,
#                 radius=radius
#             )

#             vox, side = compare_stl_files(
#                 stl_path,
#                 recon_stl,
                
#                 voxel_resolution=32,
#                 projection_resolution=32,
#                 per_axis_normalization=False
#             )

#             voxel_score=vox.get("voxel_score")
#             side_score=side["mean_boundary_similarity"]
#             if voxel_score<=voxel_goal:
#                 print("side view score: ",side_score, "Voxelscore:",voxel_score, batch, " ", sample)
#                 mesh = load_normalized_mesh(stl_path, radius)
#                 query = trimesh.proximity.ProximityQuery(mesh)

#                 # Alle vier Sets in einem gemeinsamen SDF-Aufruf erzeugen
#                 new_points_sets, new_sdf_sets = (
#                     sample_multiple_sdf_sets_from_loaded_mesh(
#                         mesh=mesh,
#                         n_sets=n_sets,
#                         n_points=n_points,
#                         tau=tau,
#                         query=query
#                     )
#                 )

#                 for k in range(n_sets):
#                     points_path = sample_folder / f"points_{n_points}_{k}.npy"
#                     sdf_path = sample_folder / f"sdf_{n_points}_{k}.npy"

#                     np.save(points_path, new_points_sets[k])
#                     np.save(sdf_path, new_sdf_sets[k])
#             count+=1
#             if count==10:
#                 found+=1
#                 #shutil.rmtree(sample_folder) 
#                 break
#     print(found)