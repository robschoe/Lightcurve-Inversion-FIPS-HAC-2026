import numpy as np
from scipy.spatial import cKDTree
import torch
from lightcurve_fips.training.utils import sdf_to_stl,parse_radius_from_stl
from pathlib import Path
import re
from lightcurve_fips.evaluation.evaluate import (make_watertight_with_pymeshfix,compare_stl_files)
from lightcurve_fips.data.generatedata import sample_sdf_from_mesh
import shutil

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
            k=k_neighbors
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

        points = np.load(points_path)
        sdf = np.load(sdf_path)

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

found=0

voxel_goal=0.98
for batch in range(102):
    for sample in range(1000):
        n_points=8192
        n_sets=4
        tau=0.1
        voxel_score=0
        count=0
        while voxel_score<=voxel_goal:
            sample_folder=PROJECT_ROOT / f"data/dataset2/batch{batch+18}/sample{sample}"

            if sample_folder.exists()==False:
                break

            all_points, all_sdf = load_all_sdf_sets(
                sample_folder,
                n_points=n_points
            )

            stl_path = sorted(
                sample_folder.glob("asteroid*.stl")
            )[0]

            radius=parse_radius_from_stl(stl_path)


            sdf_grid = sdf_samples_to_grid(
                points=all_points,
                sdf_values=all_sdf,
                resolution=64,
                k_neighbors=8
            )
            
            recon_stl=PROJECT_ROOT/"sdf_samples_reconstruction.stl"

            mesh = sdf_to_stl(
                sdf=sdf_grid,
                out_path=recon_stl,
                radius=radius
            )

            vox, side = compare_stl_files(
                stl_path,
                recon_stl,
                
                voxel_resolution=64,
                projection_resolution=128,
                per_axis_normalization=False
            )

            voxel_score=vox.get("voxel_score")
            side_score=side["mean_boundary_similarity"]
            print("side view score: ",side_score, "Voxelscore:",voxel_score)
            if voxel_score<=voxel_goal:
                for k in range(n_sets):
                    points_path = sample_folder/ f"points_{n_points}_{k}.npy"
                    sdf_path = sample_folder / f"sdf_{n_points}_{k}.npy"

                    #if points_path.exists() and sdf_path.exists():
                    #    continue

                    points, sdf = sample_sdf_from_mesh(
                        stl_path,
                        radius,
                        n_points=n_points,
                        tau=tau
                    )

                    np.save(points_path, points)
                    np.save(sdf_path, sdf)
            count+=1
            if count==10:
                found+=1
                shutil.rmtree(sample_folder) 
                break