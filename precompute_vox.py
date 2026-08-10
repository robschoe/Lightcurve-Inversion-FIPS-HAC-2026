from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
import os
import torch
import re

def stl_to_voxels_boundary(
    stl_path,
    resolution=32,
    R_max=5.313693321295838,
    boundary_value=1.0
):
    mesh = trimesh.load(stl_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    x_min, x_max = -R_max, R_max
    y_min, y_max = -R_max, R_max
    z_min, z_max = -1.0, 1.0

    pitch = 2.0 / resolution

    voxelized = mesh.voxelized(pitch=pitch).fill()
    points = voxelized.points

    ix = ((points[:,0] - x_min) / (x_max - x_min) * resolution).astype(int)
    iy = ((points[:,1] - y_min) / (y_max - y_min) * resolution).astype(int)
    iz = ((points[:,2] - z_min) / (z_max - z_min) * resolution).astype(int)

    valid = (
        (ix >= 0) & (ix < resolution) &
        (iy >= 0) & (iy < resolution) &
        (iz >= 0) & (iz < resolution)
    )

    grid[ix[valid], iy[valid], iz[valid]] = 1.0

    return grid

def signed_distance_chunked(mesh, points, chunk=200000):
    values = []

    for i in range(0, len(points), chunk):
        p = points[i:i+chunk]
        values.append(trimesh.proximity.signed_distance(mesh, p))

    return np.concatenate(values)

def sample_sdf_from_mesh(stl_path, radius, n_points=8192, tau=0.1):
    mesh = trimesh.load(stl_path)                                       #load mesh

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh = mesh.copy()

    mesh.vertices[:,0] /= radius                                        #scale coordinates
    mesh.vertices[:,1] /= radius

    n_surface = int(0.7 * n_points)                                     #70% of generated points near surface
    n_uniform = n_points - n_surface                                    #30% uniformly distributed in [-1,1]³

    surface_points = mesh.sample(n_surface)                             #sample surface points
    near_surface = surface_points + np.random.normal(scale=0.03, size=surface_points.shape)     #add noise
    uniform = np.random.uniform(-1, 1, size=(n_uniform, 3))             #sample uniform points

    points = np.concatenate([near_surface, uniform], axis=0)
    points = np.clip(points, -1, 1)                                     #respect bounding cylinder

    sdf = trimesh.proximity.signed_distance(mesh, points)               #calculate sdf for each point
    sdf = np.clip(sdf, -tau, tau)                                       #truncate since surface is to be learned
    sdf = sdf / tau                                                     #normalize sdf
    return points.astype(np.float32), sdf.astype(np.float32)

def stl_to_sdf_grid(stl_path, radius, resolution=128, tau=0.1):
    mesh = trimesh.load(stl_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    mesh = mesh.copy()

    # z auf [-1,1] normieren
    z_min = mesh.vertices[:, 2].min()
    z_max = mesh.vertices[:, 2].max()
    mesh.vertices[:, 2] = 2 * (mesh.vertices[:, 2] - z_min) / (z_max - z_min) - 1

    # x/y radius-normalisieren
    mesh.vertices[:, 0] /= radius
    mesh.vertices[:, 1] /= radius

    coords = np.linspace(-1, 1, resolution, dtype=np.float32)

    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    points = np.stack(
        [X.ravel(), Y.ravel(), Z.ravel()],
        axis=1
    )

    sdf = signed_distance_chunked(mesh, points)

    # optional: Vorzeichen prüfen!
    # falls innen positiv und du innen negativ möchtest:
    # sdf = -sdf

    sdf = np.clip(sdf, -tau, tau)
    sdf = sdf / tau

    sdf = sdf.reshape(resolution, resolution, resolution).astype(np.float32)

    return sdf

def parse_radius_from_stl(stl_path):
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", stl_path.name)

    if m is None:
        raise ValueError(f"Could not parse radius from {stl_path.name}")

    return float(m.group(1))

n_points=8192
n_sets=4
tau=0.1

for x in range(2):
    root=Path.cwd()
    root = root/f"dataset2/batch{21+x}"

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