from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
import os
import torch
import re

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