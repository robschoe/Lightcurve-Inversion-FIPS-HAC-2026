import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import measure
import re
import os
import time
import torch.nn.functional as F

def parse_radius_from_stl(stl_path):
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", stl_path.name)

    if m is None:
        raise ValueError(f"Could not parse radius from {stl_path.name}")

    return float(m.group(1))

def positional_encoding(points, num_freqs=6):
    """
    points: (B, N, 3)
    output: (B, N, 3 + 2*num_freqs*3)
    """
    enc = [points]

    for i in range(num_freqs):
        freq = 2.0 ** i
        enc.append(torch.sin(freq * torch.pi * points))
        enc.append(torch.cos(freq * torch.pi * points))

    return torch.cat(enc, dim=-1)

def reconstruct_sdf(model, lc, radius, res=128, chunk=200000, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # sicherstellen, dass radius Tensor mit Batch-Dimension ist
    if not torch.is_tensor(radius):
        radius = torch.tensor([radius], dtype=torch.float32, device=device)
    else:
        radius = radius.to(device)
        if radius.dim() == 0:
            radius = radius.unsqueeze(0)

    lc = lc.to(device)

    coords = torch.linspace(-1, 1, res, device=device)

    X, Y, Z = torch.meshgrid(
        coords, coords, coords,
        indexing="ij"
    )

    grid_points = torch.stack([X, Y, Z], dim=-1)
    grid_points = grid_points.reshape(-1, 3)

    sdf_values = []

    with torch.no_grad():
        for i in range(0, grid_points.shape[0], chunk):
            points = grid_points[i:i+chunk]

            # shape: (1, chunk, 3)
            points = points.unsqueeze(0)

            sdf = model(lc, points, radius)

            # shape: (1, chunk)
            sdf_values.append(sdf.squeeze(0).cpu())

    sdf_grid = torch.cat(sdf_values, dim=0)
    sdf_grid = sdf_grid.reshape(res, res, res).numpy()

    return sdf_grid

def sdf_to_stl(sdf, out_path, radius):
    res = sdf.shape[0]
    if torch.is_tensor(radius):
        radius = radius.detach().cpu().item()

    print("sdf min:", sdf.min())
    print("sdf max:", sdf.max())

    if not (sdf.min() <= 0 <= sdf.max()):
        raise ValueError("SDF does not cross zero. Surface cannot be extracted.")

    verts, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(2/(res-1), 2/(res-1), 2/(res-1))
    )

    # marching_cubes startet bei Koordinate 0, also nach [-1,1] verschieben
    verts -= 1.0

    # x/y zurückskalieren
    verts[:, 0] *= radius
    verts[:, 1] *= radius

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.export(out_path)

def stl_to_voxels(stl_path, resolution=64, R_max=5.313693321295838):
    mesh = trimesh.load(stl_path)

    # Optional: falls Mesh als Scene geladen wird
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            geom for geom in mesh.geometry.values()
        ])

    # Voxelgrid
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Vertices
    vertices = mesh.vertices.copy()

    # Bounding-Box des globalen Raums
    x_min, x_max = -R_max, R_max
    y_min, y_max = -R_max, R_max
    z_min, z_max = -1.0, 1.0

    # Mesh voxelisieren
    # Pitch orientiert sich an kleinster Zellgröße
    pitch_x = (x_max - x_min) / resolution
    pitch_y = (y_max - y_min) / resolution
    pitch_z = (z_max - z_min) / resolution

    pitch = min(pitch_x, pitch_y, pitch_z)

    voxelized = mesh.voxelized(pitch=pitch).fill()
    points = voxelized.points

    # Punkte in Grid-Indizes umrechnen
    ix = ((points[:,0] - x_min) / (x_max - x_min) * resolution).astype(int)
    iy = ((points[:,1] - y_min) / (y_max - y_min) * resolution).astype(int)
    iz = ((points[:,2] - z_min) / (z_max - z_min) * resolution).astype(int)

    # Nur gültige Punkte
    valid = (
        (ix >= 0) & (ix < resolution) &
        (iy >= 0) & (iy < resolution) &
        (iz >= 0) & (iz < resolution)
    )

    grid[ix[valid], iy[valid], iz[valid]] = 1.0

    return grid

def voxels_to_stl(voxels, out_path, threshold=0.5, R_max=5.313693321295838):
    verts, faces, normals, values = measure.marching_cubes(voxels, level=threshold)

    res = voxels.shape[0]

    # Indexraum -> echter Koordinatenraum
    verts[:,0] = verts[:,0] / (res - 1) * (2 * R_max) - R_max
    verts[:,1] = verts[:,1] / (res - 1) * (2 * R_max) - R_max
    verts[:,2] = verts[:,2] / (res - 1) * 2.0 - 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(out_path)

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

def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)

    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = 1 - (2 * intersection + eps) / (union + eps)

    return dice.mean()

def cylinder_mask(resolution, radius, device):
    batch_size = radius.shape[0]

    xy = torch.linspace(-R_max, R_max, resolution, device=device)
    z = torch.linspace(-1, 1, resolution, device=device)

    x, y, z = torch.meshgrid(xy, xy, z, indexing="ij")

    r_grid = (x**2 + y**2).unsqueeze(0)

    radius = radius.view(batch_size, 1, 1, 1)

    mask = r_grid <= radius**2

    return mask.float()