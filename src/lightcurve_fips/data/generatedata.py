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
    mesh = clean_mesh(mesh)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh enthält nach Bereinigung keine gültige Geometrie: {stl_path}")

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Ungültiger Radius {radius} für {stl_path}")

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

    if not np.all(np.isfinite(sdf)):
        invalid = ~np.isfinite(sdf)

        print(f"Warnung: {invalid.sum()} ungültige SDF-Werte in {stl_path}")

        # Diese Punkte neu sampeln oder zunächst mit tau ersetzen.
        # Besser wäre Neu-Sampling, aber das verhindert kaputte NPY-Dateien.
        sdf[invalid] = tau
        
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

def clean_mesh(mesh, area_epsilon=1e-12):
    """
    Entfernt problematische Vertices und degenerierte Faces.
    """
    mesh = mesh.copy()

    # Nur endliche Vertexkoordinaten behalten
    finite_vertices = np.all(np.isfinite(mesh.vertices), axis=1)

    if not np.all(finite_vertices):
        valid_faces = np.all(finite_vertices[mesh.faces], axis=1)
        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()

    # Doppelte Faces entfernen
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    # Degenerierte Faces entfernen
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()

    # Zusätzlicher Test über die Dreiecksfläche
    face_areas = mesh.area_faces
    valid_faces = np.isfinite(face_areas) & (face_areas > area_epsilon)

    mesh.update_faces(valid_faces)
    mesh.remove_unreferenced_vertices()

    # Nahezu identische Vertices zusammenführen
    try:
        mesh.merge_vertices()
    except Exception:
        pass

    # Nach dem Mergen nochmals problematische Faces entfernen
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())

    mesh.remove_unreferenced_vertices()

    # Normalen konsistent ausrichten
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    return mesh