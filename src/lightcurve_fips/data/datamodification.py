import trimesh
import numpy as np
from pathlib import Path

def normalize_mesh_height_and_get_radius(stl_in, stl_out=None):
    mesh = trimesh.load(stl_in)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    mesh = mesh.copy()

    verts = mesh.vertices  # (N,3)

    z_min = verts[:,2].min()
    z_max = verts[:,2].max()
    if z_max == z_min:
        raise ValueError("Mesh has zero z-extent; cannot normalize height.")

    scale = 2.0 / (z_max - z_min)

    z_mid = 0.5 * (z_min + z_max)

    verts_scaled = (verts - np.array([0.0, 0.0, z_mid])) * scale

    verts_scaled[:,2] = np.clip(verts_scaled[:,2], -1.0, 1.0)
    mesh_scaled = trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=True)

    radii = np.sqrt(verts_scaled[:,0]**2 + verts_scaled[:,1]**2)
    R = float(radii.max())

    if stl_out is not None:
        mesh_scaled.export(stl_out)

    return mesh_scaled, R