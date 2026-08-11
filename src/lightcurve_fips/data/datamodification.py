import trimesh
import numpy as np
from pathlib import Path

def normalize_mesh_height_and_get_radius(stl_in, stl_out=None):
    """
    Lädt stl_in, skaliert und verschiebt das Mesh so, dass
    z_min -> -1 und z_max -> +1 (Höhe = 2).
    Skaliert UNIFORM (x,y,z mit gleichem Faktor).
    Gibt (mesh_scaled, R) zurück und speichert optional stl_out.
    R ist der minimal erforderliche Radius des Zylinders D(0,R) x [-1,1].
    """
    mesh = trimesh.load(stl_in)
    if isinstance(mesh, trimesh.Scene):
        # falls Scene, vereinige alle Geometrien zu einem Mesh
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # kopieren, damit Original bleibt
    mesh = mesh.copy()

    verts = mesh.vertices  # (N,3)

    z_min = verts[:,2].min()
    z_max = verts[:,2].max()
    if z_max == z_min:
        raise ValueError("Mesh has zero z-extent; cannot normalize height.")

    # Wir wollen: new_z = (z - z_mid) * scale  mit new_z range [-1,1]
    # scale = 2 / (z_max - z_min)
    scale = 2.0 / (z_max - z_min)

    # Centering: bringe Mittelpunkt z_mid auf 0 nach Skalierung
    z_mid = 0.5 * (z_min + z_max)

    # Transform alle Vertices (uniform scale, dann translate)
    verts_scaled = (verts - np.array([0.0, 0.0, z_mid])) * scale

    # Optional numerische Korrektur: setze die Extremwerte exakt -1 und +1
    # (verhindert kleine Rundungsfehler)
    verts_scaled[:,2] = np.clip(verts_scaled[:,2], -1.0, 1.0)
    # Update mesh
    mesh_scaled = trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=True)

    # berechne Radius R = max sqrt(x^2 + y^2)
    radii = np.sqrt(verts_scaled[:,0]**2 + verts_scaled[:,1]**2)
    R = float(radii.max())

    # speichern falls gewünscht
    if stl_out is not None:
        mesh_scaled.export(stl_out)

    return mesh_scaled, R