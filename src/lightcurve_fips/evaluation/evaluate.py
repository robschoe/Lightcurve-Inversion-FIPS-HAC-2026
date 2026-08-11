import inspect
import numpy as np
import trimesh
import pymeshfix
from scipy.ndimage import binary_erosion, distance_transform_edt
from pathlib import Path


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    mesh = mesh.copy()

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

    # Unbenutzte Vertices entfernen
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    # Normalen/Konsistenz reparieren, soweit möglich
    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def normalize_mesh_to_unit_box(mesh, per_axis=False):
    """
    Normalisiert ein Mesh in eine gemeinsame Bounding Box.

    per_axis=False:
        Einheitliche Skalierung mit größter Ausdehnung.
        Seitenverhältnisse bleiben erhalten.

    per_axis=True:
        Jede Achse wird separat nach [-0.5, 0.5] skaliert.
        Das Mesh füllt die Box exakt, aber Proportionen können verändert werden.
    """
    mesh = mesh.copy()

    bounds = mesh.bounds
    min_b = bounds[0]
    max_b = bounds[1]

    center = 0.5 * (min_b + max_b)
    extents = max_b - min_b

    mesh.vertices -= center

    if per_axis:
        extents[extents == 0] = 1.0
        mesh.vertices /= extents
    else:
        scale = np.max(extents)
        if scale == 0:
            scale = 1.0
        mesh.vertices /= scale

    return mesh


def contains_points_chunked(mesh, points, chunk_size=200_000):
    """
    Prüft punktweise, ob Punkte innerhalb des Meshes liegen.
    Chunking verhindert zu hohen Speicherverbrauch.
    """
    inside = np.zeros(len(points), dtype=bool)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        inside[start:end] = mesh.contains(points[start:end])

    return inside


def voxelize_by_contains(mesh, resolution=128):
    """
    Erzeugt ein boolesches Voxelgitter.
    True bedeutet: Voxelzentrum liegt innerhalb des Meshes.
    """
    # Voxelzentren in [-0.5, 0.5]^3
    lin = np.linspace(
        -0.5 + 0.5 / resolution,
         0.5 - 0.5 / resolution,
        resolution
    )

    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    inside = contains_points_chunked(mesh, points)

    voxels = inside.reshape((resolution, resolution, resolution))

    return voxels, lin


def voxel_similarity(A, B):
    """
    Voxel-basiertes Maß:

    score = 1 - (#(A\\B) + #(B\\A)) / (#A + #B)

    Das entspricht dem Dice Score:

    score = 2 * #(A ∩ B) / (#A + #B)
    """
    A = A.astype(bool)
    B = B.astype(bool)

    count_A = np.count_nonzero(A)
    count_B = np.count_nonzero(B)

    if count_A + count_B == 0:
        return np.nan

    only_A = np.count_nonzero(A & ~B)
    only_B = np.count_nonzero(B & ~A)
    intersection = np.count_nonzero(A & B)

    score = 1.0 - (only_A + only_B) / (count_A + count_B)

    return {
        "voxel_score": score,
        "dice_score": 2.0 * intersection / (count_A + count_B),
        "count_A": count_A,
        "count_B": count_B,
        "intersection": intersection,
        "A_minus_B": only_A,
        "B_minus_A": only_B
    }


def voxel_centers_from_mask(mask, lin):
    """
    Wandelt ein Voxelgitter in die 3D-Koordinaten der belegten Voxelzentren um.
    """
    idx = np.argwhere(mask)

    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.float32)

    coords = np.column_stack([
        lin[idx[:, 0]],
        lin[idx[:, 1]],
        lin[idx[:, 2]]
    ])

    return coords


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def projection_basis(view_dir):
    """
    Erzeugt zwei orthogonale Achsen u, v senkrecht zur Blickrichtung.
    """
    view_dir = normalize(view_dir)

    up = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(view_dir, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    u = np.cross(view_dir, up)
    u = normalize(u)

    v = np.cross(u, view_dir)
    v = normalize(v)

    return u, v

def voxelize_by_trimesh_fill(mesh, resolution=128):
    """
    Schnellere Voxelisierung über trimesh.voxelized(...).fill().
    Gibt ein boolesches Voxelgitter der Form (resolution, resolution, resolution) zurück.
    Erwartet, dass das Mesh ungefähr in [-0.5, 0.5]^3 liegt.
    """
    pitch = 1.0 / resolution

    # Oberfläche voxelisieren
    voxel_grid = mesh.voxelized(pitch)

    # Innenraum füllen, falls möglich
    try:
        voxel_grid = voxel_grid.fill()
    except Exception as e:
        print("Warnung: voxel_grid.fill() fehlgeschlagen:", e)

    # Zentren belegter Voxel
    points = voxel_grid.points

    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)

    if len(points) == 0:
        lin = np.linspace(
            -0.5 + 0.5 / resolution,
             0.5 - 0.5 / resolution,
            resolution
        )
        return voxels, lin

    # Punkte nach [-0.5, 0.5]^3-Gitter mappen
    idx = np.floor((points + 0.5) * resolution).astype(int)

    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < resolution) &
        (idx[:, 1] >= 0) & (idx[:, 1] < resolution) &
        (idx[:, 2] >= 0) & (idx[:, 2] < resolution)
    )

    idx = idx[valid]

    voxels[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    lin = np.linspace(
        -0.5 + 0.5 / resolution,
         0.5 - 0.5 / resolution,
        resolution
    )

    return voxels, lin

def project_points_to_mask(points_A, points_B, view_dir, image_res=256):
    """
    Projiziert zwei 3D-Punktmengen entlang view_dir in dasselbe 2D-Bild.
    """
    u, v = projection_basis(view_dir)

    def project(points):
        px = points @ u
        py = points @ v
        return np.column_stack([px, py])

    proj_A = project(points_A)
    proj_B = project(points_B)

    all_proj = np.vstack([proj_A, proj_B])

    if len(all_proj) == 0:
        return (
            np.zeros((image_res, image_res), dtype=bool),
            np.zeros((image_res, image_res), dtype=bool)
        )

    min_xy = all_proj.min(axis=0)
    max_xy = all_proj.max(axis=0)

    # kleiner Rand, damit Projektion nicht am Bildrand klebt
    extent = max_xy - min_xy
    extent[extent == 0] = 1.0
    margin = 0.05 * extent

    min_xy -= margin
    max_xy += margin

    def rasterize(proj):
        mask = np.zeros((image_res, image_res), dtype=bool)

        if len(proj) == 0:
            return mask

        xy = (proj - min_xy) / (max_xy - min_xy)
        pix = np.floor(xy * (image_res - 1)).astype(int)
        pix = np.clip(pix, 0, image_res - 1)

        x = pix[:, 0]
        y = pix[:, 1]

        mask[y, x] = True

        return mask

    mask_A = rasterize(proj_A)
    mask_B = rasterize(proj_B)

    return mask_A, mask_B


def boundary_from_mask(mask):
    """
    Extrahiert Randpixel aus einer 2D-Binärmaske.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    boundary = mask & ~eroded

    return boundary


def symmetric_boundary_distance(mask_A, mask_B):
    """
    Symmetrische Chamfer-Distanz zwischen zwei Randkurven.

    Kleine Werte bedeuten hohe Ähnlichkeit.
    """
    boundary_A = boundary_from_mask(mask_A)
    boundary_B = boundary_from_mask(mask_B)

    if not np.any(boundary_A) or not np.any(boundary_B):
        return np.nan

    # Distanz jedes Pixels zum nächsten Randpixel von B
    dist_to_B = distance_transform_edt(~boundary_B)
    dist_to_A = distance_transform_edt(~boundary_A)

    d_A_to_B = dist_to_B[boundary_A].mean()
    d_B_to_A = dist_to_A[boundary_B].mean()

    chamfer = 0.5 * (d_A_to_B + d_B_to_A)

    # Normalisierung auf Bilddiagonale
    diag = np.sqrt(mask_A.shape[0] ** 2 + mask_A.shape[1] ** 2)
    chamfer_norm = chamfer / diag

    return {
        "boundary_distance_px": chamfer,
        "boundary_distance_normalized": chamfer_norm,
        "boundary_similarity": max(0.0, 1.0 - chamfer_norm)
    }


def default_view_directions():
    """
    Beispielhafte Blickrichtungen.
    Du kannst diese Liste beliebig erweitern.
    """
    dirs = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],

        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1],

        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ]

    return [normalize(d) for d in dirs]


def side_view_measure(A, B, lin, image_res=256, view_dirs=None):
    """
    Berechnet ein Side-view-Maß über mehrere Projektionen.
    Grundlage sind die belegten Voxel.
    """
    if view_dirs is None:
        view_dirs = default_view_directions()

    points_A = voxel_centers_from_mask(A, lin)
    points_B = voxel_centers_from_mask(B, lin)

    results = []

    for k, view_dir in enumerate(view_dirs):
        mask_A, mask_B = project_points_to_mask(
            points_A,
            points_B,
            view_dir,
            image_res=image_res
        )

        dist = symmetric_boundary_distance(mask_A, mask_B)

        if isinstance(dist, dict):
            results.append({
                "view_index": k,
                "view_dir": view_dir,
                **dist
            })

    if len(results) == 0:
        return {
            "mean_boundary_distance_px": np.nan,
            "mean_boundary_distance_normalized": np.nan,
            "mean_boundary_similarity": np.nan,
            "views": []
        }

    mean_px = np.nanmean([r["boundary_distance_px"] for r in results])
    mean_norm = np.nanmean([r["boundary_distance_normalized"] for r in results])
    mean_sim = np.nanmean([r["boundary_similarity"] for r in results])

    return {
        "mean_boundary_distance_px": mean_px,
        "mean_boundary_distance_normalized": mean_norm,
        "mean_boundary_similarity": mean_sim,
        "views": results
    }

def _is_verts_array(a):
    if not isinstance(a, (list, tuple, np.ndarray)):
        return False
    arr = np.asarray(a)
    return arr.ndim == 2 and arr.shape[1] in (3, 4) and np.issubdtype(arr.dtype, np.floating)

def _is_faces_array(a):
    if not isinstance(a, (list, tuple, np.ndarray)):
        return False
    arr = np.asarray(a)
    return arr.ndim == 2 and arr.shape[1] in (3, 4) and np.issubdtype(arr.dtype, np.integer)

def _find_vf_in_object(obj):
    # prüfe direkt Attribute
    candidates_v = {}
    candidates_f = {}

    for name in dir(obj):
        if name.startswith('_'):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue

        if _is_verts_array(attr):
            candidates_v[name] = np.asarray(attr)
        if _is_faces_array(attr):
            candidates_f[name] = np.asarray(attr)

        # falls nested (z. B. obj.mesh.vertices)
        if not candidates_v or not candidates_f:
            if hasattr(attr, "__dict__") or isinstance(attr, object):
                for subname in dir(attr):
                    if subname.startswith('_'):
                        continue
                    try:
                        subattr = getattr(attr, subname)
                    except Exception:
                        continue
                    if _is_verts_array(subattr) and subname not in candidates_v:
                        candidates_v[f"{name}.{subname}"] = np.asarray(subattr)
                    if _is_faces_array(subattr) and subname not in candidates_f:
                        candidates_f[f"{name}.{subname}"] = np.asarray(subattr)

    # Wähle plausibelstes Paar: verts mit float dtype und faces mit int dtype
    verts = None
    faces = None

    # prefer typical names
    pref_v_names = ["v", "vertices", "vert", "points"]
    pref_f_names = ["f", "faces", "triangles", "cells"]

    for n in pref_v_names:
        if n in candidates_v:
            verts = candidates_v[n]
            break
    for n in pref_f_names:
        if n in candidates_f:
            faces = candidates_f[n]
            break

    # fallback: take any
    if verts is None and candidates_v:
        verts = next(iter(candidates_v.values()))
    if faces is None and candidates_f:
        faces = next(iter(candidates_f.values()))

    return verts, faces, candidates_v.keys(), candidates_f.keys()

def make_watertight_with_pymeshfix(input_stl, output_stl):
    mesh = trimesh.load(input_stl, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh = mesh.copy()

    print("Before watertight:", mesh.is_watertight)
    print("Before faces:", len(mesh.faces))
    print("Before verts:", len(mesh.vertices))

    mf = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())

    # safe call: versuche mögliche kwargs, fallback auf kein Argument
    sig = inspect.signature(mf.repair)
    candidate_kwargs = {
        "joincomp": True,
        "remove_smallest_components": False,
        "remove_smallest_component": False,
        "preserve_genus": True,
        "verbose": True
    }
    usable_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    try:
        if usable_kwargs:
            mf.repair(**usable_kwargs)
        else:
            mf.repair()
    except TypeError:
        mf.repair()

    # jetzt robust Vertex/Face-Arrays extrahieren
    verts, faces, cand_vs, cand_fs = _find_vf_in_object(mf)

    if verts is None or faces is None:
        print("Konnte keine v/f Arrays in MeshFix-Objekt finden.")
        print("Verfügbare Kandidaten (verts):", list(cand_vs))
        print("Verfügbare Kandidaten (faces):", list(cand_fs))
        print("Dir(meshfix):", dir(mf))
        raise RuntimeError("MeshFix repariert, aber konnte v/f nicht extrahieren. Bitte gib dir(dir(meshfix)) Ausgabe und ich helfe weiter.")

    # Falls faces Float sind (manche APIs liefern float), cast zu int
    faces = np.asarray(faces)
    if not np.issubdtype(faces.dtype, np.integer):
        faces = faces.astype(np.int64)

    verts = np.asarray(verts)

    repaired = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    try:
        repaired.fix_normals()
    except Exception:
        pass

    print("After watertight:", repaired.is_watertight)
    print("After faces:", len(repaired.faces))
    print("After verts:", len(repaired.vertices))
    print("After volume:", repaired.volume)

    repaired.export(output_stl)
    return repaired

def compare_stl_files(
    true_stl,
    recon_stl,
    voxel_resolution=128,
    projection_resolution=256,
    per_axis_normalization=False
):
    true_mesh = load_mesh(true_stl)
    recon_mesh = load_mesh(recon_stl)

    print("True mesh watertight:", true_mesh.is_watertight)
    print("Recon mesh watertight:", recon_mesh.is_watertight)

    true_mesh = normalize_mesh_to_unit_box(
        true_mesh,
        per_axis=per_axis_normalization
    )

    recon_mesh = normalize_mesh_to_unit_box(
        recon_mesh,
        per_axis=per_axis_normalization
    )

    print("Voxelizing true mesh...")
    A, lin = voxelize_by_trimesh_fill(true_mesh, resolution=voxel_resolution)

    print("Voxelizing reconstructed mesh...")
    B, _ = voxelize_by_trimesh_fill(recon_mesh, resolution=voxel_resolution)

    vox_result = voxel_similarity(A, B)

    side_result = side_view_measure(
        A,
        B,
        lin,
        image_res=projection_resolution
    )

    return vox_result, side_result


if __name__ == "__main__":
    root=Path.cwd()
    make_watertight_with_pymeshfix(
        root/r"asteroid_sdf_reconstruction.stl",
        root/r"reconstruction_watertight.stl"
    )
    true_stl = root/r"asteroid3_scaled_radius0.8782467278262304.stl"
    recon_stl = root/r"reconstruction_watertight.stl"

    vox, side = compare_stl_files(
        true_stl,
        recon_stl,
        
        voxel_resolution=128,
        projection_resolution=256,
        per_axis_normalization=False
    )

    print("\n--- Voxel-based measure ---")
    for key, value in vox.items():
        print(f"{key}: {value}")

    print("\n--- Side-view measure ---")
    print("mean_boundary_distance_px:", side["mean_boundary_distance_px"])
    print("mean_boundary_distance_normalized:", side["mean_boundary_distance_normalized"])
    print("mean_boundary_similarity:", side["mean_boundary_similarity"])