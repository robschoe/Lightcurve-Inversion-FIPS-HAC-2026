from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation
from skimage import measure
import os
import torch
import re

def sample_sdf_from_mesh(stl_path, radius, n_points=8192, tau=0.1):
    mesh = load_normalized_mesh(stl_path, radius)

    return sample_sdf_from_loaded_mesh(
        mesh,
        n_points=n_points,
        tau=tau
    )

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

def load_normalized_mesh(stl_path, radius):
    mesh = trimesh.load(stl_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            list(mesh.geometry.values())
        )

    mesh = clean_mesh(mesh)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(
            f"Mesh enthält nach Bereinigung keine gültige Geometrie: {stl_path}"
        )

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Ungültiger Radius {radius} für {stl_path}")

    mesh.vertices[:, 0] /= radius
    mesh.vertices[:, 1] /= radius

    return mesh

def sample_sdf_from_loaded_mesh(
    mesh,
    n_points=8192,
    tau=0.1,
    query=None
):
    n_surface = int(0.7 * n_points)
    n_uniform = n_points - n_surface

    surface_points = mesh.sample(n_surface)

    near_surface = surface_points + np.random.normal(
        loc=0.0,
        scale=0.03,
        size=surface_points.shape
    )

    uniform = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(n_uniform, 3)
    )

    points = np.concatenate([near_surface, uniform], axis=0)
    points = np.clip(points, -1.0, 1.0)

    if query is None:
        query = trimesh.proximity.ProximityQuery(mesh)

    sdf = query.signed_distance(points)

    invalid = ~np.isfinite(sdf)

    if invalid.any():
        sdf[invalid] = tau

    sdf = np.clip(sdf, -tau, tau)
    sdf = sdf / tau

    return points.astype(np.float32), sdf.astype(np.float32)

def sample_multiple_sdf_sets_from_loaded_mesh(
    mesh,
    n_sets,
    n_points=8192,
    tau=0.1,
    query=None
):
    if query is None:
        query = trimesh.proximity.ProximityQuery(mesh)

    n_surface = int(0.7 * n_points)
    n_uniform = n_points - n_surface

    all_points = []

    for _ in range(n_sets):
        surface_points = mesh.sample(n_surface)

        near_surface = surface_points + np.random.normal(
            scale=0.03,
            size=surface_points.shape
        )

        uniform = np.random.uniform(
            -1.0,
            1.0,
            size=(n_uniform, 3)
        )

        points = np.concatenate(
            [near_surface, uniform],
            axis=0
        )

        points = np.clip(points, -1.0, 1.0)

        all_points.append(points)

    all_points = np.stack(all_points, axis=0)

    flat_points = all_points.reshape(-1, 3)

    flat_sdf = flat_sdf = signed_distance_chunked(
        query,
        flat_points,
        chunk_size=16384
    )

    invalid = ~np.isfinite(flat_sdf)

    if invalid.any():
        print(f"Warning: {invalid.sum()} invalid SDF values")
        flat_sdf[invalid] = tau

    flat_sdf = np.clip(flat_sdf, -tau, tau) / tau

    all_sdf = flat_sdf.reshape(n_sets, n_points)

    return (
        all_points.astype(np.float32),
        all_sdf.astype(np.float32)
    )

def signed_distance_chunked(query, points, chunk_size=16384):
    result = np.empty(
        len(points),
        dtype=np.float64
    )

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))

        result[start:end] = query.signed_distance(
            points[start:end]
        )

    return result


def cylinder_radius_about_z(mesh):
    """
    Radius des kleinsten Zylinders mit z-Achse als Achse.

    r = max sqrt(x^2 + y^2)
    """
    return float(
        np.linalg.norm(
            mesh.vertices[:, :2],
            axis=1,
        ).max()
    )

def rotate_and_normalize_height(mesh, rng):
    """
    Zufällige Rotation und anschließend uniforme Skalierung, sodass:

        z_min = -1
        z_max = +1

    Wichtig:
    Die Skalierung ist uniform. Die Form bleibt also geometrisch
    ähnlich und wird nicht anisotrop verzerrt.
    """
    mesh = mesh.copy()

    rotation = Rotation.random(
        random_state=rng,
    )

    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()

    mesh.apply_transform(transform)

    # Entlang der z-Achse zentrieren.
    z_min = mesh.vertices[:, 2].min()
    z_max = mesh.vertices[:, 2].max()

    z_center = 0.5 * (z_min + z_max)

    mesh.apply_translation([
        0.0,
        0.0,
        -z_center,
    ])

    # Uniform skalieren, bis die Gesamthöhe 2 beträgt.
    height = (
        mesh.vertices[:, 2].max()
        - mesh.vertices[:, 2].min()
    )

    if height <= 0:
        raise ValueError("Mesh hat keine gültige z-Ausdehnung.")

    mesh.apply_scale(2.0 / height)

    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )

    return mesh

def save_mesh_with_radius(mesh, sample_dir):
    """
    Bestimmt den Radius aus der finalen Geometrie
    und speichert das STL mit Radius im Dateinamen.
    """
    radius = cylinder_radius_about_z(mesh)

    stl_path = sample_dir / (
        f"asteroid_radius{radius:.12f}.stl"
    )

    mesh.export(stl_path)

    return stl_path, radius

def create_random_rotated_cube(rng: np.random.Generator) -> trimesh.Trimesh:
    """
    Erstellt einen Würfel, rotiert ihn zufällig im 3D-Raum und skaliert
    ihn anschließend gleichmäßig, sodass z_min=-1 und z_max=1 gilt.

    Der Würfel bleibt dabei ein Würfel, weil die Skalierung isotrop ist.
    """

    # Ausgangswürfel mit Kantenlänge 1, zentriert im Ursprung.
    mesh = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0)
    )

    # Gleichverteilte zufällige 3D-Rotation in SO(3).
    rotation = Rotation.random(
        random_state=rng
    )

    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()

    mesh.apply_transform(transform)

    # Nach Rotation zentrieren.
    z_min = mesh.vertices[:, 2].min()
    z_max = mesh.vertices[:, 2].max()

    z_center = 0.5 * (z_min + z_max)

    mesh.apply_translation(
        [0.0, 0.0, -z_center]
    )

    # Nun soll die Höhe exakt 2 sein:
    # z_min = -1, z_max = +1.
    z_height = (
        mesh.vertices[:, 2].max()
        - mesh.vertices[:, 2].min()
    )

    if z_height <= 0:
        raise ValueError("Ungültige Würfelhöhe.")

    scale = 2.0 / z_height

    # Uniforme Skalierung: Der Würfel bleibt ein Würfel.
    mesh.apply_scale(scale)

    # Numerische Bereinigung / Normalen.
    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )

    return mesh

def create_random_ellipsoid(
    rng,
    axes=(1.4, 0.8, 1.1),
    subdivisions=4,
):
    """
    Erzeugt ein Ellipsoid mit anfänglichen Halbachsen:

        axes = (a, b, c)

    Danach:
    - zufällige 3D-Rotation,
    - uniforme Skalierung auf z in [-1, 1].

    Das finale Objekt bleibt ein Ellipsoid.
    """
    a, b, c = axes

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(
            "Alle Ellipsoid-Halbachsen müssen positiv sein."
        )

    mesh = trimesh.creation.icosphere(
        subdivisions=subdivisions,
        radius=1.0,
    )

    # Kugel -> Ellipsoid
    mesh.vertices[:, 0] *= a
    mesh.vertices[:, 1] *= b
    mesh.vertices[:, 2] *= c

    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )

    mesh = rotate_and_normalize_height(
        mesh,
        rng,
    )

    return mesh

def create_random_sphere(rng, subdivisions=4):
    """
    Erzeugt eine Kugel und bringt sie in die Konvention:

        z_min = -1
        z_max = +1

    Der finale Radius ist immer ungefähr 1.
    Eine Rotation verändert eine perfekte Kugel geometrisch nicht.
    """
    sphere = trimesh.creation.icosphere(
        subdivisions=subdivisions,
        radius=1.0,
    )

    sphere = rotate_and_normalize_height(
        sphere,
        rng,
    )

    return sphere

def generate_shape_batch(
    dataset_dir,
    batch_id,
    n_samples,
    shape_type
):
    """
    shape_type:
        "sphere" oder "ellipsoid"
    """
    rng = np.random.default_rng()

    batch_dir = Path(dataset_dir) / f"batch{batch_id}"
    batch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i in range(n_samples):
        sample_dir = batch_dir / f"sample{i}"
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if shape_type == "sphere":
            mesh = create_random_sphere(
                rng,
                subdivisions=4,
            )

        elif shape_type == "ellipsoid":
            # Zufällige Halbachsen vor Rotation und z-Normierung.
            axes = rng.uniform(
                low=0.5,
                high=1.8,
                size=3,
            )

            mesh = create_random_ellipsoid(
                rng,
                axes=axes,
                subdivisions=4,
            )

        else:
            raise ValueError(
                f"Unbekannter shape_type: {shape_type}"
            )

        stl_path, radius = save_mesh_with_radius(
            mesh,
            sample_dir,
        )

        print(
            f"{sample_dir.name}: "
            f"type={shape_type}, "
            f"radius={radius:.8f}, "
            f"watertight={mesh.is_watertight}, "
            f"file={stl_path.name}"
        )

def sdf_sphere(points, center, radius):
    """Signed Distance of a sphere."""
    return np.linalg.norm(points - center, axis=-1) - radius


def sdf_ellipsoid(points, center, axes):
    """Approximation of Signed Distance of a ellipsoid."""
    a, b, c = axes

    q = points - center

    normalized_radius = np.sqrt(
        (q[..., 0] / a) ** 2
        + (q[..., 1] / b) ** 2
        + (q[..., 2] / c) ** 2
    )

    return (normalized_radius - 1.0) * min(a, b, c)


def sdf_capsule_x(points, x_start, x_end, radius):
    """SDF of a capsule, consisting of two orbs connected by a cylinder."""
    p = points.copy()

    #Determine next point on line segment.
    x_clamped = np.clip(
        p[..., 0],
        x_start,
        x_end,
    )

    closest = np.zeros_like(p)
    closest[..., 0] = x_clamped

    return np.linalg.norm(
        p - closest,
        axis=-1,
    ) - radius


def smooth_union(sdf_a, sdf_b, k=0.15):
    """
    Glatte Vereinigung zweier impliziter Formen.

    k kontrolliert die Glättung:
    - kleines k: fast harte Vereinigung mit min()
    - großes k: weicher Übergang

    Für eine harte Vereinigung genügt:
        return np.minimum(sdf_a, sdf_b)
    """
    h = np.clip(
        0.5 + 0.5 * (sdf_b - sdf_a) / k,
        0.0,
        1.0,
    )

    return (
        sdf_b * (1.0 - h)
        + sdf_a * h
        - k * h * (1.0 - h)
    )


def sdf_bone_asteroid(points, rng):
    """
    Erzeugt ein knochen-, hantel- oder kontaktbinärartiges SDF.

    Die Form besteht aus:
    - linker Lobe,
    - rechter Lobe,
    - verbindender Kapsel,
    - optionalen kleineren Unebenheiten.
    """

    # Unterschiedliche Lobenpositionen.
    left_center = np.array([
        rng.uniform(-0.95, -0.65),
        rng.uniform(-0.10, 0.10),
        rng.uniform(-0.10, 0.10),
    ])

    right_center = np.array([
        rng.uniform(0.65, 0.95),
        rng.uniform(-0.10, 0.10),
        rng.uniform(-0.10, 0.10),
    ])

    # Unterschiedliche Ellipsoidachsen für beide Enden.
    left_axes = np.array([
        rng.uniform(0.45, 0.75),
        rng.uniform(0.35, 0.65),
        rng.uniform(0.35, 0.65),
    ])

    right_axes = np.array([
        rng.uniform(0.45, 0.80),
        rng.uniform(0.35, 0.65),
        rng.uniform(0.35, 0.65),
    ])

    # Loben.
    sdf_left = sdf_ellipsoid(
        points,
        center=left_center,
        axes=left_axes,
    )

    sdf_right = sdf_ellipsoid(
        points,
        center=right_center,
        axes=right_axes,
    )

    # Verbindung zwischen den Loben.
    neck_radius = rng.uniform(0.22, 0.42)

    sdf_neck = sdf_capsule_x(
        points,
        x_start=left_center[0],
        x_end=right_center[0],
        radius=neck_radius,
    )

    # Erst linke Lobe und Steg, dann rechte Lobe hinzufügen.
    sdf = smooth_union(
        sdf_left,
        sdf_neck,
        k=0.12,
    )

    sdf = smooth_union(
        sdf,
        sdf_right,
        k=0.12,
    )

    # Kleine Nebenlobe für asymmetrische, asteroidartige Form.
    if rng.random() < 0.7:
        bump_center = np.array([
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.45, 0.45),
            rng.uniform(-0.35, 0.35),
        ])

        bump_radius = rng.uniform(0.10, 0.25)

        sdf_bump = sdf_sphere(
            points,
            center=bump_center,
            radius=bump_radius,
        )

        sdf = smooth_union(
            sdf,
            sdf_bump,
            k=0.08,
        )

    return sdf


def implicit_sdf_to_mesh(
    sdf_function,
    rng,
    resolution=128,
    grid_extent=2.0,
):
    """
    Erzeugt aus einer SDF-Funktion ein trimesh-Mesh.

    Das Feld wird auf [-grid_extent, grid_extent]^3 ausgewertet.
    Der Bereich muss groß genug sein, damit die SDF an allen
    Volumenrändern positiv ist.
    """
    coords = np.linspace(
        -grid_extent,
        grid_extent,
        resolution,
        dtype=np.float32,
    )

    X, Y, Z = np.meshgrid(
        coords,
        coords,
        coords,
        indexing="ij",
    )

    points = np.stack(
        [X, Y, Z],
        axis=-1,
    )

    sdf = sdf_function(
        points,
        rng,
    ).astype(np.float32)

    # Sicherheitscheck: Oberfläche muss innerhalb des Gitters liegen.
    boundary = np.concatenate([
        sdf[0, :, :].ravel(),
        sdf[-1, :, :].ravel(),
        sdf[:, 0, :].ravel(),
        sdf[:, -1, :].ravel(),
        sdf[:, :, 0].ravel(),
        sdf[:, :, -1].ravel(),
    ])

    if np.any(boundary <= 0.0):
        raise ValueError(
            "Die Form berührt den Rand des impliziten Gitters. "
            "grid_extent erhöhen."
        )

    voxel_size = (
        2.0 * grid_extent
    ) / (resolution - 1)

    vertices, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(
            voxel_size,
            voxel_size,
            voxel_size,
        ),
    )

    # marching_cubes startet bei Koordinate 0.
    vertices += np.array([
        -grid_extent,
        -grid_extent,
        -grid_extent,
    ])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
    )

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )

    return mesh

def create_bone_asteroid(rng, resolution=64):
    """
    Vollständige Pipeline für einen knochenförmigen Asteroiden.
    """
    mesh = implicit_sdf_to_mesh(
        sdf_function=sdf_bone_asteroid,
        rng=rng,
        resolution=resolution,
        grid_extent=2.0,
    )

    mesh = rotate_and_normalize_height(
        mesh,
        rng,
    )

    return mesh


def save_bone_asteroid_batch(
    dataset_dir,
    batch_id,
    n_samples,
):
    """
    Speichert die STL-Dateien unter:

        data/dataset/batchX/sampleY/
            asteroid_radiusR.stl
    """
    dataset_dir = Path(dataset_dir)

    batch_dir = dataset_dir / f"batch{batch_id}"
    batch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng()

    for i in range(n_samples):
        sample_dir = batch_dir / f"sample{i}"
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mesh = create_bone_asteroid(
            rng,
            resolution=64,
        )

        radius = cylinder_radius_about_z(mesh)

        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        if not np.isclose(z_min, -1.0, atol=1e-5):
            raise RuntimeError(
                f"z_min falsch: {z_min}"
            )

        if not np.isclose(z_max, 1.0, atol=1e-5):
            raise RuntimeError(
                f"z_max falsch: {z_max}"
            )

        if not mesh.is_watertight:
            print(
                f"Warning: sample{i} is not watertight."
            )

        stl_path = sample_dir / (
            f"asteroid_radius{radius:.12f}.stl"
        )

        mesh.export(stl_path)

        print(
            f"{sample_dir.name} | "
            f"radius={radius:.8f} | "
            f"faces={len(mesh.faces)} | "
            f"watertight={mesh.is_watertight}"
        )