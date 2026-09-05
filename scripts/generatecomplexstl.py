import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SEED = 42

SUBDIVISIONS = 5

MIN_CYLINDER_RADIUS = 0.5
MAX_CYLINDER_RADIUS = 6.0

BASE_RADIUS = 3.0

AXIS_SCALE = np.array([
    1.25, 
    0.85, 
    1.05,
], dtype=np.float32)

N_LARGE_LOBES = 12
N_MEDIUM_LOBES = 35
N_SMALL_LOBES = 100

N_CRATERS = 30

NOISE_STRENGTH = 0.05

SIMPLIFY = False
TARGET_FACES = 8000

def fit_mesh_into_z_cylinder(
    mesh,
    target_radius,
    target_z_min=-1.0,
    target_z_max=1.0
):
    """
    Skaliert und verschiebt ein Mesh so, dass:

    - die Zylinderachse die globale z-Achse ist,
    - z_min exakt target_z_min wird,
    - z_max exakt target_z_max wird,
    - der maximale xy-Abstand exakt target_radius wird.

    Rückgabe:
        actual_radius,
        actual_z_min,
        actual_z_max,
        xy_center_before,
        xy_scale,
        z_scale
    """

    vertices = mesh.vertices.astype(np.float64).copy()

    # --------------------------------------------------------
    # 1. Bounding Box in z bestimmen
    # --------------------------------------------------------
    z_min_before = vertices[:, 2].min()
    z_max_before = vertices[:, 2].max()

    z_center = 0.5 * (z_min_before + z_max_before)
    z_half_extent = 0.5 * (z_max_before - z_min_before)

    if z_half_extent < 1e-12:
        raise ValueError(
            "Mesh besitzt praktisch keine Ausdehnung in z-Richtung."
        )

    # Zielhöhe: z in [target_z_min, target_z_max]
    target_z_center = 0.5 * (
        target_z_min + target_z_max
    )

    target_z_half_extent = 0.5 * (
        target_z_max - target_z_min
    )

    z_scale = target_z_half_extent / z_half_extent

    # --------------------------------------------------------
    # 2. In xy-Ebene auf z-Achse zentrieren
    # --------------------------------------------------------
    #
    # Nach diesem Schritt liegt die Zylinderachse bei (0, 0).
    #
    xy_center = vertices[:, :2].mean(axis=0)

    vertices[:, 0] -= xy_center[0]
    vertices[:, 1] -= xy_center[1]

    # --------------------------------------------------------
    # 3. Aktuellen radialen Abstand zur z-Achse bestimmen
    # --------------------------------------------------------
    radial_distances = np.linalg.norm(
        vertices[:, :2],
        axis=1
    )

    current_radius = radial_distances.max()

    if current_radius < 1e-12:
        raise ValueError(
            "Mesh besitzt praktisch keine Ausdehnung in der xy-Ebene."
        )

    xy_scale = target_radius / current_radius

    # --------------------------------------------------------
    # 4. Skalierung anwenden
    # --------------------------------------------------------
    vertices[:, 0] *= xy_scale
    vertices[:, 1] *= xy_scale

    vertices[:, 2] = (
        (vertices[:, 2] - z_center)
        * z_scale
        + target_z_center
    )

    mesh.vertices = vertices

    # Cache leeren, damit trimesh Normalen/Bounds neu berechnet
    mesh._cache.clear()

    # Normalen nach anisotroper Skalierung neu berechnen
    trimesh.repair.fix_normals(
        mesh,
        multibody=True
    )

    # --------------------------------------------------------
    # 5. Tatsächliche Werte nach der Transformation bestimmen
    # --------------------------------------------------------
    final_vertices = mesh.vertices

    actual_radius = np.max(
        np.linalg.norm(
            final_vertices[:, :2],
            axis=1
        )
    )

    actual_z_min = final_vertices[:, 2].min()
    actual_z_max = final_vertices[:, 2].max()

    return (
        float(actual_radius),
        float(actual_z_min),
        float(actual_z_max),
        xy_center,
        float(xy_scale),
        float(z_scale)
    )

def random_unit_vectors(n, rng):
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors.astype(np.float32)


def angular_distance(unit_vectors_a, unit_vectors_b):
    dot_products = unit_vectors_a @ unit_vectors_b.T

    dot_products = np.clip(
        dot_products,
        -1.0,
        1.0
    )

    return np.arccos(dot_products)


def add_gaussian_lobes(
    radius,
    directions,
    n_lobes,
    angular_width_range,
    amplitude_range,
    rng,
    allow_negative=True
):

    lobe_centers = random_unit_vectors(n_lobes, rng)

    widths = rng.uniform(
        angular_width_range[0],
        angular_width_range[1],
        size=n_lobes
    )

    amplitudes = rng.uniform(
        amplitude_range[0],
        amplitude_range[1],
        size=n_lobes
    )

    if allow_negative:
        signs = rng.choice(
            [-1.0, 1.0],
            size=n_lobes,
            p=[0.45, 0.55]
        )

        amplitudes *= signs

    distances = angular_distance(
        directions,
        lobe_centers
    )

    influence = np.exp(
        -0.5
        * (distances / widths[None, :]) ** 2
    )

    deformation = influence @ amplitudes

    return radius + deformation


def add_craters(
    radius,
    directions,
    n_craters,
    rng
):
    crater_centers = random_unit_vectors(n_craters, rng)

    crater_radii = rng.uniform(
        math.radians(4.0),
        math.radians(18.0),
        size=n_craters
    )

    crater_depths = rng.uniform(
        0.05,
        0.30,
        size=n_craters
    )

    rim_heights = crater_depths * rng.uniform(
        0.15,
        0.45,
        size=n_craters
    )

    distances = angular_distance(
        directions,
        crater_centers
    )

    deformation = np.zeros(
        len(directions),
        dtype=np.float32
    )

    for crater_index in range(n_craters):
        distance = distances[:, crater_index]
        crater_radius = crater_radii[crater_index]

        normalized_distance = distance / crater_radius

        inside = normalized_distance < 1.0

        bowl = -crater_depths[crater_index] * np.exp(
            -3.0 * normalized_distance ** 2
        )

        rim = rim_heights[crater_index] * np.exp(
            -35.0
            * (normalized_distance - 0.85) ** 2
        )

        crater_effect = bowl + rim

        deformation[inside] += crater_effect[inside]

    return radius + deformation


def add_vertex_noise(radius, strength, rng):
    noise = rng.normal(
        loc=0.0,
        scale=strength,
        size=radius.shape
    )

    return radius + noise.astype(np.float32)


def smooth_vertex_radii(vertices, radius, iterations=3):
    tree = cKDTree(vertices)

    current_radius = radius.copy()

    _, neighbors = tree.query(
        vertices,
        k=10
    )

    for _ in range(iterations):
        current_radius = np.mean(
            current_radius[neighbors],
            axis=1
        )

    return current_radius

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

for h in range(10):
    root = DATASET_DIR/f"batch{103+h}"
    root.mkdir(exist_ok=True)
    for g in range (1000):
        root = DATASET_DIR/f"batch{103+h}/sample{g}"
        root.mkdir(exist_ok=True)

        rng = np.random.default_rng()

        mesh = trimesh.creation.icosphere(
            subdivisions=SUBDIVISIONS,
            radius=1.0
        )

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.copy()

        directions = vertices / np.linalg.norm(
            vertices,
            axis=1,
            keepdims=True
        )

        radius = np.full(
            len(vertices),
            BASE_RADIUS,
            dtype=np.float32
        )

        radius = add_gaussian_lobes(
            radius=radius,
            directions=directions,
            n_lobes=N_LARGE_LOBES,
            angular_width_range=(
                math.radians(20),
                math.radians(55)
            ),
            amplitude_range=(0.15, 0.70),
            rng=rng,
            allow_negative=True
        )

        radius = add_gaussian_lobes(
            radius=radius,
            directions=directions,
            n_lobes=N_MEDIUM_LOBES,
            angular_width_range=(
                math.radians(7),
                math.radians(22)
            ),
            amplitude_range=(0.04, 0.22),
            rng=rng,
            allow_negative=True
        )

        radius = add_gaussian_lobes(
            radius=radius,
            directions=directions,
            n_lobes=N_SMALL_LOBES,
            angular_width_range=(
                math.radians(2),
                math.radians(7)
            ),
            amplitude_range=(0.01, 0.07),
            rng=rng,
            allow_negative=True
        )

        radius = add_craters(
            radius=radius,
            directions=directions,
            n_craters=N_CRATERS,
            rng=rng
        )

        radius = add_vertex_noise(
            radius=radius,
            strength=NOISE_STRENGTH,
            rng=rng
        )

        radius = np.clip(
            radius,
            BASE_RADIUS * 0.20,
            None
        )

        vertices = directions * radius[:, None]

        vertices *= AXIS_SCALE[None, :]

        approx_radius = np.linalg.norm(
            vertices,
            axis=1
        )

        smoothed_radius = smooth_vertex_radii(
            vertices=vertices,
            radius=approx_radius,
            iterations=2
        )

        directions_after_scale = vertices / np.linalg.norm(
            vertices,
            axis=1,
            keepdims=True
        )

        vertices = directions_after_scale * smoothed_radius[:, None]

        asteroid = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=True
        )

        trimesh.repair.fix_normals(
            asteroid,
            multibody=True
        )

        asteroid.update_faces(asteroid.unique_faces())

        asteroid.update_faces(asteroid.nondegenerate_faces())

        asteroid.remove_unreferenced_vertices()

        if SIMPLIFY and len(asteroid.faces) > TARGET_FACES:
            print(
                f"Vereinfache Mesh: {len(asteroid.faces)} "
                f"-> {TARGET_FACES} Faces"
            )

            asteroid = asteroid.simplify_quadric_decimation(
                face_count=TARGET_FACES
            )

            trimesh.repair.fix_normals(
                asteroid,
                multibody=True
            )

        target_cylinder_radius = rng.uniform(
            MIN_CYLINDER_RADIUS,
            MAX_CYLINDER_RADIUS
        )

        (
            cylinder_radius_z,
            z_min,
            z_max,
            xy_center,
            xy_scale,
            z_scale
        ) = fit_mesh_into_z_cylinder(
            asteroid,
            target_radius=target_cylinder_radius,
            target_z_min=-1.0,
            target_z_max=1.0
        )

        print(f"Zylinder-Radius um Z: {cylinder_radius_z:.6f}")
        print(f"Z-Bereich:             [{z_min:.6f}, {z_max:.6f}]")

        OUTPUT_PATH = root / f"asteroid{g}radius{cylinder_radius_z:.8f}.stl"

        asteroid.export(OUTPUT_PATH)

        print("STL gespeichert:", OUTPUT_PATH.absolute())