from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation
from skimage import measure
import os
import torch
import re

def clean_mesh(mesh, area_epsilon=1e-12):
    """Remove invalid vertices, duplicate faces, and degenerate triangles."""

    #Work on a copy.
    mesh = mesh.copy()

    #Keep only vertices with finite coordinates
    finite_vertices = np.all(np.isfinite(mesh.vertices), axis=1)

    if not np.all(finite_vertices):
        #Remove faces referencing invalid vertices.
        valid_faces = np.all(finite_vertices[mesh.faces], axis=1)

        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()

    #Remove duplicate faces.
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    #Remove degenerate faces.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()

    #Remove faces with non-finite or close to zero areas.
    face_areas = mesh.area_faces

    valid_faces = np.isfinite(face_areas) & (face_areas > area_epsilon)

    mesh.update_faces(valid_faces)
    mesh.remove_unreferenced_vertices()

    #Merge nearly identical vertices.
    try:
        mesh.merge_vertices()
    except Exception:
        pass

    #Remove degenerate faces that may appear after merging.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())

    mesh.remove_unreferenced_vertices()

    #Repair face orientation and normal consistency.
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    return mesh

def load_normalized_mesh(stl_path, radius):
    """Load, clean, and normalize radius of an STL mesh."""

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

def cylinder_radius_about_z(mesh):
    """Radius of the smallest possible cylinder around the mesh."""
    return float(
        np.linalg.norm(
            mesh.vertices[:, :2],
            axis=1,
        ).max()
    )

def rotate_and_normalize_height(mesh, rng):
    """Apply a random rotation and uniformly scale the mesh to span z = [-1, 1]."""

    #Work on a copy.
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
    """Compute the final mesh radius and save it in the STL filename."""

    radius = cylinder_radius_about_z(mesh)

    stl_path = sample_dir / (
        f"asteroid_radius{radius:.12f}.stl"
    )

    mesh.export(stl_path)

    return stl_path, radius

def create_random_rotated_cube(rng: np.random.Generator) -> trimesh.Trimesh:
    """Create a randomly rotated cube and uniformly scale its height to [-1, 1]."""

    #Create a unit cube centered at the origin.
    mesh = trimesh.creation.box(
        extents=(1.0, 1.0, 1.0)
    )

    #Apply uniformly distributed rotation.
    rotation = Rotation.random(
        random_state=rng
    )

    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()

    mesh.apply_transform(transform)

    #Center rotated cube.
    z_min = mesh.vertices[:, 2].min()
    z_max = mesh.vertices[:, 2].max()

    z_center = 0.5 * (z_min + z_max)

    mesh.apply_translation(
        [0.0, 0.0, -z_center]
    )

    #Check height of cube, expected is the total height 2.
    z_height = (
        mesh.vertices[:, 2].max()
        - mesh.vertices[:, 2].min()
    )

    if z_height <= 0:
        raise ValueError("Ungültige Würfelhöhe.")

    #Scale to wanted height.
    scale = 2.0 / z_height
    mesh.apply_scale(scale)

    #Remove unused vertices.
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
    """Create a randomly rotated ellipsoid with height normalized to [-1, 1]."""

    a, b, c = axes

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(
            "All ellipsoid semi-axis lengths must be positive."
        )

    #Start with unit sphere.
    mesh = trimesh.creation.icosphere(
        subdivisions=subdivisions,
        radius=1.0,
    )

    #Scale each axis independently.
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
    """Create a sphere with its height normalized to [-1, 1]."""

    #Create unit sphere with its center at the origin.
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
    """Generate and save a batch of random sphere or ellipsoid meshes."""

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
                f"Unknown shape_type: {shape_type}"
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
    """Compute a smooth union of two implicit SDF shapes."""

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
    """Create a bone- or dumbbell-shaped object SDF"""

    #Sample separate centers for the left and right lobes.
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

    #Sample independent ellipsoid semi-axis lengths for both lobes.
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

    #Create the lobes.
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

    #Create a capsule-shaped bridge between the lobes.
    neck_radius = rng.uniform(0.22, 0.42)

    sdf_bridge = sdf_capsule_x(
        points,
        x_start=left_center[0],
        x_end=right_center[0],
        radius=neck_radius,
    )

    #Smoothly connect left lobe, bridge and right lobe.
    sdf = smooth_union(
        sdf_left,
        sdf_bridge,
        k=0.12,
    )

    sdf = smooth_union(
        sdf,
        sdf_right,
        k=0.12,
    )

    #Randomly add a small lobe to create more irregular shapes.
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
    """Extract a trimesh mesh from an implicit signed distance function."""

    #Create regular 3D sampling grid.
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

    #Evaluate the implicit shape on the full grid.
    sdf = sdf_function(
        points,
        rng,
    ).astype(np.float32)

    #Ensure that the zero level set does not touch the grid boundary.
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
            "The shape touches the implicit grid boundary. "
            "Increase grid_extent."
        )

    voxel_size = (
        2.0 * grid_extent
    ) / (resolution - 1)

    #Extract the SDF zero level set as a triangle mesh.
    vertices, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(
            voxel_size,
            voxel_size,
            voxel_size,
        ),
    )

    #Marching Cubes uses an origin at (0, 0, 0); shift vertices back.
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
    """Create a bone-shaped asteroid mesh from an implicit SDF."""

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
    """Generate and save a batch of bone-shaped asteroid STL meshes."""

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

        #Check the height.
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        if not np.isclose(z_min, -1.0, atol=1e-5):
            raise RuntimeError(
                f"Incorrect z_min: {z_min}"
            )

        if not np.isclose(z_max, 1.0, atol=1e-5):
            raise RuntimeError(
                f"Incorrect z_max: {z_max}"
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