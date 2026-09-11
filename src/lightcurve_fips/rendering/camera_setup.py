import numpy as np
from scipy.signal import savgol_filter
import math
import time
from pathlib import Path
import os
from tqdm.auto import tqdm

import torch
import trimesh

#This implementation relies on CUDA because geometry processing and
#lightcurve simulation are performed on the GPU.
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA-capable GPU found. This script requires CUDA-enabled PyTorch."
    )

device = torch.device("cuda:0")

def angle_to_mode(angle):
    """Simple function to match measurement angle to Top camera angle, as described in Challenge"""
    match angle:
        case 0:
            return 21
        case 45 | 135:
            return 26
        case 90:
            return 26
        case 225 | 270:
            return 24
        case 315:
            return 24
        case _:
            raise ValueError(f"Unsupported measurement angle: {angle}")

def load_face_data(path):
    """Loads an STL mesh and prepares per-face geometry data on the GPU."""
    mesh = trimesh.load(path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("STL could not be loaded as a single trimesh.Trimesh object.")

    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(mesh, multibody=True)

    triangles_np = mesh.triangles.astype(np.float32)

    edge_1 = triangles_np[:, 1] - triangles_np[:, 0]
    edge_2 = triangles_np[:, 2] - triangles_np[:, 0]

    cross = np.cross(edge_1, edge_2)
    double_area = np.linalg.norm(cross, axis=1)
    face_areas_np = 0.5 * double_area

    face_normals_np = cross / np.maximum(
        double_area[:, None],
        1e-12
    )

    #Triangles with near zero surface area are unlikely to provide meaningful normals or surface samples.
    valid = face_areas_np > 1e-12

    triangles_np = triangles_np[valid]
    face_normals_np = face_normals_np[valid]
    face_areas_np = face_areas_np[valid]

    if len(triangles_np) == 0:
        raise ValueError("The mesh contains no non-degenerate triangular faces.")

    #Move the geometry data to the GPU once so it can be reused efficiently
    #during simulation.
    return (
        torch.tensor(
            face_normals_np,
            dtype=torch.float32,
            device=device
        ),
        torch.tensor(
            face_areas_np,
            dtype=torch.float32,
            device=device
        ),
        torch.tensor(
            triangles_np,
            dtype=torch.float32,
            device=device
        ),
        mesh
    )

@torch.inference_mode()
def otsu_threshold_batch_uint8(images_uint8):
    """Compute one Otsu threshold for each uint8 image in a batch."""
    batch_size = images_uint8.shape[0]

    values = images_uint8.reshape(
        batch_size,
        -1
    ).long()

    offsets = (
        torch.arange(
            batch_size,
            device=images_uint8.device,
            dtype=torch.long
        )[:, None]
        * 256
    )

    histogram = torch.bincount(
        (values + offsets).reshape(-1),
        minlength=batch_size * 256
    ).reshape(batch_size, 256).float()

    probability = histogram / histogram.sum(
        dim=1,
        keepdim=True
    ).clamp_min(1.0)

    omega = torch.cumsum(
        probability,
        dim=1
    )

    values_0_255 = torch.arange(
        256,
        dtype=torch.float32,
        device=images_uint8.device
    )[None, :]

    mean = torch.cumsum(
        probability * values_0_255,
        dim=1
    )

    mean_total = mean[:, -1:]

    denominator = omega * (1.0 - omega)

    sigma_between = torch.zeros_like(
        denominator
    )

    valid = denominator > 1e-12

    sigma_between[valid] = (
        (
            mean_total.expand_as(omega)[valid]
            * omega[valid]
            - mean[valid]
        ) ** 2
        / denominator[valid]
    )

    return torch.argmax(
        sigma_between,
        dim=1
    ).to(torch.uint8)

def create_unique_camera_positions(radius):
    """Generate the unique camera positions defined by the measurement setup."""
    positions = []

    for i in range(8):
        #The 180-degree direction is not included in the challenge setup.
        if i == 4:
            continue

        angle = i * 45

        for mode in [0, 1, -1]:
            angle_xy = (
                (angle / 360.0) + 0.5
            ) * 2.0 * math.pi

            angle_z = (
                (math.pi / 2.0)
                * angle_to_mode(angle)
                / 90.0
                * mode
                * (-1)
                + math.pi / 2.0
            )

            #Convert spherical coordinates to Cartesian coordinates.
            x = radius * math.cos(angle_xy) * math.sin(angle_z)
            y = radius * math.sin(angle_xy) * math.sin(angle_z)
            z = radius * math.cos(angle_z)

            positions.append([x, y, z])

    return np.asarray(positions, dtype=np.float32)

def create_camera_axes(camera_positions_np):
    """Construct orthonormal view, right, and up vectors for each camera. Every camera is assumed to point from its position towards the origin."""
    views = []
    rights = []
    ups = []

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for position in camera_positions_np:
        view = position / np.linalg.norm(position)

        local_up = world_up.copy()

        #Near the north or south pole, world_up and view are nearly parallel. Use a different reference axis to avoid an unstable cross product.
        if abs(np.dot(view, local_up)) > 0.999:
            local_up = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )

        right = np.cross(view, local_up)
        right /= np.linalg.norm(right)

        up = np.cross(right, view)
        up /= np.linalg.norm(up)

        views.append(view)
        rights.append(right)
        ups.append(up)

    return (
        np.asarray(views, dtype=np.float32),
        np.asarray(rights, dtype=np.float32),
        np.asarray(ups, dtype=np.float32)
    )

def create_barycentric_samples(n_samples, device):
    """Generate approximately uniform deterministic barycentric samples."""
    grid_size = math.ceil(
        (-1.0 + math.sqrt(1.0 + 8.0 * n_samples)) / 2.0
    )

    samples = []

    for i in range(grid_size):
        for j in range(grid_size - i):
            u = (i + 1.0 / 3.0) / grid_size
            v = (j + 1.0 / 3.0) / grid_size
            w = 1.0 - u - v

            samples.append([w, u, v])

    barycentric = np.asarray(
        samples[:n_samples],
        dtype=np.float32
    )

    if len(barycentric) < n_samples:
        raise RuntimeError(
            f"Generated too few samples: {len(barycentric)} "
            f"instead of {n_samples}."
        )

    return torch.tensor(
        barycentric,
        dtype=torch.float32,
        device=device
    )

@torch.inference_mode()
def rotated_face_normals(frame_ids, face_normals, FRAMES):
    """Rotate face normals around the global z-axis for multiple rotation frames."""
    angles = (
        -2.0
        * math.pi
        * frame_ids.float()
        / FRAMES
    )

    cosine = torch.cos(angles)[:, None]
    sine = torch.sin(angles)[:, None]

    nx = face_normals[:, 0][None, :]
    ny = face_normals[:, 1][None, :]
    nz = face_normals[:, 2][None, :]

    rotated_x = cosine * nx - sine * ny
    rotated_y = sine * nx + cosine * ny
    rotated_z = nz.expand_as(rotated_x)

    return torch.stack(
        [rotated_x, rotated_y, rotated_z],
        dim=-1
    )
