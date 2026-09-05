import numpy as np
import torch
from scipy.signal import savgol_filter
from pathlib import Path
import os
import time
import sys
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from numba import njit
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.rendering.camera_setup import (create_camera)

@njit(cache=True, fastmath=True)
def rasterize_mesh_orthographic_numba(
    vertices,
    faces,
    face_normals,
    camera_pos,
    view,
    x_axis,
    y_axis,
    W,
    H,
    projection_scale,
):
    face_id = np.full((H, W), -1, dtype=np.int32)
    zbuf = np.full((H, W), -np.inf, dtype=np.float32)

    n_vertices = vertices.shape[0]

    px = np.empty(n_vertices, dtype=np.float32)
    py = np.empty(n_vertices, dtype=np.float32)
    pz = np.empty(n_vertices, dtype=np.float32)

    for i in range(n_vertices):
        rx = vertices[i, 0] - camera_pos[0]
        ry = vertices[i, 1] - camera_pos[1]
        rz = vertices[i, 2] - camera_pos[2]

        cam_x = rx * x_axis[0] + ry * x_axis[1] + rz * x_axis[2]
        cam_y = rx * y_axis[0] + ry * y_axis[1] + rz * y_axis[2]
        cam_z = rx * view[0] + ry * view[1] + rz * view[2]

        px[i] = (cam_x / projection_scale + 1.0) * 0.5 * (W - 1)
        py[i] = (cam_y / projection_scale + 1.0) * 0.5 * (H - 1)
        pz[i] = cam_z

    for face_index in range(faces.shape[0]):

        normal_dot_view = (
            face_normals[face_index, 0] * view[0]
            + face_normals[face_index, 1] * view[1]
            + face_normals[face_index, 2] * view[2]
        )

        if normal_dot_view <= 0.0:
            continue

        i0 = faces[face_index, 0]
        i1 = faces[face_index, 1]
        i2 = faces[face_index, 2]

        x0, y0, z0 = px[i0], py[i0], pz[i0]
        x1, y1, z1 = px[i1], py[i1], pz[i1]
        x2, y2, z2 = px[i2], py[i2], pz[i2]

        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)

        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), H - 1)

        if xmin > xmax or ymin > ymax:
            continue

        denominator = (
            (y1 - y2) * (x0 - x2)
            + (x2 - x1) * (y0 - y2)
        )

        if abs(denominator) < 1e-12:
            continue

        inv_denom = 1.0 / denominator

        a0 = (y1 - y2) * inv_denom
        b0 = (x2 - x1) * inv_denom
        c0 = (
            (x2 * y1 - x1 * y2)
            * inv_denom
        )

        a1 = (y2 - y0) * inv_denom
        b1 = (x0 - x2) * inv_denom
        c1 = (
            (x2 * y0 - x0 * y2)
            * inv_denom
        )

        start_x = xmin + 0.5
        start_y = ymin + 0.5

        w0_row = a0 * start_x + b0 * start_y + c0
        w1_row = a1 * start_x + b1 * start_y + c1

        depth_dx = a0 * (z0 - z2) + a1 * (z1 - z2)
        depth_dy = b0 * (z0 - z2) + b1 * (z1 - z2)

        for yy in range(ymin, ymax + 1):
            w0 = w0_row
            w1 = w1_row

            w2 = 1.0 - w0 - w1

            depth = w0 * z0 + w1 * z1 + w2 * z2

            for xx in range(xmin, xmax + 1):

                if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                    if depth > zbuf[yy, xx]:
                        zbuf[yy, xx] = depth
                        face_id[yy, xx] = face_index

                w0 += a0
                w1 += a1
                w2 = 1.0 - w0 - w1

                depth += depth_dx

            w0_row += b0
            w1_row += b1

    return face_id, zbuf

def brightness_from_raster_fast(
    face_id,
    rotated_normals,
    sun_dir,
    projection_scale,
    image_width,
    image_height,
):
    visible_faces = face_id[face_id >= 0]

    if visible_faces.size == 0:
        return 0.0

    illum = np.maximum(rotated_normals @ sun_dir, 0.0)

    pixel_counts = np.bincount(
        visible_faces,
        minlength=len(illum)
    )

    pixel_width = 2.0 * projection_scale / image_width
    pixel_height = 2.0 * projection_scale / image_height

    pixel_area = pixel_width * pixel_height

    return float(np.sum(pixel_counts * illum) * pixel_area)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "publicasteroids"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

frames = 360
brightness = []

start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"

Rs=[]

for frame in range(frames):
    theta = - frame / frames * 2 * np.pi

    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    Rs.append(R)

Rs = np.stack(Rs, axis=0)

cameras = []

for i in range(8):
    if i!=4:
        cam1 = create_camera(f"camera{i}mid", i*45,0)
        cam11 = create_camera(f"camera{i}mid", i*45,0)
        cam2 = create_camera(f"camera{i}top", i*45,1)
        cam3 = create_camera(f"camera{i}bot", i*45,-1)
        cameras.append(cam1)
        cameras.append(cam11)
        cameras.append(cam2)
        cameras.append(cam3)

view_dirs = []

for cam in cameras:
    norm = np.linalg.norm(cam)

    if norm > 0:
        cam = cam / norm
    view_dirs.append(cam)

view_dirs_np = np.array(view_dirs)

sun_dir_np = np.array([-10, 0, 0], dtype=np.float32)
norm = np.linalg.norm(sun_dir_np)
if norm > 0:
    sun_dir_np = sun_dir_np / norm

sun_dir = torch.tensor(sun_dir_np, dtype=torch.float32, device=device)
view_dirs = torch.tensor(view_dirs_np, dtype=torch.float32, device=device)

camera_pos = torch.tensor(
    np.asarray(cameras, dtype=np.float32),
    dtype=torch.float32,
    device=device
)  # (C, 3)

num_cameras = camera_pos.shape[0]

views = F.normalize(camera_pos, dim=1)  # (C, 3)

up = torch.tensor(
    [0.0, 0.0, 1.0],
    dtype=torch.float32,
    device=device
).expand(num_cameras, -1).clone()

parallel_mask = torch.abs(torch.sum(views * up, dim=1)) > 0.95

up[parallel_mask] = torch.tensor(
    [0.0, 1.0, 0.0],
    dtype=torch.float32,
    device=device
)

x_axes = F.normalize(
    torch.cross(views, up, dim=1),
    dim=1
)

y_axes = F.normalize(
    torch.cross(x_axes, views, dim=1),
    dim=1
)

print("Number of cameras:", num_cameras)

for index in range(1):
    root = DATASET_DIR / "unchanged"
    
    asteroids = list(root.glob("asteroid*.stl"))
    print("Found asteroids:", len(asteroids))
    for asteroid in asteroids:
        stl_path = asteroid
        brightness=[]

        W = 64                           #Define granularity of projection
        H = 64
        num_pixels = W * H

        projection_scale = 2.0

        mesh = trimesh.load(stl_path, force="mesh")

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        mesh = mesh.copy()

        try:
            mesh.fix_normals()
        except Exception:
            pass

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        normals = mesh.face_normals.astype(np.float32)

        
        camera_positions_np = np.asarray(cameras, dtype=np.float32)
        views_np = camera_positions_np / (
            np.linalg.norm(camera_positions_np, axis=1, keepdims=True) + 1e-8
        )
        num_cameras = len(camera_positions_np)

        up_np = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            (num_cameras, 1)
        )

        parallel_mask = np.abs(
            np.sum(views_np * up_np, axis=1)
        ) > 0.95

        up_np[parallel_mask] = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32
        )

        x_axes_np = np.cross(views_np, up_np)
        x_axes_np /= (
            np.linalg.norm(x_axes_np, axis=1, keepdims=True) + 1e-8
        )

        y_axes_np = np.cross(x_axes_np, views_np)
        y_axes_np /= (
            np.linalg.norm(y_axes_np, axis=1, keepdims=True) + 1e-8
        )

        for frame in range(frames):
            R=Rs[frame]                        #Get rotationsmatrix for given frame

            rotated_normals = normals @ R.T     #Apply rotation
            rotated_vertices = vertices @ R.T

            frame_values = [frame]

            for cam_index in range(num_cameras):
                cam_pos = camera_positions_np[cam_index]
                view = views_np[cam_index]
                x_axis = x_axes_np[cam_index]
                y_axis = y_axes_np[cam_index]

                face_id, zbuf = rasterize_mesh_orthographic_numba(
                    vertices=rotated_vertices,
                    faces=faces,
                    face_normals=rotated_normals,
                    camera_pos=cam_pos,
                    view=view,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    W=W,
                    H=H,
                    projection_scale=projection_scale,
                )

                brightness_val = brightness_from_raster_fast(
                    face_id=face_id,
                    rotated_normals=rotated_normals,
                    sun_dir=sun_dir_np,
                    projection_scale=projection_scale,
                    image_width=W,
                    image_height=H,
                )

                frame_values.append(brightness_val)

            brightness.append(frame_values)
            
            path=os.path.dirname(stl_path)
            asteroidname=os.path.basename(stl_path)

            # print(asteroid)
            # print(path)

            np.savetxt(f"{path}/brightness{asteroidname}.csv", np.asarray(brightness, dtype=np.float32), delimiter=",")
    

end = time.time()
length = end - start

print("It took", length, "seconds!")