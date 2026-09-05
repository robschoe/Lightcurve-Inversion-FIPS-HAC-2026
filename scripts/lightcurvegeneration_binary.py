import math
import time
from pathlib import Path
import os

import numpy as np
import torch
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import parse_radius_from_stl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

USE_CONCAVE_ZBUFFER = True

IMAGE_SIZE = 128

ZBUFFER_IMAGE_SIZE = IMAGE_SIZE

SAMPLES_PER_FACE = 32

CONCAVE_BATCH_SIZE = 64

FRAMES = 360

LIGHT_DIRECTION = (-1.0, 0.0, 0.0)

LIGHT_INTENSITY = 3.5

CENTER_MESH = False

KEEP_DUPLICATE_MID_CAMERA = True

BATCH_SIZE = 1024

SAVE_DEBUG_NPZ = False

if not torch.cuda.is_available():
    raise RuntimeError(
        "Keine CUDA-GPU gefunden. Dieses Script benötigt CUDA-PyTorch."
    )

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU-Speicher:",
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

torch.set_grad_enabled(False)

start_time = time.perf_counter()

def angle_to_mode(angle):
    if angle == 0:
        return 21

    if angle in {45, 90, 135}:
        return 26

    if angle in {225, 270, 315}:
        return 24

    return 21

def load_face_data(path):
    mesh = trimesh.load(path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            "STL konnte nicht als einzelnes trimesh.Trimesh geladen werden."
        )

    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(mesh, multibody=True)

    if CENTER_MESH:
        mesh.apply_translation(-mesh.vertices.mean(axis=0))

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

    valid = face_areas_np > 1e-12

    triangles_np = triangles_np[valid]
    face_normals_np = face_normals_np[valid]
    face_areas_np = face_areas_np[valid]

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

for batch in range(103):
    root = DATASET_DIR/f"batch{25+batch}"
    
    samples = sorted([
        p for p in root.rglob("sample*")
        if list(p.glob("asteroid*.stl"))
    ])
    
    asteroids = list(root.rglob("asteroid*.stl"))
    print("Found asteroids:", len(samples))
    for folder in samples:
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]

        ASTEROID_RADIUS = parse_radius_from_stl(stl_path)
        ASTEROID_HALF_HEIGHT = 1.0

        IMAGE_FILL = 0.85

        bounding_radius = math.sqrt(
            ASTEROID_RADIUS ** 2
            + ASTEROID_HALF_HEIGHT ** 2
        )

        ORTHO_SCALE = (
            2.0 * bounding_radius / IMAGE_FILL
        )

        CAMERA_RADIUS = 3.0 * bounding_radius

        face_normals, face_areas, triangles, mesh = load_face_data(
            stl_path
        )

        n_faces = len(face_areas)

        @torch.inference_mode()
        def otsu_threshold_batch_uint8(images_uint8):
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

        def create_unique_camera_positions(radius=CAMERA_RADIUS):
            positions = []

            for i in range(8):
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

                    x = radius * math.cos(angle_xy) * math.sin(angle_z)
                    y = radius * math.sin(angle_xy) * math.sin(angle_z)
                    z = radius * math.cos(angle_z)

                    positions.append([x, y, z])

            return np.asarray(positions, dtype=np.float32)

        camera_positions_np = create_unique_camera_positions()

        n_cameras = len(camera_positions_np)

        camera_positions = torch.tensor(
            camera_positions_np,
            device=device,
            dtype=torch.float32
        )

        camera_directions = torch.nn.functional.normalize(
            camera_positions,
            dim=1
        )

        def create_camera_axes(camera_positions_np):
            views = []
            rights = []
            ups = []

            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            for position in camera_positions_np:
                view = position / np.linalg.norm(position)

                local_up = world_up.copy()

                # Sonderfall: Kamera fast an Nord-/Südpol
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


        camera_views_np, camera_rights_np, camera_ups_np = (
            create_camera_axes(camera_positions_np)
        )

        camera_views = torch.tensor(
            camera_views_np,
            dtype=torch.float32,
            device=device
        )

        camera_rights = torch.tensor(
            camera_rights_np,
            dtype=torch.float32,
            device=device
        )

        camera_ups = torch.tensor(
            camera_ups_np,
            dtype=torch.float32,
            device=device
        )

        def create_barycentric_samples(n_samples, device):
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
                    f"Zu wenige Stichproben erzeugt: "
                    f"{len(barycentric)} statt {n_samples}"
                )

            return torch.tensor(
                barycentric,
                dtype=torch.float32,
                device=device
            )


        barycentric_samples = create_barycentric_samples(
            SAMPLES_PER_FACE,
            device
        )

        face_sample_points = torch.einsum(
            "fvc,sv->fsc",
            triangles,
            barycentric_samples
        )

        @torch.inference_mode()
        def render_concave_zbuffer(frame_ids, camera_ids):

            batch_size = len(frame_ids)
            n_faces = face_sample_points.shape[0]
            n_samples = face_sample_points.shape[1]
            image_size = ZBUFFER_IMAGE_SIZE

            angles = (
                -2.0
                * math.pi
                * frame_ids.float()
                / FRAMES
            )

            cosine = torch.cos(angles)
            sine = torch.sin(angles)

            px = face_sample_points[:, :, 0][None, :, :]
            py = face_sample_points[:, :, 1][None, :, :]
            pz = face_sample_points[:, :, 2][None, :, :]

            rotated_x = cosine[:, None, None] * px - sine[:, None, None] * py
            rotated_y = sine[:, None, None] * px + cosine[:, None, None] * py
            rotated_z = pz.expand(batch_size, -1, -1)

            points_world = torch.stack(
                [rotated_x, rotated_y, rotated_z],
                dim=-1
            )

            nx = face_normals[:, 0][None, :]
            ny = face_normals[:, 1][None, :]
            nz = face_normals[:, 2][None, :]

            normal_x = cosine[:, None] * nx - sine[:, None] * ny
            normal_y = sine[:, None] * nx + cosine[:, None] * ny
            normal_z = nz.expand(batch_size, -1)

            normals_world = torch.stack(
                [normal_x, normal_y, normal_z],
                dim=-1
            )

            light_cosine = torch.sum(
                normals_world * light_direction[None, None, :],
                dim=-1
            ).clamp_min(0.0)

            face_brightness = torch.round(
                (
                    light_cosine
                    * LIGHT_INTENSITY
                ).clamp(0.0, 1.0)
                * 255.0
            ).to(torch.int64)

            view = camera_views[camera_ids]
            right = camera_rights[camera_ids]
            up = camera_ups[camera_ids]

            x_camera = torch.sum(
                points_world * right[:, None, None, :],
                dim=-1
            )

            y_camera = torch.sum(
                points_world * up[:, None, None, :],
                dim=-1
            )

            depth = torch.sum(
                points_world * view[:, None, None, :],
                dim=-1
            )

            x_pixel = torch.floor(
                (
                    x_camera / ORTHO_SCALE
                    + 0.5
                )
                * image_size
            ).long()

            y_pixel = torch.floor(
                (
                    y_camera / ORTHO_SCALE
                    + 0.5
                )
                * image_size
            ).long()

            inside = (
                (x_pixel >= 0)
                & (x_pixel < image_size)
                & (y_pixel >= 0)
                & (y_pixel < image_size)
            )

            n_pixels = image_size * image_size
            x_pixel = x_pixel.reshape(batch_size, -1)
            y_pixel = y_pixel.reshape(batch_size, -1)
            depth = depth.reshape(batch_size, -1)
            inside = inside.reshape(batch_size, -1)

            brightness_points = face_brightness[:, :, None].expand(
                -1,
                -1,
                n_samples
            ).reshape(batch_size, -1)

            pixel_index = y_pixel * image_size + x_pixel

            pixel_index = torch.where(
                inside,
                pixel_index,
                torch.zeros_like(pixel_index)
            )

            depth = torch.where(
                inside,
                depth,
                torch.full_like(depth, -torch.inf)
            )

            batch_offsets = (
                torch.arange(
                    batch_size,
                    device=device,
                    dtype=torch.long
                )[:, None]
                * n_pixels
            )

            global_pixel_index = pixel_index + batch_offsets

            zbuffer = torch.full(
                (batch_size * n_pixels,),
                -torch.inf,
                dtype=torch.float32,
                device=device
            )

            zbuffer.scatter_reduce_(
                dim=0,
                index=global_pixel_index.reshape(-1),
                src=depth.reshape(-1),
                reduce="amax",
                include_self=True
            )

            visible_depth = zbuffer[
                global_pixel_index
            ]

            frontmost = (
                inside
                & (depth >= visible_depth - 1e-5)
            )

            brightness_front = torch.where(
                frontmost,
                brightness_points,
                torch.zeros_like(brightness_points)
            )

            image_flat = torch.zeros(
                batch_size * n_pixels,
                dtype=torch.int64,
                device=device
            )

            image_flat.scatter_reduce_(
                dim=0,
                index=global_pixel_index.reshape(-1),
                src=brightness_front.reshape(-1),
                reduce="amax",
                include_self=True
            )

            image = image_flat.reshape(
                batch_size,
                image_size,
                image_size
            ).to(torch.uint8)

            return image

        light_direction = torch.tensor(
            LIGHT_DIRECTION,
            device=device,
            dtype=torch.float32
        )

        light_direction = torch.nn.functional.normalize(
            light_direction,
            dim=0
        )

        pixel_world_size = ORTHO_SCALE / IMAGE_SIZE
        pixel_world_area = pixel_world_size ** 2

        frame_ids_all = torch.arange(
            FRAMES,
            device=device,
            dtype=torch.long
        ).repeat_interleave(n_cameras)

        camera_ids_all = torch.arange(
            n_cameras,
            device=device,
            dtype=torch.long
        ).repeat(FRAMES)

        n_combinations = len(frame_ids_all)

        @torch.inference_mode()
        def rotated_face_normals(frame_ids):
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


        @torch.inference_mode()
        def calculate_projected_values(frame_ids, camera_ids):
            normals_world = rotated_face_normals(
                frame_ids
            )

            view_directions = camera_directions[
                camera_ids
            ]

            view_cosine = torch.sum(
                normals_world * view_directions[:, None, :],
                dim=-1
            )

            visible_factor = view_cosine.clamp_min(0.0)

            projected_area = (
                face_areas[None, :]
                * visible_factor
            )

            projected_pixels = projected_area / pixel_world_area

            light_cosine = torch.sum(
                normals_world * light_direction[None, None, :],
                dim=-1
            )

            diffuse = (
                light_cosine.clamp_min(0.0)
                * LIGHT_INTENSITY
            ).clamp(0.0, 1.0)

            diffuse = diffuse * (
                visible_factor > 0.0
            ).float()

            brightness_uint8 = torch.round(
                diffuse * 255.0
            ).to(torch.uint8)

            return brightness_uint8, projected_pixels

        reference_frame_ids = torch.zeros(
            n_cameras,
            dtype=torch.long,
            device=device
        )

        reference_camera_ids = torch.arange(
            n_cameras,
            dtype=torch.long,
            device=device
        )

        reference_brightness, reference_projected_pixels = (
            calculate_projected_values(
                reference_frame_ids,
                reference_camera_ids
            )
        )

        histogram = torch.zeros(
            (n_cameras, 256),
            dtype=torch.float32,
            device=device
        )

        offsets = (
            torch.arange(
                n_cameras,
                device=device,
                dtype=torch.long
            )[:, None]
            * 256
        )

        histogram_flat = torch.zeros(
            n_cameras * 256,
            dtype=torch.float32,
            device=device
        )

        histogram_flat.scatter_add_(
            dim=0,
            index=(
                reference_brightness.long()
                + offsets
            ).reshape(-1),
            src=reference_projected_pixels.reshape(-1)
        )

        histogram = histogram_flat.reshape(
            n_cameras,
            256
        )

        total_image_pixels = float(
            IMAGE_SIZE * IMAGE_SIZE
        )

        object_pixels = reference_projected_pixels.sum(
            dim=1
        ).clamp(
            min=0.0,
            max=total_image_pixels
        )

        background_pixels = (
            total_image_pixels - object_pixels
        ).clamp_min(0.0)

        histogram[:, 0] += background_pixels


        reference_frame_ids = torch.zeros(
            n_cameras,
            dtype=torch.long,
            device=device
        )

        reference_camera_ids = torch.arange(
            n_cameras,
            dtype=torch.long,
            device=device
        )

        reference_images = render_concave_zbuffer(
            reference_frame_ids,
            reference_camera_ids
        )

        thresholds = otsu_threshold_batch_uint8(
            reference_images
        )

        brightness_unique = torch.empty(
            (FRAMES, n_cameras),
            dtype=torch.int64,
            device=device
        )

        for start in range(0, n_combinations, CONCAVE_BATCH_SIZE):
            end = min(
                start + CONCAVE_BATCH_SIZE,
                n_combinations
            )

            current_frame_ids = frame_ids_all[start:end]
            current_camera_ids = camera_ids_all[start:end]

            gray_images = render_concave_zbuffer(
                current_frame_ids,
                current_camera_ids
            )

            current_thresholds = thresholds[
                current_camera_ids
            ]

            binary_images = (
                gray_images
                > current_thresholds[:, None, None]
            )

            counts = binary_images.sum(
                dim=(1, 2)
            ).to(torch.int64)

            brightness_unique[
                current_frame_ids,
                current_camera_ids
            ] = counts

            if start % max(CONCAVE_BATCH_SIZE * 10, 1) == 0:
                progress = 100.0 * end / n_combinations

        torch.cuda.synchronize()

        brightness_unique_np = brightness_unique.cpu().numpy()

        if KEEP_DUPLICATE_MID_CAMERA:
            columns = []

            for direction_index in range(7):
                index = direction_index * 3

                mid = brightness_unique_np[
                    :,
                    index:index + 1
                ]

                top = brightness_unique_np[
                    :,
                    index + 1:index + 2
                ]

                bottom = brightness_unique_np[
                    :,
                    index + 2:index + 3
                ]

                columns.extend(
                    [
                        mid,
                        mid.copy(),
                        top,
                        bottom
                    ]
                )

            brightness_values = np.concatenate(
                columns,
                axis=1
            )

        else:
            brightness_values = brightness_unique_np

        brightness_values = brightness_values.astype(np.float64)

        mean_per_camera = brightness_values.mean(
            axis=0,
            keepdims=True
        )

        mean_per_camera[mean_per_camera == 0] = 1.0

        brightness_normalized = brightness_values / mean_per_camera

        frame_column = np.arange(
            1,
            FRAMES + 1,
            dtype=np.int64
        )[:, None]

        result = np.concatenate(
            [frame_column, brightness_normalized],
            axis=1
        )

        formats = ["%d"] + ["%.8f"] * brightness_normalized.shape[1]

        asteroid=os.path.basename(stl_path)

        CSV_PATH=root / f"{folder}/lc_bin_{asteroid}.csv"

        CSV_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savetxt(
            CSV_PATH,
            result,
            delimiter=",",
            fmt=formats
        )

        if SAVE_DEBUG_NPZ:
            debug_path = CSV_PATH.with_suffix(".npz")

            np.savez(
                debug_path,
                thresholds=thresholds.cpu().numpy(),
                face_areas=face_areas.cpu().numpy(),
                face_normals=face_normals.cpu().numpy(),
                camera_positions=camera_positions.cpu().numpy()
            )

    end_time = time.perf_counter()
    print(
        f"Gesamtzeit: {end_time - start_time:.3f} Sekunden"
    )