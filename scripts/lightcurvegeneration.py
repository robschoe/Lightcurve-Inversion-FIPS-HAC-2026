import math
import time
from pathlib import Path
import os
from tqdm.auto import tqdm

import numpy as np
import torch
import trimesh
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.training.utils import (parse_radius_from_stl, get_original_stl)
from lightcurve_fips.rendering.camera_setup import load_face_data, otsu_threshold_batch_uint8, create_unique_camera_positions, create_camera_axes, create_barycentric_samples, rotated_face_normals
from lightcurve_fips.data.lightcurves import render_concave_zbuffer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

USE_CONCAVE_ZBUFFER = True

IMAGE_SIZE = 256

ZBUFFER_IMAGE_SIZE = IMAGE_SIZE

SAMPLES_PER_FACE = 32

CONCAVE_BATCH_SIZE = 128

FRAMES = 360

LIGHT_DIRECTION = (-1.0, 0.0, 0.0)

LIGHT_INTENSITY = 3.5

KEEP_DUPLICATE_MID_CAMERA = True

BATCH_SIZE = 1024

SAVE_DEBUG_NPZ = False

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA-GPU found. This script needs CUDA-PyTorch."
    )

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU-Speicher:",
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

torch.set_grad_enabled(False)

start_time = time.perf_counter()

for batch in range(150):
    root = DATASET_DIR/f"batch{10+batch}"
    
    samples = sorted([
        p
        for p in root.rglob("sample*")
        if p.is_dir()
        and any(p.glob("asteroid*.stl"))
    ])
    
    asteroids = list(root.rglob("asteroid*.stl"))
    print("Found asteroids:", len(samples))
    for folder in tqdm(
        samples,
        desc=f"Batch {10 + batch}",
        unit="asteroid",
    ):
        asteroid_start = time.perf_counter()

        stl_path = get_original_stl(folder)

        if stl_path is None:
            continue

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

        camera_positions_np = create_unique_camera_positions(CAMERA_RADIUS)

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

        barycentric_samples = create_barycentric_samples(
            SAMPLES_PER_FACE,
            device
        )

        face_sample_points = torch.einsum(
            "fvc,sv->fsc",
            triangles,
            barycentric_samples
        )

        

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

        reference_parts = []

        REFERENCE_BATCH_SIZE = 1

        for start in range(0, n_cameras, REFERENCE_BATCH_SIZE):
            end = min(start + REFERENCE_BATCH_SIZE, n_cameras)

            reference_parts.append(
                render_concave_zbuffer(
                    reference_frame_ids[start:end],
                    reference_camera_ids[start:end], 
                    face_sample_points, 
                    ZBUFFER_IMAGE_SIZE, 
                    FRAMES, 
                    face_normals, 
                    light_direction, 
                    LIGHT_INTENSITY, 
                    ORTHO_SCALE, 
                    camera_views, 
                    camera_rights,
                    camera_ups
                )
            )

        reference_images = torch.cat(
            reference_parts,
            dim=0,
        )

        thresholds = otsu_threshold_batch_uint8(
            reference_images
        )

        brightness_bin = torch.empty(
            (FRAMES, n_cameras),
            dtype=torch.int64,
            device=device
        )

        brightness_int = torch.empty(
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
                current_camera_ids, 
                face_sample_points, 
                ZBUFFER_IMAGE_SIZE, 
                FRAMES, 
                face_normals, 
                light_direction,
                LIGHT_INTENSITY, 
                ORTHO_SCALE, 
                camera_views, 
                camera_rights,
                camera_ups
            )

            INTENSITY_THRESHOLD = 25.5

            gray_float = gray_images.float()

            intensities = (
                gray_float
                * (gray_float > INTENSITY_THRESHOLD)
            ).sum(dim=(1, 2))

            brightness_int[
                current_frame_ids,
                current_camera_ids
            ] = intensities.to(torch.int64)

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

            brightness_bin[
                current_frame_ids,
                current_camera_ids
            ] = counts

            if start % max(CONCAVE_BATCH_SIZE * 10, 1) == 0:
                progress = 100.0 * end / n_combinations

        torch.cuda.synchronize()

        brightness_bin_np = brightness_bin.cpu().numpy()
        brightness_int_np = brightness_int.cpu().numpy()

        if KEEP_DUPLICATE_MID_CAMERA:
            columns_int = []
            columns_bin = []

            for direction_index in range(7):
                index = direction_index * 3

                mid_bin = brightness_bin_np[
                    :,
                    index:index + 1
                ]

                top_bin = brightness_bin_np[
                    :,
                    index + 1:index + 2
                ]

                bottom_bin = brightness_bin_np[
                    :,
                    index + 2:index + 3
                ]

                mid_int = brightness_int_np[
                    :,
                    index:index + 1
                ]

                top_int = brightness_int_np[
                    :,
                    index + 1:index + 2
                ]

                bottom_int = brightness_int_np[
                    :,
                    index + 2:index + 3
                ]

                columns_bin.extend(
                    [
                        mid_bin,
                        mid_bin.copy(),
                        top_bin,
                        bottom_bin
                    ]
                )

                columns_int.extend(
                    [
                        mid_int,
                        mid_int.copy(),
                        top_int,
                        bottom_int
                    ]
                )

            brightness_values_bin = np.concatenate(
                columns_bin,
                axis=1
            )

            brightness_values_int = np.concatenate(
                columns_int,
                axis=1
            )

        else:
            brightness_values_int = brightness_int_np
            brightness_values_bin = brightness_bin_np

        brightness_values_int = brightness_values_int.astype(np.float64)
        brightness_values_bin = brightness_values_bin.astype(np.float64)

        mean_per_camera_bin = brightness_values_bin.mean(
            axis=0,
            keepdims=True
        )

        mean_per_camera_int = brightness_values_int.mean(
            axis=0,
            keepdims=True
        )

        mean_per_camera_int[mean_per_camera_int == 0] = 1.0
        mean_per_camera_bin[mean_per_camera_bin == 0] = 1.0

        brightness_normalized_bin = brightness_values_bin / mean_per_camera_bin
        brightness_normalized_int = brightness_values_int / mean_per_camera_int

        frame_column = np.arange(
            1,
            FRAMES + 1,
            dtype=np.int64
        )[:, None]

        result_int = np.concatenate(
            [frame_column, brightness_normalized_int],
            axis=1
        )

        result_bin = np.concatenate(
            [frame_column, brightness_normalized_bin],
            axis=1
        )

        formats_int = ["%d"] + ["%.8f"] * brightness_normalized_int.shape[1]
        formats_bin = ["%d"] + ["%.8f"] * brightness_normalized_bin.shape[1]

        asteroid=os.path.basename(stl_path)

        CSV_PATH_BIN=root / f"{folder}/lc_bin_{asteroid}.csv"

        CSV_PATH_BIN.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savetxt(
            CSV_PATH_BIN,
            result_bin,
            delimiter=",",
            fmt=formats_bin
        )

        CSV_PATH_INT=root / f"{folder}/lc_intens_{asteroid}.csv"
        
        CSV_PATH_INT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savetxt(
            CSV_PATH_INT,
            result_int,
            delimiter=",",
            fmt=formats_int
        )

        asteroid_time = time.perf_counter() - asteroid_start

    end_time = time.perf_counter()
    print(
        f"Time needed: {end_time - start_time:.3f} seconds"
    )