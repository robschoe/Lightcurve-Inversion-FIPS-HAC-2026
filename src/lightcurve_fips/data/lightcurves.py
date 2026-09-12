import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import measure
import re
import os
import time
import torch.nn.functional as F
import math

def load_lightcurve(csv_path):
    """
    Load given lightcurve csv or txt file and return 21 normalized camera channels.

    The first column is ignored since it only contains the frame. The duplicate middle view
    of each direction is removed, leaving 21 channels.

    Each camera channel is normalized by its mean over all frames.    
    """
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    #Drop first non-camera column, the frame index.
    values = values[:, 1:]

    if values.shape[1] != 28:
        raise ValueError(
            f"28 cameracolumns expected, got: {values.shape[1]}"
        )

    #Array to keep track where relevant data is located and where the duplicate is stored.
    keep_indices = []

    for direction in range(7):
        start = direction * 4

        keep_indices.extend([
            start,       #Mid View 1
            start + 2,   #Top View
            start + 3,   #Bottom View
        ])

    values = values[:, keep_indices]

    #Normalize each camera lightcurve by its temporal mean.
    values = values / (
        values.mean(axis=0, keepdims=True) + 1e-8
    )

    return values

@torch.inference_mode()
def render_concave_zbuffer(
    frame_ids, 
    camera_ids, 
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
):
    """Render rotating concave meshes with orthographic z-buffering."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-GPU found. This script needs CUDA-PyTorch."
        )

    device = torch.device("cuda:0")

    batch_size = len(frame_ids)
    n_faces = face_sample_points.shape[0]
    n_samples = face_sample_points.shape[1]
    image_size = ZBUFFER_IMAGE_SIZE

    #Calculate one rotation angle per frame.
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

    #Rotate all surface points around the z-axis.
    rotated_x = cosine[:, None, None] * px - sine[:, None, None] * py
    rotated_y = sine[:, None, None] * px + cosine[:, None, None] * py
    rotated_z = pz.expand(batch_size, -1, -1)

    #Similarly, the face normals are rotated.
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

    #Calculate the Lambertian brightness for each face.
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

    #Depending on batch, select camera settings.
    view = camera_views[camera_ids]
    right = camera_rights[camera_ids]
    up = camera_ups[camera_ids]

    #Project rotated points onto the orthographic camera plane. 
    x_camera = (
        rotated_x * right[:, None, None, 0]
        + rotated_y * right[:, None, None, 1]
        + rotated_z * right[:, None, None, 2]
    )

    y_camera = (
        rotated_x * up[:, None, None, 0]
        + rotated_y * up[:, None, None, 1]
        + rotated_z * up[:, None, None, 2]
    )

    depth = (
        rotated_x * view[:, None, None, 0]
        + rotated_y * view[:, None, None, 1]
        + rotated_z * view[:, None, None, 2]
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

    #Keep only samples that project inside the image bounds.
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

    #Assign face brightness to each sample point on that face.
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

    #Hidden samples contribute no brightness.
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