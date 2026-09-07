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

def compare_lightcurves(
    lc_true,
    lc_pred,
    remove_frame_column=True,
    allow_circular_shift=True,
    normalize_per_camera=True,
):
    lc_true = np.asarray(lc_true, dtype=np.float64)
    lc_pred = np.asarray(lc_pred, dtype=np.float64)

    if remove_frame_column:
        lc_true = lc_true[:, 1:]
        lc_pred = lc_pred[:, 1:]

    if lc_true.shape != lc_pred.shape:
        raise ValueError(
            f"Lightcurve shapes differ: "
            f"true={lc_true.shape}, pred={lc_pred.shape}"
        )

    if lc_true.ndim != 2:
        raise ValueError(
            "Expected shape (frames, cameras)."
        )

    valid = np.isfinite(lc_true) & np.isfinite(lc_pred)

    if not np.all(valid):
        print(
            f"Warning: {(~valid).sum()} invalid values found. "
            "They are replaced by 0."
        )

        lc_true = np.where(np.isfinite(lc_true), lc_true, 0.0)
        lc_pred = np.where(np.isfinite(lc_pred), lc_pred, 0.0)

    if normalize_per_camera:
        true_mean = np.mean(lc_true, axis=0, keepdims=True)
        pred_mean = np.mean(lc_pred, axis=0, keepdims=True)

        true_std = np.std(lc_true, axis=0, keepdims=True)
        pred_std = np.std(lc_pred, axis=0, keepdims=True)

        true_std = np.maximum(true_std, 1e-8)
        pred_std = np.maximum(pred_std, 1e-8)

        true_compare = (lc_true - true_mean) / true_std
        pred_compare = (lc_pred - pred_mean) / pred_std
    else:
        true_compare = lc_true.copy()
        pred_compare = lc_pred.copy()

    n_frames, n_cameras = true_compare.shape

    best_shift = 0
    best_rmse = np.inf
    best_pred = pred_compare

    if allow_circular_shift:
        for shift in range(n_frames):
            shifted_pred = np.roll(pred_compare, shift, axis=0)

            rmse = np.sqrt(
                np.mean((true_compare - shifted_pred) ** 2)
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_shift = shift
                best_pred = shifted_pred
    else:
        best_pred = pred_compare
        best_rmse = np.sqrt(
            np.mean((true_compare - best_pred) ** 2)
        )

    rmse_per_camera = np.sqrt(
        np.mean((true_compare - best_pred) ** 2, axis=0)
    )

    mae_per_camera = np.mean(
        np.abs(true_compare - best_pred),
        axis=0
    )

    corr_per_camera = np.zeros(n_cameras)

    for cam in range(n_cameras):
        y_true = true_compare[:, cam]
        y_pred = best_pred[:, cam]

        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            corr_per_camera[cam] = np.nan
        else:
            corr_per_camera[cam] = np.corrcoef(
                y_true,
                y_pred
            )[0, 1]

    mean_corr = np.nanmean(corr_per_camera)
    mean_rmse = np.mean(rmse_per_camera)
    mean_mae = np.mean(mae_per_camera)

    objective = mean_rmse + 0.5 * (1.0 - mean_corr)

    return {
        "objective": float(objective),
        "best_shift": int(best_shift),

        "mean_rmse": float(mean_rmse),
        "mean_mae": float(mean_mae),
        "mean_correlation": float(mean_corr),

        "rmse_per_camera": rmse_per_camera,
        "mae_per_camera": mae_per_camera,
        "correlation_per_camera": corr_per_camera,
    }