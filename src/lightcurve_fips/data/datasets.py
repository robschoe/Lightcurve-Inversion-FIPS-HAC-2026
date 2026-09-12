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
from lightcurve_fips.data.lightcurves import load_lightcurve
from lightcurve_fips.training.utils import parse_radius_from_stl

R_max = 6

def get_batch_number(sample_folder):
    """Simple function to get batch id in data"""
    match = re.search(
        r"batch(\d+)",
        sample_folder.parent.name
    )

    if match is None:
        raise ValueError(
            f"No batchnumber found: {sample_folder}"
        )

    return int(match.group(1))

class AsteroidSDFPointDatasetCombined(Dataset):
    """Dataset for lightcurves, SDF query points, SDF values, and radii."""

    def __init__(self, root, n_points=8192, max_samples=None, random_sampling=True):

        # Resolve the dataset path relative to the project root.
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.root = PROJECT_ROOT / root
        self.random_sampling = random_sampling
        self.n_points = n_points
        self.samples = []

        #Search for all sample directories.
        for sample_folder in sorted(self.root.rglob("sample*")):
            if not sample_folder.is_dir():
                continue

            #Each sample needs ground-truth STL file.
            has_stl = any(sample_folder.glob("asteroid*.stl"))

            if not has_stl:
                continue

            #Require both lightcurve variants.
            has_lc_bin = any(sample_folder.glob("lc_bin*.csv"))
            has_lc_intens = any(sample_folder.glob("lc_intens*.csv"))

            if not has_lc_bin or not has_lc_intens:
                continue

            #Make sure at least one set of points and SDF values exists.
            point_files = list(
                sample_folder.glob(
                    f"points_{self.n_points}_*.npy"
                )
            )

            sdf_files = list(
                sample_folder.glob(
                    f"sdf_{self.n_points}_*.npy"
                )
            )

            if not point_files or not sdf_files:
                continue

            stl_path = sorted(sample_folder.glob("asteroid*.stl"))[0]

            self.samples.append({
                "folder": sample_folder,
                "radius_real": parse_radius_from_stl(stl_path),
            })

            #Stop if wanted maximum of samples is reached.
            if (
                max_samples is not None
                and len(self.samples) >= max_samples
            ):
                break

        print(f"Found {len(self.samples)} valid samples")

    def __len__(self):
        """Return the number of valid dataset samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one lightcurve sample and one associated SDF point subset."""

        sample = self.samples[idx]

        folder = sample["folder"]
        radius_real = sample["radius_real"]

        csv_path_bin = sorted(folder.glob("lc_bin*.csv"))[0]
        csv_path_intens = sorted(folder.glob("lc_intens*.csv"))[0]

        lc_bin = load_lightcurve(csv_path_bin)
        lc_intens = load_lightcurve(csv_path_intens)
        lc = np.stack([lc_bin, lc_intens], axis=0)

        #Ignore incomplete temporary files.
        point_files = sorted(
            path
            for path in folder.glob(f"points_{self.n_points}_*.npy")
            if "tmp" not in path.stem
        )

        sdf_files = sorted(
            path
            for path in folder.glob(f"sdf_{self.n_points}_*.npy")
            if "tmp" not in path.stem
        )

        if not point_files:
            raise FileNotFoundError(
                f"No point files found in {folder}. "
                f"Expected pattern: points2_{self.n_points}_*.npy"
            )

        if not sdf_files:
            raise FileNotFoundError(
                f"No SDF files found in {folder}. "
                f"Expected pattern: sdf2_{self.n_points}_*.npy"
            )

        if len(point_files) != len(sdf_files):
            raise ValueError(
                f"Different numbers of point and SDF files in {folder}: "
                f"{len(point_files)} vs. {len(sdf_files)}"
            )

        if self.random_sampling:
            k = np.random.randint(len(point_files))
        else:
            k = 0

        points = np.load(point_files[k]).astype(np.float32)
        sdf = np.load(sdf_files[k]).astype(np.float32)

        #Normalize the physical radius for the neural network input.
        radius_input = radius_real / R_max

        lc = np.transpose(lc, (0, 2, 1))

        lc = torch.from_numpy(lc)
        points = torch.from_numpy(points)
        sdf = torch.from_numpy(sdf)
        radius = torch.tensor(radius_input, dtype=torch.float32)

        return lc, points, sdf, radius