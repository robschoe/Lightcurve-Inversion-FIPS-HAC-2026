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

R_max = 5.313693321295838

class AsteroidSDFPointDataset(Dataset):
    def __init__(self, root, n_points=8192):
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        DATASET_ROOT = PROJECT_ROOT / root
        self.root = DATASET_ROOT
        self.n_points = n_points

        self.samples = sorted([
            p for p in self.root.rglob("sample*")
            if list(p.glob("brightness*.csv"))
            and list(p.glob(f"points_{n_points}_*.npy"))
            and list(p.glob(f"sdf_{n_points}_*.npy"))
        ])

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        csv_path = sorted(folder.glob("brightness*.csv"))[0]

        lc = load_lightcurve(csv_path)

        point_files = sorted(folder.glob(f"points_{self.n_points}_*.npy"))
        sdf_files = sorted(folder.glob(f"sdf_{self.n_points}_*.npy"))

        k = np.random.randint(len(point_files))

        points = np.load(point_files[k]).astype(np.float32)
        sdf = np.load(sdf_files[k]).astype(np.float32)

        stl_path = sorted(folder.glob("asteroid*.stl"))[0]

        radius_real = parse_radius_from_stl(stl_path)

        radius_input = radius_real / R_max

        lc = torch.tensor(lc.T, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)
        sdf = torch.tensor(sdf, dtype=torch.float32)
        radius = torch.tensor(radius_input, dtype=torch.float32)

        return lc, points, sdf, radius

class AsteroidDataset(Dataset):
    def __init__(self, root, resolution=32):
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        DATASET_ROOT = PROJECT_ROOT / root
        self.root = DATASET_ROOT
        self.resolution = resolution

        self.samples = sorted([
            p for p in self.root.rglob("sample*")
            if list(p.glob("brightness*.csv"))
            and list(p.glob("voxels.npy"))
        ])

        print(f"Found {len(self.samples)} samples in {self.root}")
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {self.root}. "
                "Expected folders like sample_00000/lightcurve.csv and shape.stl"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        lc = load_lightcurve(list(folder.glob("brightness*.csv"))[0])
        vox = np.load(folder / "voxels.npy")
        radius=re.search("radius(.*).stl",os.path.basename(list(folder.glob("brightness*.csv"))[0]))

        # Lightcurve: (frames, cameras) → (cameras, frames)
        lc = torch.tensor(lc.T, dtype=torch.float32)

        # Voxels: (D,H,W)
        vox = torch.tensor(vox, dtype=torch.float32)

        radius = torch.tensor(float(radius.group(1)), dtype=torch.float32)

        return lc, vox, radius