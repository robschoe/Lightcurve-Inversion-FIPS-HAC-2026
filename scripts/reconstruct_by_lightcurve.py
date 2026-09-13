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
import sys

from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet
from lightcurve_fips.data.lightcurves import load_lightcurve
from lightcurve_fips.training.utils import (reconstruct_sdf,sdf_to_stl,parse_radius_from_stl)
from lightcurve_fips.evaluation.evaluate import evaluate_geometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = PROJECT_ROOT / "data" / "secretasteroids"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load trained model.
checkpoint = torch.load(CHECKPOINT_DIR/"best_by_voxel_score_sdf_newsdf_freq8.pth", map_location=device)

R_max = checkpoint["R_max"]

#The radius of the object that is to be reconstructed.
RADIUS=3.95

PATH_TO_BINARY_LC = DATASET_DIR / "lightcurves" / "Asteroid010_lightcurve_binary_blender.txt"
PATH_TO_INTENSITY_LC = DATASET_DIR / "lightcurves" / "Asteroid010_lightcurve_intensity_blender.txt"

OUTPUT_STL = DATASET_DIR / "reconstructions"

GRID_EXTENT = 1.1

#Create model architecture stored in the checkpoint.
model = LightcurveSDFNet(
    num_cameras=checkpoint["num_cameras"],
    latent_dim=checkpoint["latent_dim"],
    num_freqs=checkpoint["num_freqs"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#Load binary and intensity lightcurves.
lc_bin = load_lightcurve(PATH_TO_BINARY_LC)
lc_intens = load_lightcurve(PATH_TO_INTENSITY_LC)

lc = np.stack([lc_bin, lc_intens], axis=0)
lc = np.transpose(lc, (0, 2, 1))
lc = torch.tensor(lc, dtype=torch.float32,device=device).unsqueeze(0)

#Normalize radius for model.
radius_model = torch.tensor([RADIUS / R_max], dtype=torch.float32, device=device)

#Evaluate SDF on a regular 3D grid.
sdf = reconstruct_sdf(
    model=model,
    lc=lc,
    radius=radius_model,
    grid_extent=GRID_EXTENT,
    res=512,
    device=device,
)

#Extract the SDF zero level set and export it as STL.
mesh = sdf_to_stl(
    sdf=sdf,
    out_path=OUTPUT_STL / "ASTEROID_RECONSTRUCTION.stl",
    radius=RADIUS,
    grid_extent=1.0,
)

if mesh is None:
    print("No valid mesh could be extracted because the SDF does not cross zero.")
else:
    print(f"Reconstruction saved to: {OUTPUT_STL}")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Faces: {len(mesh.faces)}")