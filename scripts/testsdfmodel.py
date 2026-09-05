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
from lightcurve_fips.training.utils import (reconstruct_sdf,sdf_to_stl)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sdf"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_DIR/"best_by_voxel_score_sdf_combined.pth", map_location=device)

R_max = 5.313693321295838

model = LightcurveSDFNet(
    num_cameras=21,
    latent_dim=checkpoint["latent_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

asteroid="data/publicasteroids/Asteroid02_lightcurve_binary_blender.txt"
lc_bin = load_lightcurve(PROJECT_ROOT/asteroid)
asteroid="data/publicasteroids/Asteroid02_lightcurve_intensity_blender.txt"
lc_intens = load_lightcurve(PROJECT_ROOT/asteroid)
# m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", asteroid)
# radius_value = float(m.group(1))
lc = np.stack([lc_bin, lc_intens], axis=0)

lc = np.transpose(lc, (0, 2, 1))

lc = torch.tensor(lc, dtype=torch.float32,device=device)

lc = lc.unsqueeze(0)

radius_value=0.88

radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

sdf = reconstruct_sdf(model, lc, radius_model, res=32)

sdf_to_stl(
    sdf,
    PROJECT_ROOT/"data/dataset2/test/asteroid_sdf_reconstruction.stl",
    radius=radius_value
)