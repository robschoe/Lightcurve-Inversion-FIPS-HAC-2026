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

checkpoint = torch.load(CHECKPOINT_DIR/"checkpoint6400_sdf.pth", map_location=device)

R_max = 5.313693321295838

model = LightcurveSDFNet(
    num_cameras=checkpoint["num_cameras"],
    latent_dim=checkpoint["latent_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

asteroid="data/dataset2/test/brightnessasteroid2_scaled_radius1.4142135623730951.stl.csv"
lc = load_lightcurve(PROJECT_ROOT/asteroid)
lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", asteroid)
radius_value = float(m.group(1))

radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

sdf = reconstruct_sdf(model, lc, radius_model, res=32)

sdf_to_stl(
    sdf,
    PROJECT_ROOT/"data/dataset2/test/asteroid_sdf_reconstruction.stl",
    radius=radius_value
)