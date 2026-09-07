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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sdf"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = PROJECT_ROOT / "data" / "dataset2" / "test" / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_DIR/"checkpoint80_sdf_69k_net2_combined_residual_beta0_1_newsdf.pth", map_location=device)

R_max = 6

model = LightcurveSDFNet(
    num_cameras=checkpoint["num_cameras"],
    latent_dim=checkpoint["latent_dim"],
    num_freqs=checkpoint["num_freqs"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
asteroids = [
    path
    for path in DATASET_DIR.glob("asteroid*.stl")
    if "_reconstruction" not in path.stem
]
for asteroid in asteroids:
    asteroidname=os.path.basename(asteroid)
    asteroidbin=f"lc_bin_{asteroidname}.csv"
    asteroidint=f"lc_intens_{asteroidname}.csv"
    lc_bin = load_lightcurve(DATASET_DIR/asteroidbin)
    lc_intens = load_lightcurve(DATASET_DIR/asteroidint)
    # m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", asteroid)
    # radius_value = float(m.group(1))
    lc = np.stack([lc_bin, lc_intens], axis=0)

    lc = np.transpose(lc, (0, 2, 1))

    lc = torch.tensor(lc, dtype=torch.float32,device=device)

    lc = lc.unsqueeze(0)

    radius_value=parse_radius_from_stl(asteroid)

    radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

    with torch.inference_mode():
        sdf = reconstruct_sdf(
            model=model,
            lc=lc,
            radius=radius_value / checkpoint["R_max"],
            grid_extent=1.0,
            res=64,
            device=device,
        )

    mesh = sdf_to_stl(
        sdf=sdf,
        out_path=DATASET_DIR / f"{asteroid}_reconstruction.stl",
        radius=radius_value,
        grid_extent=1.0,
    )