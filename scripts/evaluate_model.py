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
import time

from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet
from lightcurve_fips.data.lightcurves import load_lightcurve
from lightcurve_fips.training.utils import (reconstruct_sdf,sdf_to_stl)
from lightcurve_fips.evaluation.evaluate import (make_watertight_with_pymeshfix,compare_stl_files)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sdf"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelname="best_by_voxel_score_sdf_99k_net1_avgpool8"

checkpoint = torch.load(CHECKPOINT_DIR/f"{modelname}.pth", map_location=device)

R_max = 5.313693321295838

model = LightcurveSDFNet(
    num_cameras=checkpoint["num_cameras"],
    latent_dim=checkpoint["latent_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

TEST_SET_DIR=PROJECT_ROOT / "data/dataset2/test"

asteroids = sorted([
            p for p in TEST_SET_DIR.rglob("sample*")
            if list(p.glob("brightness*.csv"))
        ])

print(f"Found {len(asteroids)} samples")

voxel_score_total=0
side_view_total=0
count=0

for asteroid in asteroids:
    start = time.time()
    count+=1
    csv_path = sorted(asteroid.glob("brightness*.csv"))[0]
    
    lc = load_lightcurve(csv_path)
    lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", csv_path.name)
    radius_value = float(m.group(1))

    radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

    sdf = reconstruct_sdf(model, lc, radius_model, res=32)

    recon_stl=PROJECT_ROOT/"data/dataset2/test/asteroid_sdf_reconstruction.stl"

    sdf_to_stl(
        sdf,
        recon_stl,
        radius=radius_value
    )

    recon_mesh = trimesh.load(recon_stl, force="mesh")

    if recon_mesh is None:
        print(f"Skipping {csv_path.name}: no zero crossing")
        continue

    if not recon_mesh.is_watertight:
        repaired_stl = PROJECT_ROOT / "data/dataset2/test/reconstruction_watertight.stl"

        make_watertight_with_pymeshfix(
            recon_stl,
            repaired_stl
        )

        recon_stl = repaired_stl

    true_stl = sorted(asteroid.glob("asteroid*.stl"))[0]

    vox, side = compare_stl_files(
        true_stl,
        recon_stl,
        
        voxel_resolution=64,
        projection_resolution=128,
        per_axis_normalization=False
    )

    epoch_time = time.time() - start

    # print("\n--- Voxel-based measure ---")
    #for key, value in vox.items():
    #     print(f"{key}: {value}")
    voxel_score=vox.get("voxel_score")
    voxel_score_total+=voxel_score

    # print("\n--- Side-view measure ---")
    # print("mean_boundary_distance_px:", side["mean_boundary_distance_px"])
    # print("mean_boundary_distance_normalized:", side["mean_boundary_distance_normalized"])
    # print("mean_boundary_similarity:", side["mean_boundary_similarity"])
    side_score=side["mean_boundary_similarity"]
    side_view_total+=side_score
    print(count,"Mean side view score: ",side_view_total/count, "Mean Voxelscore:",voxel_score_total/count, "Zeit:",epoch_time)
    with open(f"data/dataset2/test/modelevaluation/{modelname}_evaluation.txt", "a") as f:
        f.write(f"Asteroid {count}: Side view score: {side_score} Voxelscore: {voxel_score} \n")

with open(f"data/dataset2/test/modelevaluation/{modelname}_evaluation.txt", "a") as f:
    f.write(f"Mean side view score: {side_view_total/count} Mean Voxelscore: {voxel_score_total/count} \n")