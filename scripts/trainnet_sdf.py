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
from lightcurve_fips.data.datasets import AsteroidSDFPointDataset
from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet

start = time.time()
start2=start

R_max = 5.313693321295838

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = AsteroidSDFPointDataset("data/dataset2", n_points=8192)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

commence=0
if commence==1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("checkpoint1800_sdf.pth", map_location=device)

    model = LightcurveSDFNet(
        num_cameras=checkpoint["num_cameras"],
        latent_dim=checkpoint["latent_dim"],
        num_freqs=6
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] +1
    model.train()
else:

    # Beispielwerte anpassen
    num_cameras = 28
    frames = 360
    resolution = 64

    model = LightcurveSDFNet(num_cameras=28, latent_dim=256, num_freqs=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0

loss_fn = nn.SmoothL1Loss()

for epoch in range(start_epoch, start_epoch+1000):
    model.train()
    total_loss = 0

    for lc, points, sdf, radius in loader:
        lc = lc.to(device)
        points = points.to(device)
        sdf = sdf.to(device)
        radius = radius.to(device)

        pred_sdf = model(lc, points, radius)

        loss_raw = torch.nn.functional.smooth_l1_loss(pred_sdf, sdf, reduction="none")
        weights = 1.0 + 5.0 * (torch.abs(sdf) < 0.2).float()
        loss = (weights * loss_raw).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch%100==0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(loader),
            "num_cameras": 28,
            "latent_dim": 256,
            "R_max": R_max
        }, f"checkpoints/sdf/checkpoint{epoch}_sdf.pth")
        print("Checkpoint saved!")
    
    end2 = time.time()
    length = end2 - start2

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.6f},",length, "seconds!")
    with open("checkpoints/sdf/losses.txt", "a") as myfile:
        myfile.write(f"Epoch: {epoch} Loss: {total_loss / len(loader):.6f} \n")
    start2=end2