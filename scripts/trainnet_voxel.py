import trimesh
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage import measure
import re
import os
import time
from lightcurve_fips.data.datasets import AsteroidDataset
from lightcurve_fips.models.lightcurve_encoder import LightcurveToVoxelNet
from lightcurve_fips.training.utils import (dice_loss, cylinder_mask, voxels_to_stl)
from lightcurve_fips.data.lightcurves import load_lightcurve

start = time.time()
start2=start

R_max = 5.313693321295838

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AsteroidDataset("data/dataset", resolution=64)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Beispielwerte anpassen
num_cameras = 28
frames = 360
resolution = 64

model = LightcurveToVoxelNet(num_cameras, frames, resolution).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(2):
    print(epoch)
    total_loss = 0

    for lc, vox, radius in loader:
        lc = lc.to(device)
        radius = radius.to(device)
        vox = vox.to(device)

        pred = model(lc,radius)

        mask = cylinder_mask(resolution, radius, device)

        pred = pred.masked_fill(mask == 0, -20.0)

        prob = torch.sigmoid(pred)

        outside_loss = (prob * (1-mask)).mean()

        loss = loss_fn(pred, vox) + dice_loss(pred, vox) + 0.1 * outside_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch%10==0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(loader),
            "num_cameras": num_cameras,
            "frames": frames,
            "resolution": resolution,
            "R_max": R_max
        }, f"checkpoints/voxel/checkpoint{epoch}_voxel.pth")
        print("Checkpoint saved!")

    end2 = time.time()
    length = end2 - start2

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.6f}",length, "seconds!")
    with open("checkpoints/voxel/losses.txt", "a") as myfile:
        myfile.write(f"Epoch: {epoch} Loss: {total_loss / len(loader):.6f} \n")
    start2=end2

end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")