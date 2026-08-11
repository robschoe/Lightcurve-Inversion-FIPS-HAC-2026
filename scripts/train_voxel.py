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

start = time.time()

R_max = 5.313693321295838

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AsteroidDataset("dataset", resolution=64)
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

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss / len(loader),
        "num_cameras": num_cameras,
        "frames": frames,
        "resolution": resolution,
        "R_max": R_max
    }, "checkpoint.pth")

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")

model.eval()

# checkpoint = torch.load("checkpoint.pth", map_location=device)

# model = LightcurveToVoxelNet(
#     num_cameras=checkpoint["num_cameras"],
#     frames=checkpoint["frames"],
#     resolution=checkpoint["resolution"]
# ).to(device)

# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()
# R_max = checkpoint["R_max"]

lc = load_lightcurve("C:/Users/rober/Python Datengenerierung/brightnessasteroid5radius2.0475867806576935.stl.csv")
lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
radius = torch.tensor([2.0475867806576935], dtype=torch.float32, device=device)

with torch.no_grad():
    pred = model(lc,radius)
    mask = cylinder_mask(resolution, radius, device)
    pred = pred.masked_fill(mask == 0, -20.0)

    pred = torch.sigmoid(pred)

voxels = pred[0].cpu().numpy()

voxels_to_stl(voxels, "predicted_asteroid.stl", threshold=0.5)

end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")