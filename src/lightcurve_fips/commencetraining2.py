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

R_max = 5.313693321295838

class LightcurveToVoxelNet(nn.Module):
    def __init__(self, num_cameras, frames, resolution=32):
        super().__init__()

        self.resolution = resolution

        self.encoder = nn.Sequential(
            nn.Conv1d(num_cameras, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 64 + 1, 256 * 4 * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256,128,4,stride=2,padding=1), # 8³
            nn.ReLU(),
            nn.ConvTranspose3d(128,64,4,stride=2,padding=1),  # 16³
            nn.ReLU(),
            nn.ConvTranspose3d(64,32,4,stride=2,padding=1),   # 32³
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),  # 32³ -> 64³
            nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=1)
        )

    def forward(self, x, radius):
        x = self.encoder(x)

        radius= radius.view(-1,1)

        x=torch.cat([x,radius], dim=1)

        x = self.fc(x)
        x = x.view(-1,256,4,4,4)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x

class AsteroidDataset(Dataset):
    def __init__(self, root, resolution=32):
        self.root = Path("C:/Users/rober/Python Datengenerierung/dataset")
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


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)

    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = 1 - (2 * intersection + eps) / (union + eps)

    return dice.mean()

def cylinder_mask(resolution, radius, device):
    batch_size = radius.shape[0]

    xy = torch.linspace(-R_max, R_max, resolution, device=device)
    z = torch.linspace(-1, 1, resolution, device=device)

    x, y, z = torch.meshgrid(xy, xy, z, indexing="ij")

    r_grid = (x**2 + y**2).unsqueeze(0)

    radius = radius.view(batch_size, 1, 1, 1)

    mask = r_grid <= radius**2

    return mask.float()

def load_lightcurve(csv_path):
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    # Falls erste Spalte Framezahl ist:
    values = values[:, 1:]

    values = values / (values.mean(axis=0, keepdims=True) + 1e-8)

    return values


def stl_to_voxels(stl_path, resolution=64, R_max=5.313693321295838):
    mesh = trimesh.load(stl_path)

    # Optional: falls Mesh als Scene geladen wird
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            geom for geom in mesh.geometry.values()
        ])

    # Voxelgrid
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Vertices
    vertices = mesh.vertices.copy()

    # Bounding-Box des globalen Raums
    x_min, x_max = -R_max, R_max
    y_min, y_max = -R_max, R_max
    z_min, z_max = -1.0, 1.0

    # Mesh voxelisieren
    # Pitch orientiert sich an kleinster Zellgröße
    pitch_x = (x_max - x_min) / resolution
    pitch_y = (y_max - y_min) / resolution
    pitch_z = (z_max - z_min) / resolution

    pitch = min(pitch_x, pitch_y, pitch_z)

    voxelized = mesh.voxelized(pitch=pitch).fill()
    points = voxelized.points

    # Punkte in Grid-Indizes umrechnen
    ix = ((points[:,0] - x_min) / (x_max - x_min) * resolution).astype(int)
    iy = ((points[:,1] - y_min) / (y_max - y_min) * resolution).astype(int)
    iz = ((points[:,2] - z_min) / (z_max - z_min) * resolution).astype(int)

    # Nur gültige Punkte
    valid = (
        (ix >= 0) & (ix < resolution) &
        (iy >= 0) & (iy < resolution) &
        (iz >= 0) & (iz < resolution)
    )

    grid[ix[valid], iy[valid], iz[valid]] = 1.0

    return grid

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AsteroidDataset("dataset", resolution=64)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

checkpoint = torch.load("checkpoint.pth", map_location=device)

num_cameras = checkpoint["num_cameras"]
frames = checkpoint["frames"]
resolution = checkpoint["resolution"]
R_max = checkpoint["R_max"]

model = LightcurveToVoxelNet(num_cameras, frames, resolution).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] +1

loss_fn = nn.BCEWithLogitsLoss()

# for epoch in range(start_epoch, start_epoch+50):
#     print(epoch)
#     total_loss = 0

#     for lc, vox, radius in loader:
#         lc = lc.to(device)
#         radius = radius.to(device)
#         vox = vox.to(device)

#         pred = model(lc,radius)

#         mask = cylinder_mask(resolution, radius, device)

#         pred = pred.masked_fill(mask == 0, -20.0)

#         prob = torch.sigmoid(pred)

#         outside_loss = (prob * (1-mask)).mean()

#         loss = loss_fn(pred, vox) + dice_loss(pred, vox) + 0.1 * outside_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")
    
#     torch.save({
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "loss": total_loss / len(loader),
#         "num_cameras": num_cameras,
#         "frames": frames,
#         "resolution": resolution,
#         "R_max": R_max
#     }, "checkpoint.pth")

def voxels_to_stl(voxels, out_path, threshold=0.5, R_max=5.313693321295838):
    verts, faces, normals, values = measure.marching_cubes(voxels, level=threshold)

    res = voxels.shape[0]

    # Indexraum -> echter Koordinatenraum
    verts[:,0] = verts[:,0] / (res - 1) * (2 * R_max) - R_max
    verts[:,1] = verts[:,1] / (res - 1) * (2 * R_max) - R_max
    verts[:,2] = verts[:,2] / (res - 1) * 2.0 - 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(out_path)

model.eval()

lc = load_lightcurve("C:/Users/rober/Python Datengenerierung/brightnessasteroid52radius0.583516496508435.stl.csv")
lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
radius = torch.tensor([0.583516496508435], dtype=torch.float32, device=device)

with torch.no_grad():
    pred = model(lc,radius)
    mask = cylinder_mask(resolution, radius, device)
    pred = pred.masked_fill(mask == 0, -20.0)

    pred = torch.sigmoid(pred)

voxels = pred[0].cpu().numpy()

voxels_to_stl(voxels, "predicted_asteroid.stl", threshold=0.5)