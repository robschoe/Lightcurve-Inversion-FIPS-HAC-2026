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

class LightcurveEncoder(nn.Module):
    def __init__(self, num_cameras, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_cameras, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

    def forward(self, lc):
        return self.net(lc)

class OccupancyDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, latent, points, radius):
        B, N, _ = points.shape

        if radius.dim() == 0:
            radius = radius.unsqueeze(0)

        radius = radius.view(B)

        latent_expanded = latent[:, None, :].expand(-1, N, -1)
        radius_expanded = radius[:, None, None].expand(-1, N, 1)

        x = torch.cat([latent_expanded, points, radius_expanded], dim=-1)

        logits = self.net(x)

        return logits.squeeze(-1)

class LightcurveOccupancyNet(nn.Module):
    def __init__(self, num_cameras, latent_dim=256):
        super().__init__()

        self.encoder = LightcurveEncoder(num_cameras, latent_dim)
        self.decoder = OccupancyDecoder(latent_dim)

    def forward(self, lc, points, radius):
        latent = self.encoder(lc)
        logits = self.decoder(latent, points, radius)
        return logits

def sample_points(batch_size, n_points, device):
    points = torch.rand(batch_size, n_points, 3, device=device)

    # in [-1,1]^3 transformieren
    points = points * 2 - 1

    return points

def sample_occupancy_from_mesh(stl_path, radius, n_points=8192):
    mesh = trimesh.load(stl_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    mesh = mesh.copy()

    # Radius-normalisieren
    mesh.vertices[:, 0] /= radius
    mesh.vertices[:, 1] /= radius

    n_surface = n_points // 2
    n_uniform = n_points - n_surface

    # Punkte auf der Oberfläche
    surface_points = mesh.sample(n_surface)

    # kleine Störung um die Oberfläche
    noise = np.random.normal(scale=0.03, size=surface_points.shape)
    near_surface = surface_points + noise

    # zufällige Punkte im Raum
    uniform_points = np.random.uniform(-1, 1, size=(n_uniform, 3))

    points = np.concatenate([near_surface, uniform_points], axis=0)

    # auf [-1,1] clippen
    points = np.clip(points, -1, 1)

    inside = mesh.contains(points).astype(np.float32)

    return points.astype(np.float32), inside

class AsteroidOccupancyDataset(Dataset):
    def __init__(self, root, n_points=8192):
        self.root = Path("C:/Users/rober/Python Datengenerierung/dataset/batch1")
        self.n_points = n_points

        self.samples = sorted([
            p for p in self.root.rglob("sample*")
            if list(p.glob("brightness*.csv"))
            and list(p.glob("asteroid*.stl"))
        ])

        print("Found", len(self.samples), "samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        csv_path = sorted(folder.glob("brightness*.csv"))[0]
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]

        lc = load_lightcurve(csv_path)

        radius_match = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", stl_path.name)
        radius_real = float(radius_match.group(1))
        radius_input = radius_real / R_max

        radius = torch.tensor(radius_input, dtype=torch.float32)

        points, occ = sample_occupancy_from_mesh(
            stl_path,
            radius,
            n_points=self.n_points
        )

        lc = torch.tensor(lc.T, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)
        occ = torch.tensor(occ, dtype=torch.float32)
        radius = torch.tensor(radius, dtype=torch.float32)

        return lc, points, occ, radius

def load_lightcurve(csv_path):
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    # Falls erste Spalte Framezahl ist:
    values = values[:, 1:]

    values = values / (values.mean(axis=0, keepdims=True) + 1e-8)

    return values

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AsteroidOccupancyDataset("dataset", n_points=8192)
loader = DataLoader(dataset, batch_size=1, shuffle=True,num_workers=0)

model = LightcurveOccupancyNet(num_cameras=28).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(0):
    model.train()
    total_loss = 0

    for lc, points, occ, radius in loader:
        lc = lc.to(device)
        points = points.to(device)
        occ = occ.to(device)
        radius = radius.to(device)

        pred = model(lc, points, radius)

        loss = loss_fn(pred, occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
        
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "num_cameras": 28,
        "latent_dim": 256,
        "n_points": dataset.n_points,
        "R_max": R_max,
    }, "checkpoint.pth")

    print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

def stl_to_voxels_normalized(stl_path, radius, resolution=64):
    mesh = trimesh.load(stl_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Kopie der Vertices
    vertices = mesh.vertices.copy()

    # x,y durch Radius normieren
    vertices[:, 0] /= radius
    vertices[:, 1] /= radius

    mesh.vertices = vertices

    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.0, 1.0

    pitch = 2.0 / resolution

    voxelized = mesh.voxelized(pitch=pitch).fill()
    points = voxelized.points

    ix = ((points[:, 0] - x_min) / 2.0 * resolution).astype(int)
    iy = ((points[:, 1] - y_min) / 2.0 * resolution).astype(int)
    iz = ((points[:, 2] - z_min) / 2.0 * resolution).astype(int)

    valid = (
        (ix >= 0) & (ix < resolution) &
        (iy >= 0) & (iy < resolution) &
        (iz >= 0) & (iz < resolution)
    )

    grid[ix[valid], iy[valid], iz[valid]] = 1

    return grid

def voxels_to_stl_normalized(voxels, out_path, radius, threshold=0.5):
    if threshold is None:
        threshold = 0.5 * (voxels.min() + voxels.max())

    verts, faces, normals, values = measure.marching_cubes(voxels, level=threshold)

    res = voxels.shape[0]

    # Voxelindex → [-1,1]
    verts = verts / (res - 1) * 2.0 - 1.0

    # x,y zurückskalieren
    verts[:, 0] *= radius
    verts[:, 1] *= radius

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(out_path)

def reconstruct_occupancy(model, lc, radius, res=512, chunk=200000):
    if not torch.is_tensor(radius):
        radius = torch.tensor([radius], dtype=torch.float32, device=device)
    else:
        radius = radius.to(device)
        if radius.dim() == 0:
            radius = radius.unsqueeze(0)
    coords = torch.linspace(-1, 1, res)

    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1)
    points = grid.reshape(-1, 3)

    occ_values = []

    model.eval()

    with torch.no_grad():
        for i in range(0, len(points), chunk):
            p = points[i:i+chunk].unsqueeze(0).to(device)

            logits = model(lc, p, radius)
            probs = torch.sigmoid(logits)

            occ_values.append(probs.cpu())

    occ = torch.cat(occ_values, dim=1)
    occ = occ.reshape(res, res, res).numpy()

    return occ


lc = load_lightcurve("C:/Users/rober/Python Datengenerierung/brightnessasteroid5radius2.0475867806576935.stl.csv")
lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
radius_real = float(2.0475867806576935)
radius_input = radius_real / R_max

radius = torch.tensor(radius_input, dtype=torch.float32)
voxels = reconstruct_occupancy(model, lc, radius, res=64)
radius_value = radius.detach().cpu().item()
threshold = (voxels.min() + voxels.max()) / 2
voxels_to_stl_normalized(voxels, "predicted.stl", radius_value,threshold)

end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")