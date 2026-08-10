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

start = time.time()
start2=start

R_max = 5.313693321295838

class AsteroidSDFPointDataset(Dataset):
    def __init__(self, root, n_points=8192):
        self.root = Path.cwd()
        self.root=self.root/"dataset"
        self.n_points = n_points

        self.samples = sorted([
            p for p in self.root.rglob("sample*")
            if list(p.glob("brightness*.csv"))
            and list(p.glob(f"points_{n_points}_*.npy"))
            and list(p.glob(f"sdf_{n_points}_*.npy"))
        ])

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        csv_path = sorted(folder.glob("brightness*.csv"))[0]

        lc = load_lightcurve(csv_path)

        point_files = sorted(folder.glob(f"points_{self.n_points}_*.npy"))
        sdf_files = sorted(folder.glob(f"sdf_{self.n_points}_*.npy"))

        k = np.random.randint(len(point_files))

        points = np.load(point_files[k]).astype(np.float32)
        sdf = np.load(sdf_files[k]).astype(np.float32)

        stl_path = sorted(folder.glob("asteroid*.stl"))[0]

        radius_real = parse_radius_from_stl(stl_path)

        radius_input = radius_real / R_max

        lc = torch.tensor(lc.T, dtype=torch.float32)
        points = torch.tensor(points, dtype=torch.float32)
        sdf = torch.tensor(sdf, dtype=torch.float32)
        radius = torch.tensor(radius_input, dtype=torch.float32)

        return lc, points, sdf, radius

def parse_radius_from_stl(stl_path):
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", stl_path.name)

    if m is None:
        raise ValueError(f"Could not parse radius from {stl_path.name}")

    return float(m.group(1))

def load_lightcurve(csv_path):
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    # Falls erste Spalte Framezahl ist:
    values = values[:, 1:]

    values = values / (values.mean(axis=0, keepdims=True) + 1e-8)

    return values

class LightcurveEncoder(nn.Module):
    def __init__(self, num_cameras, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_cameras, 64, kernel_size=7, padding=3),       #cameras as input channels
            nn.SiLU(),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(1),                                    #pooling for global curve descriptor
            nn.Flatten(),

            nn.Linear(256, latent_dim),                                 #produce latent code
            nn.SiLU()
        )

    def forward(self, lc):
        return self.net(lc)

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_freqs=6):
        super().__init__()

        self.num_freqs = num_freqs

        point_dim = 3 + 2 * num_freqs * 3

        self.net = nn.Sequential(
            nn.Linear(latent_dim + point_dim + 1, 512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.SiLU(),

            nn.Linear(512, 256),
            nn.SiLU(),

            nn.Linear(256, 1)
        )

    def forward(self, latent, points, radius):
        """
        latent: (B, latent_dim)
        points: (B, N, 3)
        radius: (B,)
        """

        B, N, _ = points.shape

        if radius.dim() == 0:
            radius = radius.unsqueeze(0)

        radius = radius.view(B)

        points_enc = positional_encoding(points, self.num_freqs)

        latent_expanded = latent[:, None, :].expand(-1, N, -1)
        radius_expanded = radius[:, None, None].expand(-1, N, 1)

        x = torch.cat(
            [latent_expanded, points_enc, radius_expanded],
            dim=-1
        )

        sdf = self.net(x)

        return sdf.squeeze(-1)
    
def positional_encoding(points, num_freqs=6):
    """
    points: (B, N, 3)
    output: (B, N, 3 + 2*num_freqs*3)
    """
    enc = [points]

    for i in range(num_freqs):
        freq = 2.0 ** i
        enc.append(torch.sin(freq * torch.pi * points))
        enc.append(torch.cos(freq * torch.pi * points))

    return torch.cat(enc, dim=-1)

class LightcurveSDFNet(nn.Module):
    def __init__(self, num_cameras, latent_dim=256, num_freqs=6):
        super().__init__()

        self.encoder = LightcurveEncoder(
            num_cameras=num_cameras,
            latent_dim=latent_dim
        )

        self.decoder = SDFDecoder(
            latent_dim=latent_dim,
            num_freqs=num_freqs
        )

    def forward(self, lc, points, radius):
        latent = self.encoder(lc)
        sdf = self.decoder(latent, points, radius)
        return sdf

def reconstruct_sdf(model, lc, radius, res=128, chunk=200000, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # sicherstellen, dass radius Tensor mit Batch-Dimension ist
    if not torch.is_tensor(radius):
        radius = torch.tensor([radius], dtype=torch.float32, device=device)
    else:
        radius = radius.to(device)
        if radius.dim() == 0:
            radius = radius.unsqueeze(0)

    lc = lc.to(device)

    coords = torch.linspace(-1, 1, res, device=device)

    X, Y, Z = torch.meshgrid(
        coords, coords, coords,
        indexing="ij"
    )

    grid_points = torch.stack([X, Y, Z], dim=-1)
    grid_points = grid_points.reshape(-1, 3)

    sdf_values = []

    with torch.no_grad():
        for i in range(0, grid_points.shape[0], chunk):
            points = grid_points[i:i+chunk]

            # shape: (1, chunk, 3)
            points = points.unsqueeze(0)

            sdf = model(lc, points, radius)

            # shape: (1, chunk)
            sdf_values.append(sdf.squeeze(0).cpu())

    sdf_grid = torch.cat(sdf_values, dim=0)
    sdf_grid = sdf_grid.reshape(res, res, res).numpy()

    return sdf_grid

def sdf_to_stl(sdf, out_path, radius):
    res = sdf.shape[0]
    if torch.is_tensor(radius):
        radius = radius.detach().cpu().item()

    print("sdf min:", sdf.min())
    print("sdf max:", sdf.max())

    if not (sdf.min() <= 0 <= sdf.max()):
        raise ValueError("SDF does not cross zero. Surface cannot be extracted.")

    verts, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(2/(res-1), 2/(res-1), 2/(res-1))
    )

    # marching_cubes startet bei Koordinate 0, also nach [-1,1] verschieben
    verts -= 1.0

    # x/y zurückskalieren
    verts[:, 0] *= radius
    verts[:, 1] *= radius

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.export(out_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AsteroidSDFPointDataset("dataset", n_points=8192)
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
    
    # print("sdf target min/max:", sdf.min().item(), sdf.max().item())
    # print("pred min/max:", pred_sdf.min().item(), pred_sdf.max().item())

    if epoch%100==0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(loader),
            "num_cameras": 28,
            "latent_dim": 256,
            "R_max": R_max
        }, f"checkpoint{epoch}_sdf.pth")
        print("Checkpoint saved!")
    
    end2 = time.time()
    length = end2 - start2

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f},",length, "seconds!")
    with open("losses.txt", "a") as myfile:
        myfile.write(f"Epoch: {epoch} Loss: {total_loss / len(loader):.4f} \n")
    start2=end2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoint1800_sdf.pth", map_location=device)

model = LightcurveSDFNet(
    num_cameras=checkpoint["num_cameras"],
    latent_dim=checkpoint["latent_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

root=Path.cwd()
lc = load_lightcurve(root/"brightnessasteroid3_scaled_radius0.8782467278262304.stl.csv")
lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
radius_value = 0.8782467278262304

radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

sdf = reconstruct_sdf(model, lc, radius_model, res=32)

sdf_to_stl(
    sdf,
    "asteroid_sdf_reconstruction.stl",
    radius=radius_value
)

end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")