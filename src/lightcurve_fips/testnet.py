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

repo_path = Path("C:/Users/rober/diffusion-net-master/diffusion-net-master/src").resolve()
sys.path.insert(0, str(repo_path))

import diffusion_net

R_max = 5.313693321295838

class LightcurveEncoder(nn.Module):
    def __init__(self, num_cameras, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_cameras, 64, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(256, latent_dim),
            nn.SiLU()
        )

    def forward(self, lc):
        return self.net(lc)

class DiffusionMeshDecoder(nn.Module):
    def __init__(self, latent_dim=256, C_width=128):
        super().__init__()

        # Pro Vertex geben wir ein:
        # 3 Koordinaten + latent_dim + 1 Radius-Parameter
        C_in = 3 + latent_dim + 1

        # Pro Vertex geben wir einen Radius aus
        C_out = 1

        self.diffnet = diffusion_net.layers.DiffusionNet(
            C_in=C_in,
            C_out=C_out,
            C_width=C_width,
            last_activation=None,
            outputs_at="vertices"
        )

    def forward(self, latent, radius, template_verts, faces, mass, L, evals, evecs, gradX, gradY):
        B = latent.shape[0]
        V = template_verts.shape[0]
        outputs = []

        # stelle sicher, dass template_verts auf device ist
        template_verts = template_verts.to(latent.device)

        for b in range(B):
            z_b = latent[b]       # (latent_dim,)
            r_b = radius[b].view(1)

            z_per_vertex = z_b[None, :].expand(V, -1).to(latent.device)
            r_per_vertex = r_b[None, :].expand(V, -1).to(latent.device)

            # template_verts already on same device
            features = torch.cat([template_verts, z_per_vertex, r_per_vertex], dim=-1)

            raw = self.diffnet(
                features,
                mass,
                L=L,
                evals=evals,
                evecs=evecs,
                gradX=gradX,
                gradY=gradY,
                faces=faces
            )

            raw = raw.squeeze(-1)
            pred_r = F.softplus(raw) + 1e-4
            outputs.append(pred_r)

        pred_radii = torch.stack(outputs, dim=0)
        return pred_radii

class LightcurveDiffusionNet(nn.Module):
    def __init__(self, num_cameras, latent_dim=256, C_width=128):
        super().__init__()

        self.encoder = LightcurveEncoder(
            num_cameras=num_cameras,
            latent_dim=latent_dim
        )

        self.decoder = DiffusionMeshDecoder(
            latent_dim=latent_dim,
            C_width=C_width
        )

    def forward(
        self,
        lc,
        radius,
        template_verts,
        faces,
        mass,
        L,
        evals,
        evecs,
        gradX,
        gradY
    ):
        latent = self.encoder(lc)

        pred_radii = self.decoder(
            latent,
            radius,
            template_verts,
            faces,
            mass,
            L,
            evals,
            evecs,
            gradX,
            gradY
        )

        return pred_radii

def reconstruct_mesh_diffusionnet(
    model,
    lc,
    radius_model,      # scalar Tensor (B=1) or python float
    radius_real,       # original radius float for rescaling x/y
    template_verts,    # torch.Tensor (V,3) - can be CPU or CUDA
    template_faces,    # torch.Tensor (F,3) or numpy array
    mass, L, evals, evecs, gradX, gradY,
    out_path="recon.stl"
):
    model.eval()

    # Device where the model lives
    model_device = next(model.parameters()).device

    # Ensure input lightcurve and radius are on model device
    if not torch.is_tensor(lc):
        raise ValueError("lc must be a torch tensor")
    lc = lc.to(model_device)

    if not torch.is_tensor(radius_model):
        radius_model = torch.tensor([radius_model], dtype=torch.float32, device=model_device)
    else:
        radius_model = radius_model.to(model_device)
        if radius_model.dim() == 0:
            radius_model = radius_model.unsqueeze(0)

    # Ensure template_verts on same device as model (so multiplication works)
    template_verts_dev = template_verts.to(model_device, dtype=torch.float32)

    # Also ensure operators and faces are on the expected device
    # (you probably moved them earlier; if not, do it here)
    mass = mass.to(model_device) if torch.is_tensor(mass) else mass
    L = L.to(model_device) if torch.is_tensor(L) else L
    evals = evals.to(model_device) if torch.is_tensor(evals) else evals
    evecs = evecs.to(model_device) if torch.is_tensor(evecs) else evecs
    gradX = gradX.to(model_device) if torch.is_tensor(gradX) else gradX
    gradY = gradY.to(model_device) if torch.is_tensor(gradY) else gradY
    faces_dev = template_faces.to(model_device, dtype=torch.long) if torch.is_tensor(template_faces) else torch.from_numpy(template_faces).to(model_device)

    # Run model -> predict radii (B,V)
    with torch.no_grad():
        pred_radii = model(
            lc,
            radius_model,
            template_verts_dev,
            faces_dev,
            mass,
            L,
            evals,
            evecs,
            gradX,
            gradY
        )

    # pred_radii should be on model_device and shape (B, V) ; handle B=1
    if pred_radii.dim() == 1:
        pred_radii = pred_radii.unsqueeze(0)

    # Debug print: devices and shapes
    print("Devices and shapes:")
    print(" model device:", model_device)
    print(" template_verts_dev:", template_verts_dev.device, template_verts_dev.shape)
    print(" pred_radii:", pred_radii.device, pred_radii.shape)

    # Use first batch element
    pred_r = pred_radii[0]                # shape (V,)
    dirs = F.normalize(template_verts_dev, dim=-1)  # (V,3), on same device

    # Ensure pred_r is broadcastable: (V,) -> (V,1)
    pred_verts = dirs * pred_r[:, None]  # elementwise on model_device

    # Convert to cpu numpy for trimesh export
    pred_verts_np = pred_verts.detach().cpu().numpy()
    if torch.is_tensor(template_faces):
        faces_np = template_faces.detach().cpu().numpy()
    else:
        faces_np = template_faces  # assume numpy

    # Adjust coordinates like in your SDF pipeline (x/y scaled by radius_real)
    pred_verts_np[:, 0] *= radius_real
    pred_verts_np[:, 1] *= radius_real

    mesh = trimesh.Trimesh(vertices=pred_verts_np, faces=faces_np, process=True)
    mesh.export(out_path)

    return mesh

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

def load_lightcurve(csv_path):
    data = pd.read_csv(csv_path, header=None)

    values = data.values.astype(np.float32)

    # Falls erste Spalte Framezahl ist:
    values = values[:, 1:]

    values = values / (values.mean(axis=0, keepdims=True) + 1e-8)

    return values

def create_template_sphere(subdivisions=5):
    """
    Erstellt ein festes Kugel-Template.
    
    subdivisions=3 -> grober
    subdivisions=4 -> meist guter Start
    subdivisions=5 -> feiner, aber langsamer
    """
    mesh = trimesh.creation.icosphere(
        subdivisions=subdivisions,
        radius=1.0
    )

    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    return verts, faces


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(x, device=device, dtype=torch.float32):
    """
    Wandelt x nach Torch-Tensor konvertiert und verschiebt ihn auf device.
    Unterstützt: numpy.ndarray, torch.Tensor, torch.sparse.*, oder None.
    """
    if x is None:
        return None

    # NumPy -> Torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)

    # SciPy sparse (falls zufällig) -> dense Torch (vorsicht Speicher)
    try:
        import scipy.sparse as sp
        if isinstance(x, sp.spmatrix):
            arr = x.toarray().astype(np.float32)
            return torch.from_numpy(arr).to(device=device, dtype=dtype)
    except Exception:
        pass

    # already torch tensor
    if torch.is_tensor(x):
        # Falls schon ein SparseTensor (torch >= 1.13: torch.sparse_coo_tensor)
        if x.is_sparse:
            # Sparse tensor: stelle dtype sicher, dann auf device verschieben
            # einige ältere APIs liefern deprecated torch.sparse.FloatTensor; hier coalesce nutzen
            try:
                x = x.coalesce()
            except Exception:
                pass
            # dtype und device
            return x.to(device=device).to(dtype=dtype)
        else:
            return x.to(device=device, dtype=dtype)

    # Fallback: versuche Konvertierung über np.array
    try:
        arr = np.array(x)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    except Exception as err:
        raise TypeError(f"Kann Objekt vom Typ {type(x)} nicht in Torch konvertieren: {err}")

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

template_verts, template_faces = create_template_sphere(subdivisions=5)

frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
    template_verts, template_faces, op_cache_dir="diffusionnet_cache"
)

mass  = to_torch(mass, device=device)
L     = to_torch(L, device=device)
evals = to_torch(evals, device=device)
evecs = to_torch(evecs, device=device)
gradX = to_torch(gradX, device=device)
gradY = to_torch(gradY, device=device)

net=1
if net==0:
    checkpoint = torch.load("checkpoint4900_sdf.pth", map_location=device)

    model = LightcurveSDFNet(
        num_cameras=checkpoint["num_cameras"],
        latent_dim=checkpoint["latent_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    root=Path.cwd()
    asteroid="dataset2/test/brightnessasteroid51radius4.534746888643775.stl.csv"
    lc = load_lightcurve(root/asteroid)
    lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", asteroid)
    radius_value = float(m.group(1))

    radius_model = torch.tensor([radius_value / R_max], dtype=torch.float32, device=device)

    sdf = reconstruct_sdf(model, lc, radius_model, res=32)

    sdf_to_stl(
        sdf,
        "asteroid_sdf_reconstruction.stl",
        radius=radius_value
    )
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("checkpoint290_diffusionnet.pth", map_location=device)

    model = LightcurveDiffusionNet(
                num_cameras=28,
                latent_dim=256,
                C_width=128
            ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    root=Path.cwd()
    asteroid="dataset2/test/brightnessasteroid36radius1.6183293841683848.stl.csv"
    lc = load_lightcurve(root/asteroid)
    lc = torch.tensor(lc.T, dtype=torch.float32).unsqueeze(0).to(device)
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", asteroid)
    radius_value = float(m.group(1))

    radius_model = torch.tensor(
    [radius_value / R_max],
        dtype=torch.float32,
        device=device
    )

    mesh = reconstruct_mesh_diffusionnet(
        model=model,
        lc=lc,
        radius_model=radius_model,
        radius_real=radius_value,
        template_verts=template_verts,
        template_faces=template_faces,
        mass=mass,
        L=L,
        evals=evals,
        evecs=evecs,
        gradX=gradX,
        gradY=gradY,
        out_path="asteroid_diffusionnet_reconstruction.stl"
    )