import numpy as np
import torch
from scipy.signal import savgol_filter
from pathlib import Path
import os
import time
import sys
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.rendering.camera_setup import (create_camera)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

frames = 360
brightness = []

start = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"

Rs=[]

for frame in range(frames):
    theta = - frame / frames * 2 * torch.pi
    theta = torch.tensor(theta, dtype=torch.float32, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
        torch.stack([s,  c, torch.zeros_like(c)], dim=-1),
        torch.tensor([0.,0.,1.], device=device)
    ], dim=0)
    Rs.append(R)

cameras = []

for i in range(8):
    if i!=4:
        cam1 = create_camera(f"camera{i}mid", i*45,0)
        cam11 = create_camera(f"camera{i}mid", i*45,0)
        cam2 = create_camera(f"camera{i}top", i*45,1)
        cam3 = create_camera(f"camera{i}bot", i*45,-1)
        cameras.append(cam1)
        cameras.append(cam11)
        cameras.append(cam2)
        cameras.append(cam3)

view_dirs = []

for cam in cameras:
    norm = np.linalg.norm(cam)

    if norm > 0:
        cam = cam / norm
    view_dirs.append(cam)

view_dirs_np = np.array(view_dirs)

sun_dir_np = np.array([-10, 0, 0], dtype=np.float32)
norm = np.linalg.norm(sun_dir_np)
if norm > 0:
    sun_dir_np = sun_dir_np / norm

sun_dir = torch.tensor(sun_dir_np, dtype=torch.float32, device=device)
view_dirs = torch.tensor(view_dirs_np, dtype=torch.float32, device=device)

camera_pos = torch.tensor(
    np.asarray(cameras, dtype=np.float32),
    dtype=torch.float32,
    device=device
)  # (C, 3)

num_cameras = camera_pos.shape[0]

# Blickrichtung: Ursprung -> Kamera
views = F.normalize(camera_pos, dim=1)  # (C, 3)

# Standard-Up-Vektor
up = torch.tensor(
    [0.0, 0.0, 1.0],
    dtype=torch.float32,
    device=device
).expand(num_cameras, -1).clone()

# Falls view fast parallel zu z ist, alternativen Up-Vektor wählen
parallel_mask = torch.abs(torch.sum(views * up, dim=1)) > 0.95

up[parallel_mask] = torch.tensor(
    [0.0, 1.0, 0.0],
    dtype=torch.float32,
    device=device
)

# Lokale Kameraachsen
x_axes = F.normalize(
    torch.cross(views, up, dim=1),
    dim=1
)

y_axes = F.normalize(
    torch.cross(x_axes, views, dim=1),
    dim=1
)

print("Number of cameras:", num_cameras)

g=18
for index in range(10):
    root = DATASET_DIR/f"batch{23+index}"
    
    samples = sorted([
        p for p in root.rglob("sample*")
        if list(p.glob("asteroid*.stl"))
    ])
    
    asteroids = list(root.rglob("asteroid*.stl"))
    print("Found asteroids:", len(samples))
    for folder in tqdm(samples):
        stl_path = sorted(folder.glob("asteroid*.stl"))[0]
        brightnesses = list(folder.rglob("brightness*.csv"))

        if brightnesses==[]:
            brightness=[]

            W = 512                           #Define granularity of projection
            H = 512
            num_pixels = W * H

            centers = []
            normals = []
            areas = []

            mesh = trimesh.load(stl_path, force="mesh")

            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            centers = torch.tensor(
                mesh.triangles_center,
                dtype=torch.float32,
                device=device
            )

            normals = torch.tensor(
                mesh.face_normals,
                dtype=torch.float32,
                device=device
            )

            areas = torch.tensor(
                mesh.area_faces,
                dtype=torch.float32,
                device=device
            )

            for frame in range(frames):
                R=Rs[frame]                         #Get rotationsmatrix for given frame

                rotated_normals = normals @ R.T     #Apply rotation
                rotated_centers = centers @ R.T

                illum = torch.clamp(
                    rotated_normals @ sun_dir,
                    min=0.0
                )

                cx = (rotated_centers @ x_axes.T).T
                cy = (rotated_centers @ y_axes.T).T
                cz = (rotated_centers @ views.T).T

                # Sichtfaktor aller Faces für alle Kameras:
                # Ergebnis: (C, F)
                vis = torch.clamp(
                    (rotated_normals @ views.T).T,
                    min=0.0
                )

                # illum: (F,)
                # Zu (C, F) erweitern
                illum_all = illum.unsqueeze(0)

                # Normierung in Pixelkoordinaten, jeweils pro Kamera
                cx_min = cx.min(dim=1, keepdim=True).values
                cx_max = cx.max(dim=1, keepdim=True).values

                cy_min = cy.min(dim=1, keepdim=True).values
                cy_max = cy.max(dim=1, keepdim=True).values

                # Schutz gegen Division durch 0
                cx_range = (cx_max - cx_min).clamp_min(1e-8)
                cy_range = (cy_max - cy_min).clamp_min(1e-8)

                px = ((cx - cx_min) / cx_range * (W - 1)).long()
                py = ((cy - cy_min) / cy_range * (H - 1)).long()

                px = torch.clamp(px, 0, W - 1)
                py = torch.clamp(py, 0, H - 1)

                # (C, F): Jeder Face-Mittelpunkt erhält pro Kamera einen Pixelindex
                flat_index = py * W + px

                # ------------------------------------------------------------
                # Batched Z-Buffer für alle Kameras
                # ------------------------------------------------------------

                # Da bei deiner view-Definition größere cz-Werte näher an der
                # Kamera liegen sollten, verwenden wir amax.
                zbuf = torch.full(
                    (num_cameras, num_pixels),
                    -float("inf"),
                    dtype=torch.float32,
                    device=device
                )

                zbuf.scatter_reduce_(
                    dim=1,
                    index=flat_index,
                    src=cz,
                    reduce="amax",
                    include_self=True
                )

                visible_mask = cz == zbuf.gather(1, flat_index)

                # ------------------------------------------------------------
                # Helligkeit pro Kamera
                # ------------------------------------------------------------

                brightness_per_camera = torch.sum(
                    areas.unsqueeze(0) *
                    illum.unsqueeze(0) *
                    vis *
                    visible_mask,
                    dim=1
                )

                frame_values = [frame] + brightness_per_camera.detach().cpu().tolist()
                brightness.append(frame_values)
            
            path=folder
            asteroid=os.path.basename(stl_path)

            # print(asteroid)
            # print(path)

            np.savetxt(f"{path}/brightness{asteroid}.csv", brightness, delimiter=",")
        g+=1
    

end = time.time()
length = end - start

print("It took", length, "seconds!")