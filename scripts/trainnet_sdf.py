from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys
from pathlib import Path
import torch.nn.functional as F
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.datasets import AsteroidSDFPointDataset
from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet

start = time.time()
start2=start

R_max = 5.313693321295838

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print(f"Using device: {device}")

dataset = AsteroidSDFPointDataset("data/dataset", n_points=8192)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=16,
    pin_memory=use_cuda,
    persistent_workers=True,
    prefetch_factor=2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sdf"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

commence=0
if commence==1:
    checkpoint_path = CHECKPOINT_DIR / "/sdf_39k/checkpoint2700_sdf_39k.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = LightcurveSDFNet(
        num_cameras=checkpoint["num_cameras"],
        latent_dim=checkpoint["latent_dim"],
        num_freqs=checkpoint.get("num_freqs", 6)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] +1
    print(f"Resuming from epoch {start_epoch}")
else:
    model = LightcurveSDFNet(num_cameras=28,latent_dim=256,num_freqs=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    start_epoch = 0

run = wandb.init(
    project="lightcurve-asteroid-reconstruction",
    name="sdf-training-39ksamples",

    config={
        "model": "LightcurveSDFNet",
        "representation": "truncated_sdf",
        "num_cameras": 28,
        "latent_dim": 256,
        "num_freqs": 6,
        "n_points": 8192,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "surface_weight": 5.0,
        "surface_threshold": 0.2,
        "R_max": R_max,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16" if use_cuda else "disabled",
        "checkpoint_loaded": commence == 1,
        "start_epoch": start_epoch,
    }
)

scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

for epoch in range(start_epoch, start_epoch+2400):
    model.train()
    total_loss = 0.0
    total_raw_loss = 0.0

    if use_cuda:
        torch.cuda.synchronize()

    epoch_start = time.time()

    for batch_idx, (lc, points, sdf, radius) in enumerate(loader):
        lc = lc.to(device, non_blocking=use_cuda)
        points = points.to(device, non_blocking=use_cuda)
        sdf = sdf.to(device, non_blocking=use_cuda)
        radius = radius.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_cuda
        ):
            pred_sdf = model(lc, points, radius)

            loss_raw = F.smooth_l1_loss(
                pred_sdf,
                sdf,
                reduction="none"
            )

            weights = 1.0 + 5.0 * (torch.abs(sdf) < 0.2).float()
            sdf_loss = (weights * loss_raw).mean()

        scaler.scale(sdf_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_raw_loss = loss_raw.mean().item()
        batch_loss = sdf_loss.item()

        total_loss += batch_loss
        total_raw_loss += batch_raw_loss

        if batch_idx % 10 == 0:
            wandb.log({
                "train/batch_weighted_sdf_loss": batch_loss,
                "train/batch_raw_sdf_loss": batch_raw_loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

    if use_cuda:
        torch.cuda.synchronize()

    epoch_time = time.time() - epoch_start

    mean_loss = total_loss / len(loader)
    mean_raw_loss = total_raw_loss / len(loader)

    epoch_log = {
        "epoch": epoch,
        "train/epoch_weighted_sdf_loss": mean_loss,
        "train/epoch_raw_sdf_loss": mean_raw_loss,
        "time/epoch_seconds": epoch_time,
        "train/learning_rate": optimizer.param_groups[0]["lr"],
    }

    if use_cuda:
        epoch_log.update({
            "gpu/memory_allocated_gb":
                torch.cuda.memory_allocated() / 1024**3,

            "gpu/memory_reserved_gb":
                torch.cuda.memory_reserved() / 1024**3,
        })

    wandb.log(epoch_log)

    if epoch % 100 == 0:
        checkpoint_path = CHECKPOINT_DIR / f"sdf_39k/checkpoint{epoch}_sdf_39k.pth"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": mean_loss,
            "raw_loss": mean_raw_loss,
            "num_cameras": 28,
            "latent_dim": 256,
            "num_freqs": 6,
            "R_max": R_max,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        artifact = wandb.Artifact(
            name=f"sdf-checkpoint-epoch-{epoch}",
            type="model",
            metadata={
                "epoch": epoch,
                "weighted_sdf_loss": mean_loss,
                "raw_sdf_loss": mean_raw_loss,
                "latent_dim": 256,
                "num_freqs": 6,
                "n_points": 8192,
            }
        )

        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    print(
        f"Epoch {epoch}: "
        f"weighted loss = {mean_loss:.6f}, "
        f"raw loss = {mean_raw_loss:.6f}, "
        f"time = {epoch_time:.2f} s"
    )

wandb.finish()