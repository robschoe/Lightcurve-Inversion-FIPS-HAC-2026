from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import sys
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightcurve_fips.data.datasets import AsteroidSDFPointDatasetCombined
from lightcurve_fips.models.lightcurve_encoder import LightcurveSDFNet
from lightcurve_fips.evaluation.evaluate import evaluate_geometry
from lightcurve_fips.rendering.camera_setup import create_camera

start = time.time()
start2=start

R_max = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print(f"Using device: {device}")

dataset = AsteroidSDFPointDatasetCombined("data/dataset", n_points=8192, max_samples=None)

val_fraction=0.1 #relative size of validation set

n_total = len(dataset)
n_val = int(n_total * val_fraction)
n_train = n_total - n_val

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=generator
)

geometry_val_count = min(20, len(val_dataset))

geometry_val_indices = val_dataset.indices[:geometry_val_count]

geometry_val_folders = [
    dataset.samples[i]["folder"]
    for i in geometry_val_indices
]

TRAIN_BATCH_SIZE = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sdf"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


model = LightcurveSDFNet(num_cameras=21,latent_dim=256,num_freqs=8).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-5,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    threshold=1e-3,
    threshold_mode="rel",
    cooldown=2,
    min_lr=1e-7,
)

start_epoch = 0

run = wandb.init(
    project="lightcurve-asteroid-reconstruction",
    name="sdf-159k-net2-combined-residual-beta0.1-newsdf-freq8",

    config={
        "model": "LightcurveSDFNet",
        "representation": "truncated_sdf",
        "num_cameras": 21,
        "num_modalities": 2,
        "latent_dim": 256,
        "num_freqs": 8,
        "n_points": 8192,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": 5e-5,
        "surface_weight": 5.0,
        "surface_threshold": 0.2,
        "R_max": R_max,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16" if use_cuda else "disabled",
        "checkpoint_loaded": False,
        "start_epoch": start_epoch,
        "train_fraction": 1.0 - val_fraction,
        "validation_fraction": val_fraction,
        "split_seed": 42,
    }
)

best_geometry_voxel_score = -float("inf")

for epoch in range(start_epoch, start_epoch+2400):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_raw_loss = torch.zeros((), device=device)

    if use_cuda:
        torch.cuda.synchronize()

    epoch_start = time.time()

    for batch_idx, (lc, points, sdf, radius) in enumerate(train_loader):
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
                beta=0.2,
                reduction="none"
            )

            weights = 1.0 + 5.0 * (torch.abs(sdf) < 0.2).float()
            sdf_loss = (
                weights * loss_raw
            ).sum() / weights.sum().clamp_min(1e-8)

        sdf_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += sdf_loss.detach()
        total_raw_loss += loss_raw.mean().detach()

        if batch_idx % 10 == 0:
            wandb.log({
                "train/batch_weighted_sdf_loss": sdf_loss.item(),
                "train/batch_raw_sdf_loss": loss_raw.mean().item(),
                "train/gradient_norm": float(gradient_norm),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

    model.eval()

    val_weighted_loss_sum = torch.zeros(
        (),
        device=device
    )

    val_raw_loss_sum = torch.zeros(
        (),
        device=device
    )

    with torch.inference_mode():
        for lc, points, sdf, radius in val_loader:
            lc = lc.to(device, non_blocking=use_cuda)
            points = points.to(device, non_blocking=use_cuda)
            sdf = sdf.to(device, non_blocking=use_cuda)
            radius = radius.to(device, non_blocking=use_cuda)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_cuda
            ):
                pred_sdf = model(lc, points, radius)

                val_loss_raw = F.smooth_l1_loss(
                    pred_sdf,
                    sdf,
                    beta=0.2,
                    reduction="none"
                )

                weights = 1.0 + 5.0 * (torch.abs(sdf) < 0.2).float()

                val_loss = (
                    weights * val_loss_raw
                ).sum() / weights.sum().clamp_min(1e-8)
                val_raw_loss = val_loss_raw.mean()

            val_weighted_loss_sum += val_loss.item()
            val_raw_loss_sum += val_raw_loss.item()

    mean_train_loss = (
        total_loss / len(train_loader)
    ).item()

    mean_train_raw_loss = (
        total_raw_loss / len(train_loader)
    ).item()

    mean_val_loss = (
        val_weighted_loss_sum / len(val_loader)
    ).item()

    mean_val_raw_loss = (
        val_raw_loss_sum / len(val_loader)
    ).item()

    scheduler.step(mean_val_raw_loss)

    eval_every = 5

    if epoch % eval_every == 0:
        geometry_metrics = evaluate_geometry(
            model=model,
            sample_folders=geometry_val_folders,
            device=device,
            r_max=R_max,

            sdf_resolution=64,
            voxel_resolution=128,
            projection_resolution=128,

            output_dir=PROJECT_ROOT / "validation_reconstructions" / f"epoch_{epoch}"
        )

        print(
            f"Geometry validation | "
            f"Voxel: {geometry_metrics['voxel_score']:.4f} | "
            f"Side view: {geometry_metrics['side_view_score']:.4f} | "
            f"Evaluated: {geometry_metrics['n_evaluated']} | "
            f"Skipped: {geometry_metrics['n_skipped']}| "
            f"Real_Voxel1: {geometry_metrics['real_voxel1']}| "
            f"Real_Voxel2: {geometry_metrics['real_voxel2']}"
        )

        wandb.log({
            "epoch": epoch,
            "geometry_val/voxel_score": geometry_metrics["voxel_score"],
            "geometry_val/side_view_score": geometry_metrics["side_view_score"],
            "geometry_val/n_evaluated": geometry_metrics["n_evaluated"],
            "geometry_val/n_skipped": geometry_metrics["n_skipped"],
            "geometry_val/real_voxel1": geometry_metrics['real_voxel1'],
            "geometry_val/real_voxel2": geometry_metrics['real_voxel2'],
        })

        current_voxel_score = geometry_metrics["voxel_score"]

        if np.isfinite(current_voxel_score):
            if current_voxel_score > best_geometry_voxel_score:
                best_geometry_voxel_score = current_voxel_score

                best_path = CHECKPOINT_DIR / "best_by_voxel_score_sdf.pth"

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),

                    "train_indices": train_dataset.indices,
                    "val_indices": val_dataset.indices,

                    "num_cameras": 21,
                    "num_modalities": 2,
                    "latent_dim": 256,
                    "num_freqs": 8,
                    "n_points": 8192,
                    "batch_size": TRAIN_BATCH_SIZE,
                    "surface_weight": 5.0,
                    "surface_threshold": 0.2,
                    "R_max": R_max,
                }, best_path)

                print(
                    f"New best geometry checkpoint saved: "
                    f"voxel_score={current_voxel_score:.4f}"
                )

                wandb.log({
                    "geometry_val/best_voxel_score":
                        best_geometry_voxel_score
                })

    wandb.log({
        "epoch": epoch,

        "train/epoch_weighted_sdf_loss": mean_train_loss,
        "train/epoch_raw_sdf_loss": mean_train_raw_loss,

        "val/epoch_weighted_sdf_loss": mean_val_loss,
        "val/epoch_raw_sdf_loss": mean_val_raw_loss,

        "train/learning_rate": optimizer.param_groups[0]["lr"],
    })

    if use_cuda:
        torch.cuda.synchronize()

    epoch_time = time.time() - epoch_start

    if epoch % 10 == 0:
        checkpoint_path = CHECKPOINT_DIR / "sdf_99k" / f"checkpoint{epoch}_sdf_159k_net2_combined_residual_beta0_1_newsdf_freq8.pth"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),

            "train_indices": train_dataset.indices,
            "val_indices": val_dataset.indices,

            "num_cameras": 21,
            "num_modalities": 2,
            "latent_dim": 256,
            "num_freqs": 8,
            "n_points": 8192,
            "batch_size": TRAIN_BATCH_SIZE,
            "surface_weight": 5.0,
            "surface_threshold": 0.2,
            "R_max": R_max,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        artifact = wandb.Artifact(
            name=f"sdf-checkpoint-epoch-{epoch}",
            type="model",
            metadata={
                "epoch": epoch,
                "weighted_sdf_loss": mean_train_loss,
                "raw_sdf_loss": mean_train_raw_loss,
                "latent_dim": 256,
                "num_freqs": 8,
                "n_points": 8192,
            }
        )

        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    print(
        f"Epoch {epoch}: "
        f"weighted loss = {mean_train_loss:.6f}, "
        f"raw loss = {mean_train_raw_loss:.6f}, "
        f"time = {epoch_time:.2f} s"
    )

wandb.finish()