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
from lightcurve_fips.models.sdf_decoder import SDFDecoder2,SDFDecoder,FiLMSDFDecoder
from lightcurve_fips.training.utils import positional_encoding

class LightcurveEncoderResidual(nn.Module):
    def __init__(
        self,
        num_cameras,
        num_modalities=2,
        latent_dim=256,
        pool_length=16,
        dropout=0.05,
        use_first_derivative=False,
        use_second_derivative=False,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.num_modalities = num_modalities
        self.use_first_derivative = use_first_derivative
        self.use_second_derivative = use_second_derivative

        n_feature_groups = 1

        if use_first_derivative:
            n_feature_groups += 1

        if use_second_derivative:
            n_feature_groups += 1

        input_channels = num_modalities * n_feature_groups * num_cameras

        self.input_projection = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=7,
                padding=3,
                padding_mode="circular",
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=64,
            ),
            nn.SiLU()
        )

        self.temporal_blocks = nn.Sequential(
            CircularResidualBlock(
                channels=64,
                kernel_size=7,
                dilation=1,
                dropout=dropout
            ),
            CircularResidualBlock(
                channels=64,
                kernel_size=7,
                dilation=2,
                dropout=dropout
            ),
            CircularResidualBlock(
                channels=64,
                kernel_size=7,
                dilation=4,
                dropout=dropout
            ),
            CircularResidualBlock(
                channels=64,
                kernel_size=7,
                dilation=8,
                dropout=dropout
            ),
            CircularResidualBlock(
                channels=64,
                kernel_size=7,
                dilation=16,
                dropout=dropout
            )
        )

        self.feature_projection = nn.Sequential(
            nn.Conv1d(
                64,
                128,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=128,
            ),
            nn.SiLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(pool_length)

        self.latent_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pool_length, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
        )

    def forward(self, lc):
        batch_size, modalities, cameras, frames = lc.shape

        if modalities != self.num_modalities:
            raise ValueError(
                f"Erwartete {self.num_modalities} Modalitäten, "
                f"erhalten: {modalities}"
            )

        if cameras != self.num_cameras:
            raise ValueError(
                f"Erwartete {self.num_cameras} Kameras, "
                f"erhalten: {cameras}"
            )

        x = lc.reshape(
            batch_size,
            modalities * cameras,
            frames
        )

        x = self.input_projection(x)
        x = self.temporal_blocks(x)
        x = self.feature_projection(x)
        x = self.pool(x)

        return self.latent_projection(x)

class LightcurveSDFNet(nn.Module):
    def __init__(self, num_cameras, latent_dim=256, num_freqs=6):
        super().__init__()

        self.encoder = LightcurveEncoderResidual(
            num_cameras=num_cameras,
            latent_dim=latent_dim
        )

        self.decoder = SDFDecoder2(
            latent_dim=latent_dim,
            num_freqs=num_freqs
        )

    def forward(self, lc, points, radius):
        latent = self.encoder(lc)
        sdf = self.decoder(latent, points, radius)
        return sdf

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


class PerCameraLightcurveEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.Conv1d(64, feature_dim, kernel_size=7, padding=3),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, lc):
        B, C, T = lc.shape

        x = lc.reshape(B * C, 1, T)

        features = self.net(x)

        return features.reshape(B, C, -1)

class CameraSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=256,
        hidden_dim=256,
        n_heads=8,
        n_layers=4
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU()
        )

    def forward(self, camera_tokens):
        x = self.input_proj(camera_tokens)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.to_latent(x)

class GeometryAwareLightcurveEncoder(nn.Module):
    def __init__(
        self,
        num_cameras=28,
        lightcurve_feature_dim=128,
        latent_dim=256,
        camera_freqs=4
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.camera_freqs = camera_freqs

        self.lightcurve_encoder = PerCameraLightcurveEncoder(
            feature_dim=lightcurve_feature_dim
        )

        camera_geom_dim = 3 + 2 * camera_freqs * 3

        self.camera_encoder = CameraSetEncoder(
            input_dim=lightcurve_feature_dim + camera_geom_dim,
            latent_dim=latent_dim,
            hidden_dim=256,
            n_heads=8,
            n_layers=4
        )

    def forward(self, lc, camera_dirs):
        B, C, T = lc.shape

        if C != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {C}"
            )

        lc_features = self.lightcurve_encoder(lc)

        if camera_dirs.dim() == 2:
            camera_dirs = camera_dirs.unsqueeze(0).expand(B, -1, -1)

        camera_dirs = camera_dirs.to(
            device=lc.device,
            dtype=lc.dtype
        )

        camera_geom = positional_encoding(
            camera_dirs,
            num_freqs=self.camera_freqs
        )

        camera_tokens = torch.cat(
            [lc_features, camera_geom],
            dim=-1
        )

        latent = self.camera_encoder(camera_tokens)

        return latent

class CircularResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=7,
        dilation=1,
        dropout=0.0
    ):
        super().__init__()

        padding = (
            (kernel_size - 1) * dilation
        ) // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=channels
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=channels
        )

        self.dropout = nn.Dropout1d(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return self.act(x + residual)

class LightcurveEncoder(nn.Module):
    def __init__(
        self,
        num_cameras,
        num_modalities=2,
        latent_dim=256
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.num_modalities = num_modalities

        input_channels = num_modalities * num_cameras

        self.net = nn.Sequential(
            nn.Conv1d(input_channels,64,kernel_size=7,padding=3,padding_mode="circular"),
            nn.SiLU(),

            nn.Conv1d(64,128,kernel_size=7,padding=3,padding_mode="circular"),
            nn.SiLU(),

            nn.Conv1d(128,256,kernel_size=7,padding=3,padding_mode="circular"),
            nn.SiLU(),

            nn.Conv1d(256,256,kernel_size=7,padding=3,padding_mode="circular"),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),

            nn.Linear(256 * 16, latent_dim),
            nn.SiLU()
        )

    def forward(self, lc):
        if lc.ndim != 4:
            raise ValueError(
                "Erwartete Form [Batch, Modalitäten, Kameras, Frames], "
                f"erhalten: {tuple(lc.shape)}"
            )

        batch_size, modalities, cameras, frames = lc.shape

        if modalities != self.num_modalities:
            raise ValueError(
                f"Erwartete {self.num_modalities} Modalitäten, "
                f"erhalten: {modalities}"
            )

        if cameras != self.num_cameras:
            raise ValueError(
                f"Erwartete {self.num_cameras} Kameras, "
                f"erhalten: {cameras}"
            )

        lc = lc.reshape(
            batch_size,
            modalities * cameras,
            frames
        )

        return self.net(lc)