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