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
from lightcurve_fips.training.utils import positional_encoding

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