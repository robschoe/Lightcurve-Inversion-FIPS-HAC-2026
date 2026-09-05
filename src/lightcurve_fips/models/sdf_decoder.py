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

class SDFDecoder2(nn.Module):
    def __init__(self, latent_dim=256, num_freqs=6):
        super().__init__()

        self.num_freqs = num_freqs
        point_dim = 3 + 2 * num_freqs * 3

        input_dim = latent_dim + point_dim + 1

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512 + input_dim, 512)

        self.fc4 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

        self.act = nn.SiLU()

    def forward(self, latent, points, radius):
        B, N, _ = points.shape

        radius = radius.view(B)

        points_enc = positional_encoding(points, self.num_freqs)

        latent_expanded = latent[:, None, :].expand(-1, N, -1)
        radius_expanded = radius[:, None, None].expand(-1, N, 1)

        # (B, N, latent_dim + point_dim + 1)
        x0 = torch.cat(
            [latent_expanded, points_enc, radius_expanded],
            dim=-1
        )

        h = self.act(self.fc1(x0))
        h = self.act(self.fc2(h))

        # Skip Connection
        h = torch.cat([h, x0], dim=-1)

        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))

        sdf = self.out(h)

        return sdf.squeeze(-1)

class FiLMSDFDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_freqs=6, hidden_dim=512):
        super().__init__()

        self.num_freqs = num_freqs

        point_dim = 3 + 2 * num_freqs * 3
        input_dim = point_dim + 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, 1)

        self.film1 = nn.Linear(latent_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(latent_dim, 2 * hidden_dim)
        self.film3 = nn.Linear(latent_dim, 2 * hidden_dim)
        self.film4 = nn.Linear(latent_dim, 2 * hidden_dim)

        self.act = nn.SiLU()

    def apply_film(self, x, latent, film_layer):
        gamma_beta = film_layer(latent)

        gamma, beta = torch.chunk(
            gamma_beta,
            chunks=2,
            dim=-1
        )

        gamma = gamma[:, None, :]
        beta = beta[:, None, :]

        return gamma * x + beta

    def forward(self, latent, points, radius):
        B, N, _ = points.shape

        points_enc = positional_encoding(points, self.num_freqs)

        radius = radius.view(B, 1, 1).expand(-1, N, 1)

        x = torch.cat([points_enc, radius], dim=-1)

        h = self.fc1(x)
        h = self.apply_film(h, latent, self.film1)
        h = self.act(h)

        h = self.fc2(h)
        h = self.apply_film(h, latent, self.film2)
        h = self.act(h)

        h = self.fc3(h)
        h = self.apply_film(h, latent, self.film3)
        h = self.act(h)

        h = self.fc4(h)
        h = self.apply_film(h, latent, self.film4)
        h = self.act(h)

        return self.out(h).squeeze(-1)