import torch
import torch.nn as nn
from lightcurve_fips.training.utils import positional_encoding

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

        #(B, N, latent_dim + point_dim + 1)
        x0 = torch.cat(
            [latent_expanded, points_enc, radius_expanded],
            dim=-1
        )

        h = self.act(self.fc1(x0))
        h = self.act(self.fc2(h))

        #Skip Connection
        h = torch.cat([h, x0], dim=-1)

        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))

        sdf = self.out(h)

        return sdf.squeeze(-1)