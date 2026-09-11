import torch
import torch.nn as nn
from lightcurve_fips.training.utils import positional_encoding

class SDFDecoder2(nn.Module):
    """Decode a latent lightcurve representation into SDF values at 3D points."""
    def __init__(self, latent_dim=256, num_freqs=6):
        super().__init__()

        self.num_freqs = num_freqs

        #Original 3D coordinates plus sine/cosine positional encoding.
        point_dim = 3 + 2 * num_freqs * 3

        #Latent code, encoded query point, and object radius.
        input_dim = latent_dim + point_dim + 1

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)

        #Skip connection brings in the original input again.
        self.fc3 = nn.Linear(512 + input_dim, 512)

        self.fc4 = nn.Linear(512, 256)

        #Predicting one signed distance value per query point.
        self.out = nn.Linear(256, 1)

        self.act = nn.SiLU()

    def forward(self, latent, points, radius):
        """Predict SDF values for a batch of 3D query points."""
        B, N, _ = points.shape

        #Making sure there is only one radius value per batch item.
        radius = radius.view(B)

        #Add multi-frequency features to represent fine details.
        points_enc = positional_encoding(points, self.num_freqs)

        #Repeat the global latent vector for every query point.
        latent_expanded = latent[:, None, :].expand(-1, N, -1)

        #Repeat the object radius for every query point.
        radius_expanded = radius[:, None, None].expand(-1, N, 1)

        #(B, N, latent_dim + point_dim + 1)
        x0 = torch.cat(
            [latent_expanded, points_enc, radius_expanded],
            dim=-1
        )

        h = self.act(self.fc1(x0))
        h = self.act(self.fc2(h))

        #Skip Connection, readd the original input features.
        h = torch.cat([h, x0], dim=-1)

        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))

        #Removing the singleton channel dimension.
        sdf = self.out(h)

        return sdf.squeeze(-1)