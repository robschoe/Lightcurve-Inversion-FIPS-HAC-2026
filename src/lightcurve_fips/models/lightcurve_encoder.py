import torch.nn as nn
from lightcurve_fips.models.sdf_decoder import SDFDecoder2

class LightcurveEncoderResidual(nn.Module):
    """Encode multi-camera lightcurves into a fixed-size latent representation."""
    def __init__(
        self,
        num_cameras,
        num_modalities=2,
        latent_dim=256,
        pool_length=16,
        dropout=0.05,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.num_modalities = num_modalities

        #Each camera and modality becomes one input channel.
        input_channels = num_modalities *  num_cameras

        #Initial convolution for lightcurve signals. Since they are periodic, the convolution is circular.
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

        #Dilated residual blocks capture patterns over short and long ranges.
        self.temporal_blocks = nn.Sequential(
            CircularResidualBlock(channels=64, kernel_size=7, dilation=1, dropout=dropout),
            CircularResidualBlock(channels=64, kernel_size=7, dilation=2, dropout=dropout),
            CircularResidualBlock(channels=64, kernel_size=7, dilation=4, dropout=dropout),
            CircularResidualBlock(channels=64, kernel_size=7, dilation=8, dropout=dropout),
            CircularResidualBlock(channels=64, kernel_size=7, dilation=16, dropout=dropout),
        )

        #Project temporal features to a higher-dimensional feature space.
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

        #Reduce variable-length temporal features to a fixed length.
        self.pool = nn.AdaptiveAvgPool1d(pool_length)

        #Map pooled temporal features to the latent object representation.
        self.latent_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pool_length, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
        )

    def forward(self, lc):
        """Encode lightcurves of shape (B, modalities, cameras, frames)."""
        batch_size, modalities, cameras, frames = lc.shape

        if modalities != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {modalities}."
            )

        if cameras != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {cameras}."
            )

        #Merge modalities and cameras into Conv1d input channels.
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
    """Predict SDF values from lightcurves and 3D query points."""
    def __init__(self, num_cameras, latent_dim=256, num_freqs=6):
        super().__init__()

        #Encode the lightcurve sequence into an object-level latent code.
        self.encoder = LightcurveEncoderResidual(
            num_cameras=num_cameras,
            latent_dim=latent_dim
        )

        #Decode the latent code at arbitrary 3D query positions.
        self.decoder = SDFDecoder2(
            latent_dim=latent_dim,
            num_freqs=num_freqs
        )

    def forward(self, lc, points, radius):
        #Create a latent shape representation from the input lightcurves.
        latent = self.encoder(lc)

        #Predict one SDF value for each 3D query point.
        sdf = self.decoder(latent, points, radius)

        return sdf

class CircularResidualBlock(nn.Module):
    """Residual 1D convolution block with circular padding."""
    
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

