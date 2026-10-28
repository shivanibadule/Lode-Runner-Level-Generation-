"""Conditional Variational Autoencoder (CVAE) for Lode Runner Level Generation.

Architecture:
- Encoder: Input (B, 8, 22, 32) concat broadcast condition c (B, 5, 22, 32) -> 13 channels.
  3x Conv2d(k3, s2, p1) + BatchNorm + LeakyReLU + Dropout (22x32 -> 11x16 -> 6x8 -> 3x4).
  Linear heads for mu and logvar (latent dim 32).
- Decoder: Concat z (32) and c (5) -> 37 dims. Linear projection -> reshape (128, 3, 4).
  3x Upsampling blocks with explicit target size enforcement (22, 32).
  Outputs raw logits (8 classes).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants


class LodeRunnerCVAE(nn.Module):
    def __init__(
        self,
        num_classes: int = constants.NUM_CLASSES,
        cond_dim: int = 5,
        latent_dim: int = 32,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # Encoder: 8 one-hot channels + 5 condition channels = 13 channels
        in_channels = num_classes + cond_dim
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1) # -> (11, 16)
        self.enc_bn1 = nn.BatchNorm2d(32)

        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> (6, 8)
        self.enc_bn2 = nn.BatchNorm2d(64)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> (3, 4)
        self.enc_bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.flatten_dim = 128 * 3 * 4 # 1536
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: Latent z (32) + condition c (5) = 37 dims
        self.dec_fc = nn.Linear(latent_dim + cond_dim, self.flatten_dim)

        self.dec_conv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
        ) # (3, 4) -> (6, 8)
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.dec_conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)
        ) # (6, 8) -> (11, 16)
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.dec_conv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
        ) # (11, 16) -> (22, 32)
        self.dec_bn3 = nn.BatchNorm2d(16)

        self.dec_final = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Broadcast condition c into extra channels and encode to mu and logvar."""
        B, _, H, W = x.shape
        # Broadcast c (B, 5) -> (B, 5, H, W)
        c_spatial = c.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W)
        x_in = torch.cat([x, c_spatial], dim=1) # (B, 13, 22, 32)

        h = self.leaky_relu(self.enc_bn1(self.enc_conv1(x_in)))
        h = self.dropout(h)
        h = self.leaky_relu(self.enc_bn2(self.enc_conv2(h)))
        h = self.dropout(h)
        h = self.leaky_relu(self.enc_bn3(self.enc_conv3(h)))
        h = self.dropout(h)

        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * std with reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent z conditioned on c into logits of shape (B, 8, 22, 32)."""
        B = z.shape[0]
        zc = torch.cat([z, c], dim=1) # (B, 37)
        h = self.dec_fc(zc)
        h = h.view(B, 128, 3, 4)

        h = self.leaky_relu(self.dec_bn1(self.dec_conv1(h)))
        h = self.leaky_relu(self.dec_bn2(self.dec_conv2(h)))
        h = self.leaky_relu(self.dec_bn3(self.dec_conv3(h)))

        logits = self.dec_final(h)

        # Force exact final dimensions (22, 32)
        if logits.shape[-2:] != (constants.ROWS, constants.COLS):
            logits = F.interpolate(
                logits, size=(constants.ROWS, constants.COLS), mode="bilinear", align_corners=False
            )
        return logits

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning (logits, mu, logvar)."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, c)
        return logits, mu, logvar
