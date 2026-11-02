"""Frame autoencoder for latent diffusion: 256x256x3 frames <-> 64x64x4 latents, applied frame by frame.

The factor of 4 is deliberately small: anomalies cover a median 1.2% of a ShanghaiTech frame
(baselines/liu2018/NOTES.md), and a coarser latent would lose them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                 nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class FrameAutoencoder(nn.Module):
    def __init__(self, base=64, z_ch=4):
        super().__init__()
        self.z_ch = z_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), ResBlock(base, base),
            nn.Conv2d(base, base, 3, stride=2, padding=1), ResBlock(base, base * 2),
            nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1), ResBlock(base * 2, base * 4), ResBlock(base * 4, base * 4),
            nn.GroupNorm(8, base * 4), nn.SiLU(), nn.Conv2d(base * 4, 2 * z_ch, 3, padding=1))
        self.dec_in = nn.Sequential(nn.Conv2d(z_ch, base * 4, 3, padding=1), ResBlock(base * 4, base * 4), ResBlock(base * 4, base * 2))
        self.dec_up1 = ResBlock(base * 2, base)
        self.dec_up2 = ResBlock(base, base)
        self.dec_out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def encode(self, x):
        """Returns (mean, logvar) of the latent. Latent diffusion uses the mean."""
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec_up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.dec_up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.dec_out(h)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        kl = 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean()
        return self.decode(z), kl
