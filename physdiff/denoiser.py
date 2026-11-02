"""Factorised space-time U-Net (Ho et al., 2022) that denoises the future frame's latent given past latents.

Input is (B, T, C, H, W): the T-1 clean past latents followed by the noisy future latent.
Output is the v-prediction for the future frame only, (B, C, H, W).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=1)


def zero_(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class SpaceTimeBlock(nn.Module):
    """A spatial residual block on each frame, then a temporal convolution across frames."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), zero_(nn.Conv2d(out_ch, out_ch, 3, padding=1)))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.temporal = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), zero_(nn.Conv1d(out_ch, out_ch, 3, padding=1)))

    def forward(self, x, emb):
        b, f, c, h, w = x.shape
        x2d = x.reshape(b * f, c, h, w)
        y = self.conv1(x2d) + self.emb(emb).repeat_interleave(f, dim=0)[:, :, None, None]
        y = self.conv2(y) + self.skip(x2d)
        c = y.shape[1]
        y1d = y.reshape(b, f, c, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, c, f)
        y1d = y1d + self.temporal(y1d)
        return y1d.reshape(b, h, w, c, f).permute(0, 4, 3, 1, 2)


class Attention(nn.Module):
    """Self-attention over a sequence (B, N, C)."""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = zero_(nn.Linear(ch, ch))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return x + self.proj(y)


class SpaceTimeAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.spatial = Attention(ch)
        self.temporal = Attention(ch)

    def forward(self, x):
        b, f, c, h, w = x.shape
        y = self.spatial(x.permute(0, 1, 3, 4, 2).reshape(b * f, h * w, c)).reshape(b, f, h, w, c)
        y = self.temporal(y.permute(0, 2, 3, 1, 4).reshape(b * h * w, f, c)).reshape(b, h, w, f, c)
        return y.permute(0, 3, 4, 1, 2)


class SpaceTimeUNet(nn.Module):
    def __init__(self, z_ch=4, base=64, mults=(1, 2, 4), blocks=2):
        super().__init__()
        emb_dim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.head = nn.Conv2d(z_ch + 1, base, 3, padding=1)  # +1: a channel marking the noisy frame

        self.downs, self.samplers_down = nn.ModuleList(), nn.ModuleList()
        ch, skips = base, [base]
        for i, mult in enumerate(mults):
            level = nn.ModuleList()
            for _ in range(blocks):
                level.append(SpaceTimeBlock(ch, base * mult, emb_dim))
                ch = base * mult
                skips.append(ch)
            self.downs.append(level)
            if i < len(mults) - 1:
                self.samplers_down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)

        self.mid1 = SpaceTimeBlock(ch, ch, emb_dim)
        self.mid_attn = SpaceTimeAttention(ch)
        self.mid2 = SpaceTimeBlock(ch, ch, emb_dim)

        self.ups, self.samplers_up = nn.ModuleList(), nn.ModuleList()
        for i, mult in reversed(list(enumerate(mults))):
            level = nn.ModuleList()
            for _ in range(blocks + 1):
                level.append(SpaceTimeBlock(ch + skips.pop(), base * mult, emb_dim))
                ch = base * mult
            self.ups.append(level)
            if i > 0:
                self.samplers_up.append(nn.Conv2d(ch, ch, 3, padding=1))

        self.tail = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), zero_(nn.Conv2d(ch, z_ch, 3, padding=1)))

    @staticmethod
    def _per_frame(fn, x):
        b, f = x.shape[:2]
        y = fn(x.reshape(b * f, *x.shape[2:]))
        return y.reshape(b, f, *y.shape[1:])

    def forward(self, past, noisy, t):
        """past: (B, T-1, C, H, W), noisy: (B, C, H, W), t: (B,) integer timesteps."""
        x = torch.cat([past, noisy[:, None]], dim=1)
        mark = torch.zeros_like(x[:, :, :1])
        mark[:, -1] = 1
        x = torch.cat([x, mark], dim=2)
        emb = self.time_mlp(timestep_embedding(t, self.base))

        h = self._per_frame(self.head, x)
        hs = [h]
        for i, level in enumerate(self.downs):
            for block in level:
                h = block(h, emb)
                hs.append(h)
            if i < len(self.samplers_down):
                h = self._per_frame(self.samplers_down[i], h)
                hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for i, level in enumerate(self.ups):
            for block in level:
                h = block(torch.cat([h, hs.pop()], dim=2), emb)
            if i < len(self.samplers_up):
                up = lambda y: F.interpolate(y, scale_factor=2, mode="nearest")  # noqa: E731
                h = self._per_frame(lambda y: self.samplers_up[i](up(y)), h)

        return self.tail(h[:, -1])
