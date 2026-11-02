"""Gaussian diffusion with a cosine schedule, v-prediction and deterministic DDIM sampling.

v-prediction keeps the x0 estimate stable at high noise, which few-step sampling depends on
and which the physics guidance will be evaluated on later.
"""
import math

import torch


class Diffusion:
    def __init__(self, num_steps=1000, s=0.008, device="cpu"):
        self.num_steps = num_steps
        grid = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
        f = torch.cos((grid / num_steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).float().to(device)

    def _coeffs(self, t, ndim):
        ab = self.alpha_bar[t].reshape(-1, *[1] * (ndim - 1))
        return ab.sqrt(), (1 - ab).sqrt()

    def q_sample(self, x0, t, noise):
        a, s = self._coeffs(t, x0.ndim)
        return a * x0 + s * noise

    def v_target(self, x0, t, noise):
        a, s = self._coeffs(t, x0.ndim)
        return a * noise - s * x0

    def x0_eps_from_v(self, x, t, v):
        a, s = self._coeffs(t, x.ndim)
        return a * x - s * v, a * v + s * x

    def ddim_timesteps(self, steps):
        return torch.linspace(self.num_steps - 1, 0, steps).round().long().tolist()

    @torch.no_grad()
    def ddim_sample(self, model, past, noise, steps):
        """Deterministic DDIM from `noise` to the predicted future latent, using `steps` model evaluations."""
        x = noise
        ts = self.ddim_timesteps(steps)
        for i, step in enumerate(ts):
            t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            x0, eps = self.x0_eps_from_v(x, t, model(past, x, t))
            if i == len(ts) - 1:
                return x0
            ab_prev = self.alpha_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
