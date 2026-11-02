import torch

from physdiff.autoencoder import FrameAutoencoder
from physdiff.denoiser import SpaceTimeBlock, SpaceTimeUNet
from physdiff.diffusion import Diffusion


def test_autoencoder_shapes():
    ae = FrameAutoencoder(base=16)
    x = torch.randn(2, 3, 64, 64)
    mean, _ = ae.encode(x)
    assert mean.shape == (2, 4, 16, 16)
    rec, kl = ae(x)
    assert rec.shape == x.shape and kl.ndim == 0


def test_denoiser_shape_and_conditioning():
    net = SpaceTimeUNet(base=16)
    past, noisy = torch.randn(2, 4, 4, 16, 16), torch.randn(2, 4, 16, 16)
    t = torch.tensor([10, 500])

    # Temporal layers start at zero, so they must receive gradients or past frames could never be used.
    torch.nn.init.normal_(net.tail[-1].weight, std=0.1)
    net(past, noisy, t).pow(2).mean().backward()
    temporal = [m.temporal[-1].weight.grad for m in net.modules() if isinstance(m, SpaceTimeBlock)]
    assert any(g.abs().sum() > 0 for g in temporal)

    # With non-zero temporal layers, both the past and the noisy frame must reach the output.
    for m in net.modules():
        if isinstance(m, SpaceTimeBlock):
            torch.nn.init.normal_(m.temporal[-1].weight, std=0.1)
    out = net(past, noisy, t)
    assert out.shape == noisy.shape
    assert not torch.allclose(out, net(past + 1, noisy, t))
    assert not torch.allclose(out, net(past, noisy + 1, t))


def test_schedule_runs_from_clean_to_noise():
    ab = Diffusion().alpha_bar
    assert len(ab) == 1000 and ab[0] > 0.999 and ab[-1] < 1e-3
    assert (ab[1:] < ab[:-1]).all()


def test_v_prediction_inverts():
    d = Diffusion()
    x0, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    t = torch.tensor([0, 400, 999])
    x_t = d.q_sample(x0, t, noise)
    rec_x0, rec_eps = d.x0_eps_from_v(x_t, t, d.v_target(x0, t, noise))
    assert torch.allclose(rec_x0, x0, atol=1e-4)
    assert torch.allclose(rec_eps, noise, atol=1e-4)


def test_ddim_recovers_x0_with_a_perfect_model():
    d = Diffusion()
    x0, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    calls = []

    def oracle(past, x, t):
        calls.append(1)
        a, s = d._coeffs(t, x.ndim)
        eps = (x - a * x0) / s  # the noise that would give x from the true x0
        return a * eps - s * x0

    out = d.ddim_sample(oracle, None, noise, steps=10)
    assert len(calls) == 10
    assert torch.allclose(out, x0, atol=1e-3)
