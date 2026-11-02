"""Physics-free latent diffusion baseline on ShanghaiTech (step 2, after physdiff/train_ae.py).

Given 4 past frames, predicts the next frame's latent with few-step DDIM, decodes it, and scores each frame with
-PSNR against the autoencoder's reconstruction of the real frame. Liu et al. score against the real frame itself, but
the autoencoder's own error (38.8 dB on normal test frames, below Liu's 39.8 dB) would then cap the score. This choice
was fixed from reconstruction quality alone, before any diffusion score existed; the real-frame score is kept as a
secondary result. Every test frame starts from the same noise, so scores do not jitter between frames because of the
noise draw. The result to report is the fixed endpoint.

Usage:
    python physdiff/train_ldm.py
    python physdiff/train_ldm.py --iters 50 --max_test_videos 2   # smoke test
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.shanghaitech import ClipDataset, load_test_labels  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from physdiff.autoencoder import FrameAutoencoder  # noqa: E402
from physdiff.denoiser import SpaceTimeUNet  # noqa: E402
from physdiff.diffusion import Diffusion  # noqa: E402

NUM_PAST = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/Admin/datasets/shanghaitech")
    p.add_argument("--ae_ckpt", default=str(ROOT / "outputs" / "physdiff" / "ae" / "seed0" / "latest.pt"))
    # 40k steps at batch 8 sees the same number of clips as Liu et al. (80k steps at batch 4).
    p.add_argument("--iters", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ddim_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=-1, help="-1 to evaluate only at the end (one evaluation takes hours)")
    p.add_argument("--max_test_videos", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "physdiff" / "ldm_nophysics"))
    p.add_argument("--resume", default=None)
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def encode_clip(ae, x, scale):
    """(B, T, 3, H, W) or (B, 3, H, W) frames -> scaled latents of the same layout."""
    frames = x.reshape(-1, *x.shape[-3:])
    z = ae.encode(frames)[0] * scale
    return z.reshape(*x.shape[:-3], *z.shape[-3:])


@torch.no_grad()
def latent_scale(ae, loader, device, batches=50):
    """1 / std of training latents, so the diffusion model sees unit-scale data (Rombach et al., 2022)."""
    zs = []
    for i, batch in enumerate(loader):
        if i == batches:
            break
        zs.append(ae.encode(batch["target"].to(device))[0].flatten().cpu())
    return 1.0 / torch.cat(zs).std().item()


@torch.no_grad()
def run_eval(model, ae, diffusion, scale, args, device):
    model.eval()
    ds = ClipDataset(args.data_root, split="testing", num_past=NUM_PAST)
    videos = sorted(ds.videos)[: args.max_test_videos] if args.max_test_videos else sorted(ds.videos)
    keep = set(videos)
    ds.index = [it for it in ds.index if it[0] in keep]
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=args.workers)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = None
    psnr = {v: np.zeros(len(ds.videos[v])) for v in videos}
    psnr_real = {v: np.zeros(len(ds.videos[v])) for v in videos}
    t0, n = time.time(), 0
    for batch in loader:
        target = batch["target"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            past = encode_clip(ae, batch["past"].to(device), scale)
            if noise is None:
                noise = torch.randn(1, *past.shape[2:], device=device, generator=gen)
            z = diffusion.ddim_sample(model, past, noise.expand(len(past), -1, -1, -1), args.ddim_steps)
            pred = ae.decode(z / scale)
            recon = ae.decode(ae.encode(target)[0])
        pred = pred.float().clamp(-1, 1)
        for store, ref in ((psnr, recon.float().clamp(-1, 1)), (psnr_real, target)):
            mse = ((pred - ref) ** 2).flatten(1).mean(1)
            p = (10 * torch.log10(4.0 / mse.clamp_min(1e-10))).cpu().numpy()
            for v, t, val in zip(batch["video"], batch["t"].tolist(), p):
                store[v][t] = val
        n += len(target)
        if n % 800 == 0:
            print(f"eval {n}/{len(ds)} frames | {n / (time.time() - t0):.1f} FPS", flush=True)
    fps = n / (time.time() - t0)

    labels = load_test_labels(args.data_root, videos)
    for v in videos:
        assert len(labels[v]) == len(psnr[v]), f"{v}: {len(labels[v])} labels vs {len(psnr[v])} frames"
    res = evaluate({v: -psnr[v] for v in videos}, labels, skip_first=NUM_PAST)
    real = evaluate({v: -psnr_real[v] for v in videos}, labels, skip_first=NUM_PAST)
    res["secondary_real_frame"] = {k: v for k, v in real.items() if k != "per_video_auroc"}
    res["fps_incl_loading"] = fps
    model.train()
    return res, psnr, psnr_real


def save_eval(out, step, res, psnr, psnr_real):
    (out / f"results_step{step}.json").write_text(json.dumps({"step": step, **res}, indent=2))
    np.savez(out / f"psnr_step{step}.npz", **psnr)
    np.savez(out / f"psnr_real_step{step}.npz", **psnr_real)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    ae_ck = torch.load(args.ae_ckpt, map_location=device)
    ae = FrameAutoencoder(z_ch=ae_ck["z_ch"]).to(device).eval()
    ae.load_state_dict(ae_ck["model"])
    for prm in ae.parameters():
        prm.requires_grad_(False)
    print(f"Loaded autoencoder at step {ae_ck['step']}")

    model = SpaceTimeUNet(z_ch=ae_ck["z_ch"]).to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(device=device)
    print(f"Denoiser parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_ds = ClipDataset(args.data_root, split="training", num_past=NUM_PAST)
    print(f"Training clips: {len(train_ds)} from {len(train_ds.videos)} videos")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0)

    step, scale = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, scale = ck["step"], ck["scale"]
        print(f"Resumed from {args.resume} at step {step}")
    if scale is None:
        scale = latent_scale(ae, loader, device)
    print(f"Latent scale: {scale:.4f}")

    if args.eval_only:
        res, psnr_all, psnr_real = run_eval(ema, ae, diffusion, scale, args, device)
        print(json.dumps({k: v for k, v in res.items() if k != "per_video_auroc"}, indent=2))
        save_eval(out, step, res, psnr_all, psnr_real)
        return

    log = open(out / "train_log.txt", "a")
    model.train()
    t_last = time.time()
    while step < args.iters:
        for batch in loader:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                past = encode_clip(ae, batch["past"].to(device, non_blocking=True), scale)
                x0 = encode_clip(ae, batch["target"].to(device, non_blocking=True), scale)
            past, x0 = past.float(), x0.float()
            t = torch.randint(0, diffusion.num_steps, (len(x0),), device=device)
            noise = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_v = model(past, diffusion.q_sample(x0, t, noise), t)
                loss = F.mse_loss(pred_v, diffusion.v_target(x0, t, noise))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1 - args.ema)
            step += 1

            if step % 50 == 0:
                dt = (time.time() - t_last) / 50
                t_last = time.time()
                msg = f"[{step}/{args.iters}] loss {loss.item():.4f} | {dt:.3f}s/it ETA {dt * (args.iters - step) / 3600:.1f}h"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()

            if step % args.save_every == 0 or step == args.iters:
                torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                            "opt": opt.state_dict(), "scale": scale}, out / "latest.pt")

            if (args.eval_every > 0 and step % args.eval_every == 0) or step == args.iters:
                torch.save({"step": step, "ema": ema.state_dict(), "scale": scale}, out / f"ema_step{step}.pt")
                res, psnr_all, psnr_real = run_eval(ema, ae, diffusion, scale, args, device)
                summary = {k: v for k, v in res.items() if k != "per_video_auroc"}
                print(f"EVAL step {step}: {json.dumps(summary)}", flush=True)
                log.write(f"EVAL step {step}: {json.dumps(summary)}\n")
                save_eval(out, step, res, psnr_all, psnr_real)

            if step >= args.iters:
                break
    log.close()


if __name__ == "__main__":
    main()
