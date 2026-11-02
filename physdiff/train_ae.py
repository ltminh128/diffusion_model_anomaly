"""Train the frame autoencoder on normal training frames (step 1 of the physics-free baseline).

The anomaly score is computed in pixel space, so reconstruction error sets a floor on it. After training,
reconstruction PSNR is measured on a sample of test frames, as a check only: nothing is selected on it.

Usage:
    python physdiff/train_ae.py --iters 20000
    python physdiff/train_ae.py --iters 50 --max_test_videos 2   # smoke test
"""
import argparse
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
from physdiff.autoencoder import FrameAutoencoder  # noqa: E402


class FrameDataset(ClipDataset):
    """Single frames, so training does not decode whole clips."""

    def __init__(self, root, split="training"):
        super().__init__(root, split=split, num_past=0)

    def __getitem__(self, i):
        name, t = self.index[i]
        return self._load(self.videos[name][t])


def gradient_loss(pred, target):
    dx = (pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return dx.abs().mean() + dy.abs().mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/Admin/datasets/shanghaitech")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lam_grad", type=float, default=1.0)
    p.add_argument("--lam_kl", type=float, default=1e-6)
    p.add_argument("--z_ch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--sample_every", type=int, default=20, help="use every Nth test frame for the final check")
    p.add_argument("--max_test_videos", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "physdiff" / "ae"))
    p.add_argument("--resume", default=None)
    return p.parse_args()


@torch.no_grad()
def check_reconstruction(model, args, device):
    model.eval()
    ds = FrameDataset(args.data_root, split="testing")
    videos = sorted(ds.videos)[: args.max_test_videos]
    labels = load_test_labels(args.data_root, videos)
    psnrs = {0: [], 1: []}
    for v in videos:
        frames = ds.videos[v]
        for start in range(0, len(frames), args.sample_every * 16):
            ts = list(range(start, min(start + args.sample_every * 16, len(frames)), args.sample_every))
            x = torch.stack([ds._load(frames[t]) for t in ts]).to(device)
            mean, _ = model.encode(x)
            mse = ((model.decode(mean).clamp(-1, 1) - x) ** 2).flatten(1).mean(1)
            for t, val in zip(ts, (10 * torch.log10(4.0 / mse)).tolist()):
                psnrs[int(labels[v][t])].append(val)
    model.train()
    return {"psnr_normal": float(np.mean(psnrs[0])) if psnrs[0] else None,
            "psnr_anomalous": float(np.mean(psnrs[1])) if psnrs[1] else None,
            "frames": len(psnrs[0]) + len(psnrs[1])}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    model = FrameAutoencoder(z_ch=args.z_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Autoencoder parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"Resumed from {args.resume} at step {step}")

    ds = FrameDataset(args.data_root)
    print(f"Training frames: {len(ds)} from {len(ds.videos)} videos")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)

    log = open(out / "train_log.txt", "a")
    t_last = time.time()
    while step < args.iters:
        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec, kl = model(x)
                l1 = F.l1_loss(rec, x)
                grad = gradient_loss(rec, x)
                # Rombach et al. weight a KL sum by 1e-6 against a reconstruction loss summed over pixels. Ours is a
                # per-pixel mean, so the KL is divided by the pixel count to keep that balance.
                loss = l1 + args.lam_grad * grad + args.lam_kl * kl / x[0].numel()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            if step % 50 == 0:
                dt = (time.time() - t_last) / 50
                t_last = time.time()
                psnr = (10 * torch.log10(4.0 / ((rec.float().clamp(-1, 1) - x) ** 2).flatten(1).mean(1))).mean().item()
                msg = (f"[{step}/{args.iters}] l1 {l1.item():.4f} grad {grad.item():.4f} kl {kl.item():.1f} "
                       f"psnr {psnr:.2f} | {dt:.3f}s/it ETA {dt * (args.iters - step) / 3600:.1f}h")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()

            if step % args.save_every == 0 or step == args.iters:
                torch.save({"step": step, "z_ch": args.z_ch, "model": model.state_dict(), "opt": opt.state_dict()},
                           out / "latest.pt")

            if step >= args.iters:
                break
    log.close()

    res = check_reconstruction(model, args, device)
    print(f"RECON {json.dumps(res)}", flush=True)
    (out / "reconstruction.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
