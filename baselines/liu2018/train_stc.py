"""Liu et al., CVPR 2018 ("Future frame prediction for anomaly detection") on ShanghaiTech.

Uses the generator, discriminator and losses of the PyTorch reimplementation in ./repo
(github.com/feiyuhuahuo/Anomaly_Prediction) unchanged, with our shared data loader and metrics.
Deviations from that repo, all documented in baselines/liu2018/NOTES.md:
  1. Frozen flow network is torchvision RAFT-small (official weights) instead of LiteFlowNet/FlowNet2-SD.
  2. Gradients flow through the predicted-frame optical flow into the generator, as in the original
     TensorFlow code. The reimplementation detached it, which silently disabled the flow loss.
  3. ShanghaiTech labels are read from testing/test_frame_mask/*.npy (the repo's loader was broken).
  4. The discriminator's gradients are zeroed after the generator's backward pass, so the generator's
     adversarial loss no longer leaks into the discriminator update (as in the original TF code).

Usage:
    python baselines/liu2018/train_stc.py --iters 80000            # full run
    python baselines/liu2018/train_stc.py --iters 50 --max_test_videos 2   # smoke test
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "repo"))

from data.shanghaitech import ClipDataset, load_test_labels  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from losses import (Adversarial_Loss, Discriminate_Loss, Flow_Loss,  # noqa: E402
                    Gradient_Loss, Intensity_Loss)
from models.pix2pix_networks import PixelDiscriminator  # noqa: E402
from models.unet import UNet  # noqa: E402
from utils import weights_init_normal  # noqa: E402

NUM_PAST = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/Admin/datasets/shanghaitech")
    p.add_argument("--iters", type=int, default=80000)
    p.add_argument("--batch_size", type=int, default=4)
    # Learning rates and decay step follow the original ShanghaiTech configuration.
    p.add_argument("--g_lr", type=float, default=2e-4)
    p.add_argument("--d_lr", type=float, default=2e-5)
    p.add_argument("--lr_decay_at", type=int, default=50000, help="multiply both lrs by 0.1 here")
    p.add_argument("--lam_int", type=float, default=1.0)
    p.add_argument("--lam_grad", type=float, default=1.0)
    p.add_argument("--lam_flow", type=float, default=2.0)
    p.add_argument("--lam_adv", type=float, default=0.05)
    p.add_argument("--raft_iters", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=20000, help="-1 to evaluate only at the end")
    p.add_argument("--max_test_videos", type=int, default=None, help="for quick smoke tests")
    p.add_argument("--out", default=str(ROOT / "outputs" / "liu2018"))
    p.add_argument("--resume", default=None, help="path to a checkpoint .pt")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raft(device):
    net = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return net


@torch.no_grad()
def run_eval(generator, args, device):
    """Per-frame PSNR on the test set -> anomaly score = -PSNR, evaluated with eval/metrics.py."""
    generator.eval()
    ds = ClipDataset(args.data_root, split="testing", num_past=NUM_PAST)
    videos = sorted(ds.videos)[: args.max_test_videos] if args.max_test_videos else sorted(ds.videos)
    keep = set(videos)
    ds.index = [it for it in ds.index if it[0] in keep]
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=args.workers)

    psnr = {v: np.zeros(len(ds.videos[v])) for v in videos}
    t0, n = time.time(), 0
    for batch in loader:
        past = batch["past"].to(device).flatten(1, 2)  # (B, 12, H, W)
        target = batch["target"].to(device)
        pred = generator(past)
        mse = ((pred - target) ** 2).flatten(1).mean(1)
        p = (10 * torch.log10(4.0 / mse.clamp_min(1e-10))).cpu().numpy()  # range 2 for [-1, 1]
        for v, t, val in zip(batch["video"], batch["t"].tolist(), p):
            psnr[v][t] = val
        n += len(p)
    fps = n / (time.time() - t0)

    labels = load_test_labels(args.data_root, videos)
    for v in videos:
        assert len(labels[v]) == len(psnr[v]), f"{v}: {len(labels[v])} labels vs {len(psnr[v])} frames"
    scores = {v: -psnr[v] for v in videos}  # low PSNR = anomalous
    res = evaluate(scores, labels, skip_first=NUM_PAST)
    res["fps_incl_loading"] = fps
    generator.train()
    return res, psnr


def save_ckpt(path, step, generator, discriminator, opt_g, opt_d):
    torch.save({"step": step, "net_g": generator.state_dict(), "net_d": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = Path(args.out) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    generator = UNet(input_channels=3 * NUM_PAST, output_channel=3).to(device)
    discriminator = PixelDiscriminator(input_nc=3).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.g_lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ck["net_g"])
        discriminator.load_state_dict(ck["net_d"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"Resumed from {args.resume} at step {step}")
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    if args.eval_only:
        res, _ = run_eval(generator, args, device)
        print(json.dumps({k: v for k, v in res.items() if k != "per_video_auroc"}, indent=2))
        return

    flow_net = load_raft(device)
    int_loss, grad_loss = Intensity_Loss().to(device), Gradient_Loss(3).to(device)
    flow_loss, adv_loss, d_loss_fn = Flow_Loss().to(device), Adversarial_Loss().to(device), Discriminate_Loss().to(device)

    train_ds = ClipDataset(args.data_root, split="training", num_past=NUM_PAST)
    print(f"Training clips: {len(train_ds)} from {len(train_ds.videos)} videos")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=args.workers > 0)

    log = open(out / "train_log.txt", "a")
    generator.train()
    discriminator.train()
    t_last = time.time()
    while step < args.iters:
        for batch in loader:
            if step == args.lr_decay_at:
                for g in opt_g.param_groups + opt_d.param_groups:
                    g["lr"] *= 0.1
            past = batch["past"].to(device, non_blocking=True)  # (B, 4, 3, H, W), RGB in [-1, 1]
            target = batch["target"].to(device, non_blocking=True)
            last = past[:, -1]

            pred = generator(past.flatten(1, 2))

            with torch.no_grad():
                flow_gt = flow_net(last, target, num_flow_updates=args.raft_iters)[-1]
            # No detach here: the flow loss must reach the generator (deviation 2).
            flow_pred = flow_net(last, pred, num_flow_updates=args.raft_iters)[-1]

            l_int, l_grad = int_loss(pred, target), grad_loss(pred, target)
            l_flow, l_adv = flow_loss(flow_pred, flow_gt), adv_loss(discriminator(pred))
            g_total = args.lam_int * l_int + args.lam_grad * l_grad + args.lam_flow * l_flow + args.lam_adv * l_adv
            d_total = d_loss_fn(discriminator(target), discriminator(pred.detach()))

            # G backward first, then zero D so the adversarial term does not leak into D's grads
            # (deviation 4). No parameter changes before both backward passes are done.
            opt_g.zero_grad(set_to_none=True)
            g_total.backward()
            opt_d.zero_grad(set_to_none=True)
            d_total.backward()
            opt_g.step()
            opt_d.step()
            step += 1

            if step % 50 == 0:
                dt = (time.time() - t_last) / 50
                t_last = time.time()
                with torch.no_grad():
                    psnr = (10 * torch.log10(4.0 / ((pred - target) ** 2).flatten(1).mean(1))).mean().item()
                eta_h = dt * (args.iters - step) / 3600
                msg = (f"[{step}/{args.iters}] int {l_int.item():.4f} grad {l_grad.item():.4f} "
                       f"flow {l_flow.item():.4f} adv {l_adv.item():.4f} D {d_total.item():.4f} "
                       f"psnr {psnr:.2f} | {dt:.3f}s/it ETA {eta_h:.1f}h")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()

            if step % args.save_every == 0 or step == args.iters:
                save_ckpt(out / "latest.pt", step, generator, discriminator, opt_g, opt_d)

            if (args.eval_every > 0 and step % args.eval_every == 0) or step == args.iters:
                res, psnr_all = run_eval(generator, args, device)
                summary = {k: v for k, v in res.items() if k != "per_video_auroc"}
                print(f"EVAL step {step}: {json.dumps(summary)}", flush=True)
                log.write(f"EVAL step {step}: {json.dumps(summary)}\n")
                (out / f"results_step{step}.json").write_text(json.dumps({"step": step, **res}, indent=2))
                np.savez(out / f"psnr_step{step}.npz", **psnr_all)

            if step >= args.iters:
                break
    log.close()


if __name__ == "__main__":
    main()
