"""Diagnostics for the trained Liu et al. model: where does the prediction error actually land?

Writes to <run>/diagnostics:
  <video>.png  score curve, plus real / predicted / error map for two chosen frames:
               A = anomalous frame with the highest PSNR (hardest to detect),
               B = normal frame with the lowest PSNR (most likely false alarm).
  stats.json   for a sample of anomalous frames over all test videos, the share of the frame covered by
               the anomaly and how concentrated the error is inside it.

Usage:
    python baselines/liu2018/diagnose.py --videos 10_0042 04_0004 01_0132
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "repo"))

from data.shanghaitech import ClipDataset, load_test_labels  # noqa: E402
from models.unet import UNet  # noqa: E402

NUM_PAST = 4
SERIES, MASK = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ERR_VMAX = 0.5  # same error scale on every map, so maps can be compared


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/Admin/datasets/shanghaitech")
    p.add_argument("--run", default=str(ROOT / "outputs" / "liu2018" / "seed0"))
    p.add_argument("--step", type=int, default=80000, help="which saved PSNR file matches the checkpoint")
    p.add_argument("--videos", nargs="+", default=["10_0042", "04_0004", "01_0132"])
    p.add_argument("--sample_every", type=int, default=10, help="use every Nth anomalous frame for the stats")
    return p.parse_args()


@torch.no_grad()
def predict(generator, ds, video, times, device, batch_size=16):
    """Returns (pred, target) as CPU tensors in [-1, 1] for the given frame indices of one video."""
    frames = ds.videos[video]
    preds, targets = [], []
    for i in range(0, len(times), batch_size):
        chunk = times[i:i + batch_size]
        past = torch.stack([torch.cat([ds._load(frames[k]) for k in range(t - NUM_PAST, t)]) for t in chunk])
        targets.append(torch.stack([ds._load(frames[t]) for t in chunk]))
        preds.append(generator(past.to(device)).cpu())
    return torch.cat(preds), torch.cat(targets)


def to_image(x):
    return ((x.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def load_mask(data_root, video):
    return np.load(Path(data_root) / "testing" / "test_pixel_mask" / f"{video}.npy", mmap_mode="r")


def mask_at(masks, t):
    return cv2.resize(np.asarray(masks[t], dtype=np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0


def video_figure(video, psnr, label, picks, pred, target, masks, auroc, path):
    fig = plt.figure(figsize=(11, 11), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.8, 1, 1], hspace=0.28, wspace=0.05)

    ax = fig.add_subplot(grid[0, :], facecolor=SURFACE)
    xs = np.arange(NUM_PAST, len(psnr))
    ax.fill_between(xs, 0, 1, where=label[NUM_PAST:] > 0, transform=ax.get_xaxis_transform(),
                    color=MASK, alpha=0.15, linewidth=0, step="mid")
    ax.plot(xs, psnr[NUM_PAST:], color=SERIES, linewidth=2)
    for name, t, _ in picks:
        ax.plot(t, psnr[t], "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(name, (t, psnr[t]), xytext=(0, 10), textcoords="offset points", ha="center",
                    color=INK, fontweight="bold")
    ax.set_title(f"{video}: PSNR per frame, shaded where labelled anomalous (AUROC {auroc:.3f})",
                 color=INK, loc="left")
    ax.set_xlabel("Frame", color=INK_2)
    ax.set_ylabel("PSNR (dB)", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    for row, (name, t, desc) in enumerate(picks, start=1):
        err = (pred[row - 1] - target[row - 1]).abs().mean(0).numpy()
        m = mask_at(masks, t)
        panels = [(to_image(target[row - 1]), f"{name}: real frame {t}, {desc}"),
                  (to_image(pred[row - 1]), f"Predicted, PSNR {psnr[t]:.1f} dB"),
                  (err, "Error map (darker = larger error)")]
        for col, (img, title) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            if col == 2:
                ax.imshow(img, cmap="Blues", vmin=0, vmax=ERR_VMAX)
            else:
                ax.imshow(img)
            if m.any() and col != 1:
                ax.contour(m, levels=[0.5], colors=MASK, linewidths=1.5)
            ax.set_title(title, color=INK, fontsize=9, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda")
    out = Path(args.run) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    generator = UNet(input_channels=3 * NUM_PAST, output_channel=3).to(device)
    ck = torch.load(Path(args.run) / "latest.pt", map_location=device)
    generator.load_state_dict(ck["net_g"])
    generator.eval()
    print(f"Loaded checkpoint at step {ck['step']}")
    assert ck["step"] == args.step, "checkpoint does not match the saved PSNR file"

    ds = ClipDataset(args.data_root, split="testing", num_past=NUM_PAST)
    saved = np.load(Path(args.run) / f"psnr_step{args.step}.npz")
    per_video = json.loads((Path(args.run) / f"results_step{args.step}.json").read_text())["per_video_auroc"]
    videos = sorted(ds.videos)
    labels = load_test_labels(args.data_root, videos)

    for video in args.videos:
        psnr, label = saved[video], labels[video]
        ts = np.arange(NUM_PAST, len(psnr))
        anomalous, normal = ts[label[NUM_PAST:] > 0], ts[label[NUM_PAST:] == 0]
        picks = [("A", int(anomalous[psnr[anomalous].argmax()]), "anomalous"),
                 ("B", int(normal[psnr[normal].argmin()]), "normal")]
        pred, target = predict(generator, ds, video, [t for _, t, _ in picks], device)
        # The saved scores came from batches of 16 in eval mode, so these must agree.
        check = 10 * torch.log10(4.0 / ((pred - target) ** 2).flatten(1).mean(1)).numpy()
        print(video, "recomputed PSNR", check.round(2), "saved", psnr[[t for _, t, _ in picks]].round(2))
        video_figure(video, psnr, label, picks, pred, target, load_mask(args.data_root, video),
                     per_video[video], out / f"{video}.png")

    rows = {}
    for video in videos:
        ts = np.arange(NUM_PAST, len(labels[video]))
        anomalous = ts[labels[video][NUM_PAST:] > 0][:: args.sample_every]
        if len(anomalous) == 0:
            continue
        masks = load_mask(args.data_root, video)
        pred, target = predict(generator, ds, video, anomalous.tolist(), device)
        err = ((pred - target) ** 2).mean(1).numpy()
        area, ratio, share = [], [], []
        for e, t in zip(err, anomalous):
            m = mask_at(masks, t)
            if m.any() and not m.all():
                area.append(m.mean())
                ratio.append(e[m].mean() / max(e[~m].mean(), 1e-10))
                share.append(e[m].sum() / max(e.sum(), 1e-10))
        if area:
            rows[video] = {"frames": len(area), "area": float(np.median(area)), "ratio": float(np.median(ratio)),
                           "error_share": float(np.median(share)), "auroc": per_video.get(video)}

    scored = [r for r in rows.values() if r["auroc"] is not None]
    area, ratio, share, auroc = (np.array([r[k] for r in scored]) for k in ("area", "ratio", "error_share", "auroc"))
    rho_area, p_area = spearmanr(area, auroc)
    rho_ratio, p_ratio = spearmanr(ratio, auroc)
    summary = {
        "videos": len(scored),
        "median_anomaly_area": float(np.median(area)),
        "median_error_ratio_inside_vs_outside": float(np.median(ratio)),
        "median_error_share_inside": float(np.median(share)),
        "videos_with_ratio_below_1": int((ratio < 1).sum()),
        "spearman_area_vs_auroc": [float(rho_area), float(p_area)],
        "spearman_ratio_vs_auroc": [float(rho_ratio), float(p_ratio)],
    }
    print(json.dumps(summary, indent=2))
    for video in args.videos:
        print(video, rows.get(video))
    (out / "stats.json").write_text(json.dumps({"summary": summary, "per_video": rows}, indent=2))


if __name__ == "__main__":
    main()
