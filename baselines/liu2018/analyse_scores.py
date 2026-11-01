"""Analysis of the saved test PSNR: the numbers quoted in NOTES.md. Uses no GPU and needs no checkpoint.

Usage:
    python baselines/liu2018/analyse_scores.py
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
NUM_PAST = 4
PERIOD = 24
STEPS = (20000, 40000, 60000, 80000)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def zscore(x):
    return (x - x.mean()) / x.std()


def micro(scores, labels, norm=None):
    vids = sorted(scores)
    s = np.concatenate([norm(scores[v]) if norm else scores[v] for v in vids])
    return roc_auc_score(np.concatenate([labels[v] for v in vids]), s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="C:/Users/Admin/datasets/shanghaitech")
    p.add_argument("--run", default=str(ROOT / "outputs" / "liu2018" / "seed0"))
    args = p.parse_args()

    labels = None
    print("step | micro min-max | micro z-score | micro raw | macro | within-video PSNR gap | "
          "dip at multiples of 24 | micro min-max without those frames")
    for step in STEPS:
        z = np.load(Path(args.run) / f"psnr_step{step}.npz")
        vids = sorted(z.files)
        if labels is None:
            labels = {v: np.load(Path(args.data_root) / "testing" / "test_frame_mask" / f"{v}.npy").astype(int)[NUM_PAST:]
                      for v in vids}
        psnr = {v: z[v][NUM_PAST:] for v in vids}
        scores = {v: -psnr[v] for v in vids}
        two_class = [v for v in vids if 0 < labels[v].sum() < len(labels[v])]
        macro = np.mean([roc_auc_score(labels[v], scores[v]) for v in two_class])
        gap = np.mean([psnr[v][labels[v] == 0].mean() - psnr[v][labels[v] == 1].mean() for v in two_class])

        idx = {v: np.arange(NUM_PAST, NUM_PAST + len(psnr[v])) for v in vids}
        depth = {v: median_filter(psnr[v], size=9, mode="nearest") - psnr[v] for v in vids}
        at_mult = np.concatenate([depth[v][idx[v] % PERIOD == 0] for v in vids])
        keep = {v: idx[v] % PERIOD != 0 for v in vids}  # dropped before min-max: cannot score or set the range
        without = micro({v: scores[v][keep[v]] for v in vids}, {v: labels[v][keep[v]] for v in vids}, minmax)
        print(f"{step} | {micro(scores, labels, minmax):.4f} | {micro(scores, labels, zscore):.4f} | "
              f"{micro(scores, labels):.4f} | {macro:.4f} | {gap:.2f} | {at_mult.mean():.2f} | {without:.4f}")

    # Everything below is at the final step.
    lowest = sum((int(np.argmin(psnr[v])) + NUM_PAST) % PERIOD == 0 for v in vids)
    print(f"\nVideos whose lowest-PSNR frame is at a multiple of {PERIOD}: {lowest} of {len(vids)}")

    per_video = {v: roc_auc_score(labels[v], scores[v]) for v in two_class}
    print(f"Videos below 0.5 AUROC: {sum(a < 0.5 for a in per_video.values())} of {len(per_video)}")
    print("Macro AUROC without scene 01:", round(np.mean([a for v, a in per_video.items() if not v.startswith("01_")]), 4))
    print("\nscene | videos scored | macro | pooled micro")
    for scene in sorted({v.split("_")[0] for v in vids}):
        in_scene = [v for v in vids if v.split("_")[0] == scene]
        scored = [v for v in in_scene if v in per_video]
        print(f"{scene} | {len(scored)} | {np.mean([per_video[v] for v in scored]):.3f} | "
              f"{micro({v: scores[v] for v in in_scene}, labels, minmax):.3f}")


if __name__ == "__main__":
    main()
