"""Analysis of the saved test PSNR of the physics-free baseline, next to Liu et al.: the numbers quoted in NOTES.md.
Uses no GPU and needs no checkpoint.

Usage:
    python physdiff/analyse_scores.py
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NUM_PAST = 4
PERIOD = 24


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
    p.add_argument("--run", default=str(ROOT / "outputs" / "physdiff" / "ldm_nophysics" / "seed0"))
    p.add_argument("--liu", default=str(ROOT / "outputs" / "liu2018" / "seed0" / "psnr_step80000.npz"))
    p.add_argument("--step", type=int, default=40000)
    args = p.parse_args()

    files = {"Diffusion, reconstruction ref": Path(args.run) / f"psnr_step{args.step}.npz",
             "Diffusion, real frame ref": Path(args.run) / f"psnr_real_step{args.step}.npz",
             "Liu, real frame ref": Path(args.liu)}
    loaded = {k: np.load(f) for k, f in files.items()}
    vids = sorted(loaded["Liu, real frame ref"].files)
    labels = {v: np.load(Path(args.data_root) / "testing" / "test_frame_mask" / f"{v}.npy").astype(int)[NUM_PAST:]
              for v in vids}
    two_class = [v for v in vids if 0 < labels[v].sum() < len(labels[v])]

    per_video, scene_rows = {}, {}
    for name, z in loaded.items():
        psnr = {v: z[v][NUM_PAST:] for v in vids}
        scores = {v: -psnr[v] for v in vids}
        pv = {v: roc_auc_score(labels[v], scores[v]) for v in two_class}
        per_video[name] = pv
        all_psnr = np.concatenate([psnr[v] for v in vids])
        all_lab = np.concatenate([labels[v] for v in vids])
        gap = np.mean([psnr[v][labels[v] == 0].mean() - psnr[v][labels[v] == 1].mean() for v in two_class])

        idx = {v: np.arange(NUM_PAST, NUM_PAST + len(psnr[v])) for v in vids}
        depth = np.concatenate([(median_filter(psnr[v], size=9, mode="nearest") - psnr[v])[idx[v] % PERIOD == 0]
                                for v in vids])
        keep = {v: idx[v] % PERIOD != 0 for v in vids}  # dropped before min-max: cannot score or set the range
        without = micro({v: scores[v][keep[v]] for v in vids}, {v: labels[v][keep[v]] for v in vids}, minmax)
        lowest = sum((int(np.argmin(psnr[v])) + NUM_PAST) % PERIOD == 0 for v in vids)

        print(f"\n=== {name}")
        print(f"micro min-max {micro(scores, labels, minmax):.4f} | z-score {micro(scores, labels, zscore):.4f} | "
              f"raw {micro(scores, labels):.4f} | macro {np.mean(list(pv.values())):.4f}")
        print(f"PSNR normal {all_psnr[all_lab == 0].mean():.2f} | anomalous {all_psnr[all_lab == 1].mean():.2f} | "
              f"within-video gap {gap:.2f} dB | mean within-video std {np.mean([psnr[v].std() for v in vids]):.2f} dB")
        print(f"dip at multiples of {PERIOD}: {depth.mean():.2f} dB | lowest frame there in {lowest}/{len(vids)} videos | "
              f"micro min-max without those frames {without:.4f}")
        print(f"videos below 0.5 AUROC: {sum(a < 0.5 for a in pv.values())}/{len(pv)} | "
              f"macro without scene 01: {np.mean([a for v, a in pv.items() if not v.startswith('01_')]):.4f}")
        for scene in sorted({v.split("_")[0] for v in vids}):
            in_scene = [v for v in vids if v.split("_")[0] == scene]
            scored = [pv[v] for v in in_scene if v in pv]
            scene_rows.setdefault(scene, []).append(
                (np.mean(scored), micro({v: scores[v] for v in in_scene}, labels, minmax)))

    print("\nscene | macro, then pooled micro, in the order above")
    for scene, row in sorted(scene_rows.items()):
        print(scene, "|", " | ".join(f"{a:.3f} {b:.3f}" for a, b in row))

    print("\nWilcoxon signed-rank on per-video AUROC (one seed, so descriptive only)")
    names = list(per_video)
    for a, b in ((names[1], names[2]), (names[0], names[2]), (names[1], names[0])):
        x = np.array([per_video[a][v] for v in two_class])
        y = np.array([per_video[b][v] for v in two_class])
        print(f"{a} vs {b}: mean diff {(x - y).mean():+.4f}, higher in {(x > y).sum()}/{len(x)} videos, "
              f"p={wilcoxon(x, y).pvalue:.4f}")


if __name__ == "__main__":
    main()
