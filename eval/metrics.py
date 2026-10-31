"""Frame-level evaluation shared by every method, so all numbers are computed the same way.

Scores are per-frame anomaly scores (higher = more anomalous). Following common practice,
scores are min-max normalised within each video before pooling.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def psnr(pred, target):
    """PSNR between images in [-1, 1]; works on numpy arrays or torch tensors (per sample)."""
    mse = ((pred - target) ** 2).reshape(len(pred), -1).mean(1)
    # peak-to-peak range is 2 for [-1, 1] images
    return 10 * np.log10(4.0 / np.maximum(np.asarray(mse, dtype=np.float64), 1e-10))


def normalise_per_video(scores: dict) -> dict:
    out = {}
    for v, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        rng = s.max() - s.min()
        out[v] = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
    return out


def fpr_at_tpr(labels, scores, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(fpr[np.searchsorted(tpr, tpr_target, side="left")])


def evaluate(scores: dict, labels: dict, skip_first: int = 0) -> dict:
    """
    scores, labels: {video_name: per-frame array}, same length per video.
    skip_first: frames at the start of each video with no prediction (num_past for Liu et al.).
    Returns micro AUROC, macro AUROC, FPR@95TPR and per-video AUROC (for the Wilcoxon test).
    """
    scores = normalise_per_video({v: s[skip_first:] for v, s in scores.items()})
    labels = {v: np.asarray(labels[v])[skip_first:] for v in scores}

    all_s = np.concatenate([scores[v] for v in scores])
    all_l = np.concatenate([labels[v] for v in scores])

    per_video = {}
    for v in scores:
        if 0 < labels[v].sum() < len(labels[v]):  # AUROC undefined for single-class videos
            per_video[v] = roc_auc_score(labels[v], scores[v])

    return {
        "micro_auroc": float(roc_auc_score(all_l, all_s)),
        "macro_auroc": float(np.mean(list(per_video.values()))) if per_video else float("nan"),
        "fpr@95tpr": fpr_at_tpr(all_l, all_s),
        "per_video_auroc": per_video,
    }
