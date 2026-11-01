# Liu et al. (CVPR 2018) — reproduction notes

**Paper:** W. Liu, W. Luo, D. Lian, S. Gao, "Future frame prediction for anomaly detection – A new baseline", CVPR 2018.
Published ShanghaiTech result: **72.8% frame-level AUROC (micro)**.

## Code used
- Official code (TensorFlow 1.x): https://github.com/StevenLiuWen/ano_pred_cvpr2018.
  TF 1.x has no builds for CUDA 12 / RTX 50-series GPUs, so it cannot run on this machine.
- PyTorch reimplementation used instead: https://github.com/feiyuhuahuo/Anomaly_Prediction
  (cloned into `repo/`, commit `6295f5d`). Its README reports Ped2 95.6% and Avenue 84.6%
  (original: 95.4% / 84.9%). It was never run on ShanghaiTech.
- Our entry point: `train_stc.py`, which imports the repo's UNet generator, pixel discriminator and losses unchanged.

## Deviations from the reimplementation (and why)
| # | Change | Reason |
|---|---|---|
| 1 | Frozen flow network = torchvision **RAFT-small** (official weights) instead of LiteFlowNet / FlowNet2-SD | LiteFlowNet needs `cupy` custom kernels and weights from an unofficial Google Drive link; RAFT is a stronger, maintained, differentiable flow estimator with official weights. |
| 2 | **Gradient flows through the predicted-frame flow** | The reimplementation calls `.detach()` on `flow_pred`, so the flow loss never reaches the generator — the flow constraint was silently disabled. The original TF code back-propagates through the frozen flow network. |
| 3 | ShanghaiTech labels from `testing/test_frame_mask/*.npy` | The repo's `load_shanghaitech()` globs a directory instead of the `.npy` files and would fail. |
| 4 | Discriminator grads zeroed **after** the generator backward pass | In the reimplementation the generator's adversarial loss also accumulates gradients in the discriminator. The original TF code optimises the two networks over separate variable lists. |
| 5 | Shared data loader (`data/shanghaitech.py`) and metrics (`eval/metrics.py`) | So every method is evaluated identically (micro/macro AUROC, FPR@95TPR, per-video AUROC). |

## Hyper-parameters (original ShanghaiTech setting)
Input 4 frames at 256×256 → predict 5th · batch 4 · 80k iterations ·
G lr 2e-4, D lr 2e-5, both ×0.1 at 50k · loss weights: intensity 1, gradient 1, flow 2, adversarial 0.05.
Anomaly score = −PSNR, min–max normalised per video; first 4 frames of each video excluded.

Verified against the official `Codes/training_hyper_params/hyper_params.ini` ([shanghaitech] section:
LAM_LP=1 with L2, LAM_GDL=1, LAM_FLOW=2, LAM_ADV=0.05, LRATE_G=[2e-4, 2e-5], LRATE_D=[2e-5, 2e-6],
boundaries [50000]) and the official README training command (`--iters 80000`, default batch 4).

## Results (ShanghaiTech standard split, seed 0)
| Step | Micro AUROC | Macro AUROC | FPR@95TPR |
|---|---|---|---|
| 20k | 72.55% | 80.41% | 81.51% |
| 40k | 71.63% | 80.45% | 81.12% |
| 60k | 70.37% | 80.21% | 85.10% |
| **80k (fixed endpoint)** | **70.20%** | **80.20%** | **85.54%** |
| *Published* | *72.8%* | — | — |

Test speed at 80k: 51 FPS including data loading. Per-video AUROC is in `results/`; diagnostic figures are in `figures/`.

**Comparing with the paper.** The official README says the model is chosen as the best one on the test set, and its code
saves a checkpoint every 1,000 steps and keeps the highest AUROC. Its scoring otherwise matches ours (per-video min-max,
first 4 frames skipped, frames pooled). Selected the same way from the 4 checkpoints we evaluated, our result is 72.55%,
against 72.8%. The README does not state specifically how 72.8% was produced, so this is the likely explanation, not a proven one.
We report 70.20% when comparing methods, because selecting on the test set inflates the score.

**Limits.** One seed. The standard split, not the leave-scenes-out folds. The run was interrupted at step 71.6k when the laptop
slept and resumed from the 70k checkpoint, so `train_log.txt` contains steps 70k–71.6k twice. Only the latest checkpoint is kept.

## Analysis
Reproduce with `analyse_scores.py` and `diagnose.py`.

- **Micro AUROC falls during training, but the model is not getting worse at separating frames.** From 20k to 80k the within-video
  PSNR gap between normal and anomalous frames grows from 2.53 to 3.51 dB, micro AUROC without normalisation rises from 59.5% to 66.7%,
  and macro stays flat. With z-score instead of min-max normalisation, micro barely moves (74.15% to 73.82%). So most of the fall comes
  from min-max normalisation. These alternatives are diagnostics only, not results.
- **Periodic dips.** PSNR drops by about 3 dB at every multiple of 24 frames, regardless of content, and the lowest-PSNR frame of 97 of
  107 test videos falls on one. Likely compression keyframes (not confirmed). Removing those frames gives 70.93% at 80k, and micro still falls
  during training (72.73% to 70.93%), so they explain only part of the fall. Any PSNR-based score may be affected.
- **Error lands on the anomaly but is diluted by whole-frame PSNR.** On every 10th anomalous frame of the 106 scored videos: anomalies cover
  a median 1.2% of the frame, per-pixel error is 37× higher inside than outside (above 1× in every video), yet only 33% of the frame's
  error is inside. Per-video AUROC correlates with the inside/outside ratio (Spearman 0.44, p < 0.001) and with anomaly area (0.30, p = 0.002).
- **Typical failures.** Tiny, distant anomalies (`10_0042`, AUROC 17.6%) and slow ones (`04_0004`, 30.0%). Normal people close to the camera produce large errors.
- **Uneven across scenes.** Macro ranges from about 90% (scene 01) to about 67–68% (scenes 10 and 12), and pooled micro in scene 10 is 48%. Scene 01 holds
  34 of the 106 scored videos: without it, macro is about 75.5%. 6 videos score below 50%.
