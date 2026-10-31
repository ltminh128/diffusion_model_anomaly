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
