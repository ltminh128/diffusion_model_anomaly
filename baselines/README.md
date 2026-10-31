# Baselines

Each baseline is cloned into `baselines/<name>/repo/` (git-ignored) and run on our splits.
Scores are exported as `{video_name: per-frame score}` and evaluated with `eval/metrics.py`,
so every method is measured with exactly the same protocol.

| Baseline | Official code | Notes |
|---|---|---|
| Liu et al., CVPR 2018 | https://github.com/StevenLiuWen/ano_pred_cvpr2018 (TensorFlow 1.x) | TF 1.x does not run on RTX 50-series GPUs. Plan: PyTorch port https://github.com/feiyuhuahuo/Anomaly_Prediction, report it as a reimplementation next to the published 72.8% micro AUROC. Confirm with Prof Wang. |
| Physics-free diffusion | ours (`models/`) | Same architecture and training budget as the full model, all physics terms off. |
| MNAD, HF2-VAD, FPDM, MA-PDM, MoCoDAD, STG-NF, Attribute-based, AED-MAE | see literature review | Later phase. |
