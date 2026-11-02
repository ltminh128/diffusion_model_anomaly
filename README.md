# VAD-PhysDiff

Physics-informed diffusion models for unsupervised video anomaly detection.
URECA project, NTU — supervisor: Prof Wang Lipo.

## Setup

```bash
conda create -n vad python=3.11
conda activate vad
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Datasets live outside the repo; set their paths in `configs/paths.yaml`.

### ShanghaiTech Campus
1. Download from https://svip-lab.github.io/dataset/campus_dataset.html
2. Extract to `C:/Users/Admin/datasets/shanghaitech`
3. Extract training frames: `python scripts/extract_frames.py --root C:/Users/Admin/datasets/shanghaitech`

## Layout

| Folder | Contents |
|---|---|
| `data/` | Dataset loaders (clips of past frames + target frame) |
| `physdiff/` | Our physics-informed latent diffusion model |
| `baselines/` | Wrappers and notes for reproduced baselines |
| `eval/` | Shared metrics: micro/macro AUROC, FPR@95TPR, per-video AUROC |
| `configs/` | Dataset paths and leave-scenes-out folds |
| `scripts/` | Data preparation and run scripts |
| `docs/` | Literature review and technical report (LaTeX) |

## Status
- [x] Reproduce Liu et al. (CVPR 2018) on ShanghaiTech: 70.2% micro / 80.2% macro AUROC at the fixed endpoint, 72.6% when selected on the test set as the paper does (published 72.8%). See `baselines/liu2018/NOTES.md`.
- [x] Physics-free latent diffusion baseline on ShanghaiTech: 72.2% micro / 81.8% macro AUROC scored against the real frame, as Liu et al. do; 66.8% micro / 80.0% macro scored against the autoencoder's reconstruction, the score chosen as primary before testing. Read the reference-frame section of `physdiff/NOTES.md` before quoting either.
