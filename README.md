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
- [ ] Reproduce Liu et al. (CVPR 2018) on ShanghaiTech
- [ ] Physics-free latent diffusion baseline on ShanghaiTech
