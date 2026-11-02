# Physics-free latent diffusion baseline — notes

The diffusion model with no physics terms, on ShanghaiTech (standard split, seed 0). The physics model will be compared against this.

## Method

1. `train_ae.py`: a frame-by-frame KL autoencoder (256 to 64, 8 latent channels, 4.50M parameters), trained for 20,000 steps at batch 8 on normal frames, with L1, gradient and weak KL losses.
2. `train_ldm.py`: a factorised space-time U-Net (16.37M parameters) predicts the next frame's latent from 4 past latents. It uses a cosine schedule, v-prediction and EMA 0.999, and was trained for 40,000 steps at batch 8, the same number of clips as Liu et al. (80,000 steps at batch 4).
3. Testing: 10-step deterministic DDIM from the same fixed noise on every frame, decoded to pixels, then -PSNR, per-video min-max and the first 4 frames skipped, using the shared `eval/metrics.py`.

There was one evaluation, at the fixed endpoint, and nothing was selected on test scores except where stated under "Choice of reference frame".

## Results

| Method | Micro AUROC | Macro AUROC | FPR@95TPR | Test speed |
|---|---|---|---|---|
| Diffusion, real frame as reference | 72.18 | 81.78 | 80.09 | 5.1 FPS |
| Diffusion, autoencoder reconstruction as reference | 66.75 | 79.99 | 87.83 | 5.1 FPS |
| Liu et al. at fixed endpoint (80k), real frame as reference | 70.20 | 80.20 | 85.54 | 51 FPS |

Both diffusion scores come from the same model and the same evaluation. The speed includes data loading and the extra reconstruction step.

Against Liu on per-video AUROC (one seed, so descriptive only; Prof Wang's protocol needs 5):

- Real frame reference: +1.58 points on average, higher in 78 of 106 videos, Wilcoxon p < 0.0001.
- Reconstruction reference: -0.21 points on average, higher in 52 of 106 videos, p = 0.98.

## Choice of reference frame

**Read this before quoting the table.** The two references give different conclusions, and the choice was not made cleanly.

- Liu et al. score against the real frame. On normal test frames the autoencoder reaches 38.8 dB, below Liu's 39.8 dB for prediction, so I expected its error to mask the score.
- Before any diffusion score existed, I therefore made the reconstruction reference the primary score and kept the real frame as secondary. `results_step40000.json` still uses those labels.
- A smoke test on one video with a 50-step model ran before the full run. It showed about 95% for both scores, which carries no information, and the choice had already been made.
- That expectation was wrong: the real frame reference did better on all three metrics.

The real frame reference is the like-for-like comparison with Liu, because it is the same protocol. However, the pre-stated primary score does not beat Liu, and preferring the real frame now would be informed by test results. Both should always be reported together, with this history.

Which score to use for the physics comparison is an open decision for Prof Wang. Scoring every later model with both costs almost nothing.

## Analysis

From `analyse_scores.py`, which needs no GPU.

| | Reconstruction ref | Real frame ref | Liu |
|---|---|---|---|
| PSNR on normal / anomalous frames (dB) | 42.90 / 39.44 | 36.79 / 35.55 | 39.77 / 37.29 |
| Within-video PSNR gap (dB) | 4.90 | 2.41 | 3.51 |
| Mean within-video PSNR std (dB) | 4.46 | 2.05 | 3.15 |
| Micro with z-score instead of min-max | 73.36 | 75.34 | 73.82 |
| Micro with raw scores | 67.18 | 59.13 | 66.65 |
| Dip at multiples of 24 frames (dB) | 3.95 | 2.21 | 3.04 |
| Videos whose lowest frame is at a multiple of 24 | 100/107 | 100/107 | 97/107 |
| Micro with those frames removed (diagnostic only) | 68.72 | 72.60 | 70.93 |
| Macro without scene 01 | 75.72 | 78.00 | 75.46 |
| Videos below 0.5 AUROC | 7/106 | 6/106 | 6/106 |

- **Why the expectation failed.** The autoencoder's error did about halve the gap (4.90 to 2.41 dB), but it also halved the normal variation within each video (4.46 to 2.05 dB), so frames stayed about as separable. Gap divided by variation is 1.10 and 1.18. This suggests why it did no harm, but does not fully explain the 5.4-point micro difference.
- **Part of the difference is normalisation.** Min-max depends on each video's extremes, and the reconstruction reference has deeper dips. It loses 6.6 points going from z-score to min-max, against 3.2 for the real frame. Removing the 24-frame dips closes only about 1.5 of the 5.4 points (the difference falls to 3.9). Without normalisation the order reverses (67.18 against 59.13), which fits the autoencoder's error varying by scene (34.6 to 42 dB).
- **The 24-frame dips are not specific to a model.** They appear at the same multiples in Liu and in both diffusion scores, which points at the data rather than at either model. Compression keyframes remain the likely cause, but that is unconfirmed.
- **Weak scenes are shared.** Scenes 02, 10 and 12 are weak in all three, and scene 10's pooled micro stays below 0.5 in all three. The method's largest gains over Liu are in scenes 12 and 10 by macro (+6.7 and +4.8 points with the real frame).
- **False alarms remain the main weakness.** All three still have FPR@95TPR of at least 80%.

## Limits

- One seed and one split, so differences this small may not hold across seeds.
- Prof Wang's plan calls for a video autoencoder; this one works frame by frame.
- The autoencoder generalises imperfectly: 41.1 dB on training frames against 38.8 dB on test frames, measured the same way on about 500 frames each, lower on test in all 12 scenes. The cause is unknown.
- The 40,000-step budget matches Liu's clip count but may leave the model undertrained. The loss has been roughly flat at about 0.04-0.05 since step 10,000, which does not settle this. Any extension must be decided from training behaviour, not from test scores.
- It runs about 10 times slower than Liu at test time.

## Run history

- The first autoencoder run used 4 latent channels and a KL weight I later found to be too strong. Its results were discarded for reconstruction quality, before any diffusion run.
- Training was interrupted by laptop sleep twice and resumed from checkpoints at steps 10,000 and 20,000. `train_log.txt` therefore repeats steps 10,000-13,150 and 20,000-20,450. Resuming restores the weights, EMA and optimiser, but not the data order or random state.
- The first end-of-training evaluation froze when a data worker could not start because of low system memory, so no result was produced. It was rerun with `--eval_only` and 2 workers from the same step-40,000 checkpoint. That is the only evaluation with results.

## Reproduce

```bash
python physdiff/train_ae.py --z_ch 8
python physdiff/train_ldm.py
python physdiff/analyse_scores.py
```

Saved configs and results are in `physdiff/results/`.
