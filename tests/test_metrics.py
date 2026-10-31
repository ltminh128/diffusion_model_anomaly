import numpy as np

from eval.metrics import evaluate, psnr


def test_perfect_scores_give_auroc_one():
    labels = {"01_0001": np.array([0, 0, 1, 1, 0]), "02_0001": np.array([0, 1, 1, 0, 0])}
    scores = {v: l.astype(float) for v, l in labels.items()}
    r = evaluate(scores, labels)
    assert r["micro_auroc"] == 1.0
    assert r["macro_auroc"] == 1.0
    assert r["fpr@95tpr"] == 0.0


def test_skip_first_drops_frames():
    labels = {"01_0001": np.array([1, 1, 0, 0, 1, 1])}
    scores = {"01_0001": np.array([0.0, 0.0, 0.1, 0.2, 0.9, 0.8])}
    # first two frames would be wrongly scored, skipping them gives a perfect result
    assert evaluate(scores, labels, skip_first=2)["micro_auroc"] == 1.0


def test_psnr_identical_images_is_high():
    x = np.zeros((2, 3, 8, 8))
    assert (psnr(x, x) > 90).all()
