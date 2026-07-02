"""Encoder shoot-out: which frozen backbone best predicts pairwise preference?

Embeds a subset of pawpularity photos with each candidate (CLIP B/32, SigLIP,
DINOv2), builds score-gap preference pairs, and cross-validates a logistic head
on embedding differences. GROUPED folds: a photo appears on one side of one
split only. The winner gets pinned as taste_features.ENCODER (manual edit, on
purpose -- the pin is a reviewed decision, not a side effect).

Runs as a Hopsworks job (torch env):

    hops job deploy taste-benchmark tools/benchmark_encoders.py \
        --env torch-training-pipeline --run --wait --overwrite

Prints per-encoder pairwise accuracy (CV mean +/- SE) and the zero-shot appeal
baseline. Also writes data/benchmark_encoders.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/taste-machine")):
        if os.path.exists(os.path.join(p, "taste_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from taste_features import embed_images, zero_shot_appeal, ENCODERS   # noqa: E402

DATA = os.path.join(ROOT, "data")
N_IMAGES = 3000          # benchmark subset; full embed happens in the fleet
MIN_GAP = 10             # score-gap threshold for a confident preference pair
PAIRS_PER_FOLD = 4000
FOLDS = 5
SEED = 7


def load_subset():
    meta = pd.read_csv(os.path.join(DATA, "pawpularity", "train.csv"))
    rng = np.random.default_rng(SEED)
    meta = meta.iloc[rng.permutation(len(meta))[:N_IMAGES]].reset_index(drop=True)
    imgs, ids, scores = [], [], []
    for _, m in meta.iterrows():
        p = os.path.join(DATA, "pawpularity", "train", f"{m.Id}.jpg")
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        ids.append(m.Id)
        scores.append(int(m.Pawpularity))
    return imgs, np.array(ids), np.array(scores)


def make_pairs(idx, scores, rng, n_pairs):
    """Sample index pairs with a confident score gap; y=1 if first wins."""
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.choice(idx, 2, replace=False)
        if abs(scores[a] - scores[b]) < MIN_GAP:
            continue
        pairs.append((a, b, 1 if scores[a] > scores[b] else 0))
    return pairs


def cv_pairwise_acc(emb, scores, rng):
    """Grouped CV: photos split into folds; pairs drawn within fold-train /
    fold-test so no photo crosses the boundary."""
    from sklearn.linear_model import LogisticRegression
    n = len(scores)
    fold_of = rng.integers(0, FOLDS, n)
    accs = []
    for k in range(FOLDS):
        tr_idx = np.where(fold_of != k)[0]
        te_idx = np.where(fold_of == k)[0]
        tr = make_pairs(tr_idx, scores, rng, PAIRS_PER_FOLD)
        te = make_pairs(te_idx, scores, rng, PAIRS_PER_FOLD // 4)
        Xtr = np.stack([emb[a] - emb[b] for a, b, _ in tr])
        ytr = np.array([y for _, _, y in tr])
        Xte = np.stack([emb[a] - emb[b] for a, b, _ in te])
        yte = np.array([y for _, _, y in te])
        clf = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
        clf.fit(Xtr, ytr)
        accs.append(float((clf.predict(Xte) == yte).mean()))
    return float(np.mean(accs)), float(np.std(accs) / np.sqrt(FOLDS))


def main():
    imgs, ids, scores = load_subset()
    print(f"{len(imgs)} photos, score mean {scores.mean():.1f}", flush=True)
    rng = np.random.default_rng(SEED)
    results = {}

    # zero-shot appeal baseline (no training at all)
    zs = zero_shot_appeal(imgs)
    te = make_pairs(np.arange(len(ids)), scores, rng, PAIRS_PER_FOLD)
    zs_acc = float(np.mean([(zs[a] > zs[b]) == bool(y) for a, b, y in te]))
    results["zero_shot_clip"] = {"acc": zs_acc}
    print(f"zero-shot appeal baseline: {zs_acc:.3f}", flush=True)

    for key in ENCODERS:
        emb = embed_images(imgs, encoder=key)
        acc, se = cv_pairwise_acc(emb, scores, np.random.default_rng(SEED))
        results[key] = {"acc": acc, "se": se}
        print(f"{key}: pairwise acc {acc:.3f} +/- {se:.3f}", flush=True)

    with open(os.path.join(DATA, "benchmark_encoders.json"), "w") as f:
        json.dump(results, f, indent=2)
    best = max(ENCODERS, key=lambda k: results[k]["acc"])
    print(f"WINNER: {best} -- pin taste_features.ENCODER accordingly", flush=True)


if __name__ == "__main__":
    main()
