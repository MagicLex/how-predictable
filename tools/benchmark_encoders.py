"""Encoder shoot-out: which frozen backbone best predicts pairwise preference?

Embeds a subset of pawpularity photos with each candidate (CLIP B/32, SigLIP,
DINOv2), builds score-gap preference pairs (taste_pairs rules -- the same rules
training uses), and cross-validates a logistic head on embedding differences.
The winner gets pinned as taste_features.ENCODER (manual edit, on purpose --
the pin is a reviewed decision, not a side effect).

The zero-shot appeal baseline (CLIP prompts, no training) is computed here once
and written to the JSON; train.py quotes it from there. Only CLIP's BPE
tokenizer ever runs (the siglip tokenizer needs sentencepiece, absent from the
stock torch env -- see BLOCKERS).

    hops job deploy predictable-benchmark tools/benchmark_encoders.py \
        --env torch-training-pipeline --run --wait --overwrite

Output: data/benchmark_encoders.json
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
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/how-predictable")):
        if os.path.exists(os.path.join(p, "taste_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from taste_features import embed_images, zero_shot_appeal, ENCODERS     # noqa: E402
from taste_pairs import assign_folds, make_pairs, pair_features         # noqa: E402

DATA = os.path.join(ROOT, "data")
N_IMAGES = 3000          # benchmark subset; the full embed happens in the fleet
PAIRS_PER_FOLD = 4000
FOLDS = 5
SEED = 7


def load_subset():
    meta = pd.read_csv(os.path.join(DATA, "pawpularity", "train.csv"))
    rng = np.random.default_rng(SEED)
    meta = meta.iloc[rng.permutation(len(meta))[:N_IMAGES]].reset_index(drop=True)
    imgs, scores = [], []
    for _, m in meta.iterrows():
        p = os.path.join(DATA, "pawpularity", "train", f"{m.Id}.jpg")
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        scores.append(int(m.Pawpularity))
    return imgs, np.array(scores)


def cv_pairwise_acc(emb, scores, rng):
    from sklearn.linear_model import LogisticRegression
    fold_of = assign_folds(len(scores), FOLDS, rng)
    accs = []
    for k in range(FOLDS):
        tr = make_pairs(np.where(fold_of != k)[0], scores, rng, PAIRS_PER_FOLD)
        te = make_pairs(np.where(fold_of == k)[0], scores, rng, PAIRS_PER_FOLD // 4)
        Xtr, ytr = pair_features(emb, tr)
        Xte, yte = pair_features(emb, te)
        clf = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
        clf.fit(Xtr, ytr)
        accs.append(float((clf.predict(Xte) == yte).mean()))
    return float(np.mean(accs)), float(np.std(accs) / np.sqrt(FOLDS))


def main():
    imgs, scores = load_subset()
    print(f"{len(imgs)} photos, score mean {scores.mean():.1f}", flush=True)
    results = {}

    zs = zero_shot_appeal(imgs)
    te = make_pairs(np.arange(len(scores)), scores,
                    np.random.default_rng(SEED), PAIRS_PER_FOLD)
    zs_acc = float(np.mean([(zs[a] > zs[b]) == bool(y) for a, b, y in te]))
    results["zero_shot"] = {"acc": zs_acc}
    print(f"zero-shot appeal baseline: {zs_acc:.3f}", flush=True)

    for key in ENCODERS:
        emb = embed_images(imgs, encoder=key)
        acc, se = cv_pairwise_acc(emb, scores, np.random.default_rng(SEED))
        results[key] = {"acc": acc, "se": se}
        print(f"{key}: pairwise acc {acc:.3f} +/- {se:.3f}", flush=True)

    best = max(ENCODERS, key=lambda k: results[k]["acc"])
    results["winner"] = best
    with open(os.path.join(DATA, "benchmark_encoders.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"WINNER: {best} -- pin taste_features.ENCODER accordingly", flush=True)


if __name__ == "__main__":
    main()
