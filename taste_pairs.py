"""Preference-pair construction from scored photos. Shared by the encoder
benchmark, the training pipeline, and the flywheel retrain -- one definition of
what a pair is, one grouping rule.

Grouping rule (leak rule of this project): photos are assigned to folds; pairs
are drawn WITHIN a fold's photo set only, so no photo ever appears on any side
of two different splits.
"""
import numpy as np

MIN_GAP = 10        # score gap below which a preference is noise, not signal


def assign_folds(n, n_folds, rng):
    return rng.integers(0, n_folds, n)


def make_pairs(idx, scores, rng, n_pairs, min_gap=MIN_GAP):
    """Sample (a, b, y) index pairs from idx with |score gap| >= min_gap;
    y=1 if a wins. Balanced by construction (a,b order is random)."""
    idx = np.asarray(idx)
    pairs, tries = [], 0
    while len(pairs) < n_pairs:
        tries += 1
        if tries > n_pairs * 200:
            break                        # tiny fold; return what we have
        a, b = rng.choice(idx, 2, replace=False)
        if abs(int(scores[a]) - int(scores[b])) < min_gap:
            continue
        pairs.append((int(a), int(b), 1 if scores[a] > scores[b] else 0))
    return pairs


def pair_features(emb, pairs):
    X = np.stack([emb[a] - emb[b] for a, b, _ in pairs])
    y = np.array([y for _, _, y in pairs])
    return X, y


def pair_gaps(scores, pairs):
    return np.array([abs(int(scores[a]) - int(scores[b])) for a, b, _ in pairs])
