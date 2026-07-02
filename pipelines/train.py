"""Training pipeline (T stage): pawpularity pairs -> Bradley-Terry global prior.

Reads taste_fv (photo embeddings + engagement score), draws grouped preference
pairs (taste_pairs rules), cross-validates the Bradley-Terry logistic head
against the baselines it must beat, fits the final weights on everything, and
registers `pet_taste` in the model registry with card + images.

Baselines, in ascending order of effort:
  coin        0.5 by construction (sanity floor)
  zero_shot   frozen-encoder appeal direction (data/appeal_direction.npy,
              written by the benchmark job) -- costs nothing to "train"
  ridge_rank  Ridge emb->score per photo, rank pairs by predicted score
  bt          Bradley-Terry logistic on emb diffs (champion candidate)

Honesty: C for bt is picked on a validation split of each fold's TRAIN photos
(never test photos); headline metric is 5-fold CV mean +/- SE on gap>=10 test
pairs; accuracy-vs-gap is reported on all-gap pairs.

Runs as a Hopsworks job (pandas env):
    hops job deploy taste-train pipelines/train.py \
        --env pandas-training-pipeline --run --wait --overwrite
"""
import glob
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/taste-machine")):
        if os.path.exists(os.path.join(p, "taste_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from taste_pairs import assign_folds, make_pairs, pair_features, pair_gaps  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models", "pet_taste")
FOLDS = 5
TRAIN_PAIRS = 40000
TEST_PAIRS = 8000
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
SEED = 7


def load_data():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fv = fs.get_feature_view("taste_fv", 1)
    X, y = fv.training_data(description="pawpularity pairs base")
    df = X.copy()
    df["score"] = y["score"].values
    emb = np.stack(df["emb"].map(np.asarray).values).astype(np.float64)
    scores = df["score"].values.astype(int)
    print(f"{len(df):,} photos, emb dim {emb.shape[1]}", flush=True)
    return emb, scores


def fit_bt(Xtr, ytr, Xval, yval):
    from sklearn.linear_model import LogisticRegression
    best, best_acc = None, -1
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, fit_intercept=False)
        clf.fit(Xtr, ytr)
        acc = (clf.predict(Xval) == yval).mean()
        if acc > best_acc:
            best, best_acc, best_C = clf, acc, C
    return best, best_C


def evaluate():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score

    emb, scores = load_data()
    rng = np.random.default_rng(SEED)
    fold_of = assign_folds(len(scores), FOLDS, rng)
    appeal_path = os.path.join(DATA, "appeal_direction.npy")
    w_appeal = np.load(appeal_path) if os.path.exists(appeal_path) else None
    if w_appeal is None:
        print("WARNING: no appeal_direction.npy -- zero-shot baseline skipped",
              flush=True)

    acc = {m: [] for m in ["bt", "ridge_rank", "zero_shot"]}
    aucs, chosen_C, all_gap_records, calib_records = [], [], [], []

    for k in range(FOLDS):
        tr_photos = np.where(fold_of != k)[0]
        te_photos = np.where(fold_of == k)[0]
        # C selection on a held-back slice of TRAIN photos
        cut = int(len(tr_photos) * 0.85)
        fit_p, val_p = tr_photos[:cut], tr_photos[cut:]
        fit_pairs = make_pairs(fit_p, scores, rng, TRAIN_PAIRS)
        val_pairs = make_pairs(val_p, scores, rng, TEST_PAIRS // 2)
        te_pairs = make_pairs(te_photos, scores, rng, TEST_PAIRS)
        te_all_gap = make_pairs(te_photos, scores, rng, TEST_PAIRS, min_gap=1)

        Xf, yf = pair_features(emb, fit_pairs)
        Xv, yv = pair_features(emb, val_pairs)
        Xt, yt = pair_features(emb, te_pairs)

        bt, C = fit_bt(Xf, yf, Xv, yv)
        chosen_C.append(C)
        p = bt.predict_proba(Xt)[:, 1]
        acc["bt"].append(float(((p > 0.5) == yt).mean()))
        aucs.append(float(roc_auc_score(yt, p)))
        calib_records.append((p, yt))

        Xg, yg = pair_features(emb, te_all_gap)
        pg = bt.predict_proba(Xg)[:, 1]
        all_gap_records.append((pair_gaps(scores, te_all_gap), (pg > 0.5) == yg))

        ridge = Ridge(alpha=10.0).fit(emb[fit_p], scores[fit_p])
        s = ridge.predict(emb)
        acc["ridge_rank"].append(
            float(np.mean([(s[a] > s[b]) == bool(y) for a, b, y in te_pairs])))

        if w_appeal is not None:
            za = emb @ w_appeal
            acc["zero_shot"].append(
                float(np.mean([(za[a] > za[b]) == bool(y) for a, b, y in te_pairs])))
        print(f"fold {k}: bt={acc['bt'][-1]:.3f} (C={C}) "
              f"ridge={acc['ridge_rank'][-1]:.3f}", flush=True)

    summary = {m: {"acc": float(np.mean(v)), "se": float(np.std(v) / np.sqrt(FOLDS))}
               for m, v in acc.items() if v}
    summary["coin"] = {"acc": 0.5, "se": 0.0}
    summary["bt"]["auc"] = float(np.mean(aucs))
    final_C = max(set(chosen_C), key=chosen_C.count)

    # final fit on everything with the winning C
    from sklearn.linear_model import LogisticRegression
    pairs = make_pairs(np.arange(len(scores)), scores, rng, TRAIN_PAIRS * 2)
    Xa, ya = pair_features(emb, pairs)
    final = LogisticRegression(C=final_C, max_iter=3000, fit_intercept=False)
    final.fit(Xa, ya)
    w_global = final.coef_[0]

    return summary, final_C, w_global, all_gap_records, calib_records


def make_plots(summary, all_gap_records, calib_records, out):
    os.makedirs(out, exist_ok=True)
    order = [m for m in ["coin", "zero_shot", "ridge_rank", "bt"] if m in summary]
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [summary[m]["acc"] for m in order]
    errs = [summary[m]["se"] for m in order]
    ax.bar(order, vals, yerr=errs, color=["#999", "#7aa", "#59c", "#e63"][:len(order)])
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_ylim(0.45, max(vals) + 0.05)
    ax.set_ylabel("pairwise accuracy (5-fold CV)")
    ax.set_title("Who predicts the crowd's pick?")
    fig.tight_layout(); fig.savefig(os.path.join(out, "model_comparison.png"), dpi=120)

    gaps = np.concatenate([g for g, _ in all_gap_records])
    hits = np.concatenate([h for _, h in all_gap_records])
    bins = [(1, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    accs = [hits[(gaps >= a) & (gaps < b)].mean() for a, b in bins]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f"{a}-{b}" for a, b in bins], accs, "o-", c="#e63")
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xlabel("score gap"); ax.set_ylabel("pairwise accuracy")
    ax.set_title("Easy pairs are easy: accuracy vs score gap")
    fig.tight_layout(); fig.savefig(os.path.join(out, "acc_vs_gap.png"), dpi=120)

    p = np.concatenate([p for p, _ in calib_records])
    yt = np.concatenate([y for _, y in calib_records])
    edges = np.linspace(0, 1, 11)
    mids, frac = [], []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() > 20:
            mids.append(p[m].mean()); frac.append(yt[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", c="k", lw=0.8)
    ax.plot(mids, frac, "o-", c="#e63")
    ax.set_xlabel("predicted P(left wins)"); ax.set_ylabel("observed")
    ax.set_title("Calibration (CV test pairs)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "calibration.png"), dpi=120)
    plt.close("all")


def register(summary, final_C, w_global, plots_dir):
    import hopsworks
    from taste_features import ENCODER, MODEL_ID, EMBED_DIM

    stage = os.path.join("/tmp", "pet_taste_artifact")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    np.save(os.path.join(stage, "w_global.npy"), w_global)
    meta = {"encoder": ENCODER, "model_id": MODEL_ID, "dim": EMBED_DIM,
            "C": final_C, "cv": summary, "sigma0_default": 0.3}
    with open(os.path.join(stage, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for png in glob.glob(os.path.join(plots_dir, "*.png")):
        shutil.copy(png, stage)

    proj = hopsworks.login()
    mr = proj.get_model_registry()
    model = mr.python.create_model(
        name="pet_taste",
        metrics={"cv_pairwise_acc": summary["bt"]["acc"],
                 "cv_pairwise_se": summary["bt"]["se"],
                 "cv_auc": summary["bt"]["auc"],
                 "baseline_ridge_rank": summary["ridge_rank"]["acc"],
                 "baseline_zero_shot": summary.get("zero_shot", {}).get("acc", 0.5)},
        description=f"Bradley-Terry global taste prior on frozen {ENCODER} "
                    f"embeddings. Trained on pawpularity preference pairs "
                    f"(gap>={10}), grouped CV. The per-user online layer "
                    f"(taste_online.py) starts from these weights.")
    model.save(stage)          # consumes the dir; plots were copied in first
    print(f"registered pet_taste v{model.version}", flush=True)


def main():
    summary, final_C, w_global, gap_rec, calib_rec = evaluate()
    print(json.dumps(summary, indent=2), flush=True)
    plots = os.path.join(DATA, "plots")
    make_plots(summary, gap_rec, calib_rec, plots)
    os.makedirs(MODELS, exist_ok=True)
    np.save(os.path.join(MODELS, "w_global.npy"), w_global)
    register(summary, final_C, w_global, plots)


if __name__ == "__main__":
    main()
