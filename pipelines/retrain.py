"""Flywheel (I stage): swipes -> swipe_events FG -> challenger prior.

1. Ingest data/feedback/*.jsonl (written by the app) into swipe_events v1.
   Idempotent: PK (session_id, swipe_idx) + event_time=swiped_at, so re-runs
   converge (last-activity pattern).
2. Retrain the Bradley-Terry prior on pawpularity pairs + the swipes' TRAIN
   pairs (the actively-selected ones; measure pairs stay out of training so the
   live metric never trains on itself).
3. Evaluate on the FIXED pawpularity CV folds (same SEED as train.py -- eval
   cells never move) and on measure swipes. Register a new pet_taste version
   ONLY if it holds the pawpularity CV (within 1 SE of the champion) and beats
   it on measure swipes. Otherwise log and exit -- no silent champion churn.

Supervised flywheel, NOT RL: swipes are labels, nothing optimizes engagement.

    hops job deploy predictable-retrain pipelines/retrain.py \
        --env pandas-training-pipeline --run --wait --overwrite
Schedule daily once swipes accumulate.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/how-predictable")):
        if os.path.exists(os.path.join(p, "taste_features.py")):
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from taste_pairs import assign_folds, make_pairs, pair_features   # noqa: E402

FEEDBACK = os.path.join(ROOT, "data", "feedback")
FOLDS = 5
TRAIN_PAIRS = 40000
TEST_PAIRS = 8000
SEED = 7                     # MUST match train.py -- fixed eval cells


def ingest(fs):
    rows = []
    for path in sorted(glob.glob(os.path.join(FEEDBACK, "*.jsonl"))):
        with open(path) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    if not rows:
        print("no feedback yet", flush=True)
        return None
    df = pd.DataFrame(rows)
    df["swiped_at"] = pd.to_datetime(df["swiped_at"], utc=True)
    df = df.drop(columns=["model_version"])
    for c in ("swipe_idx", "left_id", "right_id", "chose_left", "model_pick_left"):
        df[c] = df[c].astype("int64")
    fg = fs.get_feature_group("swipe_events", 1)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"ingested {len(df)} swipes from {df.session_id.nunique()} sessions",
          flush=True)
    return df


def load_pawpularity(fs):
    fv = fs.get_feature_view("taste_fv", 1)
    X, y = fv.training_data(description="retrain base")
    df = X.copy()
    df["score"] = y["score"].values
    emb = np.stack(df["emb"].map(np.asarray).values).astype(np.float64)
    return emb, df["score"].values.astype(int)


def swipe_pair_features(pool, swipes):
    emb_of = dict(zip(pool["pet_id"].astype(int),
                      pool["emb"].map(np.asarray)))
    tr = swipes[swipes.pair_kind == "train"]
    me = swipes[swipes.pair_kind == "measure"]

    def feats(df):
        X, y = [], []
        for _, r in df.iterrows():
            l, rr = emb_of.get(int(r.left_id)), emb_of.get(int(r.right_id))
            if l is None or rr is None:
                continue
            X.append(np.asarray(l, float) - np.asarray(rr, float))
            y.append(int(r.chose_left))
        return (np.stack(X), np.array(y)) if X else (None, None)

    return feats(tr), feats(me)


def main():
    import hopsworks
    from sklearn.linear_model import LogisticRegression

    proj = hopsworks.login()
    fs = proj.get_feature_store()
    swipes = ingest(fs)
    if swipes is None or (swipes.pair_kind == "train").sum() < 200:
        print("not enough train swipes to retrain (<200) -- done", flush=True)
        return

    emb, scores = load_pawpularity(fs)
    pool = fs.get_feature_group("pet_embeddings", 1).read()
    (Xs, ys), (Xm, ym) = swipe_pair_features(pool, swipes)

    mr = proj.get_model_registry()
    champ = max(mr.get_models("pet_taste"), key=lambda m: m.version)
    champ_dir = champ.download()
    meta = json.load(open(os.path.join(champ_dir, "meta.json")))
    w_champ = np.load(os.path.join(champ_dir, "w_global.npy"))
    C = meta["C"]

    rng = np.random.default_rng(SEED)
    fold_of = assign_folds(len(scores), FOLDS, rng)
    accs = []
    for k in range(FOLDS):
        tr_pairs = make_pairs(np.where(fold_of != k)[0], scores, rng, TRAIN_PAIRS)
        te_pairs = make_pairs(np.where(fold_of == k)[0], scores, rng, TEST_PAIRS)
        Xtr, ytr = pair_features(emb, tr_pairs)
        Xte, yte = pair_features(emb, te_pairs)
        if Xs is not None:
            Xtr, ytr = np.vstack([Xtr, Xs]), np.concatenate([ytr, ys])
        clf = LogisticRegression(C=C, max_iter=3000, fit_intercept=False)
        clf.fit(Xtr, ytr)
        accs.append(float((clf.predict(Xte) == yte).mean()))
    cv_acc, cv_se = float(np.mean(accs)), float(np.std(accs) / np.sqrt(FOLDS))

    # measure-swipe accuracy: challenger (full refit) vs champion weights
    all_pairs = make_pairs(np.arange(len(scores)), scores, rng, TRAIN_PAIRS * 2)
    Xa, ya = pair_features(emb, all_pairs)
    if Xs is not None:
        Xa, ya = np.vstack([Xa, Xs]), np.concatenate([ya, ys])
    final = LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(Xa, ya)
    w_new = final.coef_[0]
    m_new = m_champ = None
    if Xm is not None:
        m_new = float((((Xm @ w_new) > 0).astype(int) == ym).mean())
        m_champ = float((((Xm @ w_champ) > 0).astype(int) == ym).mean())

    champ_cv = champ.training_metrics.get("cv_pairwise_acc", 0)
    champ_se = champ.training_metrics.get("cv_pairwise_se", 0.01)
    print(f"challenger: cv {cv_acc:.4f}+/-{cv_se:.4f} (champ {champ_cv:.4f}), "
          f"measure swipes {m_new} vs champ {m_champ}", flush=True)

    holds_cv = cv_acc >= champ_cv - champ_se
    beats_live = m_new is not None and m_champ is not None and m_new > m_champ
    if not (holds_cv and beats_live):
        print("challenger NOT promoted (needs: hold CV within 1 SE AND beat "
              "champion on measure swipes)", flush=True)
        return

    import shutil
    from taste_online import TasteSpace
    stage = "/tmp/pet_taste_challenger"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    np.save(os.path.join(stage, "w_global.npy"), w_new)
    pool_emb = np.stack(pool["emb"].map(np.asarray).values).astype(np.float64)
    TasteSpace.fit(w_new, pool_emb, k=24).save(
        os.path.join(stage, "taste_space.npz"))
    meta["cv"] = {"bt": {"acc": cv_acc, "se": cv_se}}
    meta["n_swipe_train_pairs"] = int(len(ys)) if ys is not None else 0
    with open(os.path.join(stage, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    model = mr.python.create_model(
        name="pet_taste",
        metrics={"cv_pairwise_acc": cv_acc, "cv_pairwise_se": cv_se,
                 "measure_swipe_acc": m_new, "champion_measure_acc": m_champ,
                 "n_swipe_train_pairs": float(len(ys))},
        description=f"Flywheel challenger: pawpularity pairs + {len(ys)} swipe "
                    f"train pairs. Promoted over v{champ.version} on live "
                    f"measure-swipe accuracy.")
    model.save(stage)
    print(f"PROMOTED pet_taste v{model.version}", flush=True)


if __name__ == "__main__":
    main()
