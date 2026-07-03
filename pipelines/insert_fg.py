"""Insert embedded parquets into the feature store.

Three FGs, one FV:
  pet_embeddings     v1  pool pets: pet_id PK, emb, n_photos. ONLINE (the
                         predictor looks pairs up at request time).
  pawpularity_photos v1  scored photos: photo_id PK, emb, score. Offline; the
                         training pairs come from here.
  swipe_events       v1  the game's own clicks (flywheel). Created empty-schema
                         here so the app can insert from day one. event_time =
                         swiped_at (last-activity pattern).
  taste_fv           v1  pawpularity_photos[emb -> score] for training.

    python pipelines/insert_fg.py
"""
import glob
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
EMB = os.path.join(ROOT, "data", "emb")


def main():
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    pool_paths = sorted(glob.glob(os.path.join(EMB, "pool_*.parquet")))
    pool = pd.concat([pd.read_parquet(p) for p in pool_paths], ignore_index=True)
    pool = pool.drop_duplicates(subset=["pet_id"]).reset_index(drop=True)
    pool["emb"] = pool["emb"].map(lambda v: np.asarray(v, dtype=np.float32))
    pool["ingested_at"] = pd.Timestamp.utcnow()
    print(f"pool: {len(pool):,} pets from {len(pool_paths)} shards")

    fg_pool = fs.get_or_create_feature_group(
        name="pet_embeddings", version=1,
        description="Frozen-encoder embeddings (L2-normalized) of petfinder lead "
                    "photos -- the pair pool the game draws from. Photos live on "
                    "HopsFS data/pool/<pet_id>.jpg; the store holds vectors.",
        primary_key=["pet_id"], event_time="ingested_at", online_enabled=True)
    fg_pool.insert(pool, write_options={"wait_for_job": False})
    print("pet_embeddings v1 inserted (materializing async)")

    paw_path = os.path.join(EMB, "pawpularity.parquet")
    if os.path.exists(paw_path):
        paw = pd.read_parquet(paw_path)
        paw["emb"] = paw["emb"].map(lambda v: np.asarray(v, dtype=np.float32))
        paw["ingested_at"] = pd.Timestamp.utcnow()
        fg_paw = fs.get_or_create_feature_group(
            name="pawpularity_photos", version=1,
            description="Pawpularity photos as embeddings + engagement score "
                        "(0-100, PetFinder.my page analytics). The cold-start "
                        "prior trains on preference pairs drawn from these.",
            primary_key=["photo_id"], event_time="ingested_at",
            online_enabled=False)
        fg_paw.insert(paw, write_options={"wait_for_job": False})
        print(f"pawpularity_photos v1 inserted ({len(paw):,} rows)")

        query = fg_paw.select(["photo_id", "emb", "score"])
        fs.get_or_create_feature_view(
            name="taste_fv", version=1, query=query, labels=["score"],
            description="Embedding -> engagement score. Training draws "
                        "preference pairs grouped by photo (no photo on both "
                        "sides of a fold).")
        print("taste_fv v1 ready")
    else:
        print("no pawpularity.parquet yet (kaggle step pending) -- skipped")

    swipes = pd.DataFrame({
        "session_id": pd.Series(["bootstrap"], dtype="string"),
        "swipe_idx": pd.Series([-1], dtype="int64"),
        "left_id": pd.Series([0], dtype="int64"),
        "right_id": pd.Series([0], dtype="int64"),
        "chose_left": pd.Series([0], dtype="int64"),
        "model_pick_left": pd.Series([0], dtype="int64"),
        "p_left_global": pd.Series([0.5], dtype="float64"),
        "p_left_personal": pd.Series([0.5], dtype="float64"),
        "pair_kind": pd.Series(["measure"], dtype="string"),
        "swiped_at": pd.Series([pd.Timestamp.utcnow()]),
    })
    fg_sw = fs.get_or_create_feature_group(
        name="swipe_events", version=1,
        description="Every played pair: what was shown, what the models "
                    "predicted, what the user chose. pair_kind=train pairs are "
                    "actively selected and feed retraining; pair_kind=measure "
                    "pairs are random and are the ONLY ones accuracy is "
                    "computed on. The flywheel input.",
        primary_key=["session_id", "swipe_idx"], event_time="swiped_at",
        online_enabled=False)
    fg_sw.insert(swipes, write_options={"wait_for_job": False})
    print("swipe_events v1 ready (bootstrap row)")


if __name__ == "__main__":
    main()
