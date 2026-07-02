"""how-predictable -- the machine learns your taste, live.

Two pets, click the one you like. Before you click, the model has secretly
predicted your pick. The line at the top is how often it reads you right --
watch it climb as the per-user posterior sharpens.

This app IS the v1 online inference pipeline: features from the feature store
(pet_embeddings FG, read once per pod), weights from the model registry
(pet_taste champion, pulled at pod start), per-user layer in session state,
swipes appended to data/feedback/*.jsonl for the scheduled flywheel job.

Honesty rules, enforced here:
- accuracy is computed ONLY on randomly-chosen "measure" pairs; the actively
  selected "train" pairs update the posterior but never score.
- the chart shows the frozen global model next to the personalized one; the
  gap is the personalization, the rest is "everyone likes puppies".
"""
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from taste_online import UserPosterior, global_prob, select_pair   # noqa: E402

POOL_DIR = ROOT / "data" / "pool"
FEEDBACK_DIR = ROOT / "data" / "feedback"
MEASURE_EVERY = 3          # every 3rd pair is a random measure pair
CAND_PAIRS = 40            # candidate pool for active selection
ROLL = 12                  # rolling window (measure pairs) for the headline

st.set_page_config(page_title="how predictable.", page_icon="🐶", layout="centered")


@st.cache_resource
def _login():
    import hopsworks
    return hopsworks.login()


@st.cache_resource
def load_model():
    proj = _login()
    mr = proj.get_model_registry()
    models = mr.get_models("pet_taste")
    champ = max(models, key=lambda m: m.version)
    d = champ.download()
    w = np.load(os.path.join(d, "w_global.npy"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return w, meta, champ.version


@st.cache_resource
def load_pool():
    proj = _login()
    fs = proj.get_feature_store()
    fg = fs.get_feature_group("pet_embeddings", 1)
    df = fg.read()
    have_photo = {int(p.stem) for p in POOL_DIR.glob("*.jpg")}
    df = df[df["pet_id"].isin(have_photo)].reset_index(drop=True)
    emb = np.stack(df["emb"].map(np.asarray).values).astype(np.float64)
    return df["pet_id"].to_numpy(), emb


def _log_swipe(row):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIR / f"{st.session_state.sid}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _new_pair(pet_ids, emb, post, rng):
    n = len(pet_ids)
    k = st.session_state.n_swipes
    kind = "measure" if k % MEASURE_EVERY == 0 else "train"
    if kind == "measure":
        i, j = rng.choice(n, 2, replace=False)
    else:
        cands = [tuple(rng.choice(n, 2, replace=False)) for _ in range(CAND_PAIRS)]
        feats = [emb[a] - emb[b] for a, b in cands]
        i, j = cands[select_pair(post, feats, rng=rng)]
    x = emb[i] - emb[j]
    st.session_state.pair = {
        "i": int(i), "j": int(j), "kind": kind,
        "p_personal": post.predict(x),
        "p_global": global_prob(st.session_state.w_global, x),
    }


def _on_click(chose_left):
    s = st.session_state
    pair = s.pair
    x = s.emb[pair["i"]] - s.emb[pair["j"]]
    y = 1 if chose_left else 0
    pick_p = pair["p_personal"] > 0.5
    pick_g = pair["p_global"] > 0.5
    if pair["kind"] == "measure":
        s.hits_personal.append(pick_p == bool(y))
        s.hits_global.append(pick_g == bool(y))
    s.last = {"kind": pair["kind"], "read_you": pick_p == bool(y),
              "p": pair["p_personal"] if chose_left else 1 - pair["p_personal"]}
    s.posterior.update(x, y)
    _log_swipe({
        "session_id": s.sid, "swipe_idx": s.n_swipes,
        "left_id": int(s.pet_ids[pair["i"]]), "right_id": int(s.pet_ids[pair["j"]]),
        "chose_left": y, "model_pick_left": int(pick_p),
        "p_left_global": round(pair["p_global"], 4),
        "p_left_personal": round(pair["p_personal"], 4),
        "pair_kind": pair["kind"], "model_version": s.model_version,
    })
    s.n_swipes += 1
    _new_pair(s.pet_ids, s.emb, s.posterior, s.rng)


def _init_session():
    s = st.session_state
    if "sid" in s:
        return
    w, meta, version = load_model()
    pet_ids, emb = load_pool()
    s.sid = uuid.uuid4().hex[:12]
    s.w_global = w
    s.model_version = version
    s.pet_ids, s.emb = pet_ids, emb
    s.posterior = UserPosterior(w, sigma0=meta.get("sigma0_default", 0.3))
    s.rng = np.random.default_rng()
    s.hits_personal, s.hits_global = [], []
    s.n_swipes = 0
    s.last = None
    _new_pair(pet_ids, emb, s.posterior, s.rng)


def _pct(hits, roll=None):
    if not hits:
        return None
    h = hits[-roll:] if roll else hits
    return 100.0 * sum(h) / len(h)


_init_session()
s = st.session_state

st.title("how predictable.")
st.caption("Click the pet you like more. The machine has already guessed which "
           "one you'll pick. Watch it learn you.")

pp = _pct(s.hits_personal, ROLL)
pg = _pct(s.hits_global, ROLL)
c1, c2, c3 = st.columns(3)
c1.metric("machine reads you", f"{pp:.0f}%" if pp is not None else "…",
          help="rolling accuracy of the personalized model on random measure "
               "pairs -- the actively-selected training pairs never count")
c2.metric("crowd model alone", f"{pg:.0f}%" if pg is not None else "…",
          help="the frozen global model on the same pairs; the gap to the left "
               "number is what the machine learned about YOU")
c3.metric("swipes", s.n_swipes)

if len(s.hits_personal) >= 2:
    import pandas as pd
    k = np.arange(1, len(s.hits_personal) + 1)
    chart = pd.DataFrame({
        "you, learned": 100 * np.cumsum(s.hits_personal) / k,
        "crowd model": 100 * np.cumsum(s.hits_global) / k,
    })
    st.line_chart(chart, height=180)

if s.last is not None:
    if s.last["read_you"]:
        st.success(f"how predictable. (it was "
                   f"{100*max(s.last['p'], 1-s.last['p']):.0f}% sure)")
    else:
        st.error("you surprised the machine. it is taking notes.")

pair = s.pair
left_id, right_id = s.pet_ids[pair["i"]], s.pet_ids[pair["j"]]
col_l, col_r = st.columns(2)
with col_l:
    st.image(str(POOL_DIR / f"{left_id}.jpg"), use_container_width=True)
    st.button("this one", key="pick_left", use_container_width=True,
              on_click=_on_click, args=(True,))
with col_r:
    st.image(str(POOL_DIR / f"{right_id}.jpg"), use_container_width=True)
    st.button("this one", key="pick_right", use_container_width=True,
              on_click=_on_click, args=(False,))

st.divider()
st.caption(f"session {s.sid} · pet_taste v{s.model_version} · photos: PetFinder "
           f"shelter listings (2023 archive) · supervised online preference "
           f"learning, not RL · every swipe becomes training data for the next "
           f"crowd model")
