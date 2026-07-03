"""how predictable. -- the game, served properly.

Custom Hopsworks app (FastAPI, server-rendered, no front-end framework, no CDN).
This IS the v1 online inference pipeline: features from the feature store
(pet_embeddings, read once at boot), weights from the model registry (pet_taste
champion + TasteSpace), per-user Bayesian layer in server-side session state,
swipes appended to data/feedback/*.jsonl for the scheduled flywheel job.

The model's pick for the CURRENT pair never leaves the server -- the client
only sees image ids, so devtools cannot cheat the accuracy line.

Honesty rules, enforced here:
- accuracy counts ONLY randomly-chosen "measure" pairs (every 3rd); actively
  selected "train" pairs update the posterior but never score.
- the chart shows the frozen crowd model next to the personalized one.
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

def _find_root():
    import glob
    cand = Path(__file__).resolve().parents[1]
    for p in [cand] + [Path(g) for g in sorted(glob.glob("/hopsfs/Users/*/how-predictable"))]:
        if (p / "taste_features.py").exists():
            return p
    raise RuntimeError("repo root not found")

ROOT = _find_root()
sys.path.insert(0, str(ROOT))
from taste_online import (TasteSpace, UserPosterior, global_prob,   # noqa: E402
                          select_pair)

POOL_DIR = ROOT / "data" / "pool"
FEEDBACK_DIR = ROOT / "data" / "feedback"
BASE = os.environ.get("APP_BASE_URL_PATH", "").rstrip("/")
MEASURE_EVERY = 3
CAND_PAIRS = 40
SESSION_TTL = 7200

app = FastAPI()
STATE = {"space": None, "version": None, "pet_ids": None, "emb": None}
SESSIONS = {}


@app.on_event("startup")
def boot():
    import hopsworks
    proj = hopsworks.login()
    mr = proj.get_model_registry()
    champ = max(mr.get_models("pet_taste"), key=lambda m: m.version)
    d = champ.download()
    STATE["space"] = TasteSpace.load(os.path.join(d, "taste_space.npz"))
    STATE["version"] = champ.version
    fg = proj.get_feature_store().get_feature_group("pet_embeddings", 1)
    df = fg.read()
    have = {int(p.stem) for p in POOL_DIR.glob("*.jpg")}
    df = df[df["pet_id"].isin(have)].reset_index(drop=True)
    STATE["pet_ids"] = df["pet_id"].to_numpy()
    STATE["emb"] = np.stack(df["emb"].map(np.asarray).values).astype(np.float64)
    print(f"boot: pet_taste v{STATE['version']}, pool {len(df):,} pets", flush=True)


def _new_pair(s):
    emb, rng, post = STATE["emb"], s["rng"], s["posterior"]
    n = len(STATE["pet_ids"])
    kind = "measure" if s["n_swipes"] % MEASURE_EVERY == 0 else "train"
    if kind == "measure":
        i, j = rng.choice(n, 2, replace=False)
    else:
        cands = [tuple(rng.choice(n, 2, replace=False)) for _ in range(CAND_PAIRS)]
        feats = [emb[a] - emb[b] for a, b in cands]
        i, j = cands[select_pair(post, feats, rng=rng)]
    x = emb[i] - emb[j]
    s["pair"] = {"i": int(i), "j": int(j), "kind": kind,
                 "p_personal": post.predict(x),
                 "p_global": global_prob(STATE["space"], x)}


def _session(request, response):
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if now - v["seen"] > SESSION_TTL]:
        del SESSIONS[k]
    sid = request.cookies.get("sid")
    if sid and sid in SESSIONS:
        s = SESSIONS[sid]
        s["seen"] = now
        return s
    sid = uuid.uuid4().hex[:12]
    s = {"sid": sid, "seen": now, "rng": np.random.default_rng(),
         "posterior": UserPosterior(STATE["space"]), "n_swipes": 0,
         "hits_personal": [], "hits_global": [], "last": None}
    _new_pair(s)
    SESSIONS[sid] = s
    response.set_cookie("sid", sid, max_age=SESSION_TTL, httponly=True,
                        samesite="lax", path=BASE or "/")
    return s


def _log_swipe(s, pair, y, pick_p):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    row = {"session_id": s["sid"], "swipe_idx": s["n_swipes"],
           "left_id": int(STATE["pet_ids"][pair["i"]]),
           "right_id": int(STATE["pet_ids"][pair["j"]]),
           "chose_left": y, "model_pick_left": int(pick_p),
           "p_left_global": round(pair["p_global"], 4),
           "p_left_personal": round(pair["p_personal"], 4),
           "pair_kind": pair["kind"], "model_version": STATE["version"],
           "swiped_at": datetime.now(timezone.utc).isoformat()}
    with open(FEEDBACK_DIR / f"{s['sid']}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _payload(s):
    pair = s["pair"]
    k = np.arange(1, len(s["hits_personal"]) + 1)
    return {"left": int(STATE["pet_ids"][pair["i"]]),
            "right": int(STATE["pet_ids"][pair["j"]]),
            "n_swipes": s["n_swipes"],
            "curve_personal": list((100 * np.cumsum(s["hits_personal"]) / k).round(1)),
            "curve_global": list((100 * np.cumsum(s["hits_global"]) / k).round(1)),
            "last": s["last"], "model_version": STATE["version"]}


@app.get("/health")
def health():
    return {"ok": STATE["emb"] is not None}


@app.get("/img/{pet_id}.jpg")
def img(pet_id: int):
    p = POOL_DIR / f"{pet_id}.jpg"
    if not p.exists():
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/swipe")
async def swipe(request: Request):
    response = JSONResponse(None)
    s = _session(request, response)
    body = await request.json()
    if body.get("n") != s["n_swipes"]:            # stale double-click; resend state
        response.body = json.dumps(_payload(s)).encode()
        return response
    pair = s["pair"]
    x = STATE["emb"][pair["i"]] - STATE["emb"][pair["j"]]
    y = 1 if body.get("choice") == "left" else 0
    pick_p = pair["p_personal"] > 0.5
    pick_g = pair["p_global"] > 0.5
    if pair["kind"] == "measure":
        s["hits_personal"].append(bool(pick_p) == bool(y))
        s["hits_global"].append(bool(pick_g) == bool(y))
    conf = pair["p_personal"] if y else 1 - pair["p_personal"]
    s["last"] = {"read_you": bool(pick_p) == bool(y),
                 "confidence": round(100 * max(conf, 1 - conf))}
    s["posterior"].update(x, y)
    _log_swipe(s, pair, y, pick_p)
    s["n_swipes"] += 1
    _new_pair(s)
    response.body = json.dumps(_payload(s)).encode()
    return response


@app.get("/")
def index(request: Request):
    response = HTMLResponse(PAGE)
    s = _session(request, response)
    boot_data = json.dumps(_payload(s))
    response.body = PAGE.replace("__BASE__", BASE).replace(
        "__BOOT__", boot_data).encode()
    return response


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>how predictable.</title>
<style>
  :root{--bg:#0b0d10;--panel:#14171c;--line:#232830;--txt:#e8e6e1;--dim:#8a8f98;
        --you:#ff6b4a;--crowd:#5b6472;--ok:#2fbf71}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--txt);min-height:100vh;
       font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column;align-items:center;padding:2rem 1rem 4rem}
  header{width:100%;max-width:920px;margin-bottom:1.25rem}
  h1{font-size:clamp(1.8rem,4vw,2.6rem);letter-spacing:-0.03em;font-weight:800}
  h1 em{color:var(--you);font-style:normal}
  .tag{color:var(--dim);margin-top:.35rem;font-size:.95rem}
  .board{width:100%;max-width:920px;background:var(--panel);border:1px solid var(--line);
         border-radius:16px;padding:1.1rem 1.25rem;margin-bottom:1.25rem}
  .stats{display:flex;gap:2.5rem;align-items:baseline;flex-wrap:wrap}
  .stat b{font-size:2rem;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .stat.you b{color:var(--you)} .stat.crowd b{color:var(--crowd)}
  .stat span{display:block;color:var(--dim);font-size:.78rem;text-transform:uppercase;
             letter-spacing:.08em;margin-top:.15rem}
  svg{width:100%;height:120px;margin-top:.8rem;display:block}
  .verdict{height:1.6rem;margin:.2rem 0 .9rem;font-size:.98rem;color:var(--dim);
           transition:opacity .25s}
  .verdict.hit{color:var(--you);font-weight:700}
  .verdict.miss{color:var(--ok)}
  .arena{display:grid;grid-template-columns:1fr 1fr;gap:1rem;width:100%;max-width:920px}
  .card{position:relative;border-radius:16px;overflow:hidden;cursor:pointer;
        border:1px solid var(--line);background:var(--panel);aspect-ratio:1/1.05;
        transition:transform .15s ease,border-color .15s ease}
  .card:hover{transform:translateY(-4px);border-color:var(--you)}
  .card img{width:100%;height:100%;object-fit:cover;display:block}
  .card .pick{position:absolute;inset:auto 0 0 0;padding:.7rem;text-align:center;
        background:linear-gradient(transparent,rgba(0,0,0,.75));color:#fff;
        font-weight:700;letter-spacing:.05em;font-size:.9rem;text-transform:uppercase}
  .card.chosen{transform:scale(.97)}
  .hint{color:var(--dim);font-size:.8rem;margin-top:1rem}
  footer{color:var(--dim);font-size:.75rem;margin-top:2.5rem;max-width:920px;
         line-height:1.6;text-align:center}
  @media(max-width:560px){.arena{grid-template-columns:1fr}.card{aspect-ratio:4/3}}
</style></head><body>
<header>
  <h1>how <em>predictable.</em></h1>
  <div class="tag">Click the pet you like more. The machine has already guessed
  which one you&rsquo;ll pick &mdash; watch it learn you.</div>
</header>
<div class="board">
  <div class="stats">
    <div class="stat you"><b id="acc-you">&hellip;</b><span>machine reads you</span></div>
    <div class="stat crowd"><b id="acc-crowd">&hellip;</b><span>crowd model alone</span></div>
    <div class="stat"><b id="n">0</b><span>swipes</span></div>
  </div>
  <svg id="chart" viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true"></svg>
</div>
<div class="verdict" id="verdict">&nbsp;</div>
<div class="arena">
  <div class="card" id="card-left" onclick="pick('left')">
    <img id="img-left" alt="pet A"><div class="pick">this one</div></div>
  <div class="card" id="card-right" onclick="pick('right')">
    <img id="img-right" alt="pet B"><div class="pick">this one</div></div>
</div>
<div class="hint">or use &larr; &rarr; on a keyboard</div>
<footer>the top number counts only randomly-interleaved measure pairs; the
actively-selected training pairs never score. gap between the curves = what it
learned about <i>you</i>, not the crowd. supervised online preference learning,
not RL &middot; photos: PetFinder shelter listings (2023 archive) &middot;
model <span id="ver"></span> &middot; every swipe trains the next crowd model</footer>
<script>
const BASE="__BASE__";
let S=__BOOT__, busy=false;
function render(){
  document.getElementById("img-left").src=BASE+"/img/"+S.left+".jpg";
  document.getElementById("img-right").src=BASE+"/img/"+S.right+".jpg";
  document.getElementById("n").textContent=S.n_swipes;
  document.getElementById("ver").textContent="pet_taste v"+S.model_version;
  const cp=S.curve_personal, cg=S.curve_global;
  const yv=document.getElementById("acc-you"), cv=document.getElementById("acc-crowd");
  yv.innerHTML = cp.length? Math.round(cp[cp.length-1])+"%" : "&hellip;";
  cv.innerHTML = cg.length? Math.round(cg[cg.length-1])+"%" : "&hellip;";
  drawChart(cp,cg);
  const v=document.getElementById("verdict");
  if(S.last){ if(S.last.read_you){v.className="verdict hit";
      v.textContent="how predictable. ("+S.last.confidence+"% sure)";}
    else {v.className="verdict miss";
      v.textContent="you surprised the machine. it is taking notes.";} }
}
function drawChart(cp,cg){
  const svg=document.getElementById("chart");
  const n=Math.max(cp.length,2);
  const px=i=>i*(600/(n-1)), py=v=>110-(v/100)*100;
  const path=a=>a.map((v,i)=>(i?"L":"M")+px(i).toFixed(1)+" "+py(v).toFixed(1)).join(" ");
  let s='<line x1="0" y1="60" x2="600" y2="60" stroke="#232830" stroke-dasharray="4 4"/>' ;
  s+='<text x="4" y="56" fill="#8a8f98" font-size="9">50%</text>';
  if(cg.length>1) s+='<path d="'+path(cg)+'" fill="none" stroke="#5b6472" stroke-width="2"/>';
  if(cp.length>1) s+='<path d="'+path(cp)+'" fill="none" stroke="#ff6b4a" stroke-width="2.5"/>';
  if(cp.length) s+='<circle cx="'+px(cp.length-1)+'" cy="'+py(cp[cp.length-1])+'" r="3.5" fill="#ff6b4a"/>';
  svg.innerHTML=s;
}
async function pick(side){
  if(busy) return; busy=true;
  document.getElementById("card-"+side).classList.add("chosen");
  try{
    const r=await fetch(BASE+"/swipe",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({choice:side,n:S.n_swipes})});
    S=await r.json(); render();
  } finally {
    document.getElementById("card-left").classList.remove("chosen");
    document.getElementById("card-right").classList.remove("chosen");
    busy=false;
  }
}
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")pick("left");
  if(e.key==="ArrowRight")pick("right");});
render();
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)))
