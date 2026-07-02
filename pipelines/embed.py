"""Embed pipeline (F stage): petfinder zips + pawpularity -> embedding parquets.

Two modes, both writing data/emb/ and resumable (a re-run skips existing output):

  pool:  stream a HF petfinder-dogs zip, keep the LEAD photo (-1.jpg) per pet up
         to a per-job pet cap, embed it, AND save the 300px jpg to data/pool/
         (the 2023 cloudfront originals are dead from the pod; the game serves
         photos from HopsFS). Only lead photos + vectors leave the job.

    python pipelines/embed.py pool --zip 587 --cap 15000

  pawpularity: embed data/pawpularity/train/*.jpg with the pinned encoder.

    python pipelines/embed.py pawpularity

Output: data/emb/pool_<zip>.parquet (pet_id, emb, n_photos),
        data/emb/pawpularity.parquet (photo_id, emb, score),
        data/pool/<pet_id>.jpg
"""
import argparse
import glob
import io
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from PIL import Image

# Job copies run from Resources/jobs/<name>/; anchor on the repo.
def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/taste-machine")):
        if os.path.exists(os.path.join(p, "taste_features.py")):
            return p
    raise RuntimeError("repo root with taste_features.py not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
from taste_features import embed_images, ENCODER            # noqa: E402

DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
POOL = os.path.join(DATA, "pool")
HF = "https://huggingface.co/datasets/drzraf/petfinder-dogs/resolve/main"
BATCH = 64
LEAD = re.compile(r"(\d+)-1\.jpg$")


def _download(url, dest):
    for attempt in range(3):        # truncated streams happen; verify before use
        t0 = time.time()
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if zipfile.is_zipfile(dest):
            print(f"downloaded {os.path.getsize(dest)/1e9:.2f} GB in "
                  f"{time.time()-t0:.0f}s", flush=True)
            return
        print(f"corrupt download (attempt {attempt+1}), retrying", flush=True)
        os.remove(dest)
    raise RuntimeError("3 corrupt downloads, giving up")


def _flush(rows, imgs, pend):
    vecs = embed_images(imgs, BATCH)
    rows.extend([p + (v,) for p, v in zip(pend, list(vecs))])


def pool_mode(zip_name, cap):
    out_path = os.path.join(EMB, f"pool_{zip_name}.parquet")
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping", flush=True)
        return
    os.makedirs(EMB, exist_ok=True)
    os.makedirs(POOL, exist_ok=True)
    zpath = f"/tmp/{zip_name}.zip"
    if not zipfile.is_zipfile(zpath):
        _download(f"{HF}/{zip_name}.zip", zpath)

    rows, imgs, pend = [], [], []
    kept = bad = 0
    t0 = time.time()
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if LEAD.search(n)]
        # photos-per-pet from the full listing, before we drop non-lead files
        counts = pd.Series([os.path.basename(n).split("-")[0]
                            for n in z.namelist() if n.endswith(".jpg")]).value_counts()
        print(f"{zip_name}: {len(names)} pets with a lead photo, cap {cap}", flush=True)
        for name in names:
            if kept >= cap:
                break
            pet_id = int(LEAD.search(name).group(1))
            raw = z.read(name)
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                bad += 1
                continue
            with open(os.path.join(POOL, f"{pet_id}.jpg"), "wb") as f:
                f.write(raw)
            kept += 1
            imgs.append(im)
            pend.append((pet_id, int(counts.get(str(pet_id), 1))))
            if len(imgs) >= BATCH * 4:
                _flush(rows, imgs, pend)
                imgs, pend = [], []
                rate = len(rows) / (time.time() - t0)
                print(f"  {len(rows)} embedded ({rate:.1f}/s)", flush=True)
    if imgs:
        _flush(rows, imgs, pend)
    os.remove(zpath)
    df = pd.DataFrame(rows, columns=["pet_id", "n_photos", "emb"])
    df.to_parquet(out_path)
    print(f"{zip_name}: kept {len(df)} pets, {bad} bad, encoder={ENCODER}, "
          f"{time.time()-t0:.0f}s", flush=True)


def pawpularity_mode():
    out_path = os.path.join(EMB, "pawpularity.parquet")
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping", flush=True)
        return
    os.makedirs(EMB, exist_ok=True)
    meta = pd.read_csv(os.path.join(DATA, "pawpularity", "train.csv"))
    img_dir = os.path.join(DATA, "pawpularity", "train")
    rows, imgs, pend = [], [], []
    t0 = time.time()
    for _, m in meta.iterrows():
        p = os.path.join(img_dir, f"{m.Id}.jpg")
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        pend.append((m.Id, int(m.Pawpularity)))
        if len(imgs) >= BATCH * 4:
            _flush(rows, imgs, pend)
            imgs, pend = [], []
            print(f"  {len(rows)}/{len(meta)} embedded "
                  f"({len(rows)/(time.time()-t0):.1f}/s)", flush=True)
    if imgs:
        _flush(rows, imgs, pend)
    df = pd.DataFrame(rows, columns=["photo_id", "score", "emb"])
    df.to_parquet(out_path)
    print(f"pawpularity: {len(df)} embedded, encoder={ENCODER}, "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "pawpularity"])
    ap.add_argument("--zip", dest="zip_name")
    ap.add_argument("--cap", type=int, default=15000)
    a = ap.parse_args()
    if a.mode == "pool":
        if not a.zip_name:
            sys.exit("pool mode needs --zip")
        pool_mode(a.zip_name, a.cap)
    else:
        pawpularity_mode()


if __name__ == "__main__":
    main()
