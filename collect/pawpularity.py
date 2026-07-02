"""Download the Pawpularity dataset (the cold-start prior's labels).

Kaggle competition `petfinder-pawpularity-score`: 9,912 shelter-pet photos with
a real engagement score (0-100) derived from PetFinder.my page analytics.
Needs ~/.kaggle/kaggle.json AND the competition rules accepted on the website,
otherwise the download 403s.

    python collect/pawpularity.py

Output: data/pawpularity/{train.csv, train/*.jpg}
"""
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "pawpularity")
COMP = "petfinder-pawpularity-score"


def main():
    cred = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred):
        sys.exit(f"missing {cred} -- create an API token on kaggle.com and put it there")
    os.makedirs(DEST, exist_ok=True)
    csv = os.path.join(DEST, "train.csv")
    if os.path.exists(csv):
        print("already downloaded")
        return
    zpath = os.path.join("/tmp", f"{COMP}.zip")
    subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download",
                    "-c", COMP, "-p", "/tmp"], check=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DEST)
    os.remove(zpath)
    import pandas as pd
    df = pd.read_csv(csv)
    print(f"{len(df)} scored photos, score range "
          f"{df.Pawpularity.min()}-{df.Pawpularity.max()}")


if __name__ == "__main__":
    main()
