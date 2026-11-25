"""Copy a small, balanced furniture set from the extracted ABO classroom subset into data/furniture/<label>/.

Run from the repository root after downloading and extracting abo_furniture.tar.gz into data/abo_raw:
    python scripts/prepare_furniture.py
    python scripts/prepare_furniture.py --labels bed chair table --per-class 300
Source layout expected: <src>/<split>/<label>/<image_id>.jpg  (default src = data/abo_raw/abo)
"""
import argparse, random, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/abo_raw/abo")
ap.add_argument("--dst", default="data/furniture")
ap.add_argument("--labels", nargs="*", default=["bed", "chair", "table"])
ap.add_argument("--per-class", type=int, default=300)
ap.add_argument("--split", default="train")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

src, dst = Path(a.src) / a.split, Path(a.dst)
if not src.exists():
    raise SystemExit(f"Not found: {src}. Run 'dir data\\abo_raw' and pass the right folder with --src.")
rng = random.Random(a.seed)
for label in a.labels:
    files = sorted((src / label).glob("*.jpg"))
    if not files:
        print(f"WARNING: no images for '{label}' in {src / label}")
        continue
    chosen = rng.sample(files, min(a.per_class, len(files)))
    (dst / label).mkdir(parents=True, exist_ok=True)
    for f in chosen:
        shutil.copy2(f, dst / label / f.name)
    print(f"{label}: copied {len(chosen)} of {len(files)} images")
print("Done. Folders in", dst)
