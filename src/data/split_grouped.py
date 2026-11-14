"""Leakage-safe split: all near-identical images (augmented copies of one photo) go to the SAME split.

Groups are built from (1) Roboflow-style file names (<original>_jpg.rf.<hash>.jpg share <original>)
and (2) perceptual-hash similarity (Hamming distance <= GROUP_PHASH_THRESHOLD), merged with union-find.
Then a stratified group split gives about 71 / 14 / 14 (5 of 7 folds train, 1 val, 1 test).

Run INSTEAD of src.data.split:  python -m src.data.split_grouped
"""
import re, shutil
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from src import config as C
from src.data.clean import phash

GROUP_PHASH_THRESHOLD = getattr(C, "GROUP_PHASH_THRESHOLD", 10)
RF = re.compile(r"^(?:(?:train|valid|test)_)?(.*?)_(?:jpg|jpeg|png)\.rf\.", re.I)


def find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]; i = p[i]
    return i


def main():
    files = sorted(C.INTERIM.rglob("*.jpg"))
    labels = [f.parent.name for f in files]
    n = len(files)
    parent = list(range(n))

    def union(a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[rb] = ra

    # 1) group by original file name (works if the export kept Roboflow names)
    first = {}
    named = 0
    for i, f in enumerate(files):
        m = RF.match(f.name)
        if m:
            named += 1
            key = (f.parent.name, m.group(1))
            union(i, first.setdefault(key, i))
    print(f"files matching Roboflow naming: {named}/{n}")

    # 2) group by perceptual-hash similarity
    H = np.array([phash(Image.open(f).convert("RGB")) for f in tqdm(files, desc="hashing")])
    for i in tqdm(range(n), desc="grouping"):
        for j in np.where((H[i + 1:] != H[i]).sum(1) <= GROUP_PHASH_THRESHOLD)[0]:
            union(i, i + 1 + int(j))
    groups = np.array([find(parent, i) for i in range(n)])
    sizes = Counter(groups)
    big = sizes.most_common(1)[0][1]
    print(f"{len(sizes)} groups for {n} images, largest group = {big} images ({big / n:.1%})")
    if big > 0.05 * n:
        print("WARNING: one group holds more than 5% of the data. Lower GROUP_PHASH_THRESHOLD in this file.")

    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=C.SEED)
                 .split(np.zeros(n), labels, groups))
    idx = {"test": folds[0][1], "val": folds[1][1]}
    idx["train"] = np.setdiff1d(np.arange(n), np.concatenate([idx["test"], idx["val"]]))

    if C.PROCESSED.exists():
        shutil.rmtree(C.PROCESSED)
    for name, ids in idx.items():
        for i in ids:
            dst = C.PROCESSED / name / labels[i]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(files[i], dst / files[i].name)
        print(name, len(ids), dict(Counter(labels[i] for i in ids)))
    # sanity check: no group spans two splits
    where = {}
    for name, ids in idx.items():
        for i in ids:
            where.setdefault(groups[i], set()).add(name)
    print("groups spanning more than one split:", sum(len(v) > 1 for v in where.values()))


if __name__ == "__main__":
    main()
