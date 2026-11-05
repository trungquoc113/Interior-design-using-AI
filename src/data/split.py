"""Buoc 2 - Splitting: stratified 70/15/15, co dinh seed.
Chay: python -m src.data.split
"""
import shutil
from sklearn.model_selection import train_test_split
from src import config as C


def main():
    items = [(p, p.parent.name) for p in sorted(C.INTERIM.rglob("*.jpg"))]
    paths, labels = zip(*items)
    tr, rest, _, yrest = train_test_split(
        paths, labels, test_size=1 - C.SPLIT[0], stratify=labels, random_state=C.SEED)
    val_frac = C.SPLIT[1] / (C.SPLIT[1] + C.SPLIT[2])
    va, te, _, _ = train_test_split(
        rest, yrest, train_size=val_frac, stratify=yrest, random_state=C.SEED)
    if C.PROCESSED.exists():
        shutil.rmtree(C.PROCESSED)
    for name, group in (("train", tr), ("val", va), ("test", te)):
        for p in group:
            dst = C.PROCESSED / name / p.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        print(name, len(group))


if __name__ == "__main__":
    main()
