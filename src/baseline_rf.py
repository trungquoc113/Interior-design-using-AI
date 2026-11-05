"""Baseline phu: Random Forest tren histogram mau HSV (khong dung deep learning).
Chay: python -m src.baseline_rf
"""
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src import config as C


def feats(path, bins=8):
    hsv = np.asarray(Image.open(path).convert("RGB").resize((128, 128)).convert("HSV"))
    h = [np.histogram(hsv[..., i], bins=bins, range=(0, 256))[0] for i in range(3)]
    f = np.concatenate(h).astype(np.float32)
    return f / f.sum()


def load(split, classes):
    X, y = [], []
    for i, c in enumerate(classes):
        for p in sorted((C.PROCESSED / split / c).glob("*.jpg")):
            X.append(feats(p)); y.append(i)
    return np.array(X), np.array(y)


def main():
    classes = sorted(p.name for p in (C.PROCESSED / "train").iterdir() if p.is_dir())
    Xtr, ytr = load("train", classes)
    Xte, yte = load("test", classes)
    rf = RandomForestClassifier(n_estimators=300, random_state=C.SEED, n_jobs=-1).fit(Xtr, ytr)
    p = rf.predict(Xte)
    print({"accuracy": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro")})


if __name__ == "__main__":
    main()
