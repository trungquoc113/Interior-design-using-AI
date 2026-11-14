"""Check for near-duplicate leakage between train and test.
For every test image, find the closest train image by perceptual hash (pHash) and report
how many test images have a near-identical twin in train.
Run: python -m src.leakage_check
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src import config as C
from src.data.clean import phash


def hashes(split):
    files = sorted((C.PROCESSED / split).rglob("*.jpg"))
    H = np.array([phash(Image.open(f).convert("RGB")) for f in tqdm(files, desc=split)])
    return files, H


def main():
    ftr, Htr = hashes("train")
    fte, Hte = hashes("test")
    D = [(Htr != h).sum(1) for h in Hte]
    d = np.array([x.min() for x in D])          # distance to closest train image
    nearest = [str(ftr[x.argmin()]) for x in D]  # which train image that is
    print(f"Test images: {len(d)}")
    for t in (4, 8, 12, 16):
        print(f"closest train image within Hamming distance <= {t:2d}: {(d <= t).sum():4d} ({(d <= t).mean():.1%})")
    pd.DataFrame({"test_file": [str(f) for f in fte], "nearest_train_file": nearest, "min_dist_to_train": d}).sort_values(
        "min_dist_to_train").to_csv(C.ROOT / "reports" / "leakage_check.csv", index=False)
    print("Saved reports/leakage_check.csv (smallest distances first, open the top rows to inspect them)")


if __name__ == "__main__":
    main()
