"""Buoc 1 - Cleaning: loai anh hong, anh qua nho, anh trung/gan trung (pHash).
Loai trung TRUOC khi chia tap de tranh ro ri du lieu giua train va test.

Chay: python -m src.data.clean
"""
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src import config as C

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def phash(img: Image.Image, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    n = hash_size * highfreq
    a = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float32)
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    low = (dct @ a @ dct.T)[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def main():
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    rows, kept_hashes = [], []
    files = [p for p in C.RAW.rglob("*") if p.suffix.lower() in EXTS]
    for p in tqdm(sorted(files), desc="cleaning"):
        label, status = p.parent.name, "kept"
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im = im.convert("RGB")
                if min(im.size) < C.MIN_SIDE:
                    status = "too_small"
                else:
                    h = phash(im)
                    if kept_hashes and (np.array(kept_hashes) != h).sum(1).min() <= C.PHASH_THRESHOLD:
                        status = "duplicate"
                    else:
                        kept_hashes.append(h)
                        out = C.INTERIM / label
                        out.mkdir(parents=True, exist_ok=True)
                        im.save(out / f"{p.stem}.jpg", quality=95)
        except Exception:
            status = "corrupt"
        rows.append({"file": str(p.relative_to(C.RAW)), "label": label, "status": status})
    df = pd.DataFrame(rows)
    df.to_csv(C.ROOT / "reports" / "cleaning_log.csv", index=False)
    print(df["status"].value_counts())
    print(df[df.status == "kept"]["label"].value_counts())


if __name__ == "__main__":
    main()
