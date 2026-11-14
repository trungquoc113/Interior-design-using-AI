"""Draw test images next to their closest train image, to judge leakage by eye.
Run after src.leakage_check:  python -m src.show_pairs
Output: reports/figures/leakage_pairs.png
Left of each pair = test image, right = closest train image, title = Hamming distance.
Rows: the 8 closest pairs, then 8 pairs sampled between distance 9 and 12.
"""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src import config as C


def main():
    df = pd.read_csv(C.ROOT / "reports" / "leakage_check.csv").sort_values("min_dist_to_train")
    close = df.head(8)
    mid = df[(df.min_dist_to_train >= 9) & (df.min_dist_to_train <= 12)]
    mid = mid.sample(min(8, len(mid)), random_state=C.SEED).sort_values("min_dist_to_train")
    rows = pd.concat([close, mid])
    n = len(rows)
    fig, axs = plt.subplots(n // 2 + n % 2, 4, figsize=(14, 3.2 * (n // 2 + n % 2)))
    axs = axs.reshape(-1, 4)
    for ax in axs.ravel():
        ax.axis("off")
    for i, (_, r) in enumerate(rows.iterrows()):
        row, col = divmod(i, 2)
        for k, (path, tag) in enumerate(((r.test_file, "test"), (r.nearest_train_file, "train"))):
            ax = axs[row, col * 2 + k]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(f"{tag}  d={int(r.min_dist_to_train)}", fontsize=9)
    plt.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGS / "leakage_pairs.png", dpi=110)
    print("Saved reports/figures/leakage_pairs.png")


if __name__ == "__main__":
    main()
