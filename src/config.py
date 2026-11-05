from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"            # data/raw/<style>/*.jpg
INTERIM = ROOT / "data" / "interim"    # sau khi lam sach
PROCESSED = ROOT / "data" / "processed"  # train/val/test
MODELS = ROOT / "outputs" / "models"
FIGS = ROOT / "reports" / "figures"

SEED = 42
IMG_SIZE = 224
MIN_SIDE = 128              # anh nho hon se bi loai
PHASH_THRESHOLD = 4         # khoang cach Hamming <= 4 xem la trung/gan trung
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
