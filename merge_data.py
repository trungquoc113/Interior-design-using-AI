import shutil
from pathlib import Path

src = Path(r"D:\trainphoto")        # thư mục vừa giải nén
dst = Path(r"D:\interior-style-ai\data\raw")

count = 0
for split in ("train", "valid", "test"):
    for cls_dir in (src / split).glob("*"):
        if cls_dir.is_dir():
            out = dst / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in cls_dir.iterdir():
                shutil.copy2(f, out / f"{split}_{f.name}")
                count += 1
print("Copied", count, "files")