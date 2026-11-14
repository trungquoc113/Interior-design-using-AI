"""Grad-CAM for several test images at once: the most confident WRONG predictions plus a few correct ones.
Needs a ResNet50 checkpoint and reports/errors_<tag>.csv (created by src.evaluate).
Run: python -m src.gradcam_batch --tag resnet50_finetuned
Output: reports/figures/gradcam_errors.png (title = true label | predicted label, confidence)
"""
import argparse, random
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src import config as C
from src.dataset import get_transforms
from src.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-wrong", type=int, default=6)
    ap.add_argument("--n-right", type=int, default=3)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.MODELS / f"{a.tag}.pt", map_location=device)
    if ck["model"] != "resnet50":
        raise SystemExit("gradcam_batch supports resnet50 only")
    classes = ck["classes"]
    model = build_model("resnet50", len(classes), dropout=ck["dropout"])
    model.load_state_dict(ck["state"]); model.to(device).eval()

    store = {}
    def fwd_hook(m, i, o):
        store["act"] = o
        o.register_hook(lambda g: store.update(grad=g))
    model.layer4[-1].register_forward_hook(fwd_hook)
    tf = get_transforms(False)

    err = pd.read_csv(C.ROOT / "reports" / f"errors_{a.tag}.csv")
    wrong = [(r.file, r.true) for r in err.head(a.n_wrong).itertuples()]
    wrong_files = set(err.file)
    pool = [(str(p), p.parent.name) for p in sorted((C.PROCESSED / "test").rglob("*.jpg")) if str(p) not in wrong_files]
    random.Random(C.SEED).shuffle(pool)
    items = wrong + pool[:a.n_right]

    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, (path, true) in zip(axs, items):
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(x)
        cls = out.argmax(1).item()
        conf = torch.softmax(out, 1)[0, cls].item()
        out[0, cls].backward()
        w = store["grad"].mean((2, 3), keepdim=True)
        cam = F.relu((w * store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear")[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=0.4)
        ok = classes[cls] == true
        ax.set_title(f"{'OK' if ok else 'WRONG'}  true: {true}\npred: {classes[cls]} ({conf:.0%})", fontsize=9,
                     color="green" if ok else "red")
    plt.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGS / "gradcam_errors.png", dpi=110)
    print("Saved reports/figures/gradcam_errors.png")


if __name__ == "__main__":
    main()
