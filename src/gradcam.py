"""Grad-CAM cho RQ3: mo hinh nhin vao vung nao cua anh. Ho tro resnet50 (layer4[-1]).
Chay: python -m src.gradcam --tag resnet50_frozen --image path/to/room.jpg
"""
import argparse
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
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    ck = torch.load(C.MODELS / f"{a.tag}.pt", map_location="cpu")
    model = build_model(ck["model"], len(ck["classes"]), dropout=ck["dropout"])
    model.load_state_dict(ck["state"]); model.eval()
    layer = model.layer4[-1]
    acts, grads = {}, {}
    layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
    layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))

    img = Image.open(a.image).convert("RGB")
    x = get_transforms(False)(img).unsqueeze(0)
    out = model(x)
    cls = out.argmax(1).item()
    model.zero_grad()
    out[0, cls].backward()
    w = grads["v"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear")[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    plt.imshow(img); plt.imshow(cam, cmap="jet", alpha=0.4); plt.axis("off")
    plt.title(f"Pred: {ck['classes'][cls]}")
    C.FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGS / "gradcam_example.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
