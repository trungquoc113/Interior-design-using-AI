"""SECONDARY check: does our ResNet50 recognise the requested style in the generated images?
Caution: the classifier was trained on retail-store photos, so rooms are outside its training domain.
Treat the result as a rough signal; human ratings are the main evidence.
Run: python -m src.design.classify_designs --tag resnet50_finetuned
"""
import argparse
import pandas as pd
import torch
from PIL import Image
from src import config as C
from src.dataset import get_transforms
from src.models import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.MODELS / f"{a.tag}.pt", map_location=device)
    model = build_model(ck["model"], len(ck["classes"]), dropout=ck["dropout"])
    model.load_state_dict(ck["state"]); model.to(device).eval()
    tf = get_transforms(False)
    df = pd.read_csv(C.ROOT / "outputs" / "designs" / "manifest.csv")
    preds = []
    for p in df["image"]:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        preds.append(ck["classes"][model(x).argmax(1).item()])
    df["predicted"] = preds
    df["match"] = df["predicted"] == df["style"]
    print(df.groupby("style")["match"].agg(["mean", "count"]).round(2))
    print("Overall agreement:", round(df["match"].mean(), 3))
    df[["image", "style", "predicted", "match"]].to_csv(C.ROOT / "reports" / "design_classifier_check.csv", index=False)


if __name__ == "__main__":
    main()
