"""Danh gia tren tap test: Accuracy, Macro-F1, classification report, confusion matrix.
Chay: python -m src.evaluate --tag small_cnn
"""
import argparse, json
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src import config as C
from src.dataset import get_loaders
from src.models import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(C.MODELS / f"{a.tag}.pt", map_location=device)
    classes = ck["classes"]
    model = build_model(ck["model"], len(classes), dropout=ck["dropout"])
    model.load_state_dict(ck["state"]); model.to(device).eval()
    loaders, _, _ = get_loaders()
    ys, ps = [], []
    for x, y in loaders["test"]:
        ps += model(x.to(device)).argmax(1).cpu().tolist(); ys += y.tolist()
    res = {"accuracy": accuracy_score(ys, ps), "macro_f1": f1_score(ys, ps, average="macro")}
    print(res)
    print(classification_report(ys, ps, target_names=classes))
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGS / f"cm_{a.tag}.png", dpi=150)
    (C.ROOT / "reports" / f"metrics_{a.tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
