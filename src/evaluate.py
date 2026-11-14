"""Evaluate a trained model on the held-out test set.

Outputs: Accuracy, Top-2 accuracy, Macro-F1, Macro AUC (one-vs-rest), per-class F1,
most confused style pairs, confusion matrix figure, and a list of misclassified images.
Run: python -m src.evaluate --tag small_cnn
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score, top_k_accuracy_score)
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
    n = len(classes)
    model = build_model(ck["model"], n, dropout=ck["dropout"])
    model.load_state_dict(ck["state"]); model.to(device).eval()
    loaders, _, _ = get_loaders()

    ys, probs = [], []
    for x, y in loaders["test"]:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); ys += y.tolist()
    P, ys = np.concatenate(probs), np.array(ys)
    ps = P.argmax(1)

    res = {"model": ck["model"],
           "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "accuracy": accuracy_score(ys, ps),
           "macro_f1": f1_score(ys, ps, average="macro")}
    if n > 2:
        res["top2_accuracy"] = top_k_accuracy_score(ys, P, k=2, labels=list(range(n)))
    try:
        res["macro_auc"] = roc_auc_score(ys, P, multi_class="ovr", average="macro", labels=list(range(n)))
    except ValueError:
        res["macro_auc"] = None  # a class is missing from the test set
    rep = classification_report(ys, ps, target_names=classes, output_dict=True, zero_division=0)
    res["per_class_f1"] = {c: round(rep[c]["f1-score"], 4) for c in classes}

    cm = confusion_matrix(ys, ps, labels=list(range(n)))
    off = [(cm[i, j], classes[i], classes[j]) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    res["top_confusions"] = [{"true": t, "predicted": p, "count": int(c)} for c, t, p in sorted(off, reverse=True)[:3]]

    print(json.dumps(res, indent=2))
    print(classification_report(ys, ps, target_names=classes, zero_division=0))

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix (row-normalised): {a.tag}")
    plt.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(C.FIGS / f"cm_{a.tag}.png", dpi=150)

    paths = [s[0] for s in loaders["test"].dataset.samples]
    bad = [{"file": paths[i], "true": classes[ys[i]], "predicted": classes[ps[i]],
            "confidence": round(float(P[i].max()), 3)} for i in np.where(ps != ys)[0]]
    pd.DataFrame(bad).sort_values("confidence", ascending=False).to_csv(
        C.ROOT / "reports" / f"errors_{a.tag}.csv", index=False)
    (C.ROOT / "reports" / f"metrics_{a.tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
