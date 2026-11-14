"""Compare all evaluated models: table + bar chart + training curves.
Run after evaluating each model:  python -m src.compare
Reads reports/metrics_*.json and outputs/models/*_history.json.
"""
import json
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C

COLS = ["accuracy", "macro_f1", "macro_auc", "top2_accuracy", "params_millions"]


def main():
    rows = []
    for f in sorted((C.ROOT / "reports").glob("metrics_*.json")):
        m = json.loads(f.read_text())
        rows.append({"model": f.stem.replace("metrics_", ""), **{k: m.get(k) for k in COLS}})
    if not rows:
        raise SystemExit("No metrics found. Run src.evaluate (and src.baseline_rf) first.")
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(C.ROOT / "reports" / "model_comparison.csv", index=False)
    print(df.round(4).to_string(index=False))

    C.FIGS.mkdir(parents=True, exist_ok=True)
    ax = df.set_index("model")[["accuracy", "macro_f1"]].plot.bar(figsize=(8, 5), ylim=(0, 1))
    ax.set_ylabel("Score"); ax.set_title("Model comparison (test set)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(C.FIGS / "model_comparison.png", dpi=150); plt.close()

    hist = [h for h in sorted(C.MODELS.glob("*_history.json")) if not h.stem.startswith("sweep_")]
    if hist:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for h in hist:
            d = pd.DataFrame(json.loads(h.read_text())); tag = h.stem.replace("_history", "")
            axs[0].plot(d["epoch"], d["val_loss"], label=tag)
            axs[1].plot(d["epoch"], d["val_f1"], label=tag)
        axs[0].set_title("Validation loss"); axs[1].set_title("Validation macro-F1")
        for a in axs:
            a.set_xlabel("Epoch"); a.legend()
        plt.tight_layout(); plt.savefig(C.FIGS / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
