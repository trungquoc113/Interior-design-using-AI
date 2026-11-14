"""Small hyperparameter grid search. Selection uses the VALIDATION macro-F1 (never the test set).
Run: python -m src.sweep --model resnet50 --epochs 5
Results: reports/sweep_results.csv
"""
import argparse, itertools, json, subprocess, sys
import pandas as pd
from src import config as C

GRID = {"lr": [1e-3, 3e-4], "dropout": [0.3, 0.5], "batch_size": [16, 32]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["small_cnn", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    rows = []
    for lr, do, bs in itertools.product(GRID["lr"], GRID["dropout"], GRID["batch_size"]):
        tag = f"sweep_{a.model}_lr{lr}_do{do}_bs{bs}"
        cmd = [sys.executable, "-m", "src.train", "--model", a.model, "--epochs", str(a.epochs),
               "--lr", str(lr), "--dropout", str(do), "--batch-size", str(bs),
               "--class-weights", "--tag", tag]
        if a.model != "small_cnn":
            cmd.append("--freeze")
        subprocess.run(cmd, check=True)
        h = json.loads((C.MODELS / f"{tag}_history.json").read_text())
        rows.append({"lr": lr, "dropout": do, "batch_size": bs,
                     "best_val_macro_f1": max(e["val_f1"] for e in h),
                     "best_val_acc": max(e["val_acc"] for e in h)})
    df = pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False)
    df.to_csv(C.ROOT / "reports" / "sweep_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
