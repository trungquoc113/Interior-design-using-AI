"""Huan luyen. Vi du:
  python -m src.train --model small_cnn --epochs 30
  python -m src.train --model resnet50 --epochs 15 --freeze --class-weights --tag resnet50_frozen
"""
import argparse, json, random
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from src import config as C
from src.dataset import get_loaders, class_weights
from src.models import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, loss_fn, device, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss, preds, ys = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            preds += out.argmax(1).cpu().tolist(); ys += y.cpu().tolist()
    n = len(ys)
    acc = float(np.mean(np.array(preds) == np.array(ys)))
    return tot_loss / n, acc, f1_score(ys, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small_cnn", choices=["small_cnn", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, classes, targets = get_loaders(a.batch_size)
    model = build_model(a.model, len(classes), a.freeze, a.dropout).to(device)
    w = class_weights(targets, len(classes)).to(device) if a.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    tag = a.tag or a.model
    C.MODELS.mkdir(parents=True, exist_ok=True)
    best, hist = -1, []
    for ep in range(1, a.epochs + 1):
        tl, ta, _ = run_epoch(model, loaders["train"], loss_fn, device, opt)
        vl, va, vf = run_epoch(model, loaders["val"], loss_fn, device)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va, val_f1=vf))
        print(f"ep{ep:02d} train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} f1 {vf:.3f}")
        if vf > best:
            best = vf
            torch.save({"model": a.model, "state": model.state_dict(), "classes": classes,
                        "dropout": a.dropout}, C.MODELS / f"{tag}.pt")
    (C.MODELS / f"{tag}_history.json").write_text(json.dumps(hist, indent=2))
    print("best val macro-F1:", round(best, 4))


if __name__ == "__main__":
    main()
