"""Training with optional two-phase transfer learning.

Examples
  Baseline from scratch:
    python -m src.train --model small_cnn --epochs 30 --tag small_cnn
  Transfer learning (phase 1: new head only, phase 2: fine-tune last block):
    python -m src.train --model resnet50 --freeze --epochs 8 --finetune-epochs 8 \
        --class-weights --label-smoothing 0.1 --tag resnet50_finetuned
"""
import argparse, json, random
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from src import config as C
from src.dataset import get_loaders, class_weights
from src.models import build_model, unfreeze_last_blocks


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def freeze_bn(model):
    """Keep BatchNorm layers of frozen blocks in eval mode so their statistics do not drift."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def run_epoch(model, loader, loss_fn, device, opt=None):
    train = opt is not None
    model.train(train)
    if train:
        freeze_bn(model)
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


def make_opt(model, lr, wd):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small_cnn", choices=["small_cnn", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20, help="epochs of phase 1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0, help="early stopping on val macro-F1 (0 = off)")
    ap.add_argument("--freeze", action="store_true", help="phase 1 trains only the new head")
    ap.add_argument("--finetune-epochs", type=int, default=0, help="phase 2 epochs (0 = no fine-tuning)")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=1)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.finetune_epochs and not a.freeze:
        print("Note: --finetune-epochs without --freeze: the backbone is already trainable in phase 1.")

    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, classes, targets = get_loaders(a.batch_size)
    model = build_model(a.model, len(classes), a.freeze, a.dropout).to(device)
    w = class_weights(targets, len(classes)).to(device) if a.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)

    tag = a.tag or a.model
    ckpt = C.MODELS / f"{tag}.pt"
    C.MODELS.mkdir(parents=True, exist_ok=True)
    phases = [("head" if a.freeze else "full", a.epochs, a.lr)]
    if a.finetune_epochs > 0:
        phases.append(("finetune", a.finetune_epochs, a.finetune_lr))

    best, hist, ep_total = -1.0, [], 0
    for phase, n_ep, lr in phases:
        if phase == "finetune":
            model.load_state_dict(torch.load(ckpt, map_location=device)["state"])  # start from best head
            unfreeze_last_blocks(model, a.model, a.unfreeze_blocks)
        opt = make_opt(model, lr, a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        bad = 0
        for ep in range(1, n_ep + 1):
            ep_total += 1
            tl, ta, _ = run_epoch(model, loaders["train"], loss_fn, device, opt)
            vl, va, vf = run_epoch(model, loaders["val"], loss_fn, device)
            sched.step()
            hist.append(dict(epoch=ep_total, phase=phase, train_loss=tl, train_acc=ta,
                             val_loss=vl, val_acc=va, val_f1=vf))
            print(f"[{phase}] ep{ep_total:02d} train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} f1 {vf:.3f}")
            if vf > best:
                best, bad = vf, 0
                torch.save({"model": a.model, "state": model.state_dict(), "classes": classes,
                            "dropout": a.dropout}, ckpt)
            else:
                bad += 1
            if a.patience and bad >= a.patience:
                print(f"Early stopping in phase '{phase}'."); break

    (C.MODELS / f"{tag}_history.json").write_text(json.dumps(hist, indent=2))
    (C.MODELS / f"{tag}_config.json").write_text(json.dumps({**vars(a), "best_val_macro_f1": best}, indent=2))
    print("best val macro-F1:", round(best, 4))


if __name__ == "__main__":
    main()
