"""Melatih CNN transfer learning untuk klasifikasi penyakit daun tomat.

Contoh:
  python train.py --dataset plantvillage --name pv_ce
  python train.py --dataset plantvillage --loss focal --class-weight --name pv_focal
  python train.py --dataset taiwan --init ../outputs/runs/pv_focal/best.pt --name tw_from_pv

Hasil disimpan di outputs/runs/<name>/:
  best.pt, history.csv, curves.png, confusion_test.png, metrics.json
"""
import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import OUTPUT_DIR, SEED
from data import make_loaders
from evaluate import metrics, plot_confusion, plot_history, predict, save_json
from models import FocalLoss, build_model, class_weights, head_params, load_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["plantvillage", "taiwan"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_large"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="epoch awal di mana hanya lapisan klasifikasi yang dilatih")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--patience", type=int, default=4, help="early stopping (epoch tanpa perbaikan val macro-F1)")
    p.add_argument("--init", help="checkpoint untuk inisialisasi backbone (fine-tuning)")
    p.add_argument("--max-batches", type=int, default=0, help="untuk uji cepat; 0 = semua")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, scaler, device, max_batches):
    model.train()
    total_loss = correct = n = 0
    for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="train")):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_loss(model, loader, criterion, device, max_batches):
    model.eval()
    total_loss = correct = n = 0
    preds, targets = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        total_loss += criterion(logits.float(), y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    return total_loss / n, correct / n, torch.cat(preds).numpy(), torch.cat(targets).numpy()


def set_backbone_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_params(model):
        p.requires_grad = True


def main():
    args = parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = OUTPUT_DIR / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, classes, counts = make_loaders(args.dataset, args.batch_size, args.workers)
    model = build_model(args.model, len(classes), args.dropout).to(device)
    if args.init:
        loaded, total = load_backbone(model, args.init)
        print(f"Backbone dimuat dari {args.init}: {loaded}/{total} tensor")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | model: {args.model} ({n_params / 1e6:.2f} M parameter) | kelas: {len(classes)}")
    print("Jumlah train per kelas:", dict(zip(classes, counts.tolist())))

    weight = class_weights(counts).to(device) if args.class_weight else None
    if weight is not None:
        print("Class weight:", {c: round(w, 2) for c, w in zip(classes, weight.tolist())})
    criterion = (FocalLoss(gamma=2.0, weight=weight) if args.loss == "focal"
                 else nn.CrossEntropyLoss(weight=weight)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_f1, bad_epochs = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, epoch > args.freeze_epochs)
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, args.max_batches)
        va_loss, va_acc, va_pred, va_true = eval_loss(model, loaders["val"], criterion, device, args.max_batches)
        va_f1 = metrics(va_true, va_pred, classes)["macro_f1"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "lr": optimizer.param_groups[0]["lr"], "time_s": round(time.time() - start)})
        improved = va_f1 > best_f1
        print(f"epoch {epoch:2}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}{' *' if improved else ''}")
        if improved:
            best_f1, bad_epochs = va_f1, 0
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args), "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {args.patience} epoch tanpa perbaikan val macro-F1")
                break

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    plot_history(history, run_dir / "curves.png", f"{args.name}: loss & akurasi")

    # Evaluasi akhir pada data test memakai checkpoint terbaik (dipilih dari val, bukan test)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    probs, y_true = predict(model, loaders["test"], device)
    y_pred = probs.argmax(1)
    result = metrics(y_true, y_pred, classes)
    result.update({"run": args.name, "args": vars(args), "best_val_macro_f1": best_f1,
                   "epochs_trained": len(history), "train_time_s": round(time.time() - start),
                   "n_params": n_params})
    save_json(result, run_dir / "metrics.json")
    plot_confusion(y_true, y_pred, classes, run_dir / "confusion_test.png",
                   f"{args.name} (test) acc={result['accuracy']:.3f} F1={result['macro_f1']:.3f}")
    print(f"\nTEST {args.name}: accuracy {result['accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f}")
    for c, m in result["per_class"].items():
        print(f"  {c:24} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1-score']:.3f} (n={int(m['support'])})")


if __name__ == "__main__":
    main()
