"""Melatih satu model pada gabungan PlantVillage + Taiwan (13 kelas).

Foto lapangan Taiwan jauh lebih sedikit (420 vs 10.164 data train), jadi data train
diambil seimbang per kombinasi (sumber, kelas). Epoch terbaik dipilih dari rata-rata
macro-F1 validasi PlantVillage dan Taiwan, dan evaluasi akhir dipisah per sumber agar
akurasi PlantVillage yang tinggi tidak menutupi performa pada foto lapangan.

Contoh:
  python train_gabungan.py --name gabungan_v2 --init ../outputs/runs/pv_focal_v2/best.pt \
      --extra-aug --epochs 10 --freeze-epochs 1 --lr 5e-4 --patience 4 \
      --samples-per-epoch 6000 --batch-size 48 --workers 8

Hasil disimpan di outputs/runs/<name>/. Di metrics.json:
  accuracy, macro_f1 = data uji Taiwan, memilih dari semua 13 kelas (seperti di aplikasi)
  taiwan.closed      = data uji Taiwan, memilih dari 6 kelas Taiwan saja
                       (sebanding dengan model tw_from_pv yang hanya mengenal 6 kelas)
"""
import argparse
import csv
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import DATA_DIR, DATASET_DIR, OUTPUT_DIR, SEED, TW_LABELS
from data import EVAL_TF, TRAIN_TF, LeafDataset
from evaluate import plot_confusion, plot_history, predict, save_json
from models import FocalLoss, build_model, load_backbone
from train import eval_loss, run_epoch, seed_everything, set_backbone_trainable

SOURCES = ("plantvillage", "taiwan")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gabungan_13")
    p.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_large"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--init", help="checkpoint untuk inisialisasi backbone")
    p.add_argument("--samples-per-epoch", type=int, default=0, help="0 = sebanyak data train")
    p.add_argument("--extra-aug", action="store_true", help="tambah varian augmentasi dari foto train Taiwan")
    p.add_argument("--smoke", action="store_true", help="uji cepat dengan sedikit data; hasil tidak disimpan di outputs")
    args = p.parse_args()
    args.dataset = "gabungan"
    return args


def load_combined():
    frames = [pd.read_csv(DATA_DIR / f"{s}.csv").assign(source=s) for s in SOURCES]
    df = pd.concat(frames, ignore_index=True)
    # Foto yang sama tidak boleh berada di dua split berbeda setelah digabung
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(df[df.split == a].md5) & set(df[df.split == b].md5)
        assert not shared, f"kebocoran data {a}/{b}: {len(shared)} foto"
    return df


def taiwan_augmented_rows(df):
    """Varian di folder 'data augmentation', hanya dari foto yang termasuk data train Taiwan.

    Folder itu mencampur varian foto validasi/test ke Train dan Test, jadi dicocokkan
    per foto asli agar daun di data validasi dan test tidak ikut dilatih."""
    tw_train = df[(df.source == "taiwan") & (df.split == "train")]
    keys = {(Path(p).parent.name, Path(p).stem) for p in tw_train.path}
    rows = []
    for f in sorted((DATASET_DIR / "taiwan" / "taiwan" / "data augmentation").rglob("*")):
        # File tanpa "_" adalah salinan foto asli yang sudah ada di data train
        if f.is_file() and "_" in f.stem and (f.parent.name, f.stem.split("_", 1)[0]) in keys:
            rows.append({"path": f.relative_to(DATASET_DIR).as_posix(), "label": TW_LABELS[f.parent.name],
                         "split": "train", "md5": "", "source": "taiwan"})
    return pd.DataFrame(rows)


def make_loader(part, classes, tf, batch_size, workers, sampler=None):
    ds = LeafDataset(part, classes, tf)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=True)


def balanced_sampler(train, num_samples=0):
    """Setiap kombinasi (sumber, kelas) mendapat porsi sampel yang sama."""
    group_size = train.groupby(["source", "label"]).label.transform("count")
    weights = torch.as_tensor((1.0 / group_size).to_numpy(copy=True), dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=num_samples or len(train), replacement=True)


def score(y_true, y_pred, ids, classes):
    names = [classes[i] for i in ids]
    report = classification_report(y_true, y_pred, labels=ids, target_names=names,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in names},
    }


def evaluate_source(model, loader, ids, classes, device):
    probs, y_true = predict(model, loader, device)
    open_pred = probs.argmax(1)
    closed_pred = np.asarray(ids)[probs[:, ids].argmax(1)]
    result = {"n": len(y_true),
              "open": score(y_true, open_pred, ids, classes),
              "closed": score(y_true, closed_pred, ids, classes),
              "predicted_outside_source_classes": float((~np.isin(open_pred, ids)).mean())}
    return result, y_true, open_pred


def main():
    args = parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(tempfile.mkdtemp()) if args.smoke else OUTPUT_DIR / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()
    classes = sorted(df.label.unique())
    source_ids = {s: sorted(classes.index(l) for l in df[df.source == s].label.unique()) for s in SOURCES}
    if args.extra_aug:
        extra = taiwan_augmented_rows(df)
        print(f"Varian augmentasi Taiwan yang aman: {len(extra)}")
        df = pd.concat([df, extra], ignore_index=True)
    if args.smoke:
        df = df.groupby(["source", "split"], group_keys=False).head(48)

    train, val = df[df.split == "train"], df[df.split == "val"]
    sampler = balanced_sampler(train, args.samples_per_epoch)
    train_loader = make_loader(train, classes, TRAIN_TF, args.batch_size, args.workers, sampler)
    val_loader = make_loader(val, classes, EVAL_TF, args.batch_size, args.workers)
    val_is_taiwan = (val.source == "taiwan").to_numpy()

    model = build_model(args.model, len(classes), args.dropout).to(device)
    if args.init:
        loaded, total = load_backbone(model, args.init)
        print(f"Backbone dimuat dari {args.init}: {loaded}/{total} tensor")
    print(f"Device: {device} | kelas: {len(classes)} | train: {len(train)} | val: {len(val)}")
    print("Train per sumber:", train.source.value_counts().to_dict())

    criterion = FocalLoss(gamma=2.0).to(device)  # penyeimbangan sudah dilakukan oleh sampler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_f1, bad_epochs = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, epoch > args.freeze_epochs)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, 0)
        va_loss, va_acc, va_pred, va_true = eval_loss(model, val_loader, criterion, device, 0)
        f1_tw = f1_score(va_true[val_is_taiwan], va_pred[val_is_taiwan], labels=source_ids["taiwan"],
                         average="macro", zero_division=0)
        f1_pv = f1_score(va_true[~val_is_taiwan], va_pred[~val_is_taiwan], labels=source_ids["plantvillage"],
                         average="macro", zero_division=0)
        va_f1 = (f1_tw + f1_pv) / 2
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_taiwan_macro_f1": f1_tw,
                        "val_plantvillage_macro_f1": f1_pv, "val_score": va_f1,
                        "lr": optimizer.param_groups[0]["lr"], "time_s": round(time.time() - start)})
        # Epoch saat backbone masih dibekukan tidak dipakai sebagai model akhir
        improved = epoch > args.freeze_epochs and va_f1 > best_f1
        print(f"epoch {epoch:2}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val acc {va_acc:.4f} | F1 PV {f1_pv:.4f} Taiwan {f1_tw:.4f} skor {va_f1:.4f}{' *' if improved else ''}")
        if epoch <= args.freeze_epochs:
            continue
        if improved:
            best_f1, bad_epochs = va_f1, 0
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args), "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {args.patience} epoch tanpa perbaikan skor validasi")
                break

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    plot_history(history, run_dir / "curves.png", f"{args.name}: loss & akurasi")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    test = df[df.split == "test"]
    result = {"run": args.name, "args": vars(args), "classes": classes, "best_val_score": best_f1,
              "epochs_trained": len(history), "train_time_s": round(time.time() - start)}
    for source in SOURCES:
        part = test[test.source == source]
        loader = make_loader(part, classes, EVAL_TF, args.batch_size, args.workers)
        result[source], y_true, y_pred = evaluate_source(model, loader, source_ids[source], classes, device)
        plot_confusion(y_true, y_pred, classes, run_dir / f"confusion_test_{source}.png",
                       f"{args.name} pada test {source}")
        for mode in ("open", "closed"):
            r = result[source][mode]
            print(f"TEST {source} ({mode}): accuracy {r['accuracy']:.4f} | macro-F1 {r['macro_f1']:.4f}")
    result["accuracy"] = result["taiwan"]["open"]["accuracy"]
    result["macro_f1"] = result["taiwan"]["open"]["macro_f1"]
    save_json(result, run_dir / "metrics.json")
    print("Disimpan:", run_dir)


if __name__ == "__main__":
    main()
