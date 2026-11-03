"""Uji silang: model yang dilatih di PlantVillage diuji pada foto lapangan Taiwan.

Hanya 3 kelas yang ada di kedua dataset (bacterial_spot, healthy, late_blight).
Dua cara penilaian:
  - open    : model bebas memilih dari 10 kelas PlantVillage (kondisi nyata)
  - shared  : prediksi dibatasi ke 3 kelas yang sama (mengukur kemampuan membedakan)
Sebagai pembanding, model yang sama juga diuji pada data test PlantVillage untuk 3 kelas itu.

Contoh: python cross_eval.py --checkpoint ../outputs/runs/pv_focal/best.pt
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import OUTPUT_DIR, SHARED_LABELS
from data import EVAL_TF, LeafDataset, load_split_df
from evaluate import metrics, plot_confusion, predict, save_json
from models import build_model


def evaluate_on(df, model, train_classes, device):
    df = df[(df.split == "test") & df.label.isin(SHARED_LABELS)]
    loader = DataLoader(LeafDataset(df, train_classes, EVAL_TF), batch_size=64, num_workers=2)
    probs, y_true = predict(model, loader, device)
    shared_idx = [train_classes.index(c) for c in SHARED_LABELS]
    y_open = probs.argmax(1)
    y_shared = np.array(shared_idx)[probs[:, shared_idx].argmax(1)]
    # Ubah indeks 10 kelas -> indeks 3 kelas untuk metrik
    to_shared = {i: k for k, i in enumerate(shared_idx)}
    t = np.array([to_shared[i] for i in y_true])
    s = np.array([to_shared[i] for i in y_shared])
    o = np.array([to_shared.get(i, -1) for i in y_open])  # -1 = memilih kelas di luar 3 kelas
    return {
        "n": len(t),
        "open_accuracy": float((o == t).mean()),
        "open_outside_shared": float((o == -1).mean()),
        "shared": metrics(t, s, SHARED_LABELS),
    }, y_true, y_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes, run_args = ckpt["classes"], ckpt["args"]
    model = build_model(run_args["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    out_dir = OUTPUT_DIR / "cross_eval" / run_args["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": args.checkpoint, "classes": SHARED_LABELS}
    for name in ("plantvillage", "taiwan"):
        res, y_true, y_open = evaluate_on(load_split_df(name), model, classes, device)
        result[name] = res
        rows = [classes.index(c) for c in SHARED_LABELS]
        # Confusion matrix 3 kelas sebenarnya x 10 kelas prediksi (terlihat ke mana salahnya)
        remap = {c: i for i, c in enumerate(rows)}
        plot_confusion(np.array([remap[y] for y in y_true]), y_open, SHARED_LABELS,
                       out_dir / f"confusion_{name}.png",
                       f"{run_args['name']} -> {name} test (3 kelas)", pred_classes=classes)
        print(f"{name:12} n={res['n']:4} | akurasi (10 kelas) {res['open_accuracy']:.3f} "
              f"(prediksi di luar 3 kelas: {res['open_outside_shared']:.1%}) | "
              f"akurasi (3 kelas) {res['shared']['accuracy']:.3f} macro-F1 {res['shared']['macro_f1']:.3f}")
    save_json(result, out_dir / "metrics.json")


if __name__ == "__main__":
    main()
