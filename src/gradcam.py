"""Fitur unik: Grad-CAM untuk melihat bagian gambar yang dipakai model saat memprediksi.

Dipakai untuk mengecek apakah model melihat gejala penyakit di daun, atau justru
latar belakang (risiko shortcut learning yang ditemukan saat EDA).

Contoh:
  python gradcam.py --checkpoint ../outputs/runs/tw_from_pv/best.pt --dataset taiwan
  python gradcam.py --checkpoint ../outputs/runs/pv_focal/best.pt --dataset taiwan --shared-only
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import DATASET_DIR, IMG_SIZE, OUTPUT_DIR, SEED, SHARED_LABELS
from data import EVAL_TF, load_split_df
from models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=["plantvillage", "taiwan"], required=True)
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--shared-only", action="store_true", help="hanya 3 kelas yang sama (untuk uji silang)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes, run_args = ckpt["classes"], ckpt["args"]
    model = build_model(run_args["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    cam = GradCAM(model=model, target_layers=[model.features[-1]])

    df = load_split_df(args.dataset)
    df = df[df.split == "test"]
    labels = SHARED_LABELS if args.shared_only else sorted(df.label.unique())
    samples = [r for lbl in labels
               for r in df[df.label == lbl].sample(args.per_class, random_state=SEED).itertuples()]

    n_cols = args.per_class * 2
    fig, axes = plt.subplots(len(labels), n_cols, figsize=(n_cols * 1.7, len(labels) * 2.0))
    axes = np.atleast_2d(axes)
    for k, row in enumerate(samples):
        img = Image.open(DATASET_DIR / row.path).convert("RGB")
        x = EVAL_TF(img)[None].to(device)
        with torch.no_grad():
            prob = model(x).softmax(1)[0]
        pred = int(prob.argmax())
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]  # CAM untuk kelas yang diprediksi
        rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255
        r, c = divmod(k, args.per_class)
        ax_img, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
        ax_img.imshow(rgb)
        ax_cam.imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        ok = classes[pred] == row.label
        ax_cam.set_title(f"{classes[pred]}\n{prob[pred]:.0%}", fontsize=7, color="green" if ok else "red")
        if c == 0:
            ax_img.set_ylabel(row.label, fontsize=8)
        for ax in (ax_img, ax_cam):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Grad-CAM {run_args['name']} pada test {args.dataset} "
                 f"(judul = prediksi; hijau benar, merah salah)", fontsize=10)
    fig.tight_layout()
    out_dir = OUTPUT_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_args['name']}_on_{args.dataset}.png"
    fig.savefig(out, dpi=130)
    print("Disimpan:", out)


if __name__ == "__main__":
    main()
