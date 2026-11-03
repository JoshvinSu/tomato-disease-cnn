"""Fitur unik: deteksi input di luar domain (out-of-distribution) dengan confidence threshold.

Masalah yang diselesaikan: model PlantVillage salah dengan keyakinan tinggi saat diberi
foto lapangan (uji silang: akurasi ~36%). Fitur ini menolak prediksi ("tidak yakin")
jika skor keyakinan di bawah ambang batas, alih-alih memberi label yang salah.

Dua skor dibandingkan (makin tinggi = makin mirip data training):
  - MSP    : probabilitas softmax tertinggi (Hendrycks & Gimpel, 2017)
  - Energy : logsumexp(logits) (Liu et al., 2020)
Ambang batas dipilih dari data VALIDASI agar 95% gambar in-domain tetap diterima
(tidak pernah memakai data test untuk memilih ambang).

Eksperimen:
  1. Model PlantVillage (pv_ce): in-domain = test PlantVillage, out-of-domain = test Taiwan.
  2. Model Taiwan (tw_from_pv): prediksi selektif, kurva risk-coverage pada test Taiwan.

Hasil: outputs/ood/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import OUTPUT_DIR
from data import EVAL_TF, LeafDataset, load_split_df
from evaluate import predict_logits, save_json
from models import build_model

KEEP_IN_DOMAIN = 0.95  # 95% gambar validasi in-domain harus tetap diterima
OUT_DIR = OUTPUT_DIR / "ood"


def load_model(run, device):
    ckpt = torch.load(OUTPUT_DIR / "runs" / run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"]["model"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt["classes"]


def run_split(model, dataset, split, model_classes, device):
    """Logits + label (nama) untuk satu split. Label di luar kelas model tetap boleh."""
    df = load_split_df(dataset)
    df = df[df.split == split]
    ds = LeafDataset(df, sorted(df.label.unique()), EVAL_TF)
    logits, _ = predict_logits(model, DataLoader(ds, batch_size=64, num_workers=2), device)
    pred = [model_classes[i] for i in logits.argmax(1).tolist()]
    return logits, np.array(df.label.tolist()), np.array(pred)


def scores(logits):
    return {"msp": logits.softmax(1).max(1).values.numpy(),
            "energy": torch.logsumexp(logits, 1).numpy()}


def threshold_from_val(val_scores):
    return float(np.quantile(val_scores, 1 - KEEP_IN_DOMAIN))


def experiment_ood(device):
    model, classes = load_model("pv_ce", device)
    val_logits, _, _ = run_split(model, "plantvillage", "val", classes, device)
    id_logits, id_true, id_pred = run_split(model, "plantvillage", "test", classes, device)
    ood_logits, ood_true, ood_pred = run_split(model, "taiwan", "test", classes, device)

    result = {"model": "pv_ce", "keep_in_domain_val": KEEP_IN_DOMAIN,
              "n_in_domain_test": len(id_true), "n_out_of_domain_test": len(ood_true)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ("msp", "energy")):
        s_val, s_id, s_ood = scores(val_logits)[name], scores(id_logits)[name], scores(ood_logits)[name]
        thr = threshold_from_val(s_val)
        id_ok, ood_ok = s_id >= thr, s_ood >= thr
        # Foto Taiwan dari kelas yang juga dikenal model: berapa prediksi salah yang berhasil ditolak?
        known = np.isin(ood_true, classes)
        wrong = known & (ood_pred != ood_true)
        result[name] = {
            "threshold": thr,
            "auroc": float(roc_auc_score(np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))], np.r_[s_id, s_ood])),
            "in_domain_test_accepted": float(id_ok.mean()),
            "in_domain_test_acc_all": float((id_pred == id_true).mean()),
            "in_domain_test_acc_accepted": float((id_pred == id_true)[id_ok].mean()),
            "out_of_domain_rejected": float((~ood_ok).mean()),
            "taiwan_known_class_wrong_rejected": float((~ood_ok[wrong]).mean()) if wrong.any() else None,
            "taiwan_known_class_wrong_n": int(wrong.sum()),
        }
        bins = np.linspace(min(s_id.min(), s_ood.min()), max(s_id.max(), s_ood.max()), 40)
        ax.hist(s_id, bins=bins, alpha=0.6, density=True, label="PlantVillage test (in-domain)")
        ax.hist(s_ood, bins=bins, alpha=0.6, density=True, label="Taiwan test (out-of-domain)")
        ax.axvline(thr, color="black", linestyle="--", label=f"ambang batas = {thr:.3f}")
        if name == "msp":
            ax.set_yscale("log")  # sebagian besar skor menumpuk dekat 1.0
        ax.set_title(f"Skor {name.upper()} (AUROC {result[name]['auroc']:.3f})")
        ax.set_xlabel("skor (makin tinggi = makin yakin)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Deteksi out-of-domain: model PlantVillage menerima foto lab, menolak foto lapangan?")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ood_histogram.png", dpi=150)
    plt.close(fig)
    return result


def experiment_selective(device):
    model, classes = load_model("tw_from_pv", device)
    val_logits, _, _ = run_split(model, "taiwan", "val", classes, device)
    logits, y_true, y_pred = run_split(model, "taiwan", "test", classes, device)
    correct = y_pred == y_true

    result = {"model": "tw_from_pv", "n_test": len(y_true), "acc_all": float(correct.mean())}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ("msp", "energy"):
        s_val, s = scores(val_logits)[name], scores(logits)[name]
        thr = threshold_from_val(s_val)
        ok = s >= thr
        result[name] = {"threshold": thr, "coverage": float(ok.mean()),
                        "acc_accepted": float(correct[ok].mean()),
                        "wrong_rejected": float((~ok[~correct]).mean()),
                        "correct_rejected": float((~ok[correct]).mean())}
        order = np.argsort(-s)  # dari paling yakin
        cov = np.arange(1, len(s) + 1) / len(s)
        acc = np.cumsum(correct[order]) / np.arange(1, len(s) + 1)
        ax.plot(cov, acc, label=f"{name.upper()}")
        ax.scatter([ok.mean()], [correct[ok].mean()], zorder=3)
    ax.axhline(correct.mean(), color="gray", linestyle=":", label=f"tanpa penolakan ({correct.mean():.1%})")
    ax.set_xlabel("Coverage (proporsi gambar yang diberi prediksi)")
    ax.set_ylabel("Akurasi pada gambar yang diterima")
    ax.set_title("Prediksi selektif model Taiwan (titik = ambang dari validasi)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_coverage_taiwan.png", dpi=150)
    plt.close(fig)
    return result


def predict_with_rejection(model, x, classes, threshold, score="msp"):
    """Untuk inferensi/aplikasi: kembalikan (label, skor) atau ("tidak yakin", skor)."""
    with torch.no_grad():
        logits = model(x).float().cpu()
    s = float(scores(logits)[score][0])
    label = classes[int(logits.argmax(1))]
    return (label if s >= threshold else "tidak yakin / di luar domain"), s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"ood_plantvillage_model": experiment_ood(device),
              "selective_taiwan_model": experiment_selective(device)}
    save_json(result, OUT_DIR / "metrics.json")

    r = result["ood_plantvillage_model"]
    print("1) Model PlantVillage: foto lab (in-domain) vs foto lapangan Taiwan (out-of-domain)")
    for name in ("msp", "energy"):
        m = r[name]
        print(f"  {name:6} AUROC {m['auroc']:.3f} | ambang {m['threshold']:.3f} | "
              f"PV test diterima {m['in_domain_test_accepted']:.1%} (akurasi {m['in_domain_test_acc_all']:.2%} -> "
              f"{m['in_domain_test_acc_accepted']:.2%}) | Taiwan ditolak {m['out_of_domain_rejected']:.1%} | "
              f"prediksi salah 3 kelas ditolak {m['taiwan_known_class_wrong_rejected']:.1%} "
              f"(dari {m['taiwan_known_class_wrong_n']})")
    r = result["selective_taiwan_model"]
    print(f"2) Model Taiwan: akurasi tanpa penolakan {r['acc_all']:.1%} (n={r['n_test']})")
    for name in ("msp", "energy"):
        m = r[name]
        print(f"  {name:6} ambang {m['threshold']:.3f} | coverage {m['coverage']:.1%} | "
              f"akurasi diterima {m['acc_accepted']:.1%} | salah ditolak {m['wrong_rejected']:.1%} | "
              f"benar ikut ditolak {m['correct_rejected']:.1%}")


if __name__ == "__main__":
    main()
