"""
Aplikasi Streamlit untuk proyek klasifikasi penyakit daun tomat
(PlantVillage vs Taiwan).

Taruh file ini di folder utama project (sejajar dengan folder src/, data/, notebooks/),
lalu jalankan dari terminal:

    streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR, DATASET_DIR, OUTPUT_DIR, SEED, SHARED_LABELS  # noqa: E402

RUNS_DIR = OUTPUT_DIR / "runs"

st.set_page_config(page_title="Penyakit Daun Tomat", page_icon="🍅", layout="wide")


# ---------- Data ----------
@st.cache_data
def load_data():
    pv = pd.read_csv(DATA_DIR / "plantvillage.csv")
    tw = pd.read_csv(DATA_DIR / "taiwan.csv")
    return {"PlantVillage": pv, "Taiwan": tw}


def split_table(df):
    t = df.pivot_table(index="label", columns="split", values="path",
                       aggfunc="count", fill_value=0)
    t = t[["train", "val", "test"]]
    t["total"] = t.sum(axis=1)
    t["persen"] = (100 * t["total"] / t["total"].sum()).round(1)
    return t.sort_values("total", ascending=False)


@st.cache_data
def image_stats(name, n_per_class=60):
    df = load_data()[name]
    rows = []
    for lbl, g in df.groupby("label"):
        for p in g.sample(min(n_per_class, len(g)), random_state=SEED).path:
            arr = np.asarray(Image.open(DATASET_DIR / p).convert("RGB"), dtype=np.float32) / 255
            rows.append({"label": lbl, "R": arr[..., 0].mean(), "G": arr[..., 1].mean(),
                         "B": arr[..., 2].mean(), "brightness": arr.mean(), "contrast": arr.std()})
    return pd.DataFrame(rows)


@st.cache_resource
def load_model(ckpt_path, mtime):
    """Muat checkpoint best.pt hasil train.py (berisi state_dict, classes, args).

    mtime ikut menjadi kunci cache agar checkpoint yang diperbarui ikut dimuat ulang."""
    import torch
    from models import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = build_model(args["model"], len(ckpt["classes"]), args.get("dropout", 0.3), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["classes"], args


def preprocess(img):
    # Sama seperti saat training: resize 224x224 + normalisasi ImageNet
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


data = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🍅 Penyakit Daun Tomat")
halaman = st.sidebar.radio("Halaman", ["Ringkasan", "Distribusi kelas",
                                        "Contoh gambar", "Statistik citra", "Prediksi"])

# ---------- Halaman ----------
if halaman == "Ringkasan":
    st.title("Klasifikasi Penyakit Daun Tomat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gambar PlantVillage", f"{len(data['PlantVillage']):,}",
              f"{data['PlantVillage'].label.nunique()} kelas", delta_color="off")
    c2.metric("Gambar Taiwan", f"{len(data['Taiwan']):,}",
              f"{data['Taiwan'].label.nunique()} kelas", delta_color="off")
    c3.metric("Kelas yang sama", len(SHARED_LABELS))
    st.write("Kelas yang ada di kedua dataset:", ", ".join(f"`{l}`" for l in SHARED_LABELS))
    st.info("PlantVillage berisi foto lab (latar polos), sedangkan Taiwan foto lapangan "
            "(latar ramai). Perbedaan ini yang diuji lewat uji silang PlantVillage → Taiwan.")

elif halaman == "Distribusi kelas":
    st.title("Distribusi kelas")
    nama = st.selectbox("Dataset", list(data))
    t = split_table(data[nama])
    st.caption(f"Rasio kelas terbesar : terkecil = {t.total.max() / t.total.min():.1f} : 1")
    kiri, kanan = st.columns([3, 2])
    with kiri:
        counts = data[nama].label.value_counts()
        colors = ["#d95f02" if l in SHARED_LABELS else "#1b9e77" for l in counts.index]
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(counts) + 1))
        ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
        for i, v in enumerate(counts.values[::-1]):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
        ax.set_xlabel("Jumlah gambar")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.caption("Oranye = kelas yang ada di kedua dataset")
    with kanan:
        st.dataframe(t, width="stretch")

elif halaman == "Contoh gambar":
    st.title("Contoh gambar per kelas")
    nama = st.selectbox("Dataset", list(data))
    df = data[nama]
    label = st.selectbox("Kelas", sorted(df.label.unique()))
    n = st.slider("Jumlah gambar", 3, 10, 5)
    acak = st.button("Acak ulang")
    seed = np.random.randint(0, 10_000) if acak else SEED
    paths = df[df.label == label].sample(n, random_state=seed).path
    cols = st.columns(n)
    for col, p in zip(cols, paths):
        col.image(Image.open(DATASET_DIR / p).convert("RGB"), width="stretch")

elif halaman == "Statistik citra":
    st.title("Statistik citra: kecerahan & warna")
    st.write("Perbedaan besar di sini menandakan *domain shift* (foto lab vs foto lapangan).")
    pv_s, tw_s = image_stats("PlantVillage"), image_stats("Taiwan")
    cols = ["R", "G", "B", "brightness", "contrast"]
    st.dataframe(pd.DataFrame({"PlantVillage": pv_s[cols].mean(),
                               "Taiwan": tw_s[cols].mean()}).round(3))
    fitur = st.radio("Fitur", ["brightness", "contrast"], horizontal=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(pv_s[fitur], bins=30, alpha=0.6, density=True, label="PlantVillage")
    ax.hist(tw_s[fitur], bins=30, alpha=0.6, density=True, label="Taiwan")
    ax.set_title(f"Distribusi {fitur}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

elif halaman == "Prediksi":
    import json

    import torch

    st.title("Prediksi penyakit dari foto daun")

    runs = sorted(p.parent.name for p in RUNS_DIR.glob("*/best.pt"))
    if not runs:
        st.error(f"Belum ada model di `{RUNS_DIR}`. Latih dulu, contoh:\n\n"
                 "`python train.py --dataset plantvillage --name pv_ce`")
        st.stop()

    default = runs.index("gabungan_v2") if "gabungan_v2" in runs else 0
    run = st.selectbox("Pilih model (folder di outputs/runs)", runs, index=default)
    ckpt_file = RUNS_DIR / run / "best.pt"
    model, classes, args = load_model(str(ckpt_file), ckpt_file.stat().st_mtime)

    info = f"Arsitektur **{args['model']}**, dilatih pada data **{args['dataset']}**, {len(classes)} kelas"
    metrics_file = RUNS_DIR / run / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        if args["dataset"] == "gabungan":
            info += (f" · akurasi uji foto lapangan **{m['taiwan']['open']['accuracy']:.3f}**, "
                     f"foto lab **{m['plantvillage']['open']['accuracy']:.3f}**")
        else:
            info += f" · test accuracy **{m['accuracy']:.3f}**, macro-F1 **{m['macro_f1']:.3f}**"
    st.caption(info)
    if not metrics_file.exists():
        st.warning("Model ini masih dilatih atau pelatihannya belum selesai, jadi hasilnya belum bisa dipercaya.")

    file = st.file_uploader("Upload foto daun tomat", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        with torch.no_grad():
            prob = model(preprocess(img)).softmax(1)[0].numpy()
        hasil = pd.Series(prob, index=classes).sort_values(ascending=False)

        kiri, kanan = st.columns([1, 1.4])
        kiri.image(img, width="stretch")
        with kanan:
            top, p = hasil.index[0], hasil.iloc[0]
            if top == "healthy":
                st.success(f"### Sehat ({p:.1%})")
            else:
                st.error(f"### {top.replace('_', ' ').title()} ({p:.1%})")
            if p < 0.6:
                st.warning("Keyakinan model rendah. Coba foto lain yang lebih jelas.")
            fig, ax = plt.subplots(figsize=(6, 0.4 * len(hasil) + 0.8))
            ax.barh(hasil.index[::-1], hasil.values[::-1],
                    color=["#d95f02" if c == top else "#9e9e9e" for c in hasil.index[::-1]])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilitas")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        if args["dataset"] == "plantvillage":
            st.info("Model PlantVillage dilatih dengan foto lab berlatar polos. "
                    "Untuk foto lapangan, hasilnya bisa kurang akurat (domain shift).")
