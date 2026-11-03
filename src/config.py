"""Path dan konstanta yang dipakai bersama oleh semua skrip."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR.parent / "Dataset of Tomato Leaves"

PV_DIR = DATASET_DIR / "plantvillage" / "Preprocessed data" / "Preprocessed data"
TW_DIR = DATASET_DIR / "taiwan" / "taiwan" / "Preprocessed data"

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SEED = 42
IMG_SIZE = 224  # ukuran input standar model pretrained ImageNet

# Nama folder asli -> nama kelas yang seragam.
# Kelas yang ada di kedua dataset diberi nama sama agar bisa diuji silang.
PV_LABELS = {
    "Bacterial_spot227": "bacterial_spot",
    "Early_blight227": "early_blight",
    "Late_blight227": "late_blight",
    "Leaf_Mold227": "leaf_mold",
    "Septoria_leaf_spot227": "septoria_leaf_spot",
    "Target_Spot227": "target_spot",
    "Tomato_Yellow_Leaf_Curl_Virus227": "yellow_leaf_curl_virus",
    "Tomato_mosaic_virus227": "mosaic_virus",
    "Two-spotted_spider_mite227": "spider_mite",
    "healthy227": "healthy",
}
TW_LABELS = {
    "Bacterial spot": "bacterial_spot",
    "Black mold": "black_mold",
    "Gray spot": "gray_spot",
    "Late blight": "late_blight",
    "health": "healthy",
    "powdery mildew": "powdery_mildew",
}
SHARED_LABELS = sorted(set(PV_LABELS.values()) & set(TW_LABELS.values()))
