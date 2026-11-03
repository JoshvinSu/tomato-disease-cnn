"""Mengumpulkan metrik semua run ke satu tabel (outputs/ringkasan_hasil.csv) untuk laporan."""
import csv
import json

from config import OUTPUT_DIR

rows = []
for path in sorted((OUTPUT_DIR / "runs").glob("*/metrics.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    a = m["args"]
    rows.append({
        "skenario": m["run"], "dataset": a["dataset"], "loss": a["loss"],
        "class_weight": a["class_weight"], "init": "PlantVillage" if a["init"] else "ImageNet",
        "epoch": m["epochs_trained"], "val_macro_f1": round(m["best_val_macro_f1"], 4),
        "test_accuracy": round(m["accuracy"], 4), "test_macro_f1": round(m["macro_f1"], 4),
        "waktu_menit": round(m["train_time_s"] / 60, 1),
    })
for path in sorted((OUTPUT_DIR / "cross_eval").glob("*/metrics.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    name = path.parent.name
    for ds in ("plantvillage", "taiwan"):
        r = m[ds]
        rows.append({
            "skenario": f"{name} -> {ds} (3 kelas)", "dataset": ds, "loss": "", "class_weight": "",
            "init": "", "epoch": "", "val_macro_f1": "",
            "test_accuracy": round(r["shared"]["accuracy"], 4), "test_macro_f1": round(r["shared"]["macro_f1"], 4),
            "waktu_menit": "",
        })

out = OUTPUT_DIR / "ringkasan_hasil.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(" | ".join(str(v) for v in r.values()))
print("Disimpan:", out)
