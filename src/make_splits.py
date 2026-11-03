"""Membuat pembagian train/val/test yang bebas kebocoran data.

Hasil: data/plantvillage.csv dan data/taiwan.csv
Kolom: path, label, split, md5

- PlantVillage: duplikat (isi file identik) dibuang, lalu dibagi 70/15/15 stratified.
- Taiwan: hanya foto asli ("Preprocessed data"). Test bawaan dipertahankan,
  15% dari Train dipisah menjadi validasi. Folder "data augmentation" tidak
  dipakai karena versi augmentasi dari daun yang sama ada di Train dan Test.
"""
import csv
import hashlib
import random
from collections import Counter, defaultdict

from config import DATA_DIR, DATASET_DIR, PV_DIR, PV_LABELS, SEED, TW_DIR, TW_LABELS

IMG_EXT = {".jpg", ".jpeg", ".png"}


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def stratified_split(items, fractions, rng):
    """items: list of (path, label). fractions: dict split -> fraction (jumlah = 1)."""
    by_label = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)
    out = []
    names = list(fractions)
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        n = len(group)
        start = 0
        for i, name in enumerate(names):
            end = n if i == len(names) - 1 else start + round(n * fractions[name])
            out += [(p, lbl, name) for p, lbl in group[start:end]]
            start = end
    return out


def dedupe(items):
    """Buang file yang isinya identik dengan file lain (pertahankan yang pertama)."""
    seen, kept, dropped = {}, [], []
    for path, label in items:
        h = md5(path)
        if h in seen:
            dropped.append((path, seen[h]))
            continue
        seen[h] = path
        kept.append((path, label, h))
    return kept, dropped


def write_csv(name, rows):
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / name
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", "md5"])
        w.writerows(rows)
    counts = Counter((r[2], r[1]) for r in rows)
    print(f"\n{out.name}: {len(rows)} gambar")
    for split in ("train", "val", "test"):
        total = sum(v for (s, _), v in counts.items() if s == split)
        print(f"  {split:5} {total}")


def build_plantvillage(rng):
    items = [(p, PV_LABELS[d.name]) for d in sorted(PV_DIR.iterdir()) for p in list_images(d)]
    kept, dropped = dedupe(items)
    print(f"PlantVillage: {len(items)} gambar, {len(dropped)} duplikat dibuang")
    for dup, orig in dropped:
        print(f"  {dup.parent.name}/{dup.name} == {orig.name}")
    hashes = {p: h for p, _, h in kept}
    split = stratified_split([(p, lbl) for p, lbl, _ in kept],
                             {"train": 0.70, "val": 0.15, "test": 0.15}, rng)
    write_csv("plantvillage.csv", [(p.relative_to(DATASET_DIR).as_posix(), lbl, s, hashes[p]) for p, lbl, s in split])


def build_taiwan(rng):
    rows = []
    for src_split in ("Train", "Test"):
        items = [(p, TW_LABELS[d.name]) for d in sorted((TW_DIR / src_split).iterdir())
                 for p in list_images(d)]
        kept, dropped = dedupe(items)
        assert not dropped, dropped
        hashes = {p: h for p, _, h in kept}
        if src_split == "Train":
            split = stratified_split([(p, lbl) for p, lbl, _ in kept],
                                     {"train": 0.85, "val": 0.15}, rng)
        else:
            split = [(p, lbl, "test") for p, lbl, _ in kept]
        rows += [(p.relative_to(DATASET_DIR).as_posix(), lbl, s, hashes[p]) for p, lbl, s in split]
    write_csv("taiwan.csv", rows)


def check_no_leak(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["md5"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_split[a] & by_split[b]
        assert not overlap, f"{name}: {len(overlap)} gambar ada di {a} dan {b}"
    print(f"  cek kebocoran {name}: OK (tidak ada gambar yang sama antar split)")


if __name__ == "__main__":
    build_plantvillage(random.Random(SEED))
    build_taiwan(random.Random(SEED))
    check_no_leak("plantvillage.csv")
    check_no_leak("taiwan.csv")
