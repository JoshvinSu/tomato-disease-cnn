"""Fungsi evaluasi: prediksi, metrik, confusion matrix, kurva training."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        targets.append(y)
    return torch.cat(logits), torch.cat(targets).numpy()


def predict(model, loader, device):
    logits, targets = predict_logits(model, loader, device)
    return logits.softmax(1).numpy(), targets


def metrics(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(len(classes)), zero_division=0),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
    }


def plot_confusion(y_true, y_pred, classes, path, title, pred_classes=None):
    pred_classes = pred_classes or classes
    cm = confusion_matrix(y_true, y_pred, labels=range(max(len(classes), len(pred_classes))))
    cm = cm[:len(classes), :len(pred_classes)]
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(pred_classes), 1.0 + 0.6 * len(classes)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(pred_classes)), pred_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path, title):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], marker="o", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if key == "loss" else "Akurasi")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
