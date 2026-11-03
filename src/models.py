"""Model CNN (transfer learning) dan fungsi loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_model(name, num_classes, dropout=0.3, pretrained=True):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        in_features = m.classifier[3].in_features
        m.classifier[2] = nn.Dropout(dropout)
        m.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"model tidak dikenal: {name}")
    return m


def head_params(model):
    """Parameter lapisan klasifikasi (yang dilatih saat backbone dibekukan)."""
    return list(model.classifier.parameters())


def load_backbone(model, checkpoint_path):
    """Muat bobot dari checkpoint lain kecuali lapisan klasifikasi terakhir
    (jumlah kelas bisa berbeda, misalnya PlantVillage 10 kelas -> Taiwan 6 kelas)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return len(keep), len(own)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): menurunkan bobot contoh yang sudah mudah
    diklasifikasi sehingga pelatihan lebih fokus pada contoh/kelas yang sulit."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, target, weight=self.weight, reduction="none")
        p_t = log_p.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - p_t) ** self.gamma * ce).mean()


def class_weights(counts):
    """Bobot kelas berbanding terbalik dengan jumlah data, dinormalisasi rata-rata 1."""
    w = counts.sum() / (len(counts) * counts.float())
    return w / w.mean()
