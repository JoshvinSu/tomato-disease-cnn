"""Dataset dan transformasi citra."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DATA_DIR, DATASET_DIR, IMG_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentasi hanya untuk data train. Crop acak dan perubahan warna dipakai agar
# model tidak bergantung pada latar/pencahayaan tertentu (temuan EDA).
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class LeafDataset(Dataset):
    def __init__(self, df, classes, transform):
        self.paths = df.path.tolist()
        self.targets = [classes.index(lbl) for lbl in df.label]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(DATASET_DIR / self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]


def load_split_df(dataset):
    return pd.read_csv(DATA_DIR / f"{dataset}.csv")


def make_loaders(dataset, batch_size, num_workers=4):
    df = load_split_df(dataset)
    classes = sorted(df.label.unique())
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split]
        ds = LeafDataset(part, classes, TRAIN_TF if split == "train" else EVAL_TF)
        # persistent_workers sengaja tidak dipakai: di Windows worker yang tetap hidup
        # bisa macet (deadlock) saat evaluasi test setelah iterasi dihentikan di tengah.
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=split == "train",
                                    num_workers=num_workers, pin_memory=True)
    counts = torch.bincount(torch.tensor(loaders["train"].dataset.targets), minlength=len(classes))
    return loaders, classes, counts
