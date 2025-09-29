import os
from pathlib import Path
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class COCOTrainImageDataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_list = sorted(glob(os.path.join(img_dir, "*.jpg")))
        label_dict = {Path(f).stem: f for f in glob(os.path.join(labels_dir, "*.cls"))}
        self.img_list = [img for img in self.img_list if Path(img).stem in label_dict]
        self.label_list = [label_dict[Path(img).stem] for img in self.img_list]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        labels_path = self.label_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        with open(labels_path) as f:
            labels = [int(x) for x in f.read().splitlines()]
        labels_tensor = torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)
        return image, labels_tensor

class COCOTestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_list = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, Path(img_path).stem
