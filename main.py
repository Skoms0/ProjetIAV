"""
Main training script for COCO multi-label classification.
Supports optional freezing of backbone layers.
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    resnet18, ResNet18_Weights,
)

# Local modules
from dataset import COCOTrainImageDataset, COCOTestImageDataset
from loops import train_loop, validation_loop
from utils import (
    update_graphs,
    collect_val_probs_and_labels,
    sweep_thresholds,
)
from config import CONFIG


# ==============================================================
# Data preprocessing
# ==============================================================
def get_preprocessing_transform(model_name: str, train: bool = True, pretrained: bool = True) -> transforms.Compose:
    """Return preprocessing transforms depending on model and phase."""
    model_name = model_name.lower()

    # Select normalization weights
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if weights is not None:
        norm_mean = weights.transforms().mean
        norm_std = weights.transforms().std
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    if train:
        # Training transform with augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(CONFIG["image_size"], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    # Validation/test transform (deterministic)
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


# ==============================================================
# Model selection and modification
# ==============================================================
def get_model(config: dict, device: torch.device, freeze_backbone: bool = False) -> torch.nn.Module:
    """
    Load base model, optionally freeze backbone, and modify classifier for 80 COCO classes.
    """
    model_name = config["model"].lower()
    pretrained = config.get("pretrained", True)

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 80),
        )
        if freeze_backbone:
            # Freeze all layers except classifier
            for param in model.features.parameters():
                param.requires_grad = False

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 80)
        if freeze_backbone:
            # Freeze all layers except final fully connected
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    return model.to(device)


# ==============================================================
# Training pipeline
# ==============================================================
def main() -> None:
    """Main training function."""
    # ---------------- Device setup ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using CUDA.")
    else:
        print("No GPU detected. Training on CPU.")
        choice = input("Continue on CPU? (y/n): ").strip().lower()
        if choice == "y":
            device = torch.device("cpu")
        else:
            print("Exiting. Please use a GPU machine.")
            return
    print(f"Using device: {device}")

    # ---------------- Datasets and loaders ----------------
    train_transform = get_preprocessing_transform(CONFIG["model"], train=True)
    val_transform = get_preprocessing_transform(CONFIG["model"], train=False)

    full_train_dataset = COCOTrainImageDataset(
        img_dir=CONFIG["train_dir"],
        annotations_dir=CONFIG["label_dir"],
        transform=train_transform,
    )
    test_dataset = COCOTestImageDataset(
        img_dir=CONFIG["test_dir"],
        transform=val_transform,
    )

    val_size = int(CONFIG["validation_split"] * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["max_cpus"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["max_cpus"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["max_cpus"])

    print(f"DataLoaders ready: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # Check sample batch
    image, label = next(iter(train_loader))
    print(f"Sample batch shapes - images: {image.shape}, labels: {label.shape}")

    # ---------------- Model, loss, optimizer ----------------
    freeze_backbone = CONFIG.get("freeze_backbone", False)
    model = get_model(CONFIG, device, freeze_backbone=freeze_backbone)
    criterion = nn.BCEWithLogitsLoss()

    optimizer_name = CONFIG["optimizer"].lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")

    # ---------------- TensorBoard logging ----------------
    run_name = f"{CONFIG['model']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard logs at: runs/{run_name}")

    # ---------------- Training loop ----------------
    best_f1 = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")

        _ = train_loop(train_loader, model, criterion, optimizer, device)
        train_results = validation_loop(train_loader, model, criterion, num_classes=80, device=device, multi_task=True)
        val_results = validation_loop(val_loader, model, criterion, num_classes=80, device=device, multi_task=True)

        update_graphs(writer, epoch, train_results, val_results)
        # Each time we save the layer with the best F1 because 
        # the challenge is about to have the best F1
        if val_results["f1"] > best_f1:
            best_f1 = val_results["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1 = {best_f1:.4f}")

    # ---------------- Threshold sweep ----------------
    val_probs, val_labels = collect_val_probs_and_labels(val_loader, model, device)
    best_t, best_f1_sweep, best_prec, best_rec = sweep_thresholds(val_probs, val_labels)
    print(f"Best threshold = {best_t:.2f} | F1 = {best_f1_sweep:.4f} | Precision = {best_prec:.4f} | Recall = {best_rec:.4f}")

    writer.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
