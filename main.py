# ============================
# Imports
# ============================
import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
from datetime import datetime

# Imports persos
from dataset import COCOTrainImageDataset, COCOTestImageDataset
from loops import train_loop, validation_loop
from utils import update_graphs
from config import CONFIG

# Import modèles
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)

# ============================
# Préprocessing
# ============================
def get_preprocessing_transform(model_name, train=True, pretrained=True):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base_transform = weights.transforms() if weights is not None else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            base_transform
        ])
    else:
        return base_transform

# ============================
# Modèle
# ============================
def get_model(config, device):
    model_name = config["model"].lower()
    pretrained = config.get("pretrained", True)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 80)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 80)
    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    return model.to(device)

# ============================
# Fonction d'entraînement
# ============================
def run_training(config, device):
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter

    # --- Transforms ---
    train_transform = get_preprocessing_transform(config["model"], train=True)
    val_transform   = get_preprocessing_transform(config["model"], train=False)

    # --- Datasets ---
    full_train_dataset = COCOTrainImageDataset(
        img_dir=config["train_dir"],
        annotations_dir=config["label_dir"],
        transform=train_transform
    )

    test_dataset = COCOTestImageDataset(
        img_dir=config["test_dir"],
        transform=val_transform
    )

    # --- Split ---
    val_size = int(config["validation_split"] * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # --- Loaders ---
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["max_cpus"])
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["max_cpus"])
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["max_cpus"])

    print(f"DataLoaders ready: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # --- Model ---
    model = get_model(config, device)

    # --- Loss & Optimizer ---
    criterion = torch.nn.BCEWithLogitsLoss()

    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # --- TensorBoard ---
    run_name = f"{config['model']}_epochs{config['num_epochs']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard logs → runs/{run_name}")

    # --- Training ---
    best_f1 = 0.0
    best_model_path = f"best_{config['model']}_e{config['num_epochs']}.pth"

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)

        # Validation
        train_results = validation_loop(train_loader, model, criterion, num_classes=80, device=device, multi_task=True)
        val_results   = validation_loop(val_loader, model, criterion, num_classes=80, device=device, multi_task=True)

        # Update graphs
        update_graphs(writer, epoch, train_results, val_results)

        # Save best model
        if val_results["f1"] > best_f1:
            best_f1 = val_results["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1={best_f1:.4f}")

    writer.close()
    print(f"Training finished for {config['model']} with {config['num_epochs']} epochs.")

# ============================
# Main
# ============================
def main():
    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using CUDA for training.")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected. Training on CPU will be slow.")

    # --- Expériences ---
    for model_name in ["resnet18", "resnet50"]:
        for num_epochs in [10, 20]:
            print(f"\n=== Training {model_name} for {num_epochs} epochs ===")
            CONFIG["model"] = model_name
            CONFIG["num_epochs"] = num_epochs
            run_training(CONFIG, device)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
