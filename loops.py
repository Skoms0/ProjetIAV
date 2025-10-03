# ============================
# Imports
# ============================
import torch
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
from PIL import Image

# Custom modules
from dataset import COCOTrainImageDataset, COCOTestImageDataset
from loops import train_loop, validation_loop
from utils import update_graphs
from config import CONFIG

# Torchvision models
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)

# ============================
# Préprocessing
# ============================
def get_preprocessing_transform(model_name, train=True, pretrained=True):
    model_name = model_name.lower()

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
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
    dropout_p = config.get("dropout", 0.3)

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 80)
        )

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 80)
        )

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 80)
        )

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 80)
        )

    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    # Freeze backbone if requested
    freeze_backbone = config.get("freeze_backbone", False)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # activate only the head
        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model.to(device)


# ============================
# Main
# ============================
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    # Imports
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    # Transforms
    train_transform = get_preprocessing_transform(CONFIG["model"], train=True)
    val_transform   = get_preprocessing_transform(CONFIG["model"], train=False)

    # Datasets
    full_train_dataset = COCOTrainImageDataset(
        img_dir=CONFIG["train_dir"],
        annotations_dir=CONFIG["label_dir"],
        transform=train_transform
    )
    test_dataset = COCOTestImageDataset(
        img_dir=CONFIG["test_dir"],
        transform=val_transform
    )

    # Split
    val_size = int(CONFIG["validation_split"] * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["max_cpus"])
    val_loader   = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["max_cpus"])
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["max_cpus"])

    print(f"DataLoaders ready: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # Model
    model = get_model(CONFIG, device)

    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)  # Can add class_weights if needed

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # LR Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"],
        eta_min=1e-6
    )

    # TensorBoard
    run_name = f"{CONFIG['model']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard logs → runs/{run_name}")

    # Training loop
    best_f1 = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

        # Unfreeze backbone if staged unfreeze
        unfreeze_epoch = CONFIG.get("unfreeze_backbone_epoch", None)
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            print("Unfreezing backbone for fine-tuning.")
            for name, param in model.named_parameters():
                param.requires_grad = True
            # Update optimizer with new params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG.get("unfreeze_learning_rate", CONFIG["learning_rate"]),
                weight_decay=CONFIG["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG["num_epochs"] - unfreeze_epoch,
                eta_min=1e-6
            )

        # Train
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)

        # Validation
        train_results = validation_loop(train_loader, model, criterion, num_classes=80, device=device, multi_task=True, threshold=CONFIG["threshold"])
        val_results   = validation_loop(val_loader, model, criterion, num_classes=80, device=device, multi_task=True, threshold=CONFIG["threshold"])

        # Update graphs
        update_graphs(writer, epoch, train_results, val_results)

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_results["f1"] > best_f1:
            best_f1 = val_results["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1={best_f1:.4f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
