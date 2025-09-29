# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
import json
import numpy as np

from dataset import COCOTrainImageDataset, COCOTestImageDataset
from model import EfficientNetMultiLabel
from utils import train_loop, validation_loop, predict_test, save_model_weights_json, visualize_predictions

# -----------------------------
# Load parameters
# -----------------------------
with open("parameters.json", "r") as f:
    params = json.load(f)

device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Classes
# -----------------------------
classes = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
)

# -----------------------------
# Transforms
# -----------------------------
if params.get("use_augmentation", True):
    transform_train = transforms.Compose([
        transforms.Resize(params["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
else:
    transform_train = transforms.Compose([
        transforms.Resize(params["image_size"]),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.Resize(params["image_size"]),
    transforms.ToTensor(),
])

# -----------------------------
# Datasets
# -----------------------------
full_dataset = COCOTrainImageDataset(params["train_image_dir"], params["train_label_dir"], transform_train)
total_size = len(full_dataset)
train_size = int(params["train_val_split"] * total_size)
val_size = total_size - train_size
generator = torch.Generator().manual_seed(params["seed"])
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

test_dataset = COCOTestImageDataset(params["test_image_dir"], transform_test)

# -----------------------------
# DataLoaders
# -----------------------------
max_workers = min(params["num_workers"], os.cpu_count())
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,
                          num_workers=max_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False,
                        num_workers=max_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False,
                         num_workers=max_workers, pin_memory=True)

print("Train dataset:", len(train_dataset))
print("Val dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))

# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = EfficientNetMultiLabel(num_classes=len(classes), freeze_backbone=params.get("freeze_backbone", True)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

# -----------------------------
# Training with automatic threshold tuning
# -----------------------------
best_micro_f1 = 0.0
best_threshold = params["threshold"]

for epoch in range(params["num_epochs"]):
    print(f"\nEpoch {epoch+1}/{params['num_epochs']}")
    train_loop(train_loader, model, criterion, optimizer, device)

    # Validate at multiple thresholds to find the best
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_epoch_f1 = 0.0
    best_epoch_threshold = 0.5
    for t in thresholds:
        val_loss, micro_f1, macro_f1, per_class_f1 = validation_loop(val_loader, model, criterion, device, threshold=t)
        print(f"Threshold {t:.2f} | Micro F1: {micro_f1:.4f} | Macro F1: {macro_f1:.4f}")
        if micro_f1 > best_epoch_f1:
            best_epoch_f1 = micro_f1
            best_epoch_threshold = t

    print(f"Best threshold this epoch: {best_epoch_threshold:.2f} | Best micro F1: {best_epoch_f1:.4f}")
    if best_epoch_f1 > best_micro_f1:
        best_micro_f1 = best_epoch_f1
        best_threshold = best_epoch_threshold
        # Optionally save best model
        torch.save(model.state_dict(), "best_model.pth")

print(f"\nTraining completed. Best threshold: {best_threshold:.2f} | Best micro F1: {best_micro_f1:.4f}")

# -----------------------------
# Save final model weights
# -----------------------------
save_model_weights_json(model)

# -----------------------------
# Test predictions
# -----------------------------
results = predict_test(test_loader, model, device, threshold=best_threshold)
with open("test_predictions.json", "w") as f:
    json.dump(results, f, indent=4)
print("Predictions saved in test_predictions.json")

# -----------------------------
# Visualize predictions
# -----------------------------
visualize_predictions(test_loader, model, device, classes, threshold=best_threshold,
                      n_images=params["n_images_visualize"])
