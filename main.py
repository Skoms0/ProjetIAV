import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os

from dataset import COCOTrainImageDataset, COCOTestImageDataset
from model import EfficientNetMultiLabel  # Updated import
from utils import train_loop, validation_loop, predict_test, save_model_weights_json, visualize_predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    classes = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
               "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
               "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
               "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
               "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

    # -----------------------------
    # Transforms
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # -----------------------------
    # Datasets
    # -----------------------------
    full_dataset = COCOTrainImageDataset("ms-coco/images/train-resized", "ms-coco/labels/train", transform)

    # 80% train / 20% validation split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    test_dataset = COCOTestImageDataset("ms-coco/images/test", transform)

    # -----------------------------
    # DataLoaders
    # -----------------------------
    max_workers = min(8, os.cpu_count())
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=max_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=max_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=max_workers, pin_memory=True)

    print("Train dataset:", len(train_dataset))
    print("Val dataset:", len(val_dataset))
    print("Test dataset:", len(test_dataset))

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = EfficientNetMultiLabel(num_classes=len(classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use AdamW for better convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # -----------------------------
    # Training
    # -----------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loop(train_loader, model, criterion, optimizer, device)
        val_loss = validation_loop(val_loader, model, criterion, device)
        print(f"Validation loss: {val_loss:.4f}")

    # -----------------------------
    # Save model weights
    # -----------------------------
    save_model_weights_json(model)

    # -----------------------------
    # Test predictions
    # -----------------------------
    results = predict_test(test_loader, model, device)
    import json
    with open("test_predictions.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Predictions saved in test_predictions.json")

    # -----------------------------
    # Visualize predictions
    # -----------------------------
    visualize_predictions(test_loader, model, device, classes, threshold=0.5, n_images=5)
