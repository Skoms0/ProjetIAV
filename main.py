import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import COCOTrainImageDataset, COCOTestImageDataset
from model import ConvNeXtMultiLabel
from utils import train_loop, validation_loop, predict_test, save_model_weights_json, visualize_predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    train_dataset = COCOTrainImageDataset("ms-coco/images/train", "ms-coco/labels/train", transform)
    val_dataset   = COCOTrainImageDataset("ms-coco/images/val", "ms-coco/labels/val", transform)
    test_dataset  = COCOTestImageDataset("ms-coco/images/test", transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = ConvNeXtMultiLabel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    # TRAINING
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}")
        train_loop(train_loader, model, criterion, optimizer, device)
        val_loss = validation_loop(val_loader, model, criterion, device)
        print(f"Validation loss: {val_loss:.4f}")

    # SAVE MODEL WEIGHTS JSON
    save_model_weights_json(model)

    # TEST + SAVE PREDICTIONS JSON
    results = predict_test(test_loader, model, device)
    with open("test_predictions.json", "w") as f:
        import json
        json.dump(results, f, indent=4)
    print("Predictions saved in test_predictions.json")

    # VISUALISATION
    visualize_predictions(test_loader, model, device, classes, threshold=0.5, n_images=5)
