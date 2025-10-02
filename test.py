# import statements for python, torch and companion libraries and your own modules
# TIP: use the python standard json module to write python dictionaries as JSON files
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import json
import os
from pathlib import Path
import ssl, certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

## Your custom dataset class
from dataset import COCOTestImageDataset

## Your model helper function
from main import get_model  # assuming get_model is in main.py

# global variables defining inference hyper-parameters among other things
# DON'T forget the multi-task classification probability threshold
from config import CONFIG

# data, trained model and output directories/filenames initialization
model_path = "best_model.pth"
output_json = "test_predictions.json"

# device initialization
## Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU detected. Using CUDA for training.")
else:
    print("No GPU detected. Testing on CPU will be significantly slower. Please be sure of the model you are using before continuing.")
    choice = input("Do you want to continue using CPU? (y/n): ").strip().lower()
    if choice == "y":
        device = torch.device("cpu")
        print("Continuing on CPU...")
    else:
        print("Exiting program. Please use a machine with GPU.")
        exit()  # stops the program

print("Using device:", device)

# instantiation of transforms, dataset and data loader
## Import weights for all supported models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights
)
from torchvision import transforms

def get_test_transform(config):
    """
    Returns the deterministic test/validation transform according to
    config['model'] and config['pretrained'].
    """
    model_name = config["model"].lower()
    pretrained = config.get("pretrained", True)

    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
    elif model_name == "efficientnet_b0":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        

    else:
        raise ValueError(f"Unsupported model: {config['model']}")

    if weights is not None:
        return weights.transforms()
    else:
        # fallback: basic deterministic transforms
        return transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

test_transform = get_test_transform(CONFIG)

test_dataset = COCOTestImageDataset(
    img_dir=CONFIG["test_dir"],
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["max_cpus"]
)

# load network model from saved file
model = get_model(CONFIG, device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # important for inference


# initialize output dictionary
predictions = {}  # keys: image filenames or IDs, values: list of predicted class indices


# prediction loop over test_loader
model.eval()
with torch.no_grad():
    for images, image_ids in test_loader:  # adjust if your dataset returns IDs
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs >= CONFIG["threshold"]).cpu().numpy()
        
        for img_id, pred in zip(image_ids, preds):
            predictions[img_id] = pred.nonzero()[0].tolist()  # indices of classes predicted


# write JSON file
with open(output_json, "w") as f:
    json.dump(predictions, f, indent=4)