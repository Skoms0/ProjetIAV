# import statements for python, torch and companion libraries and your own modules
import torch
import torch.nn as nn
from torchvision import transforms
import os
from glob import glob
from pathlib import Path
from PIL import Image

# import Jupyter variables 
from dataset import COCOTrainImageDataset, COCOTestImageDataset
from loops import train_loop, validation_loop
from utils import update_graphs

# global variables defining training hyper-parameters among other things, data directories initialization
from config import CONFIG

# Import model weights
from torchvision.models import (
    #mobilenet_v3_small, MobileNet_V3_Small_Weights,
    #efficientnet_b0, EfficientNet_B0_Weights,
    #resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights
)

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
        model = resnet18(weights=weights)
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, 80)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, 80)


    # Base transforms from pretrained weights (resize, crop, normalize, etc.)
    base_transform = weights.transforms() if weights is not None else transforms.Compose([
        transforms.Resize((224, 224)),  # fallback
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # simple augmentation
            base_transform
        ])
    else:
        return base_transform


def get_model(config, device):
    """
    Returns a model instance based on config["model"] and config["pretrained"],
    with the final classifier adapted for 80 multi-label outputs (MS COCO).
    """
    model_name = config["model"].lower()
    pretrained = config.get("pretrained", True)
    
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 80)

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 80)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 80)
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 80)
    
    else:
        raise ValueError(f"Unsupported model: {config['model']}")
    
    # Move model to device (CPU or GPU)
    return model.to(device)

def main():
    # device initialization

    ## Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using CUDA for training.")
    else:
        print("No GPU detected. Training on CPU will be significantly slower. Please be sure of the model you are using before continuing.")
        choice = input("Do you want to continue using CPU? (y/n): ").strip().lower()
        if choice == "y":
            device = torch.device("cpu")
            print("Continuing on CPU...")
        else:
            print("Exiting program. Please use a machine with GPU.")
            exit()  # stops the program

    print("Using device:", device)

    # instantiation of transforms, datasets and data loaders
    # TIP : use torch.utils.data.random_split to split the training set into train and validation subsets

    ## Transforms
    from torch.utils.data import DataLoader, random_split


    ### Training transform: include optional augmentation
    train_transform = get_preprocessing_transform(CONFIG["model"], train=True)

    ### Validation / Test transform: deterministic
    val_transform   = get_preprocessing_transform(CONFIG["model"], train=False)

    ## Datasets
    full_train_dataset = COCOTrainImageDataset(
        img_dir=CONFIG["train_dir"],
        annotations_dir=CONFIG["label_dir"],
        transform=train_transform
    )

    test_dataset = COCOTestImageDataset(
        img_dir=CONFIG["test_dir"],
        transform=val_transform
    )


    ## Train/Validation Split
    val_size = int(CONFIG["validation_split"] * len(full_train_dataset))  # 20% validation
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    ## DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["max_cpus"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["max_cpus"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["max_cpus"]
    )

    print("DataLoaders ready: train={}, val={}, test={}".format(
        len(train_loader), len(val_loader), len(test_loader)
    ))
    
    image, label = next(iter(train_loader))
    print(image.shape, label.shape)



    # instantiation and preparation of network model
    model = get_model(CONFIG, device)


    # instantiation of loss criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # instantiation of optimizer, registration of network parameters
    if CONFIG["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    elif CONFIG["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    elif CONFIG["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {CONFIG['optimizer']}")

    # Tensorboard
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    run_name = f"{CONFIG['model']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    print(f"TensorBoard logs → runs/{run_name}")


    # definition of current best model path
    best_f1 = 0.0
    best_model_path = "best_model.pth"

    # main training loop
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        ## Train
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        
        ## Validate
        train_results = validation_loop(train_loader, model, criterion, num_classes=80, device=device, multi_task=True)
        val_results = validation_loop(val_loader, model, criterion, num_classes=80, device=device, multi_task=True)
        
        ## Update graphs (optional)
        update_graphs(writer, epoch, train_results, val_results)
        
        ## Save model if better
        if val_results["f1"] > best_f1:
            best_f1 = val_results["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1={best_f1:.4f}")

    # close tensorboard SummaryWriter if created (optional)
    writer.close()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

