from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os
CONFIG = {
    # Model and backbone
    "model": "resnet18",
    "pretrained": True,
    "freeze_backbone": True,                  # Start with frozen backbone
    "unfreeze_backbone_epoch": 10,           # Unfreeze backbone at this epoch for fine-tuning
    "dropout": 0.5,                           # Strong regularization to reduce overfitting

    # Training parameters
    "num_epochs": 30,                         # Enough epochs for convergence
    "batch_size": 64,                         # GPU-friendly batch size
    "image_size": 224,                        

    # Optimizer
    "optimizer": "adamw",                     
    "learning_rate": 5e-5,                    # LR for frozen backbone
    "unfreeze_learning_rate": 1e-4,           # LR after backbone unfreeze
    "weight_decay": 5e-4,                     # Regularization for weights

    # Loss function
    "loss_function": "BCEWithLogitsLoss",     # Standard for multi-label
    "class_weights": True,                     # Optional: handle class imbalance

    # Threshold for multi-label classification
    "threshold": 0.4,                         # Slightly lower to improve recall → better F1

    # Dataset paths
    "train_dir": "ms-coco/images/train-resized",
    "label_dir": "ms-coco/labels/train/train",
    "test_dir": "ms-coco/images/test-resized",
    "validation_split": 0.2,                  # 20% for validation

    # Resources
    "max_cpus": 14,                           

    # Scheduler
    "scheduler": "cosine"                     # Cosine annealing LR schedule for smooth convergence
}
