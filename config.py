import os
"""
Configuration for COCO multi-label classification training.
Defines model type, preprocessing, optimizer, and training hyperparameters.
Specifies dataset directories and batch/data loader settings.
Includes validation split and threshold for multi-label predictions.
"""

CONFIG = {
    "model": "mobilenet_v3_small",  # Options: mobilenet_v3_small, efficientnet_b0, resnet50, resnet18
    "pretrained": True,              # Use pretrained weights
    "batch_size": 32,                # Batch size for training
    "image_size": 224,               # Input image size
    "num_epochs": 15,                # Number of training epochs
    "learning_rate": 1e-4,           # Optimizer learning rate
    "optimizer": "adamw",            # Options: adam, adamw, sgd
    "weight_decay": 1e-3,            # Weight decay for AdamW
    "max_cpus": 14,                  # Number of CPU threads for data loading
    "train_dir": "ms-coco/images/train-resized/train-resized",  # Training images
    "label_dir": "ms-coco/labels/train/train",                  # Training labels
    "test_dir": "ms-coco/images/test-resized/test-resized",     # Test images
    "validation_split": 0.2,         # Fraction of train set used for validation
    "threshold": 0.5,                # Default threshold for multi-label predictions
    "freeze_backbone": False,        # Freeze feature extractor/backbone for fine-tuning
}