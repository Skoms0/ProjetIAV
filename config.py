from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os

CONFIG = {
    "model": "resnet18", # mobilenet_v3_small, efficientnet_b0, resnet50
    "pretrained": True,
    "batch_size": 64,
    "image_size": 224,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "optimizer": "adamw",  # adam, adamw, sgd
    "weight_decay": 1e-4,
    "threshold": 0.5,
    "max_cpus": 14,
    "train_dir": "ms-coco/images/train-resized",
    "label_dir": "ms-coco/labels/train/train",
    "test_dir": "ms-coco/images/test-resized",
    "validation_split": 0.2,
    "threshold": 0.5,
    "dropout":0.4,
}