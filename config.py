from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os

CONFIG = {
    "model": "convnext_tiny",
    "pretrained": True,
    "batch_size": 32,
    "image_size": 224,
    "num_epochs": 1,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "threshold": 0.5,
    "max_cpus": 6,
    "train_dir": "ms-coco/images/train-resized/train-resized",
    "label_dir": "ms-coco/labels/train/train",
    "test_dir": "ms-coco/images/test-resized/test-resized",
    "validation_split": 0.2,
    "threshold": 0.5,
}