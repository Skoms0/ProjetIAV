from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os

CONFIG = {
    "model": "mobilenet_v3_small", # mobilenet_v3_small, efficientnet_b0, resnet50
    "pretrained": True,
    "batch_size": 64,
    "image_size": 256,
    "num_epochs": 20,
    "learning_rate": 3e-4,
    "optimizer": "adamw",  # adam, adamw, sgd
    "weight_decay": 1e-3,
    "threshold": 0.5,
    "max_cpus": 14,
    "train_dir": "ms-coco/images/train-resized/train-resized",
    "label_dir": "ms-coco/labels/train/train",
    "test_dir": "ms-coco/images/test-resized/test-resized",
    "validation_split": 0.2,
    "threshold": 0.5,
}