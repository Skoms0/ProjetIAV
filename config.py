CONFIG = {
    "model": "convnext_tiny",
    "pretrained": True,
    "batch_size": 32,
    "image_size": 224,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "threshold": 0.5,
    "max_cpus": 4,
    "train_dir": "ms-coco/images/train",
    "label_dir": "ms-coco/images/label",
    "test_dir": "ms-coco/images/test"
}