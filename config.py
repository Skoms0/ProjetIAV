from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os
CONFIG = {
  "freeze_backbone": true,                     // Start with backbone frozen
  "unfreeze_backbone_epoch": 10,              // Unfreeze after 10 epochs for fine-tuning
  "model": "resnet18",
  "pretrained": true,
  "batch_size": 64,
  "image_size": 224,
  "num_epochs": 30,                           // Enough epochs for convergence
  "learning_rate": 5e-5,                      // LR for frozen backbone
  "unfreeze_learning_rate": 1e-4,             // Higher LR when backbone is unfrozen
  "optimizer": "adamw",
  "weight_decay": 5e-4,
  "threshold": 0.4,                           // Lower threshold for better recall
  "max_cpus": 14,
  "train_dir": "ms-coco/images/train-resized",
  "label_dir": "ms-coco/labels/train/train",
  "test_dir": "ms-coco/images/test-resized",
  "validation_split": 0.2,
  "dropout": 0.5,                             // Regularization to prevent overfitting
  "loss_function": "BCEWithLogitsLoss",       // Best for multi-label classification
  "scheduler": "cosine",                      // Cosine annealing LR schedule for smooth convergence
  "class_weights": true                       // Optional: handle MS-COCO label imbalance
}
