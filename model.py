# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetMultiLabel(nn.Module):
    def __init__(self, num_classes=80, freeze_backbone=True):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace original classifier with identity
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # New multi-label classifier head
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out
