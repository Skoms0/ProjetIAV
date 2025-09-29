import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtMultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out
