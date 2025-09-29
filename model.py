import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtMultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Backbone ConvNeXt Tiny pré-entraîné
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Geler le backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Extraire le nombre de features avant le classifier
        in_features = self.backbone.classifier[2].in_features

        # Retirer le classifier original
        self.backbone.classifier = nn.Identity()

        # Global average pooling + nouvelle tête
        self.pool = nn.AdaptiveAvgPool2d(1)  # réduit HxW à 1x1
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone.features(x)   # récupère uniquement la partie convolutionnelle
        features = self.pool(features)         # (batch, 768, 1, 1)
        features = torch.flatten(features, 1)  # (batch, 768)
        out = self.head(features)              # (batch, 80)
        return out
