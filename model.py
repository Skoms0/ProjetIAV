import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetMultiLabel(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Backbone EfficientNet-B0 pré-entraîné
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Geler le backbone (optionnel : peut commenter si fine-tuning complet)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Nombre de features en sortie avant le classifier
        in_features = self.backbone.classifier[1].in_features

        # Retirer le classifier original
        self.backbone.classifier = nn.Identity()

        # Nouvelle tête de classification multilabel
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)             # EfficientNet gère déjà pooling
        out = self.head(features)               # (batch, num_classes)
        return out