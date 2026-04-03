"""ImageNet-pretrained backbones with binary classification heads."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def get_resnet18(freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def get_densenet121(freeze_backbone: bool = False) -> nn.Module:
    model = models.densenet121(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model


def get_efficientnet_b0(freeze_backbone: bool = False) -> nn.Module:
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model
