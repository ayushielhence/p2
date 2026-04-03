"""Construct models by name and training mode."""

from __future__ import annotations

import torch.nn as nn

from models.custom_cnn import CustomCNN
from models.pretrained import get_densenet121, get_efficientnet_b0, get_resnet18


def build_model(model_name: str, mode: str) -> nn.Module:
    """
    model_name: custom_cnn | resnet18 | densenet121 | efficientnet_b0
    mode: scratch (custom only) | frozen | finetune
    """
    name = model_name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes=1)

    freeze = mode.lower() == "frozen"
    if name == "resnet18":
        return get_resnet18(freeze_backbone=freeze)
    if name == "densenet121":
        return get_densenet121(freeze_backbone=freeze)
    if name == "efficientnet_b0":
        return get_efficientnet_b0(freeze_backbone=freeze)
    raise ValueError(f"Unknown model: {model_name}")
