"""Single-epoch training / evaluation loop."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
    grad_clip_norm: float | None = 1.0,
) -> tuple[float, float, list[float], list[float]]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze(1).float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required when train=True")
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs > 0.5).long()
            correct += (preds == labels.cpu().long()).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            all_probs.extend(probs.numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_probs, all_labels
