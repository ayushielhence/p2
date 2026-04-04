"""PneumoniaMNIST dataset loaders and transform pipelines."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PneumoniaMNIST


def _medmnist_root() -> str:
    root = Path(os.environ.get("MEDMNIST_ROOT", Path.home() / ".medmnist"))
    root.mkdir(parents=True, exist_ok=True)
    return str(root)

# --- Transform for custom CNN (28x28 grayscale) ---
scratch_train_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

scratch_val_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# --- Transform for pretrained models (224x224 RGB) ---
pretrain_train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

pretrain_val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_dataloaders(
    transform_train,
    transform_val,
    batch_size: int = 64,
    download: bool = True,
    num_workers: int = 4,
    root: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = root if root is not None else _medmnist_root()
    train_ds = PneumoniaMNIST(
        split="train", transform=transform_train, download=download, root=data_root
    )
    val_ds = PneumoniaMNIST(split="val", transform=transform_val, download=download, root=data_root)
    test_ds = PneumoniaMNIST(split="test", transform=transform_val, download=download, root=data_root)

    # Pinned memory only helps async H2D for CUDA; MPS/CPU emit a warning if True.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
