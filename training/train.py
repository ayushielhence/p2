"""Train custom or pretrained binary classifiers on PneumoniaMNIST."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    get_dataloaders,
    pretrain_train_transform,
    pretrain_val_transform,
    scratch_train_transform,
    scratch_val_transform,
)
from models.factory import build_model  # noqa: E402
from training.engine import run_epoch  # noqa: E402
from utils.helpers import ensure_dir, get_torch_device, load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PneumoniaMNIST classifier")
    p.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "resnet18", "densenet121", "efficientnet_b0"],
        help="Model architecture",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="scratch",
        choices=["scratch", "frozen", "finetune"],
        help="scratch: custom CNN; frozen/finetune: pretrained backbones",
    )
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs (default: from config; frozen uses num_epochs_frozen)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Output checkpoint path (default: checkpoints/{model}_{mode}_best.pth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)

    if args.model == "custom_cnn" and args.mode != "scratch":
        raise SystemExit("custom_cnn must use --mode scratch")
    if args.model != "custom_cnn" and args.mode == "scratch":
        raise SystemExit("Pretrained models require --mode frozen or finetune")

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    batch_size = int(cfg["training"]["batch_size"])
    download = bool(cfg["data"]["download"])
    num_workers = int(cfg["data"]["num_workers"])
    patience = int(cfg["training"]["early_stopping_patience"])
    grad_clip = cfg["training"].get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    if args.model == "custom_cnn":
        train_t, val_t = scratch_train_transform, scratch_val_transform
        lr = float(cfg["optimizer"]["lr_scratch"])
        num_epochs = int(cfg["training"]["num_epochs"])
    else:
        train_t, val_t = pretrain_train_transform, pretrain_val_transform
        if args.mode == "frozen":
            lr = float(cfg["optimizer"]["lr_scratch"])
            num_epochs = int(cfg["training"]["num_epochs_frozen"])
        else:
            lr = float(cfg["optimizer"]["lr_finetune"])
            num_epochs = int(cfg["training"]["num_epochs"])

    if args.epochs is not None:
        num_epochs = args.epochs

    train_loader, val_loader, _ = get_dataloaders(
        train_t,
        val_t,
        batch_size=batch_size,
        download=download,
        num_workers=num_workers,
    )

    device = get_torch_device()
    print(f"Device: {device}")
    model = build_model(args.model, args.mode).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = ROOT / cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    ensure_dir(ckpt_dir)
    if args.checkpoint:
        best_path = Path(args.checkpoint)
        if not best_path.is_absolute():
            best_path = ROOT / best_path
    else:
        best_path = ckpt_dir / f"{args.model}_{args.mode}_best.pth"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            grad_clip_norm=grad_clip,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
            grad_clip_norm=None,
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
"""Train custom or pretrained binary classifiers on PneumoniaMNIST."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    get_dataloaders,
    pretrain_train_transform,
    pretrain_val_transform,
    scratch_train_transform,
    scratch_val_transform,
)
from models.factory import build_model  # noqa: E402
from training.engine import run_epoch  # noqa: E402
from utils.helpers import ensure_dir, get_torch_device, load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PneumoniaMNIST classifier")
    p.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "resnet18", "densenet121", "efficientnet_b0"],
        help="Model architecture",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="scratch",
        choices=["scratch", "frozen", "finetune"],
        help="scratch: custom CNN; frozen/finetune: pretrained backbones",
    )
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs (default: from config; frozen uses num_epochs_frozen)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Output checkpoint path (default: checkpoints/{model}_{mode}_best.pth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)

    if args.model == "custom_cnn" and args.mode != "scratch":
        raise SystemExit("custom_cnn must use --mode scratch")
    if args.model != "custom_cnn" and args.mode == "scratch":
        raise SystemExit("Pretrained models require --mode frozen or finetune")

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    batch_size = int(cfg["training"]["batch_size"])
    download = bool(cfg["data"]["download"])
    num_workers = int(cfg["data"]["num_workers"])
    patience = int(cfg["training"]["early_stopping_patience"])
    grad_clip = cfg["training"].get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    if args.model == "custom_cnn":
        train_t, val_t = scratch_train_transform, scratch_val_transform
        lr = float(cfg["optimizer"]["lr_scratch"])
        num_epochs = int(cfg["training"]["num_epochs"])
    else:
        train_t, val_t = pretrain_train_transform, pretrain_val_transform
        if args.mode == "frozen":
            lr = float(cfg["optimizer"]["lr_scratch"])
            num_epochs = int(cfg["training"]["num_epochs_frozen"])
        else:
            lr = float(cfg["optimizer"]["lr_finetune"])
            num_epochs = int(cfg["training"]["num_epochs"])

    if args.epochs is not None:
        num_epochs = args.epochs

    train_loader, val_loader, _ = get_dataloaders(
        train_t,
        val_t,
        batch_size=batch_size,
        download=download,
        num_workers=num_workers,
    )

    device = get_torch_device()
    print(f"Device: {device}")
    model = build_model(args.model, args.mode).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["scheduler"]["factor"]),
        patience=int(cfg["scheduler"]["patience"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = ROOT / cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    ensure_dir(ckpt_dir)
    if args.checkpoint:
        best_path = Path(args.checkpoint)
        if not best_path.is_absolute():
            best_path = ROOT / best_path
    else:
        best_path = ckpt_dir / f"{args.model}_{args.mode}_best.pth"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            grad_clip_norm=grad_clip,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
            grad_clip_norm=None,
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
