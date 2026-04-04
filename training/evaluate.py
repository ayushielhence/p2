"""Evaluate a trained checkpoint on the PneumoniaMNIST test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    get_dataloaders,
    pretrain_val_transform,
    scratch_val_transform,
)
from models.factory import build_model  # noqa: E402
from training.engine import run_epoch  # noqa: E402
from utils.helpers import get_torch_device, load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PneumoniaMNIST classifier")
    p.add_argument("--model", type=str, required=True, choices=["custom_cnn", "resnet18", "densenet121", "efficientnet_b0"])
    p.add_argument("--mode", type=str, required=True, choices=["scratch", "frozen", "finetune"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model state_dict (.pth)")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--out-dir", type=str, default=".", help="Where to save confusion_matrix.png and roc_curve.png")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving confusion matrix and ROC figures (metrics only)",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write metrics JSON to this path (relative paths resolved from project root)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["training"]["seed"]))

    if args.model == "custom_cnn" and args.mode != "scratch":
        raise SystemExit("custom_cnn requires --mode scratch")
    if args.model != "custom_cnn" and args.mode == "scratch":
        raise SystemExit("Pretrained models require --mode frozen or finetune")

    batch_size = int(cfg["training"]["batch_size"])
    download = bool(cfg["data"]["download"])
    num_workers = int(cfg["data"]["num_workers"])

    val_transform = scratch_val_transform if args.model == "custom_cnn" else pretrain_val_transform
    _, _, test_loader = get_dataloaders(
        val_transform,
        val_transform,
        batch_size=batch_size,
        download=download,
        num_workers=num_workers,
    )

    device = get_torch_device()
    print(f"Device: {device}")
    model = build_model(args.model, args.mode).to(device)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    criterion = nn.BCEWithLogitsLoss()
    _, _, probs, labels = run_epoch(
        model,
        test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        grad_clip_norm=None,
    )

    preds = [1 if p > 0.5 else 0 for p in probs]
    labels_int = [int(l) for l in labels]

    print(
        classification_report(
            labels_int,
            preds,
            target_names=["Normal", "Pneumonia"],
        )
    )
    acc = accuracy_score(labels_int, preds)
    prec = precision_score(labels_int, preds, zero_division=0)
    rec = recall_score(labels_int, preds, zero_division=0)
    f1 = f1_score(labels_int, preds, zero_division=0)
    auc = roc_auc_score(labels_int, probs)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    metrics = {
        "model": args.model,
        "mode": args.mode,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
    }
    if args.json_out:
        json_path = Path(args.json_out)
        if not json_path.is_absolute():
            json_path = ROOT / json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        cm = confusion_matrix(labels_int, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"],
        )
        plt.title("Confusion Matrix")
        plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

        fpr, tpr, _ = roc_curve(labels_int, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
