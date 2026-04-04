"""Seeds, config loading, and shared utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def get_torch_device() -> torch.device:
    """Prefer CUDA, then Apple MPS (Metal), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)
