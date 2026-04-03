"""Train all five rubric experiments and evaluate; write metrics + update README."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS: list[tuple[str, str, str, str]] = [
    ("1", "custom_cnn", "scratch", "Custom CNN"),
    ("2", "resnet18", "frozen", "ResNet-18"),
    ("3", "resnet18", "finetune", "ResNet-18"),
    ("4", "densenet121", "finetune", "DenseNet-121"),
    ("5", "efficientnet_b0", "finetune", "EfficientNet-B0"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all five PneumoniaMNIST experiments")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional cap on training epochs per experiment (omit for config defaults)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated experiment numbers to run (1-5), e.g. 4,5 for the last two only",
    )
    args = parser.parse_args()
    extra_train: list[str] = []
    if args.epochs is not None:
        extra_train = ["--epochs", str(args.epochs)]

    experiments = list(EXPERIMENTS)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        experiments = [e for e in EXPERIMENTS if e[0] in want]
        if not experiments:
            raise SystemExit(f"No experiments matched --only {args.only!r}")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []

    for num, model, mode, label in experiments:
        run(
            [sys.executable, str(ROOT / "training" / "train.py"), "--model", model, "--mode", mode]
            + extra_train
        )
        ckpt = ROOT / "checkpoints" / f"{model}_{mode}_best.pth"
        out_sub = results_dir / f"{num}_{model}_{mode}"
        json_path = results_dir / f"{num}_{model}_{mode}_metrics.json"
        run(
            [
                sys.executable,
                str(ROOT / "training" / "evaluate.py"),
                "--model",
                model,
                "--mode",
                mode,
                "--checkpoint",
                str(ckpt.relative_to(ROOT)),
                "--out-dir",
                str(out_sub.relative_to(ROOT)),
                "--json-out",
                str(json_path.relative_to(ROOT)),
            ]
        )
        with open(json_path, encoding="utf-8") as f:
            all_metrics.append(json.load(f))

    # Merge on-disk metrics for full table when running a subset
    metrics_by_num: dict[str, dict] = {}
    for p in results_dir.glob("*_metrics.json"):
        if p.name == "all_metrics.json":
            continue
        num_key = p.name.split("_", 1)[0]
        if num_key.isdigit():
            with open(p, encoding="utf-8") as f:
                metrics_by_num[num_key] = json.load(f)
    for (num, _m, _mode, _label), m in zip(experiments, all_metrics, strict=True):
        metrics_by_num[num] = m

    missing = [n for n, _, _, _ in EXPERIMENTS if n not in metrics_by_num]
    if missing:
        print(f"Note: incomplete metrics on disk for experiment #s: {missing}")

    # Write combined summary (all files found)
    summary_path = results_dir / "all_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([metrics_by_num[n] for n, _, _, _ in EXPERIMENTS if n in metrics_by_num], f, indent=2)

    # Build markdown table rows (full 5 rows when metrics exist)
    lines = [
        "| # | Model | Mode | ROC-AUC | F1 | Accuracy | Notes |",
        "|---|-------|------|---------|-----|----------|--------|",
    ]
    for num, model, mode, label in EXPERIMENTS:
        if num not in metrics_by_num:
            lines.append(f"| {num} | {label} | {mode} | | | | |")
            continue
        m = metrics_by_num[num]
        lines.append(
            f"| {num} | {label} | {mode} | {m['roc_auc']:.4f} | {m['f1']:.4f} | {m['accuracy']:.4f} | See `results/{num}_{model}_{mode}/` |"
        )

    table_md = "\n".join(lines) + "\n"

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->"
    replacement = "<!-- RESULTS_TABLE_START -->\n" + table_md + "<!-- RESULTS_TABLE_END -->"
    if not re.search(pattern, text, re.DOTALL):
        raise SystemExit("README.md: missing RESULTS_TABLE markers")
    text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)

    # Environment snippet
    py_ver = (
        subprocess.check_output([sys.executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
        .decode()
        .strip()
    )
    try:
        import torch

        torch_ver = torch.__version__
        cuda = torch.cuda.is_available()
        cuda_str = torch.version.cuda if cuda else "N/A (CPU)"
        device_str = torch.cuda.get_device_name(0) if cuda else "CPU"
    except Exception:
        torch_ver = "?"
        cuda_str = "?"
        device_str = "?"

    env_block = (
        f"- **Python:** {py_ver}\n"
        f"- **PyTorch:** {torch_ver}\n"
        f"- **CUDA (if any):** {cuda_str}\n"
        f"- **Device:** {device_str}\n"
    )
    env_pattern = r"<!-- ENV_BLOCK_START -->.*?<!-- ENV_BLOCK_END -->"
    env_replacement = "<!-- ENV_BLOCK_START -->\n" + env_block + "<!-- ENV_BLOCK_END -->"
    if not re.search(env_pattern, text, re.DOTALL):
        raise SystemExit("README.md: missing ENV_BLOCK markers")
    text = re.sub(env_pattern, env_replacement, text, count=1, flags=re.DOTALL)

    readme.write_text(text, encoding="utf-8")
    print(f"Wrote {summary_path} and updated {readme}")


if __name__ == "__main__":
    main()
