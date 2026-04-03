# PneumoniaMNIST CNN

Binary classification of chest X-rays as **Normal** vs **Pneumonia** using the [PneumoniaMNIST](https://medmnist.com/) dataset (28×28 grayscale, MedMNIST).

## Dataset

- **PneumoniaMNIST** — see [MedMNIST](https://medmnist.com/) for documentation and citations.
- **Classes:** `0` = Normal, `1` = Pneumonia.
- **Splits:** train / validation / test are provided by the library (~5,856 images total across splits; class balance is handled by the benchmark — inspect counts in the notebook or via `len(dataset)` per split).

## Installation

```bash
git clone <your-repo-url>
cd p2
python -m pip install -r requirements.txt
```

Run all five experiments and refresh the results table in this README:

```bash
python scripts/run_all_experiments.py
```

Optional: limit epochs for a quick smoke test: `python scripts/run_all_experiments.py --epochs 3`.

Optional (Grad-CAM interpretability): `pip install grad-cam`

## Training

From the project root (this folder):

```bash
python training/train.py --model resnet18 --mode finetune
```

Examples:

| Experiment | Command |
|------------|---------|
| Custom CNN (scratch) | `python training/train.py --model custom_cnn --mode scratch` |
| ResNet-18 frozen | `python training/train.py --model resnet18 --mode frozen` |
| ResNet-18 fine-tune | `python training/train.py --model resnet18 --mode finetune` |
| DenseNet-121 fine-tune | `python training/train.py --model densenet121 --mode finetune` |
| EfficientNet-B0 fine-tune | `python training/train.py --model efficientnet_b0 --mode finetune` |

Hyperparameters live in `configs/config.yaml` (learning rates, epochs, early stopping, batch size).

## Evaluation

After training, the best weights are saved under `checkpoints/<model>_<mode>_best.pth` by default. Run:

```bash
python training/evaluate.py --model resnet18 --mode finetune --checkpoint checkpoints/resnet18_finetune_best.pth
```

For the custom CNN:

```bash
python training/evaluate.py --model custom_cnn --mode scratch --checkpoint checkpoints/custom_cnn_scratch_best.pth
```

This prints accuracy, precision, recall, F1, ROC-AUC, and a classification report, and writes `confusion_matrix.png` and `roc_curve.png` to the directory given by `--out-dir` (default: project root).

## Results

<!-- RESULTS_TABLE_START -->
| # | Model | Mode | ROC-AUC | F1 | Accuracy | Notes |
|---|-------|------|---------|-----|----------|--------|
| 1 | Custom CNN | scratch | | | | Run `python scripts/run_all_experiments.py` |
| 2 | ResNet-18 | frozen | | | | |
| 3 | ResNet-18 | finetune | | | | |
| 4 | DenseNet-121 | finetune | | | | |
| 5 | EfficientNet-B0 | finetune | | | | |
<!-- RESULTS_TABLE_END -->

## Environment

<!-- ENV_BLOCK_START -->
- **Python:** (run `python scripts/run_all_experiments.py` to fill)
- **PyTorch / CUDA:** 
- **GPU:** 
<!-- ENV_BLOCK_END -->

## Project layout

- `data/dataset.py` — transforms and `PneumoniaMNIST` dataloaders
- `models/` — `CustomCNN`, pretrained backbones, `factory.py` builder
- `training/train.py` — training with early stopping and checkpointing
- `training/evaluate.py` — test metrics and plots
- `configs/config.yaml` — centralized hyperparameters
- `notebooks/exploration.ipynb` — EDA and visualization

## Reproducibility

A fixed random seed (`training.seed` in `configs/config.yaml`, default `42`) is applied at the start of training and evaluation.
