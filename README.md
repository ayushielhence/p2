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
| 1 | Custom CNN | scratch | 0.9651 | 0.9108 | 0.8798 | See `results/1_custom_cnn_scratch/` |
| 2 | ResNet-18 | frozen | 0.9074 | 0.8704 | 0.8205 | See `results/2_resnet18_frozen/` |
| 3 | ResNet-18 | finetune | 0.9818 | 0.9205 | 0.8926 | See `results/3_resnet18_finetune/` |
| 4 | DenseNet-121 | finetune | 0.9673 | 0.9063 | 0.8718 | See `results/4_densenet121_finetune/` |
| 5 | EfficientNet-B0 | finetune | 0.9724 | 0.9138 | 0.8830 | See `results/5_efficientnet_b0_finetune/` |
<!-- RESULTS_TABLE_END -->

## Environment

<!-- ENV_BLOCK_START -->
- **Python:** 3.12.10
- **PyTorch:** 2.11.0
- **CUDA (if any):** N/A (Apple Metal / MPS)
- **Device:** MPS (Apple GPU)
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
