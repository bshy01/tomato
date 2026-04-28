# Tomato Leaf Disease Classification

This project is designed to classify tomato leaf diseases using various Deep Learning architectures. It provides a robust pipeline to compare the performance of state-of-the-art models with model-specific preprocessing.

## Key Features
- **Model Comparison**: Supports and compares **ResNet50**, **EfficientNet-B0**, and **Vision Transformer (ViT)**.
- **Automated Preprocessing**: Automatically applies model-specific input sizes and normalization (e.g., ImageNet specs for ResNet/EfficientNet, 0.5 mean/std for ViT).
- **Experiment Management**: Uses YAML configuration files to manage hyperparameters for different experiments.

## Project Structure
- `configs/`: Hyperparameter management (YAML files)
- `data/`: Datasets (organized as `train/` and `val/` subfolders) - *Ignored by Git*
- `src/`: Core source code
  - `datasets/`: Data loading and dynamic preprocessing
  - `models/`: Model factory and architecture definitions
  - `utils/`: Reproducibility (seeding) and common utilities
- `train.py`: Main execution script
- `requirements.txt`: Python dependencies

## How to Run
1. Prepare your data in the `data/` directory.
2. Run an experiment by specifying its config:
```bash
# Compare ResNet50
python train.py --config configs/experiment_v1.yaml

# Compare Vision Transformer (ViT)
python train.py --config configs/experiment_v2.yaml
```
