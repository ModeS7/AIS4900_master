# MedGen - Synthetic Medical Image Generation

A modular framework for generating synthetic brain MRI images using diffusion models.

## Quick Start

```bash
# Install
pip install -e .

# Train (default: DDPM, 128px, bravo mode)
python -m medgen.scripts.train

# Override via CLI
python -m medgen.scripts.train strategy=rflow model.image_size=256 training.epochs=100

# Cluster training
python -m medgen.scripts.train paths=cluster
```

## Features

- **Two diffusion strategies**: DDPM and Rectified Flow
- **Three training modes**: Segmentation masks, single-image, dual-image generation
- **Hydra configuration**: Composable YAML configs with CLI overrides
- **Training optimizations**: EMA, Min-SNR weighting, mixed precision (bfloat16)
- **Multi-GPU support**: Distributed training with DDP

## Project Structure

```
AIS4900_master/
├── pyproject.toml           # Package definition (pip install -e .)
├── configs/                  # Hydra configuration files
│   ├── config.yaml           # Main config (composes sub-configs)
│   ├── model/default.yaml    # Model architecture (override image_size via CLI)
│   ├── strategy/             # ddpm.yaml, rflow.yaml
│   ├── mode/                 # seg.yaml, bravo.yaml, dual.yaml
│   ├── training/             # default.yaml
│   └── paths/                # local.yaml, cluster.yaml
├── src/medgen/               # Main package
│   ├── core/                 # Constants, path utilities
│   ├── data/                 # NIfTI dataset, transforms
│   ├── diffusion/            # Trainer, strategies, modes
│   └── scripts/              # train.py, generate.py
├── IDUN/                     # SLURM template
├── DETAILES.md               # Detailed technical documentation
└── FUTURE_WORK.md            # Planned improvements
```

## Architecture

**Strategy + Mode** design pattern:

- **Strategy** (HOW to diffuse): `ddpm` or `rflow`
- **Mode** (WHAT to generate): `seg`, `bravo`, or `dual`

This gives 6 combinations (2 strategies × 3 modes).

## Training Modes

| Mode | Description | Input | Output |
|------|-------------|-------|--------|
| `seg` | Unconditional mask generation | noise [B,1,H,W] | mask [B,1,H,W] |
| `bravo` | Mask → MRI image | [noise, mask] [B,2,H,W] | image [B,1,H,W] |
| `dual` | Mask → T1 pre + T1 post | [noise×2, mask] [B,3,H,W] | [pre, post] [B,2,H,W] |

## Usage Examples

```bash
# See resolved config without running
python -m medgen.scripts.train --cfg job

# DDPM 128 bravo (default)
python -m medgen.scripts.train

# RFlow 128 bravo
python -m medgen.scripts.train strategy=rflow

# DDPM 256 dual
python -m medgen.scripts.train mode=dual model.image_size=256

# Segmentation mask generation
python -m medgen.scripts.train mode=seg

# Custom training params
python -m medgen.scripts.train training.epochs=1000 training.batch_size=32

# Disable EMA and Min-SNR
python -m medgen.scripts.train training.use_ema=false training.use_min_snr=false

# Cluster with multi-GPU
python -m medgen.scripts.train paths=cluster training.use_multi_gpu=true
```

## Generation

```bash
python -m medgen.scripts.generate \
    --checkpoint runs/bravo/ddpm_128_*/best_model.pt \
    --num_images 100 \
    --output_dir synthetic_output
```

## Cluster (IDUN)

```bash
# Edit IDUN/train_template.slurm to uncomment desired config
sbatch IDUN/train_template.slurm
```

## Installation

```bash
# Clone and install
git clone <repo>
cd AIS4900_master
pip install -e .

# Or with conda
conda create -n medgen python=3.10
conda activate medgen
pip install -e .
```

### Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA
- See requirements.txt for full list

## Documentation

- `DETAILES.md` - Detailed technical documentation
- `FUTURE_WORK.md` - Planned improvements and experiments
