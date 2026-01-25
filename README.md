# MedGen - Synthetic Medical Image Generation

[![Tests](https://github.com/ModeS7/AIS4900_master/actions/workflows/test.yml/badge.svg)](https://github.com/ModeS7/AIS4900_master/actions/workflows/test.yml)
[![Nightly](https://github.com/ModeS7/AIS4900_master/actions/workflows/tests-nightly.yml/badge.svg)](https://github.com/ModeS7/AIS4900_master/actions/workflows/tests-nightly.yml)
[![codecov](https://codecov.io/gh/ModeS7/AIS4900_master/branch/main/graph/badge.svg)](https://codecov.io/gh/ModeS7/AIS4900_master)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A modular framework for generating synthetic brain MRI images using diffusion models. Supports both pixel-space and latent-space diffusion with VAE compression.

## Quick Start

```bash
# Install
pip install -e .

# Train diffusion model (default: DDPM, 128px, bravo mode)
python -m medgen.scripts.train

# Train VAE for latent diffusion
python -m medgen.scripts.train_vae mode=dual

# Find optimal learning rate
python -m medgen.scripts.lr_finder mode=dual model_type=vae

# Generate synthetic images
python -m medgen.scripts.generate \
    --seg_model runs/seg/model.pt \
    --image_model runs/bravo/model.pt \
    --num_images 1000 --output_dir synthetic_output
```

## Features

### Training Capabilities
- **Two diffusion strategies**: DDPM (noise prediction) and Rectified Flow (velocity prediction)
- **Three training modes**: Segmentation masks, single-image (bravo), dual-image (T1 pre + T1 post)
- **VAE training**: AutoencoderKL with GAN loss for latent diffusion preparation
- **Progressive VAE**: Multi-resolution training (64 -> 128 -> 256) with plateau detection

### Training Optimizations
- **EMA weights**: Exponential moving average for higher quality samples
- **Min-SNR weighting**: Reweight loss across timesteps to prevent high-noise step domination
- **Mixed precision**: bfloat16 automatic mixed precision
- **torch.compile**: Fused forward pass optimization (~10% speedup)
- **Multi-GPU support**: Distributed training with DDP

### Data Augmentation
- **Standard augmentation**: Applied to clean images before noise addition
  - `diffusion`: Conservative (flip, discrete translate) - preserves distribution
  - `vae`: Aggressive (+ noise, blur, elastic, scale, rotate) - learns robust features
- **ScoreAug**: Applied to noisy images during training (rotation, flip, translation, cutout)
  - Omega conditioning for non-invertible transforms (rotation/flip)
- **Batch augmentation** (VAE): Mixup and CutMix for regularization
- **Visualization tool**: `python -m medgen.scripts.visualize_augmentations`

### Configuration & Monitoring
- **Hydra configuration**: Composable YAML configs with CLI overrides
- **LR Finder**: Automatic learning rate discovery for both diffusion and VAE
- **TensorBoard logging**: Comprehensive metrics and visualization
- **Checkpoint management**: Best model, latest, and periodic saves

## Project Structure

```
AIS4900_master/
├── configs/                     # Hydra configuration files
│   ├── config.yaml              # Main diffusion training config
│   ├── train_vae.yaml           # VAE training config
│   ├── train_vae_progressive.yaml  # Progressive VAE config
│   ├── lr_finder.yaml           # LR finder config
│   ├── model/default.yaml       # Model architecture
│   ├── strategy/                # ddpm.yaml, rflow.yaml
│   ├── mode/                    # seg.yaml, bravo.yaml, dual.yaml, ...
│   ├── training/                # default.yaml, fast_debug.yaml
│   └── paths/                   # local.yaml, cluster.yaml
│
├── src/medgen/                  # Main package
│   ├── core/                    # Constants, ModeType enum, CUDA setup
│   ├── data/                    # NIfTI dataset, transforms, dataloaders
│   │   ├── dataset.py           # NiFTIDataset class
│   │   ├── loaders/             # Dataloader factory functions
│   │   ├── augmentation.py      # Standard augmentation builders
│   │   ├── score_aug.py         # ScoreAug + omega conditioning
│   │   ├── mode_embed.py        # Mode embedding for multi-modality
│   │   └── combined_embed.py    # Combined omega + mode conditioning
│   ├── pipeline/                # Training pipeline components
│   │   ├── trainer.py           # DiffusionTrainer class
│   │   ├── vae_trainer.py       # VAETrainer class
│   │   ├── strategies.py        # DDPM, RFlow strategies
│   │   ├── modes.py             # Seg, Bravo, Dual modes
│   │   ├── spaces.py            # Pixel/Latent space abstractions
│   │   ├── metrics/             # MetricsTracker, quality, regional
│   │   ├── tracking/            # Gradient, FLOPs, worst batch
│   │   └── visualization.py     # ValidationVisualizer
│   └── scripts/                 # Training entry points
│       ├── train.py             # Diffusion training
│       ├── train_vae.py         # VAE training
│       ├── train_vae_progressive.py  # Progressive VAE
│       ├── lr_finder.py         # Learning rate finder
│       ├── generate.py          # Image generation
│       └── visualize_augmentations.py  # Augmentation visualization
│
├── IDUN/                        # SLURM job scripts for cluster
│   ├── train/diffusion/         # Diffusion training jobs
│   ├── train/vae/               # VAE training jobs
│   └── generate/                # Generation jobs
│
├── misc/preprocessing/          # Data preprocessing tools
│   └── preprocess.py            # Unified preprocessing CLI tool
│
├── docs/                        # Additional documentation
├── CLAUDE.md                    # Claude Code context file
├── FUTURE_WORK.md               # Planned improvements
└── RESEARCH_NOTES.md            # Research notes and references
```

## Architecture

### Strategy + Mode Pattern

The framework uses a **Strategy + Mode** design pattern:

- **Strategy** (HOW to diffuse): `ddpm` or `rflow`
- **Mode** (WHAT to generate): `seg`, `bravo`, or `dual`

This gives 6 combinations (2 strategies x 3 modes).

### Training Modes

| Mode | Description | Input Channels | Output Channels |
|------|-------------|----------------|-----------------|
| `seg` | Unconditional mask generation | 1 (noise) | 1 (mask) |
| `bravo` | Mask-conditioned MRI | 2 (noise + mask) | 1 (image) |
| `dual` | Mask-conditioned T1 pair | 3 (noise x2 + mask) | 2 (pre + post) |

### VAE Training Modes

VAE trains on images **without** segmentation conditioning:

| Mode | Channels | Description |
|------|----------|-------------|
| `bravo` | 1 | Single bravo image |
| `dual` | 2 | T1 pre + T1 post (no seg) |
| `seg` | 1 | Segmentation mask only |

## Training Scripts

### 1. Diffusion Training (`train.py`)

```bash
# Default: DDPM, 128px, bravo mode
python -m medgen.scripts.train

# Rectified Flow, dual mode
python -m medgen.scripts.train strategy=rflow mode=dual

# Custom settings
python -m medgen.scripts.train \
    model.image_size=256 \
    training.epochs=1000 \
    training.batch_size=32 \
    training.use_ema=true \
    training.use_min_snr=true

# Latent space diffusion (requires trained VAE)
python -m medgen.scripts.train \
    latent.enabled=true \
    latent.vae_checkpoint=/path/to/vae.pt

# Cluster with multi-GPU
python -m medgen.scripts.train paths=cluster training.use_multi_gpu=true
```

### 2. VAE Training (`train_vae.py`)

```bash
# Default: dual mode VAE
python -m medgen.scripts.train_vae

# Single modality (bravo, seg, t1_pre, t1_gd)
python -m medgen.scripts.train_vae mode=bravo

# Custom VAE settings
python -m medgen.scripts.train_vae \
    vae.latent_channels=4 \
    model.image_size=256 \
    training.epochs=200
```

### 3. Progressive VAE Training (`train_vae_progressive.py`)

Multi-resolution training with automatic plateau detection:

```bash
# Full progressive training (64 -> 128 -> 256)
python -m medgen.scripts.train_vae_progressive

# Resume from checkpoint
python -m medgen.scripts.train_vae_progressive \
    progressive.resume_from=/path/to/progressive_state.pt

# Custom plateau detection
python -m medgen.scripts.train_vae_progressive \
    progressive.plateau.min_improvement=1.0 \
    progressive.final_phase.epochs=200
```

### 4. Learning Rate Finder (`lr_finder.py`)

Find optimal learning rate before training:

```bash
# Diffusion LR finder
python -m medgen.scripts.lr_finder mode=dual strategy=rflow

# VAE LR finder
python -m medgen.scripts.lr_finder mode=dual model_type=vae

# Custom LR range
python -m medgen.scripts.lr_finder min_lr=1e-8 max_lr=1e-2 num_steps=300
```

### 5. Image Generation (`generate.py`)

```bash
# BRAVO mode generation
python -m medgen.scripts.generate \
    --strategy ddpm \
    --mode bravo \
    --seg_model runs/seg/best.pt \
    --image_model runs/bravo/best.pt \
    --num_images 15000 \
    --output_dir gen_bravo

# Dual mode generation
python -m medgen.scripts.generate \
    --strategy rflow \
    --mode dual \
    --seg_model runs/seg/best.pt \
    --image_model runs/dual/best.pt \
    --num_images 10000 \
    --output_dir gen_dual
```

## Configuration

### Viewing Resolved Config

```bash
# See full resolved config without running
python -m medgen.scripts.train --cfg job

# See specific config group
python -m medgen.scripts.train --cfg job training
```

### Common CLI Overrides

```bash
# Image size
model.image_size=256

# Training params
training.epochs=1000
training.batch_size=32
training.learning_rate=5e-5

# Enable features
training.use_ema=true
training.use_min_snr=true

# Disable torch.compile (for debugging)
training.compile_fused_forward=false

# Debug mode (fast iteration)
training=fast_debug

# Cluster paths
paths=cluster
```

### Named Experiments

Add a prefix to run directory:

```bash
# Creates: runs/diffusion_2d/dual/exp1_rflow_128_20241214-...
python -m medgen.scripts.train training.name=exp1_ mode=dual strategy=rflow
```

## Output Directory Structure

```
runs/
├── diffusion_2d/
│   └── {mode}/{exp_name}{strategy}_{size}_{timestamp}/
│       ├── .hydra/config.yaml    # Resolved config
│       ├── metadata.json         # Training metadata
│       ├── best.pt               # Best validation model
│       ├── latest.pt             # Latest checkpoint
│       ├── epoch_*.pt            # Periodic checkpoints
│       └── tensorboard/          # TensorBoard logs
│
├── vae_2d/
│   ├── {mode}/{exp_name}{size}_{timestamp}/
│   └── progressive/{timestamp}/
│       ├── phase_64/, phase_128/, phase_256/
│       ├── final_model.pt
│       └── progressive_state.pt  # For resuming
│
└── lr_finder/
    ├── diffusion_2d/{mode}_{size}_{timestamp}/
    └── vae_2d/{mode}_{size}_{timestamp}/
        └── lr_finder.png         # LR vs Loss plot
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
- MONAI 1.3+
- See `requirements.txt` for full list

### Hardware Requirements

- **Minimum**: 8GB VRAM (128px, batch_size=8)
- **Recommended**: 24GB VRAM (256px, batch_size=32)
- **Multi-GPU**: DDP support for scaling

## Cluster (IDUN)

```bash
# Training jobs
sbatch IDUN/train/diffusion/exp1_rflow_128_baseline.slurm
sbatch IDUN/train/vae/exp1_progressive_baseline.slurm

# Prefer H100, fallback to A100 after 10 min
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm
```

IDUN structure:
```
IDUN/
├── train/
│   ├── diffusion/    # Diffusion training experiments
│   └── vae/          # VAE training experiments
├── generate/         # Generation jobs
├── output/           # SLURM logs
│   ├── train/diffusion/
│   ├── train/vae/
│   └── generate/
└── submit_prefer_h100.sh  # H100 with A100 fallback script
```

## Data Preprocessing

Unified preprocessing tool (`misc/preprocessing/preprocess.py`) with subcommands:

```bash
# Resize images (pad to 240x240 centered, resize to 256x256)
python misc/preprocessing/preprocess.py resize -i /path/to/raw -o /path/to/processed

# Align all modalities to same slice count (truncate/pad to match)
python misc/preprocessing/preprocess.py align --data_dir /path/to/data -t 150 --dry_run
python misc/preprocessing/preprocess.py align --data_dir /path/to/data -t 150

# Pad volumes to target slice count (add zeros at end)
python misc/preprocessing/preprocess.py pad --data_dir /path/to/data -t 150

# Auto-trim empty slices (detect and remove empty leading/trailing slices)
python misc/preprocessing/preprocess.py trim-auto --data_dir /path/to/data --trim_trailing

# Manual trimming (interactively specify slices to remove per patient)
python misc/preprocessing/preprocess.py trim-manual --data_dir /path/to/data/test

# Split test set into val and test_new
python misc/preprocessing/preprocess.py split --data_dir /path/to/data --val_ratio 0.5
```

Available subcommands:
| Command | Description |
|---------|-------------|
| `resize` | Pad to 240x240 (centered) then resize to 256x256 |
| `align` | Align all modalities per patient to same slice count |
| `pad` | Pad volumes to target slice count (add zeros) |
| `trim-auto` | Auto-detect and trim empty slices |
| `trim-manual` | Interactive per-patient slice trimming |
| `split` | Split test into val and test_new |

Resize pipeline:
- **Images**: Load → Pad to 240x240 (centered) → Resize to 256x256 (bilinear)
- **Segmentation**: Load → Pad to 240x240 (centered) → Resize to 256x256 (nearest) → Binarize

## Logging & Monitoring

TensorBoard logging with configurable metrics:

```yaml
# configs/training/default.yaml
logging:
  grad_norm: true              # Gradient norm tracking
  timestep_losses: true        # Loss by diffusion timestep
  regional_losses: true        # Tumor vs background region losses
  timestep_region_losses: true # 2D heatmap: timestep x region
  ssim: true                   # Structural similarity
  psnr: true                   # Peak signal-to-noise ratio
  lpips: true                  # Perceptual similarity
  worst_batch: true            # Highest loss batch visualization
  flops: true                  # Model FLOPs measurement
```

Key TensorBoard metrics:
- `regional/tumor_loss`, `regional/background_loss` - Per-pixel regional error (pixel-weighted)
- `regional/{tiny,small,medium,large}` - Error by tumor size (Feret diameter, RANO-BM thresholds)
- `Validation/worst_batch` - Worst performing batch (for debugging)
- `training/timestep_region_heatmap` - 2D heatmap of loss by timestep and region

## Documentation

- `docs/architecture.md` - Architecture reference and TensorBoard metrics
- `docs/common-pitfalls.md` - Common issues and solutions
- `FUTURE_WORK.md` - Planned improvements and experiments
- `docs/pure_bf16_attempts.md` - Notes on pure bf16 training experiments
