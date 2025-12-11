# Synthetic Medical Image Generation

A modular framework for generating synthetic MRI images using diffusion models (DDPM and Rectified Flow).

## Overview

This project implements a flexible medical image synthesis pipeline with:
- **Two diffusion strategies**: DDPM (Denoising Diffusion Probabilistic Models) and RFlow (Rectified Flow)
- **Three training modes**: Unconditional segmentation masks, conditional single-image generation, and conditional dual-image generation
- **Comprehensive evaluation**: FID, LPIPS, MS-SSIM, MAE, PSNR metrics
- **Cluster support**: SLURM job scheduling with multi-GPU training

## Key Features

- Modular architecture separating diffusion strategies from training modes
- Strategy pattern for easy extension (DDPM, RFlow, custom strategies)
- Multiple training modes for different medical imaging scenarios
- Perceptual loss (RadImageNet ResNet50) for medical image quality
- Mixed precision training (bfloat16) with torch.compile optimization
- **EMA (Exponential Moving Average)** for stable generation quality
- **Min-SNR loss weighting** for improved training dynamics
- **Automatic LR finder** for optimal learning rate selection
- Distributed training support (DDP)
- Automatic checkpointing with best model tracking
- TensorBoard logging and visualization

## Architecture

### Design Pattern: Strategy + Mode

```
┌───────────────────────────────────────────────┐
│            DiffusionTrainer                   │
│  ┌──────────────┐        ┌─────────────┐      │
│  │   Strategy   │        │    Mode     │      │
│  │  (HOW)       │        │   (WHAT)    │      │
│  ├──────────────┤        ├─────────────┤      │
│  │ • DDPM       │        │ • Seg       │      │
│  │ • RFlow      │   +    │ • Bravo     │      │
│  │ • Custom     │        │ • Dual      │      │
│  └──────────────┘        └─────────────┘      │
└───────────────────────────────────────────────┘
```

**Strategy** = HOW to diffuse (DDPM vs RFlow)  
**Mode** = WHAT to train (segmentation masks, single image, dual images)

### Project Structure

```
Generation/TrainGen/
├── core/
│   ├── strategies.py    # Diffusion algorithms (DDPM, RFlow)
│   ├── modes.py         # Training modes (Seg, Bravo, Dual)
│   ├── trainer.py       # Unified training orchestrator
│   ├── data.py          # Data loading and preprocessing
│   └── utils.py         # Helpers (logging, checkpointing)
├── train.py             # Training entry point
└── generate.py          # Generation/inference script

config/
└── paths.py             # Centralized path configuration (PathConfig)

eval/obj_eval/
├── metrics/
│   ├── fid.py           # Fréchet Inception Distance
│   ├── lpips_eval.py    # Perceptual similarity
│   ├── ms_ssim.py       # Multi-scale structural similarity
│   └── mae_psnr.py      # Pixel-level metrics
├── core/
│   └── data_loader.py   # Evaluation data loading
├── run_evaluation.py    # Unified evaluation runner
└── generate_report.py   # Report generation
```

## Training Modes

### Mode 1: Segmentation (Unconditional)
**Purpose**: Generate synthetic segmentation masks without conditioning

```bash
python Generation/TrainGen/train.py \
    --mode seg \
    --strategy ddpm \
    --epochs 500 \
    --batch_size 16 \
    --image_size 128
```

- **Input**: Random noise [B, 1, H, W]
- **Output**: Segmentation masks [B, 1, H, W]
- **Use case**: Data augmentation for segmentation models

### Mode 2: Bravo (Conditional Single)
**Purpose**: Generate MRI images conditioned on segmentation masks

```bash
python Generation/TrainGen/train.py \
    --mode bravo \
    --strategy rflow \
    --epochs 500 \
    --batch_size 16 \
    --image_size 128
```

- **Input**: [noise, segmentation_mask] [B, 2, H, W]
- **Output**: MRI image [B, 1, H, W]
- **Use case**: Controlled synthetic MRI generation

### Mode 3: Dual (Conditional Multi-Image)
**Purpose**: Generate paired MRI sequences (T1 pre + T1 gd) with anatomical consistency

```bash
python Generation/TrainGen/train.py \
    --mode dual \
    --strategy ddpm \
    --epochs 500 \
    --batch_size 16 \
    --image_size 128
```

- **Input**: [noise_pre, noise_gd, segmentation_mask] [B, 3, H, W]
- **Output**: [T1_pre, T1_gd] [B, 2, H, W]
- **Use case**: Multi-modal medical imaging with anatomical coherence

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.6.0+cu124
- CUDA 12.4+

### Setup

```bash
# Clone repository
git clone <repository_url>
cd AIS4005_IP

# Create virtual environment
conda create -n ais4005 python=3.10
conda activate ais4005

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
monai>=1.4.0
ema-pytorch>=0.7.0
tqdm
tensorboard
einops
scikit-image
nibabel
pillow
opencv-python
lpips
pandas
numpy
gdown
matplotlib
tabulate
```

## Usage

### Training

**Local Training (Single GPU)**
```bash
python Generation/TrainGen/train.py \
    --mode bravo \
    --strategy ddpm \
    --epochs 500 \
    --batch_size 16 \
    --image_size 128 \
    --compute local
```

**Cluster Training (Multi-GPU)**
```bash
sbatch IDUN/slurmT/ddpm_128_bravo.slurm
```

**Multi-GPU Training**
```bash
python Generation/TrainGen/train.py \
    --mode dual \
    --strategy rflow \
    --multi_gpu \
    --batch_size 32 \
    --compute cluster
```

### Generation

**Generate Synthetic Images**
```bash
python Generation/TrainGen/generate.py \
    --strategy ddpm \
    --mode bravo \
    --seg_model "Diffusion_seg_128_*/Epoch199_of_200" \
    --image_model "Diffusion_bravo_128_*/Epoch199_of_200" \
    --num_images 15000 \
    --num_steps 1000 \
    --batch_size 8 \
    --output_dir "synthetic_dataset"
```

### Evaluation

**Run All Metrics**
```bash
python eval/obj_eval/run_evaluation.py \
    --synthetic_dir /path/to/synthetic \
    --real_dir /path/to/real \
    --metrics all \
    --positive-only
```

**Individual Metrics**
```bash
# FID (Fréchet Inception Distance)
python eval/obj_eval/metrics/fid.py \
    --synthetic-dir /path/to/synthetic \
    --real-dir /path/to/real \
    --image-type bravo

# LPIPS (Perceptual Quality)
python eval/obj_eval/metrics/lpips_eval.py \
    --synthetic-dir /path/to/synthetic \
    --real-dir /path/to/real \
    --mode quality

# MS-SSIM (Structural Similarity)
python eval/obj_eval/metrics/ms_ssim.py \
    --synthetic-dir /path/to/synthetic \
    --mode diversity

# MAE & PSNR
python eval/obj_eval/metrics/mae_psnr.py \
    --synthetic-dir /path/to/synthetic \
    --real-dir /path/to/real
```

## SLURM Configuration (IDUN Cluster)

### Available GPU Constraints
```
p100, v100, a100, h100
gpu16g, gpu32g, gpu40g, gpu80g
sxm4
```

### Example SLURM Script
```bash
#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4005

python Generation/TrainGen/train.py --compute cluster --mode bravo
```

### Monitor Training
```bash
# Watch job
./IDUN/watch_job.sh <JOB_ID>

# Check logs
tail -f IDUN/output/train/ddpm_128_bravo_*.out
```

## Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **FID** | Distributional similarity (RadImageNet/ImageNet) | [0, ∞) | Lower |
| **LPIPS** | Perceptual similarity | [0, 1] | Lower (quality) / Higher (diversity) |
| **MS-SSIM** | Structural similarity | [0, 1] | Higher (quality) / Lower (diversity) |
| **MAE** | Mean absolute error | [0, ∞) | Lower |
| **PSNR** | Peak signal-to-noise ratio | [0, ∞) dB | Higher |

## Advanced Features

### EMA (Exponential Moving Average)
Maintains a moving average of model weights for more stable generation quality.
```bash
# Enabled by default, disable with:
python train.py --mode bravo --no_ema
```

### Min-SNR Loss Weighting
Applies signal-to-noise ratio weighting to the diffusion loss for improved training dynamics across timesteps.
```bash
# Enabled by default, disable with:
python train.py --mode bravo --no_min_snr
```

### Automatic Learning Rate Finder
Runs a learning rate range test before training to find the optimal learning rate.
```bash
python train.py --mode bravo --find_lr
```

### Custom Strategies
Extend `DiffusionStrategy` to implement new algorithms:
```python
class CustomStrategy(DiffusionStrategy):
    def setup_scheduler(self, num_timesteps, image_size):
        # Implement scheduler setup
        pass
    
    def compute_loss(self, prediction, target_images, noise, timesteps):
        # Implement loss computation
        pass
```

### Custom Training Modes
Extend `TrainingMode` for new conditioning schemes:
```python
class CustomMode(TrainingMode):
    @property
    def is_conditional(self):
        return True
    
    def prepare_batch(self, batch, device):
        # Implement batch preparation
        pass
```


## Command Line Arguments

### Training Arguments (`train.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 500 | Number of training epochs |
| `--val_interval` | int | 30 | Validation interval in epochs |
| `--batch_size` | int | 16 | Batch size for training |
| `--image_size` | int | 128 | Image size for training |
| `--compute` | str | local | Compute environment: `local` or `cluster` |
| `--num_timesteps` | int | 1000 | Number of diffusion timesteps |
| `--warmup_epochs` | int | 5 | Number of epochs for learning rate warmup |
| `--mode` | str | seg | Training mode: `seg`, `bravo`, or `dual` |
| `--strategy` | str | ddpm | Diffusion strategy: `ddpm` or `rflow` |
| `--multi_gpu` | flag | - | Enable multi-GPU distributed training |
| `--find_lr` | flag | - | Run LR finder before training |
| `--no_ema` | flag | - | Disable EMA weight tracking |
| `--no_min_snr` | flag | - | Disable Min-SNR loss weighting |

**Example with all options:**
```bash
python Generation/TrainGen/train.py \
    --mode bravo \
    --strategy rflow \
    --epochs 500 \
    --batch_size 16 \
    --image_size 128 \
    --num_timesteps 1000 \
    --val_interval 30 \
    --warmup_epochs 5 \
    --compute local \
    --find_lr
```

## Performance Optimizations

- Mixed precision training (bfloat16)
- torch.compile for model optimization
- CUDA optimizations (TF32, Flash Attention, efficient SDP)
- Pin memory and optimized data loading
- Auto batch size adjustment based on VRAM