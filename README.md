# MedGen - Synthetic Medical Image Generation

[![Tests](https://github.com/ModeS7/AIS4900_master/actions/workflows/test.yml/badge.svg)](https://github.com/ModeS7/AIS4900_master/actions/workflows/test.yml)
[![Nightly](https://github.com/ModeS7/AIS4900_master/actions/workflows/tests-nightly.yml/badge.svg)](https://github.com/ModeS7/AIS4900_master/actions/workflows/tests-nightly.yml)
[![codecov](https://codecov.io/gh/ModeS7/AIS4900_master/branch/main/graph/badge.svg)](https://codecov.io/gh/ModeS7/AIS4900_master)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A modular framework for generating synthetic brain MRI images using diffusion models and compression autoencoders. Supports 2D/3D pixel-space and latent-space diffusion with multiple compression backends (VAE, VQ-VAE, DC-AE).

## Quick Start

```bash
# Install
pip install -e .

# === DIFFUSION ===
python -m medgen.scripts.train mode=bravo strategy=rflow                       # 2D
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3  # 3D

# === COMPRESSION ===
python -m medgen.scripts.train_compression --config-name=vae mode=multi_modality     # VAE
python -m medgen.scripts.train_compression --config-name=vqvae mode=multi_modality   # VQ-VAE
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality    # DC-AE 32x
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality  # DC-AE 64x

# === LATENT DIFFUSION ===
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt

# === DOWNSTREAM SEGMENTATION ===
python -m medgen.scripts.train_segmentation scenario=baseline

# === GENERATION ===
python -m medgen.scripts.generate checkpoint_path=runs/.../best.pt mode=bravo strategy=rflow

# === LR FINDER ===
python -m medgen.scripts.lr_finder mode=dual strategy=rflow
```

## Features

### Diffusion Training
- **Two strategies**: DDPM (noise prediction) and Rectified Flow (velocity prediction, continuous timesteps)
- **Multiple modes**: seg, bravo, dual, multi, seg_conditioned (tumor-size-conditioned mask generation)
- **Two architectures**: UNet (MONAI) and DiT (Scalable Interpolant Transformer, S/B/L variants)
- **Pixel and latent space**: Direct pixel diffusion or compressed latent diffusion
- **2D and 3D**: Unified trainer via `model.spatial_dims` parameter
- **ControlNet**: Pixel-resolution conditioning for latent diffusion (two-stage training)
- **3D pixel transforms**: Space-to-depth rearrangement and Haar wavelet decomposition
- **Classifier-free guidance**: Train with condition dropout, control strength at inference

### Compression Training
- **VAE**: AutoencoderKL with PatchDiscriminator (8x compression)
- **VQ-VAE**: Vector-quantized discrete latent space (8x, 512 codebook)
- **DC-AE**: Deep Compression Autoencoder (32x/64x/128x spatial compression)
- **DC-AE 1.5**: Structured latent space with channel masking for faster diffusion convergence
- **Seg mask compression**: BCE + Dice + Boundary loss for binary mask encoding
- **All 2D/3D**: Unified trainers via `.create_3d()` factory methods

### Regularization
- **ScoreAug / v2**: Augments noisy data (rotation, flip, translation, cutout, learned patterns)
- **SDA**: Shifted Data Augmentation (clean data augmentation with timestep compensation)
- **SAM/ASAM**: Sharpness-Aware Minimization for flat minima
- **Clean regularization**: Gradient noise, curriculum timesteps, timestep jitter, noise augmentation, feature perturbation, self-conditioning

### Training Infrastructure
- **EMA weights** for higher quality samples
- **Mixed precision**: BF16 autocast + optional pure BF16 weights
- **torch.compile** fused forward pass optimization
- **Multi-GPU** distributed training with DDP
- **Gradient checkpointing** for 3D memory reduction
- **Regional loss weighting** to upweight small tumor regions
- **Generation metrics**: KID, CMMD, FID tracking during training

### Evaluation
- **Downstream segmentation**: SegResNet trainer with per-tumor-size Dice (RANO-BM thresholds)
- **Test evaluation**: L1, MS-SSIM, PSNR, LPIPS, FID, KID, CMMD
- **TensorBoard**: Comprehensive metrics, worst batch visualization, denoising trajectory
- **Regional metrics**: Per-tumor error by clinical size category (tiny/small/medium/large)

## Architecture

### Strategy + Mode Pattern

The framework uses a **Strategy + Mode** design pattern:

- **Strategy** (HOW to diffuse): `ddpm` or `rflow`
- **Mode** (WHAT to generate): `seg`, `bravo`, `dual`, `multi`, `seg_conditioned`

### Trainer Hierarchy

```
BaseTrainer
├── DiffusionTrainerBase (abstract)
│   └── DiffusionTrainer (unified 2D/3D via spatial_dims)
├── BaseCompressionTrainer
│   ├── VAETrainer (unified 2D/3D via .create_3d())
│   ├── VQVAETrainer (unified 2D/3D via .create_3d())
│   └── DCAETrainer (unified 2D/3D via .create_3d())
└── SegmentationTrainer (downstream, unified 2D/3D)
```

### Diffusion Modes

| Mode | Input Channels | Output Channels | Conditioning |
|------|----------------|-----------------|--------------|
| `seg` | 1 | 1 | None |
| `bravo` | 2 | 1 | Seg mask |
| `dual` | 3 | 2 | Seg mask |
| `multi` | 2 | 1 | Seg mask + mode_id |
| `seg_conditioned` | 1 | 1 | Tumor size bins (FiLM embedding) |
| `seg_conditioned_input` | 1 + 7 bin maps | 1 | Tumor size bins (channel concat) |

### Compression Modes (no seg conditioning)

| Mode | Channels | Description |
|------|----------|-------------|
| `bravo` | 1 | Single bravo image |
| `dual` | 2 | T1 pre + T1 post (no seg) |
| `seg` | 1 | Segmentation mask only |
| `multi_modality` | 1 | Pools all modalities (bravo, flair, t1_pre, t1_gd) |
| `seg_compression` | 1 | Seg mask compression (DC-AE, BCE+Dice+Boundary loss) |

### Diffusion Spaces

| Space | Compression | Use Case |
|-------|-------------|----------|
| `PixelSpace` | 1x | Direct pixel diffusion (default) |
| `LatentSpace` | 8x-128x | VAE/DC-AE compressed latent diffusion |
| `SpaceToDepthSpace` | 2x | Lossless 2x2x2 rearrangement (3D) |
| `WaveletSpace` | 2x | 3D Haar wavelet frequency decomposition |

## Project Structure

```
AIS4900_master/
├── configs/                         # Hydra configuration files
│   ├── diffusion.yaml               # 2D diffusion training
│   ├── vae.yaml / vae_3d.yaml       # VAE training (2D/3D)
│   ├── vqvae.yaml / vqvae_3d.yaml   # VQ-VAE training (2D/3D)
│   ├── dcae.yaml / dcae_3d.yaml     # DC-AE training (2D/3D)
│   ├── segmentation.yaml            # Downstream segmentation
│   ├── generate.yaml                # Generation/inference
│   ├── lr_finder.yaml               # Learning rate finder
│   ├── controlnet/                  # ControlNet conditioning
│   ├── latent/                      # Latent diffusion settings
│   ├── space_to_depth/              # 3D space-to-depth transform
│   ├── wavelet/                     # 3D Haar wavelet transform
│   ├── dcae/{f32,f64,f128}.yaml     # DC-AE compression variants
│   ├── model/{default,dit,...}.yaml  # Model architectures
│   ├── strategy/{ddpm,rflow}.yaml   # Diffusion strategies
│   ├── mode/{seg,bravo,...}.yaml    # Generation modes
│   ├── training/                    # Training hyperparameters
│   └── paths/{local,cluster}.yaml   # Data paths
│
├── src/medgen/                      # Main package
│   ├── augmentation/                # ScoreAug, SDA, standard transforms
│   ├── core/                        # Constants, enums, schedulers, utilities
│   ├── data/                        # NiFTI dataset, loaders, mask codec
│   │   └── loaders/                 # Dataloader factories (17 modules)
│   ├── diffusion/                   # Strategies (DDPM/RFlow), modes, spaces
│   ├── downstream/                  # Downstream segmentation evaluation
│   ├── evaluation/                  # Test evaluation, validation runners
│   ├── losses/                      # Perceptual, segmentation, regional losses
│   ├── metrics/                     # Unified metrics, quality, generation, regional
│   │   ├── regional/               # Per-tumor metrics by size
│   │   └── tracking/               # Gradient, FLOPs, codebook, worst batch
│   ├── models/                      # UNet, DiT, DC-AE 3D, ControlNet, wavelets
│   │   └── wrappers/               # Conditioning wrappers (mode, omega, size bin)
│   ├── pipeline/                    # Trainers, configs, checkpoint management
│   │   └── optimizers/             # SAM/ASAM optimizer
│   └── scripts/                     # Entry points
│       ├── train.py                 # Diffusion training (2D/3D)
│       ├── train_compression.py     # Compression (VAE/VQ-VAE/DC-AE, 2D/3D)
│       ├── train_segmentation.py    # Downstream segmentation
│       ├── generate.py              # Image generation
│       ├── encode_latents.py        # Pre-encode for latent diffusion
│       ├── lr_finder.py             # Learning rate finder
│       └── visualize_augmentations.py
│
├── IDUN/                            # SLURM cluster jobs
│   ├── train/compression/           # Compression experiments
│   ├── train/diffusion/             # Diffusion experiments
│   ├── downstream/                  # Segmentation evaluation jobs
│   ├── generate/                    # Generation jobs
│   └── submit_prefer_h100.sh       # H100 with A100 fallback
│
├── misc/                            # Utilities
│   ├── data_processing/             # Preprocessing (resize, align, trim, split)
│   ├── analysis/                    # Dataset and tumor analysis
│   ├── profiling/                   # Memory profiling
│   ├── visualization/               # NIfTI viewer, scheduler plots
│   └── tensorboard/                 # TensorBoard utilities
│
├── tests/                           # Test suite (1126 tests)
│   ├── unit/                        # Unit tests (36 files)
│   ├── integration/                 # Integration tests
│   ├── e2e/                         # End-to-end pipeline tests
│   └── benchmarks/                  # Performance benchmarks
│
├── docs/                            # Documentation
│   ├── architecture.md              # Architecture reference, TensorBoard metrics
│   ├── commands.md                  # Full command reference
│   └── common-pitfalls.md           # 71 known issues and solutions
│
└── papers/                          # Reference papers + PAPERS.md catalog
```

## Training Scripts

| Script | Purpose | Config | Usage |
|--------|---------|--------|-------|
| `train.py` | Diffusion (2D/3D) | `diffusion.yaml` | `model.spatial_dims=3` for 3D |
| `train_compression.py` | Compression (VAE/VQ-VAE/DC-AE, 2D/3D) | `--config-name=vae/vqvae/dcae` | Append `_3d` for 3D configs |
| `train_segmentation.py` | Downstream segmentation | `segmentation.yaml` | `scenario=baseline/synthetic/mixed` |
| `generate.py` | Image generation | `generate.yaml` | |
| `encode_latents.py` | Pre-encode dataset to latent space | N/A | |
| `lr_finder.py` | Learning rate finder | `lr_finder.yaml` | |
| `visualize_augmentations.py` | Debug augmentation pipelines | `visualize_augmentations.yaml` | |

## Configuration

Hydra-based composable YAML configs with CLI overrides.

```bash
# View resolved config without running
python -m medgen.scripts.train --cfg job

# Common overrides
python -m medgen.scripts.train \
    strategy=rflow \
    mode=dual \
    model.image_size=256 \
    training.epochs=1000 \
    training.batch_size=32 \
    training.use_ema=true \
    training.name=exp1_

# Debug mode
python -m medgen.scripts.train training=fast_debug

# Cluster paths
python -m medgen.scripts.train paths=cluster
```

For the full command reference, see `docs/commands.md`.

## Output Directory Structure

```
runs/
├── diffusion_2d/{mode}/{strategy}_{size}_{timestamp}/
│   ├── .hydra/config.yaml    # Resolved config
│   ├── metadata.json         # Training metadata
│   ├── best.pt               # Best validation model
│   ├── latest.pt             # Latest checkpoint
│   └── tensorboard/          # TensorBoard logs
├── compression_2d/{mode}/{size}_{timestamp}/
├── diffusion_3d/
├── compression_3d/
├── segmentation/
└── lr_finder/
```

## Installation

```bash
git clone <repo>
cd AIS4900_master
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- MONAI 1.3+
- See `requirements.txt` for full list

### Hardware Requirements

- **Minimum**: 8GB VRAM (128px 2D, batch_size=8)
- **Recommended**: 24-40GB VRAM (256px 2D, batch_size=16-32)
- **3D training**: 40-80GB VRAM (gradient checkpointing recommended)
- **Multi-GPU**: DDP support for scaling

## Cluster (IDUN)

```bash
# Submit training jobs
sbatch IDUN/train/compression/exp9_dcae_f32.slurm

# Prefer H100, fallback to A100 after 10 min
./IDUN/submit_prefer_h100.sh IDUN/train/compression/exp9_dcae_f32.slurm

# Run in background
./IDUN/submit_prefer_h100.sh IDUN/train/compression/exp9_dcae_f32.slurm --bg

# Validate before submit (catches syntax/import/config errors)
./misc/validate_before_submit.sh IDUN/train/your_job.slurm
```

## Data Preprocessing

Unified preprocessing tool:

```bash
python misc/data_processing/preprocessing/preprocess.py resize -i /path/to/raw -o /path/to/processed
python misc/data_processing/preprocessing/preprocess.py align --data_dir /path/to/data -t 150
python misc/data_processing/preprocessing/preprocess.py trim-auto --data_dir /path/to/data
python misc/data_processing/preprocessing/preprocess.py split --data_dir /path/to/data
```

| Command | Description |
|---------|-------------|
| `resize` | Pad to 240x240 (centered) then resize to 256x256 |
| `align` | Align all modalities per patient to same slice count |
| `pad` | Pad volumes to target slice count |
| `trim-auto` | Auto-detect and trim empty slices |
| `trim-manual` | Interactive per-patient slice trimming |
| `split` | Split test into val and test_new |

## Logging & Monitoring

TensorBoard logging with configurable metrics:

```yaml
logging:
  grad_norm: true              # Gradient norm tracking
  timestep_losses: true        # Loss by diffusion timestep (10 bins)
  regional_losses: true        # Tumor vs background per-pixel error
  ssim: true                   # Structural similarity
  psnr: true                   # Peak signal-to-noise ratio
  lpips: true                  # Perceptual similarity
  worst_batch: true            # Highest loss batch visualization
  flops: true                  # Model FLOPs measurement
```

## Documentation

| Doc | Contents |
|-----|----------|
| `docs/architecture.md` | Architecture reference, TensorBoard metrics, config details |
| `docs/commands.md` | Full command reference with all options |
| `docs/common-pitfalls.md` | 71 known issues, bug fixes, and gotchas |
| `papers/PAPERS.md` | Reference papers (VAE, DDPM, RFlow, DC-AE, etc.) |
| `CLAUDE.md` | Claude Code context file |
