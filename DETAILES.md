# MedGen - Technical Details

Detailed technical documentation for the synthetic medical image generation framework.

## Project Structure

```
AIS4900_master/
├── pyproject.toml           # Package definition, install with pip install -e .
├── README.md                # User-facing documentation
├── DETAILES.md              # This file - detailed technical docs
├── CLAUDE.md                # Claude Code context file
├── FUTURE_WORK.md           # Planned improvements and experiments
├── RESEARCH_NOTES.md        # Research notes on diffusion models
├── requirements.txt         # Python dependencies
│
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main diffusion config (composes others via defaults)
│   ├── train_vae.yaml       # VAE training config
│   ├── train_vae_progressive.yaml  # Progressive resolution VAE
│   ├── lr_finder.yaml       # Learning rate finder config
│   ├── model/
│   │   └── default.yaml     # Model architecture
│   ├── strategy/
│   │   ├── ddpm.yaml        # DDPM strategy settings
│   │   └── rflow.yaml       # Rectified Flow settings
│   ├── mode/
│   │   ├── seg.yaml         # Unconditional segmentation mask generation
│   │   ├── bravo.yaml       # Conditional single image (mask -> MRI)
│   │   ├── dual.yaml        # Conditional dual image (mask -> T1 pre + post)
│   │   ├── t1_pre.yaml      # Single T1 pre-contrast
│   │   ├── t1_gd.yaml       # Single T1 post-contrast
│   │   └── multi_modality.yaml  # For progressive VAE
│   ├── training/
│   │   ├── default.yaml     # Default training hyperparameters
│   │   └── fast_debug.yaml  # Quick debug configuration
│   └── paths/
│       ├── local.yaml       # Local Linux development
│       └── cluster.yaml     # NTNU IDUN cluster
│
├── src/medgen/              # Main Python package
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py     # ModeType enum, thresholds, defaults
│   │   └── cuda_utils.py    # CUDA optimization setup
│   ├── data/
│   │   ├── __init__.py
│   │   └── nifti_dataset.py # NiFTI dataset, transforms, all dataloaders
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── trainer.py       # DiffusionTrainer class
│   │   ├── vae_trainer.py   # VAETrainer class
│   │   ├── strategies.py    # DDPM, RFlow strategies
│   │   ├── modes.py         # Seg, Bravo, Dual modes
│   │   ├── spaces.py        # Pixel/Latent space abstractions
│   │   ├── metrics.py       # MetricsTracker for logging
│   │   ├── visualization.py # ValidationVisualizer
│   │   ├── losses.py        # Custom loss functions
│   │   └── utils.py         # Checkpointing, logging utilities
│   └── scripts/
│       ├── __init__.py
│       ├── train.py         # Diffusion training script
│       ├── train_vae.py     # VAE training script
│       ├── train_vae_progressive.py  # Progressive resolution VAE
│       ├── lr_finder.py     # Learning rate finder
│       └── generate.py      # Image generation script
│
├── IDUN/
│   └── train_template.slurm # SLURM job template
│
└── docs/
    └── pure_bf16_attempts.md  # Notes on bf16 training experiments
```

## Architecture

### Strategy + Mode Pattern

The framework uses a **Strategy + Mode** design pattern that separates concerns:

- **Strategy** (HOW to diffuse): Defines the noise/velocity prediction and sampling
- **Mode** (WHAT to generate): Defines the data format and conditioning

This gives 6 combinations (2 strategies x 3 modes).

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `DiffusionTrainer` | `diffusion/trainer.py` | Main diffusion training orchestrator |
| `VAETrainer` | `diffusion/vae_trainer.py` | VAE training with GAN |
| `DDPMStrategy` | `diffusion/strategies.py` | DDPM noise prediction |
| `RFlowStrategy` | `diffusion/strategies.py` | Rectified flow velocity |
| `SegmentationMode` | `diffusion/modes.py` | Unconditional mask gen |
| `ConditionalSingleMode` | `diffusion/modes.py` | Mask -> single image |
| `ConditionalDualMode` | `diffusion/modes.py` | Mask -> dual images |
| `PixelSpace` | `diffusion/spaces.py` | Direct pixel diffusion |
| `LatentSpace` | `diffusion/spaces.py` | VAE-compressed diffusion |
| `MetricsTracker` | `diffusion/metrics.py` | Comprehensive metrics logging |
| `ValidationVisualizer` | `diffusion/visualization.py` | Validation sample generation |

## Hydra Configuration

Config files are in `configs/`. Entry points use different main configs:

| Script | Config | Description |
|--------|--------|-------------|
| `train.py` | `config.yaml` | Diffusion training |
| `train_vae.py` | `train_vae.yaml` | VAE training |
| `train_vae_progressive.py` | `train_vae_progressive.yaml` | Progressive VAE |
| `lr_finder.py` | `lr_finder.yaml` | LR range test |

### How Hydra Works

1. `defaults:` list specifies which sub-configs to compose
2. CLI args override any value: `python -m medgen.scripts.train training.batch_size=32`
3. Hydra creates timestamped output dirs with config snapshots

### CLI Override Examples

```bash
# See resolved config without running
python -m medgen.scripts.train --cfg job

# Quick debug run
python -m medgen.scripts.train training.epochs=2 training.batch_size=4

# Change image size
python -m medgen.scripts.train model.image_size=256

# Disable training features
python -m medgen.scripts.train training.use_ema=false training.use_min_snr=false

# Full cluster training
python -m medgen.scripts.train paths=cluster strategy=rflow training.epochs=1000

# Named experiment (prefix for run directory)
python -m medgen.scripts.train training.name=exp1_
```

## Data

### Dataset
- **Source**: BrainMetShare (brain MRI with metastasis)
- **Format**: NIfTI 3D volumes (.nii.gz)
- **Modalities**: bravo, t1_pre, t1_gd, flair, seg (segmentation mask)
- **Processing**: 3D volumes -> 2D slice extraction

### Dataloader Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_dataloader()` | Single modality with seg conditioning | Tensor [B, 2, H, W] |
| `create_dual_image_dataloader()` | Dual images + seg | Dict with keys |
| `create_vae_dataloader()` | Images WITHOUT seg | Tensor [B, C, H, W] |
| `create_multi_modality_dataloader()` | Mixed modalities | Tensor [B, 1, H, W] |

### Data Format by Mode

**Diffusion Training:**
| Mode | Batch Format | Shape |
|------|--------------|-------|
| seg | Tensor | `[B, 1, H, W]` |
| bravo | Tensor | `[B, 2, H, W]` (bravo + seg) |
| dual | Dict | `{t1_pre, t1_gd, seg}` each `[B, 1, H, W]` |

**VAE Training:**
| Mode | Batch Format | Shape |
|------|--------------|-------|
| bravo | Tensor | `[B, 1, H, W]` |
| dual | Tensor | `[B, 2, H, W]` (t1_pre + t1_gd, NO seg) |
| seg | Tensor | `[B, 1, H, W]` |

## Training Features

### Diffusion Trainer Features

- **EMA (Exponential Moving Average)**: Slowly-updated weight copy for higher quality samples
- **Min-SNR Loss Weighting**: Per-sample loss weighting by timestep to prevent high-noise domination
- **Perceptual Loss**: Additional loss term using pretrained network features
- **Mixed Precision**: bfloat16 automatic mixed precision
- **torch.compile**: Fused forward pass optimization (~10% speedup)
- **Multi-GPU**: Distributed training with DDP
- **Gradient Clipping**: Configurable gradient norm clipping

### VAE Trainer Features

- **GAN Training**: PatchDiscriminator for adversarial loss
- **Perceptual Loss**: SqueezeNet-based perceptual loss
- **KL Regularization**: Latent space regularization
- **EMA**: Optional EMA for generator
- **Warmup + Cosine Scheduler**: Learning rate scheduling

### LR Finder Features

- **Diffusion LR Finder**: Tests model with strategy and mode
- **VAE LR Finder**: Tests VAE without GAN for stability
- **10x Before Divergence**: Industry-standard LR suggestion algorithm
- **Automatic Plot Generation**: Saves lr_finder.png

## Loss Functions

### Diffusion Loss
```python
total_loss = mse_loss + perceptual_weight * perceptual_loss
# MSE: noise/velocity prediction error
# Perceptual: Feature-based similarity (SqueezeNet)
# Optional: Min-SNR weighting per sample
```

### VAE Loss
```python
g_loss = L1_loss + perceptual_weight * perceptual + kl_weight * KL + adv_weight * adversarial
# L1: Pixel reconstruction (default weight: 1.0)
# Perceptual: Feature similarity (default weight: 0.002)
# KL: Latent regularization (default weight: 1e-8)
# Adversarial: GAN loss (default weight: 0.005)

d_loss = discriminator_loss  # PatchGAN discriminator
```

## Checkpoint Formats

### Diffusion Checkpoint
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'ema_state_dict': ema.state_dict(),  # if use_ema=true
    'epoch': int,
    'best_loss': float,
}
```

### VAE Checkpoint
```python
{
    'model_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_g_state_dict': optimizer_g.state_dict(),
    'optimizer_d_state_dict': optimizer_d.state_dict(),
    'scheduler_g_state_dict': scheduler_g.state_dict(),
    'scheduler_d_state_dict': scheduler_d.state_dict(),
    'epoch': int,
    'best_loss': float,
    'config': vae_architecture_config,
    'disc_config': discriminator_config,
    'ema_state_dict': ema.state_dict(),  # if use_ema=true
}
```

## Metrics and Logging

### TensorBoard Metrics

**Training Metrics:**
- `train/loss`: Total training loss
- `train/mse_loss`: MSE component
- `train/perceptual_loss`: Perceptual component
- `train/grad_norm`: Gradient norm
- `train/lr`: Learning rate

**VAE-Specific:**
- `train/g_loss`: Generator total loss
- `train/d_loss`: Discriminator loss
- `train/recon_loss`: L1 reconstruction
- `train/kl_loss`: KL divergence
- `train/adv_loss`: Adversarial loss

**Validation Metrics:**
- `val/ssim`: Structural similarity
- `val/psnr`: Peak signal-to-noise ratio
- `val/lpips`: Learned perceptual metric
- `val/loss`: Validation loss

### Logging Options (training.logging)

| Option | Description | Default |
|--------|-------------|---------|
| `grad_norm` | Track gradient norm | true |
| `timestep_losses` | Loss by diffusion timestep | true |
| `regional_losses` | Loss by tumor vs background | true |
| `ssim` | SSIM metric | true |
| `psnr` | PSNR metric | true |
| `lpips` | LPIPS metric | true |
| `boundary_sharpness` | Edge quality | true |
| `intermediate_steps` | Denoising trajectory | true |
| `worst_batch` | Highest loss batch | true |
| `flops` | Model FLOPs | true |

## Important Paths

| Environment | Data | Models |
|------------|------|--------|
| Local | `/home/mode/NTNU/MedicalDataSets/brainmetshare-3` | `./runs/` |
| Cluster | `/cluster/work/modestas/MedicalDataSets/brainmetshare-3` | `/cluster/work/modestas/AIS4900_master/runs` |

## Output Directory Structure

```
runs/
├── diffusion_2d/
│   └── {mode}/{exp_name}{strategy}_{size}_{timestamp}/
│       ├── .hydra/
│       │   └── config.yaml       # Resolved config
│       ├── metadata.json         # Training metadata
│       ├── best.pt               # Best validation model
│       ├── latest.pt             # Latest checkpoint
│       ├── epoch_*.pt            # Periodic checkpoints
│       └── tensorboard/          # TensorBoard logs
│
├── vae_2d/
│   ├── {mode}/{exp_name}{size}_{timestamp}/
│   │   ├── best.pt
│   │   ├── latest.pt
│   │   └── tensorboard/
│   └── progressive/{timestamp}/
│       ├── phase_64/             # 64x64 phase
│       ├── phase_128/            # 128x128 phase
│       ├── phase_256/            # 256x256 phase
│       ├── final_model.pt        # Final model
│       └── progressive_state.pt  # For resuming
│
└── lr_finder/
    ├── diffusion_2d/{mode}_{size}_{timestamp}/
    │   └── lr_finder.png
    └── vae_2d/{mode}_{size}_{timestamp}/
        └── lr_finder.png
```

## Extending the Framework

### Adding a New Strategy

1. Create class extending `DiffusionStrategy` in `strategies.py`:
```python
class NewStrategy(DiffusionStrategy):
    def predict_noise_or_velocity(self, model, noisy_images, timesteps):
        # Implementation
        pass

    def compute_loss(self, prediction, images, noise, noisy_images, timesteps):
        # Implementation
        pass
```

2. Add config file in `configs/strategy/new.yaml`
3. Register in trainer's strategy factory

### Adding a New Mode

1. Create class extending `TrainingMode` in `modes.py`:
```python
class NewMode(TrainingMode):
    def prepare_batch(self, batch, device):
        # Implementation
        pass

    def format_model_input(self, noisy_images, labels_dict):
        # Implementation
        pass
```

2. Add config file in `configs/mode/new.yaml`
3. Register in trainer's mode factory

### Adding a New Dataloader

1. Add function in `data/nifti_dataset.py`:
```python
def create_new_dataloader(cfg, ...):
    # Implementation
    pass
```

2. Export in `data/__init__.py`
3. Update training script to use new dataloader

## Running on IDUN Cluster

```bash
# Edit IDUN/train_template.slurm to configure:
# - Job name, time limit, resources
# - Training command and arguments

# Submit job
sbatch IDUN/train_template.slurm

# Monitor job
squeue -u $USER

# Cancel job
scancel <job_id>
```

## Common Issues and Solutions

### CUDA Out of Memory

1. Reduce batch size: `training.batch_size=8`
2. Reduce image size: `model.image_size=128`
3. Disable torch.compile: `training.compile_fused_forward=false`
4. Use gradient accumulation (not implemented yet)

### Slow Training

1. Enable torch.compile: `training.compile_fused_forward=true`
2. Increase num_workers: `training.num_workers=8`
3. Use larger batch size (if VRAM allows)
4. Use latent space diffusion with trained VAE

### Poor Generation Quality

1. Train longer: `training.epochs=1000`
2. Enable EMA: `training.use_ema=true`
3. Enable Min-SNR: `training.use_min_snr=true`
4. Use LR finder to find optimal learning rate
5. Check validation samples in TensorBoard

### VAE Training Instability

1. Use LR finder first: `python -m medgen.scripts.lr_finder model_type=vae`
2. Reduce discriminator LR: `vae.disc_lr=2e-5`
3. Reduce adversarial weight: `vae.adv_weight=0.001`
4. Increase gradient clipping: `training.gradient_clip_norm=0.5`
