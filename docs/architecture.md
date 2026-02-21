# MedGen Architecture Reference

Reference this file with `@docs/architecture.md` when you need detailed project information.

## Directory Structure

```
src/medgen/
├── augmentation/                # Data augmentation
│   ├── augmentation.py          # Standard transforms (diffusion/vae)
│   ├── score_aug.py             # ScoreAug/v2 transforms
│   ├── score_aug_omega.py       # Omega conditioning encoding
│   ├── score_aug_patterns.py    # Fixed pattern masks (v2)
│   ├── score_aug_wrapper.py     # ScoreAug model wrapper
│   └── sda.py                   # Shifted Data Augmentation
├── core/                        # Constants, ModeType enum, CUDA setup, validation
│   ├── constants.py             # ModeType enum, thresholds
│   ├── cuda_utils.py            # CUDA setup and optimization
│   ├── defaults.py              # Default configuration values
│   ├── dict_utils.py            # Dictionary utilities
│   ├── distributed.py           # DDP utilities
│   ├── mode_factory.py          # Mode creation from config
│   ├── model_utils.py           # Model utility functions
│   ├── schedulers.py            # LR schedulers (cosine, warmup, plateau, constant)
│   ├── spatial_utils.py         # 2D/3D spatial helpers
│   └── validation.py            # Config validation
├── data/
│   ├── dataset.py               # NiFTIDataset class
│   ├── lossless_mask_codec.py   # Lossless binary mask encoding to DC-AE latent shape
│   ├── utils.py                 # Slice extraction, merge utilities
│   └── loaders/                 # Dataloader factory functions
│       ├── base.py              # Base loader abstractions
│       ├── builder_2d.py        # LoaderSpec pattern for 2D
│       ├── common.py            # DataLoaderConfig, GroupedBatchSampler, MODALITY_KEYS
│       ├── compression_detection.py # Auto-detect compression type from checkpoint
│       ├── datasets.py          # Dataset construction helpers
│       ├── dual.py              # Dual image dataloader
│       ├── latent.py            # Pre-encoded latent dataloader (2D/3D)
│       ├── multi_diffusion.py   # Multi-modality diffusion with mode_id
│       ├── multi_modality.py    # Multi-modality compression loaders
│       ├── seg.py               # Segmentation-only loaders
│       ├── seg_compression.py   # Seg mask compression loaders
│       ├── seg_conditioned.py   # Seg conditioned on tumor sizes (2D/3D)
│       ├── single.py            # Single modality loaders (seg, bravo)
│       ├── unified.py           # Unified loader dispatch
│       ├── vae.py               # VAE dataloaders
│       └── volume_3d.py         # 3D volumetric loaders + VolumeConfig
├── diffusion/                   # Diffusion strategies, modes, spaces
│   ├── batch_data.py            # BatchData standardized unpacking
│   ├── conditioning.py          # ConditioningContext (frozen dataclass)
│   ├── diffrs.py                # DiffRS (Diffusion Rejection Sampling)
│   ├── loading.py               # Model checkpoint loading utilities
│   ├── modes.py                 # Seg, Bravo, Dual, Multi, SegConditioned modes
│   ├── protocols.py             # Strategy/Mode protocols
│   ├── spaces.py                # Pixel/Latent/SpaceToDepth/Wavelet space
│   ├── strategies.py            # Shared strategy base
│   ├── strategy_ddpm.py         # DDPM strategy
│   └── strategy_rflow.py        # RFlow strategy
├── downstream/                  # Downstream task evaluation
│   ├── data.py                  # Segmentation data loading
│   └── segmentation_trainer.py  # SegResNet trainer (2D/3D)
├── evaluation/                  # Test evaluation and validation
│   ├── evaluation.py            # BaseTestEvaluator, CompressionTestEvaluator
│   ├── evaluation_3d.py         # 3D-specific evaluation
│   ├── evaluation_logging.py    # Evaluation result logging
│   ├── validation.py            # ValidationRunner for compression trainers
│   └── visualization.py         # Validation visualization
├── losses/                      # Loss functions
│   ├── losses.py                # PerceptualLoss, SegmentationLoss (BCE+Dice+Boundary)
│   ├── perceptual_manager.py    # Perceptual loss lifecycle management
│   └── regional_weighting.py    # Adaptive regional loss weighting
├── metrics/                     # Quality metrics
│   ├── unified.py               # Unified metrics system (MANDATORY for all trainers)
│   ├── unified_history.py       # Metric history tracking
│   ├── unified_logging.py       # Unified TensorBoard logging
│   ├── unified_visualization.py # Metric visualization
│   ├── quality.py               # MS-SSIM, PSNR, LPIPS
│   ├── generation.py            # KID, CMMD, FID (2D)
│   ├── generation_3d.py         # Generation metrics for 3D
│   ├── generation_computation.py # Metric computation helpers
│   ├── generation_sampling.py   # Sample generation for metrics
│   ├── feature_extractors.py    # ResNet50, BiomedCLIP extractors
│   ├── figures.py               # Reconstruction figures
│   ├── constants.py             # RANO-BM tumor size thresholds
│   ├── brain_mask.py            # Brain mask utilities
│   ├── codebook_manager.py      # VQ-VAE codebook tracking
│   ├── dispatch.py              # Metric dispatch by trainer type
│   ├── metric_computation.py    # Metric computation utilities
│   ├── metric_logger.py         # Metric logging
│   ├── regional_manager.py      # Regional metrics management
│   ├── sampler.py               # Metric sampling
│   ├── visualization_constants.py # Viz constants
│   ├── regional/               # Regional metrics (per-tumor)
│   │   ├── base.py             # Regional base
│   │   ├── tracker.py          # Image regional tracker
│   │   └── tracker_seg.py      # Seg regional tracker (Dice/IoU by size)
│   └── tracking/               # Training diagnostics
│       ├── codebook.py         # VQ-VAE codebook health
│       ├── flops.py            # FLOPs measurement
│       ├── gradient.py         # Gradient norm tracking
│       └── worst_batch.py      # Worst batch capture
├── models/                      # Model architectures
│   ├── factory.py               # Model factory (UNet, DiT, HDiT, UViT)
│   ├── dit.py                   # DiT (Scalable Interpolant Transformer)
│   ├── dit_blocks.py            # Transformer blocks with adaLN-Zero
│   ├── hdit.py                  # HDiT (Hierarchical Diffusion Transformer)
│   ├── uvit.py                  # UViT (ViT with Skip Connections)
│   ├── embeddings.py            # Patch/timestep/conditioning embeddings
│   ├── controlnet.py            # ControlNet for latent diffusion
│   ├── autoencoder_dc_3d.py     # 3D DC-AE architecture
│   ├── dcae_3d_ops.py           # 3D DC-AE operations
│   ├── dcae_adaptive_layers.py  # DC-AE adaptive resolution layers
│   ├── dcae_structured.py       # DC-AE 1.5 structured latent
│   ├── haar_wavelet_3d.py       # 3D Haar wavelet transform
│   └── wrappers/               # Model conditioning wrappers
│       ├── base_embed.py       # Base embedding (zero-init MLP)
│       ├── combined_embed.py   # Combined omega + mode wrapper
│       ├── device_utils.py     # Device management
│       ├── mode_embed.py       # Mode embedding for multi-modality
│       └── size_bin_embed.py   # Size bin embedding for seg_conditioned
├── pipeline/                    # Trainers and training infrastructure
│   ├── base_trainer.py          # BaseTrainer (distributed, TensorBoard, checkpoints)
│   ├── base_config.py           # BaseTrainingConfig dataclass
│   ├── diffusion_trainer_base.py # DiffusionTrainerBase (abstract)
│   ├── diffusion_config.py      # DiffusionTrainerConfig dataclass
│   ├── diffusion_init_helpers.py # Diffusion initialization helpers
│   ├── diffusion_model_setup.py # Model setup for diffusion
│   ├── trainer.py               # DiffusionTrainer (unified 2D/3D via spatial_dims)
│   ├── compression_trainer.py   # BaseCompressionTrainer
│   ├── compression_arch_config.py # Compression architecture configs
│   ├── compression_checkpointing.py # Compression checkpoint logic
│   ├── compression_metrics.py   # Compression-specific metrics
│   ├── compression_training.py  # Compression training loop
│   ├── compression_validation.py # Compression validation
│   ├── vae_trainer.py           # VAETrainer (unified 2D/3D via .create_3d())
│   ├── vqvae_trainer.py         # VQVAETrainer (unified 2D/3D via .create_3d())
│   ├── dcae_trainer.py          # DCAETrainer (unified 2D/3D via .create_3d())
│   ├── checkpoint_manager.py    # Checkpoint save/load management
│   ├── checkpointing.py         # Checkpoint utilities
│   ├── compile_manager.py       # torch.compile management
│   ├── discriminator_manager.py # GAN discriminator lifecycle
│   ├── gen_metrics_manager.py   # Generation metrics management
│   ├── evaluation.py            # Legacy test evaluation
│   ├── validation.py            # Legacy validation
│   ├── visualization.py         # Validation visualization
│   ├── loss_helper.py           # Loss computation helpers
│   ├── losses.py                # Pipeline-level loss wrappers
│   ├── profiling.py             # Training profiling
│   ├── results.py               # TrainingStepResult dataclass
│   ├── training_tricks.py       # Training trick configs
│   ├── utils.py                 # Shared utilities
│   └── optimizers/
│       └── sam.py               # SAM/ASAM optimizer
└── scripts/                     # Training entry points
    ├── train.py                 # Unified diffusion training (2D/3D via model.spatial_dims)
    ├── train_compression.py     # Unified compression training (VAE/VQ-VAE/DC-AE, 2D/3D)
    ├── train_segmentation.py    # Downstream segmentation training
    ├── train_diffrs_discriminator.py  # DiffRS rejection sampling discriminator training
    ├── generate.py              # Generation/inference script
    ├── encode_latents.py        # Pre-encode datasets for latent diffusion
    ├── eval_ode_solvers.py      # Evaluate ODE solvers for RFlow generation
    ├── find_optimal_steps.py    # Golden-section search for optimal Euler step count
    ├── measure_latent_std.py    # Measure latent space statistics
    ├── lr_finder.py             # Learning rate finder
    ├── common.py                # Shared utilities
    └── visualize_augmentations.py  # Debug augmentation pipelines

configs/
├── diffusion.yaml               # 2D diffusion training config
├── diffusion_3d.yaml            # 3D diffusion config (deprecated, use model.spatial_dims=3)
├── vae.yaml                     # VAE training config
├── vae_3d.yaml                  # 3D VAE training config
├── vqvae.yaml                   # VQ-VAE training config
├── vqvae_3d.yaml                # 3D VQ-VAE training config
├── dcae.yaml                    # DC-AE 2D training config
├── dcae_3d.yaml                 # DC-AE 3D training config
├── segmentation.yaml            # Downstream segmentation config
├── generate.yaml                # Generation config
├── lr_finder.yaml               # LR finder config
├── visualize_augmentations.yaml # Augmentation visualization config
├── controlnet/default.yaml      # ControlNet conditioning
├── latent/default.yaml          # Latent diffusion settings
├── space_to_depth/default.yaml  # 3D space-to-depth rearrangement
├── wavelet/default.yaml         # 3D Haar wavelet decomposition
├── model/default.yaml           # UNet architecture (2D)
├── model/default_3d.yaml        # UNet architecture (3D)
├── model/default_3d_5lvl.yaml   # UNet 3D (5-level variant)
├── model/dit.yaml               # DiT architecture (2D)
├── model/dit_3d.yaml            # DiT architecture (3D, S/B/L/XL variants)
├── model/hdit_3d.yaml           # HDiT architecture (3D, hierarchical transformer)
├── model/uvit_3d.yaml           # UViT architecture (3D, skip-connection ViT)
├── model/wdm_3d.yaml            # WDM UNet (3D wavelet diffusion)
├── model/smoke_test.yaml        # Minimal model for fast testing
├── vae/default.yaml             # VAE architecture
├── vae_3d/default.yaml          # 3D VAE architecture
├── vqvae/default.yaml           # VQ-VAE architecture
├── vqvae_3d/default.yaml        # 3D VQ-VAE architecture
├── dcae/default.yaml            # DC-AE 2D architecture
├── dcae/f32.yaml                # DC-AE f32c32 (32× compression)
├── dcae/f64.yaml                # DC-AE f64c128 (64× compression)
├── dcae/f128.yaml               # DC-AE f128c512 (128× compression)
├── dcae_3d/default.yaml         # DC-AE 3D architecture
├── dcae_3d/f32_d4.yaml          # DC-AE 3D f32 with 4x depth compression
├── volume/default.yaml          # 3D volume dimensions (256×256×160)
├── paths/{local,cluster}.yaml
├── strategy/{ddpm,rflow}.yaml
├── mode/{seg,bravo,bravo_seg_cond,dual,multi,multi_modality,...}.yaml
├── mode/{seg_compression,seg_conditioned,seg_conditioned_3d,...}.yaml
├── mode/{seg_conditioned_input,seg_conditioned_input_3d}.yaml
└── training/{default,fast_debug}.yaml
```

---

## Training Scripts

| Script | Purpose | Config File | Usage |
|--------|---------|-------------|-------|
| `train.py` | Train diffusion model (2D/3D) | `diffusion.yaml` | `model.spatial_dims=3` for 3D |
| `train_compression.py` | Train compression model (VAE/VQ-VAE/DC-AE, 2D/3D) | Varies | `--config-name=vae`, `vae_3d`, `dcae`, etc. |
| `train_segmentation.py` | Train downstream segmentation model | `segmentation.yaml` | `scenario=baseline/synthetic/mixed` |
| `train_diffrs_discriminator.py` | Train DiffRS rejection sampling discriminator | `diffrs.yaml` | Post-hoc quality filtering |
| `generate.py` | Generate synthetic images | `generate.yaml` | |
| `encode_latents.py` | Pre-encode dataset to latent space | N/A | |
| `eval_ode_solvers.py` | Evaluate ODE solvers for RFlow generation | N/A | Compare solver quality/speed |
| `find_optimal_steps.py` | Find optimal Euler step count | N/A | Golden-section search |
| `measure_latent_std.py` | Measure latent space statistics | N/A | For scale_factor calibration |
| `lr_finder.py` | Find optimal learning rate | `lr_finder.yaml` | |
| `common.py` | Shared utilities (get_image_keys, run_test_evaluation, etc.) | N/A | Not a script |
| `visualize_augmentations.py` | Debug augmentation pipelines | `visualize_augmentations.yaml` | |

### Compression Training Config Names

| Config Name | Model Type | Spatial Dims |
|-------------|------------|--------------|
| `vae` | VAE | 2D |
| `vae_3d` | VAE | 3D |
| `vqvae` | VQ-VAE | 2D |
| `vqvae_3d` | VQ-VAE | 3D |
| `dcae` | DC-AE | 2D |
| `dcae_3d` | DC-AE | 3D |

---

## Trainers

| Trainer | Model | Loss | Purpose |
|---------|-------|------|---------|
| `DiffusionTrainer` | UNet, DiT, HDiT, or UViT | MSE + Perceptual | 2D/3D image generation (via spatial_dims) |
| `VAETrainer` | AutoencoderKL + PatchDiscriminator | L1 + Perceptual + KL + Adversarial | 2D/3D compression (via .create_3d()) |
| `VQVAETrainer` | VQVAE + PatchDiscriminator | L1 + Perceptual + VQ + Adversarial | 2D/3D discrete latent compression |
| `DCAETrainer` | AutoencoderDC | L1 + Perceptual + optional GAN | High-compression 2D/3D (32×/64×) |
| `DCAETrainer (seg_mode)` | AutoencoderDC | BCE + Dice + Boundary | Seg mask compression |

**Note**: 3D variants support `disable_gan=true` to skip discriminator creation entirely (saves ~15GB VRAM).

**Note**: DCAETrainer with `seg_mode=true` uses segmentation-specific losses (BCE + Dice + Boundary) and metrics (Dice/IoU instead of PSNR/LPIPS/MS-SSIM).

### Trainer Class Hierarchy

```
BaseTrainer (base_trainer.py)
├── DiffusionTrainerBase (diffusion_trainer_base.py) - abstract diffusion base
│   └── DiffusionTrainer (trainer.py) - unified 2D/3D via spatial_dims
├── BaseCompressionTrainer (compression_trainer.py)
│   ├── VAETrainer (vae_trainer.py) - unified 2D/3D via .create_3d() factory
│   ├── VQVAETrainer (vqvae_trainer.py) - unified 2D/3D via .create_3d() factory
│   └── DCAETrainer (dcae_trainer.py) - unified 2D/3D via .create_3d() factory
└── SegmentationTrainer (downstream/segmentation_trainer.py) - unified 2D/3D
```

### 2D vs 3D Training

**Diffusion**: Use `model.spatial_dims=3` parameter
```bash
python -m medgen.scripts.train mode=bravo model.spatial_dims=3 strategy=rflow
```

**Compression**: Use `--config-name` to select 3D config
```bash
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality
```

**BaseTrainer Abstract Methods** (all trainers must implement):
- `setup_model()` - Initialize model, optimizer, scheduler
- `train_step(batch) -> TrainingStepResult` - Single training step
- `train_epoch(loader, epoch) -> Dict[str, float]` - Train for one epoch
- `compute_validation_losses(epoch, log_figures) -> Dict[str, float]` - Validation metrics
- `_save_checkpoint(epoch, name)` - Save model checkpoint
- `_get_trainer_type() -> str` - Return trainer type for metadata (e.g., 'vae', 'diffusion')

**BaseTrainer Template Methods**:
- `_save_metadata()` - Saves config.yaml + metadata.json (uses `_get_trainer_type()` and `_get_metadata_extra()`)
- `train()` - Main training loop with hooks

**BaseTrainer Optional Hooks**:
- `_get_metadata_extra() -> Dict` - Add trainer-specific fields to metadata.json (default: empty)
- `_on_training_start()`, `_on_epoch_start()`, `_on_epoch_end()`, `_on_training_end()`

**TrainingStepResult** (`pipeline/results.py`):
```python
@dataclass
class TrainingStepResult:
    total_loss: float
    reconstruction_loss: float = 0.0
    perceptual_loss: float = 0.0
    regularization_loss: float = 0.0  # KL or VQ loss
    adversarial_loss: float = 0.0
    discriminator_loss: float = 0.0
    mse_loss: float = 0.0

    def to_legacy_dict(self, reg_key: str = 'kl') -> Dict[str, float]:
        """Convert to legacy format for train_epoch averaging."""
```

**Shared Methods in BaseCompressionTrainer** (2D):
- `_prepare_batch()` - Batch preparation with MetaTensor handling
- `evaluate_test_set()` - Full test evaluation via `TestEvaluator`
- `_get_config_value()` - Generic config extraction from model-specific sections
- GAN training logic, EMA, perceptual loss, gradient tracking

**Subclass-Specific Overrides**:
- `_test_forward(model, images)` - Model-specific forward pass for test evaluation
- `_forward_for_validation(images)` - Model-specific forward for validation
- `_get_model_config()` - Returns model-specific config section

---

## Model Architectures

### UNet (Default)

MONAI's `DiffusionModelUNet` - convolutional encoder-decoder with attention.

```yaml
model:
  type: unet
  channels: [128, 256, 256]
  attention_levels: [false, true, true]
  num_res_blocks: 1
```

### DiT (Scalable Interpolant Transformer)

Vision transformer designed for flow matching / diffusion. Better scaling than UNet at larger sizes.

**Note**: In our experiments on medical imaging (~14K samples), DiT showed no improvement over UNet. UNet's convolutional inductive bias is better suited for small medical datasets where transformers cannot learn spatial priors from data alone.

```yaml
model:
  type: dit
  variant: B          # S (33M), B (130M), L (458M), XL (675M)
  patch_size: 2       # 1, 2, 4, or 8
  conditioning: concat  # concat or cross_attn
  qk_norm: true       # QK-normalization for stability
```

| Variant | Params | Hidden | Depth | Heads | Use Case |
|---------|--------|--------|-------|-------|----------|
| S | 33M | 384 | 12 | 6 | Latent space or high-token-count pixel space |
| B | 130M | 768 | 12 | 12 | Standard training |
| L | 458M | 1024 | 24 | 16 | Latent space only |
| XL | 675M | 1152 | 28 | 16 | Latent space only |

**3D DiT token count**: `tokens = (D/patch) × (H/patch) × (W/patch)`. Memory is O(n²) in tokens (attention), so patch size and spatial resolution are the dominant factors.

**3D DiT VRAM profiling**: Use `misc/profile_dit_memory.py` to sweep all variants × resolutions × patch sizes on GPU.

Reference: [arxiv.org/abs/2401.08740](https://arxiv.org/abs/2401.08740)

### HDiT (Hierarchical Diffusion Transformer)

U-shaped transformer with token merging/splitting for multi-resolution processing. Uses the same adaLN-Zero DiTBlocks as standard DiT but processes tokens hierarchically — most compute happens at reduced sequence lengths.

**Key advantage**: Enables `patch_size=4` for 3D volumes at manageable cost (vs DiT which needs `patch_size=8+` for 3D).

```yaml
model:
  type: hdit
  variant: S          # S/B/L/XL (same DiT variant sizes)
  patch_size: 4       # Fine patches (HDiT's sweet spot)
  level_depths: [2, 4, 6, 4, 2]  # Blocks per level (must be odd-length)
  qk_norm: true
  conditioning: concat
```

**Architecture**:
1. Patchify at `patch_size` → full-resolution tokens
2. Encoder: Process blocks, then merge tokens (2x2x2 → 8x reduction per level)
3. Bottleneck: Process at lowest resolution
4. Decoder: Split tokens back, add skip connections from encoder
5. Unpatchify to output

**Token count example** (128x128x160, patch=4):
- Level 0: 40,960 tokens (4 blocks)
- Level 1: 5,120 tokens (8 blocks)
- Bottleneck: 640 tokens (6 blocks)

| Variant | Hidden | Heads | Config |
|---------|--------|-------|--------|
| S | 384 | 6 | `model=hdit_3d model.variant=S` |
| B | 768 | 12 | `model=hdit_3d model.variant=B` |
| L | 1024 | 16 | `model=hdit_3d model.variant=L` |
| XL | 1152 | 16 | `model=hdit_3d model.variant=XL` |

**VRAM profiling**: Use `misc/profiling/profile_hdit_uvit_memory.py`. See `docs/profiling_results.md` for results.

Inspired by U-DiT (Tian et al., NeurIPS 2024) but adapted for 3D patchified sequences.

### UViT (Vision Transformer with Skip Connections)

Token-based conditioning ViT with skip connections between encoder and decoder halves. Key difference from DiT: no adaLN modulation — timestep is prepended as a token and conditioning flows through self-attention.

```yaml
model:
  type: uvit
  variant: S          # S (512d/13L), S-Deep (512d/17L), M (768d/17L), L (1024d/21L)
  patch_size: 8
  conditioning: concat
  qk_norm: false       # Paper default
```

**Key differences from DiT**:
- Token-based conditioning (timestep prepended as token, not adaLN modulation)
- Standard Pre-LN ViT blocks (no adaLN-Zero)
- Skip connections between encoder and decoder halves
- Depth must be odd (encoder + 1 mid + decoder)
- `qkv_bias=False`, no qk_norm by default
- Final conv layer to prevent patch-boundary artifacts

| Variant | Params | Hidden | Depth | Heads |
|---------|--------|--------|-------|-------|
| S | ~44M | 512 | 13 | 8 |
| S-Deep | ~58M | 512 | 17 | 8 |
| M | ~131M | 768 | 17 | 12 |
| L | ~304M | 1024 | 21 | 16 |

Reference: [arxiv.org/abs/2209.12152](https://arxiv.org/abs/2209.12152) (Bao et al., CVPR 2023)

### VQ-VAE (Vector Quantized VAE)

Discrete latent space using vector quantization instead of KL regularization.
Better for discrete latent diffusion.

```yaml
vqvae:
  num_embeddings: 512      # Codebook size
  embedding_dim: 3         # Latent channels
  commitment_cost: 0.25    # VQ commitment loss weight
  perceptual_weight: 0.002 # Perceptual loss weight
  adv_weight: 0.005        # GAN loss weight
  disable_gan: false       # Set true for pure VQ-VAE
```

Reference: [MONAI Generative](https://arxiv.org/abs/2307.15208)

### 3D VAE

Volumetric autoencoder for 3D medical imaging (256×256×160 → 32×32×20 latent).

**Memory optimizations:**
- `training.gradient_checkpointing=true` - Trades compute for memory (~50% reduction)
- `vae_3d.disable_gan=true` - Skips discriminator creation entirely (~15GB savings)
- No attention layers (O(n²) too expensive for 3D)
- 2.5D perceptual loss (sample 25% slices)

```yaml
vae_3d:
  latent_channels: 3
  channels: [32, 64, 128, 128]
  use_2_5d_perceptual: true
  disable_gan: false       # Set true to skip discriminator (saves ~15GB VRAM)
```

### 3D VQ-VAE

Volumetric VQ-VAE with discrete codebook for 3D volumes.

```yaml
vqvae_3d:
  num_embeddings: 512
  embedding_dim: 3
  commitment_cost: 0.25
  disable_gan: false       # Set true for pure VQ-VAE
```

### DC-AE (Deep Compression Autoencoder)

High-compression 2D autoencoder from MIT HAN Lab for extreme spatial compression (32×/64×).
Uses deterministic encoder (no KL divergence) with EfficientViT blocks and pixel shuffle/unshuffle.

**Key differences from VAE:**
- Deterministic encoder (no stochastic sampling, no KL loss)
- Much higher compression: 32× (8×8 spatial) or 64× (4×4 spatial)
- EfficientViT blocks with multi-scale linear attention
- Residual autoencoding with space-to-channel transforms
- Supports pretrained ImageNet models from HuggingFace

**Compression variants:**

| Variant | Input | Latent | Spatial Compression | Scaling Factor |
|---------|-------|--------|---------------------|----------------|
| f32c32 | 256×256×1 | 8×8×32 | 32× | 0.3189 |
| f64c128 | 256×256×1 | 4×4×128 | 64× | 0.2889 |
| f128c512 | 256×256×1 | 2×2×512 | 128× | 0.25 |

**For 150-slice volume (latent diffusion):**
- Per-slice latent: `[B, 32, 8, 8]` (f32) or `[B, 128, 4, 4]` (f64) or `[B, 512, 2, 2]` (f128)
- Stacked for 3D diffusion: `[B, 32, 150, 8, 8]` or `[B, 128, 150, 4, 4]` or `[B, 512, 150, 2, 2]`

```yaml
# configs/dcae/f32.yaml (default)
latent_channels: 32
compression_ratio: 32
scaling_factor: 0.3189

# configs/dcae/f64.yaml (higher compression)
latent_channels: 128
compression_ratio: 64
scaling_factor: 0.2889

# configs/dcae/f128.yaml (extreme compression)
latent_channels: 512
compression_ratio: 128
scaling_factor: 0.25
```

**3-phase training (from paper):**
1. **Phase 1**: L1 + Perceptual loss (no GAN) - main training phase
2. **Phase 2**: High-res adaptation (skip for 256×256)
3. **Phase 3**: GAN fine-tuning (optional, freeze encoder, train decoder head)

```yaml
training:
  phase: 1          # 1=no GAN (default), 3=with GAN
dcae:
  pretrained: null  # or "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"
  adv_weight: 0.0   # Set >0 for Phase 3 GAN training
```

Reference: [arxiv.org/abs/2410.10733](https://arxiv.org/abs/2410.10733)

### DC-AE 1.5: Structured Latent Space (ICCV 2025)

DC-AE 1.5 introduces **structured latent space** via channel masking during autoencoder training, enabling faster diffusion convergence.

**Key findings from paper**:
- 6× faster convergence on UViT-H (gFID 26.44 → 17.31)
- MUST be used with **augmented diffusion training** (both techniques required together)
- **NOT recommended for small channel counts** (c=32). Use only for c≥64 (f64, f128)

**How it works**:
1. During AE training, randomly mask channels [min_channels : latent_channels]
2. Creates progression: [16, 20, 24, ..., c] where c is total latent channels
3. Early channels encode structure, later channels encode details
4. Diffusion model trained with matching augmentation (same channel masking)

**Configuration**:
```yaml
# In dcae config (use with f64 or f128):
dcae:
  structured_latent:
    enabled: true
    min_channels: 16           # Minimum channels to keep
    channel_step: 4            # Step between options

# For diffusion training (MUST enable both):
training:
  augmented_diffusion:
    enabled: true
    min_channels: 16           # Match AE settings
    channel_step: 4
```

**Usage**:
```bash
# Step 1: Train DC-AE with structured latent space (f64 recommended)
python -m medgen.scripts.train_dcae dcae=f64 \
    dcae.structured_latent.enabled=true

# Step 2: Train diffusion with augmented diffusion training
python -m medgen.scripts.train mode=bravo strategy=rflow \
    vae_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    training.augmented_diffusion.enabled=true
```

**Important**: Structured latent space requires latent diffusion (not pixel space). The `augmented_diffusion.enabled` setting is ignored in pixel space.

---

## SAM Optimizer

**SAM** (Sharpness-Aware Minimization): Seeks flat minima for better generalization.
- Requires 2 forward-backward passes per step (~2x compute)
- Useful for combating overfitting on small datasets

```yaml
training:
  sam:
    enabled: true
    rho: 0.05          # Perturbation radius (0.01-0.1)
    adaptive: false    # Use ASAM (weight-scale invariant)
```

References:
- SAM: [arxiv.org/abs/2010.01412](https://arxiv.org/abs/2010.01412)
- ASAM: [arxiv.org/abs/2102.11600](https://arxiv.org/abs/2102.11600)

---

## Clean Regularization Techniques (Diffusion)

These techniques provide regularization WITHOUT leaking augmentation patterns into generated samples. Unlike standard data augmentation which transforms clean images and teaches the model to generate augmented versions, these methods regularize training while preserving the output distribution.

**Key insight**: ScoreAug worked because it provides regularization without affecting what the model learns to output. These techniques follow the same principle.

### Techniques Overview

| Technique | Config | How It Works | Implementation |
|-----------|--------|--------------|----------------|
| Constant LR | `scheduler=constant` | Skip cosine decay, maintain LR after warmup | `core/schedulers.py` |
| Gradient Noise | `gradient_noise.enabled` | Add decaying Gaussian noise to gradients | `trainer.py:_add_gradient_noise()` |
| Curriculum | `curriculum.enabled` | Progressive timestep range expansion | `trainer.py:_get_curriculum_range()`, `strategies.py` |
| Timestep Jitter | `timestep_jitter.enabled` | Add noise to sampled timesteps | `trainer.py:_apply_timestep_jitter()` |
| Noise Augmentation | `noise_augmentation.enabled` | Perturb noise vector before adding to image | `trainer.py:_apply_noise_augmentation()` |
| Feature Perturbation | `feature_perturbation.enabled` | Add noise to intermediate features via hooks | `trainer.py:_setup_feature_perturbation()` |
| Self-Conditioning | `self_conditioning.enabled` | Consistency loss between two forward passes | `trainer.py:_compute_self_conditioning_loss()` |

### Implementation Details

**Constant LR Scheduler** (`src/medgen/core/schedulers.py`):
```python
def create_warmup_constant_scheduler(
    optimizer, warmup_epochs, total_epochs, start_factor=0.1
) -> SequentialLR:
    """Linear warmup then constant LR (no cosine decay)."""
```

**Gradient Noise Injection** (`src/medgen/pipeline/trainer.py`):
- Noise decays as: `sigma / (1 + step)^decay`
- Injected AFTER gradient clipping, BEFORE optimizer step
- Reference: Neelakantan et al., 2015

**Curriculum Timestep Scheduling** (`src/medgen/pipeline/trainer.py`, `strategies.py`):
- Linearly interpolates timestep range from `[min_t_start, max_t_start]` to `[min_t_end, max_t_end]`
- Warmup epochs control progression speed
- Easy samples (low noise) → hard samples (high noise)

**Timestep Jitter**:
- Adds Gaussian noise (std configurable) to sampled timesteps
- Increases noise-level diversity without changing output distribution
- Applied in normalized [0, 1] range, then scaled back to discrete timesteps

**Noise Augmentation**:
- Perturbs noise vector: `noise = noise + randn * std`
- Renormalizes to maintain variance: `noise = noise / noise.std() * original_std`
- Increases noise diversity without changing what model learns to output

**Feature Perturbation**:
- Uses forward hooks to inject Gaussian noise at specified layers
- Configurable layers: `"encoder"`, `"mid"`, `"decoder"`, or list
- Like continuous dropout but applied to activations

**Self-Conditioning via Consistency**:
- With probability `prob`, runs model twice per batch
- First pass: no gradient, get prediction P1
- Second pass: compute consistency loss `MSE(prediction, P1)`
- Total loss = main_loss + consistency_weight × consistency_loss
- Works without model architecture changes

### VAETrainer.train() Signature
```python
def train(
    train_loader, train_dataset, val_loader=None,
    start_epoch=0, max_epochs=None, early_stop_fn=None
) -> int  # Returns last epoch number
```
- `max_epochs`: Override total epochs
- `early_stop_fn`: Callback `(epoch, val_loss) -> bool` for early stopping

---

## Strategies

| Strategy | Predicts | Target | Timestep Sampling |
|----------|----------|--------|-------------------|
| DDPM | Noise (epsilon) | `noise` | Uniform random (discrete) |
| RFlow | Velocity | `images - noise` | Logit-normal (continuous, biased to middle) |

**RFlow Defaults** (configurable in `configs/strategy/rflow.yaml`):
- `use_discrete_timesteps: false` - Continuous timesteps (floats, not integers)
- `sample_method: logit-normal` - Biases sampling toward middle timesteps
- `use_timestep_transform: true` - Resolution-based timestep adjustment
- `num_train_timesteps: 1000` - Total timesteps for training

**RFlow Timestep Convention**:
- `t=0` is **clean** (original image)
- `t=1` is **noise** (pure Gaussian noise)
- Interpolation: `x_t = (1 - t) * x_0 + t * noise`
- Velocity target: `v = x_0 - noise` (points from noise toward clean data)
- Inference: Goes from `t=1000` → `t=0` (noise to clean)
- Euler step: `x_{t-dt} = x_t + dt * v` (ADDITION - velocity points toward data)

---

## Modes

### Diffusion Modes

| Mode | in_channels | out_channels | Conditioning | Data Shape |
|------|-------------|--------------|--------------|------------|
| `seg` | 1 | 1 | None | `[B, 1, H, W]` |
| `bravo` | 2 | 1 | Seg mask | `[B, 2, H, W]` = [bravo, seg] |
| `dual` | 3 | 2 | Seg mask | `[B, 3, H, W]` = [t1_pre, t1_gd, seg] |
| `multi` | 2 | 1 | Seg mask + mode_id | `[B, 2, H, W]` = [image, seg] + mode embedding |
| `bravo_seg_cond` | 8 | 4 | Latent seg mask | `[B, 8, ...]` = [bravo_latent(4), seg_latent(4)] |
| `seg_conditioned` | 1 + size_bins | 1 | Size bins (FiLM) | `[B, 1, H, W]` seg + size bin embedding |
| `seg_conditioned_input` | 1 + 7 bin maps | 1 | Size bins (channel concat) | `[B, 8, H, W]` seg + 7 binary bin maps |
| `seg_conditioned_3d` | 1 + size_bins | 1 | Size bins (FiLM) | `[B, 1, D, H, W]` seg + size bin embedding |

**Note**: `bravo_seg_cond` is for latent diffusion only — generates BRAVO latents conditioned on VQ-VAE-encoded seg masks. Requires `latent.enabled=true`.

**Note**: `seg_conditioned` modes generate segmentation masks conditioned on tumor size distribution (size_bins). The conditioning is a 1D vector encoding expected tumor sizes via RANO-BM thresholds.

### VAE Modes (DIFFERENT - no seg conditioning)

| Mode | in_channels | Description |
|------|-------------|-------------|
| `bravo` | 1 | Single bravo image |
| `dual` | 2 | T1 pre + T1 post (NO seg) |
| `seg` | 1 | Segmentation mask only |
| `multi_modality` | 1 | Pools all modalities (bravo, flair, t1_pre, t1_gd) |
| `seg_compression` | 1 | Seg mask compression (DC-AE only, BCE+Dice+Boundary loss) |

**CRITICAL**: `train_compression.py` overrides `mode.in_channels`:
- dual mode: 2 channels (t1_pre + t1_gd, NO seg)
- other modes: 1 channel

---

## Spaces

| Space | scale_factor | Purpose |
|-------|--------------|---------|
| `PixelSpace` | 1 | Direct pixel diffusion (default) |
| `SpaceToDepthSpace` | 2 | 2D pixel rearrangement (no learned transform) |
| `WaveletSpace` | 2 | 3D Haar wavelet decomposition (8 subbands, per-subband normalized) |
| `LatentSpace` | 4-128 | Compressed diffusion via VAE/VQ-VAE/DC-AE (auto-detected from checkpoint) |

**LatentSpace scale factors** depend on the compression model:

| Compression | scale_factor | Latent Size (from 256×256) |
|-------------|-------------|---------------------------|
| VQ-VAE 4x | 4 | 64×64 |
| VAE 8x | 8 | 32×32 |
| VQ-VAE 8x | 8 | 32×32 |
| DC-AE 32x | 32 | 8×8 |
| DC-AE 64x | 64 | 4×4 |
| DC-AE 128x | 128 | 2×2 |

**Note**: `scale_factor` is auto-detected from the compression checkpoint at runtime and written back to the Hydra config so the model factory can derive correct spatial dimensions.

---

## Training Pipeline Order

```
1. [Optional] Find optimal LR:
   python -m medgen.scripts.lr_finder mode=dual model_type=vae

2. [Optional] Train VAE (for latent diffusion):
   python -m medgen.scripts.train_compression --config-name=vae mode=dual

3. Train Diffusion:
   python -m medgen.scripts.train mode=dual strategy=rflow

4. Generate images:
   python -m medgen.scripts.generate checkpoint_path=dual.pt mode=dual strategy=rflow
```

---

## Loss Functions

### Diffusion Training
```python
total_loss = mse_loss + perceptual_weight * perceptual_loss
# Default perceptual_weight: 0.0 (disabled by default - saves ~200MB GPU memory)
# To enable: training.perceptual_weight=0.001
# NOTE: Seg mode auto-disables perceptual loss (pretrained features don't apply to binary masks)
```

### VAE Training
```python
g_loss = L1_loss + 0.001 * perceptual + 1e-6 * KL + 0.01 * adversarial
```

---

## Learning Rates

### Diffusion
- LR: `1e-4`, Scheduler: Warmup (5 epochs) + Cosine Annealing
- `eta_min`: `1e-6` (minimum LR for cosine annealing, configurable)

### VAE
- Generator LR: `1e-4`, Discriminator LR: `5e-4`
- Same scheduler pattern

---

## Logging Configuration

All logging options are configured under `training.logging`:

```yaml
logging:
  # Training dynamics
  grad_norm: true              # Track gradient norm (catches instability)
  timestep_losses: true        # Loss by diffusion timestep (10 bins)
  regional_losses: true        # Loss by tumor vs background region
  timestep_region_losses: true # 2D heatmap: timestep x region

  # Validation metrics
  msssim: true                 # Multi-Scale Structural Similarity
  psnr: true                   # Peak signal-to-noise ratio
  lpips: true                  # Learned Perceptual Image Patch Similarity
  boundary_sharpness: true     # Edge quality in tumor regions

  # Visualization
  intermediate_steps: true     # Denoising trajectory
  worst_batch: true            # Highest loss batch (diffusion only)
  num_intermediate_steps: 5    # Steps in trajectory

  # Performance
  flops: true                  # Model FLOPs measurement
```

### Regional Loss Tracking

Tracks MSE/L1 separately for tumor vs background regions using per-tumor analysis:
- Uses connected components to identify individual tumors
- Measures tumor size via Feret diameter (longest edge-to-edge distance)
- Classifies tumors using RANO-BM clinical thresholds (tiny <10mm, small 10-20mm, medium 20-30mm, large >30mm)
- All metrics are pixel-weighted (larger tumors contribute proportionally more)

Logged metrics:
- `regional/tumor_loss` - Per-pixel error on tumor regions
- `regional/background_loss` - Error on background pixels
- `regional/tumor_bg_ratio` - Ratio of tumor to background error
- `regional/{tiny,small,medium,large}` - Per-pixel error by tumor size (Feret diameter)
- `training/timestep_region_heatmap` - 2D figure (timestep x region)

### Worst Batch Tracking

Shows the batch with highest loss for debugging:
- **Validation**: `Validation/worst_batch` - Logged at `figure_count` intervals
- **Test**: `test_best/worst_batch`, `test_latest/worst_batch` - Logged at end

---

## Data Augmentation

### Augmentation Types

| Type | Used By | Strategy |
|------|---------|----------|
| `diffusion` | Diffusion training | Conservative - preserves image distribution |
| `vae` | VAE training | Aggressive - learns robust features |

### Diffusion Augmentation (Conservative)

Only lossless spatial transforms. Distortions would teach model to generate distorted images.

```python
- HorizontalFlip (p=0.5)
- Rotate ±10° (p=0.5)
- Translate ±5% (p=0.5)
```

### VAE Augmentation (Aggressive)

More variety helps learn robust latent representations.

```python
# Spatial
- HorizontalFlip (p=0.5)
- Rotate ±15° (p=0.5)
- Translate ±10%, Scale 0.9-1.1x (p=0.5)

# Intensity
- GaussNoise std=0.01-0.05 (p=0.3)
- GaussianBlur kernel=3-5 (p=0.2)
- RandomBrightnessContrast ±10% (p=0.3)

# Elastic
- ElasticTransform alpha=50 (p=0.2)
```

### Batch-level Augmentations (VAE only)

Applied via collate function, disabled by default.

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| Mixup | 20% | Blend two images with beta-sampled lambda |
| CutMix | 20% | Paste rectangular region from another image |

Enable in config:
```yaml
training:
  batch_augment:
    enabled: true
    mixup_prob: 0.2
    cutmix_prob: 0.2
```

### Config Options

```yaml
training:
  augment: true              # Enable/disable augmentation
  augment_type: diffusion    # "diffusion" or "vae"
  batch_augment:             # VAE only
    enabled: false
    mixup_prob: 0.0
    cutmix_prob: 0.0
```

---

## Score Augmentation (ScoreAug)

Reference: [arxiv.org/abs/2508.07926](https://arxiv.org/abs/2508.07926)

ScoreAug applies transforms to **noisy data** (after noise addition) rather than clean data. This teaches equivariant denoising without changing the output distribution.

```
Traditional:  x → T(x) → add noise → denoise → T(x)  [learns augmented distribution]
ScoreAug:     x → add noise → T(x + noise) → denoise → T(x)  [learns equivariant denoising]
```

### Transforms

| Transform | Description | Requires Omega |
|-----------|-------------|----------------|
| Rotation | 90°, 180°, 270° rotations | Yes (noise is rotation-invariant) |
| Flip | Horizontal/vertical flip | Yes |
| Translation | ±40% X, ±20% Y shift with zero-padding | No |
| Cutout | Random square region zeroed (10-30%) | No |

**Omega Conditioning**: Required for rotation/flip because Gaussian noise is rotation-invariant - the model could "cheat" by detecting rotation from the noise pattern.

### ScoreAug Modes

| Mode | Config | Behavior |
|------|--------|----------|
| Single | `compose=false, v2_mode=false` | One transform sampled per step (per paper) |
| Compose | `compose=true` | Each transform applied with `compose_prob` |
| **v2** | `v2_mode=true` | Structured: non-destructive stack + one destructive |

### ScoreAug v2 (Structured Augmentation)

**v2 mode** separates transforms into two categories:

**Non-destructive** (can stack): rotation, flip, translation
- Each sampled independently with `nondestructive_prob`
- All selected transforms applied in sequence

**Destructive** (pick one): cutout OR fixed patterns
- Sampled with `destructive_prob`
- `cutout_vs_pattern`: Split between random cutout vs fixed patterns (0.5 = 50/50)

**Fixed patterns** (16 learnable masks via one-hot embedding):

| Category | IDs | Description |
|----------|-----|-------------|
| Checkerboard | 0-3 | 4×4/8×8 alternating grids (std/offset) |
| Grid dropout | 4-7 | 4×4 grid, 25%/50% cells dropped |
| Coarse dropout | 8-11 | 2-4 large holes (corners/edges) |
| Patch dropout | 12-15 | MAE-style 25%/50% patches dropped |

### Config Options

```yaml
training:
  score_aug:
    enabled: false             # Disabled by default
    # Non-destructive transforms
    rotation: true             # 90°, 180°, 270° rotations
    flip: true                 # Horizontal/vertical flip
    translation: false         # ±40% X, ±20% Y translation
    # Legacy destructive
    cutout: false              # Random rectangle cutout
    # Compose mode (legacy)
    compose: false             # Stack transforms independently
    compose_prob: 0.5          # Probability for each transform
    # v2 mode (structured)
    v2_mode: false             # Enable structured augmentation
    nondestructive_prob: 0.5   # Prob for each non-destructive
    destructive_prob: 0.5      # Prob of any destructive
    cutout_vs_pattern: 0.5     # Cutout vs fixed patterns split
    patterns:                  # Enable/disable pattern categories
      checkerboard: true
      grid_dropout: true
      coarse_dropout: true
      patch_dropout: true
    use_omega_conditioning: true  # Always use with ScoreAug
```

### Usage Examples

```bash
# ScoreAug with rotation only (omega required)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.use_omega_conditioning=true

# ScoreAug v2 (structured non-destructive/destructive)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.score_aug.enabled=true \
    training.score_aug.v2_mode=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.nondestructive_prob=0.5 \
    training.score_aug.destructive_prob=0.5 \
    training.score_aug.cutout_vs_pattern=0.5 \
    training.score_aug.use_omega_conditioning=true
```

### Implementation Notes

- **File**: `src/medgen/data/score_aug.py`
- **Wrapper**: `ScoreAugModelWrapper` injects omega conditioning into UNet's time embedding
- **torch.compile**: Inner UNet is compiled, wrapper (with data-dependent omega encoding) stays uncompiled
- **Perceptual loss**: Skipped for non-invertible transforms (translation, cutout) since original space can't be recovered

---

## Shifted Data Augmentation (SDA)

SDA is an **alternative** to ScoreAug that augments **clean data** (before noise addition) with a corresponding shift in noise level. This prevents augmentation pattern leakage while still providing regularization.

**Key difference from ScoreAug**:
- ScoreAug: Augments noisy data, requires omega conditioning
- SDA: Augments clean data with noise shift, no omega conditioning needed

```
ScoreAug:  x → add_noise(t) → T(noisy_x) → denoise
SDA:       x → T(x) → add_noise(t + shift) → denoise  [shifted timestep compensates]
```

### How It Works

1. With probability `prob`, apply augmentation to clean image
2. Shift the timestep by `noise_shift` to compensate for augmented input
3. Model learns from both augmented (shifted) and original paths
4. Loss weighted by `weight` for augmented path

### Config Options

```yaml
training:
  sda:
    enabled: false
    rotation: true              # 90°, 180°, 270° rotations
    flip: true                  # Horizontal/vertical flip
    noise_shift: 0.1            # Timestep shift amount (0.05-0.2 typical)
    prob: 0.5                   # Probability of using augmented path
    weight: 1.0                 # Loss weight for augmented path
```

### Usage

```bash
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.sda.enabled=true \
    training.sda.rotation=true \
    training.sda.flip=true \
    training.sda.noise_shift=0.1
```

**IMPORTANT**: Do NOT use SDA and ScoreAug together - they serve similar purposes differently

---

## Mode Embedding (Multi-Modality)

Enables a single diffusion model to generate multiple modalities (bravo, flair, t1_pre, t1_gd).

**Files:**
- `src/medgen/models/wrappers/mode_embed.py` - Mode embedding for single conditioning
- `src/medgen/models/wrappers/combined_embed.py` - Combined omega + mode + `create_conditioning_wrapper()` factory

**Mode IDs:**
| Modality | ID |
|----------|-----|
| bravo | 0 |
| flair | 1 |
| t1_pre | 2 |
| t1_gd | 3 |

**Usage:**
```yaml
mode:
  use_mode_embedding: true
```

The `ModeTimeEmbed` wrapper injects mode-specific conditioning into the UNet's time embedding, similar to omega conditioning in ScoreAug.

---

## DiffRS (Diffusion Rejection Sampling)

Post-hoc quality improvement for diffusion sampling without retraining the model. A tiny discriminator head (~500 params) is trained on top of the frozen UNet encoder to evaluate intermediate samples during generation.

**How it works**:
1. At each denoising step, the discriminator checks if the intermediate sample looks realistic for that noise level
2. Bad trajectories are rejected and retried with new noise
3. The diffusion model is never modified

**Architecture**:
- Feature extractor: Frozen UNet encoder (already trained)
- Classification head: GroupNorm → SiLU → Pool → Linear (~500 params)

**Training**:
```bash
python -m medgen.scripts.train_diffrs_discriminator \
    diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
    data_mode=bravo
```

**Config** (`configs/diffrs.yaml`):
```yaml
diffusion_checkpoint: null    # Required: path to trained diffusion model
data_mode: bravo              # Image type for real samples
num_generated_samples: 5000   # Samples for training
generation_num_steps: 25      # Steps for sample generation
num_epochs: 60
batch_size: 32
learning_rate: 3e-4
```

**Files**: `src/medgen/diffusion/diffrs.py`, `src/medgen/scripts/train_diffrs_discriminator.py`

Reference: DiffRS (ICML 2024)

---

## DataLoader Optimization

CPU augmentation runs in parallel workers. Configure for GPU utilization:

```yaml
training:
  dataloader:
    num_workers: 8           # Parallel data loading workers
    prefetch_factor: 4       # Batches to prefetch per worker
    pin_memory: true         # Faster CPU→GPU transfer
    persistent_workers: true # Avoid worker respawn overhead
```

---

## Precision Configuration (VAE only)

Pure BF16 training stores model weights in BF16 format for memory savings and preparation for NVIDIA 2:4 structured sparsity.

```yaml
training:
  precision:
    dtype: bf16          # bf16, fp16, fp32 - weight dtype when pure_weights=true
    pure_weights: false  # If true, model weights stored in low precision
```

### Precision Modes

| Config | Weights | Compute | Memory | Use Case |
|--------|---------|---------|--------|----------|
| `pure_weights=false` | FP32 | BF16 (autocast) | Baseline | Default (safe) |
| `pure_weights=true` | BF16 | BF16 | ~50% less | Memory-constrained |

### Implementation Details

- Weights converted after model creation, before DDP/compile wrapping
- Autocast continues to work as usual
- All other code (losses, metrics) unchanged
- Log message: `Converted model weights to torch.bfloat16`

### NVIDIA 2:4 Structured Sparsity (Future)

BF16 weights are compatible with 2:4 structured sparsity:
- **Pattern**: 2 non-zero weights per 4 consecutive (50% sparse)
- **Hardware**: A100, H100, RTX 30/40 (Compute 8.0+)
- **Benefit**: Up to 2x speedup for linear layers

---

## Dataloader Functions

### 2D Dataloaders (via `builder_2d.py` LoaderSpec pattern)

Most 2D dataloaders are built through the `LoaderSpec` pattern in `builder_2d.py`, which standardizes dataset creation, augmentation, and DataLoader wrapping.

| Function | File | Purpose | Augment Type |
|----------|------|---------|--------------|
| `create_dataloader()` | `single.py` | Single modality with conditioning (seg, bravo) | `diffusion` (default) |
| `create_dual_image_dataloader()` | `dual.py` | Dual images + seg conditioning | `diffusion` (default) |
| `create_vae_dataloader()` | `vae.py` | Images WITHOUT seg conditioning | `vae` (default) |
| `create_vae_validation_dataloader()` | `vae.py` | Images + seg for regional metrics | None |
| `create_multi_modality_dataloader()` | `multi_modality.py` | Mixed modalities (VAE) | `vae` (default) |
| `create_multi_diffusion_dataloader()` | `multi_diffusion.py` | Multi-modality diffusion with mode_id | `diffusion` (default) |
| `create_seg_compression_dataloader()` | `seg_compression.py` | Seg mask compression (DC-AE) | None |
| `create_latent_dataloader()` | `latent.py` | 2D latent diffusion (pre-encoded) | None |
| `create_seg_conditioned_dataloader()` | `seg_conditioned.py` | 2D seg-conditioned with size bins | None |
| `create_seg_dataloader()` | `seg.py` | Seg mask only (for seg mode diffusion) | `diffusion` (default) |

### 3D Dataloaders

| Function | File | Purpose |
|----------|------|---------|
| `create_volume_3d_dataloader()` | `volume_3d.py` | 3D volumetric compression (VAE/VQ-VAE/DC-AE) |
| `create_volume_3d_validation_dataloader()` | `volume_3d.py` | 3D compression validation with seg |

### Shared Infrastructure

| File | Purpose |
|------|---------|
| `base.py` | Base dataloader utilities and abstract interfaces |
| `builder_2d.py` | LoaderSpec pattern for standardized 2D dataloader construction |
| `common.py` | Shared config, distributed helpers, `GroupedBatchSampler`, `MODALITY_KEYS` |
| `datasets.py` | Dataset wrappers and composition utilities |
| `compression_detection.py` | Auto-detect compression model type from checkpoint |
| `unified.py` | Unified dataloader dispatch (routes mode → correct factory) |

---

## Checkpoint Formats

### Diffusion Checkpoint
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'ema_state_dict': ...,
    'epoch': int
}
```

### VAE Checkpoint
```python
{
    'model_state_dict': ...,           # AutoencoderKL
    'discriminator_state_dict': ...,   # PatchDiscriminator
    'optimizer_g_state_dict': ...,
    'optimizer_d_state_dict': ...,
    'epoch': int,
    'config': {...},
    'disc_config': {...}
}
```

---

## Key Code Locations

### Core
- `src/medgen/core/constants.py`: ModeType enum, thresholds
- `src/medgen/core/schedulers.py`: LR schedulers (cosine, warmup, plateau)

### Models
- `src/medgen/models/factory.py`: Model factory (UNet, DiT, HDiT, UViT)
- `src/medgen/models/dit.py`: DiT model (Scalable Interpolant Transformer)
- `src/medgen/models/dit_blocks.py`: Transformer blocks with adaLN-Zero
- `src/medgen/models/hdit.py`: HDiT (Hierarchical Diffusion Transformer)
- `src/medgen/models/uvit.py`: UViT (Vision Transformer with Skip Connections)
- `src/medgen/models/embeddings.py`: Patch, timestep, conditioning embeddings
- `src/medgen/models/controlnet.py`: ControlNet for latent diffusion
- `src/medgen/models/autoencoder_dc_3d.py`: 3D DC-AE architecture
- `src/medgen/models/haar_wavelet_3d.py`: 3D Haar wavelet transform
- `src/medgen/models/dcae_structured.py`: DC-AE 1.5 structured latent space
- `src/medgen/models/wrappers/`: Model conditioning wrappers (mode embed, size bin, omega, combined)

### Data
- `src/medgen/data/dataset.py`: NiFTIDataset class
- `src/medgen/data/loaders/`: Dataloader factory functions
- `src/medgen/data/loaders/common.py`: Shared DataLoader utilities (config, distributed, modality helpers)
  - `MODALITY_KEYS`: Centralized mapping (`'dual' → ['t1_pre', 't1_gd']`, etc.)
  - `get_modality_keys(modality)`: Expand composite modalities to keys
  - `GroupedBatchSampler`: Ensures homogeneous batches for mode embedding
- `src/medgen/data/loaders/builder_2d.py`: LoaderSpec pattern for standardized 2D dataloader construction
- `src/medgen/data/loaders/unified.py`: Unified dataloader dispatch (routes mode → correct factory)
- `src/medgen/data/loaders/compression_detection.py`: Auto-detect compression model type from checkpoint
- `src/medgen/data/lossless_mask_codec.py`: Lossless binary mask encoding to DC-AE latent

### Augmentation
- `src/medgen/augmentation/augmentation.py`: Standard transforms (diffusion/vae)
- `src/medgen/augmentation/score_aug.py`: ScoreAug transforms and omega conditioning
- `src/medgen/augmentation/sda.py`: Shifted Data Augmentation

### Diffusion
- `src/medgen/diffusion/strategies.py`: Strategy base class and registry
- `src/medgen/diffusion/strategy_ddpm.py`: DDPM strategy (noise prediction, discrete timesteps)
- `src/medgen/diffusion/strategy_rflow.py`: RFlow strategy (velocity prediction, continuous timesteps)
- `src/medgen/diffusion/modes.py`: Seg, Bravo, Dual, Multi, SegConditioned modes
- `src/medgen/diffusion/spaces.py`: Pixel/Latent/SpaceToDepth/Wavelet space abstraction
- `src/medgen/diffusion/diffrs.py`: DiffRS rejection sampling discriminator
- `src/medgen/diffusion/batch_data.py`: BatchData standardized batch unpacking
- `src/medgen/diffusion/conditioning.py`: ConditioningContext for diffusion conditioning
- `src/medgen/diffusion/protocols.py`: Protocol interfaces for strategies/modes
- `src/medgen/diffusion/loading.py`: Model checkpoint loading utilities

### Pipeline (Trainers)
- `src/medgen/pipeline/base_trainer.py`: BaseTrainer (distributed, TensorBoard, checkpoints)
- `src/medgen/pipeline/base_config.py`: BaseTrainingConfig dataclass
- `src/medgen/pipeline/diffusion_trainer_base.py`: DiffusionTrainerBase (abstract)
- `src/medgen/pipeline/diffusion_config.py`: DiffusionTrainerConfig dataclass
- `src/medgen/pipeline/diffusion_init_helpers.py`: Diffusion initialization helpers
- `src/medgen/pipeline/diffusion_model_setup.py`: Model setup for diffusion
- `src/medgen/pipeline/trainer.py`: DiffusionTrainer (unified 2D/3D via spatial_dims)
- `src/medgen/pipeline/compression_trainer.py`: BaseCompressionTrainer
- `src/medgen/pipeline/compression_arch_config.py`: Compression architecture configs
- `src/medgen/pipeline/vae_trainer.py`: VAETrainer (unified 2D/3D via .create_3d())
- `src/medgen/pipeline/vqvae_trainer.py`: VQVAETrainer (unified 2D/3D via .create_3d())
- `src/medgen/pipeline/dcae_trainer.py`: DCAETrainer (unified 2D/3D via .create_3d())
- `src/medgen/pipeline/checkpoint_manager.py`: Checkpoint save/load management
- `src/medgen/pipeline/compile_manager.py`: torch.compile management
- `src/medgen/pipeline/discriminator_manager.py`: GAN discriminator lifecycle
- `src/medgen/pipeline/gen_metrics_manager.py`: Generation metrics management
- `src/medgen/pipeline/training_tricks.py`: Training trick configs (gradient noise, curriculum, etc.)
- `src/medgen/pipeline/results.py`: TrainingStepResult dataclass
- `src/medgen/pipeline/loss_helper.py`: Loss computation helpers
- `src/medgen/pipeline/utils.py`: Shared utilities
- `src/medgen/pipeline/optimizers/sam.py`: SAM/ASAM optimizer wrapper

### Evaluation
- `src/medgen/evaluation/evaluation.py`: 2D test evaluation (L1, MS-SSIM, PSNR, LPIPS, FID, KID, CMMD)
- `src/medgen/evaluation/evaluation_3d.py`: 3D test evaluation
- `src/medgen/evaluation/evaluation_logging.py`: Evaluation result logging
- `src/medgen/evaluation/validation.py`: Validation runners
- `src/medgen/evaluation/visualization.py`: Validation visualization

### Downstream
- `src/medgen/downstream/segmentation_trainer.py`: SegmentationTrainer (SegResNet, per-tumor-size Dice)
- `src/medgen/downstream/data.py`: Downstream segmentation data loading

### Metrics
- `src/medgen/metrics/unified.py`: UnifiedMetrics (MANDATORY for all trainers)
- `src/medgen/metrics/quality.py`: MS-SSIM, PSNR, LPIPS
- `src/medgen/metrics/regional/`: Per-tumor regional metrics by size category
- `src/medgen/metrics/generation.py`: KID, CMMD, FID (ResNet50 + BiomedCLIP)
- `src/medgen/metrics/figures.py`: Reconstruction figures
- `src/medgen/metrics/tracking/`: Gradient, FLOPs, codebook, worst batch trackers

### Losses
- `src/medgen/losses/losses.py`: PerceptualLoss, SegmentationLoss (BCE+Dice+Boundary)
- `src/medgen/losses/regional_weighting.py`: Adaptive regional loss weighting

### Scripts
- `src/medgen/scripts/train.py`: Unified diffusion training (2D/3D)
- `src/medgen/scripts/train_compression.py`: Unified compression training (VAE/VQ-VAE/DC-AE)
- `src/medgen/scripts/train_segmentation.py`: Downstream segmentation training
- `src/medgen/scripts/train_diffrs_discriminator.py`: DiffRS rejection sampling discriminator training
- `src/medgen/scripts/generate.py`: Generation/inference
- `src/medgen/scripts/encode_latents.py`: Pre-encode datasets for latent diffusion
- `src/medgen/scripts/eval_ode_solvers.py`: Evaluate ODE solvers for RFlow generation
- `src/medgen/scripts/find_optimal_steps.py`: Golden-section search for optimal Euler step count
- `src/medgen/scripts/measure_latent_std.py`: Measure latent space statistics
- `src/medgen/scripts/lr_finder.py`: Learning rate finder
- `src/medgen/scripts/common.py`: Shared utilities

---

## Unified Metrics System

The unified metrics system (`src/medgen/metrics/unified.py`) provides consistent loss tracking across all trainer types.

> **IMPORTANT: All new trainers MUST use this system.**
> Do NOT implement custom TensorBoard logging. Use `TrainerMetricsConfig`, `LossAccumulator`, and `MetricsLogger` to ensure consistent metric names across all trainers.

### Why This Matters

- **Consistent TensorBoard tags** across all trainers (no drift)
- **Single source of truth** for metric names (`_TRAIN_LOSS_TAGS`, `_VAL_METRIC_TAGS`)
- **Less code duplication** - shared accumulation and logging logic
- **Easier comparison** between experiments using different trainers

### Adding a New Trainer

1. **Create a `TrainerMetricsConfig`** in your trainer's `__init__`:
   ```python
   from medgen.pipeline.metrics.unified import (
       TrainerMetricsConfig, LossAccumulator, MetricsLogger
   )

   # Option A: Use existing factory method
   self._metrics_config = TrainerMetricsConfig.for_vae(has_gan=True)

   # Option B: Create custom config for new trainer type
   self._metrics_config = TrainerMetricsConfig(
       mode=TrainerMode.CUSTOM,
       loss_keys={LossKey.GEN, LossKey.RECON, LossKey.PERC},
       validation_metrics={MetricKey.PSNR, MetricKey.MSSSIM},
   )
   ```

2. **Initialize accumulator and logger**:
   ```python
   self._loss_accumulator = LossAccumulator(self._metrics_config)
   self._metrics_logger = MetricsLogger(self.writer, self._metrics_config)
   ```

3. **Use in training loop**:
   ```python
   def train_epoch(self, loader, epoch):
       self._loss_accumulator.reset()
       for batch in loader:
           result = self.train_step(batch)
           self._loss_accumulator.update(result.to_dict())

       avg_losses = self._loss_accumulator.compute()
       self._metrics_logger.log_training(epoch, avg_losses)
   ```

4. **If adding new loss types**, extend `LossKey` and `_TRAIN_LOSS_TAGS` in `unified.py`.

### Components

**TrainerMode Enum**: Defines trainer types
- `VAE`: KL regularization
- `VQVAE`: VQ loss
- `DCAE`: No regularization (deterministic)
- `SEG`: Segmentation mode (BCE + Dice + Boundary loss)
- `DIFFUSION`: Diffusion model (MSE noise prediction)

**LossKey**: Canonical loss key names
- Generator: `gen`, `recon`, `perc`
- Regularization: `kl`, `vq`
- GAN: `disc`, `adv`
- Segmentation: `bce`, `dice`, `boundary`
- Diffusion: `mse`, `total`

**MetricKey**: Canonical validation metric names
- Image quality: `psnr`, `lpips`, `msssim`, `msssim_3d`, `l1`
- Segmentation: `dice_score`, `iou`

**Classes**:
- `LossAccumulator`: Unified epoch loss tracking across all trainer types
- `MetricsLogger`: Unified TensorBoard + console logging with modality suffixes

---

## Lossless Mask Codec

Lossless encoding of 256×256 binary masks into DC-AE latent-shaped tensors (`src/medgen/data/lossless_mask_codec.py`).

**Key insight**: 256×256 binary = 65,536 bits = 2,048 float32 values = DC-AE latent size

| Format | Spatial | Channels | Latent Shape |
|--------|---------|----------|--------------|
| f32 | 8×8 | 32 | `[32, 8, 8]` |
| f64 | 4×4 | 128 | `[128, 4, 4]` |
| f128 | 2×2 | 512 | `[512, 2, 2]` |

**API**:
- `encode_mask_lossless(mask, format)` → latent
- `decode_mask_lossless(latent, format)` → mask

**Use case**: Conditioning latent diffusion on segmentation masks without information loss.

---

## GroupedBatchSampler

For multi-modality diffusion with mode embedding, all samples in a batch must have the same modality (`src/medgen/data/loaders/common.py`).

**Problem**: Standard shuffle creates mixed-modality batches, but mode embedding expects homogeneous batches.

**Solution**: `GroupedBatchSampler` ensures:
1. Samples grouped by mode_id
2. Groups shuffled each epoch
3. Samples within groups shuffled
4. Each batch contains only one modality

---

## Output Directory Structure

```
runs/
├── diffusion_2d/{mode}/{strategy}_{size}_{timestamp}/
│   ├── .hydra/config.yaml    # Resolved config
│   ├── metadata.json         # Training metadata
│   ├── checkpoint_best.pt    # Best validation model
│   ├── checkpoint_latest.pt  # Latest checkpoint
│   └── tensorboard/          # TensorBoard logs
├── diffusion_3d/{mode}/{exp_name}_{timestamp}/
│   └── ...                   # Same structure as 2D
├── compression_2d/{mode}/{size}_{timestamp}/
│   └── ...
├── compression_3d/{mode}/{size}_{timestamp}/
│   └── ...
└── lr_finder/
```

**Auto-chaining (SLURM)**: Chained jobs reuse the same run directory via `CHAIN_RUN_DIR` environment variable, resuming from `checkpoint_latest.pt` across job segments.

---

## TensorBoard Metrics (Verified)

Complete list of all logged metrics, verified from source code. Source files noted for traceability.

### 1. Training Losses (`Loss/` prefix)

**Source**: `src/medgen/metrics/unified.py`

| TensorBoard Tag | VAE | VQVAE | DCAE | Diffusion | Condition |
|-----------------|-----|-------|------|-----------|-----------|
| `Loss/Generator_train` | ✅ | ✅ | ✅ | ❌ | Always |
| `Loss/L1_train` | ✅ | ✅ | ✅ | ❌ | `seg_mode=false` |
| `Loss/Perceptual_train` | ✅ | ✅ | ✅ | ✅ | `seg_mode=false` |
| `Loss/KL_train` | ✅ | ❌ | ❌ | ❌ | Always |
| `Loss/VQ_train` | ❌ | ✅ | ❌ | ❌ | Always |
| `Loss/Discriminator` | ✅ | ✅ | ✅ | ❌ | `has_gan=true` |
| `Loss/Adversarial` | ✅ | ✅ | ✅ | ❌ | `has_gan=true` |
| `Loss/BCE_train` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/Dice_train` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/Boundary_train` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/MSE_train` | ❌ | ❌ | ❌ | ✅ | Always |
| `Loss/Total_train` | ❌ | ❌ | ❌ | ✅ | Always |

### 2. Validation Losses (`Loss/` prefix)

**Source**: `src/medgen/metrics/unified.py`

| TensorBoard Tag | VAE | VQVAE | DCAE | Diffusion | Condition |
|-----------------|-----|-------|------|-----------|-----------|
| `Loss/Generator_val` | ✅ | ✅ | ✅ | ❌ | Always |
| `Loss/L1_val` | ✅ | ✅ | ✅ | ✅ | Always |
| `Loss/Perceptual_val` | ✅ | ✅ | ✅ | ❌ | `seg_mode=false` |
| `Loss/KL_val` | ✅ | ❌ | ❌ | ❌ | Always |
| `Loss/VQ_val` | ❌ | ✅ | ❌ | ❌ | Always |
| `Loss/BCE_val` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/Dice_val` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/Boundary_val` | ❌ | ✅ | ✅ | ❌ | `seg_mode=true` |
| `Loss/MSE_val` | ❌ | ❌ | ❌ | ✅ | Always |
| `Loss/Total_val` | ❌ | ❌ | ❌ | ✅ | Always |

### 3. Validation Quality Metrics (`Validation/` prefix)

**Source**: `src/medgen/metrics/unified.py`

| TensorBoard Tag | All Trainers | Condition |
|-----------------|--------------|-----------|
| `Validation/PSNR` | ✅ | `logging.psnr=true` |
| `Validation/PSNR_{modality}` | ✅ | Per-modality logging enabled |
| `Validation/MS-SSIM` | ✅ | `logging.msssim=true` |
| `Validation/MS-SSIM_{modality}` | ✅ | Per-modality logging enabled |
| `Validation/MS-SSIM-3D` | 3D only | `spatial_dims=3` |
| `Validation/LPIPS` | 2D only | `logging.lpips=true` |
| `Validation/LPIPS_{modality}` | 2D only | Per-modality logging enabled |
| `Validation/Dice` | VQVAE, DCAE | `seg_mode=true` |
| `Validation/IoU` | VQVAE, DCAE | `seg_mode=true` |

**Legacy Diffusion Metrics** (from `src/medgen/evaluation/visualization.py`):

| TensorBoard Tag | Mode | Condition |
|-----------------|------|-----------|
| `metrics/msssim` | bravo | `logging.msssim=true` |
| `metrics/msssim_t1_pre` | dual | `logging.msssim=true` |
| `metrics/msssim_t1_gd` | dual | `logging.msssim=true` |
| `metrics/psnr` | bravo | `logging.psnr=true` |
| `metrics/psnr_t1_pre` | dual | `logging.psnr=true` |
| `metrics/psnr_t1_gd` | dual | `logging.psnr=true` |
| `metrics/lpips` | bravo | `logging.lpips=true` |
| `metrics/lpips_t1_pre` | dual | `logging.lpips=true` |
| `metrics/lpips_t1_gd` | dual | `logging.lpips=true` |
| `metrics/boundary_sharpness` | bravo | `logging.boundary_sharpness=true` + seg available |
| `metrics/boundary_sharpness_t1_pre` | dual | `logging.boundary_sharpness=true` + seg available |
| `metrics/boundary_sharpness_t1_gd` | dual | `logging.boundary_sharpness=true` + seg available |

### 4. Regional Metrics (`regional/` prefix)

**Source**: `src/medgen/metrics/regional/base.py`

| TensorBoard Tag | All Trainers | Condition |
|-----------------|--------------|-----------|
| `regional/tumor_loss` | ✅ | `logging.regional_losses=true` |
| `regional/background_loss` | ✅ | `logging.regional_losses=true` |
| `regional/tumor_bg_ratio` | ✅ | `logging.regional_losses=true` |
| `regional/tiny` | ✅ | `logging.regional_losses=true` |
| `regional/small` | ✅ | `logging.regional_losses=true` |
| `regional/medium` | ✅ | `logging.regional_losses=true` |
| `regional/large` | ✅ | `logging.regional_losses=true` |
| `regional_{modality}/*` | ✅ | Single-modality modes (bravo, t1_pre, etc.) |

**Segmentation Regional Metrics** (`regional_seg/` prefix):

**Source**: `src/medgen/metrics/regional/tracker_seg.py`

| TensorBoard Tag | Trainers | Condition |
|-----------------|----------|-----------|
| `regional_seg/dice` | VQVAE, DCAE | `seg_mode=true` |
| `regional_seg/iou` | VQVAE, DCAE | `seg_mode=true` |
| `regional_seg/dice_tiny` | VQVAE, DCAE | `seg_mode=true` |
| `regional_seg/dice_small` | VQVAE, DCAE | `seg_mode=true` |
| `regional_seg/dice_medium` | VQVAE, DCAE | `seg_mode=true` |
| `regional_seg/dice_large` | VQVAE, DCAE | `seg_mode=true` |

### 5. Timestep Metrics (Diffusion only)

**Source**: `src/medgen/metrics/unified.py`

Timesteps are logged in normalized [0.0, 1.0] format (10 bins):

| TensorBoard Tag | Condition |
|-----------------|-----------|
| `Timestep/0.0-0.1` | `logging.timestep_losses=true` |
| `Timestep/0.1-0.2` | `logging.timestep_losses=true` |
| `Timestep/0.2-0.3` | `logging.timestep_losses=true` |
| ... (10 bins total) | |
| `Timestep/0.9-1.0` | `logging.timestep_losses=true` |
| `loss/timestep_region_heatmap` (figure) | `logging.timestep_region_losses=true` |

**What timestep losses measure**:
- **RFlow**: Velocity prediction MSE = `MSE(prediction, images - noise)`
- **DDPM**: Noise prediction MSE = `MSE(prediction, noise)`

These match the actual training loss, NOT reconstruction error.

**Expected RFlow pattern** (may seem counterintuitive):
- `t ≈ 0.0` (clean): **HIGH loss** - Model sees nearly-clean image, can't detect noise direction
- `t ≈ 1.0` (noisy): **LOW loss** - Model sees mostly noise, can learn velocity toward data

This is normal behavior. Early timesteps are harder because there's less noise signal to learn from.

**Note**: Regional losses (`regional/tumor_loss`, etc.) use reconstruction MSE instead, as they measure spatial output quality rather than training loss decomposition.

### 6. Training Diagnostics (`training/` prefix)

**Source**: `src/medgen/metrics/tracking/gradient.py`, `src/medgen/pipeline/compression_trainer.py`

**Compression trainers (VAE, VQVAE, DCAE) with GAN:**

| TensorBoard Tag | Condition |
|-----------------|-----------|
| `training/grad_norm_g_avg` | `logging.grad_norm=true` + `has_gan=true` |
| `training/grad_norm_g_max` | `logging.grad_norm=true` + `has_gan=true` |
| `training/grad_norm_d_avg` | `logging.grad_norm=true` + `has_gan=true` |
| `training/grad_norm_d_max` | `logging.grad_norm=true` + `has_gan=true` |

**Compression trainers without GAN / Diffusion trainer:**

| TensorBoard Tag | Condition |
|-----------------|-----------|
| `training/grad_norm_avg` | `logging.grad_norm=true` |
| `training/grad_norm_max` | `logging.grad_norm=true` |

### 7. Learning Rate (`LR/` prefix)

**Source**: `src/medgen/metrics/unified.py`, `src/medgen/pipeline/base_trainer.py`

| TensorBoard Tag | Trainers | Condition |
|-----------------|----------|-----------|
| `LR/Generator` | All | Always |
| `LR/Discriminator` | VAE, VQVAE, DCAE | `has_gan=true` |

### 8. VQ-VAE Codebook Metrics (`Codebook/` prefix)

**Source**: `src/medgen/metrics/tracking/codebook.py`

| TensorBoard Tag | Trainers | Condition |
|-----------------|----------|-----------|
| `Codebook/perplexity` | VQVAE | Always |
| `Codebook/utilization` | VQVAE | Always |
| `Codebook/dead_codes` | VQVAE | Always |
| `Codebook/entropy` | VQVAE | Always |
| `Codebook/perplexity_pct` | VQVAE | Always |

### 9. Generation Quality Metrics (`Generation/` prefix)

**Source**: `src/medgen/metrics/generation.py`

Tracks generation quality during diffusion training using distributional metrics. Compares generated samples against train/val distributions to detect overfitting.

| TensorBoard Tag | Frequency | Condition |
|-----------------|-----------|-----------|
| `Generation/KID_mean_train` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/KID_std_train` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/KID_mean_val` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/KID_std_val` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/CMMD_train` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/CMMD_val` | Every epoch | `generation_metrics.enabled=true` |
| `Generation/extended_KID_mean_train` | figure_interval | `generation_metrics.enabled=true` |
| `Generation/extended_KID_std_train` | figure_interval | `generation_metrics.enabled=true` |
| `Generation/extended_KID_mean_val` | figure_interval | `generation_metrics.enabled=true` |
| `Generation/extended_KID_std_val` | figure_interval | `generation_metrics.enabled=true` |
| `Generation/extended_CMMD_train` | figure_interval | `generation_metrics.enabled=true` |
| `Generation/extended_CMMD_val` | figure_interval | `generation_metrics.enabled=true` |

**Test Evaluation** (`test_best/`, `test_latest/` prefix):

| TensorBoard Tag | Condition |
|-----------------|-----------|
| `test_best/FID` | `generation_metrics.enabled=true` |
| `test_best/KID_mean` | `generation_metrics.enabled=true` |
| `test_best/KID_std` | `generation_metrics.enabled=true` |
| `test_best/CMMD` | `generation_metrics.enabled=true` |

**Metric Details:**
- **KID** (Kernel Inception Distance): Unbiased MMD using polynomial kernel on ResNet50 features (2048-dim). Lower = better. Returns mean ± std across subsets.
- **CMMD** (CLIP Maximum Mean Discrepancy): RBF kernel-based MMD on BiomedCLIP embeddings (512-dim). Lower = better. Medical domain-aware.
- **FID** (Fréchet Inception Distance): Fréchet distance between feature distributions. Lower = better. Only computed at test time.

**Feature Extractors:**
- **ResNet50Features**: ImageNet pretrained (default) or RadImageNet (medical domain). Uses `torch.compile(mode="reduce-overhead")` and AMP (bfloat16).
- **BiomedCLIPFeatures**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. Uses `torch.compile` and AMP.

**Optimizations:**
- Batched generation (batch_size inherited from `training.batch_size`)
- Full-batch rounding: 100 samples with batch_size=16 → 112 samples (7 × 16)
- Reference features cached at training start for efficiency
- `non_blocking=True` GPU transfers

**Overfitting Detection:**
- Healthy training: `KID_train` and `KID_val` decrease together, gap stays small
- Overfitting signal: `KID_train` decreases while `KID_val` stagnates/increases

### 10. Resource Metrics

**Source**: `src/medgen/pipeline/utils.py`, `src/medgen/metrics/tracking/flops.py`

| TensorBoard Tag | All Trainers | Condition |
|-----------------|--------------|-----------|
| `VRAM/allocated_GB` | ✅ | Always |
| `VRAM/reserved_GB` | ✅ | Always |
| `VRAM/max_allocated_GB` | ✅ | Always |
| `FLOPs/TFLOPs_epoch` | ✅ | `logging.flops=true` |
| `FLOPs/TFLOPs_total` | ✅ | `logging.flops=true` |

### 11. Test Evaluation Metrics (`test_best/`, `test_latest/` prefix)

**Source**: `src/medgen/pipeline/evaluation.py`

| TensorBoard Tag | All Trainers | Condition |
|-----------------|--------------|-----------|
| `test_best/L1` | ✅ | Always |
| `test_best/MSE` | Diffusion | Always |
| `test_best/MS-SSIM` | ✅ | `compute_msssim=true` |
| `test_best/MS-SSIM-3D` | 3D only | `compute_msssim_3d=true` |
| `test_best/PSNR` | ✅ | `compute_psnr=true` |
| `test_best/LPIPS` | 2D only | `compute_lpips=true` |
| `test_best/Dice` | VQVAE, DCAE | `seg_mode=true` |
| `test_best/IoU` | VQVAE, DCAE | `seg_mode=true` |
| `test_best/{metric}_{modality}` | ✅ | Per-modality test enabled |
| `test_best/worst_batch` (figure) | ✅ | Always |
| `test_best/Timestep/0.0-0.1` ... `0.9-1.0` | Diffusion | `logging.timestep_losses=true` |
| `test_best/FID` | Diffusion | `generation_metrics.enabled=true` |
| `test_best/KID_mean` | Diffusion | `generation_metrics.enabled=true` |
| `test_best/KID_std` | Diffusion | `generation_metrics.enabled=true` |
| `test_best/CMMD` | Diffusion | `generation_metrics.enabled=true` |

Same structure applies for `test_latest/` prefix.

### 12. Figures

**Source**: Various trainer files

| TensorBoard Tag | Trainers | Condition |
|-----------------|----------|-----------|
| `Validation/worst_batch` | VAE, VQVAE, DCAE | `logging.worst_batch=true` |
| `Validation/WorstBatch_3D` | VAE3D, VQVAE3D | `logging.worst_batch=true` |
| `denoising_trajectory` | Diffusion | `logging.intermediate_steps=true` |
| `Generated_Images` | Diffusion | figure_interval |
| `Generated_T1_Pre` | Diffusion dual | figure_interval |
| `Generated_T1_Gd` | Diffusion dual | figure_interval |
| `test_best/worst_batch` | All | Always |
| `test_latest/worst_batch` | All | Always |

---

### Metrics Summary by Trainer Type

#### VAE / VAE-3D
- **Training**: `Loss/Generator_train`, `Loss/L1_train`, `Loss/Perceptual_train`, `Loss/KL_train`, `Loss/Discriminator`, `Loss/Adversarial`
- **Validation**: `Validation/PSNR_{mode}`, `Validation/MS-SSIM_{mode}`, `Validation/LPIPS_{mode}`, `Loss/L1_val`, `Loss/KL_val`
- **Regional**: `regional_{mode}/tumor_loss`, `regional_{mode}/background_loss`, `regional_{mode}/tiny`, etc.
- **Diagnostics**: `training/grad_norm_g_avg`, `training/grad_norm_g_max`, `training/grad_norm_d_avg`, `training/grad_norm_d_max`
- **Figures**: `Validation/worst_batch`

#### VQ-VAE / VQ-VAE-3D
- Same as VAE, but `Loss/VQ_train/val` instead of `Loss/KL_train/val`
- **Additional**: `Codebook/perplexity`, `Codebook/utilization`, `Codebook/dead_codes`, `Codebook/entropy`, `Codebook/perplexity_pct`
- **Seg mode**: `Loss/BCE_train/val`, `Loss/Dice_train/val`, `Loss/Boundary_train/val`, `Validation/Dice`, `Validation/IoU`

#### DC-AE / DC-AE-3D
- Same as VAE but **NO** regularization loss (no KL or VQ)
- Seg mode same as VQ-VAE seg mode

#### Diffusion
- **Training**: `Loss/Total_train`, `Loss/MSE_train` (perceptual loss disabled by default)
- **Validation**: `Loss/Total_val`, `Loss/MSE_val`, `Validation/PSNR_{mode}`, `Validation/MS-SSIM_{mode}`, `Validation/LPIPS_{mode}`
- **Timestep**: `Timestep/0.0-0.1`, ..., `Timestep/0.9-1.0` (10 bins, normalized format)
- **Regional**: `regional_{mode}/tumor_loss`, `regional_{mode}/background_loss`, `regional_{mode}/tumor_bg_ratio`, by size
- **Diagnostics**: `training/grad_norm_avg`, `training/grad_norm_max`
- **Figures**: `denoising_trajectory`, `Generated_Images`, `Validation/worst_batch`

---

## Configuration Examples

```bash
# Diffusion (UNet, 2D)
python -m medgen.scripts.train strategy=rflow mode=dual model.image_size=256

# Diffusion (UNet, 3D)
python -m medgen.scripts.train strategy=rflow mode=bravo model.spatial_dims=3

# Diffusion (DiT)
python -m medgen.scripts.train model=dit model.variant=B mode=bravo strategy=rflow

# Diffusion with SAM
python -m medgen.scripts.train mode=bravo strategy=rflow training.sam.enabled=true

# VAE (2D)
python -m medgen.scripts.train_compression --config-name=vae mode=dual vae.latent_channels=4

# VAE (3D)
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality

# LR finder
python -m medgen.scripts.lr_finder mode=dual model_type=vae

# Disable specific logging
python -m medgen.scripts.train training.logging.lpips=false

# Enable regional losses for seg mode
python -m medgen.scripts.train mode=seg training.logging.regional_losses=true

# Visualize augmentations
python -m medgen.scripts.visualize_augmentations augment_type=vae

# DC-AE (32× compression, default)
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality

# DC-AE (64× compression)
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality

# DC-AE with pretrained ImageNet weights
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality \
    dcae.pretrained="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"

# DC-AE Phase 3 (GAN training)
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality \
    training.phase=3 dcae.adv_weight=0.1

# DC-AE Segmentation Mask Compression
# Uses BCE + Dice + Boundary loss, Dice/IoU metrics
python -m medgen.scripts.train_compression --config-name=dcae mode=seg dcae.seg_mode=true

# DC-AE Seg Compression with regional metrics (per-tumor Dice by size)
python -m medgen.scripts.train_compression --config-name=dcae mode=seg \
    dcae.seg_mode=true \
    training.logging.regional_losses=true
```

---

## IDUN Cluster Experiments

### Diffusion Experiments

| Experiment | Resolution | Network | GPUs | Features |
|------------|------------|---------|------|----------|
| exp1 | 128 | [128,256,256] | 1 | Baseline (no aug/EMA/Min-SNR) |
| exp2 | 128 | [128,256,256] | 1 | + 100 timesteps |
| exp3 | 128 | [128,256,256] | 1 | + augmentation |
| exp4 | 128 | [128,256,256] | 1 | + EMA |
| exp5 | 128 | [128,256,256] | 1 | + Min-SNR |
| exp6 | 256 | [128,256,256] | 4 | DDP baseline (no features) |
| exp7 | 256 | [128,256,256,512] | 4 | DDP + extended network |
| exp9_1 | 128 | [128,256,256] | 1 | ScoreAug (rotation + translation + cutout) |
| exp9_2 | 128 | [128,256,256] | 1 | ScoreAug (all transforms incl. brightness) |
| exp11_1 | 128 | [128,256,256] | 1 | SAM optimizer (rho=0.05) |
| exp11_2 | 128 | [128,256,256] | 1 | ASAM optimizer (adaptive=true) |
| exp12_1 | 128 | DiT-S | 1 | DiT Small (33M params) |
| exp12_2 | 128 | DiT-B | 1 | DiT Base (130M params) |
| exp12_3 | 128 | DiT-L | 1 | DiT Large (458M params) |
| exp19_1 | 128 | [128,256,256] | 1 | Constant LR (no cosine decay) |
| exp20_1 | 128 | [128,256,256] | 1 | Gradient Noise (sigma=0.01, decay=0.55) |
| exp21_1 | 128 | [128,256,256] | 1 | Curriculum Timesteps (50 epoch warmup, 0-0.3 start) |
| exp22_1 | 128 | [128,256,256] | 1 | Timestep Jitter (std=0.05) |
| exp23_1 | 128 | [128,256,256] | 1 | Noise Augmentation (std=0.1) |
| exp24_1 | 128 | [128,256,256] | 1 | Feature Perturbation (std=0.1, mid block) |
| exp25_1 | 128 | [128,256,256] | 1 | Self-Conditioning Consistency (prob=0.5, weight=0.1) |

### VAE Experiments

| Experiment | Type | Resolution | Features |
|------------|------|------------|----------|
| exp1 | Progressive | 64→128→256 | Full (aug, batch_aug, plateau detection) |
| exp2 | Single | 256 | Fine-tune from exp1 |
| exp3 | Single | 256 | Multi-modality (4 modalities pooled) |
| exp4 | Single | 256 | Multi-modality + 4x compression (64x64 latent) |
| exp5 | Single | 256 | Multi-modality + Pure BF16 weights |
| exp6 | VQ-VAE | 256 | Multi-modality, 512 codebook, 8x compression |
| exp7 | 3D VAE | 128×128×160 | Multi-modality, gradient checkpointing, disable_gan=true |
| exp8 | 3D VQ-VAE | 128×128×160 | Multi-modality, gradient checkpointing, GAN enabled |
| exp9 | DC-AE f32 | 256 | Multi-modality, 32× compression (8×8×32 latent) |
| exp10 | DC-AE f64 | 256 | Multi-modality, 64× compression (4×4×128 latent) |
| exp11.1 | DC-AE seg | 256 | Seg mask compression, BCE+Dice+Boundary, per-tumor Dice |

### H100 Submit Script

Prefers H100, falls back to H100|A100 after timeout:

```bash
# Submit with 10 min H100 wait (default)
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_vae_baseline.slurm

# Custom timeout (30 min)
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_vae_baseline.slurm 1800

# Run in background
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_vae_baseline.slurm --bg
```

Background mode:
- Logs to `/tmp/submit_h100_$$.log`
- Check with: `tail -f /tmp/submit_h100_$$.log`
- Kill with: `kill <PID>` (shown on launch)
