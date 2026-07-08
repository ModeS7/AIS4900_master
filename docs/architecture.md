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
│   ├── utils.py                 # Slice extraction, merge, binarize_seg, save_nifti
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
│       ├── seg_conditioned.py   # Seg conditioned on tumor sizes (2D, imports from datasets.py)
│       ├── single.py            # Single modality loaders (seg, bravo)
│       ├── unified.py           # Unified loader dispatch
│       ├── vae.py               # VAE dataloaders
│       ├── restoration_3d.py    # Paired (degraded, clean) 3D loader for restoration mode
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
│   ├── strategy_rflow.py        # RFlow strategy
│   ├── strategy_bridge.py       # Diffusion Bridge (paired restoration; Zhang et al. 2025)
│   ├── strategy_irsde.py        # IR-SDE mean-reverting SDE (Luo et al., ICML 2023)
│   └── strategy_resfusion.py    # Resfusion residual noise diffusion (Shi et al., NeurIPS 2024)
├── downstream/                  # Downstream task evaluation
│   ├── data.py                  # Segmentation data loading
│   ├── segmentation_trainer.py  # SegResNet trainer (2D/3D)
│   └── nnunet/                  # nnU-Net v2 integration
│       ├── convert_dataset.py   # Convert to nnU-Net format
│       ├── evaluate.py          # 5-fold ensemble evaluation
│       ├── splits.py            # CV splits + isolated preprocessed dir creation
│       └── trainer.py           # nnU-Net training wrapper
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
│   ├── generation_3d.py         # Generation metrics for 3D (incl. triplanar features)
│   ├── generation_computation.py # Metric computation helpers
│   ├── generation_sampling.py   # Sample generation for metrics
│   ├── fwd.py                   # Fréchet Wavelet Distance (Veeramacheneni et al., ICLR 2025)
│   ├── feature_extractors.py    # ResNet50 (ImageNet/RadImageNet), BiomedCLIP extractors
│   ├── figures.py               # Reconstruction figures
│   ├── constants.py             # RANO-BM tumor size thresholds
│   ├── brain_mask.py            # Brain mask utilities
│   ├── dispatch.py              # Metric dispatch by trainer type
│   ├── metric_computation.py    # Metric computation utilities
│   ├── metric_logger.py         # Metric logging
│   ├── regional_manager.py      # Regional metrics management
│   ├── sampler.py               # Metric sampling
│   ├── seg_metrics.py           # Segmentation-specific metrics (Dice, IoU)
│   ├── morphological.py         # Morphological comparison (Wasserstein on tumor stats)
│   ├── mc_dropout.py            # Monte Carlo dropout for uncertainty estimation
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
│   ├── factory.py               # Model factory (UNet, DiT, HDiT, UViT, Mamba, WDM)
│   ├── dit.py                   # DiT (Scalable Interpolant Transformer)
│   ├── dit_blocks.py            # Transformer blocks with adaLN-Zero
│   ├── hdit.py                  # HDiT (Hierarchical Diffusion Transformer)
│   ├── uvit.py                  # UViT (ViT with Skip Connections)
│   ├── mamba_diff.py            # LaMamba-Diff (state-space model, pixel-space only)
│   ├── mamba_blocks.py          # Mamba SSM blocks
│   ├── handoff.py               # HandoffWrapper (two-stage low-t/high-t inference)
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
│   ├── discriminator_manager.py # GAN discriminator lifecycle
│   ├── evaluation.py            # Legacy test evaluation
│   ├── validation.py            # Legacy validation
│   ├── visualization.py         # Validation visualization
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
    ├── visualize_augmentations.py  # Debug augmentation pipelines
    ├── compute_pixel_stats.py   # Compute brain-only pixel statistics for normalization
    ├── debug_ldm_roundtrip.py   # Debug latent diffusion model roundtrip
    ├── eval_compression_fid.py  # Evaluate compression models using FID
    ├── eval_diffrs.py           # Evaluate DiffRS rejection sampling
    ├── eval_restart.py          # Evaluate restart sampling
    ├── measure_distinguishability.py  # Measure real vs synthetic distinguishability
    ├── plot_tumor_detection.py  # Plot tumor detection results
    ├── recompute_latent_stats.py # Recompute latent space statistics
    ├── train_nnunet.py          # nnU-Net v2 training with TensorBoard (argparse)
    ├── eval_nnunet.py           # nnU-Net 5-fold ensemble evaluation
    ├── find_optimal_freeu.py    # Grid search for optimal FreeU parameters
    ├── find_optimal_cfg.py      # Grid search for optimal CFG scale
    ├── synthesize_phema.py      # Post-hoc EMA synthesis sweep (Karras EDM2)
    ├── eval_time_shift.py       # Evaluate time-shifted sampling schedules
    ├── compute_brain_atlas.py   # Compute brain atlas (union of training brain masks)
    ├── verify_ldm_pipeline.py   # Verify latent diffusion pipeline
    └── verify_wdm_pipeline.py   # Verify wavelet diffusion pipeline

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
├── model/mamba.yaml             # Mamba (LaMamba-Diff, 2D, pixel-space)
├── model/mamba_3d.yaml          # Mamba (LaMamba-Diff, 3D, pixel-space)
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
├── diffrs.yaml                  # DiffRS rejection sampling config
├── pixel_norm/default.yaml      # Pixel normalization (disabled)
├── pixel_norm/bravo.yaml        # Brain-only N(0,1) stats for BRAVO
├── pixel_norm/t1_pre.yaml       # Brain-only N(0,1) stats for T1 pre-contrast
├── pixel_norm/t1_gd.yaml        # Brain-only N(0,1) stats for T1 post-contrast
├── paths/{local,cluster}.yaml
├── strategy/{ddpm,rflow,bridge,irsde,resfusion}.yaml
├── mode/{seg,bravo,bravo_seg_cond,dual,multi,multi_modality}.yaml
├── mode/{seg_compression,seg_conditioned,seg_conditioned_3d}.yaml
├── mode/{seg_conditioned_input,seg_conditioned_input_3d}.yaml
├── mode/triple.yaml             # Triple mode (T1pre + T1gd + FLAIR)
├── mode/restoration.yaml        # Paired (degraded, clean) training
└── training/{default,fast_debug,smoke_test}.yaml
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
| `compute_pixel_stats.py` | Compute brain-only pixel statistics | N/A | For `pixel_norm` configs |
| `eval_diffrs.py` | Evaluate DiffRS rejection sampling | N/A | Argparse (not Hydra) |
| `eval_compression_fid.py` | Evaluate compression model FID | N/A | |
| `eval_restart.py` | Evaluate restart sampling | N/A | |
| `measure_distinguishability.py` | Measure real vs synthetic distinguishability | N/A | |
| `debug_ldm_roundtrip.py` | Debug LDM encode/decode roundtrip | N/A | |
| `recompute_latent_stats.py` | Recompute latent space statistics | N/A | |
| `plot_tumor_detection.py` | Plot tumor detection visualization | N/A | |
| `verify_ldm_pipeline.py` | Verify latent diffusion pipeline | N/A | |
| `verify_wdm_pipeline.py` | Verify wavelet diffusion pipeline | N/A | |
| `train_nnunet.py` | nnU-Net v2 training with TensorBoard logging | N/A | Argparse (not Hydra) |
| `eval_nnunet.py` | nnU-Net 5-fold ensemble inference + evaluation | N/A | Argparse (not Hydra) |
| `find_optimal_freeu.py` | Grid search for optimal FreeU (b, s) parameters | N/A | Argparse (not Hydra) |
| `find_optimal_cfg.py` | Grid search for optimal CFG scale | N/A | Argparse (not Hydra) |
| `synthesize_phema.py` | Post-hoc EMA synthesis sweep for optimal sigma_rel | N/A | Argparse (not Hydra) |
| `eval_time_shift.py` | Evaluate time-shifted sampling ratios (sweep or golden-section search) | N/A | Argparse (not Hydra) |
| `eval_steps_pca.py` | Golden-section search for optimal steps by PCA brain shape error | N/A | Argparse (not Hydra) |
| `compute_brain_atlas.py` | Compute brain atlas (union of training brain masks) | N/A | Argparse |
| `compute_brain_pca.py` | Compute PCA model from training brain masks | N/A | Argparse |

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
    reconstruction_loss: float
    perceptual_loss: float
    regularization_loss: float = 0.0  # KL or VQ loss
    adversarial_loss: float = 0.0
    discriminator_loss: float = 0.0
    base_loss: float = 0.0        # primary diffusion loss (MSE/L1/pseudo-Huber/etc.)
    aux_bin_loss: float = 0.0     # auxiliary size-bin prediction loss (0 if disabled)

    def to_legacy_dict(self, reg_key: str | None = 'kl') -> Dict[str, float]:
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
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality \
    dcae.structured_latent.enabled=true

# Step 2: Train diffusion with augmented diffusion training
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    training.augmented_diffusion.enabled=true
```

**Important**: Structured latent space requires latent diffusion (not pixel space). The `augmented_diffusion.enabled` setting is ignored in pixel space.

### Mamba (LaMamba-Diff, pixel-space only)

State-space model U-Net combining SS2D (multi-directional Mamba) for
global context, windowed self-attention for local detail, and FFN for
channel mixing — all conditioned via AdaLN-Zero. 2D uses 4-directional
SS2D scans; 3D uses 6-directional (±D, ±H, ±W).

Verbatim from `configs/model/mamba_3d.yaml` (the 3D variant):

```yaml
type: mamba
spatial_dims: 3
image_size: 256
depth_size: 160
patch_size: 2
variant: B                 # S / B / L / XL

# U-Net structure
depths: [2, 2, 2, 2]       # Blocks per encoder stage
bottleneck_depth: 2
skip: 2                    # Last N stages don't downsample

# Mamba SSM parameters
ssm_d_state: 1             # SSM state dimension (1 is sufficient)
ssm_ratio: 2.0             # SSM inner dim = ssm_ratio * embed_dim

# Attention parameters
window_size: 8             # Window size for local attention
mlp_ratio: 4.0             # FFN expansion ratio
```

**Variants** (per YAML comment block, `(embed_dim, num_heads)` →
approximate param count):
- S: 128, 4 → ~30M
- B: 192, 8 → ~80M
- L: 256, 16 → ~200M
- XL: 320, 16 → ~450M

**Usage:** `model=mamba` or `model=mamba_3d`. **No latent-space variant**
in this repo (no `configs/latent/mamba*.yaml`). See
`src/medgen/models/mamba_diff.py` and `mamba_blocks.py`.

Reference: LaMamba-Diff (Fu et al. 2024, arXiv:2408.02615; cited in
`src/medgen/models/mamba_diff.py:14`).

### WDM (Wavelet Diffusion Model, 3D-only)

Diffusion in the 3D Haar wavelet domain. `configs/model/wdm_3d.yaml` is
a 5-level UNet (channels [64, 128, 128, 256, 256], **no attention**, 2
res blocks per level, ~74M params for bravo) modeled after Friedrich et
al.'s WDM. Used with `wavelet/default.yaml` (lossless Haar decomposition)
and `strategy=ddpm` with **`strategy.prediction_type=sample`** (x₀-pred,
verified in exp19 SLURMs) — ε-prediction does not work well in this
domain per the paper.

```bash
python -m medgen.scripts.train mode=bravo strategy=ddpm \
    strategy.prediction_type=sample \
    model=wdm_3d wavelet=default model.spatial_dims=3
```

**Status:** Implemented and trained through 1000 epochs (exp19 era).
Overfitting observed beyond 500ep on bravo (exp26_1: FID 77 at 1000ep
vs 67 at 500ep). See `papers/WDM/WDM_PAPER_FINDINGS.md` for the
literature review and `docs/experiment_results_3d.md` §"Part 5b" for
run results.

Reference: Friedrich et al. 2024, "WDM: 3D Wavelet Diffusion Models for
High-Resolution Medical Image Synthesis", arXiv:2402.19043.

### HandoffWrapper (two-stage inference)

Inference-time wrapper that combines two checkpoints: a high-t (seed)
model handles t > `handoff_t`, a low-t (fine-tuned) model handles
t ≤ `handoff_t`. Used for low-t fine-tunes (exp48 family) where the
fine-tune only sees t ∈ [0, handoff_t] during training.

**No `configs/model/handoff.yaml`.** The wrapper is constructed at
generation time. `generate.py` uses Hydra config keys, while
`find_optimal_steps.py` uses argparse flags:

```bash
# generate.py — Hydra keys (note: gen_mode, not mode, per configs/generate.yaml:25)
python -m medgen.scripts.generate gen_mode=bravo \
    image_model=<low-t-fine-tune.pt> \
    image_model_high_t=<base.pt> \
    handoff_t=0.25

# find_optimal_steps.py — argparse flags
python -m medgen.scripts.find_optimal_steps \
    --high-t-checkpoint <base.pt> \
    --low-t-checkpoint <low-t-fine-tune.pt> \
    --handoff-t 0.25 \
    --data-root <root> --output-dir <out>
```

See `configs/generate.yaml` for the Hydra schema (`image_model`,
`image_model_high_t`, `handoff_t`) and `src/medgen/models/handoff.py` for
the wrapper.

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
    per_modality_val_loaders=None, start_epoch=0, max_epochs=None
) -> int  # Returns last epoch number
```
- `max_epochs`: Override total epochs
- `per_modality_val_loaders`: optional `dict[str, DataLoader]` for multi-modality metrics
- Note: `train()` is inherited from `BaseTrainer`; there is no `early_stop_fn` parameter

---

## Strategies

| Strategy | Predicts | Target | Timestep Sampling | Use case |
|----------|----------|--------|-------------------|----------|
| DDPM | Noise (epsilon) — or `sample` (x₀) when `strategy.prediction_type=sample` | `noise` / `clean` | Uniform random (discrete) | Pixel + latent generation; **x₀-prediction is what WDM uses** |
| RFlow | Velocity | `images - noise` | Logit-normal (continuous, biased to middle) | Pixel + latent generation (default) |
| Bridge | x̂₀ (`prediction_type: x0`) | clean | Uniform t∈[0,1] | Paired restoration; γ_max=0.125 for 3D brain |
| IR-SDE | Noise (ε) — score implicit via `score = -ε/σ` (`prediction_type: noise`) | `noise` | Uniform t∈[0,1] | Paired restoration (Luo et al., ICML 2023) |
| Resfusion | Resnoise (`prediction_type: resnoise`) | `ε + coeff·R` | Discrete T=12 | Paired restoration; ~5–12 reverse steps |

**Restoration strategies (Bridge / IR-SDE / Resfusion)** — see
`src/medgen/diffusion/strategy_{bridge,irsde,resfusion}.py`. All three
operate on **paired (degraded, clean)** data and require
`mode=restoration`. They differ in their forward dynamics and number
of inference steps:

- **Bridge**: x_t = (1-t)·x₀ + t·x₁ + γ_max·√(4t(1-t))·ε. Predicts x̂₀
  (Zhang et al. 2025, arXiv:2504.15267).
- **IR-SDE**: dx = θ_t(μ - x)dt + σ_t·dw, mean-reverting toward the
  degraded image. L1 loss (eq. 15), posterior sampling at inference
  (Luo et al., ICML 2023; reimplementation of
  github.com/Algolzw/image-restoration-sde).
- **Resfusion**: Short linear schedule (T=12). Resnoise = ε + coeff·R
  where R = degraded − clean. Reaches good quality in ~5–12 reverse
  steps (Shi et al., NeurIPS 2024; reimplementation of
  github.com/nkicsl/Resfusion).

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
| `triple` | 4 | 3 | Seg mask | `[B, 4, H, W]` = [t1_pre, t1_gd, flair, seg] |
| `multi` | 2 | 1 | Seg mask + mode_id | `[B, 2, H, W]` = [image, seg] + mode embedding |
| `bravo_seg_cond` | 8 | 4 | Latent seg mask | `[B, 8, ...]` = [bravo_latent(4), seg_latent(4)] |
| `seg_conditioned` | 1 + size_bins | 1 | Size bins (FiLM) | `[B, 1, H, W]` seg + size bin embedding |
| `seg_conditioned_input` | 1 + 7 bin maps | 1 | Size bins (channel concat) | `[B, 8, H, W]` seg + 7 binary bin maps |
| `seg_conditioned_3d` | 1 + size_bins | 1 | Size bins (FiLM, 3D) | `[B, 1, D, H, W]` seg + 3D RANO-BM size bins |
| `seg_conditioned_input_3d` | 1 + 7 bin maps | 1 | Size bins (channel concat, 3D) | `[B, 8, D, H, W]` seg + 7 binary bin maps |
| `restoration` | 2 | 1 | Degraded volume | `[B, 2, D, H, W]` = [x_t, degraded_volume] (verbatim from `configs/mode/restoration.yaml:14`); paired training, model architecture identical to bravo |

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

| Space | scale_factor | rescale | Purpose |
|-------|--------------|---------|---------|
| `PixelSpace` | 1 | opt-in | Direct pixel diffusion (default) |
| `SpaceToDepthSpace` | 2 | opt-in | 3D pixel rearrangement (PixelUnshuffle3D, no learned transform) |
| `WaveletSpace` | 2 | default off (`rescale: false` in `configs/wavelet/default.yaml:27`) | 3D Haar wavelet decomposition (8 subbands, per-subband normalized) |
| `LatentSpace` | 4-128 | N/A | Compressed diffusion via VAE/VQ-VAE/DC-AE (auto-detected from checkpoint) |

**[-1,1] Rescaling**: All spaces except `LatentSpace` support an optional `rescale` parameter that maps [0,1] data to [-1,1] inside `encode()` and back in `decode()`. This keeps all downstream code (metrics, viz, saving) at [0,1]. Use `training.rescale_data=true` for pixel/S2D spaces, or `wavelet.rescale=true` (opt-in; default is false) for wavelet space.

**Brain-only N(0,1) normalization**: `PixelSpace` also supports per-channel shift/scale normalization via `pixel_norm=bravo` (or `t1_pre`, `t1_gd`). This normalizes using brain-only statistics: `encode: (x - shift) / scale`, `decode: z * scale + shift`. Brain voxels get mean=0, std=1 (matching noise distribution); background maps to ~-2.44. Configs in `configs/pixel_norm/`. Conditioning (seg masks) is NOT normalized — only the noisy image channel. Priority order: shift/scale > rescale [-1,1] > identity.

**`needs_decode` property**: Use `space.needs_decode` (not `space.scale_factor > 1`) to check if `decode()` must be called to get pixel-space output. This correctly handles rescaling and normalization in PixelSpace (scale_factor=1 but still needs decode).

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
   python -m medgen.scripts.generate image_model=dual.pt gen_mode=dual strategy=rflow
```

---

## Loss Functions

### Diffusion Training
```python
total_loss = mse_loss + perceptual_weight * perceptual_loss + ffl_weight * focal_frequency_loss
# Default weights: perceptual_weight=0.0, ffl_weight=0.0 (both disabled by default)
# To enable LPIPS: training.perceptual_weight=0.001
# NOTE: Seg mode auto-disables perceptual loss (pretrained features don't apply to binary masks)
```

### Loss schedules (per-timestep weighting)

LPIPS and Focal Frequency loss can be weighted by a piecewise-linear
schedule of the diffusion timestep, so the auxiliary loss only fires in
specific noise regimes. Implemented in `pipeline/trainer.py` via
`_compute_t_schedule_weight()`.

| Knob (`training.<name>`) | Default | Effect |
|---|---|---|
| `perceptual_weight` | `0.0` | Global LPIPS coefficient |
| `perceptual_max_timestep` | `null` | Legacy: enable LPIPS only for `t < value/num_train_timesteps` (e.g. `250` enables LPIPS for low-t bottom-25%) |
| `perceptual_t_schedule` | `null` | `[t_on, t_full, t_off]` in normalized [0,1] units. Zero below `t_on`, ramps to 1 at `t_full`, plateaus, drops to 0 at `t_off`. Overrides `perceptual_max_timestep` when set |
| `focal_frequency_weight` | `0.0` | Global FFL coefficient (slice-wise FFT for 3D) |
| `focal_frequency_t_schedule` | `null` | Same `[t_on, t_full, t_off]` semantics as above, applied to FFL |

Examples used in production runs (set via SLURM hydra overrides):
- `exp32_2_*`: `perceptual_max_timestep=250` — LPIPS only at low t
- `exp37_*`:   `perceptual_t_schedule=[0.05, 0.20, 0.70]` — LPIPS at high t
- FFL experiments: `focal_frequency_t_schedule=[0.10, 0.30, 0.80]` with `focal_frequency_weight` ∈ {0.5, 0.7, 1.0} per run (verbatim from `IDUN/train/diffusion_3d/*.slurm`)

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

Only lossless spatial transforms. Verbatim from `src/medgen/augmentation/augmentation.py:134-137`:

```python
return A.Compose([
    A.HorizontalFlip(p=0.5),
    DiscreteTranslate(max_percent_x=0.2, max_percent_y=0.1, p=1.0),
])
```
Note: NO rotation (interpolation would blur). DiscreteTranslate is integer-pixel
(lossless). Translate range is asymmetric (±20% X, ±10% Y) because brain
shapes are oval/vertical.

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

Reference: [arxiv.org/abs/2508.07926](https://arxiv.org/abs/2508.07926). For
the omega-vector encoding spec (layout, identity-as-zeros invariant, 3D dim
overload), see [`@docs/scoreaug_omega.md`](scoreaug_omega.md).

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

- **File**: `src/medgen/augmentation/score_aug.py`
- **Wrapper**: `ScoreAugModelWrapper` injects omega conditioning into the model's time embedding (auto-detects `time_embed` for UNet or `t_embedder` for DiT/HDiT/UViT)
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

Post-hoc quality improvement for diffusion sampling without retraining the model. A small discriminator head (**~0.3M params for 2D, ~0.9M for 3D** with default `mid_channels=128`) is trained on top of the frozen UNet encoder to evaluate intermediate samples during generation.

**How it works**:
1. At each denoising step, the discriminator checks if the intermediate sample looks realistic for that noise level
2. Bad trajectories are rejected and retried with new noise
3. The diffusion model is never modified

**Architecture** (verbatim from `src/medgen/diffusion/diffrs.py:138-164`):
- Feature extractor: Frozen UNet encoder (already trained)
- Classification head: 3 stacked `Conv → GroupNorm → SiLU` blocks (1×1 channel-reduction conv, then two stride-2 3×3 convs), followed by `AdaptiveAvgPool → Linear`. With `mid_channels=128`: ~0.3M params (2D) / ~0.9M params (3D).

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

Verbatim factory names from `src/medgen/data/loaders/volume_3d.py`:

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `create_vae_3d_dataloader()` | `volume_3d.py:723` | 3D volumetric compression (VAE/VQ-VAE/DC-AE) train |
| `create_vae_3d_validation_dataloader()` | `volume_3d.py:768` | 3D compression validation |
| `create_vae_3d_test_dataloader()` | `volume_3d.py:791` | 3D compression test |
| `create_vae_3d_multi_modality_dataloader()` | `volume_3d.py:928` | Multi-modality 3D compression train |
| `create_vae_3d_multi_modality_validation_dataloader()` | `volume_3d.py:978` | Multi-modality 3D val |
| `create_vae_3d_multi_modality_test_dataloader()` | `volume_3d.py:999` | Multi-modality 3D test |
| `create_segmentation_dataloader()` | `volume_3d.py:1232` | 3D segmentation training |
| `create_single_modality_dataloader_with_seg()` | `volume_3d.py:1341` | Single-modality 3D with seg conditioning |

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
Verbatim from `src/medgen/pipeline/checkpoint_manager.py:155-180`:
```python
{
    'model_state_dict': ...,           # AutoencoderKL (the generator)
    'discriminator_state_dict': ...,   # PatchDiscriminator (only when present)
    'optimizer_state_dict': ...,       # generator optimizer
    'optimizer_d_state_dict': ...,     # discriminator optimizer (only when present)
    'scheduler_state_dict': ...,       # only when scheduler present
    'ema_state_dict': ...,             # only when EMA enabled
    'epoch': int,
    'best_metric': float,
    'metric_name': str,
    'checkpoint_manager_version': int,
    'config': {...},                   # only when self.config is not None
}
```

---

## Key Code Locations

### Core
- `src/medgen/core/constants.py`: ModeType enum, thresholds
- `src/medgen/core/schedulers.py`: LR schedulers (cosine, warmup, plateau)

### Models
- `src/medgen/models/factory.py`: Model factory (UNet, DiT, HDiT, UViT, Mamba, WDM)
- `src/medgen/models/dit.py`: DiT model (Scalable Interpolant Transformer)
- `src/medgen/models/dit_blocks.py`: Transformer blocks with adaLN-Zero
- `src/medgen/models/hdit.py`: HDiT (Hierarchical Diffusion Transformer)
- `src/medgen/models/uvit.py`: UViT (Vision Transformer with Skip Connections)
- `src/medgen/models/mamba_diff.py`: LaMamba-Diff (state-space backbone, pixel-space only)
- `src/medgen/models/mamba_blocks.py`: Mamba SSM blocks
- `src/medgen/models/handoff.py`: HandoffWrapper (two-stage low-t/high-t inference)
- `src/medgen/models/embeddings.py`: Patch, timestep, conditioning embeddings
- `src/medgen/models/controlnet.py`: ControlNet for latent diffusion
- `src/medgen/models/autoencoder_dc_3d.py`: 3D DC-AE architecture
- `src/medgen/models/haar_wavelet_3d.py`: 3D Haar wavelet transform
- `src/medgen/models/dcae_structured.py`: DC-AE 1.5 structured latent space
- `src/medgen/models/wrappers/`: Model conditioning wrappers (mode embed, size bin, omega, combined)

### Data
- `src/medgen/data/dataset.py`: NiFTIDataset class
- `src/medgen/data/utils.py`: Shared utilities
  - `binarize_seg(data, threshold=0.5)`: Single source of truth for seg binarization (clamp + threshold). Use for generated/augmented data.
  - `save_nifti(data, path, voxel_size)`: Save numpy array as NIfTI with voxel spacing.
  - `make_binary(image, threshold)`: Simple threshold for ground-truth data (no clamp needed).
  - Slice extraction: `extract_slices_single`, `extract_slices_dual`, `extract_slices_single_with_seg`
  - `merge_sequences()`: Merge multiple MR sequences into single dataset
- `src/medgen/data/loaders/`: Dataloader factory functions
- `src/medgen/data/loaders/common.py`: Shared DataLoader utilities (config, distributed, modality helpers)
  - `MODALITY_KEYS`: Centralized mapping (`'dual' → ['t1_pre', 't1_gd']`, etc.)
  - `get_modality_keys(modality)`: Expand composite modalities to keys
  - `GroupedBatchSampler`: Ensures homogeneous batches for mode embedding
- `src/medgen/data/loaders/datasets.py`: Canonical location for size-bin utilities (`compute_size_bins`, `compute_feret_diameter`, `create_size_bin_maps`, `DEFAULT_BIN_EDGES`)
- `src/medgen/data/loaders/builder_2d.py`: LoaderSpec pattern for standardized 2D dataloader construction
- `src/medgen/data/loaders/unified.py`: Unified dataloader dispatch (routes mode → correct factory)
- `src/medgen/data/loaders/compression_detection.py`: Auto-detect compression model type from checkpoint
- `src/medgen/data/loaders/restoration_3d.py`: Paired (degraded, clean) 3D loader for restoration mode (used by Bridge / IR-SDE / Resfusion)
- `src/medgen/data/lossless_mask_codec.py`: Lossless binary mask encoding to DC-AE latent

### Augmentation
- `src/medgen/augmentation/augmentation.py`: Standard transforms (diffusion/vae)
- `src/medgen/augmentation/score_aug.py`: ScoreAug transforms and omega conditioning
- `src/medgen/augmentation/sda.py`: Shifted Data Augmentation

### Diffusion
- `src/medgen/diffusion/strategies.py`: Strategy base class and registry
- `src/medgen/diffusion/strategy_ddpm.py`: DDPM strategy (noise prediction, discrete timesteps)
- `src/medgen/diffusion/strategy_rflow.py`: RFlow strategy (velocity prediction, continuous timesteps)
- `src/medgen/diffusion/strategy_bridge.py`: Diffusion Bridge for paired restoration (Zhang et al. 2025)
- `src/medgen/diffusion/strategy_irsde.py`: IR-SDE mean-reverting SDE (Luo et al., ICML 2023)
- `src/medgen/diffusion/strategy_resfusion.py`: Resfusion residual noise diffusion (Shi et al., NeurIPS 2024)
- `src/medgen/diffusion/modes.py`: Seg, Bravo, Dual, Multi, SegConditioned, Restoration modes
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
- `src/medgen/pipeline/discriminator_manager.py`: GAN discriminator lifecycle
- `src/medgen/pipeline/training_tricks.py`: Training trick configs (gradient noise, curriculum, etc.)
- `src/medgen/pipeline/results.py`: TrainingStepResult dataclass
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
- `src/medgen/downstream/nnunet/`: nnU-Net v2 integration (convert, train, evaluate, splits). Uses per-experiment isolated preprocessed dirs to prevent split race conditions (see pitfall #83)

### Metrics
- `src/medgen/metrics/unified.py`: UnifiedMetrics (MANDATORY for all trainers)
- `src/medgen/metrics/quality.py`: MS-SSIM, PSNR, LPIPS
- `src/medgen/metrics/seg_metrics.py`: Segmentation-specific metrics (Dice, IoU)
- `src/medgen/metrics/mc_dropout.py`: Monte Carlo dropout for uncertainty estimation
- `src/medgen/metrics/regional/`: Per-tumor regional metrics by size category
- `src/medgen/metrics/generation.py`: KID, CMMD, FID (ResNet50 + BiomedCLIP)
- `src/medgen/metrics/generation_3d.py`: 3D generation metrics + `extract_features_3d_triplanar` (axial+coronal+sagittal feature extraction)
- `src/medgen/metrics/fwd.py`: Fréchet Wavelet Distance (Veeramacheneni et al., ICLR 2025) — domain-agnostic, no pretrained backbone
- `src/medgen/metrics/morphological.py`: Morphological comparison (Wasserstein on tumor volume/feret/spatial/count)
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
- `src/medgen/scripts/eval_diffrs.py`: Evaluate DiffRS rejection sampling (argparse)
- `src/medgen/scripts/eval_compression_fid.py`: Evaluate compression model FID
- `src/medgen/scripts/eval_restart.py`: Evaluate restart sampling
- `src/medgen/scripts/find_optimal_steps.py`: Golden-section search for optimal Euler step count
- `src/medgen/scripts/measure_latent_std.py`: Measure latent space statistics
- `src/medgen/scripts/measure_distinguishability.py`: Measure real vs synthetic distinguishability
- `src/medgen/scripts/compute_pixel_stats.py`: Compute brain-only pixel statistics
- `src/medgen/scripts/recompute_latent_stats.py`: Recompute latent space statistics
- `src/medgen/scripts/lr_finder.py`: Learning rate finder
- `src/medgen/scripts/debug_ldm_roundtrip.py`: Debug LDM encode/decode roundtrip
- `src/medgen/scripts/verify_ldm_pipeline.py`: Verify latent diffusion pipeline
- `src/medgen/scripts/verify_wdm_pipeline.py`: Verify wavelet diffusion pipeline
- `src/medgen/scripts/plot_tumor_detection.py`: Plot tumor detection visualization
- `src/medgen/scripts/visualize_augmentations.py`: Debug augmentation pipelines
- `src/medgen/scripts/train_nnunet.py`: nnU-Net v2 training with TensorBoard (argparse)
- `src/medgen/scripts/eval_nnunet.py`: nnU-Net 5-fold ensemble evaluation (argparse)
- `src/medgen/scripts/find_optimal_freeu.py`: Grid search for optimal FreeU parameters (argparse)
- `src/medgen/scripts/find_optimal_cfg.py`: Grid search for optimal CFG scale (argparse)
- `src/medgen/scripts/synthesize_phema.py`: Post-hoc EMA synthesis sweep (argparse)
- `src/medgen/scripts/eval_time_shift.py`: Evaluate time-shifted sampling (sweep or golden-section search, argparse)
- `src/medgen/scripts/eval_steps_pca.py`: Golden-section search for optimal steps by PCA brain shape error (argparse)
- `src/medgen/scripts/compute_brain_atlas.py`: Compute brain atlas from training masks (argparse)
- `src/medgen/scripts/compute_brain_pca.py`: Compute PCA model from training brain masks (argparse)
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

The actual unified-metrics API lives in `src/medgen/metrics/unified.py`
(verbatim from the source — there is no `pipeline.metrics.unified`
module; classes called `TrainerMetricsConfig` / `LossAccumulator` /
`MetricsLogger` / `LossKey` / `MetricKey` / `TrainerMode` do **not**
exist in this codebase). The real classes are `UnifiedMetrics` and the
helper `SimpleLossAccumulator`:

1. **Instantiate `UnifiedMetrics`** in your trainer's `__init__` (see
   `src/medgen/pipeline/vae_trainer.py` and `dcae_trainer.py` for real
   usage):
   ```python
   from medgen.metrics.unified import UnifiedMetrics

   self.metrics = UnifiedMetrics(
       writer=self.writer,
       trainer_type='vae',  # or 'vqvae', 'dcae', 'diffusion', 'seg'
       device=self.device,
       # ... see UnifiedMetrics.__init__ at metrics/unified.py:159 for the full signature
   )
   ```

2. **Use in training loop** (`UnifiedMetrics.update_loss(key, value, phase)`
   is at `metrics/unified.py:535`):
   ```python
   self.metrics.update_loss('mse', loss.item(), phase='train')
   self.metrics.update_psnr(pred, gt)  # logged via PSNR accumulator
   ```

3. **For per-trainer specifics** (regional metrics, codebook tracking,
   timestep buckets), refer to the existing trainers as templates —
   each calls into `UnifiedMetrics` for accumulation and logging
   without rolling its own TensorBoard writer logic.

### Components

The unified-metrics module exposes the following at
`src/medgen/metrics/unified.py`:

- **`UnifiedMetrics`** (line 114): the public class. Takes a `writer`,
  `trainer_type` (one of `'vae'`, `'vqvae'`, `'dcae'`, `'diffusion'`,
  `'seg'`), a `device`, plus optional codebook/regional config. Exposes
  `update_loss(key, value, phase)`, `update_psnr/lpips/msssim/msssim_3d/dice/iou`,
  `log_seg_training`, `log_seg_validation`, `log_lr`, etc.
- **`SimpleLossAccumulator`** (line 57): epoch-loss accumulator helper
  used internally by `UnifiedMetrics`.

There are no separate `TrainerMode` / `LossKey` / `MetricKey` enums — loss
and metric keys are passed as plain strings (e.g. `update_loss('mse', ...,
phase='train')`). To add a new loss type for a new trainer, just call
`update_loss('your_key', value, phase=...)` with whatever string you like;
the metric will be logged under `Loss/your_key_train` (or `_val`).

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
- **FwD** (Fréchet Wavelet Distance, `metrics/fwd.py`): Per-frequency-band Fréchet distance using wavelet packet decomposition (Haar, configurable). Computes mean/covariance per packet, then Fréchet (Gaussian divergence). Returns overall + per-band scores. **Domain-agnostic** (no pretrained backbone — robust to domain shift). For 3D, slice-wise (axial). Reference: Veeramacheneni et al., ICLR 2025 (arXiv:2312.15289). Used in restoration eval logging.

**Feature Extractors:**
- **ResNet50Features**: ImageNet pretrained (default) or **RadImageNet** (medical domain — `feature_extractor=radimagenet_resnet50`). Uses `torch.compile(mode="reduce-overhead")` and AMP (bfloat16).
- **BiomedCLIPFeatures**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. Uses `torch.compile` and AMP.
- **Triplanar 3D features** (`extract_features_3d_triplanar` in `generation_3d.py`): Extracts axial + coronal + sagittal planes for 3D volumes, then runs each through a 2D feature extractor and concatenates. Captures orientation-sensitive structure that pure axial-slice extraction misses.

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

# Visualize augmentations (no augment_type key — script visualizes BOTH pipelines).
# Real Hydra keys per configs/visualize_augmentations.yaml: synthetic, modality, n_samples, image_size, output_dir
python -m medgen.scripts.visualize_augmentations modality=bravo n_samples=8

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
./IDUN/submit_prefer_h100.sh IDUN/train/diffusion/exp16_rflow_128_bs32.slurm

# Custom timeout (30 min)
./IDUN/submit_prefer_h100.sh IDUN/train/diffusion/exp16_rflow_128_bs32.slurm 1800

# Run in background
./IDUN/submit_prefer_h100.sh IDUN/train/diffusion/exp16_rflow_128_bs32.slurm --bg
```

Background mode:
- Logs to `/tmp/submit_h100_$$.log`
- Check with: `tail -f /tmp/submit_h100_$$.log`
- Kill with: `kill <PID>` (shown on launch)
