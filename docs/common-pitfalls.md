# Common Pitfalls

Reference this file with `@docs/common-pitfalls.md` when debugging issues.

## 1. Mode vs Strategy Confusion
- **Mode** = WHAT to generate (seg, bravo, dual)
- **Strategy** = HOW to denoise (ddpm, rflow)

## 2. VAE Dual vs Diffusion Dual
- **VAE dual**: 2 channels (t1_pre, t1_gd) - NO seg mask
- **Diffusion dual**: 3 channels (t1_pre, t1_gd, seg) - HAS seg mask

## 3. VAE Dataloader
- **ALWAYS** use `create_vae_dataloader()` for VAE training
- NEVER use `create_dual_image_dataloader()` for VAE
- `train_compression.py` handles this automatically

## 4. in_channels Override
- `train_compression.py` overrides `cfg.mode.in_channels` for VAE
- Don't rely on mode config files for VAE channel counts

## 5. Progressive vs Regular VAE
- **Regular**: Single resolution, single modality
- **Progressive**: Multi-resolution (64→128→256), multi-modality mixed

## 6. Pixel vs Latent Space
- **Pixel**: Direct image diffusion, more VRAM, compiled forward works
- **Latent**: VAE-compressed, less VRAM, compiled forward disabled

## 7. Channel Counts
- Always check mode config for in_channels/out_channels
- Dual mode: model outputs 2 channels (both images), not 3

## 8. LR Finder VAE
- Uses `create_vae_dataloader()` for proper batch format
- Disables GAN for stable LR finding

## 9. Seg Mode Perceptual Loss
- **Automatically disabled** for seg mode diffusion
- Pretrained ImageNet features don't apply to binary masks
- You'll see log message: "Seg mode: perceptual loss disabled"

## 10. Regional Losses Work for All Modes
- **Conditional modes** (bravo, dual): Uses conditioning seg mask
- **Seg mode**: Uses ground truth seg as the mask itself
- Enable with `training.logging.regional_losses=true`

## 11. Worst Batch Location
- **Validation**: Logged as `Validation/worst_batch` at each `figure_interval`
- **Test**: Logged as `test_best/worst_batch` and `test_latest/worst_batch`
- NOT tracked during training (only validation and test)

## 12. VAE Validation Needs Seg for Regional Metrics
- VAE validation dataloader loads seg separately for regional loss tracking
- Training dataloader does NOT include seg (not needed for training loss)
- If `regional_losses: true`, validation will track tumor vs background error

## 13. TensorBoard Metric Names
- `Loss/` prefix: Actual loss values used in training
- `Validation/` prefix: Quality metrics (SSIM, PSNR, LPIPS)
- `training/` prefix: Training diagnostics (grad norms, timestep breakdown)
- `test_best/` or `test_latest/` prefix: Test evaluation results

## 14. Timestep Region Heatmap
- Only meaningful for diffusion (not VAE)
- Shows 2D heatmap: X-axis = timestep bins, Y-axis = region (tumor/background)
- Helps identify if certain timesteps struggle with tumor regions

## 15. Config File Locations
- **Diffusion defaults**: `configs/training/default.yaml`
- **VAE overrides**: `configs/vae.yaml` (logging section)
- Some options are N/A for VAE (e.g., `timestep_losses`, `intermediate_steps`)

## 16. LPIPS Model Loading
- First validation may be slow (downloads AlexNet if not cached)
- Set `training.logging.lpips=false` to disable if not needed

## 17. Multi-GPU and Logging
- Only main process (rank 0) writes to TensorBoard
- All processes accumulate metrics, but logging is centralized
- Use `is_main_process` checks in custom code

## 18. Compiled Model and Validation
- `torch.compiler.cudagraph_mark_step_begin()` called before validation
- Prevents tensor caching issues with compiled perceptual loss
- Don't remove this call if using `training.use_compile=true`

## 19. torch.compile + DDP Incompatibility
- `compile_fused_forward` is **auto-disabled** when using DDP (multi-GPU)
- Error: `Dynamo does not know how to trace method 'set_runtime_stats_and_log' of class 'Logger'`
- DDP wrapper has internal logging that `torch.compile` can't trace
- You'll see: "Disabled compiled fused forward for DDP (multi-GPU)"
- Other compiled components (perceptual loss, VAE) still work

## 20. Augmentation Type Mismatch
- **Diffusion** should use `augment_type: diffusion` (conservative)
- **VAE** should use `augment_type: vae` (aggressive)
- Using aggressive augmentation for diffusion trains model to generate distorted images
- Dataloaders auto-select based on type, but can be overridden

## 21. Batch Augmentation Only for VAE
- Mixup/CutMix only work with VAE training (no seg mask in batch)
- Don't enable `batch_augment` for diffusion training
- Collate function modifies raw image tensors before stacking

## 22. DDP Validation Batch Size
- Validation runs on rank 0 only, with reduced batch size
- `world_size` is auto-passed to validation dataloaders
- Without this, 256px attention OOMs on single GPU (needs 16GB per attention head)
- Error: `Tried to allocate 16.00 GiB` during validation

## 23. Progressive VAE: Don't Include Seg in Modalities
- Seg masks are binary (0/1), other modalities are continuous (0-1)
- Mixing them causes NaN losses at small resolutions (64x64)
- Use: `image_keys: [bravo, flair, t1_pre, t1_gd]` (NO seg)
- Train seg separately if needed

## 24. Progressive VAE Memory Requirements
- Phase 3 (256x256) with batch_size=16 needs ~64GB host memory
- SLURM: `--mem=64G` minimum for full progressive training
- OOM kill at phase 3 = insufficient host RAM, not GPU memory

## 25. ScoreAug Omega Conditioning
- **Required** for rotation/flip transforms (non-invertible on noise)
- Without omega: model can "cheat" by detecting rotation from noise pattern
- Omega wraps model's time embedding with learned conditioning
- Enable: `training.score_aug.use_omega_conditioning=true`

## 26. ScoreAug vs Standard Augmentation
- **Don't combine both** - ScoreAug should have standard augmentation disabled
- Set `training.augment=false` when using ScoreAug
- ScoreAug augments noisy data, standard augments clean data - different purposes

## 27. ScoreAug Perceptual Loss
- Skipped for non-invertible transforms (translation, cutout)
- Can't compare to original space after transform
- Only applied when transform is invertible (rotation, flip with omega)

## 28. Regional Metrics: Per-Tumor with Feret Diameter
- Uses connected components to identify individual tumors
- Measures longest edge-to-edge distance (Feret diameter), not circular approximation
- Classifies using RANO-BM clinical thresholds (tiny <10mm, small 10-20mm, medium 20-30mm, large >30mm)
- All metrics are pixel-weighted (larger tumors contribute proportionally more)

## 29. Per-Modality Validation Metrics
- Both VAE and diffusion trainers log per-channel metrics for dual/multi modes
- `Validation/PSNR_t1_pre`, `Validation/LPIPS_t1_gd`, etc.
- For multi-modality training, also logs per-modality regional metrics
- `per_modality_val_loaders` attribute set by train.py/train_compression.py

## 30. Worst Batch Visualization Requirements
- Worst batch only tracked from **full-sized batches** (not last partial batch)
- Ensures consistent 8-sample grid visualization
- For dual mode, stores both channels as dict (not just first channel)
- Keys are `original`/`generated` (not `images`/`predicted`)

## 31. Regional Metrics Prefix Convention
- All regional metrics use `regional/` prefix (not `tumor/`)
- Per-modality regional: `regional_{modality}/` (e.g., `regional_t1_pre/tumor_loss`)
- Consistent across both VAE and diffusion trainers

## 32. 3D VAE KL Loss Bug (Fixed Jan 2026)
- **Previous bug**: 3D VAE KL loss was ~80,000× weaker than 2D
- Used `.mean()` over all dimensions instead of `sum()` over spatial + `mean()` over batch
- **Impact**: 3D VAE experiments before this fix had near-zero KL regularization
- **Action required**: Re-run 3D VAE experiments after updating code

## 33. 3D Loaders Don't Support Augmentation
- 2D loaders have `augment: bool = True` parameter
- 3D loaders (vae_3d.py) do NOT support data augmentation
- Reason: Albumentations is 2D-only; 3D requires MONAI transforms
- Workarounds: TTA during inference, multiple training seeds, or implement MONAI 3D transforms

## 34. Regional Metrics Use Pixel-Weighted Aggregation
- Both tumor AND background use pixel-weighted averaging (fixed Jan 2026)
- Previous bug: Background was sample-weighted, tumor was pixel-weighted
- This made tumor/background ratio meaningless
- All metrics now consistent: `error_sum / pixel_count`

## 35. Metric Caches Persist Across Runs
- MS-SSIM and LPIPS use cached instances for performance
- Call `clear_metric_caches()` when:
  - Running multiple training runs in same process
  - Switching GPU devices
  - Memory cleanup needed
- Import: `from medgen.pipeline.metrics import clear_metric_caches`

## 36. Segmentation Threshold Constants
- Ground truth masks: `BINARY_THRESHOLD_GT = 0.01` (very low, preserves all positive pixels)
- Generated masks: `BINARY_THRESHOLD_GEN = 0.1` (higher, filters noise)
- Always use constants from `medgen.core.constants`, not hardcoded values
- Previous bug: `multi_diffusion.py` used hardcoded `0.5` instead of constant

## 37. 3D LPIPS Performance
- 3D LPIPS now uses batched slice processing (fixed Jan 2026)
- Previous: 160 sequential forward passes per volume
- Now: Chunks of 32 slices processed together
- 10-100× speedup for 3D validation metrics

## 38. LatentSpace.encode() Shape Validation
- `LatentSpace.encode()` now validates input is 4D `[B, C, H, W]`
- Raises `ValueError` with helpful message if wrong dimensions
- Prevents cryptic errors from VAE encoder

## 39. Empty Validation Spurious Best Checkpoint (Fixed Jan 2026)
- **Previous bug**: Empty validation returned metrics with `loss=0.0`
- `0.0 < float('inf')` triggered "best" checkpoint save
- **Impact**: Empty validation folder caused spurious "best" model saves
- **Fix**: Added guard `if val_loss > 0 and val_loss < self.best_loss`
- Real losses are never exactly 0.0, so this safely filters empty validation

## 40. Mode Embedding Requires Same-Modality Batches (Fixed Jan 2026)
- **Previous bug**: `encode_mode_id()` only used `mode_id[0]`, ignoring rest of batch
- Multi-modality training pools ALL modalities, shuffle creates mixed batches
- Mixed batch `[0, 1, 2, 0, 3]` would apply mode 0's conditioning to all samples
- **Fix**: Added validation to ensure all mode_ids in batch are identical
- **Impact**: If `use_mode_embedding=true`, batches MUST have same modality
- Raises clear error: "Consider using a GroupedSampler or disabling shuffle"

## 41. BF16 Precision in Compiled Forward Functions (Fixed Jan 2026)
- **Previous bug**: Compiled forward functions (`_forward_single`, `_forward_dual`) computed MSE and perceptual loss in BF16
- Non-compiled fallback paths used FP32 (via explicit `.float()` casts)
- **Symptoms**:
  - Training loss consistently ~15-20% lower than expected
  - Occasional spikes to "correct" baseline when torch.compile recompiled
  - Results not reproducible between runs
- **Root cause**: Missing `.float()` casts in compiled functions:
  ```python
  # WRONG - BF16 under autocast
  mse_loss = ((prediction - target) ** 2).mean()

  # CORRECT - explicit FP32
  mse_loss = ((prediction.float() - target.float()) ** 2).mean()
  ```
- **Impact**: Diffusion experiments between Jan 3-7, 2026 need re-running
- **Rule**: Always cast to FP32 before MSE/perceptual loss computation
- Perceptual networks (SqueezeNet, VGG) are pretrained in FP32 - unstable with BF16

## 42. Validation RNG Consumption (Fixed Jan 2026)

**Problem**: Validation code consumed global RNG, causing training to follow a different random trajectory after each validation.

**Root cause**: `compute_validation_losses` and `_compute_volume_3d_msssim` called `torch.randn_like()` without preserving RNG state. Each validation consumed ~500K random numbers, shifting the global RNG sequence.

**Symptoms observed**:
- Training loss ~8% smoother (reduced batch-to-batch variation)
- Loss jumps at validation intervals (every 25 epochs)
- Training diverged from baseline when comparing commits
- RNG markers identical until first validation, then completely different

**What this bug did NOT do**:
- Break model convergence
- Reduce final model quality
- Corrupt learned features

This was a **reproducibility bug**, not a training quality bug. Models trained with this bug are fine - they just followed a different (but equally valid) random trajectory.

**Fix**: Save/restore RNG state around the ENTIRE `compute_validation_losses` method:
```python
def compute_validation_losses(self, epoch):
    if self.val_loader is None:
        return

    # Save RNG state before validation
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None

    # ... validation code with torch.randn_like() ...

    # Restore RNG state - training continues unaffected
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, self.device)
```

**Rules for future code**:
1. Any code that runs between training steps and uses `torch.randn*`, `torch.rand*`, or `torch.randint*` must preserve RNG state
2. This includes: validation, logging, visualization, metric computation
3. Test code can also corrupt RNG if called mid-training (e.g., `_compute_volume_3d_msssim` for test evaluation)
4. Use the pattern: save state → do work → restore state

## 44. Mode Embedding Requires Homogeneous Batches
**Problem**: Mode embedding for multi-modality diffusion (`mode=multi`) expects all samples in a batch to have the same mode_id.

**Root cause**: `encode_mode_id()` uses `mode_id[0]` for the entire batch. Mixed batches apply the wrong mode embedding to most samples.

**Fix**: Use `GroupedBatchSampler` from `medgen.data.loaders.common`:
- Groups samples by mode_id
- Shuffles groups and samples within groups
- Each batch contains only one modality

**Symptoms**: Training instability, inconsistent reconstruction quality per modality.

## 45. Seg Compression Mode vs Seg Mode
**Don't confuse**:
- `mode=seg`: Regular seg mask training (L1 + perceptual loss)
- `mode=seg_compression` + `dcae.seg_mode=true`: DC-AE seg compression (BCE + Dice + Boundary loss)

**Key differences**:
- Seg compression uses specialized dataloaders (`seg_compression.py`)
- Metrics are Dice/IoU instead of PSNR/LPIPS/MS-SSIM
- No perceptual loss (pretrained features don't apply to binary masks)
- No GAN in Phase 1-2

## 46. Lossless Mask Codec Format Must Match DC-AE
When using `encode_mask_lossless()` / `decode_mask_lossless()`, the format parameter must match the DC-AE compression level:

| DC-AE Config | Codec Format | Latent Shape |
|--------------|--------------|--------------|
| f32 (default) | 'f32' | `[32, 8, 8]` |
| f64 | 'f64' | `[128, 4, 4]` |
| f128 | 'f128' | `[512, 2, 2]` |

**Wrong format**: Shape mismatch error when concatenating with image latents.

## 47. FiLM Mode Embedding Strategy
The `film` strategy for mode embedding uses learned scale and shift (multiplicative + additive) instead of pure additive embedding:
```
output = scale * x + shift
```
This provides more expressive modulation but may require different learning rates.

## 48. Reclaim WSL2 Disk Space
WSL2's ext4.vhdx grows but doesn't shrink automatically. To reclaim space:

1. **Shut down WSL and stop the service:**
   ```powershell
   wsl --shutdown
   Stop-Service -Name "WslService" -Force
   sc.exe config WslService start=disabled
   ```

2. **Compact the VHDX:**
   ```powershell
   diskpart
   ```
   Then:
   ```
   select vdisk file="W:\WSL\ext4.vhdx"
   compact vdisk
   detach vdisk
   exit
   ```

3. **Re-enable WSL service:**
   ```powershell
   sc.exe config WslService start=demand
   ```

Note: Adjust the VHDX path to match your WSL installation location.

## 49. New Trainers MUST Use Unified Metrics System
- **DO NOT** implement custom TensorBoard logging in new trainers
- **USE** `TrainerMetricsConfig`, `LossAccumulator`, and `MetricsLogger` from `unified.py`
- This ensures consistent metric names across all trainers (no tag drift)
- See `@docs/architecture.md` → "Unified Metrics System" for implementation guide
- If adding new loss types, extend `LossKey` and `_TRAIN_LOSS_TAGS` in `unified.py`

## 50. Generation Metrics Dependencies and Caching
When using `training.generation_metrics.enabled=true`:

**Dependencies required:**
- `torchvision` - for ResNet50 feature extraction (KID/FID)
- `open_clip_torch` - for BiomedCLIP model loading (CMMD)

**First-run behavior:**
- ResNet50 (ImageNet pretrained) loads from torchvision cache
- BiomedCLIP model (~1GB) downloads on first use - requires internet
- Both extractors compiled with `torch.compile(mode="reduce-overhead")` - warmup needed
- Features cached to `cache_dir` (default: `.cache/generation_features`)
- Subsequent runs load cached features (experiment_id-based)

**Memory considerations:**
- Adds ~30s overhead per epoch for 100-sample generation
- Feature extractors loaded lazily (only when first needed)
- Batched generation uses `training.batch_size` for `torch.compile` consistency
- Sample counts rounded up to full batches (100 → 112 with batch_size=16)
- CPU offload after each batch to avoid OOM with CUDA graphs

**Cache management:**
- Cache key is `{experiment_id}_reference_features.pt`
- Delete cache file to force re-extraction after dataset changes
- Cached features are PyTorch `.pt` files (fast load)
- Backward compatible with old cache format (inception → resnet)

## 51. Clean Regularization Techniques Are Independent
When using the new regularization techniques (constant LR, gradient noise, curriculum, timestep jitter, noise augmentation, feature perturbation, self-conditioning):

**Test individually first**: Each technique addresses overfitting differently. Combining multiple without testing can cause:
- Conflicting regularization effects
- Destabilized training
- Worse performance than single technique

**Recommended approach**:
1. Run baseline (all disabled)
2. Test each technique individually
3. Combine only techniques that show improvement
4. Monitor validation curves for signs of under/over-regularization

**Known interactions**:
- `gradient_noise` + `noise_augmentation`: Both add noise, may over-regularize
- `curriculum` + `timestep_jitter`: Both affect timestep sampling, may conflict
- `self_conditioning`: Adds ~50% compute overhead (two forward passes)

**Config locations**: All under `training.*` in `configs/training/default.yaml`

## 52. Self-Conditioning Increases Compute Cost
`training.self_conditioning.enabled=true` runs the model twice per batch:
- First pass: No gradient, get prediction P1
- Second pass: Full gradient, compute consistency loss MSE(P2, P1)

**Overhead**: ~50% more compute per training step (not 2x because first pass has no grad)

**When to use**: Dataset severely overfitting and other techniques insufficient.

## 53. Feature Perturbation Layer Names
The `training.feature_perturbation.layers` config accepts:
- `"encoder"` - All down blocks
- `"mid"` - Mid block only (bottleneck)
- `"decoder"` - All up blocks
- List: `["mid", "decoder"]` - Specific combination

**MONAI UNet structure**: `down_blocks`, `mid_block`, `up_blocks`

**Default**: `["mid"]` - Perturbs only the bottleneck (most regularization with least disruption)

## 54. SDA and ScoreAug Are Mutually Exclusive
**Problem**: Using both SDA (Shifted Data Augmentation) and ScoreAug together causes conflicting augmentation patterns.

**How they differ**:
- **ScoreAug**: Augments noisy data (after noise addition), requires omega conditioning
- **SDA**: Augments clean data (before noise addition) with shifted timesteps

**Rule**: Only enable ONE of these techniques:
```yaml
# WRONG - both enabled
training.score_aug.enabled: true
training.sda.enabled: true

# CORRECT - choose one
training.score_aug.enabled: true
training.sda.enabled: false
# OR
training.score_aug.enabled: false
training.sda.enabled: true
```

## 55. Augmented Diffusion Only Works with Learned Latent Spaces
**Problem**: `training.augmented_diffusion.enabled=true` only works with VAE/VQ-VAE/DC-AE latent spaces. It is silently ignored for pixel space, wavelet space, and SpaceToDepth.

**Root cause**: Augmented diffusion (DC-AE 1.5) requires channel masking of **learned** VAE latents where channels encode increasingly fine features. Fixed transforms like Haar wavelets have semantically tied subbands (LLL, LLH, etc.) — randomly masking them destroys the frequency decomposition.

**Symptoms**: Setting `augmented_diffusion.enabled=true` without a `LatentSpace` has no effect (warning logged).

**Fix**: Only use with VAE/VQ-VAE/DC-AE latent diffusion:
```bash
# WRONG - pixel space (augmented_diffusion ignored)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augmented_diffusion.enabled=true

# WRONG - wavelet space (augmented_diffusion ignored, warning logged)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    wavelet.enabled=true \
    training.augmented_diffusion.enabled=true

# CORRECT - learned latent space with compression checkpoint
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    training.augmented_diffusion.enabled=true
```

## 56. DC-AE 1.5 Structured Latent Minimum Channel Requirement
**Problem**: DC-AE 1.5 structured latent space is suboptimal for small channel counts (c=32).

**Root cause**: The channel masking progression [16, 20, 24, ..., c] creates too few options for f32 (32 channels). Standard DC-AE performs better.

**Recommendation**:
- **f32 (32 channels)**: Use standard DC-AE, NOT structured latent
- **f64 (128 channels)**: Structured latent works well
- **f128 (512 channels)**: Structured latent works well

```yaml
# NOT RECOMMENDED for f32
dcae: f32
dcae.structured_latent.enabled: true

# RECOMMENDED for f64/f128
dcae: f64
dcae.structured_latent.enabled: true
```

## 57. Mode Embedding Now Supports Per-Sample Encoding
**Update to Pitfall 40/44**: The mode embedding system now supports per-sample encoding via `ModeEmbedModelWrapper`.

**Previous limitation**: All samples in batch required identical mode_id.

**Current status**: Mixed batches technically work with per-sample encoding.

**However**, `GroupedBatchSampler` is still **recommended** for:
- Training stability (empirically better)
- Reproducibility across runs
- Consistent batch statistics

```python
# Per-sample encoding now works (but not recommended)
mode_ids = [0, 1, 2, 0, 3]  # Mixed batch
# Each sample gets its own one-hot encoding

# GroupedBatchSampler still recommended for stability
mode_ids = [0, 0, 0, 0, 0]  # Homogeneous batch (preferred)
```

## 58. DC-AE Phase 3 Requires Different Optimizer Betas
**Problem**: DC-AE Phase 3 GAN training is unstable with default optimizer betas.

**Root cause**: Default AdamW betas `[0.9, 0.999]` cause GAN training instability. GAN training requires `[0.5, 0.9]`.

**Symptoms**: Discriminator loss oscillates wildly, generator quality degrades.

**Fix**: Always use `[0.5, 0.9]` for Phase 3:
```bash
# WRONG - default betas (unstable GAN)
python -m medgen.scripts.train_compression --config-name=dcae training.phase=3 dcae.adv_weight=0.1

# CORRECT - GAN-optimized betas
python -m medgen.scripts.train_compression --config-name=dcae training.phase=3 dcae.adv_weight=0.1 \
    'training.optimizer.betas=[0.5,0.9]'
```

**Reference**: This is standard practice for GAN training (see DCGAN, StyleGAN papers).

## 59. Hydra Optional Configs - Don't Add Duplicates
**Problem**: Adding `+latent=default` on command line when config already has `optional latent: default` causes "appears more than once" error.

**Root cause**: The `optional latent: default` in the defaults list ALREADY loads the config if the file exists. Adding `+latent=default` duplicates it.

**Error examples**:
```
# If you add +latent=default when optional already exists:
latent appears more than once in the final defaults list

# If the optional config file doesn't exist (rare):
Could not override 'latent.enabled'.
Key 'latent' is not in struct
```

**Fix**: Just override the values directly - the optional config is already loaded:
```bash
# CORRECT - optional config is already loaded, just override values
python -m medgen.scripts.train model.spatial_dims=3 \
    latent.enabled=true \
    latent.compression_checkpoint=path/to/checkpoint.pt \
    controlnet.enabled=true

# WRONG - causes "appears more than once" error
python -m medgen.scripts.train model.spatial_dims=3 \
    +latent=default \
    latent.enabled=true
```

**When to use + prefix**: Only if the config group is NOT in the defaults list at all (no `optional` either).

## 60. Generation Metrics Type Check for Tensor vs NumPy
**Problem**: `set_fixed_conditioning()` fails with "expected np.ndarray (got Tensor)".

**Root cause**: The check `hasattr(x, '__array__')` is True for BOTH numpy arrays AND torch.Tensor (tensors implement `__array__` for numpy interop). So it incorrectly tries `torch.from_numpy()` on a tensor.

**Fixed**: Now uses explicit `isinstance()` checks:
```python
# WRONG - __array__ exists on both types
if hasattr(images, '__array__'):
    images = torch.from_numpy(images)  # Fails if already tensor!

# CORRECT - explicit type checks
if isinstance(images, torch.Tensor):
    images = images.float()
elif isinstance(images, np.ndarray):
    images = torch.from_numpy(images).float()
```

## 61. seg_conditioned Mode Returns (seg, size_bins) Not (image, seg)
**Problem**: Code that expects `(image, seg)` tuples fails with `seg_conditioned` mode.

**Root cause**: Most diffusion modes return `(images, seg)` tuples where both are 3D tensors `[C, H, W]`. But `seg_conditioned` mode returns `(seg, size_bins)` where:
- `seg` is the mask to generate: shape `[1, H, W]`
- `size_bins` is 1D conditioning vector: shape `[num_bins]`

**Symptom**: `RuntimeError: Tensors must have same number of dimensions: got 3 and 1`

**Fix**: Check if second element is 1D before concatenating:
```python
if second.dim() == 1 and first.dim() == 3:
    # seg_conditioned mode: first element IS the seg mask
    tensor = first
else:
    # Standard mode: concatenate (images, seg)
    tensor = torch.cat([first, second], dim=0)
```

**Mode data formats**:
| Mode | Dataset Return | Shapes |
|------|----------------|--------|
| bravo | `(image, seg)` | `[1,H,W]`, `[1,H,W]` |
| dual | `(images, seg)` | `[2,H,W]`, `[1,H,W]` |
| seg_conditioned | `(seg, size_bins)` | `[1,H,W]`, `[num_bins]` |

## 62. Generation Metrics Auto-Disabled for Seg Modes
**Behavior**: Generation metrics (KID, CMMD, FID) are automatically disabled for `seg` and `seg_conditioned` modes.

**Reason**: Generation metrics use image feature extractors (ResNet50, BiomedCLIP) designed for natural/medical images. Binary segmentation masks produce meaningless features. Additionally, `seg_conditioned` uses `size_bins` as conditioning, not seg masks, so the generation code can't properly condition the model.

**Log message**: `"seg_conditioned mode: generation metrics disabled (binary masks)"`

**If you need quality metrics for seg modes**, consider:
- Dice/IoU between generated and real masks
- Visual inspection
- Downstream task performance

## 63. FLOPs Measurement Caused OOM with torch.compile
**Problem**: Training at 256x256 with batch_size=16 caused OOM during initialization, before training even started.

**Root cause**: The FLOPs measurement function did a forward pass with **full batch size** to measure model FLOPs. When torch.compile is enabled, this first forward pass triggers compilation which:
- Traces execution for specific tensor shapes
- Creates intermediate computation graphs
- Uses massive memory during tracing (much more than inference)

With batch_size=16 at 256x256, compilation memory spike exceeded 80GB.

**Error**: `torch.OutOfMemoryError: Tried to allocate 16.00 GiB` (67GB already allocated)

**Fix applied**: Changed `_measure_model_flops()` to use batch_size=1 for the forward pass:
```python
# Before: used full batch
model_input = self.mode.format_model_input(noisy_images, labels_dict)

# After: slice to batch_size=1
images = images[:1]
model_input = self.mode.format_model_input(noisy_images, labels_dict)
model_input = model_input[:1]  # Use single sample for FLOPs
```

**Additional memory optimization**: PerceptualLoss (ResNet50) is now skipped when `perceptual_weight=0` (seg/seg_conditioned modes), saving ~200MB.

**Result**: batch_size=16 at 256x256 now works on single 80GB GPU for seg_conditioned mode.


## 64. Min-SNR Weighting is DDPM-Only
**Problem**: Min-SNR loss weighting (`use_min_snr=true`) was available for all strategies, but it only makes theoretical sense for DDPM.

**Root cause**: Min-SNR uses the formula `SNR = alpha_bar / (1 - alpha_bar)` where `alpha_bar` comes from DDPM's noise schedule. RFlow uses linear interpolation `x_t = (1-t)*x_0 + t*ε` which doesn't have the same SNR concept.

**Previous behavior**: Code had a fallback for RFlow using `(1-t)/t` as a heuristic approximation, but this has no theoretical basis and may produce suboptimal results.

**Fix applied**: Added guards in both `DiffusionTrainer` and `Diffusion3DTrainer` that auto-disable Min-SNR with a warning when `strategy=rflow`:
```python
if self.use_min_snr and self.strategy_name == 'rflow':
    warnings.warn(
        "Min-SNR weighting is DDPM-specific and has no theoretical basis for RFlow. "
        "Disabling Min-SNR for RFlow training.",
        UserWarning,
    )
    self.use_min_snr = False
```

**Warning message**: `"Min-SNR weighting is DDPM-specific and has no theoretical basis for RFlow. Disabling Min-SNR for RFlow training."`

**Config also updated** (`configs/training/default.yaml`) with note: "DDPM ONLY - auto-disabled for RFlow".

## 65. RFlow Uses Continuous Timesteps by Default
**Change (Jan 2026)**: RFlow now uses continuous timesteps (`use_discrete_timesteps: false`) by default.

**Previous behavior**: Discrete integer timesteps (0, 1, 2, ..., 999)
**New behavior**: Continuous float timesteps (0.0 to 1000.0)

**Why**: The original Rectified Flow paper uses continuous t ∈ [0, 1]. Continuous timesteps are more faithful to the RFlow formulation. Sinusoidal embeddings in both UNet and SiT handle floats natively.

**TensorBoard format change**: Timestep bins now use normalized format:
- Old: `Timestep/0-9`, `Timestep/10-19`, ..., `Timestep/90-99`
- New: `Timestep/0.0-0.1`, `Timestep/0.1-0.2`, ..., `Timestep/0.9-1.0`

**Config options** (`configs/strategy/rflow.yaml`):
```yaml
use_discrete_timesteps: false     # Continuous (default)
sample_method: logit-normal       # Biased toward middle timesteps
use_timestep_transform: true      # Resolution-based adjustment
num_train_timesteps: 1000         # Total timesteps
```

**To revert to discrete**: Set `strategy.use_discrete_timesteps=true`

## 66. Perceptual Loss Disabled by Default for Diffusion
**Change (Jan 2026)**: Perceptual loss is now disabled by default for diffusion training.

**Previous**: `perceptual_weight: 0.001`
**New**: `perceptual_weight: 0.0`

**Why**:
- Perceptual loss adds compute overhead with minimal benefit for diffusion
- Saves ~200MB GPU memory (no ResNet50 loading)
- 3D diffusion already had this disabled

**To re-enable**: Set `training.perceptual_weight=0.001` on command line or in config.

**Log message when disabled**: `"Perceptual loss disabled (perceptual_weight=0), skipping ResNet50 loading"`

## 67. Data Augmentation Disabled by Default for Diffusion
**Change (Jan 2026)**: Data augmentation is now disabled by default for diffusion training.

**Previous**: `augment: true`
**New**: `augment: false`

**Why**:
- Standard augmentation can leak patterns into generated images
- ScoreAug or SDA are better alternatives when regularization is needed
- Most diffusion experiments showed better results without standard augmentation

**To re-enable**: Set `training.augment=true` on command line or in config.

## 68. RFlow Timestep Convention: t=0 is Clean, t=1 is Noise
**Important**: RFlow uses the convention where `t=0` is the clean image and `t=1` is pure noise.

**Interpolation formula**: `x_t = (1 - t) * x_0 + t * noise`
- At `t=0`: `x_t = x_0` (clean image)
- At `t=1`: `x_t = noise` (pure noise)

**Inference direction**: Goes from `t=1000` down to `t=0` (noise → clean).

**Velocity target**: `v = x_0 - noise` (points from noise toward clean data).

**Euler integration**: `x_{t-dt} = x_t + dt * v` (ADDITION, not subtraction).

## 69. Velocity MSE vs Reconstruction MSE - Know the Difference
**Problem**: Confusing what different MSE metrics measure in RFlow training.

**Two types of MSE in diffusion**:

| Metric Type | Formula | What It Measures |
|-------------|---------|------------------|
| **Velocity MSE** | `MSE(pred, x_0 - noise)` | Training loss - how well model predicts the velocity |
| **Reconstruction MSE** | `MSE(predicted_clean, x_0)` | Output quality - how close generated image is to original |

**Where each is used**:
- `Loss/MSE_train`, `Loss/MSE_val`: Velocity MSE (training objective)
- `Timestep/0.0-0.1`, etc.: Velocity MSE (matches training loss)
- `regional/tumor_loss`, etc.: Reconstruction MSE (measures output quality per region)

**Expected velocity MSE pattern by timestep**:
- `t ≈ 0` (clean): **HIGH loss** - Model sees nearly-clean image, can't detect noise direction
- `t ≈ 1` (noisy): **LOW loss** - Model sees pure noise, can learn to predict velocity toward data

This is the OPPOSITE of what you might expect. Don't panic if early timesteps show higher loss.

**Fix applied (Jan 2026)**: Timestep loss tracking changed from reconstruction MSE to velocity MSE in both 2D and 3D trainers. This ensures `Timestep/t_X.X` metrics match the actual training loss.

## 70. Regional Losses Use Reconstruction MSE (Intentionally)
**Why regional losses don't use velocity MSE**:

Regional metrics (`regional/tumor_loss`, `regional/background_loss`, etc.) intentionally use reconstruction MSE because:
1. They measure **output quality** per spatial region, not training loss
2. Velocity error is not spatially interpretable (velocity points in latent direction, not image space)
3. Users want to know "how good is the reconstruction in tumor regions" not "how well did the model predict velocity in tumor regions"

**Summary of metric types**:

| Metric Category | Uses | Why |
|-----------------|------|-----|
| Training loss | Velocity MSE | What model optimizes |
| Timestep breakdown | Velocity MSE | Decomposition of training loss |
| Regional quality | Reconstruction MSE | Spatial output quality |
| Validation metrics (PSNR, SSIM) | Reconstruction | Output quality |

## 71. CUDA Graph Overwrites with CFG + torch.compile
**Problem**: When using classifier-free guidance (CFG) with `torch.compile`, the first model output gets overwritten before it's used.

**Error**: `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`

**Root cause**: CFG makes two consecutive forward passes (conditioned and unconditioned). With CUDA graphs enabled by `torch.compile`, both passes reuse the same output buffer. The first result is overwritten before the CFG formula uses it.

**Fix**: Clone the first prediction immediately after computing it:
```python
# CFG pattern (fixed)
pred_cond = model(x, t, condition)
pred_cond = pred_cond.clone()  # Prevent CUDA graph overwrite
pred_uncond = model(x, t, no_condition)
pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
```

**Alternative**: Call `torch.compiler.cudagraph_mark_step_begin()` before each model invocation, but cloning is simpler.

**Affected code**: `src/medgen/diffusion/strategies.py` - both DDPM and RFlow generate methods when using `cfg_scale > 1.0`.

## 72. Latent scale_factor Must Be Written Back to Config (Fixed Feb 2026)
**Problem**: DiT latent diffusion with VQ-VAE 4x crashed with positional embedding size mismatch.

**Error**: `RuntimeError: The size of tensor a (163840) must match the size of tensor b (20480)`

**Root cause**: `train.py` auto-detects `scale_factor` from the compression checkpoint at runtime, but never wrote it back to the Hydra config. The model factory reads `latent.scale_factor` from config, gets `null`, and falls back to `8`. So for a VQ-VAE 4x model (scale_factor=4), the DiT was built for 256/8=32 spatial instead of 256/4=64.

**Why 8x worked by coincidence**: The fallback default (8) happened to match 8x compression, so it silently succeeded.

**Fix**: Added `with open_dict(cfg)` in `train.py` to write detected `scale_factor`, `depth_scale_factor`, and `latent_channels` back to the Hydra config after auto-detection:
```python
with open_dict(cfg):
    cfg.latent.scale_factor = scale_factor
    cfg.latent.latent_channels = latent_channels
```

**Rule**: Any auto-detected value that downstream code reads from config must be written back to config.

## 73. Regional Tracker spatial_dims=2 Default for 3D (Fixed Feb 2026)
**Problem**: Test evaluation crashed for 3D experiments (exp11_2, exp12_2) with `feret_diameter_max` failure.

**Error**: `ValueError: Surface level must be within volume data range` from skimage on degenerate tumor region.

**Root cause**: `evaluate_test_set()` in `evaluation.py` created `RegionalMetricsTracker` with default `spatial_dims=2`, even for 3D experiments. This caused `_update_2d()` to run on 3D `[B,1,D,H,W]` data, leading to incorrect connected component analysis and Feret diameter failures.

**Fix**: Changed `evaluation.py` to use `trainer._create_regional_tracker(loss_fn='mse')` which correctly handles spatial_dims based on the trainer type.

**Impact**: Training runs were valid — only the post-training test evaluation crashed. Checkpoints and training metrics are intact.

## 74. Feret Diameter Fails on Degenerate Tumor Regions (Fixed Feb 2026)
**Problem**: `feret_diameter_max` from skimage raises `QhullError` or `ValueError` on degenerate (single-pixel-thick or flat) tumor regions where convex hull computation fails.

**Fix**: Added try/except in `_get_2d_feret()` with fallback to bounding box diagonal:
```python
def _get_2d_feret(self, region) -> float:
    try:
        return region.feret_diameter_max
    except (ValueError, QhullError):
        minr, minc, maxr, maxc = region.bbox
        return float(np.sqrt((maxr - minr) ** 2 + (maxc - minc) ** 2)) or 1.0
```

**Affected code**: `src/medgen/metrics/regional/tracker.py` — both `_get_2d_feret()` and `_get_3d_feret()` (routes through the 2D method).

## 75. compute_predicted_clean() Was Clamping to [0,1] (Fixed Feb 2026)
**Problem**: Both `DDPMStrategy.compute_predicted_clean()` and `RFlowStrategy.compute_predicted_clean()` applied `torch.clamp(..., 0, 1)` to the predicted clean images. This destroyed latent codes and wavelet coefficients that are naturally outside [0, 1].

**Root cause**: The clamp was originally correct for pixel-space diffusion but was never updated when latent/wavelet space support was added.

**Impact**: Perceptual loss, validation metrics (PSNR, MS-SSIM, LPIPS), and worst-batch visualization were all computed on wrongly-clamped-then-decoded images. The main MSE training loss was unaffected since it operates on predictions vs targets directly.

**Fix**: Removed all `torch.clamp(0, 1)` from `compute_predicted_clean()` in both strategies. The clamp now happens after decoding to pixel space (in generation sampling and `sampler.py`).

**Affected code**: `src/medgen/diffusion/strategy_rflow.py`, `src/medgen/diffusion/strategy_ddpm.py`

## 76. Generation Sampling Was Clamping Before Decode (Fixed Feb 2026)
**Problem**: `generation_sampling.py` and `sampler.py` applied `torch.clamp(0, 1)` to generated samples BEFORE decoding from latent/wavelet space. Wavelet detail coefficients (negative values) were zeroed out, and latent codes were truncated.

**Root cause**: Same class of bug as #75 — pixel-space assumption leaked into latent/wavelet paths.

**Fix**: Moved clamp to after `space.decode()` in all 3 locations:
- `generation_sampling.py` line ~348 (batched 2D)
- `generation_sampling.py` line ~474 (3D streaming)
- `sampler.py` line ~405

**Affected code**: `src/medgen/metrics/generation_sampling.py`, `src/medgen/metrics/sampler.py`

## 77. scale_factor > 1 Guard Missed PixelSpace with Rescaling (Fixed Feb 2026)
**Problem**: Many locations guarded `space.decode()` calls with `if space.scale_factor > 1`, but `PixelSpace` has `scale_factor=1` even when rescaling is enabled. This meant perceptual loss, metrics, and visualization code would skip decoding for rescaled pixel-space data, producing [-1,1] values where [0,1] was expected.

**Root cause**: `scale_factor` was only designed for spatial downsampling (latent, wavelet, S2D). When [-1,1] rescaling was added to `PixelSpace`, the guard pattern broke.

**Fix**: Added `needs_decode` property to `DiffusionSpace` base class. Default returns `self.scale_factor > 1`, but `PixelSpace` overrides it to return `self._rescale`. Replaced all 20+ `scale_factor > 1` decode guards with `needs_decode` across trainer, validation, evaluation, sampler, generation_sampling, and visualization code.

**Rule**: Always use `space.needs_decode` to check if `decode()` must be called. Only use `scale_factor > 1` for spatial dimension checks (e.g., noise shape matching).

**Affected code**: `spaces.py`, `trainer.py`, `validation.py`, `evaluation.py`, `sampler.py`, `generation_sampling.py`, `visualization.py`, `evaluation/visualization.py`

## 78. Compiled Forward Paths Had [0,1] Clamps (Fixed Feb 2026)
**Problem**: `diffusion_model_setup.py`, `compile_manager.py`, and `training_tricks.py` all applied `torch.clamp(0, 1)` to predicted clean images in the compiled forward path. Same class of bug as #75 but in different code paths.

**Fix**: Removed all `torch.clamp(0, 1)` from `_forward_single` and `_forward_dual` in all three files (4 clamps each = 12 total).

**Affected code**: `src/medgen/pipeline/diffusion_model_setup.py`, `src/medgen/pipeline/compile_manager.py`, `src/medgen/pipeline/training_tricks.py`

## 79. Seg Binarization Inconsistency (Fixed)

**Problem**: The 2D generation pipeline (`generate.py`) used min-max normalization + `make_binary(threshold=0.1)` for seg binarization, while the training pipeline (`generation_sampling.py`) used `clamp(0,1) + > 0.5`. This caused inconsistent behavior between training evaluation and standalone generation.

**Root cause**: 4 different inline patterns for the same operation were scattered across 10+ call sites with no single source of truth.

**Fix**: Created `binarize_seg(data, threshold=0.5)` in `src/medgen/data/utils.py` — clamps to [0,1] then thresholds. Replaced all inline patterns:
- `generate.py`: 4 sites (including the inconsistent min-max pattern)
- `eval_ode_solvers.py`: 1 site
- `eval_bin_adherence.py`: 1 site
- `generation_sampling.py`: 2 sites (2D batched + 3D streaming)
- `sampler.py`: 1 site
- `seg.py`: 2 sites (post-augmentation re-binarization)

**Rule**: Always use `binarize_seg()` for generated or augmented seg data. Use `make_binary()` only for ground-truth data that's already in [0,1].

## 80. Duplicated Functions in seg_conditioned.py (Fixed)

**Problem**: `seg_conditioned.py` had full copies of 4 functions already in `datasets.py`: `compute_feret_diameter()`, `compute_size_bins()`, `create_size_bin_maps()`, `DEFAULT_BIN_EDGES`. The `seg_conditioned.py` copy of `compute_feret_diameter` used non-deterministic `np.random.choice` while `datasets.py` uses `np.random.default_rng(seed=...)`.

**Fix**: Deleted duplicates from `seg_conditioned.py`, now imports from `datasets.py`.

**Rule**: Canonical location for size-bin utilities is `src/medgen/data/loaders/datasets.py`. Never duplicate these functions.

## 81. Inline NIfTI Save Logic (Fixed)

**Problem**: `save_nifti()` existed in `generate.py` but `eval_ode_solvers.py` inlined the same `np.diag + Nifti1Image + nib.save` logic in 2 functions.

**Fix**: Moved `save_nifti()` to `src/medgen/data/utils.py` as shared utility. All callers now import from there.

**Rule**: Use `from medgen.data.utils import save_nifti` for NIfTI output.

## 82. Voxel Spacing Ordering Convention

**Caution**: Two different ordering conventions exist:
- `compute_voxel_size()` returns `(x, y, z)` = `(0.9375, 0.9375, 1.0)` — for NIfTI affine matrices
- Config `mode.size_bins.voxel_spacing` stores `[D, H, W]` = `[1.0, 0.9375, 0.9375]` — for `compute_feret_diameter_3d`

The training dataloader (`seg.py`) reads from config in the correct `(D, H, W)` order. `generate.py` passes `compute_voxel_size()` output directly, which works only because in-plane spacing is isotropic (x=y). If FOV or resolution ever differs between axes, this would silently produce wrong Feret diameters.
