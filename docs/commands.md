# MedGen Command Reference

Full command reference with all options. For quick start, see `CLAUDE.md`.

---

## VAE Training

```bash
# Basic VAE (2D)
python -m medgen.scripts.train_compression --config-name=vae mode=dual
python -m medgen.scripts.train_compression --config-name=vae mode=multi_modality

# VAE with Pure BF16 weights (~50% memory savings)
python -m medgen.scripts.train_compression --config-name=vae mode=multi_modality \
    'training.precision.pure_weights=true'

# 3D VAE (volumetric, 256×256×160)
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality

# 3D VAE with gradient checkpointing (~50% memory savings)
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality \
    training.gradient_checkpointing=true

# 3D VAE without discriminator (saves ~15GB VRAM)
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality \
    vae_3d.disable_gan=true
```

---

## VQ-VAE Training

```bash
# VQ-VAE (discrete latent space, 512 codebook)
python -m medgen.scripts.train_compression --config-name=vqvae mode=multi_modality

# VQ-VAE without GAN (pure VQ-VAE)
python -m medgen.scripts.train_compression --config-name=vqvae mode=multi_modality \
    vqvae.disable_gan=true

# 3D VQ-VAE
python -m medgen.scripts.train_compression --config-name=vqvae_3d mode=multi_modality

# 3D VQ-VAE without discriminator
python -m medgen.scripts.train_compression --config-name=vqvae_3d mode=multi_modality \
    vqvae_3d.disable_gan=true
```

---

## DC-AE Training

```bash
# DC-AE (32× compression, default)
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality

# DC-AE (64× compression)
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality

# DC-AE (128× compression)
python -m medgen.scripts.train_compression --config-name=dcae dcae=f128 mode=multi_modality

# DC-AE with pretrained ImageNet weights
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality \
    dcae.pretrained="mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"

# DC-AE Phase 3: GAN refinement
# CRITICAL: Must use betas=[0.5,0.9] for GAN stability
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality \
    training.phase=3 \
    dcae.adv_weight=0.1 \
    training.learning_rate=5.4e-5 \
    'training.optimizer.betas=[0.5,0.9]' \
    pretrained_checkpoint=runs/compression_2d/.../checkpoint_best.pt

# 3D DC-AE
python -m medgen.scripts.train_compression --config-name=dcae_3d mode=multi_modality

# 3D DC-AE without discriminator
python -m medgen.scripts.train_compression --config-name=dcae_3d mode=multi_modality \
    dcae_3d.disable_gan=true

# DC-AE Segmentation Mask Compression (BCE+Dice+Boundary loss)
python -m medgen.scripts.train_compression --config-name=dcae mode=seg dcae.seg_mode=true

# DC-AE 1.5: Structured Latent Space (for f64/f128 only, NOT f32)
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality \
    dcae.structured_latent.enabled=true
```

---

## Diffusion Training

```bash
# Basic diffusion (UNet)
python -m medgen.scripts.train mode=dual strategy=rflow
python -m medgen.scripts.train mode=bravo strategy=rflow

# Diffusion with DiT (Transformer)
python -m medgen.scripts.train model=dit model.variant=S mode=bravo strategy=rflow   # 33M params
python -m medgen.scripts.train model=dit model.variant=B mode=bravo strategy=rflow   # 130M params
python -m medgen.scripts.train model=dit model.variant=L mode=bravo strategy=rflow   # 458M params
python -m medgen.scripts.train model=dit model.variant=XL mode=bravo strategy=rflow  # 675M params

# Diffusion with HDiT (Hierarchical Transformer, 3D)
python -m medgen.scripts.train model=hdit_3d model.variant=S mode=bravo strategy=rflow model.spatial_dims=3
python -m medgen.scripts.train model=hdit_3d model.variant=B mode=bravo strategy=rflow model.spatial_dims=3
python -m medgen.scripts.train model=hdit_3d model.variant=XL \
    'model.level_depths=[4,6,8,6,4]' mode=bravo strategy=rflow model.spatial_dims=3

# Diffusion with UViT (ViT with skip connections, 3D)
python -m medgen.scripts.train model=uvit_3d model.variant=S mode=bravo strategy=rflow model.spatial_dims=3
python -m medgen.scripts.train model=uvit_3d model.variant=M mode=bravo strategy=rflow model.spatial_dims=3

# Seg-conditioned diffusion (generate seg masks conditioned on tumor sizes)
python -m medgen.scripts.train mode=seg_conditioned strategy=rflow
```

---

## 3D Diffusion Training

```bash
# 3D Diffusion (pixel-space)
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3

# 3D Latent Diffusion (with pre-trained compression model)
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3 \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_3d/.../checkpoint_best.pt

# 3D Seg-conditioned diffusion (generate 3D seg masks)
python -m medgen.scripts.train mode=seg_conditioned strategy=rflow model.spatial_dims=3

# 3D DiT Latent Diffusion (with VQ-VAE compression)
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo_seg_cond \
    model=dit_3d \
    model.variant=S \
    model.patch_size=1 \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_3d/.../checkpoint_latest.pt \
    latent.compression_type=vqvae

# 3D Pixel-Space with Space-to-Depth (lossless 2x2x2 rearrangement)
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    space_to_depth.enabled=true

# 3D Pixel-Space with Haar Wavelet Decomposition
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    wavelet.enabled=true

# 3D Wavelet Diffusion with dedicated WDM UNet
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    model=wdm_3d \
    wavelet.enabled=true

# 3D Diffusion with ControlNet (pixel-resolution conditioning)
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3 \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_3d/.../checkpoint_best.pt \
    controlnet.enabled=true \
    controlnet.freeze_unet=true \
    pretrained_checkpoint=runs/diffusion_3d/.../checkpoint_best.pt
```

---

## Latent Diffusion

```bash
# 2D Latent Diffusion (auto-caches latents)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt

# 3D Latent Diffusion
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3 \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_3d/.../checkpoint_best.pt

# 2D Latent Diffusion with VQ-VAE
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    latent.compression_type=vqvae

# 3D Latent Diffusion with VQ-VAE (bravo_seg_cond mode)
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo_seg_cond \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_3d/.../checkpoint_latest.pt \
    latent.compression_type=vqvae

# With ControlNet (pixel-resolution conditioning)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    controlnet.enabled=true \
    controlnet.freeze_unet=false

# ControlNet Stage 2 (freeze UNet, train ControlNet only)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    controlnet.enabled=true \
    controlnet.freeze_unet=true \
    pretrained_checkpoint=runs/diffusion_2d/.../checkpoint_best.pt

# With Augmented Diffusion Training (DC-AE 1.5)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt \
    training.augmented_diffusion.enabled=true
```

---

## Regularization Techniques

### ScoreAug (augments noisy data)

```bash
# ScoreAug compose mode
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.compose=true \
    training.score_aug.compose_prob=0.5 \
    training.score_aug.use_omega_conditioning=true

# ScoreAug v2 (structured)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.score_aug.enabled=true \
    training.score_aug.v2_mode=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.nondestructive_prob=0.5 \
    training.score_aug.destructive_prob=0.5 \
    training.score_aug.use_omega_conditioning=true
```

### SDA (augments clean data with shifted timesteps)

```bash
# NOTE: SDA and ScoreAug are mutually exclusive
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.augment=false \
    training.sda.enabled=true \
    training.sda.rotation=true \
    training.sda.flip=true \
    training.sda.noise_shift=0.1 \
    training.sda.prob=0.5
```

### Clean Regularization (no distribution shift)

```bash
# Constant LR (skip cosine decay)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.scheduler=constant

# Gradient Noise
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.gradient_noise.enabled=true \
    training.gradient_noise.sigma=0.01

# Curriculum Timesteps
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.curriculum.enabled=true \
    training.curriculum.warmup_epochs=50

# Timestep Jitter
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.timestep_jitter.enabled=true \
    training.timestep_jitter.std=0.05

# Noise Augmentation
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.noise_augmentation.enabled=true \
    training.noise_augmentation.std=0.1

# Feature Perturbation
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.feature_perturbation.enabled=true \
    training.feature_perturbation.std=0.1 \
    'training.feature_perturbation.layers=["mid"]'
```

---

## Region-Weighted Loss

```bash
# Higher loss weight on small tumors (conditional modes only)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.regional_weighting.enabled=true

# Custom weights
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.regional_weighting.enabled=true \
    training.regional_weighting.weights.tiny=3.0 \
    training.regional_weighting.weights.small=2.0
```

---

## Generation Quality Metrics

```bash
# Enable KID, CMMD, FID tracking
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.generation_metrics.enabled=true

# Custom sample counts
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.generation_metrics.enabled=true \
    training.generation_metrics.samples_per_epoch=50
```

---

## Multi-Modality Mode Embedding

```bash
# Full mode embedding (default)
python -m medgen.scripts.train mode=multi strategy=rflow

# Mode embedding dropout
python -m medgen.scripts.train mode=multi strategy=rflow \
    mode.mode_embedding_strategy=dropout \
    mode.mode_embedding_dropout=0.2

# No mode embedding (hard parameter sharing)
python -m medgen.scripts.train mode=multi strategy=rflow \
    mode.mode_embedding_strategy=none

# FiLM conditioning
python -m medgen.scripts.train mode=multi strategy=rflow \
    mode.mode_embedding_strategy=film
```

---

## Profiling

```bash
# Profile training
python -m medgen.scripts.train_compression --config-name=vae mode=dual \
    +training.profiling.enabled=true

# Profile with memory tracking
python -m medgen.scripts.train_compression --config-name=vae mode=dual \
    +training.profiling.enabled=true \
    +training.profiling.active=50
```

---

## DiffRS (Rejection Sampling)

```bash
# Train DiffRS discriminator head on a trained diffusion model
python -m medgen.scripts.train_diffrs_discriminator \
    diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
    data_mode=bravo

# Quick test run
python -m medgen.scripts.train_diffrs_discriminator \
    diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
    data_mode=bravo num_generated_samples=100 num_epochs=5

# 3D DiffRS
python -m medgen.scripts.train_diffrs_discriminator \
    diffusion_checkpoint=runs/diffusion_3d/.../checkpoint_best.pt \
    data_mode=bravo spatial_dims=3
```

---

## ODE Solver Evaluation

```bash
# Evaluate multiple solvers on a trained RFlow model
python -m medgen.scripts.eval_ode_solvers \
    checkpoint_path=runs/.../checkpoint_best.pt \
    mode=bravo strategy=rflow

# Find optimal Euler step count (golden-section search)
python -m medgen.scripts.find_optimal_steps \
    checkpoint_path=runs/.../checkpoint_best.pt \
    mode=bravo strategy=rflow
```

See `docs/eval-ode-solvers.md` for results (Euler/25 is optimal for RFlow).

---

## Measure Latent Statistics

```bash
# Measure latent space std for scale_factor calibration
python -m medgen.scripts.measure_latent_std \
    compression_checkpoint=runs/compression_3d/.../checkpoint_best.pt
```

---

## torch.compile

```bash
# Enable torch.compile for diffusion training (fused forward pass)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.use_compile=true

# Compression with compile
python -m medgen.scripts.train_compression --config-name=vae mode=multi_modality \
    training.use_compile=true
```

---

## IDUN Cluster

```bash
# Submit jobs
sbatch IDUN/train/diffusion/exp1_rflow_128_baseline.slurm

# Submit 3D experiments (auto-chaining enabled, up to 20 segments)
sbatch IDUN/train/diffusion_3d/exp13_dit_4x_bravo.slurm

# Prefer H100, fallback to H100|A100 after 10 min
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm

# Run in background
./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm --bg
```

**Auto-chaining**: 3D SLURM scripts use SIGUSR1 signal handling to automatically save checkpoint and resubmit before wall time expires. Configure `CHAIN_MAX` in the script to set max segments (default: 20).

---

## Data Preprocessing

```bash
# Resize images
python misc/preprocessing/preprocess.py resize -i /path/to/raw -o /path/to/processed

# Align modalities to same slice count
python misc/preprocessing/preprocess.py align --data_dir /path/to/data -t 150

# Auto-trim empty slices
python misc/preprocessing/preprocess.py trim-auto --data_dir /path/to/data

# Split test into val/test_new
python misc/preprocessing/preprocess.py split --data_dir /path/to/data
```

---

## Local CI / Pre-Submit Validation

```bash
# Full local CI (syntax + imports + config resolution + 1-batch dry run)
./misc/local_ci.sh

# Validate a specific SLURM script before submitting
./misc/validate_before_submit.sh IDUN/train/diffusion_3d/exp13_dit_4x_bravo.slurm

# DiT memory profiling (sweep variants × resolutions × patch sizes)
python misc/profile_dit_memory.py

# HDiT/UViT memory profiling
python misc/profiling/profile_hdit_uvit_memory.py
```

---

## Syntax Check

```bash
python3 -m py_compile src/medgen/**/*.py
```

---

## Visualize Augmentations

```bash
python -m medgen.scripts.visualize_augmentations augment_type=vae
python -m medgen.scripts.visualize_augmentations augment_type=diffusion
```
