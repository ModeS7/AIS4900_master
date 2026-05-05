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

> ⚠️ **DC-AE is legacy.** Latent-space experiments consistently underperformed
> pixel-space on this dataset (see MEMORY.md). VQ-VAE is preferred for new
> compression experiments. DC-AE commands remain for reference / historical runs.

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

# Dual mode (T1 pre + T1 gd, conditioned on seg mask)
python -m medgen.scripts.train mode=dual strategy=rflow
python -m medgen.scripts.train mode=dual strategy=rflow data.joint_normalization=true  # Joint norm

# Triple mode (T1 pre + T1 gd + FLAIR, conditioned on seg mask)
python -m medgen.scripts.train mode=triple strategy=rflow
python -m medgen.scripts.train mode=triple strategy=rflow data.joint_normalization=true

# With EMA (Exponential Moving Average)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.use_ema=true training.ema.decay=0.9999

# With gradient clipping and warmup (recommended for stability)
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.gradient_clip_norm=0.5 training.warmup_epochs=10

# With min-SNR loss weighting
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.use_min_snr=true training.min_snr_gamma=5.0

# With perceptual loss
python -m medgen.scripts.train mode=bravo strategy=rflow \
    training.perceptual_weight=0.1
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

# 3D Pixel-Space with [-1,1] rescaling (zero-centered data for diffusion)
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    training.rescale_data=true

# 3D Pixel-Space with brain-only N(0,1) normalization
# Brain voxels: mean≈0, std≈1. Background: ≈-2.44. Available: bravo, t1_pre, t1_gd
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    pixel_norm=bravo

# 2D Pixel-Space with brain-only N(0,1) normalization
python -m medgen.scripts.train mode=bravo strategy=rflow \
    pixel_norm=bravo

# 3D Pixel-Space with Haar Wavelet Decomposition
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    wavelet.enabled=true

# 3D Wavelet Diffusion with dedicated WDM UNet
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    model=wdm_3d \
    wavelet.enabled=true

# 3D Wavelet Diffusion with [-1,1] rescaling before DWT (default: on)
# wavelet.rescale=true maps [0,1] -> [-1,1] before wavelet decomposition
python -m medgen.scripts.train --config-name=diffusion_3d \
    mode=bravo strategy=rflow \
    model=wdm_3d \
    wavelet.enabled=true \
    wavelet.rescale=true

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

## Generation (Image/Volume Synthesis)

```bash
# 2D: seg -> bravo pipeline
python -m medgen.scripts.generate mode=bravo \
    seg_model=runs/seg/model.pt image_model=runs/bravo/model.pt

# 3D: size_bins -> seg -> bravo pipeline
python -m medgen.scripts.generate paths=cluster spatial_dims=3 mode=bravo \
    seg_model=runs/seg/checkpoint.pt image_model=runs/bravo/checkpoint.pt

# Custom output subdirectory and sample count
python -m medgen.scripts.generate mode=bravo output_subdir=experiment1 \
    num_samples=100 seg_model=... image_model=...

# With time-shift ratio (SD3-style schedule shift, 2.0 is optimal)
python -m medgen.scripts.generate paths=cluster spatial_dims=3 mode=bravo \
    seg_model=... image_model=... \
    shift_ratio_bravo=2.0 shift_ratio_seg=1.0

# Resume generation from a specific image counter
python -m medgen.scripts.generate paths=cluster spatial_dims=3 mode=bravo \
    seg_model=... image_model=... \
    current_image=250 num_images=525
```

---

## Downstream Segmentation Training

```bash
# Baseline (real data only)
python -m medgen.scripts.train_segmentation scenario=baseline

# Synthetic (generated data only)
python -m medgen.scripts.train_segmentation scenario=synthetic \
    data.synthetic_dir=runs/diffusion_3d/bravo/.../generated

# Mixed (real + synthetic)
python -m medgen.scripts.train_segmentation scenario=mixed \
    data.synthetic_dir=runs/diffusion_3d/bravo/.../generated \
    data.synthetic_ratio=0.5

# 3D segmentation training
python -m medgen.scripts.train_segmentation model.spatial_dims=3
```

---

## Downstream nnU-Net Training

Each experiment gets an isolated preprocessed directory (`nnUNet_preprocessed_{experiment}/`)
with symlinked data and its own `splits_final.json` to prevent race conditions during
concurrent training (see pitfall #83).

```bash
# Train baseline (real-only), all 5 folds
python -m medgen.scripts.train_nnunet \
    --experiment baseline \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet

# Train mixed with 210 synthetic volumes, fold 0 only
python -m medgen.scripts.train_nnunet \
    --experiment mixed --n-synthetic 210 \
    --fold 0 \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet

# Continue interrupted training (safe: uses same isolated dir)
python -m medgen.scripts.train_nnunet \
    --experiment baseline --fold 0 \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet \
    --continue-training

# Evaluate nnU-Net (5-fold ensemble inference + metrics)
python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet

# Evaluate existing predictions only
python -m medgen.scripts.eval_nnunet \
    --pred-dir /path/to/predictions \
    --gt-dir /path/to/labelsTs \
    --output results.json
```

---

## Post-hoc Evaluation Scripts

```bash
# FreeU grid search (inference-only, no retraining)
python -m medgen.scripts.find_optimal_freeu \
    --checkpoint runs/checkpoint_latest.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --output-dir eval_freeu

# CFG scale sweep
python -m medgen.scripts.find_optimal_cfg \
    --checkpoint runs/checkpoint_latest.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --output-dir eval_cfg

# Time-shift ratio evaluation
python -m medgen.scripts.eval_time_shift \
    --checkpoint runs/checkpoint_bravo.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --output-dir eval_time_shift

# Post-hoc EMA synthesis (Karras EDM2)
python -m medgen.scripts.synthesize_phema \
    --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
    --data-root ~/MedicalDataSets/brainmetshare-3

# Time-shift ratio golden-section search (continuous optimization)
python -m medgen.scripts.eval_time_shift \
    --checkpoint runs/checkpoint_bravo.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --output-dir eval_time_shift \
    --search --search-lo 1.0 --search-hi 5.0 --metric fid

# Multi-metric step search (FID + RadImageNet FID + PCA in one run)
# Shares generated volumes across metrics — ~50% faster than separate runs
python -m medgen.scripts.find_optimal_steps \
    --checkpoint runs/checkpoint_bravo.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_steps_combined \
    --metric fid,fid_radimagenet,pca --lo 10 --hi 100

# Optimal step search by PCA brain shape error (single metric)
python -m medgen.scripts.find_optimal_steps \
    --checkpoint runs/checkpoint_bravo.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_steps_pca \
    --metric pca --lo 10 --hi 100

# Optimal step search by morphological score
python -m medgen.scripts.find_optimal_steps \
    --checkpoint runs/checkpoint_bravo.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_steps_morph \
    --metric morphological --lo 10 --hi 100
```

---

## Analysis & Evaluation Scripts

A categorized inventory of analysis and evaluation tools beyond the
"Post-hoc Evaluation Scripts" section. Most run on a trained checkpoint
and write JSON / NIfTI / figures into a sub-folder of `runs/eval/`.

### Generation evaluation (FID / KID / step search / sampler choice)

```bash
# FID floor of a compression model (VAE / VQ-VAE / DC-AE)
python -m medgen.scripts.eval_compression_fid \
    --compression-checkpoint runs/compression_3d/.../checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3

# Test whether VQ-VAE compression equalizes synth vs real distributions
python -m medgen.scripts.fid_compression_equalize \
    --synth-dir <synth-volumes> --real-dir <real-volumes> \
    --compression-checkpoint runs/compression_3d/...

# FID comparison: original synth vs VQ-VAE-roundtripped synth vs real
python -m medgen.scripts.fid_vqvae_roundtrip_compare \
    --synth-dir <synth-volumes> --real-dir <real-volumes>

# Restart Sampling vs baseline Euler for RFlow generation
python -m medgen.scripts.eval_restart \
    --checkpoint runs/checkpoint_best.pt --output-dir eval_restart

# Light SDEdit at very low t₀ (compare across t₀ values for refinement)
python -m medgen.scripts.eval_light_sdedit \
    --checkpoint runs/checkpoint_best.pt --output-dir eval_light_sdedit

# Final comparison across blur-attack experiments
python -m medgen.scripts.eval_blur_attack_compare \
    --variant-dirs <dir-A> <dir-B> ...
```

### Spectrum / feature-emergence analysis (Phase 1 diagnostics)

These were used to diagnose the mean-blur / vessel-deficit problem and
inform exp37/exp32 fine-tunes. All take a trained checkpoint and a real
data root.

```bash
# Radial 3D power spectrum across fine-tuned generators
python -m medgen.scripts.analyze_generation_spectrum --checkpoints A.pt B.pt ...

# Frangi vesselness across generators
python -m medgen.scripts.analyze_vessel_prominence --checkpoints A.pt B.pt ...

# Cortical-shell vesselness (vessel-only restricted to cortex)
python -m medgen.scripts.analyze_cortical_vessels --checkpoints A.pt B.pt ...

# Per-t hybrid generation ablation (real-x₀ for t∈[lo,hi], gen otherwise)
python -m medgen.scripts.analyze_hybrid_generation --checkpoint runs/...

# Generation trajectory emergence (per-t feature emergence)
python -m medgen.scripts.analyze_generation_trajectory --checkpoint runs/...

# Post-process the trajectory JSONs into a timeline figure
python -m medgen.scripts.analyze_emergence_timeline --input-dir <traj-dir>

# Per-t velocity divergence map
python -m medgen.scripts.analyze_velocity_divergence --checkpoint runs/...

# Stochastic-Euler ablation (RFlow with noise injection)
python -m medgen.scripts.analyze_stochastic_sampling --checkpoint runs/...

# Timestep-response diagnostic (model output as t varies)
python -m medgen.scripts.analyze_timestep_response --checkpoint runs/...

# SR feasibility: does real_ds_us spectrum match generated-output spectrum?
python -m medgen.scripts.analyze_sr_feasibility --real-dir <r> --gen-dir <g>

# Velocity prediction quality across the noise schedule
python -m medgen.scripts.measure_velocity_breakdown --checkpoint runs/...

# Distinguishability: training distribution vs N(0,1) noise prior
python -m medgen.scripts.measure_distinguishability --data-root <r>

# Mean-blur diagnostic: stochastic prediction diversity at each t
python -m medgen.scripts.diagnose_mean_blur --checkpoint runs/...

# Frequency-mixing probe (HP real + LP synth)
python -m medgen.scripts.probe_freq_mix --real-dir <r> --gen-dir <g>

# VQ-VAE roundtrip probe on synthetic outputs
python -m medgen.scripts.probe_vqvae_roundtrip \
    --synth-dir <s> --vqvae-checkpoint runs/compression_3d/...
```

### Restoration / refinement (IR-SDE / Bridge / Resfusion / SDEdit)

```bash
# Apply a trained restoration model to (generated or degraded) volumes
python -m medgen.scripts.restore_volumes \
    --restoration-checkpoint runs/diffusion_3d/.../checkpoint_best.pt \
    --input-dir <volumes-to-restore> --output-dir <restored>

# Calibrate SDEdit degradation strength for restoration training
python -m medgen.scripts.calibrate_degradation \
    --diffusion-checkpoint runs/...

# Compare frequency-domain degradation methods (Gaussian blur, downsample, ...)
python -m medgen.scripts.compare_degradation_methods --real-dir <r>

# Pre-generate paired (degraded, clean) volumes for restoration training
python -m medgen.scripts.generate_degradation_pairs \
    --diffusion-checkpoint runs/... --output-dir <pairs>

# Pre-generate exp1_1_1000 outputs for IR-SDE restoration pair training (exp43b)
python -m medgen.scripts.pregen_restoration_pairs \
    --checkpoint runs/exp1_1_1000/checkpoint_best.pt --output-dir <pairs>

# Precompute (real, real_rt) pairs for exp43 VQ-VAE deblur training
python -m medgen.scripts.precompute_vqvae_pairs \
    --vqvae-checkpoint runs/compression_3d/... --output-dir <pairs>

# SDEdit-style refinement of synthetic volumes (blur-attack T1A)
python -m medgen.scripts.refine_sdedit_synth --synth-dir <s> --checkpoint runs/...

# Spectral equalization refinement (Wiener-style, blur-attack T1B)
python -m medgen.scripts.refine_spectral_eq --synth-dir <s> --real-dir <r>

# Pix2Pix refinement GAN training (exp42)
python -m medgen.scripts.train_refinement_gan --pairs-dir <pairs>
```

### Latent / wavelet / DC-AE pipeline tools

```bash
# Pre-encode images to latent space using a trained VAE/VQ-VAE/DC-AE
python -m medgen.scripts.encode_latents \
    --compression-checkpoint runs/compression_3d/.../checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3

# Recompute latent normalization stats from existing cache
python -m medgen.scripts.recompute_latent_stats --latent-dir <encoded-dir>

# Verify LDM training+generation pipeline before launching a real run
python -m medgen.scripts.verify_ldm_pipeline --config-name=diffusion_3d ...

# Verify WDM (wavelet diffusion) training+generation pipeline
python -m medgen.scripts.verify_wdm_pipeline --config-name=diffusion_3d ...

# Diagnose LDM by partial denoising round-trips
python -m medgen.scripts.debug_ldm_roundtrip --checkpoint runs/...
```

### PCA / atlas / morphology setup (one-time)

```bash
# Compute brain atlas (union of all training brain masks)
python -m medgen.scripts.compute_brain_atlas --data-root <r> --output-path data/brain_atlas.npz

# Compute PCA shape model from real brain masks
python -m medgen.scripts.compute_brain_pca --data-root <r> \
    --output-path data/brain_pca_256x256x160.npz --n-components 30

# Generate PCA explained-variance plots for thesis
python -m medgen.scripts.plot_pca_components --pca-path data/brain_pca_*.npz

# Ablation: PCA shape filter across resolution and component count
python -m medgen.scripts.ablate_pca_filter --pca-path data/brain_pca_*.npz
```

### Visualization & misc

```bash
# Visualize FFT amplitude: real vs generated volumes
python -m medgen.scripts.visualize_fft_comparison --real-dir <r> --gen-dir <g>

# Visualize per-timestep loss-weight schedules across all fine-tune experiments
python -m medgen.scripts.visualize_loss_schedules --output-dir docs/figures/

# Plot per-tumor detection analysis from saved JSON records
python -m medgen.scripts.plot_tumor_detection --json <results.json>

# Learning rate finder (diffusion + VAE)
python -m medgen.scripts.lr_finder --config-name=diffusion_3d \
    mode=bravo strategy=rflow model.spatial_dims=3
```

---

## Pixel Normalization Statistics

```bash
# Compute brain-only pixel stats for diffusion training
python -m medgen.scripts.compute_pixel_stats \
    --data-root /path/to/brainmetshare-3 \
    --modalities bravo seg

# With specific resolution
python -m medgen.scripts.compute_pixel_stats \
    --data-root /path/to/brainmetshare-3 \
    --image-size 256 --depth 160 \
    --modalities bravo seg
```

Output values are used in `configs/pixel_norm/*.yaml` files.

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

# Evaluate DiffRS against baseline generation (argparse, not Hydra)
python -m medgen.scripts.eval_diffrs \
    --bravo-model runs/checkpoint_bravo.pt \
    --diffrs-head runs/diffrs_head.pt \
    --data-root ~/data/brainmetshare-3 \
    --num-volumes 25 --output-dir results/eval_diffrs
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
