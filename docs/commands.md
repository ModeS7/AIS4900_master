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
python -m medgen.scripts.train mode=dual strategy=rflow mode.joint_normalization=true  # Joint norm (mode-level key, see configs/mode/dual.yaml:31)

# Triple mode (T1 pre + T1 gd + FLAIR, conditioned on seg mask)
python -m medgen.scripts.train mode=triple strategy=rflow
python -m medgen.scripts.train mode=triple strategy=rflow mode.joint_normalization=true

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

# 3D Wavelet Diffusion with [-1,1] rescaling before DWT
# (default OFF per configs/wavelet/default.yaml:27 — opt-in for the WDM-paper recipe)
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

`generate.py` is a Hydra script. The mode key is **`gen_mode`** (not `mode`)
per `configs/generate.yaml:25` and `cfg.gen_mode` reads in
`src/medgen/scripts/generate.py:317,318,368,369,...`. Sample count is
**`num_images`** (not `num_samples`), per `configs/generate.yaml:48`.

```bash
# 2D: seg -> bravo pipeline
python -m medgen.scripts.generate gen_mode=bravo \
    seg_model=runs/seg/model.pt image_model=runs/bravo/model.pt

# 3D: size_bins -> seg -> bravo pipeline
python -m medgen.scripts.generate paths=cluster spatial_dims=3 gen_mode=bravo \
    seg_model=runs/seg/checkpoint.pt image_model=runs/bravo/checkpoint.pt

# Custom output subdirectory and sample count
python -m medgen.scripts.generate gen_mode=bravo output_subdir=experiment1 \
    num_images=100 seg_model=... image_model=...

# With time-shift ratio (SD3-style schedule shift, 2.0 is optimal)
python -m medgen.scripts.generate paths=cluster spatial_dims=3 gen_mode=bravo \
    seg_model=... image_model=... \
    shift_ratio_bravo=2.0 shift_ratio_seg=1.0

# Resume generation from a specific image counter
python -m medgen.scripts.generate paths=cluster spatial_dims=3 gen_mode=bravo \
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

# Save softmax .npz alongside binary predictions (for post-hoc threshold sweep)
python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id 600 --experiment-name exp3_baseline_v2_d600 \
    --folds 0 1 2 3 4 \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet \
    --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans \
    --save-probabilities

# Per-fold inference with isolated output dirs (one --folds N each, --output-dir per fold)
# Useful for downloading per-fold raw outputs to analyse locally
for FOLD in 0 1 2 3 4; do
    python -m medgen.scripts.eval_nnunet \
        --experiment baseline --dataset-id 600 \
        --experiment-name exp3_baseline_v2_d600 \
        --folds $FOLD \
        --nnunet-base /cluster/work/modestas/nnunet \
        --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet \
        --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans \
        --save-probabilities \
        --output-dir runs/.../per_fold_test/fold_$FOLD
done

# Re-validate existing checkpoints (skips training, writes val softmax probs)
python -m medgen.scripts.train_nnunet \
    --experiment baseline --dataset-id 600 \
    --experiment-name exp3_baseline_v2_d600 --fold 0 \
    --nnunet-base /cluster/work/modestas/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet \
    --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans \
    --validation-only --export-validation-probabilities
```

### Random-split dataset (Ottesen 2025 protocol replication)

Builds Dataset 640 by pooling all 156 Stanford patients (train+val+test_new)
and randomly splitting 105 train + 51 test with a fixed seed. Used to test
whether literature results that don't use the official Grøvik 2020 hold-out
are evaluating on a systematically easier cohort.

```bash
# Build Dataset 640 with random 105/51 split (seed=42)
python -m medgen.scripts.convert_random_split \
    --real-dir /cluster/work/modestas/MedicalDataSets/brainmetshare-3 \
    --nnunet-raw /cluster/work/modestas/MedicalDataSets/nnunet/nnUNet_raw \
    --dataset-id 640 \
    --seed 42 \
    --modality bravo
# The picked patient IDs are written into case_info.json for audit/reproducibility.
```

### Threshold sweep on saved softmax probabilities

Post-hoc tunes the binarisation threshold on nnU-Net's softmax outputs
(no retraining). For our exp3 data, optimum lands at t ≈ 0.005 vs default
t = 0.5; ~+0.012 absolute Dice, concentrated in tiny lesions.

```bash
# Mode 1: sweep on a single set (upper-bound, biased if used on test directly)
python -m medgen.scripts.threshold_sweep \
    --mode sweep \
    --probs-dir runs/.../predictions \
    --gt-dir /path/to/labelsTs \
    --output runs/.../threshold_sweep.json

# Mode 2: tune-eval — pick threshold on validation, apply to test (publishable)
python -m medgen.scripts.threshold_sweep \
    --mode tune-eval \
    --tune-probs-dir runs/.../fold_X/validation \
    --tune-gt-dir /path/to/labelsTr \
    --eval-probs-dir runs/.../predictions \
    --eval-gt-dir /path/to/labelsTs \
    --output runs/.../threshold_sweep_val_tuned.json
```

### Local nnU-Net inference harness (no cluster needed)

After training on cluster, all inference + analysis can run locally on a
single RTX 3090. See [`docs/downstream_nnunet.md`](downstream_nnunet.md)
for the one-time setup of `.venv_nnunet/` and `data/nnunet_local/`.

```bash
# Run 5 fold models on the 51 test cases (saves softmax .npz)
.venv_nnunet/bin/python misc/run_per_fold_local.py

# Per-fold + ensemble threshold sweep
.venv_nnunet/bin/python misc/analyze_per_fold_threshold.py
# → runs/.../threshold_analysis_per_fold.{json,md}

# Size-stratified Dice + detection at t=0.50 vs t=0.005
.venv_nnunet/bin/python misc/compare_thresholds_by_size.py
# → runs/.../threshold_size_comparison.json
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

A categorized inventory of analysis/evaluation tools. **All flags below
are pulled verbatim from each script's `argparse.add_argument` calls.**
Run `python -m medgen.scripts.<X> --help` for the full list per script.
Some scripts are Hydra-based (no argparse) — those are noted explicitly.

### Generation evaluation (FID / KID / step search / sampler choice)

```bash
# FID floor of a compression model (VAE / VQ-VAE / DC-AE)
python -m medgen.scripts.eval_compression_fid \
    --compression-checkpoint runs/compression_3d/.../checkpoint_best.pt \
    --compression-type vae \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_compression_fid

# Test whether VQ-VAE compression equalizes synth vs real distributions
python -m medgen.scripts.fid_compression_equalize \
    --input-dir <synth-volumes> --real-dir <real-volumes> \
    --compression-checkpoint runs/compression_3d/... --compression-type vqvae \
    --output-dir fid_equalize

# FID comparison: original synth vs VQ-VAE-roundtripped synth vs real
python -m medgen.scripts.fid_vqvae_roundtrip_compare \
    --input-dirs <synth-A> <synth-B> --real-dir <real-volumes> \
    --compression-checkpoint runs/compression_3d/... --compression-type vqvae \
    --output-dir fid_roundtrip

# Restart Sampling vs baseline Euler for RFlow generation
python -m medgen.scripts.eval_restart \
    --bravo-model runs/bravo/checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_restart

# Light SDEdit at very low t₀ (compare across t₀ values for refinement)
python -m medgen.scripts.eval_light_sdedit \
    --bravo-model runs/bravo/checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --generated-dir <synth-volumes> \
    --output-dir eval_light_sdedit

# Final comparison across blur-attack experiments
python -m medgen.scripts.eval_blur_attack_compare \
    --real-dir <real-volumes> --output-dir eval_blur_attack \
    --methods <method1> <method2>
```

### Spectrum / feature-emergence analysis (Phase 1 diagnostics)

These were used to diagnose the mean-blur / vessel-deficit problem and
inform exp37/exp32 fine-tunes.

```bash
# Radial 3D power spectrum across fine-tuned generators
python -m medgen.scripts.analyze_generation_spectrum \
    --real-dir <real-volumes> --compare-dir <gen-dir-1> <gen-dir-2> \
    --output-dir analyze_spectrum

# Frangi vesselness across generators
python -m medgen.scripts.analyze_vessel_prominence \
    --real-dir <real-volumes> --compare-dir <gen-dir-1> <gen-dir-2> \
    --output-dir analyze_vessel

# Cortical-shell vesselness (restricted to cortex)
python -m medgen.scripts.analyze_cortical_vessels \
    --real-dir <real-volumes> --compare-dir <gen-dir-1> <gen-dir-2> \
    --output-dir analyze_cortical

# Per-t hybrid generation ablation (real-x₀ for t∈[lo,hi], gen otherwise)
python -m medgen.scripts.analyze_hybrid_generation \
    --baseline runs/baseline.pt --fine-tune runs/finetune.pt --ft-label <name> \
    --data-root <data> --real-dir <real> --output-dir analyze_hybrid

# Generation trajectory (per-t feature emergence)
python -m medgen.scripts.analyze_generation_trajectory \
    --bravo-model runs/bravo.pt --data-root <data> \
    --output-dir analyze_trajectory

# Post-process the trajectory JSONs into a timeline figure
python -m medgen.scripts.analyze_emergence_timeline \
    --input-dirs <traj-dir-1> <traj-dir-2> --output-dir emergence_timeline

# Per-t velocity divergence map
python -m medgen.scripts.analyze_velocity_divergence \
    --baseline runs/baseline.pt --fine-tunes runs/ft-A.pt runs/ft-B.pt \
    --data-root <data> --output-dir analyze_velocity_div

# Stochastic-Euler ablation (RFlow with noise injection at sampling time)
python -m medgen.scripts.analyze_stochastic_sampling \
    --checkpoint runs/checkpoint.pt --data-root <data> --real-dir <real> \
    --sigmas 0.0 0.1 0.5 --output-dir analyze_stochastic

# Timestep-response diagnostic (model output as t varies)
python -m medgen.scripts.analyze_timestep_response \
    --checkpoint runs/checkpoint.pt --data-root <data> --output-dir analyze_t_response

# SR feasibility: does real_ds_us spectrum match generated-output spectrum?
python -m medgen.scripts.analyze_sr_feasibility \
    --real-dir <real> --compare-dir <gen-dir> --output-dir analyze_sr

# Velocity prediction quality across the noise schedule
python -m medgen.scripts.measure_velocity_breakdown \
    --bravo-model runs/checkpoint.pt --data-root <data> \
    --output-dir measure_velocity

# Distinguishability: training distribution vs N(0,1) noise prior
python -m medgen.scripts.measure_distinguishability \
    --data-dir ~/MedicalDataSets/brainmetshare-3/train

# Mean-blur diagnostic: stochastic prediction diversity at each t
python -m medgen.scripts.diagnose_mean_blur \
    --bravo-model runs/checkpoint.pt --data-root <data> --output-dir diagnose_blur

# Frequency-mixing probe (HP real + LP synth)
python -m medgen.scripts.probe_freq_mix \
    --real-dir <real> --synth-dir <synth> --output-dir probe_freq

# VQ-VAE roundtrip probe on synthetic outputs
python -m medgen.scripts.probe_vqvae_roundtrip \
    --input-dir <synth> --real-dir <real> \
    --compression-checkpoint runs/compression_3d/... --compression-type vqvae \
    --output-dir probe_vqvae_rt
```

### Restoration / refinement (IR-SDE / Bridge / Resfusion / SDEdit)

```bash
# Apply a trained restoration model to (generated or degraded) volumes
python -m medgen.scripts.restore_volumes \
    --restoration-model runs/restoration/checkpoint_best.pt --strategy rflow \
    --input-dir <volumes-to-restore> --output-dir <restored> --num-steps 25

# Calibrate SDEdit degradation strength for restoration training
python -m medgen.scripts.calibrate_degradation \
    --bravo-model runs/bravo.pt --data-root <data> \
    --generated-dir <synth> --output-dir calibrate_degradation

# Compare frequency-domain degradation methods (empirical TF vs analytical Wiener)
python -m medgen.scripts.compare_degradation_methods \
    --real-dir <real> --generated-dir <synth> --output-dir compare_degradation

# Pre-generate paired (degraded, clean) volumes for restoration training
python -m medgen.scripts.generate_degradation_pairs \
    --bravo-model runs/bravo.pt --data-root <data> --output-dir <pairs>

# Pre-generate exp1_1_1000 outputs for IR-SDE restoration pair training (exp43b)
python -m medgen.scripts.pregen_restoration_pairs \
    --checkpoint runs/exp1_1_1000/checkpoint_best.pt \
    --data-root <data> --output-root <pairs>

# Precompute (real, real_rt) pairs for exp43 VQ-VAE deblur training
python -m medgen.scripts.precompute_vqvae_pairs \
    --compression-checkpoint runs/compression_3d/... --compression-type vqvae \
    --data-root <data> --output-dir <pairs>

# SDEdit-style refinement of synthetic volumes (blur-attack T1A)
python -m medgen.scripts.refine_sdedit_synth \
    --checkpoint runs/restoration.pt --synth-dirs <synth-A> <synth-B> \
    --real-dir <real> --output-dir refine_sdedit

# Spectral equalization refinement (Wiener-style, blur-attack T1B)
python -m medgen.scripts.refine_spectral_eq \
    --synth-dirs <synth> --real-dir <real> --output-dir refine_spectral

# Pix2Pix refinement GAN training (exp42) — many flags; see --help
python -m medgen.scripts.train_refinement_gan \
    --data-root <data> --output-dir <out> --epochs 100
```

### Latent / wavelet / DC-AE pipeline tools

```bash
# Pre-encode images to latent space using a trained VAE
# NOTE: encode_latents uses underscore-style flags (--vae_checkpoint not --vae-checkpoint)
python -m medgen.scripts.encode_latents \
    --vae_checkpoint runs/compression_3d/.../checkpoint_best.pt \
    --data_dir ~/MedicalDataSets/brainmetshare-3/train \
    --output_dir ~/MedicalDataSets/brainmetshare-3-latents/train \
    --mode multi_modality

# Recompute latent normalization stats from existing cache
python -m medgen.scripts.recompute_latent_stats \
    --data-root ~/MedicalDataSets/brainmetshare-3-latents

# Verify LDM training+generation pipeline before launching a real run
python -m medgen.scripts.verify_ldm_pipeline \
    --compression-checkpoint runs/compression_3d/... --compression-type vae \
    --data-root <data>

# Verify WDM (wavelet diffusion) training+generation pipeline
python -m medgen.scripts.verify_wdm_pipeline --data-root <data>

# Diagnose LDM by partial denoising round-trips
python -m medgen.scripts.debug_ldm_roundtrip \
    --checkpoint runs/checkpoint.pt \
    --compression-checkpoint runs/compression_3d/... --compression-type vae \
    --data-root <data> --output-dir debug_ldm
```

### PCA / atlas / morphology setup (one-time)

```bash
# Compute brain atlas (union of all training brain masks)
# NOTE: --output (not --output-path)
python -m medgen.scripts.compute_brain_atlas \
    --data-root <data> --output data/brain_atlas.npz

# Compute PCA shape model from real brain masks
# Fit on train, calibrate k + accept-threshold on a DISJOINT held-out set (test_new).
# --n-components: int or 'full' (cumEVR=1.0); threshold = held-out-real percentile.
python -m medgen.scripts.compute_brain_pca \
    --data-root <data> --output data/brain_pca_256x256x160.npz \
    --fit-splits train --heldout-splits test_new \
    --n-components full --threshold-percentile 99

# Generate PCA explained-variance plots for thesis
# NOTE: --npz (not --pca-path)
python -m medgen.scripts.plot_pca_components \
    --npz data/brain_pca_256x256x160.npz --output-dir <plots>

# Ablation: PCA shape filter across resolution and component count
python -m medgen.scripts.ablate_pca_filter \
    --pca-coarse data/brain_pca_low.npz --pca-fine data/brain_pca_256x256x160.npz \
    --gen-dir <synth> --real-dir <real> --train-root <data> --output-dir ablate_pca
```

### Visualization & misc

```bash
# Visualize FFT amplitude: real vs generated volumes
# NOTE: --generated-dir (not --gen-dir)
python -m medgen.scripts.visualize_fft_comparison \
    --real-dir <real> --generated-dir <synth> --output-dir fft_comparison

# Visualize per-timestep loss-weight schedules across fine-tune experiments
python -m medgen.scripts.visualize_loss_schedules --output-dir docs/figures/

# Plot per-tumor detection analysis (the script reads pre-saved JSONs from cwd
# by default; use --output-dir to redirect outputs)
python -m medgen.scripts.plot_tumor_detection --output-dir <plots>

# Learning rate finder (Hydra-based — same overrides as `train`)
python -m medgen.scripts.lr_finder mode=bravo strategy=rflow model.spatial_dims=3
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

Both scripts are **argparse-based** (NOT Hydra). Verbatim flags from
`src/medgen/scripts/eval_ode_solvers.py:1082+` and
`src/medgen/scripts/find_optimal_steps.py:295+`:

```bash
# Evaluate multiple solvers on a trained RFlow model
python -m medgen.scripts.eval_ode_solvers \
    --bravo-model runs/.../checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_ode_solvers --num-volumes 25

# Find optimal Euler step count (golden-section search)
python -m medgen.scripts.find_optimal_steps \
    --checkpoint runs/.../checkpoint_best.pt \
    --data-root ~/MedicalDataSets/brainmetshare-3 \
    --output-dir eval_steps --metric fid
```

See `docs/eval-ode-solvers.md` for results (Euler/25 is optimal for RFlow).

---

## Measure Latent Statistics

```bash
# Measure latent space std for scale_factor calibration
# Hydra script. Reads cfg.checkpoint per src/medgen/scripts/measure_latent_std.py:33
python -m medgen.scripts.measure_latent_std \
    checkpoint=runs/compression_3d/.../checkpoint_best.pt
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
python misc/data_processing/preprocessing/preprocess.py resize -i /path/to/raw -o /path/to/processed

# Align modalities to same slice count
python misc/data_processing/preprocessing/preprocess.py align --data_dir /path/to/data -t 150

# Auto-trim empty slices
python misc/data_processing/preprocessing/preprocess.py trim-auto --data_dir /path/to/data

# Split test into val/test_new
python misc/data_processing/preprocessing/preprocess.py split --data_dir /path/to/data
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

The script visualizes BOTH diffusion and VAE pipelines side-by-side; there
is no `augment_type` switch. Real Hydra keys per
`configs/visualize_augmentations.yaml`: `synthetic`, `modality`, `n_samples`,
`image_size`, `output_dir`.

```bash
# Default (uses paths=local, modality=t1_pre, n_samples=4)
python -m medgen.scripts.visualize_augmentations

# Cluster paths + bravo modality + more samples
python -m medgen.scripts.visualize_augmentations \
    paths=cluster modality=bravo n_samples=8

# Force synthetic data
python -m medgen.scripts.visualize_augmentations synthetic=true
```
