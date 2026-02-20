# Complete Experiment Results

Last updated: February 20, 2026. Data extracted from IDUN logs (`IDUN/output/`) and TensorBoard runs (`runs_tb/`).

---

## Table of Contents

1. [Global Summary](#global-summary)
2. [2D Diffusion Experiments](#2d-diffusion-experiments)
3. [3D Diffusion Experiments](#3d-diffusion-experiments)
4. [2D Compression Experiments](#2d-compression-experiments)
5. [3D Compression Experiments](#3d-compression-experiments)
6. [Segmentation Compression](#segmentation-compression)
7. [Downstream Segmentation](#downstream-segmentation)
8. [3D Generation & Evaluation](#3d-generation--evaluation)
9. [Infrastructure Issues](#infrastructure-issues)

---

## Global Summary

| Domain | Total Experiments | Best Result | Key Metric |
|--------|-------------------|-------------|------------|
| **2D Diffusion** | 89 | SiT-S (exp12_1c) | val_loss=0.005604 |
| **2D Diffusion (FID)** | 10 evaluated | SiT-S (exp12_1b) | FID=33.43, KID=0.038 |
| **3D Diffusion Pixel** | ~25 runs | exp1_1 256x256 | val_loss=0.00211 |
| **3D Diffusion Latent** | ~15 runs | exp9_1 LDM 4x | val_loss=0.0764 (still improving) |
| **2D Compression VAE** | 5 | exp4 64lat | SSIM=0.999, val_G=0.001356 |
| **2D Compression VQ-VAE** | 2 | exp6_1 4x | SSIM=0.997, val_G=0.002725 |
| **2D Compression DC-AE** | ~12 | exp9_3 f128 | SSIM=0.997, PSNR=40.76 |
| **3D Compression VQ-VAE** | 14 | exp8_1 4x lat4 | SSIM=0.995, PSNR=39.88 |
| **3D Compression DC-AE** | 1 | exp10_1 | SSIM=0.935, PSNR=30.91 |
| **Seg Compression 2D** | 4 | exp13_2 f64 | Dice=1.0000 |
| **Seg Compression 3D** | 1 | exp11_2 VQ-VAE | Dice=0.904 |
| **Downstream Seg 3D** | 2 completed | exp1 baseline | Test Dice=0.194 |

---

# 2D Diffusion Experiments

89 experiments total. All run on A100 (83) or H100 (6). Mode: `multi` (SiT/multi-modal) or default `bravo`.
Zero gradient spikes or divergence across all 89 experiments.

## 2D Top 15 by Best Validation Loss

| Rank | Experiment | Architecture | Epochs | Best Val | Gen Metrics |
|------|-----------|-------------|--------|----------|-------------|
| 1 | exp12_1c_sit_s | SiT-S | 500/500 | **0.005604** | -- |
| 2 | exp6_rflow_256 | UNet 256x256 | 500/500 | **0.006141** | -- |
| 3 | exp7_rflow_256_4lvl | UNet 256x256 4L | 500/500 | **0.006213** | -- |
| 4 | exp27_plateau | UNet + plateau LR | 375/500 | **0.006469** | -- |
| 5 | exp2_rflow_100steps (23962279) | UNet baseline | 500/500 | **0.006564** | FID=36.32 |
| 6 | exp1_3_continuous | UNet continuous | 500/500 | **0.006607** | FID=35.98 |
| 7 | exp12_2b_sit_b | SiT-B | 410/500 | **0.006682** | -- |
| 8 | exp12_1b_sit_s | SiT-S | 500/500 | **0.006758** | **FID=33.43** |
| 9 | exp1_2_ddpm | UNet DDPM | 500/500 | **0.006817** | -- |
| 10 | exp9_7_scoreaug_v2 | UNet + ScoreAug | 500/500 | **0.007099** | -- |
| 11 | exp9_6_scoreaug_compose08 | UNet + ScoreAug | 500/500 | **0.007110** | -- |
| 12 | exp16_bs8 | UNet bs=8 | 500/500 | **0.007135** | -- |
| 13 | exp9_5_scoreaug_compose | UNet + ScoreAug | 500/500 | **0.007183** | -- |
| 14 | exp5_1_rflow_100_m_snr | UNet + min_snr | 500/500 | **0.007254** | -- |
| 15 | exp9_5_scoreaug_compose (23873628) | UNet + ScoreAug | 500/500 | **0.007264** | -- |

## 2D Generation Metrics (FID/KID/CMMD)

10 experiments evaluated with generation metrics:

| Experiment | Best Val | FID | KID | CMMD |
|-----------|----------|-----|-----|------|
| exp12_1b_sit_s (SiT-S) | 0.006758 | **33.43** | **0.0375** | **0.2155** |
| exp1_3_continuous | 0.006607 | 35.98 | 0.0420 | 0.2268 |
| exp2_rflow_100steps (23962279) | 0.006564 | 36.32 | 0.0431 | 0.2234 |
| exp2_rflow_100steps (23914222) | 0.007364 | 36.42 | 0.0442 | 0.2304 |
| exp26_regional_weight | 0.007396 | 36.45 | 0.0433 | 0.2295 |
| exp23_1_noise_aug | 0.007410 | 36.67 | 0.0417 | 0.2270 |
| exp22_1_timestep_jitter | 0.007317 | 37.16 | 0.0452 | 0.2276 |
| exp25_1_self_cond | 0.007360 | 37.42 | 0.0409 | 0.2264 |
| exp24_1_feature_perturb | 0.007322 | 37.65 | 0.0440 | 0.2256 |
| exp18_1_sda (SDA) | 0.007294 | 40.37 | 0.0502 | 0.2296 |

SiT-S wins on all three generation metrics.

## 2D Experiment Categories

### Baseline & Strategy (exp1-2)
| Experiment | Description | Best Val | Notes |
|-----------|------------|----------|-------|
| exp1_rflow | RFlow 50 steps | 0.007661 | Original baseline |
| exp1_2_ddpm | DDPM | 0.006817 | DDPM beats RFlow on val |
| exp1_3_continuous | RFlow continuous | 0.006607 | Best UNet baseline |
| exp2_rflow_100steps (x6 runs) | RFlow 100 steps | 0.006564-0.007667 | Multiple runs, consistent |
| exp2_bf16mse | BF16 MSE test | 0.007487 | **FAILED** at epoch 107 |

### Attention & Augmentation (exp3-5)
| Experiment | Description | Best Val | Notes |
|-----------|------------|----------|-------|
| exp3_rflow_100+a (x2) | + attention | 0.007636-0.007937 | Attention hurts |
| exp4_rflow_100+a+ema | + attention + EMA | 0.007708 | Still worse |
| exp5_rflow_100+a+ema+m_snr | + min_snr | 0.007498 | Marginal improvement |
| exp5_1_rflow_100_m_snr | min_snr only | 0.007254 | Better without attention |

### Resolution (exp6-7)
| Experiment | Description | Best Val | Notes |
|-----------|------------|----------|-------|
| exp6_rflow_256 | 256x256 UNet | **0.006141** | Rank 2 overall |
| exp7_rflow_256_4lvl | 256x256 4-level | **0.006213** | Rank 3 overall |

### Architecture Sweep (exp8)
| Experiment | Variant | Best Val | Notes |
|-----------|---------|----------|-------|
| exp8_1_small | Small UNet | 0.007657 | |
| exp8_2_minimal | Minimal | 0.007738 | |
| exp8_3_tiny | Tiny | 0.007590 | |
| exp8_4_lean4lvl | Lean 4L | 0.007931 | |
| exp8_5_lean5lvl | Lean 5L | 0.008110 | Worst |
| exp8_6_bottleneck | Bottleneck | 0.008146 | Worst overall |
| exp8_7_deep_narrow | Deep narrow | 0.008120 | |
| exp8_8_adm | ADM-style | 0.007579-0.007838 | |
| exp8_9_edm | EDM-style | 0.007817 | |
| exp8_10_ddpm | DDPM | 0.008034 | |
| exp8_11_iddpm | iDDPM | 0.008047 | |

### ScoreAug (exp9)
| Experiment | Variant | Best Val | Notes |
|-----------|---------|----------|-------|
| exp9_1_scoreaug (x2) | Basic | 0.007296-0.007500 | |
| exp9_2_scoreaug_full (x2) | Full | 0.007533-0.007534 | |
| exp9_3_scoreaug_combined (x2) | Combined | 0.007527-0.007823 | |
| exp9_4_scoreaug_v2 (x2) | V2 | 0.007316-0.007392 | |
| exp9_5_scoreaug_compose (x2) | Compose | **0.007183**-0.007264 | Best ScoreAug |
| exp9_6_scoreaug_compose08 | Compose 0.8 | **0.007110** | 2nd best |
| exp9_7_scoreaug_v2 | V2 rerun | **0.007099** | Best ScoreAug |

### Multi-Modal Conditioning (exp10, 125 epochs)
| Experiment | Conditioning | Best Val | Notes |
|-----------|-------------|----------|-------|
| exp10_1_multi (x2) | Multi baseline | 0.008008-0.008145 | 1 run FAILED |
| exp10_2_multi_scoreaug (x3) | + ScoreAug | 0.007862-0.008018 | |
| exp10_3_multi_compose (x2) | + Compose | 0.007746-0.007770 | |
| exp10_4_multi | Multi v2 | 0.007916 | |
| exp10_5_multi_scoreaug | ScoreAug v2 | 0.007562 | |
| exp10_6_multi_dropout | Dropout | 0.007826 | Only 34/125 epochs |
| exp10_7_multi_none | No conditioning | 0.008013 | |
| exp10_8_multi_late | Late fusion | 0.007971 | |
| exp10_9_multi_film | FiLM | 0.007898 | |
| exp10_10_multi_film_scoreaug | FiLM+ScoreAug | 0.007561 | Best multi |

### Optimizer (exp11, SAM/ASAM)
| Experiment | Optimizer | Best Val | Notes |
|-----------|----------|----------|-------|
| exp11_1_sam | SAM | 0.007574 | |
| exp11_2_asam | ASAM | 0.007805 | |

### Transformer (exp12, SiT)
| Experiment | Architecture | Best Val | Notes |
|-----------|-------------|----------|-------|
| exp12_1_sit_s | SiT-S | 0.007712 | |
| exp12_1b_sit_s | SiT-S (improved) | 0.006758 | Best FID=33.43 |
| exp12_1c_sit_s | SiT-S (best) | **0.005604** | **#1 overall** |
| exp12_2_sit_b | SiT-B | 0.007801 | |
| exp12_2b_sit_b | SiT-B (improved) | 0.006682 | 410/500 epochs |

### EMA (exp13)
| Experiment | EMA Config | Best Val | Notes |
|-----------|-----------|----------|-------|
| exp13_1_ema_simple | Simple | 0.007540 | |
| exp13_2_ema_slow | Slow decay | 0.007561 | |

### Dropout/DropPath (exp14, exp17)
| Experiment | Rate | Best Val | Notes |
|-----------|------|----------|-------|
| exp14_1_sit_s_drop01 | 0.1 | 0.007449 | |
| exp14_2_sit_s_drop02 | 0.2 | 0.007454 | |
| exp14_3_sit_s_drop03 | 0.3 | 0.007516 | |
| exp17_1_droppath | Low | 0.007476 | |
| exp17_2_droppath | Medium | 0.007398 | |
| exp17_3_droppath | High | 0.007397 | |

### Weight Decay (exp15)
| Experiment | WD | Best Val | Notes |
|-----------|-----|----------|-------|
| exp15_1_rflow_wd001 | 0.01 | 0.007348 | |
| exp15_2_rflow_wd005 | 0.05 | 0.007363 | |
| exp15_3_rflow_wd01 | 0.1 | 0.007310 | Best WD |

### Batch Size (exp16)
| Experiment | BS | Best Val | Notes |
|-----------|-----|----------|-------|
| exp16_bs4 | 4 | 0.007300 | |
| exp16_bs8 | 8 | **0.007135** | Best BS |
| exp16_bs24 | 24 | 0.007415 | |
| exp16_bs32 | 32 | -- | No output (crash) |

### Training Tricks (exp18-27)
| Experiment | Technique | Best Val | Notes |
|-----------|----------|----------|-------|
| exp18_1_sda | SDA | 0.007294 | FID=40.37 (worst FID) |
| exp19_1_constant_lr | Constant LR | 0.007334 | |
| exp20_1_grad_noise | Gradient noise | 0.007339 | |
| exp21_1_curriculum | Curriculum | 0.007379 | |
| exp22_1_timestep_jitter | Timestep jitter | 0.007317 | FID=37.16 |
| exp23_1_noise_aug | Noise augmentation | 0.007410 | FID=36.67 |
| exp24_1_feature_perturb | Feature perturbation | 0.007322 | FID=37.65 |
| exp25_1_self_cond | Self-conditioning | 0.007360 | FID=37.42 |
| exp26_regional_weight | Regional weighting | 0.007396 | FID=36.45 |
| exp27_plateau | Plateau scheduler | **0.006469** | 375/500 epochs, still improving |

## 2D Summary Statistics

- Total: 89 experiments, 82 completed, 2 failed, 6 incomplete
- Best val loss: 0.005604 (SiT-S exp12_1c)
- Mean best val loss: 0.007477
- Range: 0.005604 - 0.008146
- Zero gradient spikes or divergence

---

# 3D Diffusion Experiments

## 3D Quick Reference: Best Results

| Category | Experiment | Val Loss | Train MSE | Epochs | Notes |
|----------|-----------|----------|-----------|--------|-------|
| **Bravo pixel 128** | exp8 (EMA) | 0.00227 | 0.00129 | 500 | Best 128x128 |
| **Bravo pixel 256** | exp1_1 (run2) | 0.00211 | 0.00284 | 500 | Best 256x256 |
| **Bravo DiT pixel** | exp7 (2000ep) | 0.00234 | 0.00312 | 2000 | DiT-B patch=8, 134M |
| **Seg pixel 128** | exp2 (run1) | 0.000373 | 0.000351 | 500 | Only stable 128 seg run |
| **Seg pixel 256** | exp2b_1 (input) | 0.000336 | 0.000289 | 500 | Best seg overall |
| **LDM 4x** | exp9_1 (mid UNet) | 0.0764 | 0.025 | 354 | **Still improving**, hit time limit |
| **LDM 8x** | exp9_ldm_8x (run1) | 0.177 | 1.009 | 500 | Collapsed late training |

## 3D Experiment Index

| ID | Type | Resolution | Architecture | Params | Strategy | Mode |
|----|------|-----------|-------------|--------|----------|------|
| exp1 | Pixel | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp1_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp1_chained | Pixel (chain test) | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp2 | Pixel | 128x128x160 | UNet 5L | 270M | rflow | seg_conditioned |
| exp2_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | seg_conditioned |
| exp2b | Pixel | 128x128x160 | UNet 5L | 270M | rflow | seg_cond_input |
| exp2b_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | seg_cond_input |
| exp2c | Pixel | 128x128x160 | UNet 5L | 271M | rflow | seg_cond_improved |
| exp2c_1 | Pixel | 256x256x160 | UNet 5L | 271M | rflow | seg_cond_improved |
| exp4 | Pixel+SDA | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp5_1 | Pixel+ScoreAug | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp6a | Pixel+ControlNet | 128x128x160 | UNet 5L | 270M | rflow | bravo (uncond) |
| exp7 | Pixel DiT | 128x128x160 | DiT-B p=8 | 135M | rflow | bravo |
| exp8 | Pixel+EMA | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp9 | LDM 4x | latent 64x64x40 | UNet 4L | 3.48B | rflow | bravo_seg_cond |
| exp9_0 | LDM 8x small | latent 20x32x32 | UNet 3L | 167M | rflow | bravo_seg_cond |
| exp9_1 | LDM 4x mid | latent 64x64x40 | UNet 4L | 666M | rflow | bravo_seg_cond |
| exp10_1 | LDM DC-AE 8x8 | latent 8x8x32 | SiT-B p=1 | 137M | rflow | bravo_seg_cond |
| exp10_2 | LDM DC-AE 4x4 | latent 4x4x128 | SiT-B/DiT-B | 132M | rflow | bravo_seg_cond |
| exp10_3 | LDM DC-AE 2x2 | latent 2x2x512 | SiT-B/DiT-B | 131M | rflow | bravo_seg_cond |
| exp11 | S2D pixel | 128x128x160 | UNet 5L | 1.42B | rflow | bravo |
| exp11_1 | S2D pixel | 256x256x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12 | Wavelet pixel | 128x128x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12_1 | Wavelet pixel | 256x256x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12_2 | Wavelet WDM | 128x128x160 | WDM 5L | ~77M | ddpm x0 | bravo |
| exp12_3 | Wavelet WDM | 128x128x160 | WDM 5L+attn | ~77M | rflow | bravo |
| exp12_4 | Wavelet WDM | 128x128x160 | WDM 5L+attn | ~77M | ddpm x0 | bravo |
| exp13 | LDM DiT 4x | latent 64x64x40 | DiT-S p=2 | 40M | rflow | bravo_seg_cond |

## 3D Part 1: Pixel-Space Bravo Experiments

### exp1: Pixel Bravo Baseline @ 128x128x160

**Config**: UNet 5L [32,64,256,512,512], rflow, LR=1e-4, batch=1, 500 epochs

| Run | Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|-----|--------|----------|-----------|---------|-------|------|
| Run 1 | 23969657 | A100-SXM4 | 500 | 0.0026 | 0.0012 | 89s | 12.7h | N/A |
| Run 2 | 23989011 | A100-SXM4 | 500 | 0.00246 | 0.00146 | 70s | 10.0h | 4.1/19.6GB |

**Run 1 test results** (best model): MSE=0.003141, PSNR=32.49, MS-SSIM=0.8196
**Run 1 test results** (latest): MSE=0.007305, PSNR=30.45, MS-SSIM=0.8730
**Run 2 test**: Failed (torch.load OmegaConf bug)

### exp1_1: Pixel Bravo @ 256x256x160

**Config**: UNet 5L [32,64,256,512,512], rflow, LR=1e-4, batch=1, 500 epochs

| Run | Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|-----|--------|----------|-----------|---------|-------|------|
| Run 1 | 23969658 | A100-SXM4 | 500 | 0.0022 | 0.0029 | 284s | 41h | N/A |
| Run 2 | 23989010 | A100-SXM4 | 500 | 0.00211 | 0.00284 | 224s | 31.4h | 3.6/43.0GB |

**Run 1 test results** (best): MSE=0.003494, PSNR=32.81, MS-SSIM=0.9090
256x256 is better than 128x128: Lower val loss (0.0021 vs 0.0025), higher MS-SSIM (0.909 vs 0.820).

### exp1_chained: Auto-Chain Test @ 128x128x160

**Fresh 100-epoch run from scratch** (NOT a continuation of exp1). Used to test auto-chaining with 30-minute SLURM segments.
14 chain segments, reached epoch 100/100. Best val loss: 0.00300. Final MSE: 0.003279.
Not comparable to exp1 (500 epochs) -- this only trained for 100 epochs total.

### exp4: Pixel Bravo + SDA @ 128x128x160

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|--------|----------|-----------|---------|-------|------|
| 23989012 | A100-SXM4 | 500 | 0.00245 | 0.00208 | 122s | 17.6h | 5.1/22.6GB |

~70% slower per epoch than baseline. Slightly worse val loss.

### exp5_1: Pixel Bravo + ScoreAug @ 128x128x160

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | VRAM |
|-----|-----|--------|----------|-----------|---------|------|
| 23989013 | A100-PCIe | 500 | 0.00213 | 0.00269 | 104s | 5.1/19.6GB |

**Second best 128x128 val loss** (0.00213).

### exp6a: Pixel Bravo ControlNet Stage 1 @ 128x128x160

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | VRAM |
|-----|-----|--------|----------|-----------|---------|------|
| 23989014 | A100-PCIe | 500 | 0.00234 | 0.00186 | 77s | 3.3/19.5GB |

Converged well. Generation metrics broken (channel mismatch).

### exp7: DiT-B Pixel Bravo @ 128x128x160

**Config**: DiT-B (768 hidden, 12 depth, 12 heads), patch=8, 5120 tokens, rflow, LR=1e-4

| Job | Epochs | Best Val | Final MSE | Time/ep | VRAM |
|-----|--------|----------|-----------|---------|------|
| 23989812 | 2000 | 0.00234 | 0.00312 | 34s | 2.5/6.1GB |
| 23989813 | 500 | 0.00289 | 0.00466 | 33s | 2.5/6.1GB |

DiT-B needs ~2000 epochs to match UNet at 500. But per-epoch cost is 34s vs 70-284s, VRAM is 6GB vs 19-43GB.

### exp8: Pixel Bravo + EMA @ 128x128x160

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | VRAM |
|-----|-----|--------|----------|-----------|---------|------|
| 23991162 | A100-PCIe | 500 | **0.00227** | **0.00129** | 77s | 5.1/20.6GB |

**Best 128x128 experiment overall.**

### 3D Bravo Val Loss Ranking

| Rank | Experiment | Val Loss | Resolution |
|------|-----------|----------|------------|
| 1 | exp1_1 (run2) | 0.00211 | 256x256 |
| 2 | exp5_1 ScoreAug | 0.00213 | 128x128 |
| 3 | exp1_1 (run1) | 0.0022 | 256x256 |
| 4 | exp8 EMA | 0.00227 | 128x128 |
| 5 | exp7 DiT 2000ep | 0.00234 | 128x128 |
| 6 | exp6a ControlNet S1 | 0.00234 | 128x128 |
| 7 | exp4 SDA | 0.00245 | 128x128 |
| 8 | exp1 (run2) | 0.00246 | 128x128 |
| 9 | exp1 (run1) | 0.0026 | 128x128 |
| 10 | exp7 DiT 500ep | 0.00289 | 128x128 |

## 3D Part 2: Pixel-Space Seg-Conditioned Experiments

### Stability Issue: 128x128 + LR=1e-4 Diverges

Three of four 128x128 seg_conditioned runs at LR=1e-4 diverged at epoch 34-37:
- exp2 run2 (23996720): Diverged ep35, best 0.001445
- exp2b (23996776): Diverged ep37, best 0.001358
- exp2c (24031953): Diverged ep35, best 0.747

### exp2: Seg Size-Bin Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 128 | 1e-4 | 23972121 | 500 | 0.000373 | 0.000351 | OK |
| run2 | 128 | 1e-4 | 23996720 | 443 | 0.001445 | 1.000 | **DIVERGED ep35** |
| exp2_1 | 256 | 5e-5 | 23996728 | 500 | **0.000337** | 0.000292 | OK |

### exp2b: Seg Input Channel Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| exp2b | 128 | 1e-4 | 23996776 | 500 | 0.001358 | 1.000 | **DIVERGED ep37** |
| exp2b_1 | 256 | 5e-5 | 23997358 | 500 | **0.000336** | 0.000289 | OK |

### exp2c: Improved Seg Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| exp2c | 128 | 1e-4 | 24031953 | 500 | 0.747 | 1.001 | **DIVERGED** + ckpt fail |
| exp2c_1 | 256 | 1e-4 | 24039864 | 25 | N/A | 0.00163 | **No ckpts saved** (2GB limit) |

### 3D Seg Summary
- **256x256 + LR=5e-5 is stable**: exp2_1 and exp2b_1 both converge to ~0.000337
- **128x128 + LR=1e-4 is unstable**: 3/4 runs diverge at epoch ~35
- exp2b_1 (input conditioning) = exp2_1 (size-bin) at 256x256

## 3D Part 3: Latent Diffusion (LDM) Experiments

### exp9: VQ-VAE 4x Latent, Large UNet (3.48B params)

Latent shape: [B, 8, 40, 64, 64]

| Run | Job | GPU | Epochs | Best Val | Final MSE | Grad Spikes | VRAM |
|-----|-----|-----|--------|----------|-----------|-------------|------|
| run1 | 23997507 | A100-SXM4 | 500 | 0.0950 | 1.032 | 0 | 53/68GB |
| run2 | 24039868 | H100 | 302 | 0.0875 | 0.880 | **26,387** | 53/67GB |

Both collapsed (MSE -> ~1.0). Run2 had catastrophic gradient spikes.

### exp9_1: VQ-VAE 4x Latent, Mid UNet (666M params) -- BEST NON-PIXEL

**Config**: UNet 4L [128,256,512,1024], warmup 10 epochs, grad clip 0.5

| Job | GPU | Epochs | Best Val | Final MSE | Grad Spikes | VRAM |
|-----|-----|--------|----------|-----------|-------------|------|
| 24036417 | A100-PCIe | 354 | **0.0764** | **0.025** | 0 | 11/65GB |

Zero gradient spikes. MSE still decreasing. Hit 14h SLURM time limit. **Needs continuation.**

### exp9: VQ-VAE 8x Latent, Large UNet (3.47B params)

| Run | Job | Epochs | Best Val | Final MSE | Grad Spikes |
|-----|-----|--------|----------|-----------|-------------|
| run1 | 23997506 | 500 | 0.177 | 1.009 | 0 |
| run2 | 24039867 | 335 | 0.185 | 0.935 | **31,363** |

8x is worse than 4x (0.177 vs 0.095).

### exp9_0: VQ-VAE 8x Latent, Small UNet (167M params)

| Run | Job | Epochs | Best Val | Final MSE | VRAM |
|-----|-----|--------|----------|-----------|------|
| run1 | 23997680 | 100 | 0.198 | 0.141 | 3.8/8.1GB |
| run2 | 23997808 | 100 | 0.192 | 0.141 | 3.8/18.9GB |

Quick test. Clean runs at 8x.

### LDM Pattern
Large UNets (3.48B) collapse; mid-size (666M) with warmup + grad clip works.

## 3D Part 4: DiT/SiT Latent Experiments

### exp10_1: SiT-B + DC-AE 8x8x32

| Job | Epochs | Best Val | VRAM | Status |
|-----|--------|----------|------|--------|
| 23998878 | 500 | 3.146 | 5.9/70.1GB | Completed but poor |

DC-AE 8x8x32 latent too compressed for meaningful generation.

### exp10_2, exp10_3: DC-AE 4x4x128 and 2x2x512 -- CRASHED

Both crashed instantly. Architecture mismatch with model patch embedding.

### exp13: DiT-S + VQ-VAE 4x (submitted, not yet run)

DiT-S (384 hidden, 12 layers), patch=2, 20,480 tokens, use_compile=true, bravo_seg_cond.

## 3D Part 5: S2D and Wavelet Experiments -- ALL FAILED

**ALL S2D and wavelet experiments collapsed to MSE ~1.0** (mean prediction).

### exp11/11_1: S2D @ 128 and 256
- run1 128: 500ep, best_val 0.905, mean collapse
- run1 256: 64ep, best_val 0.909, disk quota crash
- retry runs: OOM killed (1.42B params + 32GB RAM limit)

### exp12/12_1: Wavelet @ 128 and 256
- run1 128: 49ep, best_val 0.905, disk quota crash
- run1 256: 500ep, best_val 1.001, 738 ckpt failures
- retry runs: OOM killed + 571 gradient spikes (exp12_1 run2)

### Why S2D/Wavelet Failed
1. LR=1e-4 too high for 1.42B params in 8-channel space
2. No warmup or gradient clipping (exp9_1 succeeded with both)
3. Wavelet value range mismatch between subbands
4. Once MSE hits ~1.0, recovery is impossible

### exp12_2/12_3/12_4: WDM-style Wavelet (not yet run)
Use smaller WDM model (~77M) with attention at L3/L4. Submitted but pending.

---

# 2D Compression Experiments

## 2D VAE

| Experiment | Description | Epochs | MS-SSIM | PSNR | Val G Loss | Notes |
|-----------|------------|--------|---------|------|------------|-------|
| exp1 progressive | Progressive VAE | 200/200 | 0.997 | 41.70 | 0.002374 | Best VAE PSNR |
| exp2 finetune | Finetuned VAE | 100/100 | 0.996 | 39.92 | -- | |
| exp3 multimodal 256 | Multi-modal 256 | 125/125 | 0.995 | -- | 0.002984 | |
| exp4 multimodal 64lat | Multi-modal 64 lat | 125/125 | **0.999** | -- | **0.001356** | Best VAE overall |
| exp5 multimodal bf16 | Multi-modal BF16 | 125/125 | 0.976 | -- | 0.006149 | BF16 hurts quality |

**Key finding**: exp4 with 64-dim latent is the best 2D VAE (SSIM 0.999, lowest val loss).

## 2D VQ-VAE

| Experiment | Description | Epochs | MS-SSIM | Val G Loss | Notes |
|-----------|------------|--------|---------|------------|-------|
| exp6 vqvae 8x | VQ-VAE 8x | 125/125 | 0.994 | 0.003703 | |
| exp6_1 vqvae 4x | VQ-VAE 4x | 125/125 | **0.997** | **0.002725** | Better at 4x |

## 2D DC-AE (Multi-Modal)

| Experiment | Compression | Epochs | MS-SSIM | PSNR | Best Loss | Notes |
|-----------|-------------|--------|---------|------|-----------|-------|
| exp9_1 f32 | 32x | 125/125 | 0.984 | 34.95 | -- | |
| exp9_1 f32 phase3 | 32x phase3 | 175/175 | 0.982 | 33.98 | -- | |
| exp9_1 f32 1.5x | 32x, 1.5x sched | 125/125 | 0.973 | 33.50 | -- | 1.5x schedule worse |
| exp9_2 f64 | 64x | 125/125 | **0.996** | 39.84 | -- | Best DC-AE SSIM |
| exp9_2 f64 1.5x | 64x, 1.5x sched | 125/125 | 0.989 | 36.57 | 0.0056 | |
| exp9_3 f128 | 128x | 120/125 | **0.997** | **40.76** | 0.0032 | **Best DC-AE PSNR** |
| exp9_3 f128 1.5x | 128x, 1.5x sched | 125/125 | 0.987 | 35.76 | 0.0060 | |
| exp9_4 f128 LPIPS | 128x + LPIPS | 109/125 | 0.996 | **41.09** | 0.0039 | Best PSNR with LPIPS |

**Key findings**:
- Higher compression ratios (f128) actually give better PSNR than lower (f32): 40.76 vs 34.95
- 1.5x learning rate schedule consistently degrades quality vs default
- LPIPS loss (exp9_4) achieves highest PSNR (41.09) at f128

## 2D DC-AE Reruns (Feb 2026, 23998xxx)

| Experiment | Compression | Epochs | Best Loss | Notes |
|-----------|-------------|--------|-----------|-------|
| exp9_1 f32 1.5x rerun | 32x | incomplete | 0.0082 | |
| exp9_2 f64 1.5x rerun | 64x | 105/125 | 0.0104 | |
| exp9_3 f128 1.5x rerun | 128x | incomplete | 0.0051 | |

## MAISI VAE Evaluation (Pretrained, No Training)

| Split | PSNR | MS-SSIM |
|-------|------|---------|
| Test (new) | 26.95 +/- 0.35 | 0.946 +/- 0.007 |
| Val | 27.08 +/- 0.36 | 0.950 +/- 0.008 |
| Train | 27.17 +/- 0.40 | 0.946 +/- 0.010 |

MAISI pretrained VAE is significantly worse than our trained models (PSNR 27 vs 35-41).

---

# 3D Compression Experiments

## 3D VAE

| Experiment | Job | Epochs | MS-SSIM | PSNR | Best Loss | Notes |
|-----------|-----|--------|---------|------|-----------|-------|
| exp7 (run1) | 23877660 | 125/125 | -- | 0.45 | 0.005399 | Early version, broken metrics |
| exp7 (run2) | 23879377 | 125/125 | 0.000 | 34.17 | 0.006143 | SSIM computation broken |
| exp7 (run3) | 23883787 | 125/125 | **0.948** | **30.43** | -- | Fixed SSIM, best VAE 3D |

## 3D VQ-VAE 4x (256 -> 64 spatial)

| Experiment | Job | Latent Dim | MS-SSIM | MS-SSIM-3D | PSNR | Notes |
|-----------|-----|-----------|---------|------------|------|-------|
| exp8 (run1) | 23877682 | 4 | -- | -- | 0.50 | Broken metrics |
| exp8 (run2) | 23878983 | 4 | 0.000 | -- | 39.59 | Broken SSIM |
| **exp8_1** | 23884814 | **4** | **0.995** | **0.997** | **39.88** | **Best 4x** |
| exp8_2 | 23884815 | 8 | 0.995 | 0.997 | 39.62 | |
| exp8_3 | 23884816 | 16 | 0.994 | -- | 39.26 | |

**exp8_1 (latent_dim=4) is the best 3D VQ-VAE 4x.** Used by all LDM experiments.

## 3D VQ-VAE 8x (256 -> 32 spatial)

| Experiment | Job | Variant | MS-SSIM | MS-SSIM-3D | PSNR | Notes |
|-----------|-----|---------|---------|------------|------|-------|
| exp8_5 | 23885962 | lat4 | 0.981 | 0.987 | 35.65 | |
| exp8_6 | 23885967 | lat8 | 0.980 | 0.986 | 35.54 | |
| **exp8_7** | 23885965 | **lat16** | **0.982** | **0.987** | **35.93** | **Best 8x** |
| exp8_8 | 23885970 | lat32 | 0.981 | 0.987 | 35.58 | |
| exp8_9 | 23885968 | lat64 | 0.982 | 0.987 | 35.85 | |
| **exp8_10** | 23889692 | **wide** | **0.984** | -- | **36.16** | **Best 8x overall** |
| exp8_11 | 23889693 | cb1024 | 0.983 | -- | 35.85 | |
| exp8_12 | 23889696 | deeper | 0.983 | -- | 35.74 | |
| exp8_13 | 23889697 | highperc | 0.978 | -- | 34.83 | High perceptual loss hurts |
| exp8_14 | 23889698 | combined | 0.983 | -- | 35.72 | |

4x compression is clearly superior: PSNR 39.88 vs 36.16 (best 8x).

## 3D DC-AE

| Experiment | Job | Epochs | MS-SSIM | MS-SSIM-3D | PSNR | Best Loss |
|-----------|-----|--------|---------|------------|------|-----------|
| exp10_1 default | 23888483 | 90/125 | 0.935 | 0.935 | 30.91 | 0.0237 |

Hit time limit at epoch 90. DC-AE 3D significantly worse than VQ-VAE 3D (PSNR 30.9 vs 39.9).

## 3D Compression Summary

| Method | Compression | PSNR | MS-SSIM | Status |
|--------|-------------|------|---------|--------|
| VQ-VAE 4x (exp8_1) | 4x | **39.88** | **0.995** | Production-ready |
| VQ-VAE 8x (exp8_10) | 8x | 36.16 | 0.984 | Good |
| VAE 3D (exp7 run3) | ~4x | 30.43 | 0.948 | Mediocre |
| DC-AE 3D (exp10_1) | varies | 30.91 | 0.935 | Poor (incomplete) |
| MAISI pretrained | 4x | 26.95 | 0.946 | Baseline reference |

---

# Segmentation Compression

## 2D Seg Compression (DC-AE)

| Experiment | Compression | Epochs | Dice (Test Best) | IoU | Notes |
|-----------|-------------|--------|-------------------|-----|-------|
| exp11_1 DC-AE seg | default | 125/125 | 0.9982 | 0.9964 | |
| exp13_1 DC-AE seg f32 | 32x | 500/500 | 0.9999 | 0.9998 | |
| **exp13_2 DC-AE seg f64** | **64x** | 500/500 | **1.0000** | **0.9999** | **Perfect** |
| exp13_3 DC-AE seg f128 | 128x | 500/500 | 0.9999 | 0.9999 | |

**2D seg compression is solved.** All DC-AE variants achieve Dice >= 0.998.

## 3D Seg Compression (VQ-VAE)

| Experiment | Architecture | Epochs | Dice (Test Best) | IoU | Notes |
|-----------|-------------|--------|-------------------|-----|-------|
| exp11_2 VQ-VAE 3D seg | VQ-VAE | 125/125 | 0.904 | 0.827 | Significant quality gap vs 2D |

3D seg compression needs work: 0.904 vs 0.999+ for 2D.

---

# Downstream Segmentation

## Completed 3D Experiments

### Exp1: Baseline 3D (BRAVO only, 1ch) -- 500/500 epochs, 10.3h

| Metric | Last Val | Test |
|--------|----------|------|
| Dice | 0.159 | **0.194** |
| IoU | -- | 0.150 |
| Precision | 0.591 | 0.485 |
| Recall | 0.397 | 0.328 |
| HD95 | 26.0 | 29.15 |
| dice_tiny | 0.134 | 0.160 |
| dice_small | 0.214 | 0.324 |
| dice_medium | 0.598 | 0.412 |
| dice_large | 0.634 | 0.823 |

Heavy overfitting: train loss 0.12, val dice plateaued ~0.16.

### Exp2: Dual 3D (T1 pre + T1 gd, 2ch) -- 500/500 epochs, 10.5h

| Metric | Last Val | Test |
|--------|----------|------|
| Dice | 0.151 | **0.185** |
| IoU | -- | 0.143 |
| Precision | 0.591 | 0.483 |
| Recall | 0.412 | 0.306 |
| HD95 | 32.6 | 23.97 |
| dice_tiny | 0.120 | 0.151 |
| dice_small | 0.239 | 0.314 |
| dice_medium | 0.598 | 0.403 |
| dice_large | 0.677 | **0.877** |

Dual slightly worse overall but better HD95 (24.0 vs 29.2) and large tumor Dice.

## Incomplete 2D Experiments

### Exp1: Baseline 2D (BRAVO only) -- 237/500 epochs (cut off)

| Metric | Last Val |
|--------|----------|
| Dice | 0.252 |
| dice_small | 0.343 |
| dice_medium | 0.664 |
| dice_large | 0.763 |

### Exp2: Dual 2D (T1 pre + T1 gd) -- 159/500 epochs (cut off)

| Metric | Last Val |
|--------|----------|
| Dice | 0.259 |
| dice_small | 0.390 |
| dice_medium | 0.660 |
| dice_large | 0.774 |

2D slices have higher val Dice (~0.25) vs 3D (~0.16). Both 2D experiments need resubmission with longer SLURM time.

---

# 3D Generation & Evaluation

## Generated Datasets

| Experiment | Resolution | Samples | Invalid Mask Retries | Time |
|-----------|-----------|---------|---------------------|------|
| exp1 gen 128 | 128x128x150 | 100 | 14 (1 failed 10x) | ~55 min |
| exp1_1 gen 256 | 256x256x150 | 100 | 1 | ~113 min |

256x256 generation is much more stable (1 retry vs 14).

## Bin Adherence Evaluation (exp2b_1 input conditioned)

| Metric | CFG=1.0 |
|--------|---------|
| Exact bin match | 1.5% |
| Tumor presence | 92.3% |
| Per-bin accuracy | 42.6% |
| MAE | 1.25 |

**Bin adherence is poor.** The model generates tumors (92.3% presence) but cannot reliably control which size bins they fall in.

---

# 3D Sampling Improvement Evaluations (Feb 19-20, 2026)

All evaluations use the **exp1_1 bravo pixel 256x256x160** model (best 3D bravo).
Generated 25 volumes per configuration, evaluated against all reference splits.
Metrics: FID (ResNet50), KID (ResNet50), CMMD (BiomedCLIP).

## Baseline Euler Step Sweep

| Steps | FID (all) | KID (all) | CMMD (all) | FID (train) | Time/vol |
|-------|-----------|-----------|------------|-------------|----------|
| 10 | 34.20 | 0.0307 | 0.149 | 34.33 | 7.3s |
| **25** | **27.50** | **0.0238** | 0.113 | **26.77** | 18.1s |
| 50 | 30.25 | 0.0265 | **0.109** | 29.04 | 36.1s |

**Key finding**: 25 steps is optimal for FID/KID. Beyond 25, FID *degrades* due to ODE discretization error accumulation. CMMD marginally improves at 50 steps but FID gets 10% worse.

## Full ODE Solver Comparison (Not Yet Run)

`eval_ode_solvers.py` tests a comprehensive solver grid (33 configs total):
- **Fixed-step**: euler, midpoint, heun2, heun3, rk4 at [5, 10, 25, 50, 100] steps
- **Adaptive**: fehlberg2, bosh3, dopri5, dopri8 at tol=[1e-2, 1e-3]

SLURM job (24057196) crashed before producing results. Needs resubmission.
Script: `IDUN/eval/eval_ode_solvers.slurm`

## DiffRS (Discriminator-Guided Reflow)

**Paper**: DiffRS trains a discriminator to estimate density ratios between real and generated distributions, then uses these ratios to correct ODE trajectories during inference.

**DiffRS head training (job 24059814)**: Trained on 105 train volumes + 105 generated. 952K param discriminator, 100 epochs.
- Train accuracy converged to ~98% but val accuracy plateaued at ~82-85%
- Best val loss: 0.3634 (epoch 12), heavily overfitting by epoch 20
- **Root cause**: Dataset too small (~200 total volumes). Discriminator memorizes rather than learning meaningful density ratios.

### DiffRS Run 1: Full Correction

| Config | NFE/vol | FID (all) | CMMD (all) | vs Baseline |
|--------|---------|-----------|------------|-------------|
| DiffRS/10 | 17 (1.7x) | 36.56 | 0.157 | FID +2.4 (worse) |
| DiffRS/25 | 81 (3.2x) | 37.67 | 0.118 | FID +10.2 (much worse) |
| DiffRS/50 | 134 (2.7x) | 53.61 | 0.116 | FID +23.4 (catastrophic) |

Full correction is catastrophically worse. NFE explosion (3.2x at 25 steps) and FID degradation. The discriminator over-corrects.

### DiffRS Run 2: Lightweight Correction (Reduced Strength)

| Config | NFE/vol | FID (all) | KID (all) | CMMD (all) | vs Baseline |
|--------|---------|-----------|-----------|------------|-------------|
| DiffRS/10 | 14 (1.4x) | 34.31 | 0.0305 | 0.149 | FID +0.1 |
| DiffRS/25 | 35 (1.4x) | 27.88 | 0.0240 | 0.113 | FID +0.4 |
| DiffRS/50 | 70 (1.4x) | 31.54 | 0.0278 | 0.109 | FID +1.3 |

Lightweight correction: marginally worse FID everywhere. CMMD essentially unchanged. **No improvement at any operating point.** 1.4x NFE overhead for nothing.

### DiffRS Conclusion

**DiffRS failed for our dataset.** The discriminator cannot learn meaningful density ratios from ~200 training volumes. This is a fundamental dataset size limitation, not an implementation issue. DiffRS is designed for large-scale datasets (ImageNet, CIFAR) where the discriminator can learn general distributional differences.

## Restart Sampling (Xu et al., NeurIPS 2023)

**Paper**: Purely algorithmic ODE improvement. Alternates between adding forward noise and running backward ODE within a restart interval [tmin, tmax]. Contracts accumulated errors via stochasticity while maintaining ODE-level accuracy. No auxiliary model needed.

**Why we tried it**: Our model shows the classic error accumulation pattern (FID degrades beyond 25 steps), which is exactly what Restart fixes in the paper.

### Results (vs 'all' reference, 25 main Euler steps)

| Config | NFE/vol | FID | KID | CMMD | vs Euler/25 |
|--------|---------|-----|-----|------|-------------|
| Euler/25 (baseline) | 25 | 27.50 | 0.0234 | 0.113 | -- |
| Restart K=1, n=3, [0.1, 0.3] | 28 | 32.83 | 0.0311 | 0.113 | FID +5.3 (worse) |
| Restart K=1, n=5, [0.1, 0.3] | 30 | 27.50 | 0.0244 | 0.112 | FID +0.0 (same) |
| Restart K=2, n=3, [0.1, 0.3] | 31 | -- | -- | -- | (job timed out) |

### Results (vs 'train' reference — most favorable split)

| Config | NFE/vol | FID | KID | CMMD |
|--------|---------|-----|-----|------|
| Euler/25 | 25 | 26.77 | 0.0216 | 0.109 |
| Restart K=1, n=5 | 30 | 26.73 | 0.0214 | 0.109 |

### Restart Conclusion

**No meaningful improvement.** Best restart config (K=1, n=5 at 30 NFE) matches baseline Euler/25 FID to within noise (27.50 vs 27.50 on 'all', 26.73 vs 26.77 on 'train'). The K=1, n=3 config with fewer restart steps is significantly worse (FID 32.83), suggesting the restart backward ODE needs enough steps to be useful. But adding those steps just matches the baseline at higher cost.

**Hypothesis**: Our model's error accumulation pattern differs from what Restart targets. Restart fixes *discretization error* in the ODE solver, but our model's quality degradation beyond 25 steps may be due to the model itself (imperfect velocity field) rather than solver discretization.

## Sampling Improvement Summary

| Method | Best FID | NFE | vs Baseline 25-step | Verdict |
|--------|----------|-----|---------------------|---------|
| Euler (baseline) | **27.50** | 25 | -- | **Best** |
| DiffRS (lightweight) | 27.88 | 35 | +0.38 FID, 1.4x cost | No improvement |
| DiffRS (full) | 37.67 | 81 | +10.17 FID, 3.2x cost | Catastrophic |
| Restart K=1, n=5 | 27.50 | 30 | +0.00 FID, 1.2x cost | No improvement |
| Restart K=1, n=3 | 32.83 | 28 | +5.33 FID, 1.1x cost | Worse |

**Conclusion**: For our small-dataset 3D medical imaging setting, plain Euler at 25 steps is optimal. Neither discriminator-based correction (DiffRS) nor stochastic restart (Restart Sampling) provides any benefit. This is likely because: (1) dataset is too small for DiffRS discriminator training, and (2) the error pattern is model-quality-limited rather than solver-limited.

---

## 3D Seg Generation Evaluation (exp14_1)

### Bug: Sigmoid Without Thresholding

The `find_optimal_steps` evaluation for exp14_1 (3D pixel seg model) produced FID ~98 flat across all step counts (25-40 steps), which contradicted training metrics (CMMD 0.02, KID 6e-4).

**Root cause**: The evaluation script applied `torch.sigmoid(result)` but never thresholded to binary masks. Training metrics apply `(sample > 0.5).float()` after sigmoid. This meant evaluation compared soft probability maps (continuous 0-1 values) against binary reference masks, producing meaningless FID scores.

**Fix**: Added `> 0.5` threshold after sigmoid in `eval_ode_solvers.py` (line 514). Results before fix are invalid.

| Steps | FID (all) | KID | CMMD | Notes |
|-------|-----------|-----|------|-------|
| 25-40 | ~98 | ~0.151 | ~0.74 | **INVALID** (no threshold) |

Needs re-evaluation after fix.

---

## DC-AE 1.5 Structured Latent Space (Feb 20, 2026)

### Implementation Verification

DC-AE 1.5 (structured latent space) implementation verified correct:
- `StructuredAutoencoderDC` wrapper replaces encoder.conv_out and decoder.conv_in with adaptive weight-slicing layers
- `AdaptiveOutputConv2d` / `AdaptiveInputConv2d` slice weight tensors for variable channel counts
- Gradient isolation confirmed: inactive channels receive zero gradient
- Shortcut path handling verified for both encoder and decoder
- All 1110 existing tests pass, plus dedicated functional tests

### 2D DC-AE 1.5 SLURM Jobs (Ready for Submission)

| Experiment | Compression | Latent Shape | Notes |
|-----------|-------------|-------------|-------|
| exp9_1 f32 1.5 | 32x | 8x8x32 | Paper says NOT recommended for c=32 (comparison) |
| exp9_2 f64 1.5 | 64x | 4x4x128 | Paper recommends for c>=64 |
| exp9_3 f128 1.5 | 128x | 2x2x512 | Paper recommends for large channels |

All three SLURM files have auto-chaining enabled (12h segments, max 20 chains).

---

# Infrastructure Issues

### Checkpoint Corruption (Feb 10 runs, 24031xxx)
All jobs suffered `inline_container.cc:664` filesystem errors. ~1,768 combined failures.

### 2GB File Size Limit (Feb 13 runs)
Models >~1B params produce checkpoints exceeding 2GB. Affected S2D/wavelet retries, exp2c_1.

### OOM System Kills (Feb 13 runs)
5 runs OOM-killed: 1.42B param models with `--mem=32G` insufficient (not CUDA OOM, system OOM).

### Gradient Spike Catastrophe (LDM reruns)
exp9 4x rerun: 26K spikes. exp9 8x rerun: 31K spikes. Original runs had zero. Possibly seed-dependent.

### torch.load OmegaConf Bug
Newer-codebase runs fail test eval: `WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig`

### 3D MS-SSIM Bug
Early 3D compression runs (exp7, exp8 run1/run2) had broken MS-SSIM (0.0 or N/A) due to expecting 4D tensors but getting 5D. Fixed in later experiments.

---

# Key Takeaways

## What Works
1. **2D SiT-S** is the best 2D architecture (val 0.0056, FID 33.43)
2. **3D Pixel UNet at 256x256** is the most reliable 3D approach (val 0.0021)
3. **VQ-VAE 3D 4x (exp8_1)** is production-ready compression (PSNR 39.88, SSIM 0.995)
4. **2D seg compression is solved** (Dice 1.0 with DC-AE)
5. **EMA helps** for both 2D and 3D training
6. **25 Euler steps** is optimal for 3D bravo generation (FID 27.50)

## What Doesn't Work
1. **S2D and wavelet 3D**: All collapsed to mean prediction
2. **DC-AE 3D latent + DiT/SiT**: Architecture mismatch crashes or poor quality
3. **Large LDM UNets (3.48B)**: Collapse or gradient catastrophe
4. **128x128 seg_conditioned at LR=1e-4**: 3/4 runs diverge
5. **1.5x learning rate schedule for DC-AE**: Consistently worse
6. **DiffRS**: Discriminator overfits on small dataset (~200 volumes), no quality improvement
7. **Restart Sampling**: Matches baseline at best, no improvement on our error pattern

## Open Questions
1. Can exp9_1 LDM (666M, val 0.076 at ep354) reach pixel-space quality with more epochs?
2. Can WDM-style wavelet (exp12_2/12_3/12_4) avoid mean collapse with smaller model?
3. Can DiT-S + VQ-VAE 4x (exp13) achieve good quality with fast compile?
4. Should downstream segmentation use generated data augmentation?
5. Does DC-AE 1.5 structured latent improve reconstruction quality for 2D?
6. What does the seg generation evaluation look like with the fixed threshold?
