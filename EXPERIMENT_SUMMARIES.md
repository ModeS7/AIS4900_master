# MedGen Experiment Summaries

Generated from `runs_tb/` TensorBoard event files via `scripts/extract_tb.py` →
`runs_tb_extracted.json`. Covers **391 training runs** across 19 subdirectories,
Dec 2025 – Apr 2026.

Per-experiment entries report:
- Purpose parsed from run name and cross-referenced to memory/docs when
  applicable.
- Training timeline (start → end, wall-clock duration, epochs reached, TFLOPs,
  peak VRAM).
- Loss dynamics — train/val MSE, perceptual, regional breakdown, per-timestep
  bucketing.
- Generation metrics — per-epoch KID/CMMD/FID and extended-evaluation variants.
- Validation-quality metrics — LPIPS, PSNR, MS-SSIM, MS-SSIM-3D per modality.
- Diversity metrics when logged (inter-sample LPIPS/MSSSIM).
- PCA/morphological metrics when logged.
- Anomalies (NaN, early stop, collapsed diversity, missing expected tags).
- Ranking within the exp-number family when multiple siblings exist.

The accepted **best generation model** as of April 2026 is **exp1_1 @ 1000
epochs** (3D pixel-space RFlow + EMA at 256×256×160), visually confirmed and
100% PCA pass-rate. The runner-up on FID is **exp23 ScoreAug @ 1000 epochs**
(FID 20.38 vs 19.12 post-hoc), but exp23 was shown to damage anatomical
fidelity (~20% broken brain shapes even with real seg masks). Subsequent
effort went into:

- Mean-blur diagnosis (exp1_1 collapses to posterior mean — HF deficit grows
  with t) → exp37 t-weighted LPIPS/FFL fine-tunes.
- Phase-1 spectrum analysis confirming exp37_3 as best mid-to-high-frequency
  recovery (+27% over baseline at the high band).
- IR-SDE restoration network design (post-hoc deblur) — ongoing.
- LaMamba-Diff (Mamba + window attention) pixel-space experiments (exp34).

---

## Category index

| Category | Runs | Focus |
|---|---:|---|
| [diffusion_3d/bravo](#diffusion_3dbravo) | 117 | Primary pixel-space 3D bravo generation |
| [diffusion_3d/bravo_latent](#diffusion_3dbravo_latent) | 17 | Latent diffusion bravo (compressed-space) |
| [diffusion_3d/seg](#diffusion_3dseg) | 17 | Unconditional seg-mask generation |
| [diffusion_3d/restoration](#diffusion_3drestoration) | 10 | IR-SDE deblur / restoration |
| [diffusion_3d/cfg](#diffusion_3dcfg) | 9 | Classifier-free guidance ablations |
| [compression_3d/multi_modality](#compression_3dmulti_modality) | 23 | VAE/VQ-VAE/DC-AE for 3D latent diffusion |
| [compression_3d/seg](#compression_3dseg) | 1 | Seg VQ-VAE |
| [downstream/SegResNet](#downstreamsegresnet) | 18 | Synthetic-data downstream eval (SegResNet) |
| [downstream/nnunet](#downstreamnnunet) | 13 | nnU-Net downstream (3D fullres) |
| [diffusion_2d/bravo](#diffusion_2dbravo) | 78 | 2D pixel-space bravo — historical/ablation |
| [diffusion_2d/multi](#diffusion_2dmulti) | 12 | 2D multi-modality |
| [diffusion_2d/restoration](#diffusion_2drestoration) | 2 | 2D restoration |
| [compression_2d/multi_modality](#compression_2dmulti_modality) | 13 | 2D VAE/VQ-VAE/DC-AE |
| [compression_2d/seg](#compression_2dseg) | 4 | 2D seg compression |
| [compression_2d/progressive](#compression_2dprogressive) | 2 | Progressive growing compression |
| [compression_2d/bravo](#compression_2dbravo) | 1 | 2D bravo compression |
| [diffusion_3d/old-discard](#diffusion_3dold-discard) | 50 | Deprecated/abandoned 3D diffusion runs (bulk list) |
| [downstream/old](#downstreamold) | 4 | Deprecated downstream runs |
| [diffrs/tensorboard](#diffrstensorboard) | 0 | Empty directory |

Legend used in per-experiment blocks:
- `↓` = lower is better; `↑` = higher is better.
- `best @ ep N` = epoch where the metric hit its min/max; `last` = final value.
- **bold** run name = best of its exp-number family on the headline metric.

---
## diffusion_3d/bravo

*117 runs across 25 experiment families.*

### exp1

The **exp1 family** is the long-running baseline and its variants, spanning
Feb → April 2026. Core recipe: 3D UNet (≈270M params), RFlow strategy
(continuous timesteps), EMA, 256×256×160. Variants explored: SNR-γ reweighting
(1e), EDM preconditioning (1f), Huber / pseudo-Huber losses (1g, 1h), offset
noise (1k), adjusted offset noise (1l), global normalization (1m), CFG-Zero*
(1n), ScoreAug+EMA (1i), gradient accumulation (1j), post-hoc EMA (1o),
uniform timestep sampling (1p), attention dropout (1r, 128-only), weight decay
(1s, 128-only), mixup (1t), 156-volume train split (1_1_156), 1000-epoch runs
(1_1_1000, 1_1_1000plus), dual-mode (1v2) and triple-mode (1v3).

**Key finding (memory/finding_exp1_1_1000_best.md):** `exp1_1_1000_pixel_bravo`
at 1000 epochs is the accepted best generation model — 100% PCA pass-rate,
visually-confirmed plausible anatomy, post-hoc FID 19.12 at Euler-27 steps
(best absolute FID). ScoreAug (exp23) has lower in-training FID but
damages anatomical fidelity.

**Mean-blur diagnosis (memory/project_mean_blur_diagnostic.md):** even
exp1_1_1000 collapses to a deterministic posterior mean; HF deficit grows
monotonically with t (1% @ t=0.02 → 72% @ t=0.80). Chained Euler partially
recovers HF at mid/high t. Motivated exp37 t-weighted LPIPS/FFL fine-tunes.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp1r_pixel_bravo_attn_dropout_20260313-142342` — 0.0069
  2. 🥈 `exp1_1_156_pixel_bravo_20260411-024806` — 0.0081
  3.  `exp1v2_2_1000_pixel_dual_20260411-024806` — 0.0087
  4.  `exp1s_01_pixel_bravo_weight_decay_20260403-040937` — 0.0092
  5.  `exp1_1_1000plus_pixel_bravo_20260411-235425` — 0.0102
  6.  `exp1_1_1000_pixel_bravo_20260402-121556` — 0.0108
  7.  `exp1s_02_pixel_bravo_weight_decay_128_20260404-144426` — 0.0120
  8.  `exp1s_pixel_bravo_weight_decay_20260313-141301` — 0.0123
  9.  `exp1g_pixel_bravo_pseudo_huber_20260226-093915` — 0.0134
  10.  `exp1v2_pixel_dual_20260326-164147` — 0.0144
  11.  `exp1p_1_pixel_bravo_uniform_timestep_20260312-014123` — 0.0148
  12.  `exp1v2_1_pixel_dual_joint_norm_20260327-031630` — 0.0150
  13.  `exp1h_pixel_bravo_lpips_huber_20260226-221951` — 0.0151
  14.  `exp1v2_2_156_pixel_dual_20260412-143252` — 0.0155
  15.  `exp1_pixel_bravo_20260224-163719` — 0.0173
  16.  `exp1_1_1000_pixel_bravo_20260301-020650` — 0.0181
  17.  `exp1k_pixel_bravo_offset_noise_20260306-113529` — 0.0189
  18.  `exp1l_pixel_bravo_adjusted_offset_20260306-114524` — 0.0197
  19.  `exp1v2_2_pixel_dual_256_20260404-140254` — 0.0211
  20.  `exp1t_pixel_bravo_mixup_128_20260405-042205` — 0.0234
  21.  `exp1e_pixel_bravo_snr_gamma_rflow_128x160_20260225-015538` — 0.0254
  22.  `exp1l_1_pixel_bravo_adjusted_offset_20260309-202039` — 0.0259
  23.  `exp1t_1_pixel_bravo_mixup_20260402-131808` — 0.0266
  24.  `exp1_1_pixel_bravo_20260301-180232` — 0.0280
  25.  `exp1v3_pixel_triple_20260326-164117` — 0.0309
  26.  `exp1k_1_pixel_bravo_offset_noise_20260311-201606` — 0.0364
  27.  `exp1j_1_pixel_bravo_grad_accum_20260301-015744` — 0.0417
  28.  `exp1v3_2_pixel_triple_256_20260403-035904` — 0.0446
  29.  `exp1g_1_pixel_bravo_pseudo_huber_20260226-034319` — 0.0526
  30.  `exp1h_1_pixel_bravo_lpips_huber_20260226-202457` — 0.0559
  31.  `exp1b_20260221-172156` — 0.0600
  32.  `exp1v3_1_pixel_triple_joint_norm_20260327-033611` — 0.0635
  33.  `exp1c_20260224-172410` — 0.0767
  34.  `exp1b_1_20260310-123920` — 0.0799
  35.  `exp1e_1_pixel_bravo_snr_gamma_rflow_256x160_20260225-031917` — 0.0806
  36.  `exp1c_20260223-002347` — 0.0878
  37.  `exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714` — 0.1436
  38.  `exp1m_1_pixel_bravo_global_norm_20260306-215802` — 0.1573
  39.  `exp1o_1_pixel_bravo_20260413-023304` — 0.1619
  40.  `exp1m_pixel_bravo_global_norm_20260306-230725` — 0.1814
  41.  `exp1n_pixel_bravo_cfg_zero_star_20260309-191558` — 0.1858
  42.  `exp1f_1_pixel_bravo_edm_precond_rflow_256x160_20260225-033959` — 0.2212
  43.  `exp1f_pixel_bravo_edm_precond_rflow_128x160_20260225-022154` — 0.2677
  44.  `exp1c_1_20260223-004121` — 0.4436

#### `exp1b_20260221-172156`
*started 2026-02-21 17:21 • 500 epochs • 17h48m • 1954.2 TFLOPs • peak VRAM 23.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.8143 → 0.0048 (min 0.0036 @ ep 483)
  - `Loss/MSE_val`: 1.7111 → 0.0187 (min 0.0063 @ ep 180)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0875
  - 0.1-0.2: 0.0517
  - 0.2-0.3: 0.0422
  - 0.3-0.4: 0.0229
  - 0.4-0.5: 0.0183
  - 0.5-0.6: 0.0124
  - 0.6-0.7: 0.0100
  - 0.7-0.8: 0.0067
  - 0.8-0.9: 0.0042
  - 0.9-1.0: 0.0049

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0641, best 0.0407 @ ep 343
  - `Generation/KID_mean_train`: last 0.0662, best 0.0392 @ ep 343
  - `Generation/KID_std_val`: last 0.0080, best 0.0042 @ ep 403
  - `Generation/KID_std_train`: last 0.0089, best 0.0035 @ ep 148
  - `Generation/CMMD_val`: last 0.3818, best 0.2914 @ ep 223
  - `Generation/CMMD_train`: last 0.3838, best 0.2879 @ ep 223
  - `Generation/extended_KID_mean_val`: last 0.0827, best 0.0600 @ ep 149
  - `Generation/extended_KID_mean_train`: last 0.0840, best 0.0569 @ ep 149
  - `Generation/extended_CMMD_val`: last 0.3853, best 0.3133 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.3897, best 0.3129 @ ep 124

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4499 (min 0.3566, max 1.8742)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9281 (min 0.4202, max 0.9723)
  - `Validation/MS-SSIM_bravo`: last 0.9342 (min 0.4520, max 0.9786)
  - `Validation/PSNR_bravo`: last 31.3732 (min 11.4018, max 35.6748)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4428
  - `Generation_Diversity/extended_MSSSIM`: 0.1916

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008005
  - `regional_bravo/large`: 0.0043
  - `regional_bravo/medium`: 0.0070
  - `regional_bravo/small`: 0.0156
  - `regional_bravo/tiny`: 0.0120
  - `regional_bravo/tumor_bg_ratio`: 10.8643
  - `regional_bravo/tumor_loss`: 0.0087

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0248, max 8.6283 @ ep 0
  - `training/grad_norm_max`: last 0.1422, max 23.8791 @ ep 40

#### `exp1c_20260223-002347`
*started 2026-02-23 00:23 • 500 epochs • 17h24m • 1954.2 TFLOPs • peak VRAM 23.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 5.9891 → 0.0551 (min 0.0454 @ ep 407)
  - `Loss/MSE_val`: 5.6974 → 0.1644 (min 0.0752 @ ep 173)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1693
  - 0.1-0.2: 0.2080
  - 0.2-0.3: 0.1578
  - 0.3-0.4: 0.1679
  - 0.4-0.5: 0.1772
  - 0.5-0.6: 0.1545
  - 0.6-0.7: 0.1701
  - 0.7-0.8: 0.1514
  - 0.8-0.9: 0.1316
  - 0.9-1.0: 0.1011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2311, best 0.0987 @ ep 245
  - `Generation/KID_mean_train`: last 0.2240, best 0.0901 @ ep 245
  - `Generation/KID_std_val`: last 0.0119, best 0.0068 @ ep 249
  - `Generation/KID_std_train`: last 0.0128, best 0.0063 @ ep 359
  - `Generation/CMMD_val`: last 0.5527, best 0.4224 @ ep 359
  - `Generation/CMMD_train`: last 0.5544, best 0.4207 @ ep 359
  - `Generation/extended_KID_mean_val`: last 0.1659, best 0.0878 @ ep 249
  - `Generation/extended_KID_mean_train`: last 0.1631, best 0.0841 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.4441, best 0.4137 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.4409, best 0.4095 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2850 (min 0.0980, max 1.7887)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9767 (min 0.8067, max 0.9949)
  - `Validation/MS-SSIM_bravo`: last 0.9754 (min 0.8017, max 0.9955)
  - `Validation/PSNR_bravo`: last 35.8563 (min 21.4518, max 41.7632)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.1122
  - `Generation_Diversity/extended_MSSSIM`: 0.4476

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0003143
  - `regional_bravo/large`: 0.0036
  - `regional_bravo/medium`: 0.0053
  - `regional_bravo/small`: 0.0039
  - `regional_bravo/tiny`: 0.0045
  - `regional_bravo/tumor_bg_ratio`: 14.0539
  - `regional_bravo/tumor_loss`: 0.0044

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1689, max 20.7024 @ ep 1
  - `training/grad_norm_max`: last 2.3892, max 58.3141 @ ep 12

#### `exp1c_1_20260223-004121`
*started 2026-02-23 00:41 • 500 epochs • 40h20m • 16018.2 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 5.9958 → 0.0644 (min 0.0616 @ ep 496)
  - `Loss/MSE_val`: 5.6883 → 0.0753 (min 0.0670 @ ep 367)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2014
  - 0.1-0.2: 0.1050
  - 0.2-0.3: 0.0935
  - 0.3-0.4: 0.0649
  - 0.4-0.5: 0.0622
  - 0.5-0.6: 0.0581
  - 0.6-0.7: 0.0692
  - 0.7-0.8: 0.0771
  - 0.8-0.9: 0.0621
  - 0.9-1.0: 0.0995

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.5996, best 0.3627 @ ep 237
  - `Generation/KID_mean_train`: last 0.6046, best 0.3652 @ ep 237
  - `Generation/KID_std_val`: last 0.0144, best 0.0127 @ ep 419
  - `Generation/KID_std_train`: last 0.0157, best 0.0127 @ ep 365
  - `Generation/CMMD_val`: last 0.6666, best 0.6113 @ ep 237
  - `Generation/CMMD_train`: last 0.6649, best 0.6075 @ ep 237
  - `Generation/extended_KID_mean_val`: last 0.4436, best 0.4436 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.4393, best 0.4393 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.6342, best 0.6342 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.6304, best 0.6304 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.1322 (min 0.0967, max 1.6846)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9951 (min 0.7105, max 0.9953)
  - `Validation/MS-SSIM_bravo`: last 0.9931 (min 0.6703, max 0.9957)
  - `Validation/PSNR_bravo`: last 40.5722 (min 20.5862, max 42.2725)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.0133
  - `Generation_Diversity/extended_MSSSIM`: 0.4624

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0001285
  - `regional_bravo/large`: 0.0011
  - `regional_bravo/medium`: 0.0038
  - `regional_bravo/small`: 0.0022
  - `regional_bravo/tiny`: 0.0013
  - `regional_bravo/tumor_bg_ratio`: 16.2784
  - `regional_bravo/tumor_loss`: 0.0021

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1101, max 152.9150 @ ep 46
  - `training/grad_norm_max`: last 1.0007, max 1097.4590 @ ep 46

#### `exp1_pixel_bravo_20260224-163719`
*started 2026-02-24 16:37 • 500 epochs • 10h29m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9694 → 0.0025 (min 0.0015 @ ep 353)
  - `Loss/MSE_val`: 0.9290 → 0.0093 (min 0.0024 @ ep 204)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0466
  - 0.1-0.2: 0.0155
  - 0.2-0.3: 0.0099
  - 0.3-0.4: 0.0062
  - 0.4-0.5: 0.0037
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0210, best 0.0134 @ ep 218
  - `Generation/KID_mean_train`: last 0.0177, best 0.0120 @ ep 218
  - `Generation/KID_std_val`: last 0.0027, best 0.0014 @ ep 489
  - `Generation/KID_std_train`: last 0.0023, best 0.0012 @ ep 376
  - `Generation/CMMD_val`: last 0.2320, best 0.1857 @ ep 210
  - `Generation/CMMD_train`: last 0.2293, best 0.1784 @ ep 210
  - `Generation/extended_KID_mean_val`: last 0.0202, best 0.0173 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0167, best 0.0144 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.1944, best 0.1760 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.1933, best 0.1733 @ ep 324

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5192 (min 0.5108, max 1.8796)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9220 (min 0.2850, max 0.9573)
  - `Validation/MS-SSIM_bravo`: last 0.9355 (min 0.3145, max 0.9644)
  - `Validation/PSNR_bravo`: last 31.8224 (min 10.7008, max 33.5798)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4167
  - `Generation_Diversity/extended_MSSSIM`: 0.1569

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007463
  - `regional_bravo/large`: 0.0136
  - `regional_bravo/medium`: 0.0098
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0097
  - `regional_bravo/tumor_bg_ratio`: 15.3649
  - `regional_bravo/tumor_loss`: 0.0115

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0186, max 3.8623 @ ep 0
  - `training/grad_norm_max`: last 0.2097, max 17.9805 @ ep 50

#### `exp1c_20260224-172410`
*started 2026-02-24 17:24 • 500 epochs • 10h33m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 5.9871 → 0.0509 (min 0.0464 @ ep 459)
  - `Loss/MSE_val`: 5.7092 → 0.1441 (min 0.0746 @ ep 138)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1658
  - 0.1-0.2: 0.1515
  - 0.2-0.3: 0.1624
  - 0.3-0.4: 0.1564
  - 0.4-0.5: 0.1251
  - 0.5-0.6: 0.1181
  - 0.6-0.7: 0.1426
  - 0.7-0.8: 0.1412
  - 0.8-0.9: 0.1766
  - 0.9-1.0: 0.1171

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1120, best 0.0829 @ ep 455
  - `Generation/KID_mean_train`: last 0.1045, best 0.0792 @ ep 455
  - `Generation/KID_std_val`: last 0.0099, best 0.0058 @ ep 291
  - `Generation/KID_std_train`: last 0.0077, best 0.0054 @ ep 310
  - `Generation/CMMD_val`: last 0.5336, best 0.4859 @ ep 469
  - `Generation/CMMD_train`: last 0.5367, best 0.4878 @ ep 469
  - `Generation/extended_KID_mean_val`: last 0.0853, best 0.0767 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0777, best 0.0693 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.4855, best 0.4760 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.4851, best 0.4755 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2331 (min 0.1048, max 1.7770)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9797 (min 0.8031, max 0.9949)
  - `Validation/MS-SSIM_bravo`: last 0.9758 (min 0.8190, max 0.9954)
  - `Validation/PSNR_bravo`: last 36.5640 (min 21.8711, max 41.9550)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.0674
  - `Generation_Diversity/extended_MSSSIM`: 0.6448

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0003178
  - `regional_bravo/large`: 0.0074
  - `regional_bravo/medium`: 0.0048
  - `regional_bravo/small`: 0.0036
  - `regional_bravo/tiny`: 0.0027
  - `regional_bravo/tumor_bg_ratio`: 15.6951
  - `regional_bravo/tumor_loss`: 0.0050

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1750, max 20.3685 @ ep 0
  - `training/grad_norm_max`: last 0.9089, max 50.1574 @ ep 6

#### `exp1e_pixel_bravo_snr_gamma_rflow_128x160_20260225-015538`
*started 2026-02-25 01:55 • 500 epochs • 16h55m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8732 → 0.0006359 (min 0.000622 @ ep 498)
  - `Loss/MSE_val`: 0.9246 → 0.0100 (min 0.0024 @ ep 134)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0726
  - 0.1-0.2: 0.0460
  - 0.2-0.3: 0.0103
  - 0.3-0.4: 0.0083
  - 0.4-0.5: 0.0057
  - 0.5-0.6: 0.0040
  - 0.6-0.7: 0.0033
  - 0.7-0.8: 0.0026
  - 0.8-0.9: 0.0021
  - 0.9-1.0: 0.0015

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0223, best 0.0171 @ ep 316
  - `Generation/KID_mean_train`: last 0.0236, best 0.0158 @ ep 316
  - `Generation/KID_std_val`: last 0.0028, best 0.0016 @ ep 275
  - `Generation/KID_std_train`: last 0.0029, best 0.0016 @ ep 289
  - `Generation/CMMD_val`: last 0.2400, best 0.2007 @ ep 202
  - `Generation/CMMD_train`: last 0.2430, best 0.1965 @ ep 202
  - `Generation/extended_KID_mean_val`: last 0.0298, best 0.0254 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0264, best 0.0229 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2026, best 0.1946 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.2033, best 0.1965 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4766 (min 0.4590, max 1.8829)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9018 (min 0.2858, max 0.9563)
  - `Validation/MS-SSIM_bravo`: last 0.9024 (min 0.2890, max 0.9597)
  - `Validation/PSNR_bravo`: last 29.9060 (min 9.9256, max 33.4577)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4517
  - `Generation_Diversity/extended_MSSSIM`: 0.2025

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0144
  - `regional_bravo/medium`: 0.0184
  - `regional_bravo/small`: 0.0155
  - `regional_bravo/tiny`: 0.0099
  - `regional_bravo/tumor_bg_ratio`: 13.6763
  - `regional_bravo/tumor_loss`: 0.0153

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0095, max 3.5080 @ ep 0
  - `training/grad_norm_max`: last 0.0292, max 3.9648 @ ep 0

#### `exp1f_pixel_bravo_edm_precond_rflow_128x160_20260225-022154`
*started 2026-02-25 02:21 • 500 epochs • 16h46m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0315 → 0.0132 (min 0.0055 @ ep 395)
  - `Loss/MSE_val`: 0.0406 → 0.0217 (min 0.0030 @ ep 376)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2384
  - 0.1-0.2: 0.0658
  - 0.2-0.3: 0.0250
  - 0.3-0.4: 0.0064
  - 0.4-0.5: 0.0058
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0023
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2646, best 0.2252 @ ep 332
  - `Generation/KID_mean_train`: last 0.2624, best 0.2242 @ ep 332
  - `Generation/KID_std_val`: last 0.0095, best 0.0080 @ ep 451
  - `Generation/KID_std_train`: last 0.0087, best 0.0075 @ ep 426
  - `Generation/CMMD_val`: last 0.6323, best 0.6000 @ ep 13
  - `Generation/CMMD_train`: last 0.6366, best 0.6036 @ ep 13
  - `Generation/extended_KID_mean_val`: last 0.2779, best 0.2677 @ ep 99
  - `Generation/extended_KID_mean_train`: last 0.2759, best 0.2632 @ ep 99
  - `Generation/extended_CMMD_val`: last 0.6729, best 0.6509 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.6740, best 0.6521 @ ep 124

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.3054 (min 0.9348, max 1.5455)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9236 (min 0.5611, max 0.9251)
  - `Validation/MS-SSIM_bravo`: last 0.9204 (min 0.5813, max 0.9238)
  - `Validation/PSNR_bravo`: last 30.5260 (min 24.5616, max 30.8045)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1963
  - `Generation_Diversity/extended_MSSSIM`: 0.3134

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0010
  - `regional_bravo/large`: 0.0069
  - `regional_bravo/medium`: 0.0156
  - `regional_bravo/small`: 0.0099
  - `regional_bravo/tiny`: 0.0075
  - `regional_bravo/tumor_bg_ratio`: 10.2508
  - `regional_bravo/tumor_loss`: 0.0107

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0355, max 0.5545 @ ep 0
  - `training/grad_norm_max`: last 1.8693, max 24.7123 @ ep 5

#### `exp1e_1_pixel_bravo_snr_gamma_rflow_256x160_20260225-031917`
*started 2026-02-25 03:19 • 500 epochs • 49h06m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8751 → 0.0018 (min 0.0016 @ ep 411)
  - `Loss/MSE_val`: 0.9232 → 0.0038 (min 0.0021 @ ep 354)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0232
  - 0.1-0.2: 0.0086
  - 0.2-0.3: 0.0063
  - 0.3-0.4: 0.0027
  - 0.4-0.5: 0.0035
  - 0.5-0.6: 0.0025
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0576, best 0.0369 @ ep 342
  - `Generation/KID_mean_train`: last 0.0493, best 0.0302 @ ep 342
  - `Generation/KID_std_val`: last 0.0053, best 0.0031 @ ep 407
  - `Generation/KID_std_train`: last 0.0038, best 0.0024 @ ep 420
  - `Generation/CMMD_val`: last 0.2117, best 0.1888 @ ep 408
  - `Generation/CMMD_train`: last 0.2041, best 0.1827 @ ep 412
  - `Generation/extended_KID_mean_val`: last 0.0920, best 0.0806 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0818, best 0.0724 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.1810, best 0.1810 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.1740, best 0.1740 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5527 (min 0.4657, max 1.7783)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9505 (min 0.2365, max 0.9595)
  - `Validation/MS-SSIM_bravo`: last 0.9526 (min 0.2818, max 0.9654)
  - `Validation/PSNR_bravo`: last 32.7761 (min 10.8306, max 34.0574)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4511
  - `Generation_Diversity/extended_MSSSIM`: 0.1429

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006025
  - `regional_bravo/large`: 0.0066
  - `regional_bravo/medium`: 0.0104
  - `regional_bravo/small`: 0.0108
  - `regional_bravo/tiny`: 0.0079
  - `regional_bravo/tumor_bg_ratio`: 14.4443
  - `regional_bravo/tumor_loss`: 0.0087

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0126, max 6.0255 @ ep 23
  - `training/grad_norm_max`: last 0.0648, max 35.7338 @ ep 23

#### `exp1f_1_pixel_bravo_edm_precond_rflow_256x160_20260225-033959`
*started 2026-02-25 03:39 • 500 epochs • 49h20m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0509 → 0.0038 (min 0.0027 @ ep 401)
  - `Loss/MSE_val`: 0.0347 → 0.0036 (min 0.0027 @ ep 407)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0309
  - 0.1-0.2: 0.0100
  - 0.2-0.3: 0.0039
  - 0.3-0.4: 0.0045
  - 0.4-0.5: 0.0035
  - 0.5-0.6: 0.0028
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0013
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0016

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1283, best 0.1094 @ ep 464
  - `Generation/KID_mean_train`: last 0.1165, best 0.1005 @ ep 464
  - `Generation/KID_std_val`: last 0.0089, best 0.0048 @ ep 453
  - `Generation/KID_std_train`: last 0.0071, best 0.0046 @ ep 444
  - `Generation/CMMD_val`: last 0.4081, best 0.3430 @ ep 496
  - `Generation/CMMD_train`: last 0.4002, best 0.3334 @ ep 496
  - `Generation/extended_KID_mean_val`: last 0.2323, best 0.2212 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.2248, best 0.2158 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.3928, best 0.3866 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.3839, best 0.3761 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5473 (min 0.4605, max 1.5318)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9460 (min 0.6046, max 0.9470)
  - `Validation/MS-SSIM_bravo`: last 0.9439 (min 0.6097, max 0.9569)
  - `Validation/PSNR_bravo`: last 32.1788 (min 23.8355, max 33.1795)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3987
  - `Generation_Diversity/extended_MSSSIM`: 0.1685

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006602
  - `regional_bravo/large`: 0.0118
  - `regional_bravo/medium`: 0.0076
  - `regional_bravo/small`: 0.0072
  - `regional_bravo/tiny`: 0.0073
  - `regional_bravo/tumor_bg_ratio`: 13.4476
  - `regional_bravo/tumor_loss`: 0.0089

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0414, max 0.7756 @ ep 0
  - `training/grad_norm_max`: last 2.5815, max 26.5759 @ ep 6

#### `exp1g_1_pixel_bravo_pseudo_huber_20260226-034319`
*started 2026-02-26 03:43 • 500 epochs • 42h16m • 16018.2 TFLOPs • peak VRAM 46.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 3184.8047 → 138.4513 (min 122.5704 @ ep 445)
  - `Loss/MSE_val`: 0.9265 → 0.0046 (min 0.0022 @ ep 249)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0264
  - 0.1-0.2: 0.0090
  - 0.2-0.3: 0.0068
  - 0.3-0.4: 0.0053
  - 0.4-0.5: 0.0063
  - 0.5-0.6: 0.0034
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0614, best 0.0305 @ ep 308
  - `Generation/KID_mean_train`: last 0.0550, best 0.0239 @ ep 308
  - `Generation/KID_std_val`: last 0.0088, best 0.0026 @ ep 269
  - `Generation/KID_std_train`: last 0.0087, best 0.0025 @ ep 334
  - `Generation/CMMD_val`: last 0.3708, best 0.1833 @ ep 308
  - `Generation/CMMD_train`: last 0.3663, best 0.1794 @ ep 308
  - `Generation/extended_KID_mean_val`: last 0.0707, best 0.0526 @ ep 324
  - `Generation/extended_KID_mean_train`: last 0.0616, best 0.0471 @ ep 324
  - `Generation/extended_CMMD_val`: last 0.3472, best 0.2106 @ ep 299
  - `Generation/extended_CMMD_train`: last 0.3412, best 0.2031 @ ep 299

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5334 (min 0.4474, max 1.7785)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9236 (min 0.2355, max 0.9597)
  - `Validation/MS-SSIM_bravo`: last 0.9269 (min 0.2763, max 0.9662)
  - `Validation/PSNR_bravo`: last 30.9371 (min 10.7371, max 33.9423)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6523
  - `Generation_Diversity/extended_MSSSIM`: 0.2971

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008763
  - `regional_bravo/large`: 0.0095
  - `regional_bravo/medium`: 0.0120
  - `regional_bravo/small`: 0.0132
  - `regional_bravo/tiny`: 0.0088
  - `regional_bravo/tumor_bg_ratio`: 12.3056
  - `regional_bravo/tumor_loss`: 0.0108

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 554.7637, max 6641.2603 @ ep 2
  - `training/grad_norm_max`: last 3076.1416, max 5.123e+04 @ ep 18

#### `exp1g_pixel_bravo_pseudo_huber_20260226-093915`
*started 2026-02-26 09:39 • 500 epochs • 10h33m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1592.6180 → 66.0194 (min 53.9473 @ ep 465)
  - `Loss/MSE_val`: 0.9261 → 0.0062 (min 0.0024 @ ep 204)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0218
  - 0.1-0.2: 0.0179
  - 0.2-0.3: 0.0143
  - 0.3-0.4: 0.0053
  - 0.4-0.5: 0.0049
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0029
  - 0.7-0.8: 0.0024
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0232, best 0.0123 @ ep 491
  - `Generation/KID_mean_train`: last 0.0214, best 0.0115 @ ep 341
  - `Generation/KID_std_val`: last 0.0027, best 0.0014 @ ep 388
  - `Generation/KID_std_train`: last 0.0035, best 0.0012 @ ep 489
  - `Generation/CMMD_val`: last 0.2584, best 0.1894 @ ep 209
  - `Generation/CMMD_train`: last 0.2595, best 0.1874 @ ep 173
  - `Generation/extended_KID_mean_val`: last 0.0134, best 0.0134 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0113, best 0.0104 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2341, best 0.1713 @ ep 249
  - `Generation/extended_CMMD_train`: last 0.2372, best 0.1716 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5410 (min 0.4693, max 1.8759)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9184 (min 0.2860, max 0.9563)
  - `Validation/MS-SSIM_bravo`: last 0.9157 (min 0.3471, max 0.9604)
  - `Validation/PSNR_bravo`: last 30.6777 (min 11.3252, max 33.6615)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4376
  - `Generation_Diversity/extended_MSSSIM`: 0.1711

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009408
  - `regional_bravo/large`: 0.0100
  - `regional_bravo/medium`: 0.0105
  - `regional_bravo/small`: 0.0215
  - `regional_bravo/tiny`: 0.0107
  - `regional_bravo/tumor_bg_ratio`: 13.2615
  - `regional_bravo/tumor_loss`: 0.0125

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 954.6154, max 3467.9053 @ ep 3
  - `training/grad_norm_max`: last 3592.3113, max 2.098e+04 @ ep 4

#### `exp1h_1_pixel_bravo_lpips_huber_20260226-202457`
*started 2026-02-26 20:24 • 500 epochs • 35h22m • 16018.2 TFLOPs • peak VRAM 46.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1615.1643 → 83.1286 (min 65.3265 @ ep 473)
  - `Loss/MSE_val`: 0.9252 → 0.0035 (min 0.0023 @ ep 316)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0270
  - 0.1-0.2: 0.0084
  - 0.2-0.3: 0.0070
  - 0.3-0.4: 0.0051
  - 0.4-0.5: 0.0031
  - 0.5-0.6: 0.0034
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0023
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0588, best 0.0470 @ ep 141
  - `Generation/KID_mean_train`: last 0.0530, best 0.0433 @ ep 141
  - `Generation/KID_std_val`: last 0.0076, best 0.0032 @ ep 151
  - `Generation/KID_std_train`: last 0.0072, best 0.0032 @ ep 148
  - `Generation/CMMD_val`: last 0.3724, best 0.2233 @ ep 156
  - `Generation/CMMD_train`: last 0.3684, best 0.2173 @ ep 156
  - `Generation/extended_KID_mean_val`: last 0.0616, best 0.0559 @ ep 349
  - `Generation/extended_KID_mean_train`: last 0.0548, best 0.0487 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.3651, best 0.2385 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.3613, best 0.2276 @ ep 124

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5388 (min 0.4136, max 1.7905)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9329 (min 0.2387, max 0.9603)
  - `Validation/MS-SSIM_bravo`: last 0.9288 (min 0.2480, max 0.9678)
  - `Validation/PSNR_bravo`: last 31.0942 (min 10.1569, max 34.1306)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2701
  - `Generation_Diversity/extended_MSSSIM`: 0.1313

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008532
  - `regional_bravo/large`: 0.0158
  - `regional_bravo/medium`: 0.0118
  - `regional_bravo/small`: 0.0156
  - `regional_bravo/tiny`: 0.0097
  - `regional_bravo/tumor_bg_ratio`: 15.7466
  - `regional_bravo/tumor_loss`: 0.0134

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 372.0261, max 3784.2297 @ ep 3
  - `training/grad_norm_max`: last 1419.6277, max 3.869e+04 @ ep 3

#### `exp1h_pixel_bravo_lpips_huber_20260226-221951`
*started 2026-02-26 22:19 • 500 epochs • 10h51m • 1954.2 TFLOPs • peak VRAM 21.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 788.4956 → 38.8432 (min 25.2470 @ ep 435)
  - `Loss/MSE_val`: 0.9262 → 0.0091 (min 0.0023 @ ep 157)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0554
  - 0.1-0.2: 0.0190
  - 0.2-0.3: 0.0113
  - 0.3-0.4: 0.0101
  - 0.4-0.5: 0.0043
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0026
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0211, best 0.0138 @ ep 306
  - `Generation/KID_mean_train`: last 0.0204, best 0.0121 @ ep 227
  - `Generation/KID_std_val`: last 0.0022, best 0.0014 @ ep 401
  - `Generation/KID_std_train`: last 0.0023, best 0.0014 @ ep 341
  - `Generation/CMMD_val`: last 0.2451, best 0.1733 @ ep 143
  - `Generation/CMMD_train`: last 0.2471, best 0.1696 @ ep 143
  - `Generation/extended_KID_mean_val`: last 0.0213, best 0.0151 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0190, best 0.0116 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.2054, best 0.1877 @ ep 249
  - `Generation/extended_CMMD_train`: last 0.2067, best 0.1875 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4848 (min 0.4439, max 1.8905)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9120 (min 0.2852, max 0.9563)
  - `Validation/MS-SSIM_bravo`: last 0.9200 (min 0.3288, max 0.9638)
  - `Validation/PSNR_bravo`: last 30.6101 (min 11.1561, max 33.6779)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.8381
  - `Generation_Diversity/extended_MSSSIM`: 0.2019

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000947
  - `regional_bravo/large`: 0.0118
  - `regional_bravo/medium`: 0.0161
  - `regional_bravo/small`: 0.0203
  - `regional_bravo/tiny`: 0.0122
  - `regional_bravo/tumor_bg_ratio`: 15.8449
  - `regional_bravo/tumor_loss`: 0.0150

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 330.1992, max 1705.7703 @ ep 2
  - `training/grad_norm_max`: last 1579.5022, max 1.529e+04 @ ep 4

#### `exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714`
*started 2026-03-01 01:57 • 500 epochs • 43h48m • 16018.2 TFLOPs • peak VRAM 43.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9008 → 0.0024 (min 0.0024 @ ep 398)
  - `Loss/MSE_val`: 0.9302 → 0.0028 (min 0.0021 @ ep 471)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0263
  - 0.1-0.2: 0.0081
  - 0.2-0.3: 0.0059
  - 0.3-0.4: 0.0037
  - 0.4-0.5: 0.0032
  - 0.5-0.6: 0.0019
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1303, best 0.1227 @ ep 497
  - `Generation/KID_mean_train`: last 0.1213, best 0.1150 @ ep 474
  - `Generation/KID_std_val`: last 0.0053, best 0.0040 @ ep 487
  - `Generation/KID_std_train`: last 0.0035, best 0.0034 @ ep 408
  - `Generation/CMMD_val`: last 0.2508, best 0.2178 @ ep 445
  - `Generation/CMMD_train`: last 0.2437, best 0.2094 @ ep 445
  - `Generation/extended_KID_mean_val`: last 0.1441, best 0.1436 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.1389, best 0.1365 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2502, best 0.2502 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.2409, best 0.2409 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.8401 (min 0.6253, max 1.7928)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9587 (min 0.2349, max 0.9591)
  - `Validation/MS-SSIM_bravo`: last 0.9517 (min 0.2727, max 0.9670)
  - `Validation/PSNR_bravo`: last 32.6977 (min 10.5663, max 34.0436)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.8791
  - `Generation_Diversity/extended_MSSSIM`: 0.3105

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006184
  - `regional_bravo/large`: 0.0090
  - `regional_bravo/medium`: 0.0107
  - `regional_bravo/small`: 0.0111
  - `regional_bravo/tiny`: 0.0093
  - `regional_bravo/tumor_bg_ratio`: 16.0594
  - `regional_bravo/tumor_loss`: 0.0099

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0299, max 4.0598 @ ep 75
  - `training/grad_norm_max`: last 0.1402, max 29.6520 @ ep 74

#### `exp1j_1_pixel_bravo_grad_accum_20260301-015744`
*started 2026-03-01 01:57 • 500 epochs • 22h47m • 16018.2 TFLOPs • peak VRAM 47.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9984 → 0.0032 (min 0.0022 @ ep 487)
  - `Loss/MSE_val`: 0.9887 → 0.0037 (min 0.0023 @ ep 384)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0135
  - 0.1-0.2: 0.0124
  - 0.2-0.3: 0.0062
  - 0.3-0.4: 0.0031
  - 0.4-0.5: 0.0036
  - 0.5-0.6: 0.0019
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0429, best 0.0327 @ ep 361
  - `Generation/KID_mean_train`: last 0.0386, best 0.0276 @ ep 361
  - `Generation/KID_std_val`: last 0.0032, best 0.0025 @ ep 399
  - `Generation/KID_std_train`: last 0.0027, best 0.0021 @ ep 444
  - `Generation/CMMD_val`: last 0.2423, best 0.1937 @ ep 335
  - `Generation/CMMD_train`: last 0.2413, best 0.1833 @ ep 335
  - `Generation/extended_KID_mean_val`: last 0.0534, best 0.0417 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0432, best 0.0352 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2046, best 0.2003 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1990, best 0.1961 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7857 (min 0.5188, max 1.7932)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9487 (min 0.2307, max 0.9577)
  - `Validation/MS-SSIM_bravo`: last 0.9390 (min 0.2216, max 0.9646)
  - `Validation/PSNR_bravo`: last 32.0150 (min 9.3219, max 34.2255)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2907
  - `Generation_Diversity/extended_MSSSIM`: 0.1383

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007134
  - `regional_bravo/large`: 0.0101
  - `regional_bravo/medium`: 0.0094
  - `regional_bravo/small`: 0.0107
  - `regional_bravo/tiny`: 0.0070
  - `regional_bravo/tumor_bg_ratio`: 13.1798
  - `regional_bravo/tumor_loss`: 0.0094

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0235, max 3.9169 @ ep 0
  - `training/grad_norm_max`: last 0.1252, max 3.9478 @ ep 0

#### `exp1_1_1000_pixel_bravo_20260301-020650`
*started 2026-03-01 02:06 • 1000 epochs • 125h08m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9696 → 0.0013 (min 0.0011 @ ep 977)
  - `Loss/MSE_val`: 0.9286 → 0.0063 (min 0.0023 @ ep 220)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0584
  - 0.1-0.2: 0.0335
  - 0.2-0.3: 0.0160
  - 0.3-0.4: 0.0083
  - 0.4-0.5: 0.0035
  - 0.5-0.6: 0.0044
  - 0.6-0.7: 0.0031
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0024

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0283, best 0.0190 @ ep 878
  - `Generation/KID_mean_train`: last 0.0284, best 0.0157 @ ep 515
  - `Generation/KID_std_val`: last 0.0032, best 0.0020 @ ep 615
  - `Generation/KID_std_train`: last 0.0039, best 0.0020 @ ep 808
  - `Generation/CMMD_val`: last 0.2196, best 0.1646 @ ep 494
  - `Generation/CMMD_train`: last 0.2286, best 0.1588 @ ep 494
  - `Generation/extended_KID_mean_val`: last 0.0181, best 0.0181 @ ep 999
  - `Generation/extended_KID_mean_train`: last 0.0163, best 0.0163 @ ep 999
  - `Generation/extended_CMMD_val`: last 0.1816, best 0.1681 @ ep 349
  - `Generation/extended_CMMD_train`: last 0.1867, best 0.1728 @ ep 349

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4285 (min 0.3810, max 1.7903)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9052 (min 0.2352, max 0.9608)
  - `Validation/MS-SSIM_bravo`: last 0.9058 (min 0.2402, max 0.9647)
  - `Validation/PSNR_bravo`: last 29.5031 (min 9.8310, max 33.9822)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2915
  - `Generation_Diversity/extended_MSSSIM`: 0.1312

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0012
  - `regional_bravo/large`: 0.0150
  - `regional_bravo/medium`: 0.0161
  - `regional_bravo/small`: 0.0160
  - `regional_bravo/tiny`: 0.0092
  - `regional_bravo/tumor_bg_ratio`: 12.3546
  - `regional_bravo/tumor_loss`: 0.0143

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0062, max 3.8578 @ ep 0
  - `training/grad_norm_max`: last 0.0286, max 7.1709 @ ep 3

#### `exp1_1_pixel_bravo_20260301-180232`
*started 2026-03-01 18:02 • 500 epochs • 52h10m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9682 → 0.0026 (min 0.0018 @ ep 405)
  - `Loss/MSE_val`: 0.9254 → 0.0038 (min 0.0021 @ ep 257)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0146
  - 0.1-0.2: 0.0081
  - 0.2-0.3: 0.0047
  - 0.3-0.4: 0.0039
  - 0.4-0.5: 0.0025
  - 0.5-0.6: 0.0023
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0023
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0466, best 0.0274 @ ep 493
  - `Generation/KID_mean_train`: last 0.0443, best 0.0243 @ ep 490
  - `Generation/KID_std_val`: last 0.0043, best 0.0024 @ ep 480
  - `Generation/KID_std_train`: last 0.0039, best 0.0018 @ ep 307
  - `Generation/CMMD_val`: last 0.2396, best 0.1800 @ ep 347
  - `Generation/CMMD_train`: last 0.2371, best 0.1762 @ ep 347
  - `Generation/extended_KID_mean_val`: last 0.0316, best 0.0280 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0260, best 0.0217 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.2088, best 0.2074 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2089, best 0.2088 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5145 (min 0.4655, max 1.7913)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9454 (min 0.2388, max 0.9601)
  - `Validation/MS-SSIM_bravo`: last 0.9501 (min 0.2374, max 0.9651)
  - `Validation/PSNR_bravo`: last 32.6501 (min 9.7317, max 34.0569)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2544
  - `Generation_Diversity/extended_MSSSIM`: 0.1354

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006366
  - `regional_bravo/large`: 0.0098
  - `regional_bravo/medium`: 0.0074
  - `regional_bravo/small`: 0.0075
  - `regional_bravo/tiny`: 0.0068
  - `regional_bravo/tumor_bg_ratio`: 12.7148
  - `regional_bravo/tumor_loss`: 0.0081

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0138, max 3.8822 @ ep 0
  - `training/grad_norm_max`: last 0.0841, max 13.0312 @ ep 93

#### `exp1k_pixel_bravo_offset_noise_20260306-113529`
*started 2026-03-06 11:35 • 500 epochs • 11h23m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9791 → 0.0027 (min 0.0018 @ ep 475)
  - `Loss/MSE_val`: 0.9300 → 0.0063 (min 0.0022 @ ep 238)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0192
  - 0.1-0.2: 0.0130
  - 0.2-0.3: 0.0100
  - 0.3-0.4: 0.0067
  - 0.4-0.5: 0.0047
  - 0.5-0.6: 0.0031
  - 0.6-0.7: 0.0028
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0275, best 0.0161 @ ep 337
  - `Generation/KID_mean_train`: last 0.0232, best 0.0135 @ ep 236
  - `Generation/KID_std_val`: last 0.0041, best 0.0015 @ ep 320
  - `Generation/KID_std_train`: last 0.0030, best 0.0014 @ ep 415
  - `Generation/CMMD_val`: last 0.2371, best 0.1895 @ ep 292
  - `Generation/CMMD_train`: last 0.2325, best 0.1869 @ ep 190
  - `Generation/extended_KID_mean_val`: last 0.0208, best 0.0189 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0164, best 0.0164 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2181, best 0.1993 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.2176, best 0.2002 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5618 (min 0.4998, max 1.8817)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9262 (min 0.2829, max 0.9558)
  - `Validation/MS-SSIM_bravo`: last 0.9358 (min 0.3343, max 0.9631)
  - `Validation/PSNR_bravo`: last 31.6067 (min 10.9430, max 33.7572)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4563
  - `Generation_Diversity/extended_MSSSIM`: 0.1992

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007779
  - `regional_bravo/large`: 0.0071
  - `regional_bravo/medium`: 0.0103
  - `regional_bravo/small`: 0.0149
  - `regional_bravo/tiny`: 0.0102
  - `regional_bravo/tumor_bg_ratio`: 13.1660
  - `regional_bravo/tumor_loss`: 0.0102

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0234, max 4.4965 @ ep 27
  - `training/grad_norm_max`: last 0.2909, max 43.6660 @ ep 27

#### `exp1l_pixel_bravo_adjusted_offset_20260306-114524`
*started 2026-03-06 11:45 • 500 epochs • 10h51m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9810 → 0.0017 (min 0.0015 @ ep 443)
  - `Loss/MSE_val`: 0.9306 → 0.0058 (min 0.0025 @ ep 193)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0208
  - 0.1-0.2: 0.0228
  - 0.2-0.3: 0.0087
  - 0.3-0.4: 0.0073
  - 0.4-0.5: 0.0035
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0028
  - 0.7-0.8: 0.0024
  - 0.8-0.9: 0.0014
  - 0.9-1.0: 0.0021

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0216, best 0.0183 @ ep 376
  - `Generation/KID_mean_train`: last 0.0214, best 0.0152 @ ep 299
  - `Generation/KID_std_val`: last 0.0018, best 0.0015 @ ep 427
  - `Generation/KID_std_train`: last 0.0021, best 0.0013 @ ep 316
  - `Generation/CMMD_val`: last 0.2575, best 0.1838 @ ep 241
  - `Generation/CMMD_train`: last 0.2602, best 0.1843 @ ep 155
  - `Generation/extended_KID_mean_val`: last 0.0209, best 0.0197 @ ep 199
  - `Generation/extended_KID_mean_train`: last 0.0177, best 0.0165 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2133, best 0.1747 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.2148, best 0.1754 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6820 (min 0.5447, max 1.8886)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9198 (min 0.2828, max 0.9558)
  - `Validation/MS-SSIM_bravo`: last 0.9207 (min 0.3173, max 0.9614)
  - `Validation/PSNR_bravo`: last 30.7388 (min 10.5713, max 33.5480)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4939
  - `Generation_Diversity/extended_MSSSIM`: 0.1701

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009062
  - `regional_bravo/large`: 0.0100
  - `regional_bravo/medium`: 0.0130
  - `regional_bravo/small`: 0.0125
  - `regional_bravo/tiny`: 0.0102
  - `regional_bravo/tumor_bg_ratio`: 12.7629
  - `regional_bravo/tumor_loss`: 0.0116

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0183, max 3.9466 @ ep 0
  - `training/grad_norm_max`: last 0.0571, max 6.0944 @ ep 31

#### `exp1m_1_pixel_bravo_global_norm_20260306-215802`
*started 2026-03-06 21:58 • 500 epochs • 24h12m • 16018.2 TFLOPs • peak VRAM 46.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.7931 → 0.0121 (min 0.0102 @ ep 488)
  - `Loss/MSE_val`: 1.6977 → 0.0227 (min 0.0108 @ ep 328)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1852
  - 0.1-0.2: 0.0804
  - 0.2-0.3: 0.0154
  - 0.3-0.4: 0.0111
  - 0.4-0.5: 0.0097
  - 0.5-0.6: 0.0137
  - 0.6-0.7: 0.0047
  - 0.7-0.8: 0.0089
  - 0.8-0.9: 0.0040
  - 0.9-1.0: 0.0038

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1883, best 0.1648 @ ep 495
  - `Generation/KID_mean_train`: last 0.1741, best 0.1534 @ ep 495
  - `Generation/KID_std_val`: last 0.0110, best 0.0074 @ ep 459
  - `Generation/KID_std_train`: last 0.0093, best 0.0068 @ ep 403
  - `Generation/CMMD_val`: last 0.4420, best 0.3974 @ ep 446
  - `Generation/CMMD_train`: last 0.4350, best 0.3924 @ ep 446
  - `Generation/extended_KID_mean_val`: last 0.1666, best 0.1573 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.1531, best 0.1433 @ ep 199
  - `Generation/extended_CMMD_val`: last 0.3748, best 0.3568 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.3689, best 0.3505 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6246 (min 0.3529, max 1.7508)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9769 (min 0.3678, max 0.9780)
  - `Validation/MS-SSIM_bravo`: last 0.9634 (min 0.3408, max 0.9828)
  - `Validation/PSNR_bravo`: last 32.9497 (min 9.9248, max 34.5428)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.7892
  - `Generation_Diversity/extended_MSSSIM`: 0.2892

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006976
  - `regional_bravo/large`: 0.0138
  - `regional_bravo/medium`: 0.0114
  - `regional_bravo/small`: 0.0166
  - `regional_bravo/tiny`: 0.0157
  - `regional_bravo/tumor_bg_ratio`: 20.1477
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0321, max 13.6773 @ ep 16
  - `training/grad_norm_max`: last 0.2820, max 82.2942 @ ep 103

#### `exp1m_pixel_bravo_global_norm_20260306-230725`
*started 2026-03-06 23:07 • 500 epochs • 10h34m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.7910 → 1.0062 (min 0.0195 @ ep 44)
  - `Loss/MSE_val`: 1.6895 → 1.0072 (min 0.0208 @ ep 40)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0094
  - 0.1-0.2: 1.0025
  - 0.2-0.3: 1.0041
  - 0.3-0.4: 1.0065
  - 0.4-0.5: 1.0052
  - 0.5-0.6: 1.0128
  - 0.6-0.7: 1.0113
  - 0.7-0.8: 1.0050
  - 0.8-0.9: 1.0037
  - 0.9-1.0: 1.0077

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.7618, best 0.1338 @ ep 39
  - `Generation/KID_mean_train`: last 0.7619, best 0.1321 @ ep 39
  - `Generation/KID_std_val`: last 0.0096, best 0.0069 @ ep 39
  - `Generation/KID_std_train`: last 0.0107, best 0.0062 @ ep 35
  - `Generation/CMMD_val`: last 0.8076, best 0.4623 @ ep 39
  - `Generation/CMMD_train`: last 0.8074, best 0.4640 @ ep 39
  - `Generation/extended_KID_mean_val`: last 0.7631, best 0.1814 @ ep 24
  - `Generation/extended_KID_mean_train`: last 0.7595, best 0.1798 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.8140, best 0.5371 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.8207, best 0.5395 @ ep 24

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.8302 (min 0.9815, max 1.8608)
  - `Validation/MS-SSIM-3D_bravo`: last 0.6006 (min 0.4791, max 0.9591)
  - `Validation/MS-SSIM_bravo`: last 0.6185 (min 0.5032, max 0.9623)
  - `Validation/PSNR_bravo`: last 15.6037 (min 12.0488, max 31.6323)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1046
  - `Generation_Diversity/extended_MSSSIM`: 0.9249

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0743
  - `regional_bravo/large`: 0.0435
  - `regional_bravo/medium`: 0.0458
  - `regional_bravo/small`: 0.0751
  - `regional_bravo/tiny`: 0.0964
  - `regional_bravo/tumor_bg_ratio`: 0.7946
  - `regional_bravo/tumor_loss`: 0.0591

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0427, max 17.2770 @ ep 45
  - `training/grad_norm_max`: last 0.1673, max 99.0105 @ ep 45

#### `exp1n_pixel_bravo_cfg_zero_star_20260309-191558`
*started 2026-03-09 19:15 • 500 epochs • 87h05m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9688 → 0.0019 (min 0.0015 @ ep 485)
  - `Loss/MSE_val`: 0.9255 → 0.0060 (min 0.0023 @ ep 179)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0167
  - 0.1-0.2: 0.0131
  - 0.2-0.3: 0.0090
  - 0.3-0.4: 0.0050
  - 0.4-0.5: 0.0044
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0029
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0019
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.3776, best 0.3331 @ ep 397
  - `Generation/KID_mean_train`: last 0.3716, best 0.3294 @ ep 397
  - `Generation/KID_std_val`: last 0.0139, best 0.0091 @ ep 493
  - `Generation/KID_std_train`: last 0.0131, best 0.0082 @ ep 366
  - `Generation/CMMD_val`: last 0.5417, best 0.5069 @ ep 469
  - `Generation/CMMD_train`: last 0.5222, best 0.4857 @ ep 469
  - `Generation/extended_KID_mean_val`: last 0.1904, best 0.1858 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.1887, best 0.1859 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.3607, best 0.3471 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.3506, best 0.3357 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5109 (min 0.4548, max 1.7890)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9239 (min 0.2359, max 0.9605)
  - `Validation/MS-SSIM_bravo`: last 0.9267 (min 0.2476, max 0.9646)
  - `Validation/PSNR_bravo`: last 31.2687 (min 10.2334, max 33.6838)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1763
  - `Generation_Diversity/extended_MSSSIM`: 0.2388

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009022
  - `regional_bravo/large`: 0.0099
  - `regional_bravo/medium`: 0.0128
  - `regional_bravo/small`: 0.0120
  - `regional_bravo/tiny`: 0.0089
  - `regional_bravo/tumor_bg_ratio`: 12.0848
  - `regional_bravo/tumor_loss`: 0.0109

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0201, max 3.8759 @ ep 0
  - `training/grad_norm_max`: last 0.1070, max 7.3444 @ ep 4

#### `exp1l_1_pixel_bravo_adjusted_offset_20260309-202039`
*started 2026-03-09 20:20 • 500 epochs • 83h07m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9804 → 0.0036 (min 0.0022 @ ep 472)
  - `Loss/MSE_val`: 0.9291 → 0.0076 (min 0.0021 @ ep 305)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0400
  - 0.1-0.2: 0.0068
  - 0.2-0.3: 0.0068
  - 0.3-0.4: 0.0036
  - 0.4-0.5: 0.0026
  - 0.5-0.6: 0.0031
  - 0.6-0.7: 0.0017
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0016

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0438, best 0.0242 @ ep 454
  - `Generation/KID_mean_train`: last 0.0399, best 0.0221 @ ep 449
  - `Generation/KID_std_val`: last 0.0046, best 0.0026 @ ep 465
  - `Generation/KID_std_train`: last 0.0043, best 0.0022 @ ep 424
  - `Generation/CMMD_val`: last 0.2181, best 0.1514 @ ep 429
  - `Generation/CMMD_train`: last 0.2074, best 0.1424 @ ep 429
  - `Generation/extended_KID_mean_val`: last 0.0337, best 0.0259 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0296, best 0.0231 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1758, best 0.1634 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.1687, best 0.1548 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6013 (min 0.5298, max 1.7879)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9481 (min 0.2348, max 0.9586)
  - `Validation/MS-SSIM_bravo`: last 0.9535 (min 0.2649, max 0.9633)
  - `Validation/PSNR_bravo`: last 33.2292 (min 10.5412, max 33.8180)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2767
  - `Generation_Diversity/extended_MSSSIM`: 0.1568

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005878
  - `regional_bravo/large`: 0.0097
  - `regional_bravo/medium`: 0.0099
  - `regional_bravo/small`: 0.0100
  - `regional_bravo/tiny`: 0.0085
  - `regional_bravo/tumor_bg_ratio`: 16.3428
  - `regional_bravo/tumor_loss`: 0.0096

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0224, max 4.7347 @ ep 114
  - `training/grad_norm_max`: last 0.1292, max 40.4004 @ ep 114

#### `exp1b_1_20260310-123920`
*started 2026-03-10 12:39 • 500 epochs • 78h23m • 16018.2 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.8633 → 0.0048 (min 0.0041 @ ep 458)
  - `Loss/MSE_val`: 1.8080 → 0.0163 (min 0.0067 @ ep 175)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0508
  - 0.1-0.2: 0.0339
  - 0.2-0.3: 0.0304
  - 0.3-0.4: 0.0213
  - 0.4-0.5: 0.0182
  - 0.5-0.6: 0.0169
  - 0.6-0.7: 0.0083
  - 0.7-0.8: 0.0103
  - 0.8-0.9: 0.0070
  - 0.9-1.0: 0.0042

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1876, best 0.1040 @ ep 467
  - `Generation/KID_mean_train`: last 0.1922, best 0.1066 @ ep 467
  - `Generation/KID_std_val`: last 0.0075, best 0.0040 @ ep 441
  - `Generation/KID_std_train`: last 0.0073, best 0.0037 @ ep 339
  - `Generation/CMMD_val`: last 0.3029, best 0.1399 @ ep 359
  - `Generation/CMMD_train`: last 0.2902, best 0.1377 @ ep 274
  - `Generation/extended_KID_mean_val`: last 0.0807, best 0.0799 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0809, best 0.0809 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.1238, best 0.1238 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.1248, best 0.1248 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5208 (min 0.3491, max 1.7725)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9252 (min 0.3272, max 0.9736)
  - `Validation/MS-SSIM_bravo`: last 0.9274 (min 0.3346, max 0.9811)
  - `Validation/PSNR_bravo`: last 30.9216 (min 10.0674, max 36.2981)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.1558
  - `Generation_Diversity/extended_MSSSIM`: 0.2927

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008943
  - `regional_bravo/large`: 0.0048
  - `regional_bravo/medium`: 0.0098
  - `regional_bravo/small`: 0.0155
  - `regional_bravo/tiny`: 0.0113
  - `regional_bravo/tumor_bg_ratio`: 10.6339
  - `regional_bravo/tumor_loss`: 0.0095

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0315, max 8.9473 @ ep 1
  - `training/grad_norm_max`: last 0.2219, max 36.1751 @ ep 5

#### `exp1k_1_pixel_bravo_offset_noise_20260311-201606`
*started 2026-03-11 20:16 • 500 epochs • 42h26m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0009 → 0.0026 (min 0.0019 @ ep 422)
  - `Loss/MSE_val`: 0.9715 → 0.0041 (min 0.0026 @ ep 266)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0207
  - 0.1-0.2: 0.0140
  - 0.2-0.3: 0.0099
  - 0.3-0.4: 0.0059
  - 0.4-0.5: 0.0043
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0029
  - 0.7-0.8: 0.0023
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0397, best 0.0256 @ ep 263
  - `Generation/KID_mean_train`: last 0.0361, best 0.0228 @ ep 343
  - `Generation/KID_std_val`: last 0.0034, best 0.0028 @ ep 263
  - `Generation/KID_std_train`: last 0.0042, best 0.0025 @ ep 379
  - `Generation/CMMD_val`: last 0.2017, best 0.1642 @ ep 348
  - `Generation/CMMD_train`: last 0.1929, best 0.1507 @ ep 348
  - `Generation/extended_KID_mean_val`: last 0.0482, best 0.0364 @ ep 224
  - `Generation/extended_KID_mean_train`: last 0.0413, best 0.0295 @ ep 224
  - `Generation/extended_CMMD_val`: last 0.1737, best 0.1737 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.1623, best 0.1623 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7081 (min 0.4531, max 1.7827)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9252 (min 0.2325, max 0.9586)
  - `Validation/MS-SSIM_bravo`: last 0.9255 (min 0.2744, max 0.9611)
  - `Validation/PSNR_bravo`: last 30.6049 (min 10.5897, max 33.4156)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1817
  - `Generation_Diversity/extended_MSSSIM`: 0.0844

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009366
  - `regional_bravo/large`: 0.0110
  - `regional_bravo/medium`: 0.0121
  - `regional_bravo/small`: 0.0140
  - `regional_bravo/tiny`: 0.0103
  - `regional_bravo/tumor_bg_ratio`: 12.5404
  - `regional_bravo/tumor_loss`: 0.0117

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0303, max 4.0293 @ ep 0
  - `training/grad_norm_max`: last 0.1646, max 13.9792 @ ep 8

#### `exp1p_1_pixel_bravo_uniform_timestep_20260312-014123`
*started 2026-03-12 01:41 • 500 epochs • 75h39m • 16018.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9726 → 0.0050 (min 0.0037 @ ep 456)
  - `Loss/MSE_val`: 0.9379 → 0.0120 (min 0.0026 @ ep 441)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1035
  - 0.1-0.2: 0.0094
  - 0.2-0.3: 0.0056
  - 0.3-0.4: 0.0039
  - 0.4-0.5: 0.0029
  - 0.5-0.6: 0.0023
  - 0.6-0.7: 0.0015
  - 0.7-0.8: 0.0013
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0017

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0531, best 0.0169 @ ep 435
  - `Generation/KID_mean_train`: last 0.0542, best 0.0168 @ ep 435
  - `Generation/KID_std_val`: last 0.0040, best 0.0025 @ ep 496
  - `Generation/KID_std_train`: last 0.0064, best 0.0024 @ ep 416
  - `Generation/CMMD_val`: last 0.1904, best 0.1581 @ ep 391
  - `Generation/CMMD_train`: last 0.1795, best 0.1494 @ ep 391
  - `Generation/extended_KID_mean_val`: last 0.0148, best 0.0148 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0154, best 0.0143 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.1637, best 0.1538 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1597, best 0.1456 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6138 (min 0.4540, max 1.7304)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9556 (min 0.2339, max 0.9573)
  - `Validation/MS-SSIM_bravo`: last 0.9523 (min 0.3505, max 0.9678)
  - `Validation/PSNR_bravo`: last 34.5435 (min 14.1757, max 36.6929)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.9743
  - `Generation_Diversity/extended_MSSSIM`: 0.1133

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005653
  - `regional_bravo/large`: 0.0053
  - `regional_bravo/medium`: 0.0060
  - `regional_bravo/small`: 0.0082
  - `regional_bravo/tiny`: 0.0062
  - `regional_bravo/tumor_bg_ratio`: 10.9959
  - `regional_bravo/tumor_loss`: 0.0062

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0466, max 3.8665 @ ep 0
  - `training/grad_norm_max`: last 0.1869, max 49.5639 @ ep 21

#### `exp1s_pixel_bravo_weight_decay_20260313-141301`
*started 2026-03-13 14:13 • 500 epochs • 15h03m • 1954.2 TFLOPs • peak VRAM 13.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9668 → 0.0015 (min 0.0013 @ ep 494)
  - `Loss/MSE_val`: 0.9214 → 0.0054 (min 0.0023 @ ep 243)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0360
  - 0.1-0.2: 0.0345
  - 0.2-0.3: 0.0150
  - 0.3-0.4: 0.0060
  - 0.4-0.5: 0.0037
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0020
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0009132

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0305, best 0.0071 @ ep 221
  - `Generation/KID_mean_train`: last 0.0307, best 0.0066 @ ep 221
  - `Generation/KID_std_val`: last 0.0022, best 0.0016 @ ep 293
  - `Generation/KID_std_train`: last 0.0025, best 0.0015 @ ep 388
  - `Generation/CMMD_val`: last 0.1655, best 0.1160 @ ep 226
  - `Generation/CMMD_train`: last 0.1596, best 0.1061 @ ep 226
  - `Generation/extended_KID_mean_val`: last 0.0161, best 0.0123 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0155, best 0.0091 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.1379, best 0.1166 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.1345, best 0.1098 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5986 (min 0.4580, max 1.8933)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9184 (min 0.2873, max 0.9576)
  - `Validation/MS-SSIM_bravo`: last 0.9149 (min 0.3317, max 0.9610)
  - `Validation/PSNR_bravo`: last 30.4891 (min 10.6978, max 33.6593)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4096
  - `Generation_Diversity/extended_MSSSIM`: 0.0934

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009563
  - `regional_bravo/large`: 0.0089
  - `regional_bravo/medium`: 0.0152
  - `regional_bravo/small`: 0.0180
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 13.6452
  - `regional_bravo/tumor_loss`: 0.0130

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0236, max 4.3219 @ ep 42
  - `training/grad_norm_max`: last 0.1156, max 28.0927 @ ep 42

#### `exp1r_pixel_bravo_attn_dropout_20260313-142342`
*started 2026-03-13 14:23 • 500 epochs • 17h31m • 1954.2 TFLOPs • peak VRAM 13.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9683 → 0.0014 (min 0.0009999 @ ep 438)
  - `Loss/MSE_val`: 0.9261 → 0.0053 (min 0.0025 @ ep 154)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0646
  - 0.1-0.2: 0.0391
  - 0.2-0.3: 0.0130
  - 0.3-0.4: 0.0063
  - 0.4-0.5: 0.0050
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0030
  - 0.7-0.8: 0.0032
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0216, best 0.0078 @ ep 399
  - `Generation/KID_mean_train`: last 0.0223, best 0.0077 @ ep 295
  - `Generation/KID_std_val`: last 0.0029, best 0.0016 @ ep 246
  - `Generation/KID_std_train`: last 0.0034, best 0.0016 @ ep 434
  - `Generation/CMMD_val`: last 0.1830, best 0.1259 @ ep 312
  - `Generation/CMMD_train`: last 0.1803, best 0.1211 @ ep 312
  - `Generation/extended_KID_mean_val`: last 0.0139, best 0.0069 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0121, best 0.0061 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.1017, best 0.1017 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.0992, best 0.0992 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4964 (min 0.4597, max 1.8834)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9082 (min 0.2875, max 0.9572)
  - `Validation/MS-SSIM_bravo`: last 0.9062 (min 0.3198, max 0.9614)
  - `Validation/PSNR_bravo`: last 29.9694 (min 10.5333, max 33.7079)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4735
  - `Generation_Diversity/extended_MSSSIM`: 0.2090

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0105
  - `regional_bravo/medium`: 0.0169
  - `regional_bravo/small`: 0.0208
  - `regional_bravo/tiny`: 0.0110
  - `regional_bravo/tumor_bg_ratio`: 13.6098
  - `regional_bravo/tumor_loss`: 0.0148

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0142, max 3.8642 @ ep 0
  - `training/grad_norm_max`: last 0.0604, max 16.2849 @ ep 5

#### `exp1v3_pixel_triple_20260326-164117`
*started 2026-03-26 16:41 • 500 epochs • 22h59m • 1954.2 TFLOPs • peak VRAM 23.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9904 → 0.0029 (min 0.0021 @ ep 496)
  - `Loss/MSE_val`: 0.9680 → 0.0034 (min 0.0022 @ ep 297)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0128
  - 0.1-0.2: 0.0093
  - 0.2-0.3: 0.0067
  - 0.3-0.4: 0.0033
  - 0.4-0.5: 0.0030
  - 0.5-0.6: 0.0020
  - 0.6-0.7: 0.0029
  - 0.7-0.8: 0.0021
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0492, best 0.0117 @ ep 416
  - `Generation/KID_mean_train`: last 0.0508, best 0.0112 @ ep 274
  - `Generation/KID_std_val`: last 0.0058, best 0.0014 @ ep 448
  - `Generation/KID_std_train`: last 0.0068, best 0.0015 @ ep 448
  - `Generation/CMMD_val`: last 0.2380, best 0.1282 @ ep 401
  - `Generation/CMMD_train`: last 0.2327, best 0.1207 @ ep 401
  - `Generation/extended_KID_mean_val`: last 0.0335, best 0.0309 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0330, best 0.0330 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1497, best 0.1372 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.1481, best 0.1294 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_flair`: last 0.8255 (min 0.5650, max 1.8646)
  - `Validation/LPIPS_t1_gd`: last 0.5084 (min 0.3234, max 1.8739)
  - `Validation/LPIPS_t1_pre`: last 0.6199 (min 0.3774, max 1.8530)
  - `Validation/LPIPS_triple`: last 0.6513 (min 0.4220, max 1.8591)
  - `Validation/MS-SSIM-3D_triple`: last 0.9549 (min 0.3088, max 0.9657)
  - `Validation/MS-SSIM_flair`: last 0.9447 (min 0.4001, max 0.9623)
  - `Validation/MS-SSIM_t1_gd`: last 0.9590 (min 0.4016, max 0.9736)
  - `Validation/MS-SSIM_t1_pre`: last 0.9612 (min 0.4277, max 0.9750)
  - `Validation/MS-SSIM_triple`: last 0.9550 (min 0.4098, max 0.9702)
  - `Validation/PSNR_flair`: last 31.8991 (min 11.6276, max 33.4117)
  - `Validation/PSNR_t1_gd`: last 33.5193 (min 11.6177, max 35.2944)
  - `Validation/PSNR_t1_pre`: last 32.0354 (min 11.4926, max 33.7774)
  - `Validation/PSNR_triple`: last 32.4846 (min 11.5793, max 34.1089)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3902
  - `Generation_Diversity/extended_MSSSIM`: 0.1497

**Regional loss (final):**
  - `regional_triple/background_loss`: 0.0006616
  - `regional_triple/large`: 0.0130
  - `regional_triple/medium`: 0.0120
  - `regional_triple/small`: 0.0104
  - `regional_triple/tiny`: 0.0055
  - `regional_triple/tumor_bg_ratio`: 16.4887
  - `regional_triple/tumor_loss`: 0.0109

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0466, max 2.1688 @ ep 5
  - `training/grad_norm_max`: last 0.3054, max 27.7440 @ ep 11

#### `exp1v2_pixel_dual_20260326-164147`
*started 2026-03-26 16:41 • 500 epochs • 21h49m • 1954.2 TFLOPs • peak VRAM 23.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9861 → 0.0020 (min 0.0016 @ ep 474)
  - `Loss/MSE_val`: 0.9576 → 0.0035 (min 0.0021 @ ep 321)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0202
  - 0.1-0.2: 0.0099
  - 0.2-0.3: 0.0058
  - 0.3-0.4: 0.0050
  - 0.4-0.5: 0.0031
  - 0.5-0.6: 0.0028
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0488, best 0.0034 @ ep 404
  - `Generation/KID_mean_train`: last 0.0495, best 0.0032 @ ep 404
  - `Generation/KID_std_val`: last 0.0021, best 0.0009544 @ ep 404
  - `Generation/KID_std_train`: last 0.0027, best 0.0012 @ ep 393
  - `Generation/CMMD_val`: last 0.1668, best 0.1066 @ ep 429
  - `Generation/CMMD_train`: last 0.1523, best 0.0880 @ ep 429
  - `Generation/extended_KID_mean_val`: last 0.0144, best 0.0144 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0142, best 0.0142 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1315, best 0.1099 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.1183, best 0.0972 @ ep 449

**Validation quality:**
  - `Validation/LPIPS`: last 0.5441 (min 0.3940, max 1.8683)
  - `Validation/LPIPS_t1_gd`: last 0.4956 (min 0.3486, max 1.8805)
  - `Validation/LPIPS_t1_pre`: last 0.5926 (min 0.4389, max 1.8606)
  - `Validation/MS-SSIM`: last 0.9534 (min 0.2981, max 0.9746)
  - `Validation/MS-SSIM-3D`: last 0.9517 (min 0.2958, max 0.9711)
  - `Validation/MS-SSIM_t1_gd`: last 0.9538 (min 0.2924, max 0.9746)
  - `Validation/MS-SSIM_t1_pre`: last 0.9530 (min 0.3038, max 0.9746)
  - `Validation/PSNR`: last 32.2338 (min 9.8100, max 34.6217)
  - `Validation/PSNR_t1_gd`: last 33.1439 (min 9.8341, max 35.3778)
  - `Validation/PSNR_t1_pre`: last 31.3238 (min 9.7860, max 33.8656)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6370
  - `Generation_Diversity/extended_MSSSIM`: 0.1343

**Regional loss (final):**
  - `regional/background_loss`: 0.0006788
  - `regional/large`: 0.0110
  - `regional/medium`: 0.0053
  - `regional/small`: 0.0100
  - `regional/tiny`: 0.0074
  - `regional/tumor_bg_ratio`: 12.0734
  - `regional/tumor_loss`: 0.0082

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0559, max 6.6546 @ ep 69
  - `training/grad_norm_max`: last 0.2830, max 50.2178 @ ep 69

#### `exp1v2_1_pixel_dual_joint_norm_20260327-031630`
*started 2026-03-27 03:16 • 500 epochs • 29h53m • 1954.2 TFLOPs • peak VRAM 21.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9835 → 0.0015 (min 0.0014 @ ep 481)
  - `Loss/MSE_val`: 0.9545 → 0.0043 (min 0.0019 @ ep 294)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0306
  - 0.1-0.2: 0.0074
  - 0.2-0.3: 0.0059
  - 0.3-0.4: 0.0043
  - 0.4-0.5: 0.0032
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0016
  - 0.7-0.8: 0.0012
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0237, best 0.0063 @ ep 381
  - `Generation/KID_mean_train`: last 0.0237, best 0.0065 @ ep 329
  - `Generation/KID_std_val`: last 0.0021, best 0.0012 @ ep 307
  - `Generation/KID_std_train`: last 0.0022, best 0.0012 @ ep 307
  - `Generation/CMMD_val`: last 0.1207, best 0.1005 @ ep 324
  - `Generation/CMMD_train`: last 0.1096, best 0.0840 @ ep 324
  - `Generation/extended_KID_mean_val`: last 0.0347, best 0.0150 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.0354, best 0.0146 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.1096, best 0.1096 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.1000, best 0.1000 @ ep 499

**Validation quality:**
  - `Validation/LPIPS`: last 0.4337 (min 0.4270, max 1.8701)
  - `Validation/LPIPS_t1_gd`: last 0.4551 (min 0.4337, max 1.8733)
  - `Validation/LPIPS_t1_pre`: last 0.4122 (min 0.4122, max 1.8685)
  - `Validation/MS-SSIM`: last 0.9663 (min 0.3340, max 0.9758)
  - `Validation/MS-SSIM-3D`: last 0.9530 (min 0.2966, max 0.9705)
  - `Validation/MS-SSIM_t1_gd`: last 0.9655 (min 0.3379, max 0.9751)
  - `Validation/MS-SSIM_t1_pre`: last 0.9670 (min 0.3259, max 0.9764)
  - `Validation/PSNR`: last 33.6455 (min 10.4837, max 34.9611)
  - `Validation/PSNR_t1_gd`: last 33.3076 (min 10.5081, max 34.6578)
  - `Validation/PSNR_t1_pre`: last 33.9833 (min 10.4594, max 35.2644)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2581
  - `Generation_Diversity/extended_MSSSIM`: 0.1068

**Regional loss (final):**
  - `regional/background_loss`: 0.000475
  - `regional/large`: 0.0084
  - `regional/medium`: 0.0052
  - `regional/small`: 0.0087
  - `regional/tiny`: 0.0066
  - `regional/tumor_bg_ratio`: 14.7406
  - `regional/tumor_loss`: 0.0070

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0424, max 2.8875 @ ep 39
  - `training/grad_norm_max`: last 0.1116, max 39.3183 @ ep 39

#### `exp1v3_1_pixel_triple_joint_norm_20260327-033611`
*started 2026-03-27 03:36 • 500 epochs • 32h26m • 1954.2 TFLOPs • peak VRAM 23.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9864 → 0.0020 (min 0.0017 @ ep 430)
  - `Loss/MSE_val`: 0.9636 → 0.0029 (min 0.0019 @ ep 303)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0115
  - 0.1-0.2: 0.0077
  - 0.2-0.3: 0.0072
  - 0.3-0.4: 0.0029
  - 0.4-0.5: 0.0039
  - 0.5-0.6: 0.0020
  - 0.6-0.7: 0.0014
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0009832

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0954, best 0.0395 @ ep 309
  - `Generation/KID_mean_train`: last 0.0925, best 0.0366 @ ep 309
  - `Generation/KID_std_val`: last 0.0037, best 0.0021 @ ep 308
  - `Generation/KID_std_train`: last 0.0038, best 0.0023 @ ep 465
  - `Generation/CMMD_val`: last 0.4564, best 0.2955 @ ep 353
  - `Generation/CMMD_train`: last 0.4310, best 0.2818 @ ep 353
  - `Generation/extended_KID_mean_val`: last 0.0635, best 0.0635 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0611, best 0.0611 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.3590, best 0.3370 @ ep 299
  - `Generation/extended_CMMD_train`: last 0.3491, best 0.3285 @ ep 324

**Validation quality:**
  - `Validation/LPIPS_flair`: last 0.9601 (min 0.7550, max 1.8505)
  - `Validation/LPIPS_t1_gd`: last 0.7354 (min 0.4960, max 1.8950)
  - `Validation/LPIPS_t1_pre`: last 0.7339 (min 0.5111, max 1.8856)
  - `Validation/LPIPS_triple`: last 0.8098 (min 0.6109, max 1.8748)
  - `Validation/MS-SSIM-3D_triple`: last 0.9492 (min 0.2773, max 0.9645)
  - `Validation/MS-SSIM_flair`: last 0.9327 (min 0.3312, max 0.9624)
  - `Validation/MS-SSIM_t1_gd`: last 0.9568 (min 0.2956, max 0.9735)
  - `Validation/MS-SSIM_t1_pre`: last 0.9586 (min 0.3052, max 0.9736)
  - `Validation/MS-SSIM_triple`: last 0.9494 (min 0.3151, max 0.9692)
  - `Validation/PSNR_flair`: last 30.4760 (min 9.5497, max 32.3705)
  - `Validation/PSNR_t1_gd`: last 35.1650 (min 9.6209, max 36.8876)
  - `Validation/PSNR_t1_pre`: last 35.6156 (min 9.6385, max 37.5065)
  - `Validation/PSNR_triple`: last 33.7522 (min 9.6031, max 35.5082)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1350
  - `Generation_Diversity/extended_MSSSIM`: 0.0937

**Regional loss (final):**
  - `regional_triple/background_loss`: 0.0005789
  - `regional_triple/large`: 0.0095
  - `regional_triple/medium`: 0.0082
  - `regional_triple/small`: 0.0081
  - `regional_triple/tiny`: 0.0063
  - `regional_triple/tumor_bg_ratio`: 14.2398
  - `regional_triple/tumor_loss`: 0.0082

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0423, max 8.8072 @ ep 45
  - `training/grad_norm_max`: last 0.1929, max 60.5976 @ ep 45

#### `exp1_1_1000_pixel_bravo_20260402-121556`
*started 2026-04-02 12:15 • 1000 epochs • 95h33m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9676 → 0.0026 (min 0.0012 @ ep 863)
  - `Loss/MSE_val`: 0.9240 → 0.0085 (min 0.0020 @ ep 428)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0293
  - 0.1-0.2: 0.0175
  - 0.2-0.3: 0.0133
  - 0.3-0.4: 0.0093
  - 0.4-0.5: 0.0060
  - 0.5-0.6: 0.0040
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0026
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0226, best 0.0129 @ ep 585
  - `Generation/KID_mean_train`: last 0.0224, best 0.0125 @ ep 585
  - `Generation/KID_std_val`: last 0.0037, best 0.0019 @ ep 794
  - `Generation/KID_std_train`: last 0.0033, best 0.0022 @ ep 712
  - `Generation/CMMD_val`: last 0.1400, best 0.1283 @ ep 679
  - `Generation/CMMD_train`: last 0.1378, best 0.1205 @ ep 679
  - `Generation/extended_KID_mean_val`: last 0.0146, best 0.0108 @ ep 749
  - `Generation/extended_KID_mean_train`: last 0.0130, best 0.0077 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1258, best 0.1244 @ ep 649
  - `Generation/extended_CMMD_train`: last 0.1252, best 0.1187 @ ep 649

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4129 (min 0.4012, max 1.7868)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9066 (min 0.2356, max 0.9610)
  - `Validation/MS-SSIM_bravo`: last 0.9192 (min 0.2526, max 0.9670)
  - `Validation/PSNR_bravo`: last 30.3864 (min 10.2723, max 34.1997)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2099
  - `Generation_Diversity/extended_MSSSIM`: 0.1001

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009897
  - `regional_bravo/large`: 0.0158
  - `regional_bravo/medium`: 0.0159
  - `regional_bravo/small`: 0.0133
  - `regional_bravo/tiny`: 0.0073
  - `regional_bravo/tumor_bg_ratio`: 13.7952
  - `regional_bravo/tumor_loss`: 0.0137

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0135, max 4.4326 @ ep 57
  - `training/grad_norm_max`: last 0.4625, max 23.5858 @ ep 57

#### `exp1t_1_pixel_bravo_mixup_20260402-131808`
*started 2026-04-02 13:18 • 500 epochs • 52h45m • 16018.2 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9668 → 0.0025 (min 0.0020 @ ep 387)
  - `Loss/MSE_val`: 0.9226 → 0.0030 (min 0.0021 @ ep 454)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0117
  - 0.1-0.2: 0.0079
  - 0.2-0.3: 0.0061
  - 0.3-0.4: 0.0034
  - 0.4-0.5: 0.0024
  - 0.5-0.6: 0.0023
  - 0.6-0.7: 0.0017
  - 0.7-0.8: 0.0018
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0449, best 0.0308 @ ep 373
  - `Generation/KID_mean_train`: last 0.0434, best 0.0284 @ ep 487
  - `Generation/KID_std_val`: last 0.0043, best 0.0030 @ ep 485
  - `Generation/KID_std_train`: last 0.0054, best 0.0027 @ ep 400
  - `Generation/CMMD_val`: last 0.2836, best 0.2030 @ ep 256
  - `Generation/CMMD_train`: last 0.2702, best 0.1903 @ ep 280
  - `Generation/extended_KID_mean_val`: last 0.0471, best 0.0266 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0416, best 0.0209 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.2632, best 0.1895 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.2506, best 0.1765 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5875 (min 0.4860, max 1.7818)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9552 (min 0.2359, max 0.9601)
  - `Validation/MS-SSIM_bravo`: last 0.9506 (min 0.2881, max 0.9650)
  - `Validation/PSNR_bravo`: last 32.6973 (min 10.9845, max 33.9643)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4218
  - `Generation_Diversity/extended_MSSSIM`: 0.1632

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006064
  - `regional_bravo/large`: 0.0089
  - `regional_bravo/medium`: 0.0073
  - `regional_bravo/small`: 0.0086
  - `regional_bravo/tiny`: 0.0087
  - `regional_bravo/tumor_bg_ratio`: 13.7617
  - `regional_bravo/tumor_loss`: 0.0083

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0151, max 4.8246 @ ep 66
  - `training/grad_norm_max`: last 0.0692, max 30.9929 @ ep 67

#### `exp1v3_2_pixel_triple_256_20260403-035904`
*started 2026-04-03 03:59 • 500 epochs • 43h09m • 16018.2 TFLOPs • peak VRAM 43.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9905 → 0.0027 (min 0.0021 @ ep 478)
  - `Loss/MSE_val`: 0.9683 → 0.0023 (min 0.0020 @ ep 359)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0223
  - 0.1-0.2: 0.0068
  - 0.2-0.3: 0.0029
  - 0.3-0.4: 0.0027
  - 0.4-0.5: 0.0026
  - 0.5-0.6: 0.0018
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0012
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0475, best 0.0310 @ ep 495
  - `Generation/KID_mean_train`: last 0.0522, best 0.0324 @ ep 495
  - `Generation/KID_std_val`: last 0.0056, best 0.0032 @ ep 495
  - `Generation/KID_std_train`: last 0.0069, best 0.0032 @ ep 408
  - `Generation/CMMD_val`: last 0.2190, best 0.2170 @ ep 459
  - `Generation/CMMD_train`: last 0.2048, best 0.1999 @ ep 459
  - `Generation/extended_KID_mean_val`: last 0.0472, best 0.0446 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0494, best 0.0451 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2299, best 0.2263 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2195, best 0.2159 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_flair`: last 0.9041 (min 0.5476, max 1.7628)
  - `Validation/LPIPS_t1_gd`: last 0.7052 (min 0.4469, max 1.7796)
  - `Validation/LPIPS_t1_pre`: last 0.6550 (min 0.4656, max 1.7772)
  - `Validation/LPIPS_triple`: last 0.7548 (min 0.5030, max 1.7732)
  - `Validation/MS-SSIM-3D_triple`: last 0.9638 (min 0.2477, max 0.9664)
  - `Validation/MS-SSIM_flair`: last 0.9450 (min 0.3197, max 0.9606)
  - `Validation/MS-SSIM_t1_gd`: last 0.9632 (min 0.3127, max 0.9740)
  - `Validation/MS-SSIM_t1_pre`: last 0.9646 (min 0.3261, max 0.9760)
  - `Validation/MS-SSIM_triple`: last 0.9576 (min 0.3195, max 0.9699)
  - `Validation/PSNR_flair`: last 32.4067 (min 11.2545, max 33.7869)
  - `Validation/PSNR_t1_gd`: last 34.4823 (min 11.2819, max 35.8760)
  - `Validation/PSNR_t1_pre`: last 33.3142 (min 11.2193, max 34.8256)
  - `Validation/PSNR_triple`: last 33.4011 (min 11.2519, max 34.8295)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5247
  - `Generation_Diversity/extended_MSSSIM`: 0.2940

**Regional loss (final):**
  - `regional_triple/background_loss`: 0.0005849
  - `regional_triple/large`: 0.0111
  - `regional_triple/medium`: 0.0075
  - `regional_triple/small`: 0.0084
  - `regional_triple/tiny`: 0.0053
  - `regional_triple/tumor_bg_ratio`: 14.4061
  - `regional_triple/tumor_loss`: 0.0084

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0569, max 1.5425 @ ep 9
  - `training/grad_norm_max`: last 0.2258, max 31.5412 @ ep 12

#### `exp1s_01_pixel_bravo_weight_decay_20260403-040937`
*started 2026-04-03 04:09 • 1000 epochs • 99h50m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9678 → 0.0012 (min 0.0011 @ ep 913)
  - `Loss/MSE_val`: 0.9234 → 0.0088 (min 0.0022 @ ep 314)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0452
  - 0.1-0.2: 0.0229
  - 0.2-0.3: 0.0144
  - 0.3-0.4: 0.0065
  - 0.4-0.5: 0.0060
  - 0.5-0.6: 0.0037
  - 0.6-0.7: 0.0039
  - 0.7-0.8: 0.0024
  - 0.8-0.9: 0.0025
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0223, best 0.0132 @ ep 828
  - `Generation/KID_mean_train`: last 0.0247, best 0.0120 @ ep 828
  - `Generation/KID_std_val`: last 0.0040, best 0.0018 @ ep 294
  - `Generation/KID_std_train`: last 0.0032, best 0.0021 @ ep 460
  - `Generation/CMMD_val`: last 0.1606, best 0.1130 @ ep 735
  - `Generation/CMMD_train`: last 0.1660, best 0.1015 @ ep 735
  - `Generation/extended_KID_mean_val`: last 0.0124, best 0.0092 @ ep 949
  - `Generation/extended_KID_mean_train`: last 0.0126, best 0.0085 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.1030, best 0.1030 @ ep 999
  - `Generation/extended_CMMD_train`: last 0.1035, best 0.1035 @ ep 999

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4132 (min 0.3850, max 1.7859)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9062 (min 0.2361, max 0.9610)
  - `Validation/MS-SSIM_bravo`: last 0.9117 (min 0.2865, max 0.9687)
  - `Validation/PSNR_bravo`: last 30.1231 (min 11.0362, max 34.2685)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2221
  - `Generation_Diversity/extended_MSSSIM`: 0.1126

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0141
  - `regional_bravo/medium`: 0.0179
  - `regional_bravo/small`: 0.0202
  - `regional_bravo/tiny`: 0.0104
  - `regional_bravo/tumor_bg_ratio`: 14.6054
  - `regional_bravo/tumor_loss`: 0.0156

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0081, max 3.8917 @ ep 0
  - `training/grad_norm_max`: last 0.0494, max 11.0896 @ ep 4

#### `exp1v2_2_pixel_dual_256_20260404-140254`
*started 2026-04-04 14:02 • 500 epochs • 46h41m • 16018.2 TFLOPs • peak VRAM 42.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9863 → 0.0020 (min 0.0016 @ ep 423)
  - `Loss/MSE_val`: 0.9579 → 0.0019 (min 0.0016 @ ep 417)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0126
  - 0.1-0.2: 0.0076
  - 0.2-0.3: 0.0027
  - 0.3-0.4: 0.0020
  - 0.4-0.5: 0.0015
  - 0.5-0.6: 0.0014
  - 0.6-0.7: 0.0013
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0696, best 0.0275 @ ep 417
  - `Generation/KID_mean_train`: last 0.0694, best 0.0275 @ ep 417
  - `Generation/KID_std_val`: last 0.0032, best 0.0023 @ ep 474
  - `Generation/KID_std_train`: last 0.0029, best 0.0023 @ ep 474
  - `Generation/CMMD_val`: last 0.2499, best 0.1632 @ ep 498
  - `Generation/CMMD_train`: last 0.2374, best 0.1489 @ ep 498
  - `Generation/extended_KID_mean_val`: last 0.0375, best 0.0211 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0368, best 0.0198 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.1865, best 0.1786 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.1796, best 0.1741 @ ep 474

**Validation quality:**
  - `Validation/LPIPS`: last 0.5468 (min 0.3894, max 1.7892)
  - `Validation/LPIPS_t1_gd`: last 0.5842 (min 0.3735, max 1.7903)
  - `Validation/LPIPS_t1_pre`: last 0.5093 (min 0.3904, max 1.7882)
  - `Validation/MS-SSIM`: last 0.9682 (min 0.2671, max 0.9763)
  - `Validation/MS-SSIM-3D`: last 0.9707 (min 0.2459, max 0.9722)
  - `Validation/MS-SSIM_t1_gd`: last 0.9673 (min 0.2596, max 0.9758)
  - `Validation/MS-SSIM_t1_pre`: last 0.9690 (min 0.2746, max 0.9768)
  - `Validation/PSNR`: last 34.2957 (min 10.0674, max 35.7098)
  - `Validation/PSNR_t1_gd`: last 34.9487 (min 10.1067, max 36.2677)
  - `Validation/PSNR_t1_pre`: last 33.6427 (min 10.0281, max 35.1707)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6282
  - `Generation_Diversity/extended_MSSSIM`: 0.2129

**Regional loss (final):**
  - `regional/background_loss`: 0.0004321
  - `regional/large`: 0.0084
  - `regional/medium`: 0.0060
  - `regional/small`: 0.0076
  - `regional/tiny`: 0.0044
  - `regional/tumor_bg_ratio`: 15.6253
  - `regional/tumor_loss`: 0.0068

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0450, max 6.0491 @ ep 51
  - `training/grad_norm_max`: last 0.2412, max 27.9525 @ ep 51

#### `exp1s_02_pixel_bravo_weight_decay_128_20260404-144426`
*started 2026-04-04 14:44 • 500 epochs • 16h05m • 1954.2 TFLOPs • peak VRAM 13.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9675 → 0.0019 (min 0.0013 @ ep 468)
  - `Loss/MSE_val`: 0.9238 → 0.0054 (min 0.0023 @ ep 182)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0205
  - 0.1-0.2: 0.0249
  - 0.2-0.3: 0.0095
  - 0.3-0.4: 0.0060
  - 0.4-0.5: 0.0062
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0026
  - 0.7-0.8: 0.0026
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0189, best 0.0097 @ ep 254
  - `Generation/KID_mean_train`: last 0.0178, best 0.0088 @ ep 254
  - `Generation/KID_std_val`: last 0.0023, best 0.0016 @ ep 403
  - `Generation/KID_std_train`: last 0.0021, best 0.0012 @ ep 403
  - `Generation/CMMD_val`: last 0.1718, best 0.1174 @ ep 248
  - `Generation/CMMD_train`: last 0.1661, best 0.1079 @ ep 248
  - `Generation/extended_KID_mean_val`: last 0.0120, best 0.0120 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0112, best 0.0101 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1127, best 0.1119 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.1058, best 0.0988 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5441 (min 0.4768, max 1.8808)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9142 (min 0.2846, max 0.9576)
  - `Validation/MS-SSIM_bravo`: last 0.9138 (min 0.3544, max 0.9636)
  - `Validation/PSNR_bravo`: last 30.3360 (min 11.2667, max 34.0725)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3702
  - `Generation_Diversity/extended_MSSSIM`: 0.1222

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009886
  - `regional_bravo/large`: 0.0097
  - `regional_bravo/medium`: 0.0178
  - `regional_bravo/small`: 0.0150
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 13.8153
  - `regional_bravo/tumor_loss`: 0.0137

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0154, max 3.8599 @ ep 0
  - `training/grad_norm_max`: last 0.0975, max 4.6850 @ ep 2

#### `exp1t_pixel_bravo_mixup_128_20260405-042205`
*started 2026-04-05 04:22 • 500 epochs • 29h04m • 1954.2 TFLOPs • peak VRAM 23.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9673 → 0.0020 (min 0.0016 @ ep 472)
  - `Loss/MSE_val`: 0.9240 → 0.0048 (min 0.0022 @ ep 191)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0158
  - 0.1-0.2: 0.0113
  - 0.2-0.3: 0.0076
  - 0.3-0.4: 0.0058
  - 0.4-0.5: 0.0035
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0031
  - 0.7-0.8: 0.0013
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0559, best 0.0124 @ ep 341
  - `Generation/KID_mean_train`: last 0.0509, best 0.0118 @ ep 220
  - `Generation/KID_std_val`: last 0.0063, best 0.0018 @ ep 248
  - `Generation/KID_std_train`: last 0.0056, best 0.0017 @ ep 358
  - `Generation/CMMD_val`: last 0.3021, best 0.1143 @ ep 301
  - `Generation/CMMD_train`: last 0.2853, best 0.0978 @ ep 301
  - `Generation/extended_KID_mean_val`: last 0.0327, best 0.0234 @ ep 199
  - `Generation/extended_KID_mean_train`: last 0.0294, best 0.0201 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1831, best 0.1350 @ ep 274
  - `Generation/extended_CMMD_train`: last 0.1749, best 0.1205 @ ep 274

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5849 (min 0.4934, max 1.8829)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9341 (min 0.2881, max 0.9572)
  - `Validation/MS-SSIM_bravo`: last 0.9378 (min 0.3126, max 0.9634)
  - `Validation/PSNR_bravo`: last 32.1074 (min 10.3634, max 33.8305)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3846
  - `Generation_Diversity/extended_MSSSIM`: 0.1526

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006652
  - `regional_bravo/large`: 0.0093
  - `regional_bravo/medium`: 0.0118
  - `regional_bravo/small`: 0.0109
  - `regional_bravo/tiny`: 0.0079
  - `regional_bravo/tumor_bg_ratio`: 15.4508
  - `regional_bravo/tumor_loss`: 0.0103

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0203, max 3.8938 @ ep 0
  - `training/grad_norm_max`: last 0.0808, max 20.5688 @ ep 68

#### `exp1_1_156_pixel_bravo_20260411-024806`
*started 2026-04-11 02:48 • 1000 epochs • 169h48m • 47597.0 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9465 → 0.0017 (min 0.0010 @ ep 834)
  - `Loss/MSE_val`: 0.8795 → 0.0024 (min 0.0006384 @ ep 944)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0211
  - 0.1-0.2: 0.0040
  - 0.2-0.3: 0.0023
  - 0.3-0.4: 0.0010
  - 0.4-0.5: 0.0007811
  - 0.5-0.6: 0.0005846
  - 0.6-0.7: 0.0004508
  - 0.7-0.8: 0.000405
  - 0.8-0.9: 0.0005896
  - 0.9-1.0: 0.0006918

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0185, best 0.0124 @ ep 398
  - `Generation/KID_mean_train`: last 0.0215, best 0.0137 @ ep 398
  - `Generation/KID_std_val`: last 0.0056, best 0.0022 @ ep 771
  - `Generation/KID_std_train`: last 0.0050, best 0.0023 @ ep 761
  - `Generation/CMMD_val`: last 0.1580, best 0.1214 @ ep 616
  - `Generation/CMMD_train`: last 0.1591, best 0.1162 @ ep 616
  - `Generation/extended_KID_mean_val`: last 0.0099, best 0.0081 @ ep 899
  - `Generation/extended_KID_mean_train`: last 0.0098, best 0.0084 @ ep 899
  - `Generation/extended_CMMD_val`: last 0.1185, best 0.0987 @ ep 649
  - `Generation/extended_CMMD_train`: last 0.1214, best 0.1004 @ ep 649

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2820 (min 0.1847, max 1.7791)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9935 (min 0.2415, max 0.9935)
  - `Validation/MS-SSIM_bravo`: last 0.9892 (min 0.2918, max 0.9937)
  - `Validation/PSNR_bravo`: last 37.2627 (min 11.0887, max 38.1497)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3680
  - `Generation_Diversity/extended_MSSSIM`: 0.1654

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0002211
  - `regional_bravo/large`: 0.0023
  - `regional_bravo/medium`: 0.0020
  - `regional_bravo/small`: 0.0020
  - `regional_bravo/tiny`: 0.0022
  - `regional_bravo/tumor_bg_ratio`: 9.5657
  - `regional_bravo/tumor_loss`: 0.0021

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0080, max 4.6146 @ ep 10
  - `training/grad_norm_max`: last 0.2124, max 30.9435 @ ep 10

#### `exp1v2_2_1000_pixel_dual_20260411-024806`
*started 2026-04-11 02:48 • 1000 epochs • 134h14m • 32036.5 TFLOPs • peak VRAM 42.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9869 → 0.0012 (min 0.0008321 @ ep 994)
  - `Loss/MSE_val`: 0.9590 → 0.0048 (min 0.0017 @ ep 444)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0177
  - 0.1-0.2: 0.0124
  - 0.2-0.3: 0.0081
  - 0.3-0.4: 0.0058
  - 0.4-0.5: 0.0044
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0021
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0140, best 0.0065 @ ep 989
  - `Generation/KID_mean_train`: last 0.0146, best 0.0069 @ ep 635
  - `Generation/KID_std_val`: last 0.0027, best 0.0016 @ ep 673
  - `Generation/KID_std_train`: last 0.0033, best 0.0017 @ ep 998
  - `Generation/CMMD_val`: last 0.1428, best 0.0916 @ ep 804
  - `Generation/CMMD_train`: last 0.1383, best 0.0786 @ ep 804
  - `Generation/extended_KID_mean_val`: last 0.0152, best 0.0087 @ ep 949
  - `Generation/extended_KID_mean_train`: last 0.0163, best 0.0084 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.1291, best 0.1090 @ ep 849
  - `Generation/extended_CMMD_train`: last 0.1230, best 0.1040 @ ep 849

**Validation quality:**
  - `Validation/LPIPS`: last 0.3620 (min 0, max 1.7890)
  - `Validation/LPIPS_t1_gd`: last 0.3722 (min 0.2629, max 1.7903)
  - `Validation/LPIPS_t1_pre`: last 0.3518 (min 0.2539, max 1.7877)
  - `Validation/MS-SSIM`: last 0.9380 (min 0.2678, max 0.9778)
  - `Validation/MS-SSIM-3D`: last 0.9338 (min 0.2450, max 0.9738)
  - `Validation/MS-SSIM_t1_gd`: last 0.9394 (min 0.2647, max 0.9786)
  - `Validation/MS-SSIM_t1_pre`: last 0.9366 (min 0.2692, max 0.9772)
  - `Validation/PSNR`: last 31.4300 (min 9.8484, max 35.8954)
  - `Validation/PSNR_t1_gd`: last 32.2558 (min 9.9218, max 36.4551)
  - `Validation/PSNR_t1_pre`: last 30.6041 (min 9.7750, max 35.3357)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1936
  - `Generation_Diversity/extended_MSSSIM`: 0.1130

**Regional loss (final):**
  - `regional/background_loss`: 0.0008224
  - `regional/large`: 0.0114
  - `regional/medium`: 0.0081
  - `regional/small`: 0.0114
  - `regional/tiny`: 0.0069
  - `regional/tumor_bg_ratio`: 11.6441
  - `regional/tumor_loss`: 0.0096

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0189, max 4.1552 @ ep 65
  - `training/grad_norm_max`: last 0.0640, max 35.2132 @ ep 65

#### `exp1_1_1000plus_pixel_bravo_20260411-235425`
*started 2026-04-11 23:54 • 1000 epochs • 128h52m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0023 → 0.0011 (min 0.0005028 @ ep 787)
  - `Loss/MSE_val`: 0.0087 → 0.0101 (min 0.0029 @ ep 13)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0728
  - 0.1-0.2: 0.0384
  - 0.2-0.3: 0.0187
  - 0.3-0.4: 0.0095
  - 0.4-0.5: 0.0083
  - 0.5-0.6: 0.0048
  - 0.6-0.7: 0.0030
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0017

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0283, best 0.0087 @ ep 324
  - `Generation/KID_mean_train`: last 0.0325, best 0.0087 @ ep 324
  - `Generation/KID_std_val`: last 0.0042, best 0.0022 @ ep 123
  - `Generation/KID_std_train`: last 0.0040, best 0.0024 @ ep 252
  - `Generation/CMMD_val`: last 0.1495, best 0.1153 @ ep 243
  - `Generation/CMMD_train`: last 0.1592, best 0.1061 @ ep 131
  - `Generation/extended_KID_mean_val`: last 0.0159, best 0.0102 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0179, best 0.0108 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1100, best 0.1085 @ ep 899
  - `Generation/extended_CMMD_train`: last 0.1206, best 0.1016 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3856 (min 0, max 1.1558)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8856 (min 0.8851, max 0.9489)
  - `Validation/MS-SSIM_bravo`: last 0.8831 (min 0.8770, max 0.9521)
  - `Validation/PSNR_bravo`: last 28.6858 (min 28.4214, max 32.8994)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3769
  - `Generation_Diversity/extended_MSSSIM`: 0.1945

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0014
  - `regional_bravo/large`: 0.0151
  - `regional_bravo/medium`: 0.0210
  - `regional_bravo/small`: 0.0210
  - `regional_bravo/tiny`: 0.0114
  - `regional_bravo/tumor_bg_ratio`: 12.1844
  - `regional_bravo/tumor_loss`: 0.0171

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0058, max 0.0841 @ ep 10
  - `training/grad_norm_max`: last 0.1160, max 2.2120 @ ep 157

#### `exp1v2_2_156_pixel_dual_20260412-143252`
*started 2026-04-12 14:32 • 1000 epochs • 110h04m • 32036.5 TFLOPs • peak VRAM 42.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9998 → 0.000992 (min 0.000713 @ ep 894)
  - `Loss/MSE_val`: 0.9867 → 0.0078 (min 0.0018 @ ep 353)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0413
  - 0.1-0.2: 0.0233
  - 0.2-0.3: 0.0094
  - 0.3-0.4: 0.0071
  - 0.4-0.5: 0.0044
  - 0.5-0.6: 0.0031
  - 0.6-0.7: 0.0037
  - 0.7-0.8: 0.0028
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0016

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0076, best 0.0066 @ ep 855
  - `Generation/KID_mean_train`: last 0.0072, best 0.0067 @ ep 747
  - `Generation/KID_std_val`: last 0.0026, best 0.0014 @ ep 639
  - `Generation/KID_std_train`: last 0.0024, best 0.0017 @ ep 913
  - `Generation/CMMD_val`: last 0.1306, best 0.0834 @ ep 912
  - `Generation/CMMD_train`: last 0.1202, best 0.0722 @ ep 912
  - `Generation/extended_KID_mean_val`: last 0.0252, best 0.0155 @ ep 899
  - `Generation/extended_KID_mean_train`: last 0.0248, best 0.0164 @ ep 899
  - `Generation/extended_CMMD_val`: last 0.1156, best 0.1090 @ ep 949
  - `Generation/extended_CMMD_train`: last 0.1076, best 0.1003 @ ep 949

**Validation quality:**
  - `Validation/LPIPS`: last 0.3529 (min 0.3112, max 1.7754)
  - `Validation/LPIPS_t1_gd`: last 0.3562 (min 0.3257, max 1.7805)
  - `Validation/LPIPS_t1_pre`: last 0.3495 (min 0.2705, max 1.7703)
  - `Validation/MS-SSIM`: last 0.9362 (min 0.3110, max 0.9760)
  - `Validation/MS-SSIM-3D`: last 0.9292 (min 0.2399, max 0.9722)
  - `Validation/MS-SSIM_t1_gd`: last 0.9373 (min 0.3064, max 0.9750)
  - `Validation/MS-SSIM_t1_pre`: last 0.9350 (min 0.3155, max 0.9775)
  - `Validation/PSNR`: last 31.0560 (min 11.2509, max 35.5536)
  - `Validation/PSNR_t1_gd`: last 31.9430 (min 11.3326, max 36.0490)
  - `Validation/PSNR_t1_pre`: last 30.1690 (min 11.1693, max 35.0582)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2063
  - `Generation_Diversity/extended_MSSSIM`: 0.1161

**Regional loss (final):**
  - `regional/background_loss`: 0.0008727
  - `regional/large`: 0.0092
  - `regional/medium`: 0.0062
  - `regional/small`: 0.0095
  - `regional/tiny`: 0.0062
  - `regional/tumor_bg_ratio`: 8.9907
  - `regional/tumor_loss`: 0.0078

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0268, max 2.5491 @ ep 10
  - `training/grad_norm_max`: last 0.2045, max 50.5974 @ ep 9

#### `exp1o_1_pixel_bravo_20260413-023304`
*started 2026-04-13 02:33 • 500 epochs • 84h26m • 16018.2 TFLOPs • peak VRAM 44.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9694 → 0.0025 (min 0.0021 @ ep 495)
  - `Loss/MSE_val`: 0.9571 → 0.0030 (min 0.0028 @ ep 495)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0171
  - 0.1-0.2: 0.0093
  - 0.2-0.3: 0.0057
  - 0.3-0.4: 0.0041
  - 0.4-0.5: 0.0028
  - 0.5-0.6: 0.0028
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0021
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0021

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1508, best 0.1508 @ ep 499
  - `Generation/KID_mean_train`: last 0.1501, best 0.1481 @ ep 497
  - `Generation/KID_std_val`: last 0.0077, best 0.0049 @ ep 371
  - `Generation/KID_std_train`: last 0.0073, best 0.0044 @ ep 312
  - `Generation/CMMD_val`: last 0.2032, best 0.1775 @ ep 490
  - `Generation/CMMD_train`: last 0.1858, best 0.1572 @ ep 490
  - `Generation/extended_KID_mean_val`: last 0.1619, best 0.1619 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.1583, best 0.1583 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1769, best 0.1769 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.1587, best 0.1587 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.4066 (min 1.0660, max 1.7816)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9427 (min 0.2324, max 0.9429)
  - `Validation/MS-SSIM_bravo`: last 0.9236 (min 0.2778, max 0.9528)
  - `Validation/PSNR_bravo`: last 31.4127 (min 10.8063, max 33.3224)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3442
  - `Generation_Diversity/extended_MSSSIM`: 0.1308

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000883
  - `regional_bravo/large`: 0.0104
  - `regional_bravo/medium`: 0.0073
  - `regional_bravo/small`: 0.0117
  - `regional_bravo/tiny`: 0.0080
  - `regional_bravo/tumor_bg_ratio`: 10.5374
  - `regional_bravo/tumor_loss`: 0.0093

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0373, max 3.8553 @ ep 0
  - `training/grad_norm_max`: last 0.1640, max 16.7653 @ ep 89

### exp4

**exp4** — SDA (shifted data augmentation) variant at 128×128×160.

#### `exp4_pixel_bravo_sda_rflow_128x160_20260130-025210`
*started 2026-01-30 02:52 • 500 epochs • 17h30m • 1954.2 TFLOPs • peak VRAM 24.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9730 → 0.0021 (min 0.0016 @ ep 420)
  - `Loss/MSE_val`: 0.9353 → 0.0086 (min 0.0024 @ ep 233)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0407
  - 0.1-0.2: 0.0224
  - 0.2-0.3: 0.0099
  - 0.3-0.4: 0.0063
  - 0.4-0.5: 0.0045
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0032
  - 0.7-0.8: 0.0020
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0010

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0253, best 0.0140 @ ep 266
  - `Generation/KID_mean_train`: last 0.0231, best 0.0122 @ ep 266
  - `Generation/KID_std_val`: last 0.0023, best 0.0017 @ ep 384
  - `Generation/KID_std_train`: last 0.0022, best 0.0015 @ ep 424
  - `Generation/CMMD_val`: last 0.2240, best 0.1829 @ ep 236
  - `Generation/CMMD_train`: last 0.2228, best 0.1744 @ ep 236
  - `Generation/extended_KID_mean_val`: last 0.0337, best 0.0189 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.0300, best 0.0149 @ ep 274
  - `Generation/extended_CMMD_val`: last 0.2071, best 0.1989 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.2079, best 0.1980 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5462 (min 0.5261, max 1.8783)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9242 (min 0.2661, max 0.9565)
  - `Validation/MS-SSIM_bravo`: last 0.9357 (min 0.2702, max 0.9642)
  - `Validation/PSNR_bravo`: last 31.6862 (min 9.5230, max 34.0537)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6401
  - `Generation_Diversity/extended_MSSSIM`: 0.1227

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007727
  - `regional_bravo/large`: 0.0033
  - `regional_bravo/medium`: 0.0082
  - `regional_bravo/small`: 0.0103
  - `regional_bravo/tiny`: 0.0059
  - `regional_bravo/tumor_bg_ratio`: 8.7938
  - `regional_bravo/tumor_loss`: 0.0068

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0974, max 4.9435 @ ep 2
  - `training/grad_norm_max`: last 0.3411, max 86.9110 @ ep 71

### exp5

**exp5** — ScoreAug variant at 128×128×160 (predecessor to exp23).

#### `exp5_1_pixel_bravo_scoreaug_rflow_128x160_20260130-030053`
*started 2026-01-30 03:00 • 500 epochs • 14h46m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9186 → 0.0027 (min 0.0023 @ ep 484)
  - `Loss/MSE_val`: 0.9279 → 0.0031 (min 0.0021 @ ep 471)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0192
  - 0.1-0.2: 0.0080
  - 0.2-0.3: 0.0056
  - 0.3-0.4: 0.0038
  - 0.4-0.5: 0.0021
  - 0.5-0.6: 0.0018
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0012
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0578, best 0.0182 @ ep 460
  - `Generation/KID_mean_train`: last 0.0544, best 0.0163 @ ep 465
  - `Generation/KID_std_val`: last 0.0027, best 0.0020 @ ep 454
  - `Generation/KID_std_train`: last 0.0027, best 0.0016 @ ep 284
  - `Generation/CMMD_val`: last 0.2133, best 0.1809 @ ep 279
  - `Generation/CMMD_train`: last 0.2053, best 0.1776 @ ep 279
  - `Generation/extended_KID_mean_val`: last 0.0387, best 0.0191 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0343, best 0.0164 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.2031, best 0.1885 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.2019, best 0.1863 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6550 (min 0.4896, max 1.8823)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9583 (min 0.2821, max 0.9596)
  - `Validation/MS-SSIM_bravo`: last 0.9532 (min 0.3052, max 0.9662)
  - `Validation/PSNR_bravo`: last 32.8757 (min 10.3657, max 34.1096)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6714
  - `Generation_Diversity/extended_MSSSIM`: 0.2544

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005759
  - `regional_bravo/large`: 0.0106
  - `regional_bravo/medium`: 0.0097
  - `regional_bravo/small`: 0.0143
  - `regional_bravo/tiny`: 0.0097
  - `regional_bravo/tumor_bg_ratio`: 18.7821
  - `regional_bravo/tumor_loss`: 0.0108

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0407, max 3.7748 @ ep 0
  - `training/grad_norm_max`: last 0.1909, max 9.9913 @ ep 80

### exp6

**exp6 (ControlNet)** — seg-conditioned bravo generation via ControlNet.
exp6a = stage-1 (full-image conditioning), exp6b = stage-2 (patchified).
Per memory: exp6b at 128×128 reached FID 49.62; at 256×256 FID 76–83
(worse at high res); exp6a_1 at 256×256 had lowest KID (0.026) and
CMMD (0.175) but 72 gradient spikes.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp6b_pixel_bravo_controlnet_stage2_20260312-210758` — 0.0143
  2. 🥈 `exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305` — 0.0212
  3.  `exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034` — 0.0279

#### `exp6a_pixel_bravo_controlnet_stage1_rflow_128x160_20260130-033231`
*started 2026-01-30 03:32 • 500 epochs • 10h40m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9689 → 0.0019 (min 0.0014 @ ep 380)
  - `Loss/MSE_val`: 0.9262 → 0.0059 (min 0.0023 @ ep 107)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0280
  - 0.1-0.2: 0.0216
  - 0.2-0.3: 0.0098
  - 0.3-0.4: 0.0084
  - 0.4-0.5: 0.0031
  - 0.5-0.6: 0.0031
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0022
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0014

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6122 (min 0.4900, max 1.8823)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9226 (min 0.2874, max 0.9567)
  - `Validation/MS-SSIM_bravo`: last 0.9246 (min 0.3096, max 0.9640)
  - `Validation/PSNR_bravo`: last 30.9877 (min 10.3398, max 33.7996)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000856
  - `regional_bravo/large`: 0.0094
  - `regional_bravo/medium`: 0.0218
  - `regional_bravo/small`: 0.0139
  - `regional_bravo/tiny`: 0.0073
  - `regional_bravo/tumor_bg_ratio`: 16.7289
  - `regional_bravo/tumor_loss`: 0.0143

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0350, max 3.8716 @ ep 0
  - `training/grad_norm_max`: last 0.1617, max 19.3211 @ ep 70

#### `exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305`
*started 2026-03-11 00:33 • 500 epochs • 65h42m • 16018.2 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9684 → 0.0032 (min 0.0021 @ ep 483)
  - `Loss/MSE_val`: 0.9256 → 0.0051 (min 0.0020 @ ep 329)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0462
  - 0.1-0.2: 0.0103
  - 0.2-0.3: 0.0051
  - 0.3-0.4: 0.0035
  - 0.4-0.5: 0.0027
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0018
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0010

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0233, best 0.0166 @ ep 476
  - `Generation/KID_mean_train`: last 0.0219, best 0.0160 @ ep 468
  - `Generation/KID_std_val`: last 0.0046, best 0.0025 @ ep 360
  - `Generation/KID_std_train`: last 0.0057, best 0.0023 @ ep 404
  - `Generation/CMMD_val`: last 0.1878, best 0.1439 @ ep 467
  - `Generation/CMMD_train`: last 0.1747, best 0.1337 @ ep 391
  - `Generation/extended_KID_mean_val`: last 0.0250, best 0.0212 @ ep 349
  - `Generation/extended_KID_mean_train`: last 0.0215, best 0.0172 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.1412, best 0.1200 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1329, best 0.1088 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5509 (min 0.4757, max 1.7845)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9528 (min 0.2383, max 0.9604)
  - `Validation/MS-SSIM_bravo`: last 0.9494 (min 0.2858, max 0.9676)
  - `Validation/PSNR_bravo`: last 32.9947 (min 10.6112, max 34.0839)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2091
  - `Generation_Diversity/extended_MSSSIM`: 0.1011

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006189
  - `regional_bravo/large`: 0.0103
  - `regional_bravo/medium`: 0.0139
  - `regional_bravo/small`: 0.0098
  - `regional_bravo/tiny`: 0.0050
  - `regional_bravo/tumor_bg_ratio`: 16.4431
  - `regional_bravo/tumor_loss`: 0.0102

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0182, max 8.6442 @ ep 14
  - `training/grad_norm_max`: last 0.3331, max 69.6875 @ ep 14

#### `exp6b_pixel_bravo_controlnet_stage2_20260312-210758`
*started 2026-03-12 21:07 • 500 epochs • 12h00m • 1954.2 TFLOPs • peak VRAM 22.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0017 → 0.0019 (min 0.0012 @ ep 499)
  - `Loss/MSE_val`: 0.0059 → 0.0074 (min 0.0030 @ ep 72)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0359
  - 0.1-0.2: 0.0217
  - 0.2-0.3: 0.0101
  - 0.3-0.4: 0.0063
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0039
  - 0.6-0.7: 0.0030
  - 0.7-0.8: 0.0024
  - 0.8-0.9: 0.0012
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0301, best 0.0115 @ ep 120
  - `Generation/KID_mean_train`: last 0.0272, best 0.0112 @ ep 120
  - `Generation/KID_std_val`: last 0.0030, best 0.0017 @ ep 265
  - `Generation/KID_std_train`: last 0.0025, best 0.0016 @ ep 384
  - `Generation/CMMD_val`: last 0.1792, best 0.1309 @ ep 226
  - `Generation/CMMD_train`: last 0.1658, best 0.1224 @ ep 114
  - `Generation/extended_KID_mean_val`: last 0.0188, best 0.0143 @ ep 74
  - `Generation/extended_KID_mean_train`: last 0.0163, best 0.0129 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.1387, best 0.1097 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.1283, best 0.1033 @ ep 24

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5094 (min 0.4917, max 0.6839)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9220 (min 0.9216, max 0.9236)
  - `Validation/MS-SSIM_bravo`: last 0.9321 (min 0.9121, max 0.9367)
  - `Validation/PSNR_bravo`: last 31.3250 (min 30.3822, max 31.8760)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2077
  - `Generation_Diversity/extended_MSSSIM`: 0.0650

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008134
  - `regional_bravo/large`: 0.0100
  - `regional_bravo/medium`: 0.0186
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0066
  - `regional_bravo/tumor_bg_ratio`: 15.9995
  - `regional_bravo/tumor_loss`: 0.0130

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0, max 0 @ ep 0
  - `training/grad_norm_max`: last 0, max 0 @ ep 0

#### `exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034`
*started 2026-03-13 21:10 • 500 epochs • 48h23m • 16018.2 TFLOPs • peak VRAM 50.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0024 → 0.0034 (min 0.0021 @ ep 51)
  - `Loss/MSE_val`: 0.0040 → 0.0031 (min 0.0020 @ ep 173)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0139
  - 0.1-0.2: 0.0086
  - 0.2-0.3: 0.0051
  - 0.3-0.4: 0.0035
  - 0.4-0.5: 0.0024
  - 0.5-0.6: 0.0021
  - 0.6-0.7: 0.0015
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0524, best 0.0195 @ ep 248
  - `Generation/KID_mean_train`: last 0.0506, best 0.0185 @ ep 248
  - `Generation/KID_std_val`: last 0.0031, best 0.0022 @ ep 332
  - `Generation/KID_std_train`: last 0.0031, best 0.0020 @ ep 153
  - `Generation/CMMD_val`: last 0.1855, best 0.1520 @ ep 42
  - `Generation/CMMD_train`: last 0.1737, best 0.1409 @ ep 106
  - `Generation/extended_KID_mean_val`: last 0.0394, best 0.0279 @ ep 224
  - `Generation/extended_KID_mean_train`: last 0.0326, best 0.0238 @ ep 224
  - `Generation/extended_CMMD_val`: last 0.1751, best 0.1516 @ ep 149
  - `Generation/extended_CMMD_train`: last 0.1623, best 0.1411 @ ep 324

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6113 (min 0.4610, max 0.7379)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9583 (min 0.9579, max 0.9587)
  - `Validation/MS-SSIM_bravo`: last 0.9501 (min 0.9344, max 0.9707)
  - `Validation/PSNR_bravo`: last 32.9076 (min 31.7981, max 34.5325)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4324
  - `Generation_Diversity/extended_MSSSIM`: 0.1246

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005927
  - `regional_bravo/large`: 0.0058
  - `regional_bravo/medium`: 0.0077
  - `regional_bravo/small`: 0.0060
  - `regional_bravo/tiny`: 0.0047
  - `regional_bravo/tumor_bg_ratio`: 10.3898
  - `regional_bravo/tumor_loss`: 0.0062

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0, max 0 @ ep 0
  - `training/grad_norm_max`: last 0, max 0 @ ep 0

### exp7

**exp7 (SiT/DiT pixel-space)** — pixel-space DiT/SiT architecture sweep
with patch sizes 4/8 at 128×128 and 256×256. Bypassed because pixel-space
DiT at 3D is VRAM-prohibitive beyond small variants.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp7_sit_b_128_patch8_2000_rflow_128x160_20260130-133403` — 0.0334
  2. 🥈 `exp7_3_sit_s_128_patch4_20260216-020438` — 0.0542
  3.  `exp7_2_sit_b_256_patch8_2000_20260218-165041` — 0.0560
  4.  `exp7_sit_b_128_patch8_rflow_128x160_20260130-145402` — 0.0676
  5.  `exp7_2_sit_l_256_patch8_20260220-131107` — 0.0894
  6.  `exp7_2_sit_xl_256_patch8_20260220-131913` — 0.0908
  7.  `exp7_2_sit_b_256_patch8_20260215-023826` — 0.1193
  8.  `exp7_2_sit_s_256_patch4_20260215-023826` — 0.1321
  9.  `exp7_2_sit_s_256_patch8_2000_20260218-165041` — 1.0372

#### `exp7_sit_b_128_patch8_2000_rflow_128x160_20260130-133403`
*started 2026-01-30 13:34 • 2000 epochs • 19h02m • 550709.0 TFLOPs • peak VRAM 8.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0045 → 0.0031 (min 0.0023 @ ep 1997)
  - `Loss/MSE_val`: 1.0030 → 0.0032 (min 0.0023 @ ep 1681)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0262
  - 0.1-0.2: 0.0097
  - 0.2-0.3: 0.0083
  - 0.3-0.4: 0.0044
  - 0.4-0.5: 0.0030
  - 0.5-0.6: 0.0021
  - 0.6-0.7: 0.0023
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0009664

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0446, best 0.0216 @ ep 1709
  - `Generation/KID_mean_train`: last 0.0365, best 0.0173 @ ep 1709
  - `Generation/KID_std_val`: last 0.0050, best 0.0020 @ ep 1661
  - `Generation/KID_std_train`: last 0.0044, best 0.0017 @ ep 1702
  - `Generation/CMMD_val`: last 0.2927, best 0.1829 @ ep 1435
  - `Generation/CMMD_train`: last 0.2839, best 0.1730 @ ep 1435
  - `Generation/extended_KID_mean_val`: last 0.0381, best 0.0334 @ ep 1699
  - `Generation/extended_KID_mean_train`: last 0.0313, best 0.0280 @ ep 1699
  - `Generation/extended_CMMD_val`: last 0.2396, best 0.2157 @ ep 1799
  - `Generation/extended_CMMD_train`: last 0.2339, best 0.2085 @ ep 1799

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5477 (min 0.4634, max 1.8888)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9471 (min 0.2997, max 0.9485)
  - `Validation/MS-SSIM_bravo`: last 0.9426 (min 0.3342, max 0.9609)
  - `Validation/PSNR_bravo`: last 32.0993 (min 10.1331, max 33.3660)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3772
  - `Generation_Diversity/extended_MSSSIM`: 0.1829

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006631
  - `regional_bravo/large`: 0.0142
  - `regional_bravo/medium`: 0.0162
  - `regional_bravo/small`: 0.0207
  - `regional_bravo/tiny`: 0.0124
  - `regional_bravo/tumor_bg_ratio`: 23.8802
  - `regional_bravo/tumor_loss`: 0.0158

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0057, max 2.4123 @ ep 3
  - `training/grad_norm_max`: last 0.0104, max 11.9367 @ ep 4

#### `exp7_sit_b_128_patch8_rflow_128x160_20260130-145402`
*started 2026-01-30 14:54 • 500 epochs • 4h46m • 137677.2 TFLOPs • peak VRAM 8.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0045 → 0.0047 (min 0.0034 @ ep 326)
  - `Loss/MSE_val`: 1.0026 → 0.0044 (min 0.0029 @ ep 467)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0407
  - 0.1-0.2: 0.0142
  - 0.2-0.3: 0.0075
  - 0.3-0.4: 0.0039
  - 0.4-0.5: 0.0041
  - 0.5-0.6: 0.0023
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0021
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0700, best 0.0570 @ ep 436
  - `Generation/KID_mean_train`: last 0.0654, best 0.0505 @ ep 485
  - `Generation/KID_std_val`: last 0.0048, best 0.0039 @ ep 426
  - `Generation/KID_std_train`: last 0.0055, best 0.0034 @ ep 195
  - `Generation/CMMD_val`: last 0.3456, best 0.3176 @ ep 399
  - `Generation/CMMD_train`: last 0.3405, best 0.3132 @ ep 399
  - `Generation/extended_KID_mean_val`: last 0.0739, best 0.0676 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0682, best 0.0613 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.4000, best 0.3681 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.4001, best 0.3688 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7170 (min 0.6066, max 1.8893)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9328 (min 0.2998, max 0.9334)
  - `Validation/MS-SSIM_bravo`: last 0.9262 (min 0.3505, max 0.9442)
  - `Validation/PSNR_bravo`: last 31.3857 (min 10.6319, max 32.6888)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3470
  - `Generation_Diversity/extended_MSSSIM`: 0.1387

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007621
  - `regional_bravo/large`: 0.0142
  - `regional_bravo/medium`: 0.0121
  - `regional_bravo/small`: 0.0165
  - `regional_bravo/tiny`: 0.0104
  - `regional_bravo/tumor_bg_ratio`: 17.4078
  - `regional_bravo/tumor_loss`: 0.0133

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0064, max 2.7415 @ ep 2
  - `training/grad_norm_max`: last 0.0219, max 15.0385 @ ep 4

#### `exp7_2_sit_b_256_patch8_20260215-023826`
*started 2026-02-15 02:38 • 500 epochs • 16h04m • 550666.9 TFLOPs • peak VRAM 16.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0046 → 0.0043 (min 0.0030 @ ep 408)
  - `Loss/MSE_val`: 1.0029 → 0.0044 (min 0.0026 @ ep 457)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0320
  - 0.1-0.2: 0.0318
  - 0.2-0.3: 0.0046
  - 0.3-0.4: 0.0038
  - 0.4-0.5: 0.0039
  - 0.5-0.6: 0.0027
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0015

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0979, best 0.0809 @ ep 418
  - `Generation/KID_mean_train`: last 0.0890, best 0.0714 @ ep 418
  - `Generation/KID_std_val`: last 0.0076, best 0.0056 @ ep 291
  - `Generation/KID_std_train`: last 0.0069, best 0.0050 @ ep 397
  - `Generation/CMMD_val`: last 0.3072, best 0.2853 @ ep 468
  - `Generation/CMMD_train`: last 0.2951, best 0.2690 @ ep 468
  - `Generation/extended_KID_mean_val`: last 0.1193, best 0.1193 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.1112, best 0.1112 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.3367, best 0.3347 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.3267, best 0.3244 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4485 (min 0.4063, max 1.7821)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9432 (min 0.2390, max 0.9434)
  - `Validation/MS-SSIM_bravo`: last 0.9429 (min 0.3119, max 0.9529)
  - `Validation/PSNR_bravo`: last 32.0182 (min 11.1249, max 32.8801)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3671
  - `Generation_Diversity/extended_MSSSIM`: 0.1709

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006675
  - `regional_bravo/large`: 0.0090
  - `regional_bravo/medium`: 0.0121
  - `regional_bravo/small`: 0.0095
  - `regional_bravo/tiny`: 0.0096
  - `regional_bravo/tumor_bg_ratio`: 15.0996
  - `regional_bravo/tumor_loss`: 0.0101

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0044, max 2.6553 @ ep 3
  - `training/grad_norm_max`: last 0.0150, max 13.0409 @ ep 5

#### `exp7_2_sit_s_256_patch4_20260215-023826`
*started 2026-02-15 02:38 • 200 epochs • 165h44m • 439150.4 TFLOPs • peak VRAM 51.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0039 → 0.0050 (min 0.0037 @ ep 148)
  - `Loss/MSE_val`: 1.0010 → 0.0056 (min 0.0034 @ ep 182)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0337
  - 0.1-0.2: 0.0156
  - 0.2-0.3: 0.0076
  - 0.3-0.4: 0.0054
  - 0.4-0.5: 0.0033
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0022
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1529, best 0.1306 @ ep 178
  - `Generation/KID_mean_train`: last 0.1490, best 0.1268 @ ep 178
  - `Generation/KID_std_val`: last 0.0093, best 0.0065 @ ep 172
  - `Generation/KID_std_train`: last 0.0080, best 0.0067 @ ep 172
  - `Generation/CMMD_val`: last 0.4189, best 0.4189 @ ep 199
  - `Generation/CMMD_train`: last 0.4090, best 0.4090 @ ep 199
  - `Generation/extended_KID_mean_val`: last 0.1335, best 0.1321 @ ep 174
  - `Generation/extended_KID_mean_train`: last 0.1266, best 0.1261 @ ep 174
  - `Generation/extended_CMMD_val`: last 0.4364, best 0.4364 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.4262, best 0.4262 @ ep 199

**Validation quality:**
  - `Validation/LPIPS`: last 0.6502 (min 0.5157, max 1.5299)
  - `Validation/LPIPS_bravo`: last 1.2055 (min 0.7259, max 1.7785)
  - `Validation/MS-SSIM`: last 0.9330 (min 0.8944, max 0.9464)
  - `Validation/MS-SSIM-3D`: last 0.9369 (min 0.9130, max 0.9377)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9233 (min 0.2314, max 0.9276)
  - `Validation/MS-SSIM_bravo`: last 0.9043 (min 0.2949, max 0.9420)
  - `Validation/PSNR`: last 31.5317 (min 30.0162, max 32.2134)
  - `Validation/PSNR_bravo`: last 30.3733 (min 11.2900, max 32.0109)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5121
  - `Generation_Diversity/extended_MSSSIM`: 0.1701

**Regional loss (final):**
  - `regional/background_loss`: 0.0007765
  - `regional/large`: 0.0102
  - `regional/medium`: 0.0075
  - `regional/small`: 0.0101
  - `regional/tiny`: 0.0071
  - `regional/tumor_bg_ratio`: 11.3527
  - `regional/tumor_loss`: 0.0088
  - `regional_bravo/background_loss`: 0.00097
  - `regional_bravo/large`: 0.0100
  - `regional_bravo/medium`: 0.0106
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0102
  - `regional_bravo/tumor_bg_ratio`: 11.0647
  - `regional_bravo/tumor_loss`: 0.0107

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 6.669e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0124, max 4.2681 @ ep 2
  - `training/grad_norm_max`: last 0.0263, max 18.9261 @ ep 3

#### `exp7_3_sit_s_128_patch4_20260216-020438`
*started 2026-02-16 02:04 • 500 epochs • 36h01m • 274471.6 TFLOPs • peak VRAM 10.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0039 → 0.0034 (min 0.0029 @ ep 440)
  - `Loss/MSE_val`: 1.0004 → 0.0032 (min 0.0025 @ ep 422)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0277
  - 0.1-0.2: 0.0126
  - 0.2-0.3: 0.0089
  - 0.3-0.4: 0.0044
  - 0.4-0.5: 0.0036
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0016
  - 0.7-0.8: 0.0012
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0498, best 0.0408 @ ep 443
  - `Generation/KID_mean_train`: last 0.0459, best 0.0367 @ ep 443
  - `Generation/KID_std_val`: last 0.0057, best 0.0029 @ ep 443
  - `Generation/KID_std_train`: last 0.0061, best 0.0024 @ ep 436
  - `Generation/CMMD_val`: last 0.3110, best 0.2495 @ ep 453
  - `Generation/CMMD_train`: last 0.3070, best 0.2458 @ ep 453
  - `Generation/extended_KID_mean_val`: last 0.0692, best 0.0542 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0637, best 0.0490 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.3230, best 0.3008 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.3230, best 0.3005 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6149 (min 0.5135, max 1.8792)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9458 (min 0.2858, max 0.9463)
  - `Validation/MS-SSIM_bravo`: last 0.9409 (min 0.3101, max 0.9574)
  - `Validation/PSNR_bravo`: last 31.9709 (min 9.9719, max 32.9859)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3140
  - `Generation_Diversity/extended_MSSSIM`: 0.1164

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006582
  - `regional_bravo/large`: 0.0093
  - `regional_bravo/medium`: 0.0118
  - `regional_bravo/small`: 0.0140
  - `regional_bravo/tiny`: 0.0109
  - `regional_bravo/tumor_bg_ratio`: 17.2724
  - `regional_bravo/tumor_loss`: 0.0114

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0075, max 4.7931 @ ep 2
  - `training/grad_norm_max`: last 0.0302, max 17.7778 @ ep 2

#### `exp7_2_sit_b_256_patch8_2000_20260218-165041`
*started 2026-02-18 16:50 • 2000 epochs • 91h07m • 2202667.8 TFLOPs • peak VRAM 15.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0043 → 0.0026 (min 0.0019 @ ep 1897)
  - `Loss/MSE_val`: 1.0022 → 0.0040 (min 0.0022 @ ep 801)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0217
  - 0.1-0.2: 0.0099
  - 0.2-0.3: 0.0071
  - 0.3-0.4: 0.0046
  - 0.4-0.5: 0.0024
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0018
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0473, best 0.0370 @ ep 1572
  - `Generation/KID_mean_train`: last 0.0363, best 0.0284 @ ep 1583
  - `Generation/KID_std_val`: last 0.0064, best 0.0032 @ ep 1510
  - `Generation/KID_std_train`: last 0.0052, best 0.0027 @ ep 1450
  - `Generation/CMMD_val`: last 0.2471, best 0.2061 @ ep 1347
  - `Generation/CMMD_train`: last 0.2369, best 0.1995 @ ep 1347
  - `Generation/extended_KID_mean_val`: last 0.0731, best 0.0560 @ ep 1099
  - `Generation/extended_KID_mean_train`: last 0.0641, best 0.0460 @ ep 1099
  - `Generation/extended_CMMD_val`: last 0.2297, best 0.2203 @ ep 1899
  - `Generation/extended_CMMD_train`: last 0.2191, best 0.2088 @ ep 1899

**Validation quality:**
  - `Validation/LPIPS`: last 0.4081 (min 0.3590, max 1.2436)
  - `Validation/LPIPS_bravo`: last 0.4154 (min 0.3530, max 1.7932)
  - `Validation/MS-SSIM`: last 0.9499 (min 0.9258, max 0.9658)
  - `Validation/MS-SSIM-3D`: last 0.9487 (min 0.9434, max 0.9542)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9484 (min 0.2396, max 0.9501)
  - `Validation/MS-SSIM_bravo`: last 0.9446 (min 0.2407, max 0.9612)
  - `Validation/PSNR`: last 32.3429 (min 31.1816, max 33.5762)
  - `Validation/PSNR_bravo`: last 32.1228 (min 9.4086, max 33.3250)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3764
  - `Generation_Diversity/extended_MSSSIM`: 0.1765

**Regional loss (final):**
  - `regional/background_loss`: 0.0006517
  - `regional/large`: 0.0126
  - `regional/medium`: 0.0161
  - `regional/small`: 0.0152
  - `regional/tiny`: 0.0101
  - `regional/tumor_bg_ratio`: 20.8490
  - `regional/tumor_loss`: 0.0136
  - `regional_bravo/background_loss`: 0.0006853
  - `regional_bravo/large`: 0.0104
  - `regional_bravo/medium`: 0.0127
  - `regional_bravo/small`: 0.0165
  - `regional_bravo/tiny`: 0.0086
  - `regional_bravo/tumor_bg_ratio`: 17.2587
  - `regional_bravo/tumor_loss`: 0.0118

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0034, max 2.8937 @ ep 4
  - `training/grad_norm_max`: last 0.0084, max 14.1120 @ ep 5

#### `exp7_2_sit_s_256_patch8_2000_20260218-165041`
*started 2026-02-18 16:50 • 391 epochs • 9h05m • 107911.0 TFLOPs • peak VRAM 7.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0052 → 0.2703 (min 0.2701 @ ep 376)
  - `Loss/MSE_val`: 1.0039 → 0.2706 (min 0.2700 @ ep 357)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.3098
  - 0.1-0.2: 0.2747
  - 0.2-0.3: 0.2716
  - 0.3-0.4: 0.2702
  - 0.4-0.5: 0.2696
  - 0.5-0.6: 0.2693
  - 0.6-0.7: 0.2701
  - 0.7-0.8: 0.2699
  - 0.8-0.9: 0.2693
  - 0.9-1.0: 0.2704

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.0696, best 0.9686 @ ep 25
  - `Generation/KID_mean_train`: last 1.0656, best 0.9659 @ ep 35
  - `Generation/KID_std_val`: last 0.0174, best 0.0139 @ ep 352
  - `Generation/KID_std_train`: last 0.0195, best 0.0133 @ ep 143
  - `Generation/CMMD_val`: last 0.8038, best 0.8018 @ ep 309
  - `Generation/CMMD_train`: last 0.8076, best 0.8059 @ ep 309
  - `Generation/extended_KID_mean_val`: last 1.0578, best 1.0372 @ ep 99
  - `Generation/extended_KID_mean_train`: last 1.0587, best 1.0370 @ ep 199
  - `Generation/extended_CMMD_val`: last 0.8003, best 0.7982 @ ep 99
  - `Generation/extended_CMMD_train`: last 0.8064, best 0.8039 @ ep 99

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.7853 (min 1.7437, max 1.7937)
  - `Validation/MS-SSIM-3D_bravo`: last 0.3338 (min 0.2318, max 0.3647)
  - `Validation/MS-SSIM_bravo`: last 0.4115 (min 0.2603, max 0.4942)
  - `Validation/PSNR_bravo`: last 16.2141 (min 10.0211, max 17.6891)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0642
  - `Generation_Diversity/extended_MSSSIM`: 0.9444

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0325
  - `regional_bravo/large`: 0.0578
  - `regional_bravo/medium`: 0.0754
  - `regional_bravo/small`: 0.0468
  - `regional_bravo/tiny`: 0.0252
  - `regional_bravo/tumor_bg_ratio`: 1.6633
  - `regional_bravo/tumor_loss`: 0.0541

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 9.118e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0227, max 1.7273 @ ep 3
  - `training/grad_norm_max`: last 0.1218, max 12.3961 @ ep 5

#### `exp7_2_sit_l_256_patch8_20260220-131107`
*started 2026-02-20 13:11 • 500 epochs • 42h21m • 1952113.6 TFLOPs • peak VRAM 31.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0040 → 0.0038 (min 0.0029 @ ep 419)
  - `Loss/MSE_val`: 1.0021 → 0.0035 (min 0.0027 @ ep 461)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0418
  - 0.1-0.2: 0.0185
  - 0.2-0.3: 0.0091
  - 0.3-0.4: 0.0040
  - 0.4-0.5: 0.0032
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0940, best 0.0714 @ ep 458
  - `Generation/KID_mean_train`: last 0.0823, best 0.0632 @ ep 458
  - `Generation/KID_std_val`: last 0.0116, best 0.0053 @ ep 293
  - `Generation/KID_std_train`: last 0.0102, best 0.0037 @ ep 372
  - `Generation/CMMD_val`: last 0.3225, best 0.2896 @ ep 484
  - `Generation/CMMD_train`: last 0.3093, best 0.2788 @ ep 484
  - `Generation/extended_KID_mean_val`: last 0.0894, best 0.0894 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0806, best 0.0797 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.3543, best 0.3425 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.3450, best 0.3338 @ ep 474

**Validation quality:**
  - `Validation/LPIPS`: last 0.4309 (min 0.4208, max 1.7900)
  - `Validation/LPIPS_bravo`: last 0.4692 (min 0.3760, max 0.5243)
  - `Validation/MS-SSIM`: last 0.9504 (min 0.2662, max 0.9534)
  - `Validation/MS-SSIM-3D`: last 0.9413 (min 0.2440, max 0.9415)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9426 (min 0.9406, max 0.9428)
  - `Validation/MS-SSIM_bravo`: last 0.9389 (min 0.9290, max 0.9570)
  - `Validation/PSNR`: last 32.4457 (min 10.2618, max 32.6358)
  - `Validation/PSNR_bravo`: last 31.8009 (min 31.3037, max 33.0483)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2226
  - `Generation_Diversity/extended_MSSSIM`: 0.1253

**Regional loss (final):**
  - `regional/background_loss`: 0.0006144
  - `regional/large`: 0.0062
  - `regional/medium`: 0.0057
  - `regional/small`: 0.0090
  - `regional/tiny`: 0.0070
  - `regional/tumor_bg_ratio`: 11.0270
  - `regional/tumor_loss`: 0.0068
  - `regional_bravo/background_loss`: 0.000703
  - `regional_bravo/large`: 0.0096
  - `regional_bravo/medium`: 0.0103
  - `regional_bravo/small`: 0.0102
  - `regional_bravo/tiny`: 0.0071
  - `regional_bravo/tumor_bg_ratio`: 13.3669
  - `regional_bravo/tumor_loss`: 0.0094

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0042, max 1.9606 @ ep 3
  - `training/grad_norm_max`: last 0.0120, max 9.8701 @ ep 4

#### `exp7_2_sit_xl_256_patch8_20260220-131913`
*started 2026-02-20 13:19 • 500 epochs • 64h31m • 2881147.0 TFLOPs • peak VRAM 41.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0038 → 0.0039 (min 0.0029 @ ep 477)
  - `Loss/MSE_val`: 1.0012 → 0.0039 (min 0.0026 @ ep 497)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0210
  - 0.1-0.2: 0.0120
  - 0.2-0.3: 0.0065
  - 0.3-0.4: 0.0047
  - 0.4-0.5: 0.0036
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0017
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0941, best 0.0773 @ ep 465
  - `Generation/KID_mean_train`: last 0.0860, best 0.0675 @ ep 465
  - `Generation/KID_std_val`: last 0.0093, best 0.0049 @ ep 321
  - `Generation/KID_std_train`: last 0.0090, best 0.0051 @ ep 372
  - `Generation/CMMD_val`: last 0.3224, best 0.2856 @ ep 455
  - `Generation/CMMD_train`: last 0.3101, best 0.2706 @ ep 455
  - `Generation/extended_KID_mean_val`: last 0.0921, best 0.0908 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0836, best 0.0810 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.3437, best 0.3279 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.3348, best 0.3185 @ ep 424

**Validation quality:**
  - `Validation/LPIPS`: last 0.5733 (min 0.4827, max 1.7914)
  - `Validation/LPIPS_bravo`: last 0.4457 (min 0.3992, max 0.9886)
  - `Validation/MS-SSIM`: last 0.9339 (min 0.2562, max 0.9479)
  - `Validation/MS-SSIM-3D`: last 0.9377 (min 0.2414, max 0.9391)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9432 (min 0.9361, max 0.9436)
  - `Validation/MS-SSIM_bravo`: last 0.9463 (min 0.9217, max 0.9541)
  - `Validation/PSNR`: last 31.7499 (min 9.6904, max 32.5379)
  - `Validation/PSNR_bravo`: last 32.0091 (min 31.0597, max 32.6721)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2997
  - `Generation_Diversity/extended_MSSSIM`: 0.1685

**Regional loss (final):**
  - `regional/background_loss`: 0.0007295
  - `regional/large`: 0.0122
  - `regional/medium`: 0.0119
  - `regional/small`: 0.0126
  - `regional/tiny`: 0.0082
  - `regional/tumor_bg_ratio`: 15.6312
  - `regional/tumor_loss`: 0.0114
  - `regional_bravo/background_loss`: 0.0006541
  - `regional_bravo/large`: 0.0074
  - `regional_bravo/medium`: 0.0091
  - `regional_bravo/small`: 0.0105
  - `regional_bravo/tiny`: 0.0088
  - `regional_bravo/tumor_bg_ratio`: 13.3799
  - `regional_bravo/tumor_loss`: 0.0088

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0035, max 1.9751 @ ep 3
  - `training/grad_norm_max`: last 0.0110, max 7.8850 @ ep 3

### exp8

**exp8** — baseline UNet with EMA at 128×128×160, early pre-256 test.

#### `exp8_pixel_bravo_ema_rflow_128x160_20260131-061549`
*started 2026-01-31 06:15 • 500 epochs • 11h03m • 1954.2 TFLOPs • peak VRAM 22.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9682 → 0.0013 (min 0.0013 @ ep 499)
  - `Loss/MSE_val`: 0.9269 → 0.0043 (min 0.0023 @ ep 296)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0380
  - 0.1-0.2: 0.0133
  - 0.2-0.3: 0.0072
  - 0.3-0.4: 0.0048
  - 0.4-0.5: 0.0048
  - 0.5-0.6: 0.0027
  - 0.6-0.7: 0.0024
  - 0.7-0.8: 0.0018
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0415, best 0.0158 @ ep 487
  - `Generation/KID_mean_train`: last 0.0404, best 0.0161 @ ep 487
  - `Generation/KID_std_val`: last 0.0024, best 0.0016 @ ep 446
  - `Generation/KID_std_train`: last 0.0022, best 0.0016 @ ep 460
  - `Generation/CMMD_val`: last 0.2428, best 0.1948 @ ep 291
  - `Generation/CMMD_train`: last 0.2428, best 0.1923 @ ep 291
  - `Generation/extended_KID_mean_val`: last 0.0245, best 0.0245 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0210, best 0.0210 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.2075, best 0.1938 @ ep 299
  - `Generation/extended_CMMD_train`: last 0.2081, best 0.1925 @ ep 299

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5589 (min 0.5374, max 1.8879)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9311 (min 0.2852, max 0.9555)
  - `Validation/MS-SSIM_bravo`: last 0.9326 (min 0.3207, max 0.9627)
  - `Validation/PSNR_bravo`: last 31.2303 (min 10.5781, max 33.8047)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5502
  - `Generation_Diversity/extended_MSSSIM`: 0.1028

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008237
  - `regional_bravo/large`: 0.0080
  - `regional_bravo/medium`: 0.0110
  - `regional_bravo/small`: 0.0171
  - `regional_bravo/tiny`: 0.0092
  - `regional_bravo/tumor_bg_ratio`: 13.3759
  - `regional_bravo/tumor_loss`: 0.0110

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0433, max 3.8774 @ ep 0
  - `training/grad_norm_max`: last 0.1947, max 8.5823 @ ep 81

### exp12

**exp12** — 128×128 baseline ablations.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp12b_2_20260222-235610` — 0.1083
  2. 🥈 `exp12_2_20260222-233635` — 0.2503

#### `exp12_2_20260222-233635`
*started 2026-02-22 23:36 • 500 epochs • 13h49m • 50.2 TFLOPs • peak VRAM 27.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9055 → 0.0025 (min 0.0023 @ ep 455)
  - `Loss/MSE_val`: 0.9037 → 0.0037 (min 0.0032 @ ep 465)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.8932
  - 0.1-0.2: 1.8623
  - 0.2-0.3: 1.8589
  - 0.3-0.4: 1.8676
  - 0.4-0.5: 1.8695
  - 0.5-0.6: 1.8687
  - 0.6-0.7: 1.8763
  - 0.7-0.8: 1.8721
  - 0.8-0.9: 1.8623
  - 0.9-1.0: 1.8720

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2572, best 0.2065 @ ep 286
  - `Generation/KID_mean_train`: last 0.2523, best 0.2050 @ ep 286
  - `Generation/KID_std_val`: last 0.0115, best 0.0059 @ ep 97
  - `Generation/KID_std_train`: last 0.0152, best 0.0057 @ ep 47
  - `Generation/CMMD_val`: last 0.6126, best 0.5200 @ ep 364
  - `Generation/CMMD_train`: last 0.6121, best 0.5194 @ ep 364
  - `Generation/extended_KID_mean_val`: last 0.2812, best 0.2503 @ ep 324
  - `Generation/extended_KID_mean_train`: last 0.2715, best 0.2493 @ ep 324
  - `Generation/extended_CMMD_val`: last 0.5507, best 0.5372 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.5492, best 0.5366 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.2216 (min 1.1943, max 1.8211)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9168 (min 0.2880, max 0.9255)
  - `Validation/MS-SSIM_bravo`: last 0.9204 (min 0.2880, max 0.9445)
  - `Validation/PSNR_bravo`: last 30.8738 (min 6.4611, max 31.4164)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3984
  - `Generation_Diversity/extended_MSSSIM`: 0.0788

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0083
  - `regional_bravo/medium`: 0.0098
  - `regional_bravo/small`: 0.0095
  - `regional_bravo/tiny`: 0.0110
  - `regional_bravo/tumor_bg_ratio`: 8.8643
  - `regional_bravo/tumor_loss`: 0.0095

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0600, max 9.4024 @ ep 4
  - `training/grad_norm_max`: last 0.1172, max 9.8108 @ ep 4

#### `exp12b_2_20260222-235610`
*started 2026-02-22 23:56 • 500 epochs • 15h40m • 50.2 TFLOPs • peak VRAM 27.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9986 → 0.4351 (min 0.3427 @ ep 490)
  - `Loss/MSE_val`: 1.0367 → 0.6533 (min 0.3058 @ ep 378)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.7625
  - 0.1-0.2: 1.7150
  - 0.2-0.3: 1.5574
  - 0.3-0.4: 1.4095
  - 0.4-0.5: 1.3126
  - 0.5-0.6: 1.2793
  - 0.6-0.7: 1.1179
  - 0.7-0.8: 1.0913
  - 0.8-0.9: 1.1094
  - 0.9-1.0: 1.0840

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1532, best 0.1088 @ ep 37
  - `Generation/KID_mean_train`: last 0.1488, best 0.1021 @ ep 36
  - `Generation/KID_std_val`: last 0.0078, best 0.0059 @ ep 134
  - `Generation/KID_std_train`: last 0.0102, best 0.0055 @ ep 77
  - `Generation/CMMD_val`: last 0.5971, best 0.4576 @ ep 191
  - `Generation/CMMD_train`: last 0.6001, best 0.4602 @ ep 191
  - `Generation/extended_KID_mean_val`: last 0.1552, best 0.1083 @ ep 24
  - `Generation/extended_KID_mean_train`: last 0.1513, best 0.1041 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.5908, best 0.5113 @ ep 99
  - `Generation/extended_CMMD_train`: last 0.5903, best 0.5115 @ ep 99

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6427 (min 0.5207, max 1.4211)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9464 (min 0.2456, max 0.9498)
  - `Validation/MS-SSIM_bravo`: last 0.8938 (min 0.2470, max 0.9437)
  - `Validation/PSNR_bravo`: last 30.9567 (min 22.5591, max 33.7526)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4905
  - `Generation_Diversity/extended_MSSSIM`: 0.3827

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009405
  - `regional_bravo/large`: 0.0078
  - `regional_bravo/medium`: 0.0157
  - `regional_bravo/small`: 0.0163
  - `regional_bravo/tiny`: 0.0104
  - `regional_bravo/tumor_bg_ratio`: 13.4831
  - `regional_bravo/tumor_loss`: 0.0127

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.4938, max 3.9707 @ ep 4
  - `training/grad_norm_max`: last 2.2772, max 5.3165 @ ep 3

### exp15

**exp15** — UViT (skip-connection ViT) pixel-space test.

#### `exp15_uvit_pixel_bravo_20260219-012308`
*started 2026-02-19 01:23 • 500 epochs • 3h46m • 71476.0 TFLOPs • peak VRAM 5.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0008 → 0.0148 (min 0.0147 @ ep 429)
  - `Loss/MSE_val`: 0.9904 → 0.0149 (min 0.0143 @ ep 330)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0384
  - 0.1-0.2: 0.0215
  - 0.2-0.3: 0.0173
  - 0.3-0.4: 0.0153
  - 0.4-0.5: 0.0147
  - 0.5-0.6: 0.0144
  - 0.6-0.7: 0.0131
  - 0.7-0.8: 0.0135
  - 0.8-0.9: 0.0136
  - 0.9-1.0: 0.0134

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.4049, best 0.3494 @ ep 141
  - `Generation/KID_mean_train`: last 0.4053, best 0.3493 @ ep 138
  - `Generation/KID_std_val`: last 0.0079, best 0.0065 @ ep 149
  - `Generation/KID_std_train`: last 0.0087, best 0.0060 @ ep 467
  - `Generation/CMMD_val`: last 0.6510, best 0.6478 @ ep 118
  - `Generation/CMMD_train`: last 0.6517, best 0.6491 @ ep 392
  - `Generation/extended_KID_mean_val`: last 0.4137, best 0.3784 @ ep 149
  - `Generation/extended_KID_mean_train`: last 0.4082, best 0.3704 @ ep 149
  - `Generation/extended_CMMD_val`: last 0.6689, best 0.6605 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.6715, best 0.6642 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.7935 (min 1.6675, max 1.8907)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7796 (min 0.2953, max 0.8167)
  - `Validation/MS-SSIM_bravo`: last 0.7682 (min 0.3446, max 0.8595)
  - `Validation/PSNR_bravo`: last 26.4272 (min 10.5740, max 28.4201)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1439
  - `Generation_Diversity/extended_MSSSIM`: 0.6009

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0159
  - `regional_bravo/medium`: 0.0189
  - `regional_bravo/small`: 0.0188
  - `regional_bravo/tiny`: 0.0112
  - `regional_bravo/tumor_bg_ratio`: 6.3802
  - `regional_bravo/tumor_loss`: 0.0167

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0083, max 0.8276 @ ep 6
  - `training/grad_norm_max`: last 0.0315, max 11.8250 @ ep 36

### exp16

**exp16** — HDiT (hierarchical DiT) pixel-space baseline.

#### `exp16_hdit_pixel_bravo_20260219-044848`
*started 2026-02-19 04:48 • 500 epochs • 12h38m • 125259.7 TFLOPs • peak VRAM 5.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0040 → 0.0030 (min 0.0017 @ ep 495)
  - `Loss/MSE_val`: 1.0003 → 0.0093 (min 0.0026 @ ep 170)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0311
  - 0.1-0.2: 0.0253
  - 0.2-0.3: 0.0121
  - 0.3-0.4: 0.0065
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0020
  - 0.6-0.7: 0.0031
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0012
  - 0.9-1.0: 0.0010

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0570, best 0.0354 @ ep 371
  - `Generation/KID_mean_train`: last 0.0498, best 0.0292 @ ep 371
  - `Generation/KID_std_val`: last 0.0057, best 0.0027 @ ep 354
  - `Generation/KID_std_train`: last 0.0044, best 0.0023 @ ep 478
  - `Generation/CMMD_val`: last 0.2902, best 0.1941 @ ep 442
  - `Generation/CMMD_train`: last 0.2830, best 0.1872 @ ep 442
  - `Generation/extended_KID_mean_val`: last 0.0390, best 0.0390 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0342, best 0.0342 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.2421, best 0.2383 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.2404, best 0.2355 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4792 (min 0.4739, max 1.8891)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9206 (min 0.2953, max 0.9437)
  - `Validation/MS-SSIM_bravo`: last 0.9312 (min 0.3335, max 0.9541)
  - `Validation/PSNR_bravo`: last 31.4294 (min 10.2785, max 32.8426)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3798
  - `Generation_Diversity/extended_MSSSIM`: 0.1818

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000792
  - `regional_bravo/large`: 0.0178
  - `regional_bravo/medium`: 0.0161
  - `regional_bravo/small`: 0.0163
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 19.7027
  - `regional_bravo/tumor_loss`: 0.0156

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0093, max 5.0013 @ ep 3
  - `training/grad_norm_max`: last 0.0391, max 22.4513 @ ep 4

### exp17

**exp17 (HDiT scaling)** — HDiT size sweep at 256×256. Variants: S/p2, B/p4,
L/p8, XL/p8, S/p4. Motivation: does hierarchical patchification help 3D
pixel-space scaling?

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp17_3_hdit_xl_p8_256_20260220-141245` — 0.1566
  2. 🥈 `exp17_2_hdit_l_p8_256_20260220-135418` — 0.1764
  3.  `exp17_1_hdit_b_p4_256_20260219-210002` — 0.1974
  4.  `exp17_4_hdit_s_p4_256_20260225-111541` — 0.3994

#### `exp17_0_hdit_s_p2_256_20260219-205831`
*started 2026-02-19 20:58 • 3 epochs • 40h21m • 8827.5 TFLOPs • peak VRAM 73.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9998 → 0.1149 (min 0.1149 @ ep 2)
  - `Loss/MSE_val`: 0.9887 → 0.1475 (min 0.1475 @ ep 1)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0020
  - 0.1-0.2: 0.5365
  - 0.2-0.3: 0.2668
  - 0.3-0.4: 0.1615
  - 0.4-0.5: 0.1255
  - 0.5-0.6: 0.1083
  - 0.6-0.7: 0.0983
  - 0.7-0.8: 0.0928
  - 0.8-0.9: 0.0904
  - 0.9-1.0: 0.9738

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.6192, best 0.6192 @ ep 1
  - `Generation/KID_mean_train`: last 0.6123, best 0.6123 @ ep 1
  - `Generation/KID_std_val`: last 0.0223, best 0.0122 @ ep 0
  - `Generation/KID_std_train`: last 0.0234, best 0.0158 @ ep 0
  - `Generation/CMMD_val`: last 0.7827, best 0.7827 @ ep 1
  - `Generation/CMMD_train`: last 0.7819, best 0.7819 @ ep 1

**Validation quality:**
  - `Validation/LPIPS`: last 1.7218 (min 1.7218, max 1.7907)
  - `Validation/MS-SSIM`: last 0.4384 (min 0.2553, max 0.4384)
  - `Validation/MS-SSIM-3D`: last 0.4520 (min 0.2344, max 0.4520)
  - `Validation/PSNR`: last 16.7340 (min 9.9302, max 16.7340)

**Regional loss (final):**
  - `regional/background_loss`: 0.0239
  - `regional/large`: 0.0519
  - `regional/medium`: 0.0415
  - `regional/small`: 0.0477
  - `regional/tiny`: 0.0406
  - `regional/tumor_bg_ratio`: 1.9243
  - `regional/tumor_loss`: 0.0459

**LR schedule:**
  - `LR/Generator`: peak 4.6e-05 @ ep 1, final 4.6e-05

**Training meta:**
  - `training/grad_norm_avg`: last 2.6052, max 2.7160 @ ep 2
  - `training/grad_norm_max`: last 11.2940, max 13.0099 @ ep 2

#### `exp17_1_hdit_b_p4_256_20260219-210002`
*started 2026-02-19 21:00 • 179 epochs • 121h47m • 712316.3 TFLOPs • peak VRAM 35.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0028 → 0.0045 (min 0.0032 @ ep 177)
  - `Loss/MSE_val`: 0.9985 → 0.0063 (min 0.0030 @ ep 169)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0351
  - 0.1-0.2: 0.0116
  - 0.2-0.3: 0.0057
  - 0.3-0.4: 0.0042
  - 0.4-0.5: 0.0033
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1107, best 0.0800 @ ep 160
  - `Generation/KID_mean_train`: last 0.1034, best 0.0745 @ ep 160
  - `Generation/KID_std_val`: last 0.0056, best 0.0046 @ ep 160
  - `Generation/KID_std_train`: last 0.0045, best 0.0045 @ ep 158
  - `Generation/CMMD_val`: last 0.2758, best 0.2737 @ ep 154
  - `Generation/CMMD_train`: last 0.2646, best 0.2646 @ ep 177
  - `Generation/extended_KID_mean_val`: last 0.1974, best 0.1974 @ ep 174
  - `Generation/extended_KID_mean_train`: last 0.1881, best 0.1881 @ ep 174
  - `Generation/extended_CMMD_val`: last 0.3835, best 0.3835 @ ep 174
  - `Generation/extended_CMMD_train`: last 0.3733, best 0.3733 @ ep 174

**Validation quality:**
  - `Validation/LPIPS`: last 1.3189 (min 1.0908, max 1.7829)
  - `Validation/LPIPS_bravo`: last 0.6857 (min 0.5529, max 1.6954)
  - `Validation/MS-SSIM`: last 0.9201 (min 0.2926, max 0.9324)
  - `Validation/MS-SSIM-3D`: last 0.9106 (min 0.2389, max 0.9198)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9436 (min 0.8124, max 0.9456)
  - `Validation/MS-SSIM_bravo`: last 0.9405 (min 0.8280, max 0.9507)
  - `Validation/PSNR`: last 31.1445 (min 10.5652, max 31.4330)
  - `Validation/PSNR_bravo`: last 32.2464 (min 27.0989, max 32.2738)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3430
  - `Generation_Diversity/extended_MSSSIM`: 0.1529

**Regional loss (final):**
  - `regional/background_loss`: 0.0009135
  - `regional/large`: 0.0131
  - `regional/medium`: 0.0090
  - `regional/small`: 0.0108
  - `regional/tiny`: 0.0102
  - `regional/tumor_bg_ratio`: 11.9752
  - `regional/tumor_loss`: 0.0109
  - `regional_bravo/background_loss`: 0.0007144
  - `regional_bravo/large`: 0.0127
  - `regional_bravo/medium`: 0.0127
  - `regional_bravo/small`: 0.0114
  - `regional_bravo/tiny`: 0.0090
  - `regional_bravo/tumor_bg_ratio`: 16.4307
  - `regional_bravo/tumor_loss`: 0.0117

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 7.304e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0159, max 4.3427 @ ep 2
  - `training/grad_norm_max`: last 0.0469, max 14.4927 @ ep 2

#### `exp17_2_hdit_l_p8_256_20260220-135418`
*started 2026-02-20 13:54 • 500 epochs • 10h32m • 447455.5 TFLOPs • peak VRAM 12.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0035 → 0.0027 (min 0.0022 @ ep 470)
  - `Loss/MSE_val`: 1.0022 → 0.0047 (min 0.0032 @ ep 461)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0284
  - 0.1-0.2: 0.0225
  - 0.2-0.3: 0.0098
  - 0.3-0.4: 0.0072
  - 0.4-0.5: 0.0039
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0026
  - 0.7-0.8: 0.0022
  - 0.8-0.9: 0.0014
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0913, best 0.0819 @ ep 478
  - `Generation/KID_mean_train`: last 0.0762, best 0.0697 @ ep 462
  - `Generation/KID_std_val`: last 0.0124, best 0.0054 @ ep 287
  - `Generation/KID_std_train`: last 0.0079, best 0.0049 @ ep 263
  - `Generation/CMMD_val`: last 0.3034, best 0.2707 @ ep 460
  - `Generation/CMMD_train`: last 0.2855, best 0.2538 @ ep 460
  - `Generation/extended_KID_mean_val`: last 0.1841, best 0.1764 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.1815, best 0.1689 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.3548, best 0.3411 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.3403, best 0.3266 @ ep 474

**Validation quality:**
  - `Validation/LPIPS`: last 0.5783 (min 0.4460, max 1.7929)
  - `Validation/MS-SSIM`: last 0.9190 (min 0.2754, max 0.9456)
  - `Validation/MS-SSIM-3D`: last 0.9226 (min 0.2482, max 0.9350)
  - `Validation/PSNR`: last 30.7648 (min 9.9698, max 32.2527)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3434
  - `Generation_Diversity/extended_MSSSIM`: 0.1603

**Regional loss (final):**
  - `regional/background_loss`: 0.000882
  - `regional/large`: 0.0127
  - `regional/medium`: 0.0132
  - `regional/small`: 0.0162
  - `regional/tiny`: 0.0104
  - `regional/tumor_bg_ratio`: 14.7850
  - `regional/tumor_loss`: 0.0130

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0171, max 1.5998 @ ep 5
  - `training/grad_norm_max`: last 0.0466, max 13.2828 @ ep 11

#### `exp17_3_hdit_xl_p8_256_20260220-141245`
*started 2026-02-20 14:12 • 500 epochs • 20h07m • 565819.6 TFLOPs • peak VRAM 14.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0034 → 0.0020 (min 0.0014 @ ep 464)
  - `Loss/MSE_val`: 1.0022 → 0.0091 (min 0.0037 @ ep 249)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0776
  - 0.1-0.2: 0.0260
  - 0.2-0.3: 0.0119
  - 0.3-0.4: 0.0075
  - 0.4-0.5: 0.0051
  - 0.5-0.6: 0.0033
  - 0.6-0.7: 0.0034
  - 0.7-0.8: 0.0023
  - 0.8-0.9: 0.0024
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0816, best 0.0486 @ ep 492
  - `Generation/KID_mean_train`: last 0.0690, best 0.0410 @ ep 492
  - `Generation/KID_std_val`: last 0.0086, best 0.0048 @ ep 402
  - `Generation/KID_std_train`: last 0.0107, best 0.0037 @ ep 371
  - `Generation/CMMD_val`: last 0.2611, best 0.2232 @ ep 492
  - `Generation/CMMD_train`: last 0.2432, best 0.2096 @ ep 492
  - `Generation/extended_KID_mean_val`: last 0.1572, best 0.1566 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.1466, best 0.1466 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.2830, best 0.2778 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2692, best 0.2642 @ ep 474

**Validation quality:**
  - `Validation/LPIPS`: last 0.4328 (min 0.4024, max 1.7916)
  - `Validation/MS-SSIM`: last 0.9114 (min 0.2669, max 0.9455)
  - `Validation/MS-SSIM-3D`: last 0.9069 (min 0.2472, max 0.9302)
  - `Validation/PSNR`: last 30.3603 (min 9.7519, max 32.2565)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3177
  - `Generation_Diversity/extended_MSSSIM`: 0.1272

**Regional loss (final):**
  - `regional/background_loss`: 0.000981
  - `regional/large`: 0.0148
  - `regional/medium`: 0.0145
  - `regional/small`: 0.0143
  - `regional/tiny`: 0.0106
  - `regional/tumor_bg_ratio`: 14.0406
  - `regional/tumor_loss`: 0.0138

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0088, max 1.6195 @ ep 3
  - `training/grad_norm_max`: last 0.0270, max 9.4865 @ ep 4

#### `exp17_4_hdit_s_p4_256_20260225-111541`
*started 2026-02-25 11:15 • 55 epochs • 16h13m • 55112.6 TFLOPs • peak VRAM 17.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0038 → 0.0099 (min 0.0080 @ ep 46)
  - `Loss/MSE_val`: 1.0010 → 0.0061 (min 0.0059 @ ep 49)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1781
  - 0.1-0.2: 0.0446
  - 0.2-0.3: 0.0142
  - 0.3-0.4: 0.0064
  - 0.4-0.5: 0.0047
  - 0.5-0.6: 0.0048
  - 0.6-0.7: 0.0040
  - 0.7-0.8: 0.0046
  - 0.8-0.9: 0.0044
  - 0.9-1.0: 0.0051

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.3653, best 0.3055 @ ep 36
  - `Generation/KID_mean_train`: last 0.3569, best 0.3030 @ ep 36
  - `Generation/KID_std_val`: last 0.0138, best 0.0096 @ ep 30
  - `Generation/KID_std_train`: last 0.0128, best 0.0098 @ ep 35
  - `Generation/CMMD_val`: last 0.5675, best 0.5179 @ ep 46
  - `Generation/CMMD_train`: last 0.5655, best 0.5152 @ ep 46
  - `Generation/extended_KID_mean_val`: last 0.4685, best 0.3994 @ ep 24
  - `Generation/extended_KID_mean_train`: last 0.4708, best 0.3902 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.6067, best 0.5941 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.6056, best 0.5917 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.6426 (min 1.6332, max 1.7848)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8907 (min 0.2375, max 0.8907)
  - `Validation/MS-SSIM_bravo`: last 0.8551 (min 0.2841, max 0.8792)
  - `Validation/PSNR_bravo`: last 28.9509 (min 10.5223, max 29.5760)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1830
  - `Generation_Diversity/extended_MSSSIM`: 0.2703

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0018
  - `regional_bravo/large`: 0.0167
  - `regional_bravo/medium`: 0.0145
  - `regional_bravo/small`: 0.0099
  - `regional_bravo/tiny`: 0.0087
  - `regional_bravo/tumor_bg_ratio`: 7.4215
  - `regional_bravo/tumor_loss`: 0.0132

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 9.753e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.1574, max 4.6714 @ ep 3
  - `training/grad_norm_max`: last 3.0512, max 16.0016 @ ep 2

### exp18

**exp18** — UViT-L /p8 at 256×256.

#### `exp18_uvit_l_p8_256_20260220-170256`
*started 2026-02-20 17:02 • 500 epochs • 46h34m • 1843578.8 TFLOPs • peak VRAM 21.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9903 → 0.0029 (min 0.0026 @ ep 474)
  - `Loss/MSE_val`: 0.9673 → 0.0027 (min 0.0023 @ ep 391)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0206
  - 0.1-0.2: 0.0093
  - 0.2-0.3: 0.0060
  - 0.3-0.4: 0.0046
  - 0.4-0.5: 0.0028
  - 0.5-0.6: 0.0019
  - 0.6-0.7: 0.0017
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0644, best 0.0644 @ ep 499
  - `Generation/KID_mean_train`: last 0.0568, best 0.0568 @ ep 499
  - `Generation/KID_std_val`: last 0.0052, best 0.0047 @ ep 465
  - `Generation/KID_std_train`: last 0.0052, best 0.0033 @ ep 468
  - `Generation/CMMD_val`: last 0.2988, best 0.2656 @ ep 409
  - `Generation/CMMD_train`: last 0.2880, best 0.2504 @ ep 409
  - `Generation/extended_KID_mean_val`: last 0.1020, best 0.1020 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0912, best 0.0912 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.2865, best 0.2865 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.2733, best 0.2733 @ ep 499

**Validation quality:**
  - `Validation/LPIPS`: last 1.1285 (min 0.8842, max 1.7891)
  - `Validation/LPIPS_bravo`: last 0.7063 (min 0.5390, max 1.5649)
  - `Validation/MS-SSIM`: last 0.9349 (min 0.2557, max 0.9525)
  - `Validation/MS-SSIM-3D`: last 0.9453 (min 0.2402, max 0.9492)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9530 (min 0.9090, max 0.9534)
  - `Validation/MS-SSIM_bravo`: last 0.9457 (min 0.9071, max 0.9605)
  - `Validation/PSNR`: last 31.7688 (min 9.9401, max 32.6380)
  - `Validation/PSNR_bravo`: last 32.1487 (min 30.5529, max 33.2719)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4876
  - `Generation_Diversity/extended_MSSSIM`: 0.1520

**Regional loss (final):**
  - `regional/background_loss`: 0.0007353
  - `regional/large`: 0.0080
  - `regional/medium`: 0.0122
  - `regional/small`: 0.0119
  - `regional/tiny`: 0.0094
  - `regional/tumor_bg_ratio`: 13.8226
  - `regional/tumor_loss`: 0.0102
  - `regional_bravo/background_loss`: 0.0006585
  - `regional_bravo/large`: 0.0106
  - `regional_bravo/medium`: 0.0079
  - `regional_bravo/small`: 0.0098
  - `regional_bravo/tiny`: 0.0086
  - `regional_bravo/tumor_bg_ratio`: 14.1260
  - `regional_bravo/tumor_loss`: 0.0093

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0097, max 0.8562 @ ep 3
  - `training/grad_norm_max`: last 0.0189, max 18.7886 @ ep 189

### exp19

**exp19 (WDM / DiT ablations)** — wavelet-domain diffusion (WDM) +
DiT-S variant. Per memory: exp19_2 (WDM 270M DDPM x0) reached FID 67.32,
best wavelet result but still far from pixel-space baseline.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp19_2_20260224-022548` — 0.0614
  2. 🥈 `exp19_paper_20260224-015924` — 0.0659
  3.  `exp19_5_20260224-142309` — 0.0736
  4.  `exp19_6_wdm_dit_s_20260306-050100` — 0.1069
  5.  `exp19_0_20260224-015924` — 0.1072
  6.  `exp19_3_20260224-034637` — 0.1220
  7.  `exp19_4_20260224-092339` — 0.2944
  8.  `exp19_1_20260224-022448` — 0.2972

#### `exp19_0_20260224-015924`
*started 2026-02-24 01:59 • 500 epochs • 47h46m • 214.0 TFLOPs • peak VRAM 21.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9206 → 0.3228 (min 0.2081 @ ep 496)
  - `Loss/MSE_val`: 0.9184 → 0.3195 (min 0.1541 @ ep 473)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 2.2247
  - 0.1-0.2: 2.0261
  - 0.2-0.3: 1.9167
  - 0.3-0.4: 1.7412
  - 0.4-0.5: 1.7171
  - 0.5-0.6: 1.5715
  - 0.6-0.7: 1.4663
  - 0.7-0.8: 1.1535
  - 0.8-0.9: 1.1700
  - 0.9-1.0: 1.1160

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1705, best 0.0703 @ ep 167
  - `Generation/KID_mean_train`: last 0.1606, best 0.0585 @ ep 167
  - `Generation/KID_std_val`: last 0.0196, best 0.0076 @ ep 112
  - `Generation/KID_std_train`: last 0.0242, best 0.0069 @ ep 170
  - `Generation/CMMD_val`: last 0.4715, best 0.2169 @ ep 250
  - `Generation/CMMD_train`: last 0.4570, best 0.2042 @ ep 250
  - `Generation/extended_KID_mean_val`: last 0.1693, best 0.1072 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.1601, best 0.0937 @ ep 274
  - `Generation/extended_CMMD_val`: last 0.4634, best 0.3296 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.4454, best 0.3104 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2541 (min 0.1349, max 1.4339)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9853 (min 0.8539, max 0.9854)
  - `Validation/MS-SSIM_bravo`: last 0.9650 (min 0.8470, max 0.9863)
  - `Validation/PSNR_bravo`: last 37.1016 (min 28.7912, max 39.3667)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1845
  - `Generation_Diversity/extended_MSSSIM`: 0.0258

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000375
  - `regional_bravo/large`: 0.0012
  - `regional_bravo/medium`: 0.0120
  - `regional_bravo/small`: 0.0067
  - `regional_bravo/tiny`: 0.0044
  - `regional_bravo/tumor_bg_ratio`: 15.6885
  - `regional_bravo/tumor_loss`: 0.0059

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3449, max 2.2552 @ ep 0
  - `training/grad_norm_max`: last 3.0725, max 8.5169 @ ep 0

#### `exp19_paper_20260224-015924`
*started 2026-02-24 01:59 • 500 epochs • 43h16m • 214.0 TFLOPs • peak VRAM 21.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8887 → 0.0019 (min 0.0018 @ ep 495)
  - `Loss/MSE_val`: 0.8616 → 0.0050 (min 0.0021 @ ep 186)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.9121
  - 0.1-0.2: 1.9017
  - 0.2-0.3: 1.8966
  - 0.3-0.4: 1.8950
  - 0.4-0.5: 1.8933
  - 0.5-0.6: 1.9070
  - 0.6-0.7: 1.9075
  - 0.7-0.8: 1.9120
  - 0.8-0.9: 1.9103
  - 0.9-1.0: 1.9084

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0766, best 0.0501 @ ep 218
  - `Generation/KID_mean_train`: last 0.0730, best 0.0439 @ ep 218
  - `Generation/KID_std_val`: last 0.0067, best 0.0042 @ ep 267
  - `Generation/KID_std_train`: last 0.0072, best 0.0038 @ ep 297
  - `Generation/CMMD_val`: last 0.3085, best 0.2648 @ ep 380
  - `Generation/CMMD_train`: last 0.3069, best 0.2613 @ ep 380
  - `Generation/extended_KID_mean_val`: last 0.0733, best 0.0659 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0660, best 0.0622 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.3012, best 0.2830 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.2965, best 0.2774 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5664 (min 0.4346, max 1.6941)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8874 (min 0.3147, max 0.9414)
  - `Validation/MS-SSIM_bravo`: last 0.8852 (min 0.3155, max 0.9608)
  - `Validation/PSNR_bravo`: last 30.3453 (min 6.6680, max 33.2790)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4336
  - `Generation_Diversity/extended_MSSSIM`: 0.2213

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0013
  - `regional_bravo/large`: 0.0117
  - `regional_bravo/medium`: 0.0068
  - `regional_bravo/small`: 0.0101
  - `regional_bravo/tiny`: 0.0098
  - `regional_bravo/tumor_bg_ratio`: 7.7018
  - `regional_bravo/tumor_loss`: 0.0096

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0255, max 13.1903 @ ep 1
  - `training/grad_norm_max`: last 0.0630, max 14.5582 @ ep 1

#### `exp19_1_20260224-022448`
*started 2026-02-24 02:24 • 500 epochs • 47h10m • 214.0 TFLOPs • peak VRAM 21.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.8880 → 0.2089 (min 0.1921 @ ep 463)
  - `Loss/MSE_val`: 1.8629 → 0.2145 (min 0.1978 @ ep 473)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1817
  - 0.1-0.2: 0.1833
  - 0.2-0.3: 0.2035
  - 0.3-0.4: 0.2060
  - 0.4-0.5: 0.2164
  - 0.5-0.6: 0.2470
  - 0.6-0.7: 0.2908
  - 0.7-0.8: 0.3581
  - 0.8-0.9: 0.4493
  - 0.9-1.0: 0.4883

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.3265, best 0.2742 @ ep 227
  - `Generation/KID_mean_train`: last 0.3248, best 0.2606 @ ep 292
  - `Generation/KID_std_val`: last 0.0181, best 0.0140 @ ep 28
  - `Generation/KID_std_train`: last 0.0224, best 0.0144 @ ep 44
  - `Generation/CMMD_val`: last 0.6592, best 0.5750 @ ep 403
  - `Generation/CMMD_train`: last 0.6538, best 0.5666 @ ep 403
  - `Generation/extended_KID_mean_val`: last 0.3173, best 0.2972 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.3086, best 0.2870 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.5641, best 0.5605 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.5508, best 0.5472 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.0530 (min 0.0326, max 1.6224)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9954 (min 0.7787, max 0.9956)
  - `Validation/MS-SSIM_bravo`: last 0.9969 (min 0.8245, max 0.9978)
  - `Validation/PSNR_bravo`: last 45.7826 (min 31.0594, max 47.9670)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.8855
  - `Generation_Diversity/extended_MSSSIM`: 0.1685

**Regional loss (final):**
  - `regional_bravo/background_loss`: 3.887e-05
  - `regional_bravo/large`: 0.0006573
  - `regional_bravo/medium`: 0.0015
  - `regional_bravo/small`: 0.0007609
  - `regional_bravo/tiny`: 0.0007765
  - `regional_bravo/tumor_bg_ratio`: 23.7914
  - `regional_bravo/tumor_loss`: 0.000925

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.2054, max 3.3817 @ ep 12
  - `training/grad_norm_max`: last 4.7550, max 13.0924 @ ep 28

#### `exp19_2_20260224-022548`
*started 2026-02-24 02:25 • 500 epochs • 29h33m • 15856.5 TFLOPs • peak VRAM 23.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9508 → 0.3096 (min 0.2678 @ ep 427)
  - `Loss/MSE_val`: 0.9715 → 0.3492 (min 0.1784 @ ep 486)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 2.0668
  - 0.1-0.2: 1.8733
  - 0.2-0.3: 1.5083
  - 0.3-0.4: 1.6410
  - 0.4-0.5: 1.6774
  - 0.5-0.6: 1.4603
  - 0.6-0.7: 1.5552
  - 0.7-0.8: 1.3514
  - 0.8-0.9: 1.1474
  - 0.9-1.0: 1.1697

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0514, best 0.0409 @ ep 393
  - `Generation/KID_mean_train`: last 0.0413, best 0.0333 @ ep 393
  - `Generation/KID_std_val`: last 0.0067, best 0.0049 @ ep 482
  - `Generation/KID_std_train`: last 0.0061, best 0.0047 @ ep 376
  - `Generation/CMMD_val`: last 0.2201, best 0.1886 @ ep 422
  - `Generation/CMMD_train`: last 0.2072, best 0.1791 @ ep 422
  - `Generation/extended_KID_mean_val`: last 0.0614, best 0.0614 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0542, best 0.0516 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.2180, best 0.2180 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.2020, best 0.2020 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2315 (min 0.1464, max 1.4570)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9800 (min 0.7581, max 0.9804)
  - `Validation/MS-SSIM_bravo`: last 0.9627 (min 0.7464, max 0.9810)
  - `Validation/PSNR_bravo`: last 36.3129 (min 26.8061, max 37.8301)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5714
  - `Generation_Diversity/extended_MSSSIM`: 0.1800

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0004189
  - `regional_bravo/large`: 0.0050
  - `regional_bravo/medium`: 0.0037
  - `regional_bravo/small`: 0.0036
  - `regional_bravo/tiny`: 0.0020
  - `regional_bravo/tumor_bg_ratio`: 9.0122
  - `regional_bravo/tumor_loss`: 0.0038

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3552, max 2.4650 @ ep 0
  - `training/grad_norm_max`: last 2.4108, max 7.3752 @ ep 57

#### `exp19_3_20260224-034637`
*started 2026-02-24 03:46 • 500 epochs • 45h50m • 17925.3 TFLOPs • peak VRAM 67.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9221 → 0.3159 (min 0.2373 @ ep 205)
  - `Loss/MSE_val`: 0.9360 → 0.2470 (min 0.1762 @ ep 115)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 2.1265
  - 0.1-0.2: 2.2860
  - 0.2-0.3: 1.8246
  - 0.3-0.4: 1.8224
  - 0.4-0.5: 1.6153
  - 0.5-0.6: 1.5862
  - 0.6-0.7: 1.5396
  - 0.7-0.8: 1.4307
  - 0.8-0.9: 1.2038
  - 0.9-1.0: 1.1990

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1415, best 0.0998 @ ep 141
  - `Generation/KID_mean_train`: last 0.1298, best 0.0857 @ ep 141
  - `Generation/KID_std_val`: last 0.0159, best 0.0095 @ ep 82
  - `Generation/KID_std_train`: last 0.0141, best 0.0084 @ ep 141
  - `Generation/CMMD_val`: last 0.4275, best 0.2866 @ ep 141
  - `Generation/CMMD_train`: last 0.4124, best 0.2778 @ ep 141
  - `Generation/extended_KID_mean_val`: last 0.1402, best 0.1220 @ ep 99
  - `Generation/extended_KID_mean_train`: last 0.1312, best 0.1128 @ ep 99
  - `Generation/extended_CMMD_val`: last 0.3904, best 0.3904 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.3719, best 0.3719 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.1656 (min 0.1520, max 1.4368)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9845 (min 0.8332, max 0.9846)
  - `Validation/MS-SSIM_bravo`: last 0.9804 (min 0.8013, max 0.9830)
  - `Validation/PSNR_bravo`: last 38.1391 (min 27.9573, max 38.6658)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4283
  - `Generation_Diversity/extended_MSSSIM`: 0.0782

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0002347
  - `regional_bravo/large`: 0.0015
  - `regional_bravo/medium`: 0.0013
  - `regional_bravo/small`: 0.0032
  - `regional_bravo/tiny`: 0.0051
  - `regional_bravo/tumor_bg_ratio`: 10.5464
  - `regional_bravo/tumor_loss`: 0.0025

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3137, max 2.2429 @ ep 0
  - `training/grad_norm_max`: last 1.7902, max 7.8411 @ ep 87

#### `exp19_4_20260224-092339`
*started 2026-02-24 09:23 • 372 epochs • 35h44m • 159.2 TFLOPs • peak VRAM 21.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8824 → 0.0465 (min 0.0321 @ ep 288)
  - `Loss/MSE_val`: 0.8405 → 0.0353 (min 0.0184 @ ep 261)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0298
  - 0.1-0.2: 1.0654
  - 0.2-0.3: 1.0997
  - 0.3-0.4: 1.1400
  - 0.4-0.5: 1.1773
  - 0.5-0.6: 1.1940
  - 0.6-0.7: 1.1992
  - 0.7-0.8: 1.3737
  - 0.8-0.9: 1.3920
  - 0.9-1.0: 1.7812

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.3223, best 0.2522 @ ep 106
  - `Generation/KID_mean_train`: last 0.3075, best 0.2420 @ ep 106
  - `Generation/KID_std_val`: last 0.0182, best 0.0124 @ ep 13
  - `Generation/KID_std_train`: last 0.0237, best 0.0127 @ ep 17
  - `Generation/CMMD_val`: last 0.6424, best 0.5400 @ ep 72
  - `Generation/CMMD_train`: last 0.6367, best 0.5321 @ ep 72
  - `Generation/extended_KID_mean_val`: last 0.3218, best 0.2944 @ ep 199
  - `Generation/extended_KID_mean_train`: last 0.3038, best 0.2826 @ ep 199
  - `Generation/extended_CMMD_val`: last 0.6467, best 0.5979 @ ep 224
  - `Generation/extended_CMMD_train`: last 0.6362, best 0.5863 @ ep 224

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.0302 (min 0.0194, max 1.4122)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9947 (min 0.8181, max 0.9952)
  - `Validation/MS-SSIM_bravo`: last 0.9962 (min 0.8159, max 0.9973)
  - `Validation/PSNR_bravo`: last 44.3829 (min 28.2367, max 45.7986)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6048
  - `Generation_Diversity/extended_MSSSIM`: 0.2217

**Regional loss (final):**
  - `regional_bravo/background_loss`: 4.605e-05
  - `regional_bravo/large`: 0.0006519
  - `regional_bravo/medium`: 0.000509
  - `regional_bravo/small`: 0.0007272
  - `regional_bravo/tiny`: 0.0006681
  - `regional_bravo/tumor_bg_ratio`: 13.6652
  - `regional_bravo/tumor_loss`: 0.0006295

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1.675e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.2257, max 2.5935 @ ep 0
  - `training/grad_norm_max`: last 0.5585, max 8.0043 @ ep 6

#### `exp19_5_20260224-142309`
*started 2026-02-24 14:23 • 500 epochs • 51h54m • 214.0 TFLOPs • peak VRAM 21.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.5395 → 0.0449 (min 0.0367 @ ep 446)
  - `Loss/MSE_val`: 0.3410 → 0.0568 (min 0.0359 @ ep 392)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.7400
  - 0.1-0.2: 1.7060
  - 0.2-0.3: 1.6893
  - 0.3-0.4: 1.6835
  - 0.4-0.5: 1.6785
  - 0.5-0.6: 1.6729
  - 0.6-0.7: 1.6633
  - 0.7-0.8: 1.6620
  - 0.8-0.9: 1.6656
  - 0.9-1.0: 1.6633

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0649, best 0.0463 @ ep 415
  - `Generation/KID_mean_train`: last 0.0504, best 0.0381 @ ep 439
  - `Generation/KID_std_val`: last 0.0080, best 0.0047 @ ep 366
  - `Generation/KID_std_train`: last 0.0075, best 0.0038 @ ep 448
  - `Generation/CMMD_val`: last 0.2374, best 0.1732 @ ep 234
  - `Generation/CMMD_train`: last 0.2312, best 0.1720 @ ep 234
  - `Generation/extended_KID_mean_val`: last 0.0845, best 0.0736 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0715, best 0.0603 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2484, best 0.2285 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2371, best 0.2200 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3919 (min 0.2578, max 1.4800)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9722 (min 0.6066, max 0.9727)
  - `Validation/MS-SSIM_bravo`: last 0.9339 (min 0.6113, max 0.9734)
  - `Validation/PSNR_bravo`: last 34.5327 (min 18.7056, max 37.7675)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6632
  - `Generation_Diversity/extended_MSSSIM`: 0.1707

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006726
  - `regional_bravo/large`: 0.0090
  - `regional_bravo/medium`: 0.0112
  - `regional_bravo/small`: 0.0081
  - `regional_bravo/tiny`: 0.0052
  - `regional_bravo/tumor_bg_ratio`: 12.9009
  - `regional_bravo/tumor_loss`: 0.0087

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0885, max 9.4842 @ ep 0
  - `training/grad_norm_max`: last 0.8792, max 11.0252 @ ep 0

#### `exp19_6_wdm_dit_s_20260306-050100`
*started 2026-03-06 05:01 • 500 epochs • 9h54m • 138347.4 TFLOPs • peak VRAM 7.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9954 → 0.3924 (min 0.3101 @ ep 436)
  - `Loss/MSE_val`: 1.0560 → 0.4347 (min 0.2798 @ ep 45)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 2.0270
  - 0.1-0.2: 1.9949
  - 0.2-0.3: 1.6714
  - 0.3-0.4: 1.7129
  - 0.4-0.5: 1.4521
  - 0.5-0.6: 1.5446
  - 0.6-0.7: 1.2210
  - 0.7-0.8: 1.1561
  - 0.8-0.9: 1.1739
  - 0.9-1.0: 1.1808

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1351, best 0.1158 @ ep 284
  - `Generation/KID_mean_train`: last 0.1299, best 0.1097 @ ep 354
  - `Generation/KID_std_val`: last 0.0071, best 0.0058 @ ep 405
  - `Generation/KID_std_train`: last 0.0095, best 0.0052 @ ep 206
  - `Generation/CMMD_val`: last 0.4029, best 0.3911 @ ep 487
  - `Generation/CMMD_train`: last 0.4011, best 0.3888 @ ep 487
  - `Generation/extended_KID_mean_val`: last 0.1124, best 0.1069 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.1030, best 0.0989 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.4056, best 0.4056 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.4021, best 0.4021 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2565 (min 0.1882, max 1.4662)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9441 (min 0.4156, max 0.9460)
  - `Validation/MS-SSIM_bravo`: last 0.9260 (min 0.4213, max 0.9546)
  - `Validation/PSNR_bravo`: last 32.9921 (min 22.3960, max 34.7848)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3808
  - `Generation_Diversity/extended_MSSSIM`: 0.1864

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008084
  - `regional_bravo/large`: 0.0096
  - `regional_bravo/medium`: 0.0117
  - `regional_bravo/small`: 0.0087
  - `regional_bravo/tiny`: 0.0048
  - `regional_bravo/tumor_bg_ratio`: 11.1902
  - `regional_bravo/tumor_loss`: 0.0090

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.2076, max 0.7842 @ ep 11
  - `training/grad_norm_max`: last 1.0744, max 7.0391 @ ep 11

### exp20

**exp20 (model scaling)** — pixel-space UNet sweep 270M → 152M → 67M → 20M,
with/without attention. Per memory: 270M FID 51.17 (baseline), 152M 72.11,
67M+attn 99.86, 67M-attn 92.16, 20M 94.75. 270M→152M = 41% FID degradation;
attention does not help at 67M. Val loss and FID are uncorrelated at small
sizes (20M has val loss 0.00212 but FID 94.75).

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp20_5_pixel_bravo_mid_152m_20260314-064756` — 0.0255
  2. 🥈 `exp20_6_pixel_bravo_tiny_20m_20260314-072201` — 0.0321
  3.  `exp20_2_pixel_bravo_deep_wide_20260303-031712` — 0.0370
  4.  `exp20_1_pixel_bravo_attn_l3_20260301-015714` — 0.0418
  5.  `exp20_4_pixel_bravo_small_67m_20260314-064756` — 0.0428
  6.  `exp20_7_pixel_bravo_67m_no_attn_20260314-072441` — 0.0469
  7.  `exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712` — 0.0474

#### `exp20_1_pixel_bravo_attn_l3_20260301-015714`
*started 2026-03-01 01:57 • 318 epochs • 75h19m • 232778.5 TFLOPs • peak VRAM 68.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9674 → 0.0025 (min 0.0023 @ ep 315)
  - `Loss/MSE_val`: 0.9231 → 0.0027 (min 0.0024 @ ep 174)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0202
  - 0.1-0.2: 0.0056
  - 0.2-0.3: 0.0049
  - 0.3-0.4: 0.0038
  - 0.4-0.5: 0.0023
  - 0.5-0.6: 0.0022
  - 0.6-0.7: 0.0014
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0559, best 0.0334 @ ep 286
  - `Generation/KID_mean_train`: last 0.0540, best 0.0302 @ ep 295
  - `Generation/KID_std_val`: last 0.0031, best 0.0028 @ ep 293
  - `Generation/KID_std_train`: last 0.0030, best 0.0025 @ ep 286
  - `Generation/CMMD_val`: last 0.2248, best 0.2026 @ ep 262
  - `Generation/CMMD_train`: last 0.2239, best 0.1947 @ ep 262
  - `Generation/extended_KID_mean_val`: last 0.1481, best 0.0418 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.1315, best 0.0336 @ ep 274
  - `Generation/extended_CMMD_val`: last 0.3483, best 0.1979 @ ep 249
  - `Generation/extended_CMMD_train`: last 0.3350, best 0.1938 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6488 (min 0.4516, max 1.7819)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9562 (min 0.2381, max 0.9618)
  - `Validation/MS-SSIM_bravo`: last 0.9519 (min 0.2811, max 0.9653)
  - `Validation/PSNR_bravo`: last 32.7875 (min 10.7012, max 34.1230)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4921
  - `Generation_Diversity/extended_MSSSIM`: 0.2041

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005855
  - `regional_bravo/large`: 0.0094
  - `regional_bravo/medium`: 0.0106
  - `regional_bravo/small`: 0.0079
  - `regional_bravo/tiny`: 0.0062
  - `regional_bravo/tumor_bg_ratio`: 15.0624
  - `regional_bravo/tumor_loss`: 0.0088

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 3.105e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0543, max 6.6431 @ ep 44
  - `training/grad_norm_max`: last 0.2126, max 40.2983 @ ep 44

#### `exp20_2_pixel_bravo_deep_wide_20260303-031712`
*started 2026-03-03 03:17 • 500 epochs • 104h30m • 24969.3 TFLOPs • peak VRAM 83.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9894 → 0.0020 (min 0.0014 @ ep 494)
  - `Loss/MSE_val`: 0.9684 → 0.0064 (min 0.0023 @ ep 183)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0744
  - 0.1-0.2: 0.0164
  - 0.2-0.3: 0.0130
  - 0.3-0.4: 0.0055
  - 0.4-0.5: 0.0050
  - 0.5-0.6: 0.0033
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0021

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0310, best 0.0245 @ ep 317
  - `Generation/KID_mean_train`: last 0.0284, best 0.0239 @ ep 317
  - `Generation/KID_std_val`: last 0.0032, best 0.0023 @ ep 425
  - `Generation/KID_std_train`: last 0.0025, best 0.0023 @ ep 466
  - `Generation/CMMD_val`: last 0.2527, best 0.2001 @ ep 219
  - `Generation/CMMD_train`: last 0.2543, best 0.1981 @ ep 219
  - `Generation/extended_KID_mean_val`: last 0.0517, best 0.0370 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0474, best 0.0324 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.2060, best 0.1924 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.2092, best 0.1929 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4709 (min 0.4202, max 1.7920)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9220 (min 0.2321, max 0.9601)
  - `Validation/MS-SSIM_bravo`: last 0.9271 (min 0.2449, max 0.9628)
  - `Validation/PSNR_bravo`: last 30.7881 (min 9.9595, max 33.7378)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2228
  - `Generation_Diversity/extended_MSSSIM`: 0.1065

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008942
  - `regional_bravo/large`: 0.0122
  - `regional_bravo/medium`: 0.0130
  - `regional_bravo/small`: 0.0122
  - `regional_bravo/tiny`: 0.0099
  - `regional_bravo/tumor_bg_ratio`: 13.3718
  - `regional_bravo/tumor_loss`: 0.0120

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 19, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0370, max 3.9011 @ ep 0
  - `training/grad_norm_max`: last 2.0095, max 72.1497 @ ep 12

#### `exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712`
*started 2026-03-03 03:17 • 500 epochs • 83h36m • 372228.7 TFLOPs • peak VRAM 70.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9891 → 0.0020 (min 0.0015 @ ep 418)
  - `Loss/MSE_val`: 0.9675 → 0.0063 (min 0.0021 @ ep 172)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0186
  - 0.1-0.2: 0.0164
  - 0.2-0.3: 0.0135
  - 0.3-0.4: 0.0072
  - 0.4-0.5: 0.0057
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0028
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0014
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0315, best 0.0238 @ ep 368
  - `Generation/KID_mean_train`: last 0.0291, best 0.0217 @ ep 368
  - `Generation/KID_std_val`: last 0.0031, best 0.0024 @ ep 498
  - `Generation/KID_std_train`: last 0.0026, best 0.0021 @ ep 422
  - `Generation/CMMD_val`: last 0.2412, best 0.1886 @ ep 227
  - `Generation/CMMD_train`: last 0.2398, best 0.1777 @ ep 227
  - `Generation/extended_KID_mean_val`: last 0.0474, best 0.0474 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0458, best 0.0452 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.2168, best 0.1967 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.2177, best 0.1948 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4949 (min 0.4538, max 1.7897)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9182 (min 0.2320, max 0.9592)
  - `Validation/MS-SSIM_bravo`: last 0.9241 (min 0.2430, max 0.9637)
  - `Validation/PSNR_bravo`: last 30.6421 (min 9.8390, max 33.7009)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1656
  - `Generation_Diversity/extended_MSSSIM`: 0.0878

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009177
  - `regional_bravo/large`: 0.0125
  - `regional_bravo/medium`: 0.0122
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0090
  - `regional_bravo/tumor_bg_ratio`: 12.8212
  - `regional_bravo/tumor_loss`: 0.0118

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 19, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0168, max 3.9237 @ ep 0
  - `training/grad_norm_max`: last 0.0861, max 32.9144 @ ep 12

#### `exp20_4_pixel_bravo_small_67m_20260314-064756`
*started 2026-03-14 06:47 • 500 epochs • 30h07m • 6877.7 TFLOPs • peak VRAM 24.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9883 → 0.0029 (min 0.0018 @ ep 451)
  - `Loss/MSE_val`: 0.9682 → 0.0051 (min 0.0026 @ ep 212)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0333
  - 0.1-0.2: 0.0141
  - 0.2-0.3: 0.0096
  - 0.3-0.4: 0.0062
  - 0.4-0.5: 0.0050
  - 0.5-0.6: 0.0032
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0030
  - 0.8-0.9: 0.0025
  - 0.9-1.0: 0.0030

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0469, best 0.0264 @ ep 261
  - `Generation/KID_mean_train`: last 0.0453, best 0.0255 @ ep 261
  - `Generation/KID_std_val`: last 0.0042, best 0.0032 @ ep 261
  - `Generation/KID_std_train`: last 0.0030, best 0.0027 @ ep 256
  - `Generation/CMMD_val`: last 0.1937, best 0.1682 @ ep 364
  - `Generation/CMMD_train`: last 0.1837, best 0.1537 @ ep 364
  - `Generation/extended_KID_mean_val`: last 0.0563, best 0.0428 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0506, best 0.0363 @ ep 274
  - `Generation/extended_CMMD_val`: last 0.1798, best 0.1623 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1658, best 0.1478 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6074 (min 0.4818, max 1.7950)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9172 (min 0.2328, max 0.9561)
  - `Validation/MS-SSIM_bravo`: last 0.9177 (min 0.2170, max 0.9611)
  - `Validation/PSNR_bravo`: last 30.4111 (min 9.2231, max 33.7607)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2941
  - `Generation_Diversity/extended_MSSSIM`: 0.1208

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0010
  - `regional_bravo/large`: 0.0141
  - `regional_bravo/medium`: 0.0126
  - `regional_bravo/small`: 0.0124
  - `regional_bravo/tiny`: 0.0085
  - `regional_bravo/tumor_bg_ratio`: 12.0285
  - `regional_bravo/tumor_loss`: 0.0123

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0309, max 2.7788 @ ep 0
  - `training/grad_norm_max`: last 1.2439, max 10.6118 @ ep 161

#### `exp20_5_pixel_bravo_mid_152m_20260314-064756`
*started 2026-03-14 06:47 • 500 epochs • 37h06m • 11144.1 TFLOPs • peak VRAM 35.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9791 → 0.0023 (min 0.0016 @ ep 435)
  - `Loss/MSE_val`: 0.9485 → 0.0035 (min 0.0024 @ ep 176)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0259
  - 0.1-0.2: 0.0161
  - 0.2-0.3: 0.0083
  - 0.3-0.4: 0.0065
  - 0.4-0.5: 0.0031
  - 0.5-0.6: 0.0037
  - 0.6-0.7: 0.0026
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0281, best 0.0144 @ ep 268
  - `Generation/KID_mean_train`: last 0.0270, best 0.0122 @ ep 268
  - `Generation/KID_std_val`: last 0.0035, best 0.0023 @ ep 281
  - `Generation/KID_std_train`: last 0.0033, best 0.0022 @ ep 374
  - `Generation/CMMD_val`: last 0.1712, best 0.1472 @ ep 294
  - `Generation/CMMD_train`: last 0.1620, best 0.1425 @ ep 294
  - `Generation/extended_KID_mean_val`: last 0.0255, best 0.0255 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0227, best 0.0227 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1567, best 0.1456 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.1517, best 0.1399 @ ep 274

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6108 (min 0.4678, max 1.7749)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9258 (min 0.2349, max 0.9588)
  - `Validation/MS-SSIM_bravo`: last 0.9190 (min 0.3190, max 0.9658)
  - `Validation/PSNR_bravo`: last 30.3473 (min 11.7356, max 33.9108)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2060
  - `Generation_Diversity/extended_MSSSIM`: 0.1050

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000978
  - `regional_bravo/large`: 0.0165
  - `regional_bravo/medium`: 0.0131
  - `regional_bravo/small`: 0.0162
  - `regional_bravo/tiny`: 0.0105
  - `regional_bravo/tumor_bg_ratio`: 14.6057
  - `regional_bravo/tumor_loss`: 0.0143

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0164, max 3.3663 @ ep 0
  - `training/grad_norm_max`: last 0.2176, max 5.2228 @ ep 13

#### `exp20_6_pixel_bravo_tiny_20m_20260314-072201`
*started 2026-03-14 07:22 • 500 epochs • 22h54m • 3187.0 TFLOPs • peak VRAM 21.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9885 → 0.0025 (min 0.0021 @ ep 497)
  - `Loss/MSE_val`: 0.9690 → 0.0030 (min 0.0021 @ ep 310)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0195
  - 0.1-0.2: 0.0098
  - 0.2-0.3: 0.0049
  - 0.3-0.4: 0.0030
  - 0.4-0.5: 0.0027
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0015

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0251, best 0.0164 @ ep 389
  - `Generation/KID_mean_train`: last 0.0220, best 0.0147 @ ep 389
  - `Generation/KID_std_val`: last 0.0049, best 0.0025 @ ep 388
  - `Generation/KID_std_train`: last 0.0042, best 0.0020 @ ep 434
  - `Generation/CMMD_val`: last 0.1858, best 0.1669 @ ep 304
  - `Generation/CMMD_train`: last 0.1780, best 0.1548 @ ep 304
  - `Generation/extended_KID_mean_val`: last 0.0396, best 0.0321 @ ep 224
  - `Generation/extended_KID_mean_train`: last 0.0342, best 0.0265 @ ep 224
  - `Generation/extended_CMMD_val`: last 0.1863, best 0.1614 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.1753, best 0.1479 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5767 (min 0.4800, max 1.7828)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9512 (min 0.2325, max 0.9585)
  - `Validation/MS-SSIM_bravo`: last 0.9453 (min 0.2850, max 0.9637)
  - `Validation/PSNR_bravo`: last 32.1478 (min 10.8647, max 33.6428)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2738
  - `Generation_Diversity/extended_MSSSIM`: 0.1408

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006951
  - `regional_bravo/large`: 0.0087
  - `regional_bravo/medium`: 0.0102
  - `regional_bravo/small`: 0.0103
  - `regional_bravo/tiny`: 0.0074
  - `regional_bravo/tumor_bg_ratio`: 13.2201
  - `regional_bravo/tumor_loss`: 0.0092

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0383, max 3.3245 @ ep 9
  - `training/grad_norm_max`: last 0.1641, max 60.8510 @ ep 9

#### `exp20_7_pixel_bravo_67m_no_attn_20260314-072441`
*started 2026-03-14 07:24 • 500 epochs • 30h32m • 152.9 TFLOPs • peak VRAM 20.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9883 → 0.0044 (min 0.0020 @ ep 461)
  - `Loss/MSE_val`: 0.9673 → 0.0040 (min 0.0022 @ ep 308)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0157
  - 0.1-0.2: 0.0102
  - 0.2-0.3: 0.0059
  - 0.3-0.4: 0.0048
  - 0.4-0.5: 0.0036
  - 0.5-0.6: 0.0025
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0663, best 0.0250 @ ep 343
  - `Generation/KID_mean_train`: last 0.0623, best 0.0219 @ ep 343
  - `Generation/KID_std_val`: last 0.0060, best 0.0026 @ ep 375
  - `Generation/KID_std_train`: last 0.0061, best 0.0029 @ ep 435
  - `Generation/CMMD_val`: last 0.2223, best 0.1709 @ ep 468
  - `Generation/CMMD_train`: last 0.2078, best 0.1636 @ ep 423
  - `Generation/extended_KID_mean_val`: last 0.0517, best 0.0469 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0457, best 0.0409 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1956, best 0.1886 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1824, best 0.1730 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5755 (min 0.4970, max 1.7871)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9464 (min 0.2317, max 0.9585)
  - `Validation/MS-SSIM_bravo`: last 0.9511 (min 0.2761, max 0.9628)
  - `Validation/PSNR_bravo`: last 32.3872 (min 10.6181, max 33.4534)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5608
  - `Generation_Diversity/extended_MSSSIM`: 0.2758

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006254
  - `regional_bravo/large`: 0.0081
  - `regional_bravo/medium`: 0.0078
  - `regional_bravo/small`: 0.0105
  - `regional_bravo/tiny`: 0.0080
  - `regional_bravo/tumor_bg_ratio`: 13.4924
  - `regional_bravo/tumor_loss`: 0.0084

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1023, max 2.7682 @ ep 0
  - `training/grad_norm_max`: last 6.3937, max 42.1178 @ ep 24

### exp23

**exp23 (ScoreAug)** — Score Augmentation (Hou et al. 2025). Variants:
exp23 with rotation+flip+translation+cutout enabled; exp23_1 ScoreAug-safe
with translation+cutout only (no rotation/flip).
Per memory (1000ep post-hoc): FID 20.38 at Euler-27, RadImageNet FID 0.659
at Euler-48 (best RIN). **But**: exp23 generations have ~20% broken brain
shapes even with perfect seg masks — ScoreAug damages anatomical fidelity
despite improving FID. Exp23 translation-leak traced to an omega-encoding
bug (identity encoded as [1,0,..,0] instead of zeros); fixed Apr 21, 2026
(requires retraining to benefit).

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp23_1_pixel_bravo_scoreaug_safe_20260402-122902` — 0.0151
  2. 🥈 `exp23_pixel_bravo_scoreaug_256_20260309-220559` — 0.0152

#### `exp23_pixel_bravo_scoreaug_256_20260309-220559`
*started 2026-03-09 22:05 • 1000 epochs • 110h59m • 32036.5 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9034 → 0.0029 (min 0.0020 @ ep 804)
  - `Loss/MSE_val`: 0.9241 → 0.0042 (min 0.0020 @ ep 726)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0171
  - 0.1-0.2: 0.0107
  - 0.2-0.3: 0.0057
  - 0.3-0.4: 0.0026
  - 0.4-0.5: 0.0029
  - 0.5-0.6: 0.0018
  - 0.6-0.7: 0.0016
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0748, best 0.0193 @ ep 890
  - `Generation/KID_mean_train`: last 0.0733, best 0.0185 @ ep 405
  - `Generation/KID_std_val`: last 0.0049, best 0.0024 @ ep 837
  - `Generation/KID_std_train`: last 0.0071, best 0.0029 @ ep 954
  - `Generation/CMMD_val`: last 0.2365, best 0.1353 @ ep 782
  - `Generation/CMMD_train`: last 0.2224, best 0.1246 @ ep 782
  - `Generation/extended_KID_mean_val`: last 0.0174, best 0.0152 @ ep 799
  - `Generation/extended_KID_mean_train`: last 0.0139, best 0.0118 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.1277, best 0.1263 @ ep 949
  - `Generation/extended_CMMD_train`: last 0.1158, best 0.1136 @ ep 949

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5062 (min 0.4215, max 1.7787)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9589 (min 0.2064, max 0.9627)
  - `Validation/MS-SSIM_bravo`: last 0.9563 (min 0.2643, max 0.9716)
  - `Validation/PSNR_bravo`: last 33.4646 (min 11.2459, max 34.7496)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6842
  - `Generation_Diversity/extended_MSSSIM`: 0.3550

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005486
  - `regional_bravo/large`: 0.0097
  - `regional_bravo/medium`: 0.0107
  - `regional_bravo/small`: 0.0087
  - `regional_bravo/tiny`: 0.0050
  - `regional_bravo/tumor_bg_ratio`: 16.1387
  - `regional_bravo/tumor_loss`: 0.0089

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0156, max 3.7852 @ ep 0
  - `training/grad_norm_max`: last 0.0768, max 25.4885 @ ep 27

#### `exp23_1_pixel_bravo_scoreaug_safe_20260402-122902`
*started 2026-04-02 12:29 • 1000 epochs • 95h56m • 32036.5 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8666 → 0.0023 (min 0.0018 @ ep 997)
  - `Loss/MSE_val`: 0.9268 → 0.0037 (min 0.0021 @ ep 433)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0120
  - 0.1-0.2: 0.0094
  - 0.2-0.3: 0.0060
  - 0.3-0.4: 0.0034
  - 0.4-0.5: 0.0040
  - 0.5-0.6: 0.0024
  - 0.6-0.7: 0.0025
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0019
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0235, best 0.0176 @ ep 808
  - `Generation/KID_mean_train`: last 0.0246, best 0.0174 @ ep 934
  - `Generation/KID_std_val`: last 0.0045, best 0.0023 @ ep 669
  - `Generation/KID_std_train`: last 0.0042, best 0.0021 @ ep 669
  - `Generation/CMMD_val`: last 0.1566, best 0.1244 @ ep 826
  - `Generation/CMMD_train`: last 0.1550, best 0.1192 @ ep 826
  - `Generation/extended_KID_mean_val`: last 0.0182, best 0.0151 @ ep 849
  - `Generation/extended_KID_mean_train`: last 0.0179, best 0.0155 @ ep 849
  - `Generation/extended_CMMD_val`: last 0.1242, best 0.1242 @ ep 999
  - `Generation/extended_CMMD_train`: last 0.1232, best 0.1232 @ ep 999

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5213 (min 0.4044, max 1.7790)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9361 (min 0.2347, max 0.9617)
  - `Validation/MS-SSIM_bravo`: last 0.9379 (min 0.2767, max 0.9699)
  - `Validation/PSNR_bravo`: last 31.6375 (min 10.8837, max 34.4170)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6281
  - `Generation_Diversity/extended_MSSSIM`: 0.3323

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007708
  - `regional_bravo/large`: 0.0125
  - `regional_bravo/medium`: 0.0198
  - `regional_bravo/small`: 0.0126
  - `regional_bravo/tiny`: 0.0071
  - `regional_bravo/tumor_bg_ratio`: 17.4406
  - `regional_bravo/tumor_loss`: 0.0134

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0123, max 3.7218 @ ep 0
  - `training/grad_norm_max`: last 0.0433, max 22.6436 @ ep 111

#### `exp23_pixel_bravo_scoreaug_256_20260422-000228`
*started 2026-04-22 00:02 • 7 epochs • 41m00s • 224.3 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9108 → 0.0171 (min 0.0171 @ ep 6)
  - `Loss/MSE_val`: 0.9247 → 0.0135 (min 0.0135 @ ep 6)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0901
  - 0.1-0.2: 0.0240
  - 0.2-0.3: 0.0219
  - 0.3-0.4: 0.0146
  - 0.4-0.5: 0.0127
  - 0.5-0.6: 0.0105
  - 0.6-0.7: 0.0111
  - 0.7-0.8: 0.0105
  - 0.8-0.9: 0.0112
  - 0.9-1.0: 0.9327

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.4161, best 0.4161 @ ep 6
  - `Generation/KID_mean_train`: last 0.4234, best 0.4234 @ ep 6
  - `Generation/KID_std_val`: last 0.0345, best 0.0195 @ ep 0
  - `Generation/KID_std_train`: last 0.0259, best 0.0172 @ ep 0
  - `Generation/CMMD_val`: last 0.6360, best 0.6360 @ ep 6
  - `Generation/CMMD_train`: last 0.6216, best 0.6216 @ ep 6

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.7024 (min 1.7024, max 1.7676)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8228 (min 0.2372, max 0.8228)
  - `Validation/MS-SSIM_bravo`: last 0.8184 (min 0.3249, max 0.8184)
  - `Validation/PSNR_bravo`: last 26.9021 (min 11.7904, max 26.9021)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0032
  - `regional_bravo/large`: 0.0198
  - `regional_bravo/medium`: 0.0203
  - `regional_bravo/small`: 0.0206
  - `regional_bravo/tiny`: 0.0209
  - `regional_bravo/tumor_bg_ratio`: 6.3259
  - `regional_bravo/tumor_loss`: 0.0203

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 0.0001

**Training meta:**
  - `training/grad_norm_avg`: last 0.2683, max 3.7944 @ ep 0
  - `training/grad_norm_max`: last 2.7131, max 38.4905 @ ep 5

### exp24

**exp24 (combined techniques)** — ScoreAug + adjusted offset + post-hoc EMA +
uniform-t + dropout + weight decay, 1000 epochs. Per memory: FID 62.87
(worse than exp23 alone 62.57), 742 gradient spikes (vs 9 for exp23),
worse LPIPS 0.881 vs 0.531. Stacking techniques ≠ ScoreAug alone.

#### `exp24_pixel_bravo_combined_20260315-193239`
*started 2026-03-15 19:32 • 1000 epochs • 112h40m • 32036.5 TFLOPs • peak VRAM 44.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9047 → 0.0166 (min 0.0034 @ ep 836)
  - `Loss/MSE_val`: 0.9392 → 0.0077 (min 0.0024 @ ep 918)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0429
  - 0.1-0.2: 0.0085
  - 0.2-0.3: 0.0056
  - 0.3-0.4: 0.0027
  - 0.4-0.5: 0.0027
  - 0.5-0.6: 0.0020
  - 0.6-0.7: 0.0014
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0393, best 0.0271 @ ep 998
  - `Generation/KID_mean_train`: last 0.0362, best 0.0259 @ ep 872
  - `Generation/KID_std_val`: last 0.0065, best 0.0026 @ ep 818
  - `Generation/KID_std_train`: last 0.0040, best 0.0022 @ ep 872
  - `Generation/CMMD_val`: last 0.2360, best 0.1767 @ ep 961
  - `Generation/CMMD_train`: last 0.2193, best 0.1678 @ ep 961
  - `Generation/extended_KID_mean_val`: last 0.0330, best 0.0328 @ ep 899
  - `Generation/extended_KID_mean_train`: last 0.0298, best 0.0296 @ ep 899
  - `Generation/extended_CMMD_val`: last 0.2044, best 0.2044 @ ep 999
  - `Generation/extended_CMMD_train`: last 0.1939, best 0.1939 @ ep 999

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7421 (min 0.4820, max 1.7615)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9570 (min 0.2337, max 0.9579)
  - `Validation/MS-SSIM_bravo`: last 0.9441 (min 0.3322, max 0.9659)
  - `Validation/PSNR_bravo`: last 33.4466 (min 12.5193, max 37.0741)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3141
  - `Generation_Diversity/extended_MSSSIM`: 0.1753

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006365
  - `regional_bravo/large`: 0.0092
  - `regional_bravo/medium`: 0.0100
  - `regional_bravo/small`: 0.0068
  - `regional_bravo/tiny`: 0.0036
  - `regional_bravo/tumor_bg_ratio`: 12.3225
  - `regional_bravo/tumor_loss`: 0.0078

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1754, max 3.8376 @ ep 240
  - `training/grad_norm_max`: last 7.4730, max 70.0203 @ ep 240

### exp25

**exp25** — 20M model + full technique stack at 2000 epochs. FID 73 —
better than vanilla 20M (FID 95) but still far from 270M (FID 51).
Techniques help but don't compensate for capacity gap.

#### `exp25_pixel_bravo_combined_20m_20260315-193139`
*started 2026-03-15 19:31 • 2000 epochs • 123h36m • 12748.0 TFLOPs • peak VRAM 20.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9210 → 0.0216 (min 0.0036 @ ep 1627)
  - `Loss/MSE_val`: 0.9747 → 0.0054 (min 0.0018 @ ep 1927)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0311
  - 0.1-0.2: 0.0074
  - 0.2-0.3: 0.0064
  - 0.3-0.4: 0.0037
  - 0.4-0.5: 0.0028
  - 0.5-0.6: 0.0015
  - 0.6-0.7: 0.0015
  - 0.7-0.8: 0.0013
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0017

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0717, best 0.0377 @ ep 1225
  - `Generation/KID_mean_train`: last 0.0718, best 0.0359 @ ep 1225
  - `Generation/KID_std_val`: last 0.0037, best 0.0029 @ ep 1422
  - `Generation/KID_std_train`: last 0.0043, best 0.0027 @ ep 1368
  - `Generation/CMMD_val`: last 0.2840, best 0.2101 @ ep 1085
  - `Generation/CMMD_train`: last 0.2691, best 0.1977 @ ep 1085
  - `Generation/extended_KID_mean_val`: last 0.0387, best 0.0386 @ ep 1799
  - `Generation/extended_KID_mean_train`: last 0.0371, best 0.0337 @ ep 1799
  - `Generation/extended_CMMD_val`: last 0.2366, best 0.2366 @ ep 1999
  - `Generation/extended_CMMD_train`: last 0.2288, best 0.2288 @ ep 1999

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6901 (min 0.3664, max 1.7718)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9584 (min 0.2314, max 0.9593)
  - `Validation/MS-SSIM_bravo`: last 0.9450 (min 0.3007, max 0.9743)
  - `Validation/PSNR_bravo`: last 33.1329 (min 11.6320, max 37.1478)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3163
  - `Generation_Diversity/extended_MSSSIM`: 0.1506

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006426
  - `regional_bravo/large`: 0.0138
  - `regional_bravo/medium`: 0.0090
  - `regional_bravo/small`: 0.0117
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 17.5756
  - `regional_bravo/tumor_loss`: 0.0113

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.4104, max 2.8269 @ ep 2
  - `training/grad_norm_max`: last 13.9796, max 49.1314 @ ep 35

### exp26

**exp26 (WDM + ScoreAug / long)** — wavelet diffusion variants.
Per memory: exp26_1 at 1000ep has FID 77 (regressed from 67 at 500ep
seen at exp19_2 level), WDM overfits beyond 500ep.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp26_1_wdm_1000ep_20260327-024744` — 0.0564
  2. 🥈 `exp26_wdm_scoreaug_20260325-013334` — 0.1286

#### `exp26_wdm_scoreaug_20260325-013334`
*started 2026-03-25 01:33 • 1000 epochs • 43h52m • 31713.1 TFLOPs • peak VRAM 23.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9282 → 0.3446 (min 0.2560 @ ep 741)
  - `Loss/MSE_val`: 0.9787 → 0.2949 (min 0.1892 @ ep 939)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 2.0287
  - 0.1-0.2: 2.3871
  - 0.2-0.3: 1.9759
  - 0.3-0.4: 1.7471
  - 0.4-0.5: 1.7889
  - 0.5-0.6: 1.6136
  - 0.6-0.7: 1.3503
  - 0.7-0.8: 1.2946
  - 0.8-0.9: 1.1417
  - 0.9-1.0: 1.1150

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1446, best 0.0804 @ ep 375
  - `Generation/KID_mean_train`: last 0.1423, best 0.0734 @ ep 375
  - `Generation/KID_std_val`: last 0.0232, best 0.0138 @ ep 97
  - `Generation/KID_std_train`: last 0.0223, best 0.0133 @ ep 375
  - `Generation/CMMD_val`: last 0.4851, best 0.3688 @ ep 375
  - `Generation/CMMD_train`: last 0.4628, best 0.3525 @ ep 375
  - `Generation/extended_KID_mean_val`: last 0.1450, best 0.1286 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.1402, best 0.1199 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.4530, best 0.4280 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.4367, best 0.4168 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2281 (min 0.1449, max 1.4475)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9813 (min 0.7356, max 0.9815)
  - `Validation/MS-SSIM_bravo`: last 0.9697 (min 0.7141, max 0.9835)
  - `Validation/PSNR_bravo`: last 36.5372 (min 26.4713, max 38.3453)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2697
  - `Generation_Diversity/extended_MSSSIM`: 0.0660

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0003239
  - `regional_bravo/large`: 0.0085
  - `regional_bravo/medium`: 0.0044
  - `regional_bravo/small`: 0.0044
  - `regional_bravo/tiny`: 0.0024
  - `regional_bravo/tumor_bg_ratio`: 16.4635
  - `regional_bravo/tumor_loss`: 0.0053

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3768, max 2.4440 @ ep 0
  - `training/grad_norm_max`: last 1.6465, max 43.8351 @ ep 222

#### `exp26_1_wdm_1000ep_20260327-024744`
*started 2026-03-27 02:47 • 1000 epochs • 62h16m • 31713.0 TFLOPs • peak VRAM 23.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9478 → 0.2788 (min 0.2139 @ ep 973)
  - `Loss/MSE_val`: 0.9693 → 0.3227 (min 0.1795 @ ep 731)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.9747
  - 0.1-0.2: 2.0175
  - 0.2-0.3: 1.7418
  - 0.3-0.4: 1.7918
  - 0.4-0.5: 1.9140
  - 0.5-0.6: 1.6637
  - 0.6-0.7: 1.2018
  - 0.7-0.8: 1.1819
  - 0.8-0.9: 1.2349
  - 0.9-1.0: 1.2576

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0736, best 0.0425 @ ep 623
  - `Generation/KID_mean_train`: last 0.0630, best 0.0374 @ ep 623
  - `Generation/KID_std_val`: last 0.0119, best 0.0059 @ ep 536
  - `Generation/KID_std_train`: last 0.0115, best 0.0062 @ ep 455
  - `Generation/CMMD_val`: last 0.2375, best 0.1941 @ ep 649
  - `Generation/CMMD_train`: last 0.2216, best 0.1782 @ ep 740
  - `Generation/extended_KID_mean_val`: last 0.0800, best 0.0564 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0759, best 0.0525 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.2232, best 0.2036 @ ep 649
  - `Generation/extended_CMMD_train`: last 0.2107, best 0.1894 @ ep 649

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.2039 (min 0.1353, max 1.4427)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9800 (min 0.7659, max 0.9812)
  - `Validation/MS-SSIM_bravo`: last 0.9552 (min 0.7570, max 0.9811)
  - `Validation/PSNR_bravo`: last 36.2033 (min 27.0210, max 38.0676)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4501
  - `Generation_Diversity/extended_MSSSIM`: 0.1709

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005071
  - `regional_bravo/large`: 0.0098
  - `regional_bravo/medium`: 0.0116
  - `regional_bravo/small`: 0.0070
  - `regional_bravo/tiny`: 0.0058
  - `regional_bravo/tumor_bg_ratio`: 17.6866
  - `regional_bravo/tumor_loss`: 0.0090

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.2924, max 2.4489 @ ep 0
  - `training/grad_norm_max`: last 1.2091, max 15.5508 @ ep 9

### exp29

**exp29** — mixup augmentation baseline at 1000 epochs.

#### `exp29_pixel_bravo_mixup_1000_20260404-032937`
*started 2026-04-04 03:29 • 1000 epochs • 96h30m • 32036.5 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9692 → 0.0021 (min 0.0013 @ ep 907)
  - `Loss/MSE_val`: 0.9268 → 0.0051 (min 0.0022 @ ep 550)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0173
  - 0.1-0.2: 0.0133
  - 0.2-0.3: 0.0092
  - 0.3-0.4: 0.0047
  - 0.4-0.5: 0.0053
  - 0.5-0.6: 0.0032
  - 0.6-0.7: 0.0018
  - 0.7-0.8: 0.0012
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0787, best 0.0231 @ ep 337
  - `Generation/KID_mean_train`: last 0.0736, best 0.0201 @ ep 337
  - `Generation/KID_std_val`: last 0.0092, best 0.0030 @ ep 529
  - `Generation/KID_std_train`: last 0.0077, best 0.0027 @ ep 313
  - `Generation/CMMD_val`: last 0.4084, best 0.1547 @ ep 475
  - `Generation/CMMD_train`: last 0.3865, best 0.1470 @ ep 475
  - `Generation/extended_KID_mean_val`: last 0.0801, best 0.0157 @ ep 599
  - `Generation/extended_KID_mean_train`: last 0.0758, best 0.0135 @ ep 599
  - `Generation/extended_CMMD_val`: last 0.3535, best 0.1346 @ ep 599
  - `Generation/extended_CMMD_train`: last 0.3375, best 0.1254 @ ep 599

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5251 (min 0.4331, max 1.7885)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9244 (min 0.2363, max 0.9603)
  - `Validation/MS-SSIM_bravo`: last 0.9279 (min 0.2622, max 0.9657)
  - `Validation/PSNR_bravo`: last 31.3081 (min 10.1227, max 33.9653)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3639
  - `Generation_Diversity/extended_MSSSIM`: 0.1217

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008254
  - `regional_bravo/large`: 0.0157
  - `regional_bravo/medium`: 0.0186
  - `regional_bravo/small`: 0.0113
  - `regional_bravo/tiny`: 0.0080
  - `regional_bravo/tumor_bg_ratio`: 17.1284
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0093, max 5.0577 @ ep 35
  - `training/grad_norm_max`: last 0.0513, max 21.7546 @ ep 35

### exp30

**exp30** — weight decay + mixup + EMA combination.

#### `exp30_pixel_bravo_wd_mixup_ema_20260404-033007`
*started 2026-04-04 03:30 • 1000 epochs • 103h10m • 32036.5 TFLOPs • peak VRAM 44.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9671 → 0.0019 (min 0.0013 @ ep 955)
  - `Loss/MSE_val`: 0.9226 → 0.0047 (min 0.0020 @ ep 236)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0253
  - 0.1-0.2: 0.0122
  - 0.2-0.3: 0.0075
  - 0.3-0.4: 0.0092
  - 0.4-0.5: 0.0039
  - 0.5-0.6: 0.0030
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0541, best 0.0166 @ ep 581
  - `Generation/KID_mean_train`: last 0.0494, best 0.0149 @ ep 581
  - `Generation/KID_std_val`: last 0.0060, best 0.0024 @ ep 985
  - `Generation/KID_std_train`: last 0.0047, best 0.0025 @ ep 581
  - `Generation/CMMD_val`: last 0.2741, best 0.1525 @ ep 665
  - `Generation/CMMD_train`: last 0.2591, best 0.1495 @ ep 665
  - `Generation/extended_KID_mean_val`: last 0.0515, best 0.0276 @ ep 699
  - `Generation/extended_KID_mean_train`: last 0.0448, best 0.0263 @ ep 699
  - `Generation/extended_CMMD_val`: last 0.2704, best 0.2117 @ ep 699
  - `Generation/extended_CMMD_train`: last 0.2573, best 0.2011 @ ep 849

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5174 (min 0.4306, max 1.7827)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9245 (min 0.2375, max 0.9613)
  - `Validation/MS-SSIM_bravo`: last 0.9296 (min 0.2827, max 0.9660)
  - `Validation/PSNR_bravo`: last 31.0706 (min 10.7683, max 34.1003)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2849
  - `Generation_Diversity/extended_MSSSIM`: 0.0929

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008063
  - `regional_bravo/large`: 0.0103
  - `regional_bravo/medium`: 0.0110
  - `regional_bravo/small`: 0.0112
  - `regional_bravo/tiny`: 0.0086
  - `regional_bravo/tumor_bg_ratio`: 12.8181
  - `regional_bravo/tumor_loss`: 0.0103

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0094, max 3.8958 @ ep 0
  - `training/grad_norm_max`: last 0.1427, max 17.9808 @ ep 4

### exp31

**exp31** — combined techniques without ScoreAug, three model sizes
(270M / 67M / 17M).

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp31_0_pixel_bravo_combined_no_sa_20260406-041427` — 0.0145
  2. 🥈 `exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255` — 0.0207
  3.  `exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125` — 0.0409

#### `exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125`
*started 2026-04-05 21:11 • 1000 epochs • 44h41m • 13755.4 TFLOPs • peak VRAM 25.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9993 → 0.0052 (min 0.0032 @ ep 779)
  - `Loss/MSE_val`: 0.9731 → 0.0187 (min 0.0029 @ ep 806)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1741
  - 0.1-0.2: 0.0139
  - 0.2-0.3: 0.0069
  - 0.3-0.4: 0.0054
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0041
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0269, best 0.0269 @ ep 999
  - `Generation/KID_mean_train`: last 0.0249, best 0.0249 @ ep 999
  - `Generation/KID_std_val`: last 0.0032, best 0.0022 @ ep 674
  - `Generation/KID_std_train`: last 0.0033, best 0.0022 @ ep 733
  - `Generation/CMMD_val`: last 0.1873, best 0.1548 @ ep 936
  - `Generation/CMMD_train`: last 0.1800, best 0.1452 @ ep 936
  - `Generation/extended_KID_mean_val`: last 0.0498, best 0.0409 @ ep 699
  - `Generation/extended_KID_mean_train`: last 0.0435, best 0.0332 @ ep 699
  - `Generation/extended_CMMD_val`: last 0.1905, best 0.1641 @ ep 799
  - `Generation/extended_CMMD_train`: last 0.1819, best 0.1527 @ ep 799

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7745 (min 0.4156, max 1.7690)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9227 (min 0.2313, max 0.9546)
  - `Validation/MS-SSIM_bravo`: last 0.9307 (min 0.3027, max 0.9645)
  - `Validation/PSNR_bravo`: last 32.3310 (min 11.7625, max 35.9975)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2855
  - `Generation_Diversity/extended_MSSSIM`: 0.1517

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.000842
  - `regional_bravo/large`: 0.0060
  - `regional_bravo/medium`: 0.0127
  - `regional_bravo/small`: 0.0107
  - `regional_bravo/tiny`: 0.0070
  - `regional_bravo/tumor_bg_ratio`: 10.6403
  - `regional_bravo/tumor_loss`: 0.0090

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0562, max 5.0916 @ ep 8
  - `training/grad_norm_max`: last 1.7436, max 141.7197 @ ep 11

#### `exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255`
*started 2026-04-05 21:12 • 2000 epochs • 128h51m • 12748.0 TFLOPs • peak VRAM 20.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0030 → 0.0126 (min 0.0025 @ ep 1858)
  - `Loss/MSE_val`: 0.9724 → 0.0098 (min 0.0028 @ ep 744)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0521
  - 0.1-0.2: 0.0191
  - 0.2-0.3: 0.0102
  - 0.3-0.4: 0.0119
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0037
  - 0.6-0.7: 0.0043
  - 0.7-0.8: 0.0040
  - 0.8-0.9: 0.0022
  - 0.9-1.0: 0.0028

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0302, best 0.0191 @ ep 859
  - `Generation/KID_mean_train`: last 0.0304, best 0.0184 @ ep 751
  - `Generation/KID_std_val`: last 0.0037, best 0.0020 @ ep 1930
  - `Generation/KID_std_train`: last 0.0051, best 0.0023 @ ep 1280
  - `Generation/CMMD_val`: last 0.1999, best 0.1760 @ ep 810
  - `Generation/CMMD_train`: last 0.2001, best 0.1659 @ ep 885
  - `Generation/extended_KID_mean_val`: last 0.0207, best 0.0207 @ ep 1999
  - `Generation/extended_KID_mean_train`: last 0.0182, best 0.0182 @ ep 1999
  - `Generation/extended_CMMD_val`: last 0.1385, best 0.1375 @ ep 1599
  - `Generation/extended_CMMD_train`: last 0.1356, best 0.1324 @ ep 1599

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6672 (min 0.3721, max 1.7646)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8823 (min 0.2316, max 0.9558)
  - `Validation/MS-SSIM_bravo`: last 0.8844 (min 0.3506, max 0.9631)
  - `Validation/PSNR_bravo`: last 30.2129 (min 12.6809, max 36.2180)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3635
  - `Generation_Diversity/extended_MSSSIM`: 0.1941

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0013
  - `regional_bravo/large`: 0.0131
  - `regional_bravo/medium`: 0.0174
  - `regional_bravo/small`: 0.0140
  - `regional_bravo/tiny`: 0.0117
  - `regional_bravo/tumor_bg_ratio`: 10.8242
  - `regional_bravo/tumor_loss`: 0.0142

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1176, max 2.7804 @ ep 0
  - `training/grad_norm_max`: last 9.5821, max 34.4673 @ ep 44

#### `exp31_0_pixel_bravo_combined_no_sa_20260406-041427`
*started 2026-04-06 04:14 • 1000 epochs • 108h44m • 32036.5 TFLOPs • peak VRAM 44.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9841 → 0.0113 (min 0.0035 @ ep 969)
  - `Loss/MSE_val`: 0.9406 → 0.0047 (min 0.0025 @ ep 477)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0320
  - 0.1-0.2: 0.0106
  - 0.2-0.3: 0.0052
  - 0.3-0.4: 0.0035
  - 0.4-0.5: 0.0032
  - 0.5-0.6: 0.0020
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0020
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0016

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0266, best 0.0177 @ ep 767
  - `Generation/KID_mean_train`: last 0.0269, best 0.0186 @ ep 957
  - `Generation/KID_std_val`: last 0.0044, best 0.0021 @ ep 606
  - `Generation/KID_std_train`: last 0.0039, best 0.0025 @ ep 915
  - `Generation/CMMD_val`: last 0.1762, best 0.1456 @ ep 802
  - `Generation/CMMD_train`: last 0.1666, best 0.1371 @ ep 802
  - `Generation/extended_KID_mean_val`: last 0.0189, best 0.0145 @ ep 899
  - `Generation/extended_KID_mean_train`: last 0.0182, best 0.0147 @ ep 899
  - `Generation/extended_CMMD_val`: last 0.1446, best 0.1351 @ ep 899
  - `Generation/extended_CMMD_train`: last 0.1395, best 0.1292 @ ep 949

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6405 (min 0.4113, max 1.7637)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9521 (min 0.2344, max 0.9594)
  - `Validation/MS-SSIM_bravo`: last 0.9433 (min 0.3326, max 0.9720)
  - `Validation/PSNR_bravo`: last 32.9094 (min 12.5813, max 36.8382)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2239
  - `Generation_Diversity/extended_MSSSIM`: 0.1188

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006674
  - `regional_bravo/large`: 0.0098
  - `regional_bravo/medium`: 0.0106
  - `regional_bravo/small`: 0.0084
  - `regional_bravo/tiny`: 0.0058
  - `regional_bravo/tumor_bg_ratio`: 13.4454
  - `regional_bravo/tumor_loss`: 0.0090

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0896, max 4.6590 @ ep 61
  - `training/grad_norm_max`: last 7.2393, max 38.5628 @ ep 6

### exp32

**exp32 (perceptual / FFL)** — LPIPS / FFL auxiliary losses. Variants:
exp32_1 FFL, exp32_2 LPIPS at low-t (perceptual_max_timestep ramp),
exp32_3 pseudo-Huber. 1000-epoch versions also trained.
Per memory (mean-blur analysis): exp32_2 used a LOW-t LPIPS schedule which
is **backwards** — the HF deficit is worst at mid/high t. This motivated
exp37's HIGH-t schedule.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp32_1_1000_pixel_bravo_ffl_20260412-151416` — 0.0103
  2. 🥈 `exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027` — 0.0107
  3.  `exp32_1_pixel_bravo_ffl_20260409-192332` — 0.0108
  4.  `exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057` — 0.0110
  5.  `exp32_2_pixel_bravo_lpips_lowt_20260411-203908` — 0.0141
  6.  `exp32_pixel_bravo_perceptual_20260409-192621` — 0.0575

#### `exp32_1_pixel_bravo_ffl_20260409-192332`
*started 2026-04-09 19:23 • 100 epochs • 7h25m • 3203.6 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0017 → 0.0021 (min 0.0013 @ ep 18)
  - `Loss/MSE_val`: 0.0050 → 0.0069 (min 0.0044 @ ep 33)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0177
  - 0.1-0.2: 0.0215
  - 0.2-0.3: 0.0111
  - 0.3-0.4: 0.0082
  - 0.4-0.5: 0.0038
  - 0.5-0.6: 0.0041
  - 0.6-0.7: 0.0038
  - 0.7-0.8: 0.0028
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0247, best 0.0167 @ ep 77
  - `Generation/KID_mean_train`: last 0.0262, best 0.0181 @ ep 30
  - `Generation/KID_std_val`: last 0.0055, best 0.0019 @ ep 64
  - `Generation/KID_std_train`: last 0.0041, best 0.0021 @ ep 35
  - `Generation/CMMD_val`: last 0.1718, best 0.1394 @ ep 15
  - `Generation/CMMD_train`: last 0.1712, best 0.1345 @ ep 15
  - `Generation/extended_KID_mean_val`: last 0.0131, best 0.0108 @ ep 29
  - `Generation/extended_KID_mean_train`: last 0.0147, best 0.0100 @ ep 29
  - `Generation/extended_CMMD_val`: last 0.1342, best 0.1131 @ ep 39
  - `Generation/extended_CMMD_train`: last 0.1339, best 0.1119 @ ep 34

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4429 (min 0.4136, max 0.4869)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9047 (min 0.9047, max 0.9113)
  - `Validation/MS-SSIM_bravo`: last 0.9087 (min 0.9011, max 0.9228)
  - `Validation/PSNR_bravo`: last 30.0775 (min 29.4321, max 30.8594)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2224
  - `Generation_Diversity/extended_MSSSIM`: 0.1241

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0184
  - `regional_bravo/medium`: 0.0202
  - `regional_bravo/small`: 0.0153
  - `regional_bravo/tiny`: 0.0108
  - `regional_bravo/tumor_bg_ratio`: 15.3141
  - `regional_bravo/tumor_loss`: 0.0168

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0110, max 0.0198 @ ep 14
  - `training/grad_norm_max`: last 0.1301, max 0.5612 @ ep 55

#### `exp32_pixel_bravo_perceptual_20260409-192621`
*started 2026-04-09 19:26 • 100 epochs • 9h44m • 3203.6 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0023 → 0.0026 (min 0.0018 @ ep 17)
  - `Loss/MSE_val`: 0.0061 → 0.0054 (min 0.0041 @ ep 35)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0160
  - 0.1-0.2: 0.0141
  - 0.2-0.3: 0.0117
  - 0.3-0.4: 0.0066
  - 0.4-0.5: 0.0057
  - 0.5-0.6: 0.0052
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0022
  - 0.8-0.9: 0.0022
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0697, best 0.0230 @ ep 1
  - `Generation/KID_mean_train`: last 0.0681, best 0.0206 @ ep 1
  - `Generation/KID_std_val`: last 0.0049, best 0.0030 @ ep 89
  - `Generation/KID_std_train`: last 0.0058, best 0.0028 @ ep 87
  - `Generation/CMMD_val`: last 0.1278, best 0.1151 @ ep 89
  - `Generation/CMMD_train`: last 0.1138, best 0.0949 @ ep 89
  - `Generation/extended_KID_mean_val`: last 0.0615, best 0.0575 @ ep 69
  - `Generation/extended_KID_mean_train`: last 0.0569, best 0.0531 @ ep 69
  - `Generation/extended_CMMD_val`: last 0.1330, best 0.1330 @ ep 99
  - `Generation/extended_CMMD_train`: last 0.1181, best 0.1181 @ ep 99

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5389 (min 0.3935, max 0.6643)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9123 (min 0.8986, max 0.9158)
  - `Validation/MS-SSIM_bravo`: last 0.9126 (min 0.8996, max 0.9328)
  - `Validation/PSNR_bravo`: last 29.9998 (min 29.4619, max 31.2170)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1749
  - `Generation_Diversity/extended_MSSSIM`: 0.0980

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0093
  - `regional_bravo/medium`: 0.0177
  - `regional_bravo/small`: 0.0147
  - `regional_bravo/tiny`: 0.0099
  - `regional_bravo/tumor_bg_ratio`: 11.8344
  - `regional_bravo/tumor_loss`: 0.0128

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 19.9631, max 29.2491 @ ep 2
  - `training/grad_norm_max`: last 84.4883, max 115.6002 @ ep 8

#### `exp32_2_pixel_bravo_lpips_lowt_20260411-203908`
*started 2026-04-11 20:39 • 100 epochs • 11h25m • 3203.6 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0017 → 0.0021 (min 0.0013 @ ep 44)
  - `Loss/MSE_val`: 0.0069 → 0.0075 (min 0.0035 @ ep 60)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0218
  - 0.1-0.2: 0.0190
  - 0.2-0.3: 0.0115
  - 0.3-0.4: 0.0064
  - 0.4-0.5: 0.0083
  - 0.5-0.6: 0.0042
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0031
  - 0.8-0.9: 0.0022
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0194, best 0.0120 @ ep 33
  - `Generation/KID_mean_train`: last 0.0173, best 0.0097 @ ep 67
  - `Generation/KID_std_val`: last 0.0042, best 0.0020 @ ep 93
  - `Generation/KID_std_train`: last 0.0035, best 0.0023 @ ep 27
  - `Generation/CMMD_val`: last 0.1573, best 0.1039 @ ep 10
  - `Generation/CMMD_train`: last 0.1454, best 0.0880 @ ep 10
  - `Generation/extended_KID_mean_val`: last 0.0158, best 0.0141 @ ep 79
  - `Generation/extended_KID_mean_train`: last 0.0125, best 0.0111 @ ep 79
  - `Generation/extended_CMMD_val`: last 0.1203, best 0.1093 @ ep 84
  - `Generation/extended_CMMD_train`: last 0.1135, best 0.0954 @ ep 89

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3981 (min 0.3487, max 0.6529)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9089 (min 0.9036, max 0.9121)
  - `Validation/MS-SSIM_bravo`: last 0.9187 (min 0.8969, max 0.9296)
  - `Validation/PSNR_bravo`: last 30.3412 (min 29.2111, max 31.3564)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1202
  - `Generation_Diversity/extended_MSSSIM`: 0.0638

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009981
  - `regional_bravo/large`: 0.0126
  - `regional_bravo/medium`: 0.0179
  - `regional_bravo/small`: 0.0115
  - `regional_bravo/tiny`: 0.0098
  - `regional_bravo/tumor_bg_ratio`: 13.3431
  - `regional_bravo/tumor_loss`: 0.0133

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3385, max 0.5593 @ ep 55
  - `training/grad_norm_max`: last 5.4849, max 9.0801 @ ep 43

#### `exp32_1_1000_pixel_bravo_ffl_20260412-151416`
*started 2026-04-12 15:14 • 1000 epochs • 147h01m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0024 → 0.0013 (min 0.0010 @ ep 919)
  - `Loss/MSE_val`: 0.0071 → 0.0103 (min 0.0039 @ ep 229)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0544
  - 0.1-0.2: 0.0265
  - 0.2-0.3: 0.0178
  - 0.3-0.4: 0.0095
  - 0.4-0.5: 0.0068
  - 0.5-0.6: 0.0052
  - 0.6-0.7: 0.0040
  - 0.7-0.8: 0.0032
  - 0.8-0.9: 0.0021
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0226, best 0.0141 @ ep 296
  - `Generation/KID_mean_train`: last 0.0244, best 0.0141 @ ep 143
  - `Generation/KID_std_val`: last 0.0025, best 0.0022 @ ep 736
  - `Generation/KID_std_train`: last 0.0039, best 0.0021 @ ep 143
  - `Generation/CMMD_val`: last 0.1647, best 0.1241 @ ep 210
  - `Generation/CMMD_train`: last 0.1618, best 0.1189 @ ep 210
  - `Generation/extended_KID_mean_val`: last 0.0131, best 0.0103 @ ep 799
  - `Generation/extended_KID_mean_train`: last 0.0130, best 0.0098 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1309, best 0.1105 @ ep 949
  - `Generation/extended_CMMD_train`: last 0.1348, best 0.1142 @ ep 949

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3963 (min 0.3658, max 0.4919)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8887 (min 0.8887, max 0.9108)
  - `Validation/MS-SSIM_bravo`: last 0.8925 (min 0.8794, max 0.9250)
  - `Validation/PSNR_bravo`: last 29.3177 (min 28.5697, max 31.6122)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2615
  - `Generation_Diversity/extended_MSSSIM`: 0.1396

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0013
  - `regional_bravo/large`: 0.0177
  - `regional_bravo/medium`: 0.0235
  - `regional_bravo/small`: 0.0163
  - `regional_bravo/tiny`: 0.0110
  - `regional_bravo/tumor_bg_ratio`: 13.4851
  - `regional_bravo/tumor_loss`: 0.0177

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0082, max 0.0268 @ ep 108
  - `training/grad_norm_max`: last 0.0378, max 1.5200 @ ep 676

#### `exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027`
*started 2026-04-12 15:30 • 1000 epochs • 137h24m • 32036.5 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0019 → 0.0016 (min 0.0011 @ ep 815)
  - `Loss/MSE_val`: 0.0064 → 0.0061 (min 0.0038 @ ep 58)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0244
  - 0.1-0.2: 0.0136
  - 0.2-0.3: 0.0194
  - 0.3-0.4: 0.0055
  - 0.4-0.5: 0.0058
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0039
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0183, best 0.0092 @ ep 224
  - `Generation/KID_mean_train`: last 0.0190, best 0.0098 @ ep 371
  - `Generation/KID_std_val`: last 0.0046, best 0.0019 @ ep 724
  - `Generation/KID_std_train`: last 0.0028, best 0.0019 @ ep 96
  - `Generation/CMMD_val`: last 0.1815, best 0.0945 @ ep 26
  - `Generation/CMMD_train`: last 0.1797, best 0.0828 @ ep 26
  - `Generation/extended_KID_mean_val`: last 0.0119, best 0.0107 @ ep 799
  - `Generation/extended_KID_mean_train`: last 0.0132, best 0.0080 @ ep 799
  - `Generation/extended_CMMD_val`: last 0.1159, best 0.1096 @ ep 899
  - `Generation/extended_CMMD_train`: last 0.1157, best 0.1002 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4621 (min 0.3465, max 1.0442)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8986 (min 0.8974, max 0.9130)
  - `Validation/MS-SSIM_bravo`: last 0.8959 (min 0.8899, max 0.9300)
  - `Validation/PSNR_bravo`: last 29.3743 (min 28.9755, max 31.3049)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2830
  - `Generation_Diversity/extended_MSSSIM`: 0.1424

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0013
  - `regional_bravo/large`: 0.0186
  - `regional_bravo/medium`: 0.0198
  - `regional_bravo/small`: 0.0156
  - `regional_bravo/tiny`: 0.0095
  - `regional_bravo/tumor_bg_ratio`: 13.0655
  - `regional_bravo/tumor_loss`: 0.0166

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0876, max 0.7084 @ ep 145
  - `training/grad_norm_max`: last 1.4782, max 8.7242 @ ep 457

#### `exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057`
*started 2026-04-15 04:10 • 1000 epochs • 131h18m • 32036.5 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 117.5729 → 111.7511 (min 88.4243 @ ep 851)
  - `Loss/MSE_val`: 0.0065 → 0.0100 (min 0.0041 @ ep 301)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0922
  - 0.1-0.2: 0.0291
  - 0.2-0.3: 0.0252
  - 0.3-0.4: 0.0110
  - 0.4-0.5: 0.0067
  - 0.5-0.6: 0.0040
  - 0.6-0.7: 0.0037
  - 0.7-0.8: 0.0036
  - 0.8-0.9: 0.0019
  - 0.9-1.0: 0.0024

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0265, best 0.0137 @ ep 143
  - `Generation/KID_mean_train`: last 0.0289, best 0.0150 @ ep 143
  - `Generation/KID_std_val`: last 0.0038, best 0.0019 @ ep 334
  - `Generation/KID_std_train`: last 0.0046, best 0.0022 @ ep 937
  - `Generation/CMMD_val`: last 0.1748, best 0.1285 @ ep 152
  - `Generation/CMMD_train`: last 0.1784, best 0.1239 @ ep 152
  - `Generation/extended_KID_mean_val`: last 0.0117, best 0.0110 @ ep 599
  - `Generation/extended_KID_mean_train`: last 0.0128, best 0.0107 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.1270, best 0.1160 @ ep 699
  - `Generation/extended_CMMD_train`: last 0.1315, best 0.1171 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4185 (min 0.3609, max 0.4905)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8844 (min 0.8839, max 0.9109)
  - `Validation/MS-SSIM_bravo`: last 0.8889 (min 0.8801, max 0.9290)
  - `Validation/PSNR_bravo`: last 28.9263 (min 28.5185, max 31.1749)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2886
  - `Generation_Diversity/extended_MSSSIM`: 0.1385

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0013
  - `regional_bravo/large`: 0.0144
  - `regional_bravo/medium`: 0.0214
  - `regional_bravo/small`: 0.0181
  - `regional_bravo/tiny`: 0.0123
  - `regional_bravo/tumor_bg_ratio`: 12.3644
  - `regional_bravo/tumor_loss`: 0.0166

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 494.2967, max 687.1339 @ ep 247
  - `training/grad_norm_max`: last 2850.6045, max 1.591e+04 @ ep 972

### exp34

**exp34 (Mamba / LaMamba-Diff)** — SS2D + window attention + FFN
(LaMamba-Diff architecture). Variants: S p4, L p4 with gradient
checkpointing; 1000-epoch versions. First state-space-model pixel-space
test at 3D. Cross-merge bug identified + fixed on Apr 20, 2026 (wrong
spatial alignment of directional scan contributions).

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp34_mamba_l_p4_gc_20260416-005333` — 0.1476
  2. 🥈 `exp34_1_1000_mamba_l_p4_gc_20260419-152401` — 0.1615
  3.  `exp34_0_1000_mamba_s_p4_20260419-154005` — 0.1665
  4.  `exp34_mamba_s_p4_20260416-005333` — 0.1992

#### `exp34_mamba_l_p4_gc_20260416-005333`
*started 2026-04-16 00:53 • 500 epochs • 68h44m • 844627.1 TFLOPs • peak VRAM 37.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0045 → 0.0012 (min 0.0009781 @ ep 482)
  - `Loss/MSE_val`: 1.0013 → 0.0110 (min 0.0033 @ ep 106)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0645
  - 0.1-0.2: 0.0308
  - 0.2-0.3: 0.0228
  - 0.3-0.4: 0.0070
  - 0.4-0.5: 0.0076
  - 0.5-0.6: 0.0039
  - 0.6-0.7: 0.0034
  - 0.7-0.8: 0.0019
  - 0.8-0.9: 0.0019
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0492, best 0.0221 @ ep 373
  - `Generation/KID_mean_train`: last 0.0403, best 0.0187 @ ep 373
  - `Generation/KID_std_val`: last 0.0089, best 0.0027 @ ep 333
  - `Generation/KID_std_train`: last 0.0071, best 0.0027 @ ep 455
  - `Generation/CMMD_val`: last 0.2080, best 0.1741 @ ep 388
  - `Generation/CMMD_train`: last 0.1867, best 0.1570 @ ep 388
  - `Generation/extended_KID_mean_val`: last 0.1611, best 0.1476 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.1578, best 0.1457 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2616, best 0.2253 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2469, best 0.2118 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3261 (min 0.0245, max 1.7847)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8972 (min 0.2460, max 0.9409)
  - `Validation/MS-SSIM_bravo`: last 0.8951 (min 0.2806, max 0.9336)
  - `Validation/PSNR_bravo`: last 29.2747 (min 10.4825, max 31.4257)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2251
  - `Generation_Diversity/extended_MSSSIM`: 0.1389

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0012
  - `regional_bravo/large`: 0.0178
  - `regional_bravo/medium`: 0.0143
  - `regional_bravo/small`: 0.0127
  - `regional_bravo/tiny`: 0.0091
  - `regional_bravo/tumor_bg_ratio`: 11.4775
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0059, max 20.4800 @ ep 5
  - `training/grad_norm_max`: last 0.0272, max 55.4175 @ ep 5

#### `exp34_mamba_s_p4_20260416-005333`
*started 2026-04-16 00:53 • 500 epochs • 28h50m • 238216.8 TFLOPs • peak VRAM 50.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0057 → 0.0041 (min 0.0015 @ ep 446)
  - `Loss/MSE_val`: 1.0043 → 0.0108 (min 0.0037 @ ep 220)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1072
  - 0.1-0.2: 0.0329
  - 0.2-0.3: 0.0209
  - 0.3-0.4: 0.0074
  - 0.4-0.5: 0.0054
  - 0.5-0.6: 0.0034
  - 0.6-0.7: 0.0034
  - 0.7-0.8: 0.0021
  - 0.8-0.9: 0.0012
  - 0.9-1.0: 0.000967

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0571, best 0.0449 @ ep 467
  - `Generation/KID_mean_train`: last 0.0493, best 0.0391 @ ep 467
  - `Generation/KID_std_val`: last 0.0081, best 0.0059 @ ep 399
  - `Generation/KID_std_train`: last 0.0069, best 0.0047 @ ep 339
  - `Generation/CMMD_val`: last 0.2763, best 0.2387 @ ep 467
  - `Generation/CMMD_train`: last 0.2624, best 0.2269 @ ep 467
  - `Generation/extended_KID_mean_val`: last 0.1992, best 0.1992 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.1954, best 0.1954 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.3717, best 0.3717 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.3559, best 0.3559 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4368 (min 0.3699, max 1.7861)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9000 (min 0.2322, max 0.9366)
  - `Validation/MS-SSIM_bravo`: last 0.9016 (min 0.2908, max 0.9411)
  - `Validation/PSNR_bravo`: last 29.7297 (min 10.2600, max 31.7909)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3088
  - `Generation_Diversity/extended_MSSSIM`: 0.2168

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0179
  - `regional_bravo/medium`: 0.0135
  - `regional_bravo/small`: 0.0125
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 12.6186
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0219, max 18.7791 @ ep 7
  - `training/grad_norm_max`: last 1.1914, max 46.7893 @ ep 9

#### `exp34_1_1000_mamba_l_p4_gc_20260419-152401`
*started 2026-04-19 15:24 • 590 epochs • 56h29m • 996659.6 TFLOPs • peak VRAM 37.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0045 → 0.0012 (min 0.0008322 @ ep 585)
  - `Loss/MSE_val`: 1.0012 → 0.0158 (min 0.0036 @ ep 214)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1317
  - 0.1-0.2: 0.0417
  - 0.2-0.3: 0.0174
  - 0.3-0.4: 0.0111
  - 0.4-0.5: 0.0059
  - 0.5-0.6: 0.0029
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0017
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0366, best 0.0258 @ ep 532
  - `Generation/KID_mean_train`: last 0.0315, best 0.0229 @ ep 536
  - `Generation/KID_std_val`: last 0.0045, best 0.0033 @ ep 534
  - `Generation/KID_std_train`: last 0.0053, best 0.0025 @ ep 441
  - `Generation/CMMD_val`: last 0.2290, best 0.1464 @ ep 573
  - `Generation/CMMD_train`: last 0.2161, best 0.1332 @ ep 573
  - `Generation/extended_KID_mean_val`: last 0.1719, best 0.1615 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.1675, best 0.1574 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.2917, best 0.2485 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.2765, best 0.2299 @ ep 349

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4402 (min 0.2901, max 1.7806)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9080 (min 0.2460, max 0.9376)
  - `Validation/MS-SSIM_bravo`: last 0.9118 (min 0.3053, max 0.9323)
  - `Validation/PSNR_bravo`: last 29.7283 (min 10.8135, max 31.2974)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3422
  - `Generation_Diversity/extended_MSSSIM`: 0.1993

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0164
  - `regional_bravo/medium`: 0.0131
  - `regional_bravo/small`: 0.0110
  - `regional_bravo/tiny`: 0.0084
  - `regional_bravo/tumor_bg_ratio`: 11.4673
  - `regional_bravo/tumor_loss`: 0.0129

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 3.731e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0146, max 16.2741 @ ep 5
  - `training/grad_norm_max`: last 0.0687, max 90.7215 @ ep 43

#### `exp34_0_1000_mamba_s_p4_20260419-154005`
*started 2026-04-19 15:40 • 1000 epochs • 47h20m • 476433.2 TFLOPs • peak VRAM 50.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0057 → 0.0014 (min 0.0009871 @ ep 812)
  - `Loss/MSE_val`: 1.0044 → 0.0081 (min 0.0031 @ ep 128)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1311
  - 0.1-0.2: 0.0442
  - 0.2-0.3: 0.0194
  - 0.3-0.4: 0.0100
  - 0.4-0.5: 0.0054
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0032
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0459, best 0.0244 @ ep 958
  - `Generation/KID_mean_train`: last 0.0393, best 0.0211 @ ep 804
  - `Generation/KID_std_val`: last 0.0067, best 0.0033 @ ep 958
  - `Generation/KID_std_train`: last 0.0068, best 0.0029 @ ep 952
  - `Generation/CMMD_val`: last 0.2156, best 0.1728 @ ep 862
  - `Generation/CMMD_train`: last 0.2020, best 0.1620 @ ep 862
  - `Generation/extended_KID_mean_val`: last 0.1729, best 0.1665 @ ep 949
  - `Generation/extended_KID_mean_train`: last 0.1666, best 0.1613 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.2922, best 0.2776 @ ep 949
  - `Generation/extended_CMMD_train`: last 0.2773, best 0.2630 @ ep 949

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3254 (min 0.3124, max 1.7914)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9010 (min 0.2312, max 0.9428)
  - `Validation/MS-SSIM_bravo`: last 0.9010 (min 0.2630, max 0.9446)
  - `Validation/PSNR_bravo`: last 29.3186 (min 9.7158, max 31.9976)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3911
  - `Generation_Diversity/extended_MSSSIM`: 0.2037

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0012
  - `regional_bravo/large`: 0.0213
  - `regional_bravo/medium`: 0.0149
  - `regional_bravo/small`: 0.0146
  - `regional_bravo/tiny`: 0.0099
  - `regional_bravo/tumor_bg_ratio`: 13.2097
  - `regional_bravo/tumor_loss`: 0.0160

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0082, max 20.3538 @ ep 7
  - `training/grad_norm_max`: last 0.0335, max 50.7925 @ ep 8

### exp35

**exp35 (ScoreAug fine-tunes from exp1_1_1000)** — three variants exploring
per-t ScoreAug schedules. exp35_1 "detail" (low-t aug: [0.05, 0.10, 0.30]),
exp35_2 "structure", exp35_3 "uniform". Launched after omega-encoding bug
fix (Apr 21, 2026). Checkpoint topology auto-remap added so exp1_1_1000's
bare UNet weights load cleanly into the ScoreAug-wrapped model.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp35_2_scoreaug_structure_20260421-105310` — 0.0112
  2. 🥈 `exp35_1_scoreaug_detail_20260421-093244` — 0.0123
  3.  `exp35_3_scoreaug_uniform_20260421-115605` — 0.0163

#### `exp35_1_scoreaug_detail_20260421-093244`
*started 2026-04-21 09:32 • 149 epochs • 15h13m • 4773.4 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0036 → 0.0022 (min 0.0015 @ ep 93)
  - `Loss/MSE_val`: 0.0056 → 0.0045 (min 0.0026 @ ep 16)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0204
  - 0.1-0.2: 0.0079
  - 0.2-0.3: 0.0059
  - 0.3-0.4: 0.0058
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0036
  - 0.6-0.7: 0.0031
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0027
  - 0.9-1.0: 0.0020

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0217, best 0.0134 @ ep 5
  - `Generation/KID_mean_train`: last 0.0220, best 0.0124 @ ep 5
  - `Generation/KID_std_val`: last 0.0030, best 0.0021 @ ep 36
  - `Generation/KID_std_train`: last 0.0051, best 0.0020 @ ep 45
  - `Generation/CMMD_val`: last 0.1677, best 0.1389 @ ep 5
  - `Generation/CMMD_train`: last 0.1612, best 0.1353 @ ep 78
  - `Generation/extended_KID_mean_val`: last 0.0170, best 0.0123 @ ep 49
  - `Generation/extended_KID_mean_train`: last 0.0145, best 0.0118 @ ep 49
  - `Generation/extended_CMMD_val`: last 0.1377, best 0.1246 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.1296, best 0.1191 @ ep 24

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4384 (min 0.4046, max 0.7296)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9182 (min 0.9103, max 0.9462)
  - `Validation/MS-SSIM_bravo`: last 0.9270 (min 0.9057, max 0.9471)
  - `Validation/PSNR_bravo`: last 31.1907 (min 29.8702, max 32.5850)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5371
  - `Generation_Diversity/extended_MSSSIM`: 0.1379

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009067
  - `regional_bravo/large`: 0.0161
  - `regional_bravo/medium`: 0.0160
  - `regional_bravo/small`: 0.0121
  - `regional_bravo/tiny`: 0.0092
  - `regional_bravo/tumor_bg_ratio`: 15.3444
  - `regional_bravo/tumor_loss`: 0.0139

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 8.328e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0138, max 0.0436 @ ep 0
  - `training/grad_norm_max`: last 0.0690, max 0.7796 @ ep 96

#### `exp35_2_scoreaug_structure_20260421-105310`
*started 2026-04-21 10:53 • 173 epochs • 11h49m • 5542.3 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0023 → 0.0020 (min 0.0013 @ ep 54)
  - `Loss/MSE_val`: 0.0061 → 0.0083 (min 0.0033 @ ep 119)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0221
  - 0.1-0.2: 0.0183
  - 0.2-0.3: 0.0136
  - 0.3-0.4: 0.0078
  - 0.4-0.5: 0.0050
  - 0.5-0.6: 0.0040
  - 0.6-0.7: 0.0040
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0021
  - 0.9-1.0: 0.0017

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0224, best 0.0162 @ ep 110
  - `Generation/KID_mean_train`: last 0.0240, best 0.0140 @ ep 72
  - `Generation/KID_std_val`: last 0.0042, best 0.0025 @ ep 86
  - `Generation/KID_std_train`: last 0.0052, best 0.0024 @ ep 6
  - `Generation/CMMD_val`: last 0.1719, best 0.1230 @ ep 107
  - `Generation/CMMD_train`: last 0.1583, best 0.1196 @ ep 107
  - `Generation/extended_KID_mean_val`: last 0.0157, best 0.0112 @ ep 24
  - `Generation/extended_KID_mean_train`: last 0.0113, best 0.0100 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.1183, best 0.1075 @ ep 74
  - `Generation/extended_CMMD_train`: last 0.1156, best 0.1042 @ ep 74

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4225 (min 0.4210, max 0.5507)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9090 (min 0.9070, max 0.9193)
  - `Validation/MS-SSIM_bravo`: last 0.9200 (min 0.9038, max 0.9302)
  - `Validation/PSNR_bravo`: last 30.5678 (min 29.6363, max 31.3955)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2994
  - `Generation_Diversity/extended_MSSSIM`: 0.1419

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009661
  - `regional_bravo/large`: 0.0140
  - `regional_bravo/medium`: 0.0147
  - `regional_bravo/small`: 0.0133
  - `regional_bravo/tiny`: 0.0108
  - `regional_bravo/tumor_bg_ratio`: 13.8933
  - `regional_bravo/tumor_loss`: 0.0134

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 7.758e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0134, max 0.0231 @ ep 134
  - `training/grad_norm_max`: last 0.0542, max 0.9023 @ ep 134

#### `exp35_3_scoreaug_uniform_20260421-115605`
*started 2026-04-21 11:56 • 109 epochs • 11h46m • 3492.0 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0061 → 0.0029 (min 0.0019 @ ep 70)
  - `Loss/MSE_val`: 0.0066 → 0.0032 (min 0.0026 @ ep 88)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0502
  - 0.1-0.2: 0.0081
  - 0.2-0.3: 0.0050
  - 0.3-0.4: 0.0048
  - 0.4-0.5: 0.0033
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0019
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0460, best 0.0158 @ ep 54
  - `Generation/KID_mean_train`: last 0.0448, best 0.0147 @ ep 3
  - `Generation/KID_std_val`: last 0.0068, best 0.0024 @ ep 97
  - `Generation/KID_std_train`: last 0.0086, best 0.0027 @ ep 63
  - `Generation/CMMD_val`: last 0.2778, best 0.1364 @ ep 98
  - `Generation/CMMD_train`: last 0.2666, best 0.1333 @ ep 100
  - `Generation/extended_KID_mean_val`: last 0.0165, best 0.0163 @ ep 49
  - `Generation/extended_KID_mean_train`: last 0.0173, best 0.0146 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.1357, best 0.1312 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.1347, best 0.1254 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5358 (min 0.3914, max 0.5932)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9461 (min 0.9157, max 0.9515)
  - `Validation/MS-SSIM_bravo`: last 0.9414 (min 0.9225, max 0.9567)
  - `Validation/PSNR_bravo`: last 31.8717 (min 30.7555, max 33.8884)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4899
  - `Generation_Diversity/extended_MSSSIM`: 0.2536

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007016
  - `regional_bravo/large`: 0.0082
  - `regional_bravo/medium`: 0.0134
  - `regional_bravo/small`: 0.0113
  - `regional_bravo/tiny`: 0.0083
  - `regional_bravo/tumor_bg_ratio`: 14.6010
  - `regional_bravo/tumor_loss`: 0.0102

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 9.124e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0154, max 0.0799 @ ep 0
  - `training/grad_norm_max`: last 0.0555, max 1.8756 @ ep 10

### exp36

**exp36 (augmentation fine-tunes)** — medium / MRI-specific augmentations
at uniform vs detail schedules. Bypassed in favor of exp37's t-weighted
approach after mean-blur diagnosis showed "low-t specialist" direction
was backwards.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp36_2_augment_medium_detail_20260421-031742` — 0.0105
  2. 🥈 `exp36_4_augment_mri_detail_20260421-042913` — 0.0129
  3.  `exp36_3_augment_mri_uniform_20260421-120440` — 0.0257
  4.  `exp36_1_augment_medium_uniform_20260421-004136` — 0.0264

#### `exp36_1_augment_medium_uniform_20260421-004136`
*started 2026-04-21 00:41 • 280 epochs • 24h02m • 8970.2 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0055 → 0.0024 (min 0.0018 @ ep 145)
  - `Loss/MSE_val`: 0.0046 → 0.0028 (min 0.0019 @ ep 242)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0239
  - 0.1-0.2: 0.0061
  - 0.2-0.3: 0.0047
  - 0.3-0.4: 0.0033
  - 0.4-0.5: 0.0024
  - 0.5-0.6: 0.0018
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0011
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0617, best 0.0239 @ ep 27
  - `Generation/KID_mean_train`: last 0.0585, best 0.0235 @ ep 64
  - `Generation/KID_std_val`: last 0.0072, best 0.0030 @ ep 30
  - `Generation/KID_std_train`: last 0.0059, best 0.0030 @ ep 207
  - `Generation/CMMD_val`: last 0.3698, best 0.1858 @ ep 129
  - `Generation/CMMD_train`: last 0.3456, best 0.1738 @ ep 129
  - `Generation/extended_KID_mean_val`: last 0.0454, best 0.0264 @ ep 99
  - `Generation/extended_KID_mean_train`: last 0.0418, best 0.0238 @ ep 99
  - `Generation/extended_CMMD_val`: last 0.1972, best 0.1716 @ ep 99
  - `Generation/extended_CMMD_train`: last 0.1854, best 0.1624 @ ep 99

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5308 (min 0.4372, max 0.6399)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9604 (min 0.9265, max 0.9614)
  - `Validation/MS-SSIM_bravo`: last 0.9583 (min 0.9312, max 0.9692)
  - `Validation/PSNR_bravo`: last 33.3066 (min 31.5622, max 34.2781)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4944
  - `Generation_Diversity/extended_MSSSIM`: 0.2023

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005326
  - `regional_bravo/large`: 0.0085
  - `regional_bravo/medium`: 0.0053
  - `regional_bravo/small`: 0.0088
  - `regional_bravo/tiny`: 0.0074
  - `regional_bravo/tumor_bg_ratio`: 13.9891
  - `regional_bravo/tumor_loss`: 0.0075

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 4.782e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0190, max 0.0817 @ ep 0
  - `training/grad_norm_max`: last 0.1048, max 0.9789 @ ep 63

#### `exp36_2_augment_medium_detail_20260421-031742`
*started 2026-04-21 03:17 • 303 epochs • 21h28m • 9707.0 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0037 → 0.0017 (min 0.0012 @ ep 220)
  - `Loss/MSE_val`: 0.0062 → 0.0047 (min 0.0030 @ ep 186)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0155
  - 0.1-0.2: 0.0111
  - 0.2-0.3: 0.0034
  - 0.3-0.4: 0.0038
  - 0.4-0.5: 0.0040
  - 0.5-0.6: 0.0041
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0023
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0395, best 0.0175 @ ep 281
  - `Generation/KID_mean_train`: last 0.0389, best 0.0183 @ ep 99
  - `Generation/KID_std_val`: last 0.0048, best 0.0023 @ ep 25
  - `Generation/KID_std_train`: last 0.0039, best 0.0024 @ ep 100
  - `Generation/CMMD_val`: last 0.2051, best 0.1417 @ ep 176
  - `Generation/CMMD_train`: last 0.1887, best 0.1351 @ ep 204
  - `Generation/extended_KID_mean_val`: last 0.0149, best 0.0105 @ ep 149
  - `Generation/extended_KID_mean_train`: last 0.0120, best 0.0101 @ ep 149
  - `Generation/extended_CMMD_val`: last 0.1319, best 0.1224 @ ep 149
  - `Generation/extended_CMMD_train`: last 0.1237, best 0.1203 @ ep 149

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4784 (min 0.3929, max 0.6268)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9195 (min 0.9070, max 0.9427)
  - `Validation/MS-SSIM_bravo`: last 0.9307 (min 0.9008, max 0.9496)
  - `Validation/PSNR_bravo`: last 31.9141 (min 29.6503, max 33.0126)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2564
  - `Generation_Diversity/extended_MSSSIM`: 0.1293

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0008521
  - `regional_bravo/large`: 0.0090
  - `regional_bravo/medium`: 0.0144
  - `regional_bravo/small`: 0.0104
  - `regional_bravo/tiny`: 0.0045
  - `regional_bravo/tumor_bg_ratio`: 11.5667
  - `regional_bravo/tumor_loss`: 0.0099

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 4.137e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0106, max 0.0562 @ ep 0
  - `training/grad_norm_max`: last 0.0357, max 0.9345 @ ep 207

#### `exp36_4_augment_mri_detail_20260421-042913`
*started 2026-04-21 04:29 • 236 epochs • 20h16m • 7560.6 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0031 → 0.0021 (min 0.0014 @ ep 232)
  - `Loss/MSE_val`: 0.0035 → 0.0043 (min 0.0029 @ ep 124)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0137
  - 0.1-0.2: 0.0086
  - 0.2-0.3: 0.0053
  - 0.3-0.4: 0.0059
  - 0.4-0.5: 0.0061
  - 0.5-0.6: 0.0036
  - 0.6-0.7: 0.0028
  - 0.7-0.8: 0.0029
  - 0.8-0.9: 0.0020
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0261, best 0.0169 @ ep 214
  - `Generation/KID_mean_train`: last 0.0302, best 0.0152 @ ep 79
  - `Generation/KID_std_val`: last 0.0058, best 0.0021 @ ep 79
  - `Generation/KID_std_train`: last 0.0054, best 0.0023 @ ep 67
  - `Generation/CMMD_val`: last 0.1822, best 0.1541 @ ep 183
  - `Generation/CMMD_train`: last 0.1803, best 0.1448 @ ep 183
  - `Generation/extended_KID_mean_val`: last 0.0170, best 0.0129 @ ep 174
  - `Generation/extended_KID_mean_train`: last 0.0152, best 0.0098 @ ep 99
  - `Generation/extended_CMMD_val`: last 0.1492, best 0.1219 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.1441, best 0.1124 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4677 (min 0.3783, max 0.6113)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9079 (min 0.9079, max 0.9434)
  - `Validation/MS-SSIM_bravo`: last 0.9138 (min 0.9070, max 0.9532)
  - `Validation/PSNR_bravo`: last 30.3115 (min 29.9320, max 33.1413)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2684
  - `Generation_Diversity/extended_MSSSIM`: 0.1486

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0010
  - `regional_bravo/large`: 0.0168
  - `regional_bravo/medium`: 0.0146
  - `regional_bravo/small`: 0.0117
  - `regional_bravo/tiny`: 0.0059
  - `regional_bravo/tumor_bg_ratio`: 12.7616
  - `regional_bravo/tumor_loss`: 0.0131

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 6.047e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0146, max 0.0474 @ ep 0
  - `training/grad_norm_max`: last 0.0857, max 1.3566 @ ep 167

#### `exp36_3_augment_mri_uniform_20260421-120440`
*started 2026-04-21 12:04 • 109 epochs • 11h47m • 3492.0 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0056 → 0.0026 (min 0.0022 @ ep 68)
  - `Loss/MSE_val`: 0.0046 → 0.0026 (min 0.0022 @ ep 105)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0241
  - 0.1-0.2: 0.0073
  - 0.2-0.3: 0.0067
  - 0.3-0.4: 0.0030
  - 0.4-0.5: 0.0024
  - 0.5-0.6: 0.0023
  - 0.6-0.7: 0.0015
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0496, best 0.0183 @ ep 69
  - `Generation/KID_mean_train`: last 0.0474, best 0.0179 @ ep 69
  - `Generation/KID_std_val`: last 0.0039, best 0.0029 @ ep 74
  - `Generation/KID_std_train`: last 0.0048, best 0.0031 @ ep 69
  - `Generation/CMMD_val`: last 0.3118, best 0.1727 @ ep 69
  - `Generation/CMMD_train`: last 0.2964, best 0.1575 @ ep 69
  - `Generation/extended_KID_mean_val`: last 0.0318, best 0.0257 @ ep 74
  - `Generation/extended_KID_mean_train`: last 0.0313, best 0.0232 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.1846, best 0.1846 @ ep 99
  - `Generation/extended_CMMD_train`: last 0.1740, best 0.1740 @ ep 99

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5550 (min 0.4481, max 0.6396)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9593 (min 0.9239, max 0.9599)
  - `Validation/MS-SSIM_bravo`: last 0.9539 (min 0.9146, max 0.9656)
  - `Validation/PSNR_bravo`: last 32.8296 (min 31.0466, max 34.1209)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2555
  - `Generation_Diversity/extended_MSSSIM`: 0.1480

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005754
  - `regional_bravo/large`: 0.0103
  - `regional_bravo/medium`: 0.0096
  - `regional_bravo/small`: 0.0086
  - `regional_bravo/tiny`: 0.0068
  - `regional_bravo/tumor_bg_ratio`: 15.7753
  - `regional_bravo/tumor_loss`: 0.0091

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 9, final 9.124e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0184, max 0.0873 @ ep 0
  - `training/grad_norm_max`: last 0.1113, max 0.6896 @ ep 0

### exp37

**exp37 (t-weighted perceptual / FFL fine-tunes)** — designed after mean-blur
diagnostic. HIGH-t LPIPS+FFL schedule (opposite of exp32_2). Three variants:
exp37_1 LPIPS-only high-t, exp37_2 LPIPS+FFL high-t (500ep, regressed),
exp37_3 LPIPS+FFL short (150ep, heavier weights).

Per memory/project_phase1_spectrum_finding.md: **exp37_3 is decisively best
at mid-to-high frequencies** — +27% over baseline at high band, +11% at
very_high. Only model reaching ≥90% of real at the high band. Vessel-scale
deficit (very_high, 0.40–0.50 cycles/pixel) remains severe: 63% of real
even for exp37_3; 37% gap unsolved.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp37_1_pixel_bravo_lpips_hight_20260418-002202` — 0.0192
  2. 🥈 `exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105` — 0.0223
  3.  `exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042` — 0.0247

#### `exp37_1_pixel_bravo_lpips_hight_20260418-002202`
*started 2026-04-18 00:22 • 500 epochs • 53h00m • 16018.2 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0021 → 0.0018 (min 0.0016 @ ep 5)
  - `Loss/MSE_val`: 0.0082 → 0.0052 (min 0.0035 @ ep 150)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0279
  - 0.1-0.2: 0.0213
  - 0.2-0.3: 0.0136
  - 0.3-0.4: 0.0058
  - 0.4-0.5: 0.0056
  - 0.5-0.6: 0.0040
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0023
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0339, best 0.0168 @ ep 315
  - `Generation/KID_mean_train`: last 0.0313, best 0.0148 @ ep 315
  - `Generation/KID_std_val`: last 0.0034, best 0.0021 @ ep 321
  - `Generation/KID_std_train`: last 0.0029, best 0.0023 @ ep 319
  - `Generation/CMMD_val`: last 0.1265, best 0.1021 @ ep 294
  - `Generation/CMMD_train`: last 0.1149, best 0.0922 @ ep 294
  - `Generation/extended_KID_mean_val`: last 0.0278, best 0.0192 @ ep 324
  - `Generation/extended_KID_mean_train`: last 0.0241, best 0.0166 @ ep 324
  - `Generation/extended_CMMD_val`: last 0.1242, best 0.1103 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1118, best 0.0989 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4732 (min 0, max 0.7262)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9118 (min 0.8974, max 0.9201)
  - `Validation/MS-SSIM_bravo`: last 0.9113 (min 0.9002, max 0.9341)
  - `Validation/PSNR_bravo`: last 29.9530 (min 29.4375, max 31.6465)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1945
  - `Generation_Diversity/extended_MSSSIM`: 0.1152

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0011
  - `regional_bravo/large`: 0.0127
  - `regional_bravo/medium`: 0.0170
  - `regional_bravo/small`: 0.0140
  - `regional_bravo/tiny`: 0.0084
  - `regional_bravo/tumor_bg_ratio`: 12.5391
  - `regional_bravo/tumor_loss`: 0.0133

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 14, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 52.2288, max 98.5546 @ ep 6
  - `training/grad_norm_max`: last 241.6709, max 400.9643 @ ep 32

#### `exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105`
*started 2026-04-18 00:51 • 500 epochs • 57h34m • 16018.2 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0022 → 0.0020 (min 0.0018 @ ep 2)
  - `Loss/MSE_val`: 0.0093 → 0.0061 (min 0.0037 @ ep 77)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0238
  - 0.1-0.2: 0.0117
  - 0.2-0.3: 0.0110
  - 0.3-0.4: 0.0087
  - 0.4-0.5: 0.0047
  - 0.5-0.6: 0.0041
  - 0.6-0.7: 0.0029
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0026
  - 0.9-1.0: 0.0014

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0329, best 0.0154 @ ep 282
  - `Generation/KID_mean_train`: last 0.0305, best 0.0130 @ ep 282
  - `Generation/KID_std_val`: last 0.0040, best 0.0022 @ ep 355
  - `Generation/KID_std_train`: last 0.0036, best 0.0019 @ ep 282
  - `Generation/CMMD_val`: last 0.1510, best 0.0990 @ ep 377
  - `Generation/CMMD_train`: last 0.1373, best 0.0890 @ ep 377
  - `Generation/extended_KID_mean_val`: last 0.0256, best 0.0223 @ ep 224
  - `Generation/extended_KID_mean_train`: last 0.0219, best 0.0191 @ ep 224
  - `Generation/extended_CMMD_val`: last 0.1213, best 0.1152 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.1137, best 0.1064 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4627 (min 0, max 0.6545)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9116 (min 0.8975, max 0.9190)
  - `Validation/MS-SSIM_bravo`: last 0.9145 (min 0.9044, max 0.9398)
  - `Validation/PSNR_bravo`: last 30.1379 (min 29.5964, max 31.9022)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3028
  - `Generation_Diversity/extended_MSSSIM`: 0.0895

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0010
  - `regional_bravo/large`: 0.0131
  - `regional_bravo/medium`: 0.0129
  - `regional_bravo/small`: 0.0155
  - `regional_bravo/tiny`: 0.0114
  - `regional_bravo/tumor_bg_ratio`: 12.7464
  - `regional_bravo/tumor_loss`: 0.0132

**LR schedule:**
  - `LR/Generator`: peak 1e-05 @ ep 14, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 56.4461, max 92.4927 @ ep 1
  - `training/grad_norm_max`: last 204.7357, max 361.0537 @ ep 270

#### `exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042`
*started 2026-04-20 01:00 • 150 epochs • 10h58m • 4805.5 TFLOPs • peak VRAM 42.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0019 → 0.0031 (min 0.0018 @ ep 7)
  - `Loss/MSE_val`: 0.0076 → 0.0053 (min 0.0034 @ ep 86)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0245
  - 0.1-0.2: 0.0116
  - 0.2-0.3: 0.0119
  - 0.3-0.4: 0.0075
  - 0.4-0.5: 0.0046
  - 0.5-0.6: 0.0038
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0018
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0012

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0348, best 0.0169 @ ep 0
  - `Generation/KID_mean_train`: last 0.0309, best 0.0148 @ ep 0
  - `Generation/KID_std_val`: last 0.0041, best 0.0028 @ ep 77
  - `Generation/KID_std_train`: last 0.0040, best 0.0025 @ ep 126
  - `Generation/CMMD_val`: last 0.1611, best 0.1079 @ ep 2
  - `Generation/CMMD_train`: last 0.1445, best 0.0896 @ ep 2
  - `Generation/extended_KID_mean_val`: last 0.0393, best 0.0247 @ ep 97
  - `Generation/extended_KID_mean_train`: last 0.0339, best 0.0217 @ ep 97
  - `Generation/extended_CMMD_val`: last 0.1572, best 0.1496 @ ep 132
  - `Generation/extended_CMMD_train`: last 0.1429, best 0.1343 @ ep 132

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5599 (min 0.2890, max 0.7284)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9179 (min 0.8976, max 0.9255)
  - `Validation/MS-SSIM_bravo`: last 0.9240 (min 0.9006, max 0.9410)
  - `Validation/PSNR_bravo`: last 30.5609 (min 29.4284, max 31.9421)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1634
  - `Generation_Diversity/extended_MSSSIM`: 0.1014

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009106
  - `regional_bravo/large`: 0.0119
  - `regional_bravo/medium`: 0.0128
  - `regional_bravo/small`: 0.0146
  - `regional_bravo/tiny`: 0.0103
  - `regional_bravo/tumor_bg_ratio`: 13.5171
  - `regional_bravo/tumor_loss`: 0.0123

**LR schedule:**
  - `LR/Generator`: peak 2e-05 @ ep 9, final 1e-07

**Training meta:**
  - `training/grad_norm_avg`: last 44.0141, max 121.2567 @ ep 6
  - `training/grad_norm_max`: last 267.5623, max 577.5291 @ ep 43

---separator---
## diffusion_3d/bravo_latent

*17 runs across 6 experiment families.*

### exp9

**exp9** — first latent-diffusion bravo tests; LDM-4x (exp9_1) and LDM-8x
(exp9). Uses a 3D VAE compression model as encoder-decoder.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp9_ldm_8x_bravo_rflow_256x160_20260223-012214` — 0.1842
  2. 🥈 `exp9_1_ldm_4x_bravo_20260222-225625` — 0.1981

#### `exp9_1_ldm_4x_bravo_20260222-225625`
*started 2026-02-22 22:56 • 500 epochs • 27h16m • 17863.7 TFLOPs • peak VRAM 80.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.7077 → 0.1683 (min 0.1531 @ ep 498)
  - `Loss/MSE_val`: 1.5981 → 0.8466 (min 0.2501 @ ep 153)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.1872
  - 0.1-0.2: 0.9051
  - 0.2-0.3: 0.7129
  - 0.3-0.4: 0.5985
  - 0.4-0.5: 0.5018
  - 0.5-0.6: 0.4086
  - 0.6-0.7: 0.5259
  - 0.7-0.8: 0.4994
  - 0.8-0.9: 0.6924
  - 0.9-1.0: 0.9925

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2169, best 0.2114 @ ep 333
  - `Generation/KID_mean_train`: last 0.2105, best 0.2077 @ ep 333
  - `Generation/KID_std_val`: last 0.0275, best 0.0192 @ ep 85
  - `Generation/KID_std_train`: last 0.0355, best 0.0180 @ ep 168
  - `Generation/CMMD_val`: last 0.5691, best 0.4893 @ ep 416
  - `Generation/CMMD_train`: last 0.5469, best 0.4678 @ ep 416
  - `Generation/extended_KID_mean_val`: last 0.1981, best 0.1981 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.1948, best 0.1947 @ ep 224
  - `Generation/extended_CMMD_val`: last 0.5162, best 0.4928 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.5087, best 0.4846 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3847 (min 0.3136, max 0.4198)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7621 (min 0.7488, max 0.9939)
  - `Validation/MS-SSIM_bravo`: last 0.7630 (min 0.7371, max 0.8174)
  - `Validation/PSNR_bravo`: last 25.8852 (min 25.1618, max 28.0961)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.9034
  - `Generation_Diversity/extended_MSSSIM`: 0.2778

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.3416, max 5.3433 @ ep 46
  - `training/grad_norm_max`: last 0.7609, max 25.9917 @ ep 46

#### `exp9_ldm_8x_bravo_rflow_256x160_20260223-012214`
*started 2026-02-23 01:22 • 500 epochs • 38h10m • 68031.8 TFLOPs • peak VRAM 72.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.5294 → 1.4573 (min 0.3684 @ ep 33)
  - `Loss/MSE_val`: 1.3358 → 1.4869 (min 0.3827 @ ep 32)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.4677
  - 0.1-0.2: 1.5074
  - 0.2-0.3: 1.3214
  - 0.3-0.4: 1.5518
  - 0.4-0.5: 1.6578
  - 0.5-0.6: 1.3538
  - 0.6-0.7: 1.5593
  - 0.7-0.8: 1.5288
  - 0.8-0.9: 1.6178

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2103, best 0.1812 @ ep 313
  - `Generation/KID_mean_train`: last 0.2121, best 0.1723 @ ep 441
  - `Generation/KID_std_val`: last 0.0288, best 0.0145 @ ep 111
  - `Generation/KID_std_train`: last 0.0245, best 0.0152 @ ep 252
  - `Generation/CMMD_val`: last 0.5054, best 0.4349 @ ep 441
  - `Generation/CMMD_train`: last 0.4881, best 0.4144 @ ep 441
  - `Generation/extended_KID_mean_val`: last 0.2292, best 0.1842 @ ep 349
  - `Generation/extended_KID_mean_train`: last 0.2268, best 0.1739 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.4864, best 0.4509 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.4806, best 0.4471 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3606 (min 0.3069, max 0.3976)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9669 (min 0.9161, max 0.9799)
  - `Validation/MS-SSIM_bravo`: last 0.7785 (min 0.7455, max 0.8196)
  - `Validation/PSNR_bravo`: last 26.1916 (min 25.5248, max 27.6407)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.1324
  - `Generation_Diversity/extended_MSSSIM`: 0.3600

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.9009, max 13.4524 @ ep 4
  - `training/grad_norm_max`: last 1.7765, max 52.9279 @ ep 40

### exp13

**exp13 (DiT latent)** — DiT on VAE latents with/without normalization.
exp13 = DiT-4x/8x with norm; exp13_1 = DiT on VAE latents with/without norm.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp13_dit_4x_bravo_nonorm_20260225-034202` — 0.0440
  2. 🥈 `exp13_dit_8x_bravo_20260223-042538` — 0.0536
  3.  `exp13_dit_4x_bravo_20260222-230427` — 0.0747
  4.  `exp13_1_dit_4x_vae_bravo_20260226-032147` — 0.0879
  5.  `exp13_1_dit_4x_vae_bravo_nonorm_20260226-021643` — 0.1760

#### `exp13_dit_4x_bravo_20260222-230427`
*started 2026-02-22 23:04 • 500 epochs • 8h01m • 137158.3 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.9738 → 0.2820 (min 0.2733 @ ep 366)
  - `Loss/MSE_val`: 1.9153 → 0.2935 (min 0.2742 @ ep 393)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2688
  - 0.1-0.2: 0.2537
  - 0.2-0.3: 0.2861
  - 0.3-0.4: 0.3216
  - 0.4-0.5: 0.2976
  - 0.5-0.6: 0.4921
  - 0.6-0.7: 0.4252
  - 0.7-0.8: 0.7723
  - 0.8-0.9: 0.6446

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0979, best 0.0812 @ ep 451
  - `Generation/KID_mean_train`: last 0.0914, best 0.0755 @ ep 451
  - `Generation/KID_std_val`: last 0.0159, best 0.0092 @ ep 258
  - `Generation/KID_std_train`: last 0.0193, best 0.0089 @ ep 285
  - `Generation/CMMD_val`: last 0.4115, best 0.3540 @ ep 257
  - `Generation/CMMD_train`: last 0.3860, best 0.3301 @ ep 257
  - `Generation/extended_KID_mean_val`: last 0.0780, best 0.0747 @ ep 349
  - `Generation/extended_KID_mean_train`: last 0.0755, best 0.0678 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.3748, best 0.3399 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.3544, best 0.3226 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3749 (min 0.3110, max 0.4108)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9927 (min 0.9110, max 0.9927)
  - `Validation/MS-SSIM_bravo`: last 0.7692 (min 0.7390, max 0.8172)
  - `Validation/PSNR_bravo`: last 25.8662 (min 25.2501, max 28.0550)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2079
  - `Generation_Diversity/extended_MSSSIM`: 0.1357

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0819, max 5.4113 @ ep 5
  - `training/grad_norm_max`: last 0.1925, max 18.5494 @ ep 7

#### `exp13_dit_8x_bravo_20260223-042538`
*started 2026-02-23 04:25 • 500 epochs • 5h47m • 137088.9 TFLOPs • peak VRAM 7.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.9485 → 0.3559 (min 0.3458 @ ep 479)
  - `Loss/MSE_val`: 1.8541 → 0.4045 (min 0.3811 @ ep 255)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.4205
  - 0.1-0.2: 0.3919
  - 0.2-0.3: 0.4029
  - 0.3-0.4: 0.3636
  - 0.4-0.5: 0.5036
  - 0.5-0.6: 0.4314
  - 0.6-0.7: 0.5099
  - 0.7-0.8: 0.7102
  - 0.8-0.9: 0.8350

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0660, best 0.0585 @ ep 235
  - `Generation/KID_mean_train`: last 0.0611, best 0.0579 @ ep 235
  - `Generation/KID_std_val`: last 0.0086, best 0.0070 @ ep 462
  - `Generation/KID_std_train`: last 0.0090, best 0.0065 @ ep 250
  - `Generation/CMMD_val`: last 0.3359, best 0.2662 @ ep 270
  - `Generation/CMMD_train`: last 0.3149, best 0.2500 @ ep 270
  - `Generation/extended_KID_mean_val`: last 0.0625, best 0.0536 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0580, best 0.0513 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.2968, best 0.2605 @ ep 299
  - `Generation/extended_CMMD_train`: last 0.2810, best 0.2457 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3625 (min 0.2993, max 0.3965)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9820 (min 0.9159, max 0.9822)
  - `Validation/MS-SSIM_bravo`: last 0.7710 (min 0.7413, max 0.8180)
  - `Validation/PSNR_bravo`: last 25.9191 (min 25.3360, max 27.5827)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3167
  - `Generation_Diversity/extended_MSSSIM`: 0.1915

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.2373, max 7.6082 @ ep 3
  - `training/grad_norm_max`: last 0.5074, max 38.0843 @ ep 5

#### `exp13_dit_4x_bravo_nonorm_20260225-034202`
*started 2026-02-25 03:42 • 500 epochs • 8h44m • 137158.3 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0306 → 0.1009 (min 0.0881 @ ep 497)
  - `Loss/MSE_val`: 1.0285 → 0.1085 (min 0.0770 @ ep 228)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1631
  - 0.1-0.2: 0.1284
  - 0.2-0.3: 0.0907
  - 0.3-0.4: 0.0569
  - 0.4-0.5: 0.0628
  - 0.5-0.6: 0.0443
  - 0.6-0.7: 0.0251
  - 0.7-0.8: 0.0385
  - 0.8-0.9: 0.0344
  - 0.9-1.0: 0.0270

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0563, best 0.0466 @ ep 260
  - `Generation/KID_mean_train`: last 0.0536, best 0.0402 @ ep 420
  - `Generation/KID_std_val`: last 0.0111, best 0.0059 @ ep 403
  - `Generation/KID_std_train`: last 0.0118, best 0.0061 @ ep 361
  - `Generation/CMMD_val`: last 0.3542, best 0.2927 @ ep 414
  - `Generation/CMMD_train`: last 0.3321, best 0.2712 @ ep 414
  - `Generation/extended_KID_mean_val`: last 0.0536, best 0.0440 @ ep 424
  - `Generation/extended_KID_mean_train`: last 0.0483, best 0.0396 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.3006, best 0.2802 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.2811, best 0.2588 @ ep 324

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4185 (min 0.3700, max 0.8598)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7726 (min 0.3890, max 0.8225)
  - `Validation/MS-SSIM_bravo`: last 0.7734 (min 0.7003, max 0.8174)
  - `Validation/PSNR_bravo`: last 26.0290 (min 25.3428, max 27.7361)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2330
  - `Generation_Diversity/extended_MSSSIM`: 0.1426

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0428, max 4.5637 @ ep 2
  - `training/grad_norm_max`: last 0.0704, max 17.6858 @ ep 2

#### `exp13_1_dit_4x_vae_bravo_nonorm_20260226-021643`
*started 2026-02-26 02:16 • 500 epochs • 10h32m • 137158.3 TFLOPs • peak VRAM 17.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 3.0789 → 0.1797 (min 0.1696 @ ep 370)
  - `Loss/MSE_val`: 3.0093 → 0.1815 (min 0.1672 @ ep 474)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1664
  - 0.1-0.2: 0.1550
  - 0.2-0.3: 0.1833
  - 0.3-0.4: 0.1953
  - 0.4-0.5: 0.2640
  - 0.5-0.6: 0.2878
  - 0.6-0.7: 0.4424
  - 0.7-0.8: 0.4538
  - 0.8-0.9: 0.5439
  - 0.9-1.0: 0.8668

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2472, best 0.1725 @ ep 126
  - `Generation/KID_mean_train`: last 0.2477, best 0.1693 @ ep 126
  - `Generation/KID_std_val`: last 0.0168, best 0.0137 @ ep 188
  - `Generation/KID_std_train`: last 0.0216, best 0.0146 @ ep 163
  - `Generation/CMMD_val`: last 0.4949, best 0.4198 @ ep 291
  - `Generation/CMMD_train`: last 0.4759, best 0.3973 @ ep 291
  - `Generation/extended_KID_mean_val`: last 0.1760, best 0.1760 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.1716, best 0.1716 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.4084, best 0.3892 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.3964, best 0.3774 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3804 (min 0.3235, max 0.4647)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7592 (min 0.7450, max 0.8158)
  - `Validation/MS-SSIM_bravo`: last 0.7697 (min 0.7397, max 0.8182)
  - `Validation/PSNR_bravo`: last 25.8615 (min 25.2410, max 27.7710)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4579
  - `Generation_Diversity/extended_MSSSIM`: 0.2424

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1229, max 4.7114 @ ep 13
  - `training/grad_norm_max`: last 0.3917, max 18.9283 @ ep 14

#### `exp13_1_dit_4x_vae_bravo_20260226-032147`
*started 2026-02-26 03:21 • 500 epochs • 9h39m • 137158.3 TFLOPs • peak VRAM 17.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.9279 → 0.1503 (min 0.1469 @ ep 454)
  - `Loss/MSE_val`: 1.7624 → 0.1593 (min 0.1485 @ ep 413)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1530
  - 0.1-0.2: 0.1413
  - 0.2-0.3: 0.1529
  - 0.3-0.4: 0.1613
  - 0.4-0.5: 0.1949
  - 0.5-0.6: 0.2121
  - 0.6-0.7: 0.2515
  - 0.7-0.8: 0.2630
  - 0.8-0.9: 0.2674

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1497, best 0.0870 @ ep 319
  - `Generation/KID_mean_train`: last 0.1418, best 0.0823 @ ep 319
  - `Generation/KID_std_val`: last 0.0216, best 0.0092 @ ep 298
  - `Generation/KID_std_train`: last 0.0180, best 0.0094 @ ep 274
  - `Generation/CMMD_val`: last 0.3035, best 0.2763 @ ep 464
  - `Generation/CMMD_train`: last 0.2871, best 0.2614 @ ep 464
  - `Generation/extended_KID_mean_val`: last 0.1044, best 0.0879 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0943, best 0.0844 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2599, best 0.2599 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.2438, best 0.2438 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3852 (min 0.3227, max 0.6148)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7606 (min 0.7478, max 0.8161)
  - `Validation/MS-SSIM_bravo`: last 0.7704 (min 0.7414, max 0.8179)
  - `Validation/PSNR_bravo`: last 25.9145 (min 25.3198, max 27.7545)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2894
  - `Generation_Diversity/extended_MSSSIM`: 0.1722

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1010, max 3.1123 @ ep 9
  - `training/grad_norm_max`: last 0.3332, max 14.3794 @ ep 11

### exp21

**exp21 (LDM UNet sizing)** — UNet architectures on 4x latents: large,
MAISI-style, small. Per memory: exp21_2 (MAISI UNet 167M, LDM 4x) reached
FID 50.89 in-training.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp21_2_ldm_4x_unet_maisi_20260304-083519` — 0.0377
  2. 🥈 `exp21_3_ldm_4x_unet_small_20260304-091058` — 0.0506
  3.  `exp21_1_ldm_4x_unet_large_20260304-061012` — 0.0784

#### `exp21_1_ldm_4x_unet_large_20260304-061012`
*started 2026-03-04 06:10 • 175 epochs • 6h06m • 6252.3 TFLOPs • peak VRAM 80.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9306 → 0.0582 (min 0.0547 @ ep 167)
  - `Loss/MSE_val`: 0.8083 → 0.1220 (min 0.0767 @ ep 85)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2096
  - 0.1-0.2: 0.1592
  - 0.2-0.3: 0.0797
  - 0.3-0.4: 0.0754
  - 0.4-0.5: 0.0646
  - 0.5-0.6: 0.0514
  - 0.6-0.7: 0.0360
  - 0.7-0.8: 0.0264
  - 0.8-0.9: 0.0220
  - 0.9-1.0: 0.0284

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2647, best 0.0714 @ ep 121
  - `Generation/KID_mean_train`: last 0.2627, best 0.0695 @ ep 121
  - `Generation/KID_std_val`: last 0.0361, best 0.0069 @ ep 121
  - `Generation/KID_std_train`: last 0.0276, best 0.0059 @ ep 121
  - `Generation/CMMD_val`: last 0.5906, best 0.3299 @ ep 107
  - `Generation/CMMD_train`: last 0.5732, best 0.3164 @ ep 107
  - `Generation/extended_KID_mean_val`: last 0.1321, best 0.0784 @ ep 149
  - `Generation/extended_KID_mean_train`: last 0.1255, best 0.0736 @ ep 124
  - `Generation/extended_CMMD_val`: last 0.3304, best 0.3304 @ ep 174
  - `Generation/extended_CMMD_train`: last 0.3221, best 0.3221 @ ep 174

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4194 (min 0.3766, max 0.9855)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7869 (min 0.4117, max 0.8078)
  - `Validation/MS-SSIM_bravo`: last 0.7757 (min 0.6761, max 0.8153)
  - `Validation/PSNR_bravo`: last 26.1275 (min 25.0093, max 27.6885)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4689
  - `Generation_Diversity/extended_MSSSIM`: 0.2127

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 7.479e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.2221, max 3.1234 @ ep 1
  - `training/grad_norm_max`: last 0.4829, max 24.4173 @ ep 2

#### `exp21_2_ldm_4x_unet_maisi_20260304-083519`
*started 2026-03-04 08:35 • 500 epochs • 10h00m • 7240.2 TFLOPs • peak VRAM 20.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9808 → 0.0374 (min 0.0261 @ ep 492)
  - `Loss/MSE_val`: 0.9304 → 0.4265 (min 0.0767 @ ep 124)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.8030
  - 0.1-0.2: 0.4830
  - 0.2-0.3: 0.2392
  - 0.3-0.4: 0.1098
  - 0.4-0.5: 0.0907
  - 0.5-0.6: 0.0504
  - 0.6-0.7: 0.0382
  - 0.7-0.8: 0.0465
  - 0.8-0.9: 0.0307
  - 0.9-1.0: 0.0298

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0577, best 0.0360 @ ep 165
  - `Generation/KID_mean_train`: last 0.0523, best 0.0329 @ ep 165
  - `Generation/KID_std_val`: last 0.0129, best 0.0043 @ ep 165
  - `Generation/KID_std_train`: last 0.0152, best 0.0046 @ ep 165
  - `Generation/CMMD_val`: last 0.2085, best 0.1653 @ ep 446
  - `Generation/CMMD_train`: last 0.1908, best 0.1556 @ ep 446
  - `Generation/extended_KID_mean_val`: last 0.0497, best 0.0377 @ ep 174
  - `Generation/extended_KID_mean_train`: last 0.0440, best 0.0368 @ ep 174
  - `Generation/extended_CMMD_val`: last 0.1834, best 0.1745 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.1693, best 0.1637 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4043 (min 0.3559, max 0.7589)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7730 (min 0.3979, max 0.8188)
  - `Validation/MS-SSIM_bravo`: last 0.7839 (min 0.7037, max 0.8191)
  - `Validation/PSNR_bravo`: last 26.3034 (min 25.1635, max 27.8424)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3866
  - `Generation_Diversity/extended_MSSSIM`: 0.1897

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1642, max 3.0712 @ ep 4
  - `training/grad_norm_max`: last 0.4663, max 35.9018 @ ep 4

#### `exp21_3_ldm_4x_unet_small_20260304-091058`
*started 2026-03-04 09:10 • 500 epochs • 6h32m • 3197.2 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0089 → 0.0720 (min 0.0609 @ ep 489)
  - `Loss/MSE_val`: 0.9879 → 0.2207 (min 0.0817 @ ep 87)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.3444
  - 0.1-0.2: 0.2764
  - 0.2-0.3: 0.1491
  - 0.3-0.4: 0.1121
  - 0.4-0.5: 0.0543
  - 0.5-0.6: 0.0703
  - 0.6-0.7: 0.0346
  - 0.7-0.8: 0.0269
  - 0.8-0.9: 0.0402
  - 0.9-1.0: 0.0397

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1286, best 0.0406 @ ep 403
  - `Generation/KID_mean_train`: last 0.1223, best 0.0354 @ ep 253
  - `Generation/KID_std_val`: last 0.0199, best 0.0078 @ ep 403
  - `Generation/KID_std_train`: last 0.0201, best 0.0082 @ ep 125
  - `Generation/CMMD_val`: last 0.3925, best 0.1812 @ ep 485
  - `Generation/CMMD_train`: last 0.3740, best 0.1739 @ ep 488
  - `Generation/extended_KID_mean_val`: last 0.0675, best 0.0506 @ ep 249
  - `Generation/extended_KID_mean_train`: last 0.0588, best 0.0452 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.2526, best 0.2063 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2359, best 0.1928 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4101 (min 0.3789, max 0.8358)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7754 (min 0.4190, max 0.8208)
  - `Validation/MS-SSIM_bravo`: last 0.7836 (min 0.7154, max 0.8201)
  - `Validation/PSNR_bravo`: last 26.2553 (min 25.3862, max 27.9853)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3430
  - `Generation_Diversity/extended_MSSSIM`: 0.1164

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1604, max 4.7001 @ ep 6
  - `training/grad_norm_max`: last 0.6513, max 30.0865 @ ep 9

### exp22

**exp22 (LDM DiT sizing)** — DiT-B / DiT-L / DiT-S variants on 4x latents.
Per memory: **exp22_2 (DiT-L 478M) is the fastest training (3.53h) with
FID 47.41 — most efficient latent-space recipe**; exp22_1 (DiT-B 136M)
FID 48.99.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp22_2_ldm_4x_dit_l_20260304-100612` — 0.0358
  2. 🥈 `exp22_3_ldm_4x_dit_s_long_20260304-102347` — 0.0373
  3.  `exp22_1_ldm_4x_dit_b_20260304-100512` — 0.0388

#### `exp22_1_ldm_4x_dit_b_20260304-100512`
*started 2026-03-04 10:05 • 500 epochs • 14h38m • 548288.8 TFLOPs • peak VRAM 11.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0280 → 0.0763 (min 0.0640 @ ep 474)
  - `Loss/MSE_val`: 1.0214 → 0.1542 (min 0.0851 @ ep 126)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2654
  - 0.1-0.2: 0.1786
  - 0.2-0.3: 0.1281
  - 0.3-0.4: 0.0778
  - 0.4-0.5: 0.0566
  - 0.5-0.6: 0.0639
  - 0.6-0.7: 0.0542
  - 0.7-0.8: 0.0264
  - 0.8-0.9: 0.0309
  - 0.9-1.0: 0.0219

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0561, best 0.0403 @ ep 466
  - `Generation/KID_mean_train`: last 0.0492, best 0.0342 @ ep 356
  - `Generation/KID_std_val`: last 0.0115, best 0.0051 @ ep 334
  - `Generation/KID_std_train`: last 0.0104, best 0.0041 @ ep 318
  - `Generation/CMMD_val`: last 0.3307, best 0.2414 @ ep 333
  - `Generation/CMMD_train`: last 0.3046, best 0.2208 @ ep 333
  - `Generation/extended_KID_mean_val`: last 0.0419, best 0.0388 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0353, best 0.0326 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.2825, best 0.2356 @ ep 374
  - `Generation/extended_CMMD_train`: last 0.2597, best 0.2175 @ ep 374

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3983 (min 0.3681, max 0.9378)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8013 (min 0.4155, max 0.8221)
  - `Validation/MS-SSIM_bravo`: last 0.7877 (min 0.6890, max 0.8211)
  - `Validation/PSNR_bravo`: last 26.7414 (min 25.1302, max 27.8318)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2322
  - `Generation_Diversity/extended_MSSSIM`: 0.1460

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0670, max 5.0090 @ ep 2
  - `training/grad_norm_max`: last 0.1135, max 20.6080 @ ep 3

#### `exp22_2_ldm_4x_dit_l_20260304-100612`
*started 2026-03-04 10:06 • 500 epochs • 54h01m • 1948942.8 TFLOPs • peak VRAM 30.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0299 → 0.0638 (min 0.0544 @ ep 495)
  - `Loss/MSE_val`: 1.0267 → 0.1720 (min 0.0847 @ ep 112)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.4843
  - 0.1-0.2: 0.1993
  - 0.2-0.3: 0.1334
  - 0.3-0.4: 0.0676
  - 0.4-0.5: 0.0532
  - 0.5-0.6: 0.0587
  - 0.6-0.7: 0.0533
  - 0.7-0.8: 0.0330
  - 0.8-0.9: 0.0289
  - 0.9-1.0: 0.0332

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0535, best 0.0357 @ ep 345
  - `Generation/KID_mean_train`: last 0.0486, best 0.0326 @ ep 383
  - `Generation/KID_std_val`: last 0.0081, best 0.0046 @ ep 369
  - `Generation/KID_std_train`: last 0.0093, best 0.0045 @ ep 369
  - `Generation/CMMD_val`: last 0.4123, best 0.2461 @ ep 406
  - `Generation/CMMD_train`: last 0.3891, best 0.2302 @ ep 406
  - `Generation/extended_KID_mean_val`: last 0.0395, best 0.0358 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0331, best 0.0299 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.2822, best 0.2702 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2633, best 0.2497 @ ep 474

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4149 (min 0.3548, max 1.0124)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7848 (min 0.4038, max 0.8132)
  - `Validation/MS-SSIM_bravo`: last 0.7798 (min 0.6412, max 0.8220)
  - `Validation/PSNR_bravo`: last 26.4910 (min 24.4973, max 28.0845)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1741
  - `Generation_Diversity/extended_MSSSIM`: 0.1177

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 19, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1067, max 3.4043 @ ep 3
  - `training/grad_norm_max`: last 0.2781, max 19.8116 @ ep 4

#### `exp22_3_ldm_4x_dit_s_long_20260304-102347`
*started 2026-03-04 10:23 • 2000 epochs • 37h47m • 548633.2 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0299 → 0.0461 (min 0.0416 @ ep 1655)
  - `Loss/MSE_val`: 1.0283 → 0.3681 (min 0.0823 @ ep 240)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.1317
  - 0.1-0.2: 0.4389
  - 0.2-0.3: 0.1872
  - 0.3-0.4: 0.1042
  - 0.4-0.5: 0.0914
  - 0.5-0.6: 0.0670
  - 0.6-0.7: 0.0384
  - 0.7-0.8: 0.0267
  - 0.8-0.9: 0.0370
  - 0.9-1.0: 0.0300

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0550, best 0.0340 @ ep 919
  - `Generation/KID_mean_train`: last 0.0485, best 0.0291 @ ep 919
  - `Generation/KID_std_val`: last 0.0096, best 0.0035 @ ep 1070
  - `Generation/KID_std_train`: last 0.0078, best 0.0038 @ ep 1232
  - `Generation/CMMD_val`: last 0.3181, best 0.2140 @ ep 1646
  - `Generation/CMMD_train`: last 0.2895, best 0.1924 @ ep 1646
  - `Generation/extended_KID_mean_val`: last 0.0484, best 0.0373 @ ep 899
  - `Generation/extended_KID_mean_train`: last 0.0397, best 0.0320 @ ep 1299
  - `Generation/extended_CMMD_val`: last 0.2592, best 0.2310 @ ep 1299
  - `Generation/extended_CMMD_train`: last 0.2357, best 0.2114 @ ep 1299

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.3974 (min 0.3617, max 0.9780)
  - `Validation/MS-SSIM-3D_bravo`: last 0.8072 (min 0.3842, max 0.8219)
  - `Validation/MS-SSIM_bravo`: last 0.7876 (min 0.6721, max 0.8250)
  - `Validation/PSNR_bravo`: last 26.4229 (min 24.7669, max 27.9777)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1986
  - `Generation_Diversity/extended_MSSSIM`: 0.1172

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1167, max 4.3459 @ ep 3
  - `training/grad_norm_max`: last 0.3293, max 15.7286 @ ep 3

### exp27

**exp27 (LDM + ScoreAug)** — DiT-L on latents with ScoreAug.
Memory notes ScoreAug HURTS latent space (exp27/28 worse than non-ScoreAug LDM).

#### `exp27_ldm_4x_dit_l_scoreaug_20260325-190518`
*started 2026-03-25 19:05 • 1000 epochs • 108h25m • 3897886.2 TFLOPs • peak VRAM 39.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9299 → 0.0599 (min 0.0537 @ ep 955)
  - `Loss/MSE_val`: 1.0283 → 0.2006 (min 0.0793 @ ep 278)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.4362
  - 0.1-0.2: 0.2370
  - 0.2-0.3: 0.1184
  - 0.3-0.4: 0.0862
  - 0.4-0.5: 0.0578
  - 0.5-0.6: 0.0546
  - 0.6-0.7: 0.0330
  - 0.7-0.8: 0.0282
  - 0.8-0.9: 0.0292
  - 0.9-1.0: 0.0341

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0632, best 0.0390 @ ep 379
  - `Generation/KID_mean_train`: last 0.0577, best 0.0341 @ ep 379
  - `Generation/KID_std_val`: last 0.0106, best 0.0044 @ ep 469
  - `Generation/KID_std_train`: last 0.0118, best 0.0042 @ ep 593
  - `Generation/CMMD_val`: last 0.3756, best 0.2568 @ ep 956
  - `Generation/CMMD_train`: last 0.3556, best 0.2453 @ ep 630
  - `Generation/extended_KID_mean_val`: last 0.0528, best 0.0421 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0513, best 0.0355 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2900, best 0.2565 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.2782, best 0.2439 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4462 (min 0.3651, max 0.9681)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7764 (min 0.3931, max 0.8071)
  - `Validation/MS-SSIM_bravo`: last 0.7734 (min 0.6768, max 0.8229)
  - `Validation/PSNR_bravo`: last 26.4312 (min 25.1327, max 27.9621)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2648
  - `Generation_Diversity/extended_MSSSIM`: 0.1438

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 19, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0813, max 3.3241 @ ep 4
  - `training/grad_norm_max`: last 0.2918, max 13.2235 @ ep 4

### exp28

**exp28 (LDM MAISI + ScoreAug)** — MAISI-UNet on latents + ScoreAug, with
variants: base, mixup, v2. All degrade vs exp21_2 baseline.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp28_ldm_4x_unet_maisi_scoreaug_20260325-013810` — 0.0573
  2. 🥈 `exp28_2_ldm_4x_unet_maisi_scoreaug_v2_20260326-144538` — 0.0597
  3.  `exp28_1_ldm_4x_unet_maisi_scoreaug_mixup_20260325-211501` — 0.0947

#### `exp28_ldm_4x_unet_maisi_scoreaug_20260325-013810`
*started 2026-03-25 01:38 • 1000 epochs • 20h42m • 14480.4 TFLOPs • peak VRAM 21.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9066 → 0.0650 (min 0.0599 @ ep 731)
  - `Loss/MSE_val`: 0.9225 → 0.1897 (min 0.0704 @ ep 151)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2765
  - 0.1-0.2: 0.2180
  - 0.2-0.3: 0.1293
  - 0.3-0.4: 0.0873
  - 0.4-0.5: 0.0761
  - 0.5-0.6: 0.0435
  - 0.6-0.7: 0.0304
  - 0.7-0.8: 0.0248
  - 0.8-0.9: 0.0212
  - 0.9-1.0: 0.0350

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.1786, best 0.0439 @ ep 593
  - `Generation/KID_mean_train`: last 0.1737, best 0.0372 @ ep 453
  - `Generation/KID_std_val`: last 0.0283, best 0.0098 @ ep 393
  - `Generation/KID_std_train`: last 0.0359, best 0.0095 @ ep 318
  - `Generation/CMMD_val`: last 0.3749, best 0.1910 @ ep 593
  - `Generation/CMMD_train`: last 0.3636, best 0.1829 @ ep 593
  - `Generation/extended_KID_mean_val`: last 0.0915, best 0.0573 @ ep 599
  - `Generation/extended_KID_mean_train`: last 0.0893, best 0.0476 @ ep 599
  - `Generation/extended_CMMD_val`: last 0.2438, best 0.1990 @ ep 599
  - `Generation/extended_CMMD_train`: last 0.2302, best 0.1864 @ ep 599

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4057 (min 0.3569, max 1.0489)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7805 (min 0.3898, max 0.8198)
  - `Validation/MS-SSIM_bravo`: last 0.7721 (min 0.6752, max 0.8217)
  - `Validation/PSNR_bravo`: last 25.9898 (min 24.8990, max 27.8648)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.5001
  - `Generation_Diversity/extended_MSSSIM`: 0.1894

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0947, max 2.7905 @ ep 5
  - `training/grad_norm_max`: last 0.2263, max 31.9083 @ ep 4

#### `exp28_1_ldm_4x_unet_maisi_scoreaug_mixup_20260325-211501`
*started 2026-03-25 21:15 • 1000 epochs • 20h56m • 14480.4 TFLOPs • peak VRAM 21.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9064 → 0.0735 (min 0.0576 @ ep 995)
  - `Loss/MSE_val`: 0.9394 → 0.1174 (min 0.0682 @ ep 219)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1888
  - 0.1-0.2: 0.1487
  - 0.2-0.3: 0.1050
  - 0.3-0.4: 0.0730
  - 0.4-0.5: 0.0595
  - 0.5-0.6: 0.0434
  - 0.6-0.7: 0.0286
  - 0.7-0.8: 0.0333
  - 0.8-0.9: 0.0224
  - 0.9-1.0: 0.0418

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.2133, best 0.1052 @ ep 925
  - `Generation/KID_mean_train`: last 0.2061, best 0.1010 @ ep 925
  - `Generation/KID_std_val`: last 0.0285, best 0.0161 @ ep 332
  - `Generation/KID_std_train`: last 0.0242, best 0.0142 @ ep 332
  - `Generation/CMMD_val`: last 0.4527, best 0.3543 @ ep 122
  - `Generation/CMMD_train`: last 0.4372, best 0.3363 @ ep 122
  - `Generation/extended_KID_mean_val`: last 0.0963, best 0.0947 @ ep 949
  - `Generation/extended_KID_mean_train`: last 0.0920, best 0.0908 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.3635, best 0.3246 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.3484, best 0.3159 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4256 (min 0.3681, max 0.8179)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7926 (min 0.4007, max 0.8209)
  - `Validation/MS-SSIM_bravo`: last 0.7795 (min 0.6921, max 0.8209)
  - `Validation/PSNR_bravo`: last 26.3714 (min 25.2335, max 27.9048)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3340
  - `Generation_Diversity/extended_MSSSIM`: 0.1471

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0888, max 2.7483 @ ep 3
  - `training/grad_norm_max`: last 0.2406, max 17.0446 @ ep 3

#### `exp28_2_ldm_4x_unet_maisi_scoreaug_v2_20260326-144538`
*started 2026-03-26 14:45 • 1000 epochs • 24h29m • 14480.4 TFLOPs • peak VRAM 21.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.7999 → 0.0639 (min 0.0624 @ ep 845)
  - `Loss/MSE_val`: 0.9298 → 0.0980 (min 0.0682 @ ep 339)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1573
  - 0.1-0.2: 0.1136
  - 0.2-0.3: 0.0816
  - 0.3-0.4: 0.0578
  - 0.4-0.5: 0.0413
  - 0.5-0.6: 0.0334
  - 0.6-0.7: 0.0233
  - 0.7-0.8: 0.0420
  - 0.8-0.9: 0.0229
  - 0.9-1.0: 0.0272

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0888, best 0.0332 @ ep 574
  - `Generation/KID_mean_train`: last 0.0844, best 0.0299 @ ep 574
  - `Generation/KID_std_val`: last 0.0181, best 0.0039 @ ep 574
  - `Generation/KID_std_train`: last 0.0167, best 0.0037 @ ep 574
  - `Generation/CMMD_val`: last 0.3645, best 0.2303 @ ep 819
  - `Generation/CMMD_train`: last 0.3537, best 0.2177 @ ep 574
  - `Generation/extended_KID_mean_val`: last 0.0597, best 0.0597 @ ep 999
  - `Generation/extended_KID_mean_train`: last 0.0587, best 0.0587 @ ep 999
  - `Generation/extended_CMMD_val`: last 0.2436, best 0.2436 @ ep 999
  - `Generation/extended_CMMD_train`: last 0.2316, best 0.2316 @ ep 999

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4275 (min 0.2961, max 0.9681)
  - `Validation/MS-SSIM-3D_bravo`: last 0.7892 (min 0.3991, max 0.8208)
  - `Validation/MS-SSIM_bravo`: last 0.7742 (min 0.6774, max 0.8180)
  - `Validation/PSNR_bravo`: last 26.0131 (min 25.1112, max 27.8835)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.8552
  - `Generation_Diversity/extended_MSSSIM`: 0.2882

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0651, max 2.6494 @ ep 3
  - `training/grad_norm_max`: last 0.1476, max 16.0253 @ ep 3

---
## diffusion_3d/seg

*17 runs across 2 experiment families.*

### exp2

**exp2 family (seg generation)** — unconditional + size-bin-conditioned seg
mask generation for the 2-stage generation pipeline (seg → bravo).
Variants: base size-bin FiLM conditioning (exp2, exp2_1); input-channel
concatenation (exp2b, exp2b_1); improved conditioning (exp2c, exp2c_1);
auxiliary bin-prediction loss (exp2d — see memory/aux_bin_loss_experiment.md);
multi-level auxiliary loss (exp2e).

**Known NaN cases:** two early exp2 runs produced NaNs in `Generation/CMMD_*`
tags (likely CMMD feature extractor failing on early checkpoints).

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp2_pixel_seg_sizebin_rflow_128x160_20260127-043308` — -0.0075
  2. 🥈 `exp2_pixel_seg_sizebin_rflow_128x160_20260121-163331` — -0.0072
  3.  `exp2e_pixel_seg_multilevel_aux_20260226-130811` — -4.175e-05
  4.  `exp2d_pixel_seg_aux_bin_20260225-040703` — 8.597e-06
  5.  `exp2d_pixel_seg_aux_bin_20260225-005914` — 1.847e-05
  6.  `exp2d_1_pixel_seg_aux_bin_20260224-000544` — 0.0001237
  7.  `exp2c_pixel_seg_improved_20260218-183124` — 0.000161
  8.  `exp2c_1_pixel_seg_improved_20260218-183155` — 0.0001631
  9.  `exp2b_1_pixel_seg_input_cond_rflow_256x160_20260201-133413` — 0.0002143
  10.  `exp2_1_pixel_seg_sizebin_rflow_256x160_20260201-015003` — 0.0015
  11.  `exp2_pixel_seg_sizebin_rflow_128x160_20260201-014505` — 0.0020
  12.  `exp2b_pixel_seg_input_cond_rflow_128x160_20260201-025016` — 0.0045

#### `exp2_pixel_seg_sizebin_rflow_128x160_20260121-163331`
*started 2026-01-21 16:33 • 500 epochs • 14h03m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9618 → 0.000351 (min 0.0003171 @ ep 463)
  - `Loss/MSE_val`: 0.9170 → 0.0004175 (min 0.0003727 @ ep 336)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0010
  - 0.1-0.2: 0.0003547
  - 0.2-0.3: 0.000502
  - 0.3-0.4: 0.0005963
  - 0.4-0.5: 0.0001144
  - 0.5-0.6: 0.0001992
  - 0.6-0.7: 0.0007361
  - 0.7-0.8: 0.0004744
  - 0.8-0.9: 0.000142
  - 0.9-1.0: 0.0003026

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0021, best -0.0091 @ ep 351
  - `Generation/KID_mean_train`: last 0.0026, best -0.0081 @ ep 351
  - `Generation/KID_std_val`: last 0.0044, best 0.0030 @ ep 210
  - `Generation/KID_std_train`: last 0.0064, best 0.0018 @ ep 66
  - `Generation/CMMD_val`: last —, best 0 @ ep 14
  - `Generation/CMMD_train`: last —, best 0 @ ep 14
  - `Generation/extended_KID_mean_val`: last -0.000709, best -0.0072 @ ep 399
  - `Generation/extended_KID_mean_train`: last -0.0003647, best -0.0060 @ ep 149
  - `Generation/extended_CMMD_val`: last 0, best 0 @ ep 24
  - `Generation/extended_CMMD_train`: last 0, best 0 @ ep 24

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.5384 (min 0.0426, max 0.7120)
  - `Validation/IoU_seg_conditioned`: last 0.4657 (min 0.0362, max 0.6320)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9821 (min 0.0986, max 0.9947)
  - `Validation/PSNR_seg_conditioned`: last 43.3717 (min 11.7722, max 45.8349)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2732
  - `Generation_Diversity/extended_MSSSIM`: 0.0415
  - `Generation_Diversity/LPIPS`: 0.2230
  - `Generation_Diversity/MSSSIM`: 0.0358

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0086, max 3.8780 @ ep 0
  - `training/grad_norm_max`: last 0.0514, max 27.0617 @ ep 52

**⚠️ NaN detected in 4 tag(s):** Generation/CMMD_train, Generation/CMMD_val, Generation/extended_CMMD_train, Generation/extended_CMMD_val

#### `exp2_1_pixel_seg_sizebin_rflow_256x160_20260122-131414`
*started 2026-01-22 13:14 • 500 epochs • 39h34m • 16018.2 TFLOPs • peak VRAM 59.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9818 → 0.0002913 (min 0.000272 @ ep 495)
  - `Loss/MSE_val`: 0.9619 → 0.0003532 (min 0.0003358 @ ep 498)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0003097
  - 0.1-0.2: 0.000362
  - 0.2-0.3: 0.0002199
  - 0.3-0.4: 0.0001841
  - 0.4-0.5: 0.0001782
  - 0.5-0.6: 0.0003509
  - 0.6-0.7: 0.0006542
  - 0.7-0.8: 8.155e-05
  - 0.8-0.9: 0.0005037
  - 0.9-1.0: 0.0006298

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.5874, best 1.5874 @ ep 0
  - `Generation/KID_mean_train`: last 1.6076, best 1.6076 @ ep 0
  - `Generation/KID_std_val`: last 0.0560, best 0.0560 @ ep 0
  - `Generation/KID_std_train`: last 0.0529, best 0.0529 @ ep 0
  - `Generation/CMMD_val`: last 0.8042, best 0.8042 @ ep 0
  - `Generation/CMMD_train`: last 0.8185, best 0.8185 @ ep 0

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.7798 (min 0.0039, max 0.8387)
  - `Validation/IoU_seg_conditioned`: last 0.7087 (min 0.0020, max 0.7711)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9922 (min 0.0560, max 0.9949)
  - `Validation/PSNR_seg_conditioned`: last 45.8363 (min 9.3679, max 46.1818)

**Diversity (extended):**
  - `Generation_Diversity/LPIPS`: 0.0506
  - `Generation_Diversity/MSSSIM`: 0.9726

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 10, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0075, max 3.9354 @ ep 0
  - `training/grad_norm_max`: last 0.0498, max 4.4278 @ ep 2

#### `exp2_1_pixel_seg_sizebin_rflow_256x160_20260127-043308`
*started 2026-01-27 04:33 • 500 epochs • 40h03m • 16018.2 TFLOPs • peak VRAM 59.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9813 → 0.0002616 (min 0.0002577 @ ep 445)
  - `Loss/MSE_val`: 0.9608 → 0.0003426 (min 0.0003243 @ ep 423)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0002365
  - 0.1-0.2: 0.0001142
  - 0.2-0.3: 0.000366
  - 0.3-0.4: 0.0001099
  - 0.4-0.5: 0.0001636
  - 0.5-0.6: 0.0002741
  - 0.6-0.7: 0.0001493
  - 0.7-0.8: 0.0008316
  - 0.8-0.9: 0.0008408
  - 0.9-1.0: 0.000379

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.6138, best 1.6138 @ ep 0
  - `Generation/KID_mean_train`: last 1.6361, best 1.6361 @ ep 0
  - `Generation/KID_std_val`: last 0.0525, best 0.0525 @ ep 0
  - `Generation/KID_std_train`: last 0.0495, best 0.0495 @ ep 0
  - `Generation/CMMD_val`: last 0.8013, best 0.8013 @ ep 0
  - `Generation/CMMD_train`: last 0.8154, best 0.8154 @ ep 0

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.7995 (min 0.0502, max 0.8360)
  - `Validation/IoU_seg_conditioned`: last 0.7124 (min 0.0444, max 0.7559)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9948 (min 0.0552, max 0.9955)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9879 (min 0.0846, max 0.9962)
  - `Validation/PSNR_seg_conditioned`: last 45.0759 (min 10.9549, max 47.2193)

**Diversity (extended):**
  - `Generation_Diversity/LPIPS`: 0.0505
  - `Generation_Diversity/MSSSIM`: 0.9724

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 10, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0073, max 3.9346 @ ep 0
  - `training/grad_norm_max`: last 0.0608, max 4.9994 @ ep 4

#### `exp2_pixel_seg_sizebin_rflow_128x160_20260127-043308`
*started 2026-01-27 04:33 • 500 epochs • 18h06m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9612 → 0.0003342 (min 0.0003089 @ ep 411)
  - `Loss/MSE_val`: 0.9163 → 0.0004319 (min 0.0003477 @ ep 378)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0002566
  - 0.1-0.2: 0.0003505
  - 0.2-0.3: 0.0012
  - 0.3-0.4: 0.0001148
  - 0.4-0.5: 0.0001915
  - 0.5-0.6: 0.0003602
  - 0.6-0.7: 0.0004479
  - 0.7-0.8: 0.0002427
  - 0.8-0.9: 0.0009435
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0049, best -0.0095 @ ep 408
  - `Generation/KID_mean_train`: last 0.0019, best -0.0081 @ ep 408
  - `Generation/KID_std_val`: last 0.0129, best 0.0027 @ ep 349
  - `Generation/KID_std_train`: last 0.0042, best 0.0023 @ ep 310
  - `Generation/CMMD_val`: last —, best 0 @ ep 6
  - `Generation/CMMD_train`: last —, best 0 @ ep 6
  - `Generation/extended_KID_mean_val`: last 0.0049, best -0.0075 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0019, best -0.0062 @ ep 399
  - `Generation/extended_CMMD_val`: last —, best 0 @ ep 49
  - `Generation/extended_CMMD_train`: last —, best 0 @ ep 49

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.5917 (min 0.0040, max 0.7359)
  - `Validation/IoU_seg_conditioned`: last 0.5303 (min 0.0020, max 0.6711)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9876 (min 0.0650, max 0.9884)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9898 (min 0.0661, max 0.9942)
  - `Validation/PSNR_seg_conditioned`: last 44.8843 (min 9.6200, max 46.6260)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0806
  - `Generation_Diversity/extended_MSSSIM`: 0.0050
  - `Generation_Diversity/LPIPS`: 0.0542
  - `Generation_Diversity/MSSSIM`: 0.0029

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0090, max 3.8779 @ ep 0
  - `training/grad_norm_max`: last 0.0502, max 23.8528 @ ep 50

**⚠️ NaN detected in 4 tag(s):** Generation/CMMD_train, Generation/CMMD_val, Generation/extended_CMMD_train, Generation/extended_CMMD_val

#### `exp2_pixel_seg_sizebin_rflow_128x160_20260201-014505`
*started 2026-02-01 01:45 • 444 epochs • 119h48m • 1731.4 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9618 → 1.0004 (min 0.0015 @ ep 29)
  - `Loss/MSE_val`: 0.9180 → 1.0003 (min 0.0014 @ ep 31)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.9997
  - 0.1-0.2: 1.0002
  - 0.2-0.3: 1.0009
  - 0.3-0.4: 1.0007
  - 0.4-0.5: 0.9999
  - 0.5-0.6: 1.0000
  - 0.6-0.7: 1.0002
  - 0.7-0.8: 1.0001
  - 0.8-0.9: 1.0000
  - 0.9-1.0: 1.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.2420, best 0.0002941 @ ep 15
  - `Generation/KID_mean_train`: last 1.2503, best 3.102e-05 @ ep 19
  - `Generation/KID_std_val`: last 0.0083, best 0.0003122 @ ep 15
  - `Generation/KID_std_train`: last 0.0067, best 0.0001816 @ ep 19
  - `Generation/CMMD_val`: last 0.8697, best 0.0683 @ ep 19
  - `Generation/CMMD_train`: last 0.8899, best 0.0593 @ ep 19
  - `Generation/extended_KID_mean_val`: last 1.2345, best 0.0020 @ ep 24
  - `Generation/extended_KID_mean_train`: last 1.2465, best 0.0009177 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.8658, best 0.1344 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.8875, best 0.0962 @ ep 24

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.0466 (min 0.0021, max 0.5842)
  - `Validation/IoU_seg_conditioned`: last 0.0391 (min 0.0011, max 0.5064)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.0600 (min 0.0596, max 0.9258)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.0943 (min 0.0508, max 0.9303)
  - `Validation/PSNR_seg_conditioned`: last 10.9464 (min 8.4513, max 38.9013)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0516
  - `Generation_Diversity/extended_MSSSIM`: 0.9796

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 4.204e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0014, max 3.8785 @ ep 0
  - `training/grad_norm_max`: last 0.0105, max 21.0506 @ ep 34

#### `exp2_1_pixel_seg_sizebin_rflow_256x160_20260201-015003`
*started 2026-02-01 01:50 • 500 epochs • 70h26m • 16018.2 TFLOPs • peak VRAM 59.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9816 → 0.0002921 (min 0.0002686 @ ep 409)
  - `Loss/MSE_val`: 0.9608 → 0.0004652 (min 0.0003369 @ ep 423)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0046
  - 0.1-0.2: 0.0001851
  - 0.2-0.3: 0.000445
  - 0.3-0.4: 0.0001349
  - 0.4-0.5: 0.0001857
  - 0.5-0.6: 0.000502
  - 0.6-0.7: 0.0001981
  - 0.7-0.8: 0.0006547
  - 0.8-0.9: 0.0001522
  - 0.9-1.0: 0.0003137

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0067, best -1.976e-05 @ ep 384
  - `Generation/KID_mean_train`: last 0.0036, best -0.000133 @ ep 465
  - `Generation/KID_std_val`: last 0.0031, best 0.0003814 @ ep 247
  - `Generation/KID_std_train`: last 0.0020, best 0.0001861 @ ep 465
  - `Generation/CMMD_val`: last 0.2300, best 0.0674 @ ep 46
  - `Generation/CMMD_train`: last 0.1899, best 0.0726 @ ep 58
  - `Generation/extended_KID_mean_val`: last 0.0064, best 0.0015 @ ep 49
  - `Generation/extended_KID_mean_train`: last 0.0036, best 0.0005695 @ ep 49
  - `Generation/extended_CMMD_val`: last 0.2109, best 0.0159 @ ep 74
  - `Generation/extended_CMMD_train`: last 0.1657, best 0 @ ep 124

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.6968 (min 0.0225, max 0.8303)
  - `Validation/IoU_seg_conditioned`: last 0.6130 (min 0.0132, max 0.7605)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9951 (min 0.0551, max 0.9957)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9895 (min 0.0729, max 0.9959)
  - `Validation/PSNR_seg_conditioned`: last 43.8325 (min 10.6042, max 46.1228)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0156
  - `Generation_Diversity/extended_MSSSIM`: 0.0032
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 10, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0085, max 3.9357 @ ep 0
  - `training/grad_norm_max`: last 0.0446, max 9.8958 @ ep 8

#### `exp2b_pixel_seg_input_cond_rflow_128x160_20260201-025016`
*started 2026-02-01 02:50 • 500 epochs • 13h07m • 1954.2 TFLOPs • peak VRAM 21.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9621 → 1.0003 (min 0.0015 @ ep 32)
  - `Loss/MSE_val`: 0.9159 → 1.0004 (min 0.0014 @ ep 32)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.9995
  - 0.1-0.2: 1.0021
  - 0.2-0.3: 0.9998
  - 0.3-0.4: 1.0009
  - 0.4-0.5: 1.0007
  - 0.5-0.6: 1.0004
  - 0.6-0.7: 1.0016
  - 0.7-0.8: 0.9999
  - 0.8-0.9: 0.9993
  - 0.9-1.0: 0.9995

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.2341, best 0.000274 @ ep 16
  - `Generation/KID_mean_train`: last 1.2416, best 6.04e-05 @ ep 19
  - `Generation/KID_std_val`: last 0.0064, best 0.0002567 @ ep 16
  - `Generation/KID_std_train`: last 0.0066, best 0.0002647 @ ep 34
  - `Generation/CMMD_val`: last 0.8687, best 0.0738 @ ep 34
  - `Generation/CMMD_train`: last 0.8885, best 0.0743 @ ep 34
  - `Generation/extended_KID_mean_val`: last 1.2412, best 0.0045 @ ep 24
  - `Generation/extended_KID_mean_train`: last 1.2465, best 0.0025 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.8653, best 0.2472 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.8867, best 0.2054 @ ep 24

**Validation quality:**
  - `Validation/LPIPS_seg_conditioned_input`: last 1.8515 (min 1.5697, max 1.8565)
  - `Validation/MS-SSIM-3D_seg_conditioned_input`: last 0.0599 (min 0.0596, max 0.9210)
  - `Validation/MS-SSIM_seg_conditioned_input`: last 0.0965 (min 0.0540, max 0.9116)
  - `Validation/PSNR_seg_conditioned_input`: last 11.3888 (min 8.6780, max 37.8508)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0525
  - `Generation_Diversity/extended_MSSSIM`: 0.9808

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0016, max 3.6648 @ ep 0
  - `training/grad_norm_max`: last 0.0087, max 31.1196 @ ep 6

#### `exp2b_1_pixel_seg_input_cond_rflow_256x160_20260201-133413`
*started 2026-02-01 13:34 • 500 epochs • 42h03m • 16018.2 TFLOPs • peak VRAM 61.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9808 → 0.0002894 (min 0.0002781 @ ep 491)
  - `Loss/MSE_val`: 0.9583 → 0.0003796 (min 0.0003361 @ ep 406)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0002477
  - 0.1-0.2: 0.000405
  - 0.2-0.3: 0.0002393
  - 0.3-0.4: 0.0004331
  - 0.4-0.5: 3.576e-05
  - 0.5-0.6: 0.0005863
  - 0.6-0.7: 0.0003432
  - 0.7-0.8: 0.0004043
  - 0.8-0.9: 0.0001968
  - 0.9-1.0: 0.0004233

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0031, best -9.584e-07 @ ep 239
  - `Generation/KID_mean_train`: last 0.0016, best -5.052e-05 @ ep 462
  - `Generation/KID_std_val`: last 0.0017, best 0.0003234 @ ep 437
  - `Generation/KID_std_train`: last 0.0013, best 0.0002907 @ ep 241
  - `Generation/CMMD_val`: last 0.1494, best 0.0604 @ ep 327
  - `Generation/CMMD_train`: last 0.1036, best 0 @ ep 230
  - `Generation/extended_KID_mean_val`: last 0.0005211, best 0.0002143 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0001451, best -5.641e-06 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.0128, best 0 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.0196, best 0 @ ep 224

**Validation quality:**
  - `Validation/LPIPS_seg_conditioned_input`: last 0.0920 (min 0.0632, max 1.7517)
  - `Validation/MS-SSIM-3D_seg_conditioned_input`: last 0.9941 (min 0.0554, max 0.9954)
  - `Validation/MS-SSIM_seg_conditioned_input`: last 0.9879 (min 0.0769, max 0.9957)
  - `Validation/PSNR_seg_conditioned_input`: last 44.4647 (min 10.9904, max 45.9263)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0346
  - `Generation_Diversity/extended_MSSSIM`: 0.0089
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 10, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0100, max 3.8221 @ ep 0
  - `training/grad_norm_max`: last 0.0549, max 25.0066 @ ep 10

#### `exp2c_pixel_seg_improved_20260218-183124`
*started 2026-02-18 18:31 • 500 epochs • 20h28m • 1954.5 TFLOPs • peak VRAM 23.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9817 → 0.0003893 (min 0.0003299 @ ep 475)
  - `Loss/MSE_val`: 0.9605 → 0.0005062 (min 0.0004293 @ ep 349)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0003178
  - 0.1-0.2: 0.0006132
  - 0.2-0.3: 0.0006661
  - 0.3-0.4: 0.0004564
  - 0.4-0.5: 8.782e-05
  - 0.5-0.6: 0.0005491
  - 0.6-0.7: 0.0008
  - 0.7-0.8: 0.000274
  - 0.8-0.9: 0.0010
  - 0.9-1.0: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0039, best 2.741e-05 @ ep 94
  - `Generation/KID_mean_train`: last 0.0020, best -1.641e-05 @ ep 285
  - `Generation/KID_std_val`: last 0.0018, best 0.0001822 @ ep 142
  - `Generation/KID_std_train`: last 0.0013, best 0.0001065 @ ep 178
  - `Generation/CMMD_val`: last 0.2334, best 0.0676 @ ep 82
  - `Generation/CMMD_train`: last 0.1994, best 0 @ ep 90
  - `Generation/extended_KID_mean_val`: last 0.0019, best 0.000161 @ ep 99
  - `Generation/extended_KID_mean_train`: last 0.0007301, best 0.0001147 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.1523, best 0.0455 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.1140, best 0.0306 @ ep 74

**Validation quality:**
  - `Validation/Dice`: last 0.5930 (min 0.0129, max 0.7325)
  - `Validation/IoU`: last 0.5167 (min 0.0068, max 0.6564)
  - `Validation/MS-SSIM`: last 0.9836 (min 0.0883, max 0.9932)
  - `Validation/MS-SSIM-3D`: last 0.9828 (min 0.0614, max 0.9886)
  - `Validation/PSNR`: last 44.0600 (min 10.7648, max 45.3498)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0616
  - `Generation_Diversity/extended_MSSSIM`: 0.0077

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0145, max 3.9276 @ ep 0
  - `training/grad_norm_max`: last 0.4349, max 16.9482 @ ep 14

#### `exp2c_1_pixel_seg_improved_20260218-183155`
*started 2026-02-18 18:31 • 500 epochs • 61h45m • 16018.6 TFLOPs • peak VRAM 41.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9821 → 0.0003357 (min 0.0003339 @ ep 483)
  - `Loss/MSE_val`: 0.9615 → 0.0003534 (min 0.000334 @ ep 487)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0002549
  - 0.1-0.2: 0.0004852
  - 0.2-0.3: 0.0007309
  - 0.3-0.4: 0.0004521
  - 0.4-0.5: 3.997e-05
  - 0.5-0.6: 0.0001186
  - 0.6-0.7: 0.0003408
  - 0.7-0.8: 0.0001947
  - 0.8-0.9: 0.0001457
  - 0.9-1.0: 5.191e-05

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0028, best 7.769e-05 @ ep 368
  - `Generation/KID_mean_train`: last 0.0011, best -9.269e-05 @ ep 321
  - `Generation/KID_std_val`: last 0.0017, best 0.000315 @ ep 83
  - `Generation/KID_std_train`: last 0.0011, best 0.0002211 @ ep 321
  - `Generation/CMMD_val`: last 0.1675, best 0 @ ep 167
  - `Generation/CMMD_train`: last 0.1311, best 0 @ ep 92
  - `Generation/extended_KID_mean_val`: last 0.0024, best 0.0001631 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.0008964, best 9.504e-05 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.1074, best 0 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.0617, best 0 @ ep 374

**Validation quality:**
  - `Validation/Dice`: last 0.6359 (min 0.0262, max 0.8257)
  - `Validation/IoU`: last 0.5710 (min 0.0162, max 0.7539)
  - `Validation/MS-SSIM`: last 0.9883 (min 0.0697, max 0.9945)
  - `Validation/MS-SSIM-3D`: last 0.9929 (min 0.0547, max 0.9947)
  - `Validation/PSNR`: last 43.6381 (min 10.2281, max 46.2384)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0165
  - `Generation_Diversity/extended_MSSSIM`: 0.0136
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0084, max 3.9296 @ ep 0
  - `training/grad_norm_max`: last 0.0551, max 14.4098 @ ep 3

#### `exp2d_1_pixel_seg_aux_bin_20260224-000544`
*started 2026-02-24 00:05 • 327 epochs • 48h21m • 10476.1 TFLOPs • peak VRAM 41.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9826 → 0.000486 (min 0.0004739 @ ep 316)
  - `Loss/MSE_val`: 0.9619 → 0.0004939 (min 0.0003979 @ ep 322)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0015
  - 0.1-0.2: 0.0008746
  - 0.2-0.3: 0.0005871
  - 0.3-0.4: 0.0010
  - 0.4-0.5: 0.0003999
  - 0.5-0.6: 0.0003055
  - 0.6-0.7: 0.0001721
  - 0.7-0.8: 0.0001894
  - 0.8-0.9: 0.0006045
  - 0.9-1.0: 9.315e-05

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0074, best 9.873e-05 @ ep 306
  - `Generation/KID_mean_train`: last 0.0096, best 0.0001902 @ ep 324
  - `Generation/KID_std_val`: last 0.0034, best 0.0004164 @ ep 304
  - `Generation/KID_std_train`: last 0.0033, best 0.0003513 @ ep 175
  - `Generation/CMMD_val`: last 0.2702, best 0.0676 @ ep 240
  - `Generation/CMMD_train`: last 0.3495, best 0.0731 @ ep 202
  - `Generation/extended_KID_mean_val`: last 0.0037, best 0.0001237 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.0015, best 0.0004768 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.1292, best 0.0271 @ ep 299
  - `Generation/extended_CMMD_train`: last 0.0837, best 0.0737 @ ep 299

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.7005 (min 0.0080, max 0.8311)
  - `Validation/IoU_seg_conditioned`: last 0.6205 (min 0.0041, max 0.7484)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9885 (min 0.0550, max 0.9935)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9831 (min 0.0577, max 0.9922)
  - `Validation/PSNR_seg_conditioned`: last 43.3736 (min 9.3580, max 44.3286)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0393
  - `Generation_Diversity/extended_MSSSIM`: 0.0137
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1.459e-05

**Training meta:**
  - `training/grad_norm_avg`: last 6.2632, max 16.6103 @ ep 8
  - `training/grad_norm_max`: last 292.0957, max 489.6794 @ ep 239

#### `exp2d_pixel_seg_aux_bin_20260225-005914`
*started 2026-02-25 00:59 • 438 epochs • 24h03m • 1712.1 TFLOPs • peak VRAM 23.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9829 → 0.0005607 (min 0.00049 @ ep 421)
  - `Loss/MSE_val`: 0.9628 → 0.0005322 (min 0.0005038 @ ep 322)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.00056
  - 0.1-0.2: 0.0005209
  - 0.2-0.3: 0.0017
  - 0.3-0.4: 8.99e-05
  - 0.4-0.5: 0.000519
  - 0.5-0.6: 0.0006902
  - 0.6-0.7: 0.0005539
  - 0.7-0.8: 0.0004265
  - 0.8-0.9: 0.0001662
  - 0.9-1.0: 0.0003409

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0037, best -1.263e-05 @ ep 341
  - `Generation/KID_mean_train`: last 0.0015, best -1.041e-05 @ ep 380
  - `Generation/KID_std_val`: last 0.0012, best 0.0001937 @ ep 225
  - `Generation/KID_std_train`: last 0.0008475, best 0.0001273 @ ep 380
  - `Generation/CMMD_val`: last 0.2273, best 0.0674 @ ep 431
  - `Generation/CMMD_train`: last 0.1934, best 0 @ ep 195
  - `Generation/extended_KID_mean_val`: last 0.0009906, best 1.847e-05 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.0002587, best 7.096e-05 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.1038, best 0 @ ep 274
  - `Generation/extended_CMMD_train`: last 0.0653, best 0 @ ep 74

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.3859 (min 0.0252, max 0.6942)
  - `Validation/IoU_seg_conditioned`: last 0.3167 (min 0.0144, max 0.6169)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9825 (min 0.0610, max 0.9854)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9716 (min 0.0829, max 0.9877)
  - `Validation/PSNR_seg_conditioned`: last 40.2065 (min 10.5916, max 43.7778)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.2435
  - `Generation_Diversity/extended_MSSSIM`: 0.0443

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 2.91e-06

**Training meta:**
  - `training/grad_norm_avg`: last 5.5634, max 61.6180 @ ep 30
  - `training/grad_norm_max`: last 192.4174, max 2309.3896 @ ep 30

#### `exp2d_pixel_seg_aux_bin_20260225-040703`
*started 2026-02-25 04:07 • 306 epochs • 23h10m • 1196.2 TFLOPs • peak VRAM 23.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9826 → 0.0005647 (min 0.0005451 @ ep 296)
  - `Loss/MSE_val`: 0.9634 → 0.0006166 (min 0.0005257 @ ep 299)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0026
  - 0.1-0.2: 0.0007285
  - 0.2-0.3: 0.0011
  - 0.3-0.4: 0.0010
  - 0.4-0.5: 0.0006579
  - 0.5-0.6: 0.0004251
  - 0.6-0.7: 0.0001131
  - 0.7-0.8: 0.0004979
  - 0.8-0.9: 0.0005575
  - 0.9-1.0: 0.0001293

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0015, best 1.669e-07 @ ep 272
  - `Generation/KID_mean_train`: last 0.0028, best 1.652e-05 @ ep 269
  - `Generation/KID_std_val`: last 0.0009242, best 0.0002114 @ ep 248
  - `Generation/KID_std_train`: last 0.0014, best 0.0001693 @ ep 128
  - `Generation/CMMD_val`: last 0.1568, best 0.0683 @ ep 86
  - `Generation/CMMD_train`: last 0.1775, best 0 @ ep 247
  - `Generation/extended_KID_mean_val`: last 0.0001581, best 8.597e-06 @ ep 199
  - `Generation/extended_KID_mean_train`: last 0.0006289, best 1.434e-05 @ ep 274
  - `Generation/extended_CMMD_val`: last 0.0445, best 0 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.0834, best 0.0320 @ ep 199

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.5200 (min 0.0039, max 0.7091)
  - `Validation/IoU_seg_conditioned`: last 0.4526 (min 0.0020, max 0.6130)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9821 (min 0.0614, max 0.9879)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9705 (min 0.0622, max 0.9876)
  - `Validation/PSNR_seg_conditioned`: last 41.5784 (min 9.2051, max 43.3295)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3616
  - `Generation_Diversity/extended_MSSSIM`: 0.0327

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1.763e-05

**Training meta:**
  - `training/grad_norm_avg`: last 6.8411, max 65.6361 @ ep 50
  - `training/grad_norm_max`: last 245.4815, max 3344.2898 @ ep 51

#### `exp2e_pixel_seg_multilevel_aux_20260226-130811`
*started 2026-02-26 13:08 • 500 epochs • 21h34m • 1954.5 TFLOPs • peak VRAM 23.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9830 → 0.0005787 (min 0.0005053 @ ep 463)
  - `Loss/MSE_val`: 0.9642 → 0.0006003 (min 0.0004627 @ ep 456)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0034
  - 0.1-0.2: 0.0010
  - 0.2-0.3: 0.0003208
  - 0.3-0.4: 0.0002089
  - 0.4-0.5: 0.0006972
  - 0.5-0.6: 0.0002271
  - 0.6-0.7: 0.0009971
  - 0.7-0.8: 0.0002562
  - 0.8-0.9: 8.424e-05
  - 0.9-1.0: 0.0002474

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0037, best 2.496e-05 @ ep 320
  - `Generation/KID_mean_train`: last 0.0019, best -5.283e-06 @ ep 304
  - `Generation/KID_std_val`: last 0.0014, best 0.0001974 @ ep 106
  - `Generation/KID_std_train`: last 0.000991, best 0.0001281 @ ep 21
  - `Generation/CMMD_val`: last 0.2121, best 0 @ ep 357
  - `Generation/CMMD_train`: last 0.1740, best 0 @ ep 119
  - `Generation/extended_KID_mean_val`: last 0.0001978, best -4.175e-05 @ ep 99
  - `Generation/extended_KID_mean_train`: last 0.0001151, best 8.294e-05 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.0360, best 0.0180 @ ep 399
  - `Generation/extended_CMMD_train`: last 0, best 0 @ ep 49

**Validation quality:**
  - `Validation/Dice_seg_conditioned`: last 0.5275 (min 0.0045, max 0.7005)
  - `Validation/IoU_seg_conditioned`: last 0.4529 (min 0.0023, max 0.6162)
  - `Validation/MS-SSIM-3D_seg_conditioned`: last 0.9825 (min 0.0615, max 0.9851)
  - `Validation/MS-SSIM_seg_conditioned`: last 0.9754 (min 0.0734, max 0.9851)
  - `Validation/PSNR_seg_conditioned`: last 41.8635 (min 10.0481, max 44.1330)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.1407
  - `Generation_Diversity/extended_MSSSIM`: 0.0417

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 3.7510, max 133.9234 @ ep 11
  - `training/grad_norm_max`: last 251.7324, max 3103.6553 @ ep 170

### exp14

**exp14 family (seg unconditional)** — pure unconditional seg generation
(no size bins). exp14 baseline, exp14_1 256×160, exp14_2 smaller 67M model.
Per memory: exp14_1 is the canonical seg model for pipeline use —
`validate_size_bins=false` required since model is unconditional.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp14_pixel_seg_20260218-183155` — 4.242e-05
  2. 🥈 `exp14_2_pixel_seg_67m_20260408-035801` — 5.155e-05
  3.  `exp14_1_pixel_seg_20260217-040309` — 0.000338

#### `exp14_1_pixel_seg_20260217-040309`
*started 2026-02-17 04:03 • 500 epochs • 41h13m • 16018.2 TFLOPs • peak VRAM 42.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9814 → 0.0002717 (min 0.0002633 @ ep 445)
  - `Loss/MSE_val`: 0.9615 → 0.0004422 (min 0.0003247 @ ep 489)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0031
  - 0.1-0.2: 0.000302
  - 0.2-0.3: 0.0006389
  - 0.3-0.4: 9.358e-05
  - 0.4-0.5: 0.0001717
  - 0.5-0.6: 0.0004684
  - 0.6-0.7: 0.0004185
  - 0.7-0.8: 0.0001005
  - 0.8-0.9: 0.0001056
  - 0.9-1.0: 0.000206

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0006361, best -1.123e-05 @ ep 448
  - `Generation/KID_mean_train`: last 0.0015, best -8.204e-05 @ ep 322
  - `Generation/KID_std_val`: last 0.0008564, best 0.0003607 @ ep 398
  - `Generation/KID_std_train`: last 0.0012, best 0.000263 @ ep 322
  - `Generation/CMMD_val`: last 0.0966, best 0 @ ep 317
  - `Generation/CMMD_train`: last 0.1226, best 0 @ ep 122
  - `Generation/extended_KID_mean_val`: last 0.000597, best 0.000338 @ ep 149
  - `Generation/extended_KID_mean_train`: last -5.095e-05, best -5.095e-05 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.0170, best 0.0170 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.0345, best 0.0156 @ ep 449

**Validation quality:**
  - `Validation/Dice_seg`: last 0.7349 (min 0.0028, max 0.8320)
  - `Validation/IoU_seg`: last 0.6623 (min 0.0014, max 0.7585)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0759
  - `Generation_Diversity/extended_MSSSIM`: 0.0311

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0074, max 3.9316 @ ep 0
  - `training/grad_norm_max`: last 0.0471, max 7.3791 @ ep 300

#### `exp14_pixel_seg_20260218-183155`
*started 2026-02-18 18:31 • 500 epochs • 12h20m • 1954.2 TFLOPs • peak VRAM 23.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9817 → 0.0002952 (min 0.0002786 @ ep 484)
  - `Loss/MSE_val`: 0.9621 → 0.0004534 (min 0.0003522 @ ep 441)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0002949
  - 0.1-0.2: 0.0002914
  - 0.2-0.3: 0.0005675
  - 0.3-0.4: 0.0002187
  - 0.4-0.5: 0.0007005
  - 0.5-0.6: 0.0006776
  - 0.6-0.7: 0.0004775
  - 0.7-0.8: 0.0001379
  - 0.8-0.9: 5.344e-05
  - 0.9-1.0: 0.0013

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0046, best -1.322e-05 @ ep 232
  - `Generation/KID_mean_train`: last 0.0024, best -1.271e-05 @ ep 268
  - `Generation/KID_std_val`: last 0.0016, best 0.0001547 @ ep 239
  - `Generation/KID_std_train`: last 0.0008179, best 9.775e-05 @ ep 459
  - `Generation/CMMD_val`: last 0.2881, best 0.0499 @ ep 173
  - `Generation/CMMD_train`: last 0.2488, best 0 @ ep 17
  - `Generation/extended_KID_mean_val`: last 0.0019, best 4.242e-05 @ ep 324
  - `Generation/extended_KID_mean_train`: last 0.0008573, best 0.0001355 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.1742, best 0.0213 @ ep 324
  - `Generation/extended_CMMD_train`: last 0.1316, best 0.0229 @ ep 349

**Validation quality:**
  - `Validation/Dice_seg`: last 0.5416 (min 0.0048, max 0.6799)
  - `Validation/IoU_seg`: last 0.4584 (min 0.0024, max 0.5603)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0811
  - `Generation_Diversity/extended_MSSSIM`: 0.0076

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0099, max 3.9264 @ ep 0
  - `training/grad_norm_max`: last 0.0384, max 6.7784 @ ep 420

#### `exp14_2_pixel_seg_67m_20260408-035801`
*started 2026-04-08 03:58 • 1000 epochs • 32h40m • 13755.4 TFLOPs • peak VRAM 25.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9915 → 0.0004473 (min 0.0003115 @ ep 989)
  - `Loss/MSE_val`: 0.9829 → 0.0005355 (min 0.0003801 @ ep 954)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0009274
  - 0.1-0.2: 0.0015
  - 0.2-0.3: 0.0005058
  - 0.3-0.4: 0.0008346
  - 0.4-0.5: 0.0001931
  - 0.5-0.6: 0.0001551
  - 0.6-0.7: 0.000339
  - 0.7-0.8: 0.0001846
  - 0.8-0.9: 0.0003717
  - 0.9-1.0: 0.0002368

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0036, best 2.039e-05 @ ep 978
  - `Generation/KID_mean_train`: last 0.0019, best -3.892e-05 @ ep 901
  - `Generation/KID_std_val`: last 0.0021, best 0.0003003 @ ep 595
  - `Generation/KID_std_train`: last 0.0015, best 0.0002452 @ ep 901
  - `Generation/CMMD_val`: last 0.1183, best 0 @ ep 918
  - `Generation/CMMD_train`: last 0.0725, best 0 @ ep 289
  - `Generation/extended_KID_mean_val`: last 0.0032, best 5.155e-05 @ ep 599
  - `Generation/extended_KID_mean_train`: last 0.0014, best 5.416e-05 @ ep 949
  - `Generation/extended_CMMD_val`: last 0.1317, best 0.0140 @ ep 349
  - `Generation/extended_CMMD_train`: last 0.0908, best 0.0146 @ ep 449

**Validation quality:**
  - `Validation/Dice_seg`: last 0.5574 (min 0.1477, max 0.8478)
  - `Validation/IoU_seg`: last 0.5304 (min 0.1268, max 0.7955)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0420
  - `Generation_Diversity/extended_MSSSIM`: 0.0071
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 9, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0536, max 4.8547 @ ep 14
  - `training/grad_norm_max`: last 2.9828, max 248.6193 @ ep 251

---
## diffusion_3d/restoration

*10 runs across 1 experiment families.*

### exp33

**exp33 (IR-SDE / bridge / resfusion restoration)** — post-hoc restoration
network to fix MSE-smoothing artifacts in diffusion-generated volumes.
Per memory/project_irsde_restoration.md:
- IR-SDE (mean-reverting SDE) — variants 33_1, 33_1b, 33_1c (17M).
- Bridge — variants 33_2, 33_2b (17M), 33_5 (bridge 17M).
- Bridge + noise — variants 33_3, 33_3b (17M).
- Resfusion — variants 33_4, 33_4c (17M).
- Training data: SDEdit pairs from exp1_1_1000 @ t₀=0.50.
- Evaluation: LPIPS, FWD, L1 vs real.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp33_3_rflow_bridge_noise_restoration_20260413-014148` — 1.2933
  2. 🥈 `exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630` — 1.2980
  3.  `exp33_2b_rflow_bridge_restoration_17m_20260415-122714` — 1.3226
  4.  `exp33_2_rflow_bridge_restoration_20260413-004114` — 1.3246

#### `exp33_2_rflow_bridge_restoration_20260413-004114`
*started 2026-04-13 00:41 • 381 epochs • 82h51m • 12173.9 TFLOPs • peak VRAM 41.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0003389 → 2.352e-06 (min 2.341e-06 @ ep 369)
  - `Loss/MSE_val`: 1.0099 → 1.1136 (min 1.0099 @ ep 0)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0650
  - 0.1-0.2: 1.0972
  - 0.2-0.3: 1.0970
  - 0.3-0.4: 1.1112
  - 0.4-0.5: 1.1191
  - 0.5-0.6: 1.1288
  - 0.6-0.7: 1.1320
  - 0.7-0.8: 1.1347
  - 0.8-0.9: 1.1418
  - 0.9-1.0: 1.1410

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.3460, best 1.3186 @ ep 28
  - `Generation/KID_mean_train`: last 1.3444, best 1.3153 @ ep 28
  - `Generation/KID_std_val`: last 0.0231, best 0.0133 @ ep 312
  - `Generation/KID_std_train`: last 0.0160, best 0.0137 @ ep 267
  - `Generation/CMMD_val`: last 0.7869, best 0.7859 @ ep 182
  - `Generation/CMMD_train`: last 0.7605, best 0.7420 @ ep 215
  - `Generation/extended_KID_mean_val`: last 1.3426, best 1.3246 @ ep 24
  - `Generation/extended_KID_mean_train`: last 1.3487, best 1.3266 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.7623, best 0.7618 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.7580, best 0.7510 @ ep 24

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.7771 (min 1.7618, max 1.7979)
  - `Validation/MS-SSIM-3D_restoration`: last 0.2259 (min 0.2250, max 0.2311)
  - `Validation/MS-SSIM_restoration`: last 0.3130 (min 0.1794, max 0.3596)
  - `Validation/PSNR_restoration`: last 11.6161 (min 8.5236, max 12.9430)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0444
  - `Generation_Diversity/extended_MSSSIM`: 0.9619
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.2448
  - `regional_restoration/large`: 0.3783
  - `regional_restoration/medium`: 0.2193
  - `regional_restoration/small`: 0.2553
  - `regional_restoration/tiny`: 0.2766
  - `regional_restoration/tumor_bg_ratio`: 1.3104
  - `regional_restoration/tumor_loss`: 0.3208

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 1.494e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0022, max 0.0852 @ ep 1
  - `training/grad_norm_max`: last 0.0125, max 0.4492 @ ep 1

#### `exp33_3_rflow_bridge_noise_restoration_20260413-014148`
*started 2026-04-13 01:41 • 322 epochs • 85h25m • 10315.7 TFLOPs • peak VRAM 42.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0020 → 0.0002746 (min 0.0002627 @ ep 319)
  - `Loss/MSE_val`: 0.9560 → 0.9081 (min 0.8981 @ ep 29)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.9153
  - 0.1-0.2: 0.9125
  - 0.2-0.3: 0.9067
  - 0.3-0.4: 0.9086
  - 0.4-0.5: 0.9025
  - 0.5-0.6: 0.9076
  - 0.6-0.7: 0.9074
  - 0.7-0.8: 0.9086
  - 0.8-0.9: 0.9109
  - 0.9-1.0: 0.9080

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.2988, best 1.2772 @ ep 296
  - `Generation/KID_mean_train`: last 1.2992, best 1.2792 @ ep 296
  - `Generation/KID_std_val`: last 0.0235, best 0.0153 @ ep 224
  - `Generation/KID_std_train`: last 0.0240, best 0.0149 @ ep 69
  - `Generation/CMMD_val`: last 0.7863, best 0.7846 @ ep 56
  - `Generation/CMMD_train`: last 0.7547, best 0.7412 @ ep 53
  - `Generation/extended_KID_mean_val`: last 1.3023, best 1.2933 @ ep 224
  - `Generation/extended_KID_mean_train`: last 1.2988, best 1.2916 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.7612, best 0.7609 @ ep 74
  - `Generation/extended_CMMD_train`: last 0.7575, best 0.7510 @ ep 274

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.7835 (min 1.7557, max 1.7963)
  - `Validation/MS-SSIM-3D_restoration`: last 0.2430 (min 0.2337, max 0.2442)
  - `Validation/MS-SSIM_restoration`: last 0.2524 (min 0.2057, max 0.3795)
  - `Validation/PSNR_restoration`: last 10.5620 (min 9.2441, max 13.2283)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0458
  - `Generation_Diversity/extended_MSSSIM`: 0.9589
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.3068
  - `regional_restoration/large`: 0.5054
  - `regional_restoration/medium`: 0.3220
  - `regional_restoration/small`: 0.2737
  - `regional_restoration/tiny`: 0.3448
  - `regional_restoration/tumor_bg_ratio`: 1.3426
  - `regional_restoration/tumor_loss`: 0.4120

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 2.988e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0064, max 0.1221 @ ep 0
  - `training/grad_norm_max`: last 0.0344, max 0.3792 @ ep 0

#### `exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630`
*started 2026-04-14 04:46 • 266 epochs • 55h33m • 1689.1 TFLOPs • peak VRAM 20.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0026 → 0.0003136 (min 0.0003095 @ ep 248)
  - `Loss/MSE_val`: 0.9831 → 0.9021 (min 0.9008 @ ep 260)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.9041
  - 0.1-0.2: 0.9025
  - 0.2-0.3: 0.8994
  - 0.3-0.4: 0.9022
  - 0.4-0.5: 0.8995
  - 0.5-0.6: 0.9004
  - 0.6-0.7: 0.9020
  - 0.7-0.8: 0.9028
  - 0.8-0.9: 0.9065
  - 0.9-1.0: 0.9017

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.3005, best 1.2812 @ ep 131
  - `Generation/KID_mean_train`: last 1.2987, best 1.2798 @ ep 131
  - `Generation/KID_std_val`: last 0.0203, best 0.0156 @ ep 112
  - `Generation/KID_std_train`: last 0.0197, best 0.0157 @ ep 151
  - `Generation/CMMD_val`: last 0.7859, best 0.7844 @ ep 96
  - `Generation/CMMD_train`: last 0.7517, best 0.7420 @ ep 123
  - `Generation/extended_KID_mean_val`: last 1.3044, best 1.2980 @ ep 49
  - `Generation/extended_KID_mean_train`: last 1.3003, best 1.2883 @ ep 149
  - `Generation/extended_CMMD_val`: last 0.7613, best 0.7608 @ ep 174
  - `Generation/extended_CMMD_train`: last 0.7533, best 0.7523 @ ep 149

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.7897 (min 1.7625, max 1.7963)
  - `Validation/MS-SSIM-3D_restoration`: last 0.2402 (min 0.2314, max 0.2405)
  - `Validation/MS-SSIM_restoration`: last 0.2167 (min 0.1946, max 0.3689)
  - `Validation/PSNR_restoration`: last 9.5702 (min 8.8837, max 12.9737)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0455
  - `Generation_Diversity/extended_MSSSIM`: 0.9588
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.3651
  - `regional_restoration/large`: 0.4756
  - `regional_restoration/medium`: 0.4156
  - `regional_restoration/small`: 0.5469
  - `regional_restoration/tiny`: 0.4239
  - `regional_restoration/tumor_bg_ratio`: 1.2685
  - `regional_restoration/tumor_loss`: 0.4631

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 4.733e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0103, max 0.0982 @ ep 0
  - `training/grad_norm_max`: last 0.0479, max 0.3706 @ ep 4

#### `exp33_2b_rflow_bridge_restoration_17m_20260415-122714`
*started 2026-04-15 12:27 • 196 epochs • 26h37m • 1242.9 TFLOPs • peak VRAM 20.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0004169 → 8.973e-06 (min 8.973e-06 @ ep 195)
  - `Loss/MSE_val`: 1.0077 → 1.0841 (min 1.0077 @ ep 0)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0531
  - 0.1-0.2: 1.0737
  - 0.2-0.3: 1.0866
  - 0.3-0.4: 1.0873
  - 0.4-0.5: 1.0864
  - 0.5-0.6: 1.0868
  - 0.6-0.7: 1.0822
  - 0.7-0.8: 1.0807
  - 0.8-0.9: 1.0790
  - 0.9-1.0: 1.0808

**Generation metrics:**
  - `Generation/KID_mean_val`: last 1.3187, best 1.3104 @ ep 76
  - `Generation/KID_mean_train`: last 1.3278, best 1.3127 @ ep 45
  - `Generation/KID_std_val`: last 0.0216, best 0.0158 @ ep 192
  - `Generation/KID_std_train`: last 0.0213, best 0.0153 @ ep 193
  - `Generation/CMMD_val`: last 0.7861, best 0.7854 @ ep 24
  - `Generation/CMMD_train`: last 0.7557, best 0.7427 @ ep 42
  - `Generation/extended_KID_mean_val`: last 1.3300, best 1.3226 @ ep 74
  - `Generation/extended_KID_mean_train`: last 1.3339, best 1.3243 @ ep 149
  - `Generation/extended_CMMD_val`: last 0.7620, best 0.7615 @ ep 24
  - `Generation/extended_CMMD_train`: last 0.7573, best 0.7521 @ ep 149

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.7787 (min 1.7653, max 1.7963)
  - `Validation/MS-SSIM-3D_restoration`: last 0.2289 (min 0.2270, max 0.2307)
  - `Validation/MS-SSIM_restoration`: last 0.2916 (min 0.1959, max 0.3535)
  - `Validation/PSNR_restoration`: last 11.1193 (min 8.8386, max 12.7213)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0448
  - `Generation_Diversity/extended_MSSSIM`: 0.9601
  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.2888
  - `regional_restoration/large`: 0.3533
  - `regional_restoration/medium`: 0.4858
  - `regional_restoration/small`: 0.3883
  - `regional_restoration/tiny`: 0.3686
  - `regional_restoration/tumor_bg_ratio`: 1.3334
  - `regional_restoration/tumor_loss`: 0.3852

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 6.908e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0093, max 0.0611 @ ep 2
  - `training/grad_norm_max`: last 0.0311, max 0.2802 @ ep 6

#### `exp33_1_irsde_restoration_20260416-032309`
*started 2026-04-16 03:23 • 25 epochs • 6h32m • 178.2 TFLOPs • peak VRAM 12.2 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0026 → 0.0002639 (min 0.0002633 @ ep 23)
  - `Loss/L1_val`: 0.0038 → 0.0051 (min 0.0038 @ ep 0)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.7610
  - 0.1-0.2: 0.2251
  - 0.2-0.3: 0.1479
  - 0.3-0.4: 0.0819
  - 0.4-0.5: 0.0449
  - 0.5-0.6: 0.0370
  - 0.6-0.7: 0.0341
  - 0.7-0.8: 0.0351
  - 0.8-0.9: 0.0352
  - 0.9-1.0: 0.0353

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.3811 (min 1.3614, max 1.5329)
  - `Validation/MS-SSIM-3D_restoration`: last 0.1729 (min 0.1729, max 0.2578)
  - `Validation/MS-SSIM_restoration`: last 0.3629 (min 0.2905, max 0.4136)
  - `Validation/PSNR_restoration`: last 21.5347 (min 18.6070, max 24.1092)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.1156
  - `regional_restoration/large`: 0.0501
  - `regional_restoration/medium`: 0.0423
  - `regional_restoration/small`: 0.0826
  - `regional_restoration/tiny`: 0.0615
  - `regional_restoration/tumor_bg_ratio`: 0.5037
  - `regional_restoration/tumor_loss`: 0.0582

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 9.96e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0079, max 0.0276 @ ep 2
  - `training/grad_norm_max`: last 0.0173, max 0.0967 @ ep 1

#### `exp33_1b_irsde_restoration_20260416-034412`
*started 2026-04-16 03:44 • 25 epochs • 6h34m • 935.4 TFLOPs • peak VRAM 12.5 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0026 → 0.0002647 (min 0.0002647 @ ep 24)
  - `Loss/L1_val`: 0.0040 → 0.0049 (min 0.0040 @ ep 0)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.8127
  - 0.1-0.2: 0.2003
  - 0.2-0.3: 0.1229
  - 0.3-0.4: 0.0683
  - 0.4-0.5: 0.0494
  - 0.5-0.6: 0.0437
  - 0.6-0.7: 0.0420
  - 0.7-0.8: 0.0421
  - 0.8-0.9: 0.0416
  - 0.9-1.0: 0.0417

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.4230 (min 1.3686, max 1.5495)
  - `Validation/MS-SSIM-3D_restoration`: last 0.1693 (min 0.1693, max 0.2467)
  - `Validation/MS-SSIM_restoration`: last 0.3707 (min 0.2873, max 0.3770)
  - `Validation/PSNR_restoration`: last 22.5099 (min 18.4957, max 22.7240)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.1171
  - `regional_restoration/large`: 0.0021
  - `regional_restoration/medium`: 0.0436
  - `regional_restoration/small`: 0.0749
  - `regional_restoration/tiny`: 0.1023
  - `regional_restoration/tumor_bg_ratio`: 0.5728
  - `regional_restoration/tumor_loss`: 0.0671

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 9.96e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0083, max 0.0298 @ ep 2
  - `training/grad_norm_max`: last 0.0185, max 0.0903 @ ep 1

#### `exp33_4_resfusion_restoration_20260416-050654`
*started 2026-04-16 05:06 • 25 epochs • 5h22m • 178.2 TFLOPs • peak VRAM 12.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.5268 → 0.0159 (min 0.0159 @ ep 24)
  - `Loss/MSE_val`: 0.3887 → 0.0645 (min 0.0641 @ ep 22)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.7721
  - 0.1-0.2: 1.2199
  - 0.2-0.3: 0.8287
  - 0.3-0.4: 0.5710

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.2641 (min 1.2352, max 1.5965)
  - `Validation/MS-SSIM-3D_restoration`: last 7.986e-05 (min 7.537e-05, max 0.0001859)
  - `Validation/MS-SSIM_restoration`: last 0.4544 (min 0.2741, max 0.4544)
  - `Validation/PSNR_restoration`: last 26.0081 (min 18.5508, max 26.0081)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.0362
  - `regional_restoration/large`: 0.0315
  - `regional_restoration/medium`: 0.0295
  - `regional_restoration/small`: 0.0861
  - `regional_restoration/tiny`: 0.0496
  - `regional_restoration/tumor_bg_ratio`: 1.4838
  - `regional_restoration/tumor_loss`: 0.0538

**LR schedule:**
  - `LR/Generator`: peak 0.00011 @ ep 4, final 0.0001096

**Training meta:**
  - `training/grad_norm_avg`: last 0.0696, max 3.6985 @ ep 0
  - `training/grad_norm_max`: last 0.1687, max 5.0988 @ ep 0

#### `exp33_1c_irsde_restoration_17m_20260416-060302`
*started 2026-04-16 06:03 • 59 epochs • 9h04m • 376.1 TFLOPs • peak VRAM 19.6 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0035 → 0.0003517 (min 0.0003422 @ ep 54)
  - `Loss/L1_val`: 0.0036 → 0.0051 (min 0.0034 @ ep 27)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.9801
  - 0.1-0.2: 0.2309
  - 0.2-0.3: 0.0636
  - 0.3-0.4: 0.0593
  - 0.4-0.5: 0.0632
  - 0.5-0.6: 0.0723
  - 0.6-0.7: 0.0789
  - 0.7-0.8: 0.0812
  - 0.8-0.9: 0.0829
  - 0.9-1.0: 0.0833

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.4528 (min 1.2769, max 1.6669)
  - `Validation/MS-SSIM-3D_restoration`: last 0.6793 (min 0.6713, max 0.7630)
  - `Validation/MS-SSIM_restoration`: last 0.5348 (min 0.4856, max 0.7719)
  - `Validation/PSNR_restoration`: last 21.3918 (min 16.7943, max 30.3351)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.1435
  - `regional_restoration/large`: 0.1339
  - `regional_restoration/medium`: 0.0637
  - `regional_restoration/small`: 0.0716
  - `regional_restoration/tiny`: 0.1339
  - `regional_restoration/tumor_bg_ratio`: 0.8111
  - `regional_restoration/tumor_loss`: 0.1164

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 9.758e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0186, max 0.0247 @ ep 39
  - `training/grad_norm_max`: last 0.0676, max 0.1205 @ ep 46

#### `exp33_4c_resfusion_restoration_17m_20260416-075251`
*started 2026-04-16 07:52 • 50 epochs • 7h13m • 318.7 TFLOPs • peak VRAM 19.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9821 → 0.0130 (min 0.0092 @ ep 40)
  - `Loss/MSE_val`: 1.3803 → 0.0639 (min 0.0441 @ ep 36)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.7663
  - 0.1-0.2: 1.2242
  - 0.2-0.3: 0.8180
  - 0.3-0.4: 0.5526

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.3393 (min 1.1093, max 1.7650)
  - `Validation/MS-SSIM-3D_restoration`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_restoration`: last 0.6647 (min 0.3629, max 0.8700)
  - `Validation/PSNR_restoration`: last 23.9517 (min 12.6212, max 31.1392)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.0346
  - `regional_restoration/large`: 0.0536
  - `regional_restoration/medium`: 0.0909
  - `regional_restoration/small`: 0.0528
  - `regional_restoration/tiny`: 0.0481
  - `regional_restoration/tumor_bg_ratio`: 1.7338
  - `regional_restoration/tumor_loss`: 0.0600

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 9.838e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.3689, max 5.3926 @ ep 9
  - `training/grad_norm_max`: last 3.5775, max 38.3336 @ ep 12

#### `exp33_5_bridge_restoration_17m_20260416-081254`
*started 2026-04-16 08:12 • 75 epochs • 6h41m • 478.0 TFLOPs • peak VRAM 27.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0021 → 0.0002316 (min 0.0002091 @ ep 73)
  - `Loss/MSE_val`: 0.0033 → 0.0021 (min 0.0016 @ ep 25)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.0014
  - 0.1-0.2: 1.0037
  - 0.2-0.3: 1.0013
  - 0.3-0.4: 0.9983
  - 0.4-0.5: 0.9965
  - 0.5-0.6: 0.9927
  - 0.6-0.7: 0.9922
  - 0.7-0.8: 0.9921
  - 0.8-0.9: 0.9916
  - 0.9-1.0: 0.9927

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.5089 (min 1.2174, max 1.6834)
  - `Validation/MS-SSIM-3D_restoration`: last 0.5278 (min 0.4490, max 0.5996)
  - `Validation/MS-SSIM_restoration`: last 0.8007 (min 0.5775, max 0.8433)
  - `Validation/PSNR_restoration`: last 28.0991 (min 24.3032, max 29.7812)

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.0021
  - `regional_restoration/large`: 0.0422
  - `regional_restoration/medium`: 0.0564
  - `regional_restoration/small`: 0.0327
  - `regional_restoration/tiny`: 0.0276
  - `regional_restoration/tumor_bg_ratio`: 19.7358
  - `regional_restoration/tumor_loss`: 0.0409

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 9, final 9.576e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0421, max 0.5403 @ ep 0
  - `training/grad_norm_max`: last 0.2367, max 1.9556 @ ep 0

---
## diffusion_3d/cfg

*9 runs across 5 experiment families.*

### exp1

**exp1 (CFG debugging)** — early classifier-free-guidance ablations at
128×128×160. One run NaN'd `Generation/CMMD_*`.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp1_pixel_bravo_rflow_128x160_20260128-223458` — 0.0120
  2. 🥈 `exp1_debugging_cfg_rflow_128x160_20260128-223445` — 0.0146

#### `exp1_debugging_cfg_rflow_128x160_20260128-223445`
*started 2026-01-28 22:34 • 500 epochs • 10h00m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9675 → 0.0015 (min 0.0012 @ ep 466)
  - `Loss/MSE_val`: 0.9237 → 0.0099 (min 0.0022 @ ep 161)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0364
  - 0.1-0.2: 0.0370
  - 0.2-0.3: 0.0096
  - 0.3-0.4: 0.0058
  - 0.4-0.5: 0.0048
  - 0.5-0.6: 0.0047
  - 0.6-0.7: 0.0033
  - 0.7-0.8: 0.0022
  - 0.8-0.9: 0.0018
  - 0.9-1.0: 0.0019

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0423, best 0.0119 @ ep 410
  - `Generation/KID_mean_train`: last 0.0409, best 0.0134 @ ep 342
  - `Generation/KID_std_val`: last 0.0019, best 0.0014 @ ep 328
  - `Generation/KID_std_train`: last 0.0028, best 0.0012 @ ep 415
  - `Generation/CMMD_val`: last 0.2548, best 0.1783 @ ep 245
  - `Generation/CMMD_train`: last 0.2553, best 0.1734 @ ep 236
  - `Generation/extended_KID_mean_val`: last 0.0146, best 0.0146 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0133, best 0.0133 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.2216, best 0.1921 @ ep 174
  - `Generation/extended_CMMD_train`: last 0.2249, best 0.1894 @ ep 174

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.4893 (min 0.4665, max 1.8782)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9106 (min 0.2858, max 0.9571)
  - `Validation/MS-SSIM_bravo`: last 0.9191 (min 0.3331, max 0.9644)
  - `Validation/PSNR_bravo`: last 30.6681 (min 10.9447, max 33.7777)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3908
  - `Generation_Diversity/extended_MSSSIM`: 0.1500

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009637
  - `regional_bravo/large`: 0.0078
  - `regional_bravo/medium`: 0.0158
  - `regional_bravo/small`: 0.0144
  - `regional_bravo/tiny`: 0.0095
  - `regional_bravo/tumor_bg_ratio`: 12.6551
  - `regional_bravo/tumor_loss`: 0.0122

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0183, max 3.8795 @ ep 0
  - `training/grad_norm_max`: last 0.0902, max 10.1273 @ ep 4

#### `exp1_pixel_bravo_rflow_128x160_20260128-223458`
*started 2026-01-28 22:34 • 500 epochs • 10h06m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9676 → 0.0015 (min 0.0012 @ ep 485)
  - `Loss/MSE_val`: 0.9247 → 0.0064 (min 0.0024 @ ep 160)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0313
  - 0.1-0.2: 0.0193
  - 0.2-0.3: 0.0113
  - 0.3-0.4: 0.0079
  - 0.4-0.5: 0.0048
  - 0.5-0.6: 0.0031
  - 0.6-0.7: 0.0025
  - 0.7-0.8: 0.0017
  - 0.8-0.9: 0.0022
  - 0.9-1.0: 0.0018

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0317, best 0.0105 @ ep 254
  - `Generation/KID_mean_train`: last 0.0285, best 0.0094 @ ep 254
  - `Generation/KID_std_val`: last 0.0029, best 0.0013 @ ep 308
  - `Generation/KID_std_train`: last 0.0028, best 0.0013 @ ep 308
  - `Generation/CMMD_val`: last 0.2626, best 0.1755 @ ep 205
  - `Generation/CMMD_train`: last 0.2618, best 0.1739 @ ep 205
  - `Generation/extended_KID_mean_val`: last 0.0147, best 0.0120 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.0133, best 0.0100 @ ep 424
  - `Generation/extended_CMMD_val`: last 0.2094, best 0.1969 @ ep 224
  - `Generation/extended_CMMD_train`: last 0.2124, best 0.1962 @ ep 224

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5039 (min 0.4403, max 1.8797)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9143 (min 0.2856, max 0.9574)
  - `Validation/MS-SSIM_bravo`: last 0.9164 (min 0.3105, max 0.9613)
  - `Validation/PSNR_bravo`: last 30.4478 (min 10.4218, max 33.9451)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3243
  - `Generation_Diversity/extended_MSSSIM`: 0.1466

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009889
  - `regional_bravo/large`: 0.0098
  - `regional_bravo/medium`: 0.0163
  - `regional_bravo/small`: 0.0232
  - `regional_bravo/tiny`: 0.0113
  - `regional_bravo/tumor_bg_ratio`: 15.0779
  - `regional_bravo/tumor_loss`: 0.0149

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0172, max 3.8697 @ ep 0
  - `training/grad_norm_max`: last 0.0867, max 5.0602 @ ep 2

### exp4

**exp4** — SDA with CFG at 128.

#### `exp4_pixel_bravo_sda_rflow_128x160_20260129-022356`
*started 2026-01-29 02:23 • 400 epochs • 16h11m • 1563.3 TFLOPs • peak VRAM 24.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9700 → 0.0019 (min 0.0013 @ ep 379)
  - `Loss/MSE_val`: 0.9307 → 0.0050 (min 0.0025 @ ep 148)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0669
  - 0.1-0.2: 0.0155
  - 0.2-0.3: 0.0109
  - 0.3-0.4: 0.0067
  - 0.4-0.5: 0.0050
  - 0.5-0.6: 0.0045
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0016
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0267, best 0.0134 @ ep 266
  - `Generation/KID_mean_train`: last 0.0249, best 0.0116 @ ep 266
  - `Generation/KID_std_val`: last 0.0019, best 0.0018 @ ep 396
  - `Generation/KID_std_train`: last 0.0017, best 0.0016 @ ep 392
  - `Generation/CMMD_val`: last 0.2297, best 0.1766 @ ep 235
  - `Generation/CMMD_train`: last 0.2299, best 0.1734 @ ep 235
  - `Generation/extended_KID_mean_val`: last 0.0261, best 0.0261 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0229, best 0.0229 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.1850, best 0.1850 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.1843, best 0.1843 @ ep 399

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6083 (min 0.5304, max 1.8689)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9162 (min 0.2832, max 0.9550)
  - `Validation/MS-SSIM_bravo`: last 0.9158 (min 0.2686, max 0.9586)
  - `Validation/PSNR_bravo`: last 30.2711 (min 9.4792, max 33.3808)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.6335
  - `Generation_Diversity/extended_MSSSIM`: 0.1259

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009822
  - `regional_bravo/large`: 0.0118
  - `regional_bravo/medium`: 0.0162
  - `regional_bravo/small`: 0.0113
  - `regional_bravo/tiny`: 0.0081
  - `regional_bravo/tumor_bg_ratio`: 12.9061
  - `regional_bravo/tumor_loss`: 0.0127

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.083e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.1245, max 4.7394 @ ep 2
  - `training/grad_norm_max`: last 0.5485, max 65.6567 @ ep 116

### exp5

**exp5** — ScoreAug + CFG at 128.

#### `exp5_1_pixel_bravo_scoreaug_rflow_128x160_20260129-022356`
*started 2026-01-29 02:23 • 500 epochs • 14h38m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9067 → 0.0031 (min 0.0023 @ ep 361)
  - `Loss/MSE_val`: 0.9246 → 0.0026 (min 0.0021 @ ep 337)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0293
  - 0.1-0.2: 0.0075
  - 0.2-0.3: 0.0043
  - 0.3-0.4: 0.0028
  - 0.4-0.5: 0.0026
  - 0.5-0.6: 0.0017
  - 0.6-0.7: 0.0016
  - 0.7-0.8: 0.0018
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0016

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0694, best 0.0199 @ ep 314
  - `Generation/KID_mean_train`: last 0.0661, best 0.0164 @ ep 314
  - `Generation/KID_std_val`: last 0.0028, best 0.0017 @ ep 413
  - `Generation/KID_std_train`: last 0.0025, best 0.0016 @ ep 287
  - `Generation/CMMD_val`: last 0.2092, best 0.1678 @ ep 397
  - `Generation/CMMD_train`: last 0.2019, best 0.1612 @ ep 397
  - `Generation/extended_KID_mean_val`: last 0.0547, best 0.0197 @ ep 374
  - `Generation/extended_KID_mean_train`: last 0.0520, best 0.0163 @ ep 374
  - `Generation/extended_CMMD_val`: last 0.2003, best 0.1825 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.1991, best 0.1803 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.7287 (min 0.4924, max 1.8808)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9596 (min 0.2830, max 0.9602)
  - `Validation/MS-SSIM_bravo`: last 0.9442 (min 0.3521, max 0.9706)
  - `Validation/PSNR_bravo`: last 32.6637 (min 11.2287, max 34.5114)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4207
  - `Generation_Diversity/extended_MSSSIM`: 0.2450

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0005933
  - `regional_bravo/large`: 0.0091
  - `regional_bravo/medium`: 0.0140
  - `regional_bravo/small`: 0.0134
  - `regional_bravo/tiny`: 0.0098
  - `regional_bravo/tumor_bg_ratio`: 19.9202
  - `regional_bravo/tumor_loss`: 0.0118

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0381, max 3.7609 @ ep 0
  - `training/grad_norm_max`: last 0.2152, max 27.6189 @ ep 39

### exp6

**exp6 ControlNet stage-1 + CFG at 128** — cfg-variant of the exp6 family.

#### `exp6a_pixel_bravo_controlnet_stage1_rflow_128x160_20260129-063540`
*started 2026-01-29 06:35 • 500 epochs • 9h53m • 1954.2 TFLOPs • peak VRAM 21.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.9682 → 0.0016 (min 0.0012 @ ep 451)
  - `Loss/MSE_val`: 0.9256 → 0.0047 (min 0.0024 @ ep 189)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.1215
  - 0.1-0.2: 0.0193
  - 0.2-0.3: 0.0109
  - 0.3-0.4: 0.0079
  - 0.4-0.5: 0.0049
  - 0.5-0.6: 0.0039
  - 0.6-0.7: 0.0027
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0013
  - 0.9-1.0: 0.0022

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5292 (min 0.4759, max 1.8825)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9163 (min 0.2876, max 0.9565)
  - `Validation/MS-SSIM_bravo`: last 0.9137 (min 0.3099, max 0.9627)
  - `Validation/PSNR_bravo`: last 30.2282 (min 10.3789, max 33.5388)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0009911
  - `regional_bravo/large`: 0.0105
  - `regional_bravo/medium`: 0.0202
  - `regional_bravo/small`: 0.0141
  - `regional_bravo/tiny`: 0.0085
  - `regional_bravo/tumor_bg_ratio`: 14.4210
  - `regional_bravo/tumor_loss`: 0.0143

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0220, max 3.8955 @ ep 0
  - `training/grad_norm_max`: last 0.1056, max 8.1315 @ ep 2

### exp7

**exp7 (SiT + CFG)** — SiT (DiT) architecture sweep at 128: S, B, L, XL
with patch 8. Per memory: CFG-Zero* doesn't help — best scale=1.0 (no
guidance), all scales >1.0 dramatically worse.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp7_sit_l_128_patch8_rflow_128x160_20260129-083948` — 0.0579
  2. 🥈 `exp7_sit_xl_128_patch8_rflow_128x160_20260129-084435` — 0.0646
  3.  `exp7_sit_b_128_patch8_rflow_128x160_20260129-074908` — 0.0654
  4.  `exp7_sit_s_128_patch8_rflow_128x160_20260129-072436` — 0.6962

#### `exp7_sit_s_128_patch8_rflow_128x160_20260129-072436`
*started 2026-01-29 07:24 • 500 epochs • 3h01m • 34589.5 TFLOPs • peak VRAM 5.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0054 → 0.2668 (min 0.2662 @ ep 442)
  - `Loss/MSE_val`: 1.0041 → 0.2664 (min 0.2659 @ ep 478)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.2803
  - 0.1-0.2: 0.2698
  - 0.2-0.3: 0.2674
  - 0.3-0.4: 0.2671
  - 0.4-0.5: 0.2664
  - 0.5-0.6: 0.2656
  - 0.6-0.7: 0.2654
  - 0.7-0.8: 0.2656
  - 0.8-0.9: 0.2651
  - 0.9-1.0: 0.2657

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.6934, best 0.6838 @ ep 452
  - `Generation/KID_mean_train`: last 0.6934, best 0.6849 @ ep 452
  - `Generation/KID_std_val`: last 0.0106, best 0.0079 @ ep 226
  - `Generation/KID_std_train`: last 0.0123, best 0.0075 @ ep 223
  - `Generation/CMMD_val`: last 0.8193, best 0.8064 @ ep 150
  - `Generation/CMMD_train`: last 0.8203, best 0.8078 @ ep 150
  - `Generation/extended_KID_mean_val`: last 0.7004, best 0.6962 @ ep 449
  - `Generation/extended_KID_mean_train`: last 0.6973, best 0.6967 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.8150, best 0.8104 @ ep 249
  - `Generation/extended_CMMD_train`: last 0.8214, best 0.8174 @ ep 249

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 1.8773 (min 1.8434, max 1.8926)
  - `Validation/MS-SSIM-3D_bravo`: last 0.4333 (min 0.2862, max 0.4633)
  - `Validation/MS-SSIM_bravo`: last 0.4822 (min 0.3362, max 0.5941)
  - `Validation/PSNR_bravo`: last 15.5909 (min 10.4533, max 18.1609)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.0711
  - `Generation_Diversity/extended_MSSSIM`: 0.9435

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0370
  - `regional_bravo/large`: 0.0452
  - `regional_bravo/medium`: 0.0520
  - `regional_bravo/small`: 0.0445
  - `regional_bravo/tiny`: 0.0395
  - `regional_bravo/tumor_bg_ratio`: 1.2571
  - `regional_bravo/tumor_loss`: 0.0466

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0378, max 1.7822 @ ep 3
  - `training/grad_norm_max`: last 0.0511, max 12.8607 @ ep 6

#### `exp7_sit_b_128_patch8_rflow_128x160_20260129-074908`
*started 2026-01-29 07:49 • 500 epochs • 3h32m • 137677.2 TFLOPs • peak VRAM 8.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0044 → 0.0039 (min 0.0033 @ ep 450)
  - `Loss/MSE_val`: 1.0028 → 0.0046 (min 0.0026 @ ep 352)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0293
  - 0.1-0.2: 0.0151
  - 0.2-0.3: 0.0082
  - 0.3-0.4: 0.0045
  - 0.4-0.5: 0.0038
  - 0.5-0.6: 0.0033
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0011
  - 0.9-1.0: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0547, best 0.0500 @ ep 479
  - `Generation/KID_mean_train`: last 0.0494, best 0.0457 @ ep 483
  - `Generation/KID_std_val`: last 0.0058, best 0.0038 @ ep 382
  - `Generation/KID_std_train`: last 0.0057, best 0.0032 @ ep 382
  - `Generation/CMMD_val`: last 0.3352, best 0.3026 @ ep 397
  - `Generation/CMMD_train`: last 0.3323, best 0.2987 @ ep 397
  - `Generation/extended_KID_mean_val`: last 0.0670, best 0.0654 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0613, best 0.0606 @ ep 474
  - `Generation/extended_CMMD_val`: last 0.3877, best 0.3877 @ ep 499
  - `Generation/extended_CMMD_train`: last 0.3886, best 0.3886 @ ep 499

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6243 (min 0.5796, max 1.8810)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9330 (min 0.3008, max 0.9334)
  - `Validation/MS-SSIM_bravo`: last 0.9348 (min 0.3795, max 0.9524)
  - `Validation/PSNR_bravo`: last 31.8057 (min 11.1744, max 32.7190)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3856
  - `Generation_Diversity/extended_MSSSIM`: 0.1579

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007254
  - `regional_bravo/large`: 0.0101
  - `regional_bravo/medium`: 0.0130
  - `regional_bravo/small`: 0.0121
  - `regional_bravo/tiny`: 0.0098
  - `regional_bravo/tumor_bg_ratio`: 15.7741
  - `regional_bravo/tumor_loss`: 0.0114

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0061, max 2.9148 @ ep 3
  - `training/grad_norm_max`: last 0.0181, max 14.5234 @ ep 4

#### `exp7_sit_l_128_patch8_rflow_128x160_20260129-083948`
*started 2026-01-29 08:39 • 500 epochs • 5h56m • 488064.9 TFLOPs • peak VRAM 13.8 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0040 → 0.0043 (min 0.0031 @ ep 406)
  - `Loss/MSE_val`: 1.0015 → 0.0059 (min 0.0028 @ ep 320)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0224
  - 0.1-0.2: 0.0154
  - 0.2-0.3: 0.0092
  - 0.3-0.4: 0.0046
  - 0.4-0.5: 0.0038
  - 0.5-0.6: 0.0025
  - 0.6-0.7: 0.0023
  - 0.7-0.8: 0.0014
  - 0.8-0.9: 0.0016
  - 0.9-1.0: 0.0011

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0500, best 0.0478 @ ep 485
  - `Generation/KID_mean_train`: last 0.0454, best 0.0423 @ ep 467
  - `Generation/KID_std_val`: last 0.0038, best 0.0037 @ ep 384
  - `Generation/KID_std_train`: last 0.0035, best 0.0029 @ ep 406
  - `Generation/CMMD_val`: last 0.3459, best 0.2809 @ ep 463
  - `Generation/CMMD_train`: last 0.3439, best 0.2720 @ ep 463
  - `Generation/extended_KID_mean_val`: last 0.0579, best 0.0579 @ ep 499
  - `Generation/extended_KID_mean_train`: last 0.0540, best 0.0540 @ ep 499
  - `Generation/extended_CMMD_val`: last 0.3676, best 0.3495 @ ep 424
  - `Generation/extended_CMMD_train`: last 0.3674, best 0.3481 @ ep 424

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6051 (min 0.5706, max 1.8841)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9334 (min 0.3102, max 0.9342)
  - `Validation/MS-SSIM_bravo`: last 0.9384 (min 0.3349, max 0.9453)
  - `Validation/PSNR_bravo`: last 32.1329 (min 9.9995, max 32.4900)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.4381
  - `Generation_Diversity/extended_MSSSIM`: 0.1999

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0006676
  - `regional_bravo/large`: 0.0190
  - `regional_bravo/medium`: 0.0100
  - `regional_bravo/small`: 0.0127
  - `regional_bravo/tiny`: 0.0074
  - `regional_bravo/tumor_bg_ratio`: 18.9912
  - `regional_bravo/tumor_loss`: 0.0127

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0060, max 2.2338 @ ep 3
  - `training/grad_norm_max`: last 0.0200, max 11.3531 @ ep 7

#### `exp7_sit_xl_128_patch8_rflow_128x160_20260129-084435`
*started 2026-01-29 08:44 • 500 epochs • 8h43m • 720340.4 TFLOPs • peak VRAM 18.6 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0036 → 0.0053 (min 0.0031 @ ep 453)
  - `Loss/MSE_val`: 1.0012 → 0.0036 (min 0.0028 @ ep 355)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0703
  - 0.1-0.2: 0.0174
  - 0.2-0.3: 0.0097
  - 0.3-0.4: 0.0054
  - 0.4-0.5: 0.0026
  - 0.5-0.6: 0.0026
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0015
  - 0.8-0.9: 0.0015
  - 0.9-1.0: 0.0009417

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0556, best 0.0495 @ ep 465
  - `Generation/KID_mean_train`: last 0.0501, best 0.0439 @ ep 465
  - `Generation/KID_std_val`: last 0.0063, best 0.0036 @ ep 404
  - `Generation/KID_std_train`: last 0.0056, best 0.0030 @ ep 454
  - `Generation/CMMD_val`: last 0.3374, best 0.2777 @ ep 445
  - `Generation/CMMD_train`: last 0.3328, best 0.2717 @ ep 445
  - `Generation/extended_KID_mean_val`: last 0.0678, best 0.0646 @ ep 474
  - `Generation/extended_KID_mean_train`: last 0.0597, best 0.0593 @ ep 449
  - `Generation/extended_CMMD_val`: last 0.3814, best 0.3663 @ ep 449
  - `Generation/extended_CMMD_train`: last 0.3813, best 0.3660 @ ep 449

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6943 (min 0.5730, max 1.8817)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9328 (min 0.3146, max 0.9334)
  - `Validation/MS-SSIM_bravo`: last 0.9251 (min 0.4171, max 0.9489)
  - `Validation/PSNR_bravo`: last 31.3696 (min 11.9009, max 32.4759)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.3905
  - `Generation_Diversity/extended_MSSSIM`: 0.1637

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0007619
  - `regional_bravo/large`: 0.0114
  - `regional_bravo/medium`: 0.0131
  - `regional_bravo/small`: 0.0162
  - `regional_bravo/tiny`: 0.0088
  - `regional_bravo/tumor_bg_ratio`: 16.3817
  - `regional_bravo/tumor_loss`: 0.0125

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0065, max 2.1349 @ ep 2
  - `training/grad_norm_max`: last 0.0196, max 8.7124 @ ep 4

---
## compression_3d/multi_modality

*23 runs across 4 experiment families.*

### exp7

**exp7 (3D VAE — 128 → 256)** — initial 3D VAE compression at 128×128×160,
then trained at 128 and validated at 256 to see upscaling behavior.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp7_128x128x160_128x160_20251231-175024` — 33.6830
  2. 🥈 `exp7_train128_val256_256x160_20260106-131830` — 31.5676

#### `exp7_128x128x160_128x160_20251230-003632`
*started 2025-12-30 00:36 • 125 epochs • 51h43m*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0345 → 0.0045 (min 0.0045 @ ep 124)

**Validation quality:**
  - `Validation/L1`: last 0.0054 (min 0.0054, max 0.0233)
  - `Validation/PSNR`: last 0.4514 (min 0.3311, max 0.4629)

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_g_avg`: last 0.2578, max 10.8846 @ ep 16
  - `training/grad_norm_g_max`: last 0.7519, max 31.0381 @ ep 53

#### `exp7_128x128x160_128x160_20251231-175024`
*started 2025-12-31 17:50 • 125 epochs • 79h31m • 69.8 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0440 → 0.0048 (min 0.0048 @ ep 124)
  - `Loss/L1_val`: 0.0248 → 0.0059 (min 0.0059 @ ep 121)

**Validation quality:**
  - `Validation/LPIPS`: last 0.2899 (min 0.2853, max 1.2481)
  - `Validation/LPIPS_bravo`: last 0.3237 (min 0.3208, max 1.2587)
  - `Validation/LPIPS_flair`: last 0.3670 (min 0.3618, max 1.2878)
  - `Validation/LPIPS_t1_gd`: last 0.2025 (min 0.1970, max 1.2696)
  - `Validation/LPIPS_t1_pre`: last 0.2666 (min 0.2610, max 1.1873)
  - `Validation/MS-SSIM`: last 0.9753 (min 0.7496, max 0.9753)
  - `Validation/MS-SSIM_bravo`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_flair`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_t1_gd`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_t1_pre`: last 0 (min 0, max 0)
  - `Validation/PSNR`: last 33.9207 (min 25.9535, max 33.9617)
  - `Validation/PSNR_bravo`: last 33.6707 (min 27.4004, max 33.6830)
  - `Validation/PSNR_flair`: last 33.1884 (min 25.9819, max 33.2176)
  - `Validation/PSNR_t1_gd`: last 35.4138 (min 26.8884, max 35.4616)
  - `Validation/PSNR_t1_pre`: last 33.4076 (min 23.5317, max 33.5012)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0055
  - `regional_bravo/large`: 0.0662
  - `regional_bravo/medium`: 0.0733
  - `regional_bravo/small`: 0.0628
  - `regional_bravo/tiny`: 0.0508
  - `regional_bravo/tumor_bg_ratio`: 11.8637
  - `regional_bravo/tumor_loss`: 0.0656
  - `regional_flair/background_loss`: 0.0066
  - `regional_flair/large`: 0.0882
  - `regional_flair/medium`: 0.0833
  - `regional_flair/small`: 0.0670
  - `regional_flair/tiny`: 0.0614
  - `regional_flair/tumor_bg_ratio`: 11.8392
  - `regional_flair/tumor_loss`: 0.0780
  - `regional_t1_gd/background_loss`: 0.0047
  - `regional_t1_gd/large`: 0.0656
  - `regional_t1_gd/medium`: 0.0706
  - `regional_t1_gd/small`: 0.0668
  - `regional_t1_gd/tiny`: 0.0461
  - `regional_t1_gd/tumor_bg_ratio`: 13.8151
  - `regional_t1_gd/tumor_loss`: 0.0644
  - `regional_t1_pre/background_loss`: 0.0063
  - `regional_t1_pre/large`: 0.0649
  - `regional_t1_pre/medium`: 0.0372
  - `regional_t1_pre/small`: 0.0463
  - `regional_t1_pre/tiny`: 0.0357
  - `regional_t1_pre/tumor_bg_ratio`: 7.4153
  - `regional_t1_pre/tumor_loss`: 0.0467

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_g_avg`: last 0.2156, max 9.8164 @ ep 11
  - `training/grad_norm_g_max`: last 0.7459, max 19.1592 @ ep 12

#### `exp7_train128_val256_256x160_20260106-131830`
*started 2026-01-06 13:18 • 125 epochs • 44h02m • 69.8 TFLOPs • peak VRAM 40.0 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0351 → 0.0048 (min 0.0048 @ ep 124)
  - `Loss/L1_val`: 0.0302 → 0.0122 (min 0.0093 @ ep 45)

**Validation quality:**
  - `Validation/LPIPS`: last 0.3084 (min 0.2630, max 1.1575)
  - `Validation/LPIPS_bravo`: last 0.3318 (min 0.3214, max 1.1820)
  - `Validation/LPIPS_flair`: last 0.4149 (min 0.3291, max 1.2009)
  - `Validation/LPIPS_t1_gd`: last 0.2371 (min 0.1902, max 1.1148)
  - `Validation/LPIPS_t1_pre`: last 0.2498 (min 0.1900, max 1.1349)
  - `Validation/MS-SSIM`: last 0.9453 (min 0.8053, max 0.9529)
  - `Validation/MS-SSIM-3D`: last 0.9461 (min 0.7554, max 0.9551)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9444 (min 0.7566, max 0.9517)
  - `Validation/MS-SSIM-3D_flair`: last 0.9380 (min 0.7514, max 0.9457)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9535 (min 0.7710, max 0.9631)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9488 (min 0.7405, max 0.9614)
  - `Validation/MS-SSIM_bravo`: last 0.9430 (min 0.8057, max 0.9491)
  - `Validation/MS-SSIM_flair`: last 0.9387 (min 0.8014, max 0.9445)
  - `Validation/MS-SSIM_t1_gd`: last 0.9526 (min 0.8167, max 0.9613)
  - `Validation/MS-SSIM_t1_pre`: last 0.9468 (min 0.7975, max 0.9577)
  - `Validation/PSNR`: last 30.3425 (min 25.1565, max 31.8853)
  - `Validation/PSNR_bravo`: last 30.5718 (min 25.8292, max 31.5676)
  - `Validation/PSNR_flair`: last 29.9988 (min 25.0304, max 31.1657)
  - `Validation/PSNR_t1_gd`: last 31.5925 (min 26.1152, max 33.3472)
  - `Validation/PSNR_t1_pre`: last 29.2116 (min 23.6465, max 31.5296)

**Regional loss (final):**
  - `regional/background_loss`: 0.0122
  - `regional/large`: 0.0821
  - `regional/medium`: 0.0715
  - `regional/small`: 0.0662
  - `regional/tiny`: 0.0613
  - `regional/tumor_bg_ratio`: 5.9238
  - `regional/tumor_loss`: 0.0720
  - `regional_bravo/background_loss`: 0.0098
  - `regional_bravo/large`: 0.0726
  - `regional_bravo/medium`: 0.0757
  - `regional_bravo/small`: 0.0642
  - `regional_bravo/tiny`: 0.0605
  - `regional_bravo/tumor_bg_ratio`: 7.1195
  - `regional_bravo/tumor_loss`: 0.0695
  - `regional_flair/background_loss`: 0.0127
  - `regional_flair/large`: 0.1141
  - `regional_flair/medium`: 0.0928
  - `regional_flair/small`: 0.0698
  - `regional_flair/tiny`: 0.0689
  - `regional_flair/tumor_bg_ratio`: 7.1497
  - `regional_flair/tumor_loss`: 0.0908
  - `regional_t1_gd/background_loss`: 0.0102
  - `regional_t1_gd/large`: 0.0703
  - `regional_t1_gd/medium`: 0.0649
  - `regional_t1_gd/small`: 0.0655
  - `regional_t1_gd/tiny`: 0.0550
  - `regional_t1_gd/tumor_bg_ratio`: 6.3592
  - `regional_t1_gd/tumor_loss`: 0.0648
  - `regional_t1_pre/background_loss`: 0.0159
  - `regional_t1_pre/large`: 0.0712
  - `regional_t1_pre/medium`: 0.0529
  - `regional_t1_pre/small`: 0.0652
  - `regional_t1_pre/tiny`: 0.0604
  - `regional_t1_pre/tumor_bg_ratio`: 3.9376
  - `regional_t1_pre/tumor_loss`: 0.0628

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.2349, max 10.7628 @ ep 15
  - `training/grad_norm_max`: last 0.9731, max 19.7913 @ ep 16

### exp8

**exp8 (3D VAE sweep, 256×160)** — 12 variants (exp8_1…exp8_14) exploring
KL weight, adversarial loss, architecture depth for 3D multi-modality VAE
at 256×256×160. The `_256x160` suffix indicates input resolution.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp8_2_256x160_20260107-114547` — 37.6814
  2. 🥈 `exp8_1_256x160_20260107-031153` — 37.6325
  3.  `exp8_3_256x160_20260107-114547` — 37.4273
  4.  `exp8_256x160_20251231-123029` — 37.3608
  5.  `exp8_10_256x160_20260112-121110` — 34.8171
  6.  `exp8_11_256x160_20260112-121110` — 34.5592
  7.  `exp8_12_256x160_20260112-121110` — 34.5389
  8.  `exp8_14_256x160_20260112-121110` — 34.4359
  9.  `exp8_7_256x160_20260108-014607` — 34.4119
  10.  `exp8_9_256x160_20260108-043415` — 34.3889
  11.  `exp8_5_256x160_20260108-014607` — 34.1703
  12.  `exp8_8_256x160_20260108-094220` — 34.1431
  13.  `exp8_6_256x160_20260108-043415` — 34.1375
  14.  `exp8_13_256x160_20260112-121110` — 33.5662

#### `exp8_256x160_20251231-123029`
*started 2025-12-31 12:30 • 125 epochs • 75h21m • 158.6 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0386 → 0.0029 (min 0.0029 @ ep 95)
  - `Loss/L1_val`: 0.0375 → 0.0030 (min 0.0030 @ ep 85)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0485 (min 0.0458, max 0.8957)
  - `Validation/LPIPS_bravo`: last 0.0768 (min 0.0750, max 0.9131)
  - `Validation/LPIPS_flair`: last 0.0647 (min 0.0608, max 0.8906)
  - `Validation/LPIPS_t1_gd`: last 0.0300 (min 0.0279, max 0.8871)
  - `Validation/LPIPS_t1_pre`: last 0.0226 (min 0.0197, max 0.8920)
  - `Validation/MS-SSIM`: last 0.9966 (min 0.3720, max 0.9967)
  - `Validation/MS-SSIM_bravo`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_flair`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_t1_gd`: last 0 (min 0, max 0)
  - `Validation/MS-SSIM_t1_pre`: last 0 (min 0, max 0)
  - `Validation/PSNR`: last 39.6711 (min 20.3440, max 39.7352)
  - `Validation/PSNR_bravo`: last 37.2963 (min 21.4295, max 37.3608)
  - `Validation/PSNR_flair`: last 38.1238 (min 21.1469, max 38.3199)
  - `Validation/PSNR_t1_gd`: last 41.6725 (min 20.5517, max 41.7371)
  - `Validation/PSNR_t1_pre`: last 41.5919 (min 18.2479, max 41.6315)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0036
  - `regional_bravo/large`: 0.0335
  - `regional_bravo/medium`: 0.0322
  - `regional_bravo/small`: 0.0260
  - `regional_bravo/tiny`: 0.0254
  - `regional_bravo/tumor_bg_ratio`: 8.2987
  - `regional_bravo/tumor_loss`: 0.0301
  - `regional_flair/background_loss`: 0.0039
  - `regional_flair/large`: 0.0365
  - `regional_flair/medium`: 0.0333
  - `regional_flair/small`: 0.0331
  - `regional_flair/tiny`: 0.0304
  - `regional_flair/tumor_bg_ratio`: 8.7324
  - `regional_flair/tumor_loss`: 0.0337
  - `regional_t1_gd/background_loss`: 0.0022
  - `regional_t1_gd/large`: 0.0275
  - `regional_t1_gd/medium`: 0.0255
  - `regional_t1_gd/small`: 0.0225
  - `regional_t1_gd/tiny`: 0.0176
  - `regional_t1_gd/tumor_bg_ratio`: 10.9530
  - `regional_t1_gd/tumor_loss`: 0.0240
  - `regional_t1_pre/background_loss`: 0.0024
  - `regional_t1_pre/large`: 0.0181
  - `regional_t1_pre/medium`: 0.0144
  - `regional_t1_pre/small`: 0.0166
  - `regional_t1_pre/tiny`: 0.0141
  - `regional_t1_pre/tumor_bg_ratio`: 6.6842
  - `regional_t1_pre/tumor_loss`: 0.0160

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.8369, max 16.6368 @ ep 112
  - `training/grad_norm_d_max`: last 19.3953, max 926.2098 @ ep 15
  - `training/grad_norm_g_avg`: last 0.3637, max 0.5286 @ ep 69
  - `training/grad_norm_g_max`: last 1.0004, max 17.9611 @ ep 50

#### `exp8_1_256x160_20260107-031153`
*started 2026-01-07 03:11 • 125 epochs • 34h41m • 185.1 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0386 → 0.0029 (min 0.0029 @ ep 110)
  - `Loss/L1_val`: 0.0375 → 0.0030 (min 0.0029 @ ep 80)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0405 (min 0.0336, max 0.8853)
  - `Validation/LPIPS_bravo`: last 0.0640 (min 0.0537, max 0.9025)
  - `Validation/LPIPS_flair`: last 0.0534 (min 0.0433, max 0.8799)
  - `Validation/LPIPS_t1_gd`: last 0.0246 (min 0.0208, max 0.8768)
  - `Validation/LPIPS_t1_pre`: last 0.0200 (min 0.0160, max 0.8820)
  - `Validation/MS-SSIM`: last 0.9947 (min 0.4639, max 0.9948)
  - `Validation/MS-SSIM-3D`: last 0.9967 (min 0.3713, max 0.9968)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9954 (min 0.3970, max 0.9955)
  - `Validation/MS-SSIM-3D_flair`: last 0.9952 (min 0.3741, max 0.9953)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9979 (min 0.3743, max 0.9979)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9984 (min 0.3397, max 0.9984)
  - `Validation/MS-SSIM_bravo`: last 0.9927 (min 0.4768, max 0.9929)
  - `Validation/MS-SSIM_flair`: last 0.9924 (min 0.4798, max 0.9925)
  - `Validation/MS-SSIM_t1_gd`: last 0.9965 (min 0.4683, max 0.9965)
  - `Validation/MS-SSIM_t1_pre`: last 0.9973 (min 0.4309, max 0.9973)
  - `Validation/PSNR`: last 39.8156 (min 20.3408, max 39.9843)
  - `Validation/PSNR_bravo`: last 37.4919 (min 21.4260, max 37.6325)
  - `Validation/PSNR_flair`: last 38.4067 (min 21.1432, max 38.5766)
  - `Validation/PSNR_t1_gd`: last 41.7299 (min 20.5484, max 41.9340)
  - `Validation/PSNR_t1_pre`: last 41.6340 (min 18.2455, max 41.7940)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0282
  - `regional/medium`: 0.0258
  - `regional/small`: 0.0242
  - `regional/tiny`: 0.0216
  - `regional/tumor_bg_ratio`: 8.4450
  - `regional/tumor_loss`: 0.0254
  - `regional_bravo/background_loss`: 0.0036
  - `regional_bravo/large`: 0.0326
  - `regional_bravo/medium`: 0.0312
  - `regional_bravo/small`: 0.0256
  - `regional_bravo/tiny`: 0.0249
  - `regional_bravo/tumor_bg_ratio`: 8.1625
  - `regional_bravo/tumor_loss`: 0.0294
  - `regional_flair/background_loss`: 0.0038
  - `regional_flair/large`: 0.0358
  - `regional_flair/medium`: 0.0323
  - `regional_flair/small`: 0.0323
  - `regional_flair/tiny`: 0.0299
  - `regional_flair/tumor_bg_ratio`: 8.6806
  - `regional_flair/tumor_loss`: 0.0330
  - `regional_t1_gd/background_loss`: 0.0022
  - `regional_t1_gd/large`: 0.0268
  - `regional_t1_gd/medium`: 0.0254
  - `regional_t1_gd/small`: 0.0223
  - `regional_t1_gd/tiny`: 0.0177
  - `regional_t1_gd/tumor_bg_ratio`: 10.6567
  - `regional_t1_gd/tumor_loss`: 0.0237
  - `regional_t1_pre/background_loss`: 0.0024
  - `regional_t1_pre/large`: 0.0176
  - `regional_t1_pre/medium`: 0.0143
  - `regional_t1_pre/small`: 0.0167
  - `regional_t1_pre/tiny`: 0.0138
  - `regional_t1_pre/tumor_bg_ratio`: 6.4701
  - `regional_t1_pre/tumor_loss`: 0.0157

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.0961, max 17.9000 @ ep 119
  - `training/grad_norm_d_max`: last 12.2025, max 357.3566 @ ep 23
  - `training/grad_norm_g_avg`: last 0.0586, max 0.5419 @ ep 4
  - `training/grad_norm_g_max`: last 0.1522, max 2.6447 @ ep 80

#### `exp8_2_256x160_20260107-114547`
*started 2026-01-07 11:45 • 125 epochs • 33h14m • 290.9 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0423 → 0.0029 (min 0.0028 @ ep 92)
  - `Loss/L1_val`: 0.0385 → 0.0030 (min 0.0029 @ ep 90)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0334 (min 0.0311, max 0.9322)
  - `Validation/LPIPS_bravo`: last 0.0512 (min 0.0493, max 0.9509)
  - `Validation/LPIPS_flair`: last 0.0426 (min 0.0399, max 0.9294)
  - `Validation/LPIPS_t1_gd`: last 0.0221 (min 0.0199, max 0.9222)
  - `Validation/LPIPS_t1_pre`: last 0.0175 (min 0.0153, max 0.9264)
  - `Validation/MS-SSIM`: last 0.9945 (min 0.4599, max 0.9949)
  - `Validation/MS-SSIM-3D`: last 0.9967 (min 0.3715, max 0.9968)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9953 (min 0.3973, max 0.9955)
  - `Validation/MS-SSIM-3D_flair`: last 0.9952 (min 0.3743, max 0.9954)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9979 (min 0.3746, max 0.9979)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9984 (min 0.3398, max 0.9984)
  - `Validation/MS-SSIM_bravo`: last 0.9924 (min 0.4728, max 0.9930)
  - `Validation/MS-SSIM_flair`: last 0.9920 (min 0.4758, max 0.9927)
  - `Validation/MS-SSIM_t1_gd`: last 0.9965 (min 0.4643, max 0.9966)
  - `Validation/MS-SSIM_t1_pre`: last 0.9973 (min 0.4268, max 0.9973)
  - `Validation/PSNR`: last 39.6915 (min 20.3130, max 39.9933)
  - `Validation/PSNR_bravo`: last 37.3606 (min 21.3935, max 37.6814)
  - `Validation/PSNR_flair`: last 38.1264 (min 21.1068, max 38.6614)
  - `Validation/PSNR_t1_gd`: last 41.7016 (min 20.5216, max 41.9164)
  - `Validation/PSNR_t1_pre`: last 41.5773 (min 18.2301, max 41.7338)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0281
  - `regional/medium`: 0.0256
  - `regional/small`: 0.0244
  - `regional/tiny`: 0.0215
  - `regional/tumor_bg_ratio`: 8.3456
  - `regional/tumor_loss`: 0.0254
  - `regional_bravo/background_loss`: 0.0037
  - `regional_bravo/large`: 0.0329
  - `regional_bravo/medium`: 0.0315
  - `regional_bravo/small`: 0.0258
  - `regional_bravo/tiny`: 0.0249
  - `regional_bravo/tumor_bg_ratio`: 8.0880
  - `regional_bravo/tumor_loss`: 0.0296
  - `regional_flair/background_loss`: 0.0039
  - `regional_flair/large`: 0.0359
  - `regional_flair/medium`: 0.0322
  - `regional_flair/small`: 0.0329
  - `regional_flair/tiny`: 0.0302
  - `regional_flair/tumor_bg_ratio`: 8.5415
  - `regional_flair/tumor_loss`: 0.0332
  - `regional_t1_gd/background_loss`: 0.0022
  - `regional_t1_gd/large`: 0.0263
  - `regional_t1_gd/medium`: 0.0245
  - `regional_t1_gd/small`: 0.0223
  - `regional_t1_gd/tiny`: 0.0172
  - `regional_t1_gd/tumor_bg_ratio`: 10.5420
  - `regional_t1_gd/tumor_loss`: 0.0232
  - `regional_t1_pre/background_loss`: 0.0024
  - `regional_t1_pre/large`: 0.0171
  - `regional_t1_pre/medium`: 0.0140
  - `regional_t1_pre/small`: 0.0166
  - `regional_t1_pre/tiny`: 0.0138
  - `regional_t1_pre/tumor_bg_ratio`: 6.4153
  - `regional_t1_pre/tumor_loss`: 0.0155

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 4.9645, max 10.3330 @ ep 89
  - `training/grad_norm_d_max`: last 14.9374, max 101.3770 @ ep 5
  - `training/grad_norm_g_avg`: last 0.1071, max 0.5398 @ ep 5
  - `training/grad_norm_g_max`: last 0.9200, max 2.3647 @ ep 61

#### `exp8_3_256x160_20260107-114547`
*started 2026-01-07 11:45 • 125 epochs • 32h40m • 502.5 TFLOPs • peak VRAM 6.6 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0524 → 0.0032 (min 0.0031 @ ep 79)
  - `Loss/L1_val`: 0.0409 → 0.0034 (min 0.0031 @ ep 79)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0431 (min 0.0336, max 1.0375)
  - `Validation/LPIPS_bravo`: last 0.0655 (min 0.0531, max 1.0586)
  - `Validation/LPIPS_flair`: last 0.0528 (min 0.0431, max 1.0376)
  - `Validation/LPIPS_t1_gd`: last 0.0299 (min 0.0208, max 1.0261)
  - `Validation/LPIPS_t1_pre`: last 0.0241 (min 0.0165, max 1.0277)
  - `Validation/MS-SSIM`: last 0.9943 (min 0.4577, max 0.9946)
  - `Validation/MS-SSIM-3D`: last 0.9963 (min 0.3722, max 0.9965)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9948 (min 0.3981, max 0.9952)
  - `Validation/MS-SSIM-3D_flair`: last 0.9947 (min 0.3751, max 0.9950)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9976 (min 0.3753, max 0.9977)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9982 (min 0.3404, max 0.9983)
  - `Validation/MS-SSIM_bravo`: last 0.9921 (min 0.4707, max 0.9926)
  - `Validation/MS-SSIM_flair`: last 0.9919 (min 0.4734, max 0.9923)
  - `Validation/MS-SSIM_t1_gd`: last 0.9962 (min 0.4622, max 0.9963)
  - `Validation/MS-SSIM_t1_pre`: last 0.9970 (min 0.4244, max 0.9971)
  - `Validation/PSNR`: last 39.3699 (min 20.2322, max 39.6911)
  - `Validation/PSNR_bravo`: last 37.0543 (min 21.2991, max 37.4273)
  - `Validation/PSNR_flair`: last 38.1108 (min 21.0016, max 38.3678)
  - `Validation/PSNR_t1_gd`: last 41.2250 (min 20.4435, max 41.5863)
  - `Validation/PSNR_t1_pre`: last 41.0898 (min 18.1849, max 41.3831)

**Regional loss (final):**
  - `regional/background_loss`: 0.0034
  - `regional/large`: 0.0280
  - `regional/medium`: 0.0257
  - `regional/small`: 0.0243
  - `regional/tiny`: 0.0216
  - `regional/tumor_bg_ratio`: 7.5170
  - `regional/tumor_loss`: 0.0254
  - `regional_bravo/background_loss`: 0.0040
  - `regional_bravo/large`: 0.0329
  - `regional_bravo/medium`: 0.0316
  - `regional_bravo/small`: 0.0258
  - `regional_bravo/tiny`: 0.0252
  - `regional_bravo/tumor_bg_ratio`: 7.3959
  - `regional_bravo/tumor_loss`: 0.0296
  - `regional_flair/background_loss`: 0.0042
  - `regional_flair/large`: 0.0355
  - `regional_flair/medium`: 0.0321
  - `regional_flair/small`: 0.0322
  - `regional_flair/tiny`: 0.0294
  - `regional_flair/tumor_bg_ratio`: 7.8502
  - `regional_flair/tumor_loss`: 0.0327
  - `regional_t1_gd/background_loss`: 0.0026
  - `regional_t1_gd/large`: 0.0263
  - `regional_t1_gd/medium`: 0.0251
  - `regional_t1_gd/small`: 0.0226
  - `regional_t1_gd/tiny`: 0.0176
  - `regional_t1_gd/tumor_bg_ratio`: 9.2242
  - `regional_t1_gd/tumor_loss`: 0.0235
  - `regional_t1_pre/background_loss`: 0.0028
  - `regional_t1_pre/large`: 0.0174
  - `regional_t1_pre/medium`: 0.0141
  - `regional_t1_pre/small`: 0.0165
  - `regional_t1_pre/tiny`: 0.0141
  - `regional_t1_pre/tumor_bg_ratio`: 5.6240
  - `regional_t1_pre/tumor_loss`: 0.0156

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 6.3155, max 11.0112 @ ep 5
  - `training/grad_norm_d_max`: last 17.9076, max 1315.1924 @ ep 27
  - `training/grad_norm_g_avg`: last 0.2127, max 1.2904 @ ep 0
  - `training/grad_norm_g_max`: last 1.4645, max 10.3811 @ ep 22

#### `exp8_5_256x160_20260108-014607`
*started 2026-01-08 01:46 • 125 epochs • 40h03m • 84.2 TFLOPs • peak VRAM 6.9 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0450 → 0.0042 (min 0.0042 @ ep 92)
  - `Loss/L1_val`: 0.0395 → 0.0049 (min 0.0047 @ ep 89)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0896 (min 0.0855, max 0.9722)
  - `Validation/LPIPS_bravo`: last 0.1264 (min 0.1204, max 0.9925)
  - `Validation/LPIPS_flair`: last 0.1112 (min 0.1057, max 0.9715)
  - `Validation/LPIPS_t1_gd`: last 0.0617 (min 0.0589, max 0.9613)
  - `Validation/LPIPS_t1_pre`: last 0.0589 (min 0.0551, max 0.9636)
  - `Validation/MS-SSIM`: last 0.9797 (min 0.4685, max 0.9813)
  - `Validation/MS-SSIM-3D`: last 0.9848 (min 0.3741, max 0.9858)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9820 (min 0.4002, max 0.9832)
  - `Validation/MS-SSIM-3D_flair`: last 0.9784 (min 0.3771, max 0.9801)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9888 (min 0.3771, max 0.9897)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9899 (min 0.3418, max 0.9905)
  - `Validation/MS-SSIM_bravo`: last 0.9765 (min 0.4818, max 0.9784)
  - `Validation/MS-SSIM_flair`: last 0.9711 (min 0.4842, max 0.9735)
  - `Validation/MS-SSIM_t1_gd`: last 0.9850 (min 0.4732, max 0.9863)
  - `Validation/MS-SSIM_t1_pre`: last 0.9861 (min 0.4348, max 0.9871)
  - `Validation/PSNR`: last 35.1187 (min 20.2910, max 35.5672)
  - `Validation/PSNR_bravo`: last 33.7442 (min 21.3671, max 34.1703)
  - `Validation/PSNR_flair`: last 33.8571 (min 21.0762, max 34.5202)
  - `Validation/PSNR_t1_gd`: last 36.8988 (min 20.5009, max 37.3778)
  - `Validation/PSNR_t1_pre`: last 35.9748 (min 18.2198, max 36.4003)

**Regional loss (final):**
  - `regional/background_loss`: 0.0049
  - `regional/large`: 0.0544
  - `regional/medium`: 0.0456
  - `regional/small`: 0.0454
  - `regional/tiny`: 0.0382
  - `regional/tumor_bg_ratio`: 9.5396
  - `regional/tumor_loss`: 0.0470
  - `regional_bravo/background_loss`: 0.0053
  - `regional_bravo/large`: 0.0592
  - `regional_bravo/medium`: 0.0538
  - `regional_bravo/small`: 0.0454
  - `regional_bravo/tiny`: 0.0411
  - `regional_bravo/tumor_bg_ratio`: 9.7373
  - `regional_bravo/tumor_loss`: 0.0515
  - `regional_flair/background_loss`: 0.0062
  - `regional_flair/large`: 0.0667
  - `regional_flair/medium`: 0.0574
  - `regional_flair/small`: 0.0591
  - `regional_flair/tiny`: 0.0524
  - `regional_flair/tumor_bg_ratio`: 9.6916
  - `regional_flair/tumor_loss`: 0.0598
  - `regional_t1_gd/background_loss`: 0.0038
  - `regional_t1_gd/large`: 0.0549
  - `regional_t1_gd/medium`: 0.0456
  - `regional_t1_gd/small`: 0.0450
  - `regional_t1_gd/tiny`: 0.0333
  - `regional_t1_gd/tumor_bg_ratio`: 12.2447
  - `regional_t1_gd/tumor_loss`: 0.0461
  - `regional_t1_pre/background_loss`: 0.0045
  - `regional_t1_pre/large`: 0.0368
  - `regional_t1_pre/medium`: 0.0255
  - `regional_t1_pre/small`: 0.0321
  - `regional_t1_pre/tiny`: 0.0260
  - `regional_t1_pre/tumor_bg_ratio`: 6.8237
  - `regional_t1_pre/tumor_loss`: 0.0306

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.1971, max 10.5945 @ ep 4
  - `training/grad_norm_d_max`: last 16.9621, max 325.2851 @ ep 2
  - `training/grad_norm_g_avg`: last 0.0316, max 0.6043 @ ep 2
  - `training/grad_norm_g_max`: last 0.0856, max 2.4294 @ ep 3

#### `exp8_7_256x160_20260108-014607`
*started 2026-01-08 01:46 • 125 epochs • 40h15m • 123.9 TFLOPs • peak VRAM 6.9 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0394 → 0.0040 (min 0.0040 @ ep 95)
  - `Loss/L1_val`: 0.0378 → 0.0047 (min 0.0045 @ ep 67)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0917 (min 0.0820, max 0.8925)
  - `Validation/LPIPS_bravo`: last 0.1295 (min 0.1135, max 0.9096)
  - `Validation/LPIPS_flair`: last 0.1111 (min 0.1023, max 0.8871)
  - `Validation/LPIPS_t1_gd`: last 0.0644 (min 0.0557, max 0.8841)
  - `Validation/LPIPS_t1_pre`: last 0.0617 (min 0.0519, max 0.8892)
  - `Validation/MS-SSIM`: last 0.9811 (min 0.4782, max 0.9821)
  - `Validation/MS-SSIM-3D`: last 0.9860 (min 0.3753, max 0.9866)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9831 (min 0.4016, max 0.9838)
  - `Validation/MS-SSIM-3D_flair`: last 0.9800 (min 0.3784, max 0.9811)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9899 (min 0.3783, max 0.9902)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9909 (min 0.3427, max 0.9912)
  - `Validation/MS-SSIM_bravo`: last 0.9778 (min 0.4917, max 0.9790)
  - `Validation/MS-SSIM_flair`: last 0.9729 (min 0.4938, max 0.9748)
  - `Validation/MS-SSIM_t1_gd`: last 0.9863 (min 0.4830, max 0.9869)
  - `Validation/MS-SSIM_t1_pre`: last 0.9873 (min 0.4442, max 0.9878)
  - `Validation/PSNR`: last 35.4874 (min 20.3577, max 35.8673)
  - `Validation/PSNR_bravo`: last 34.0103 (min 21.4446, max 34.4119)
  - `Validation/PSNR_flair`: last 34.1792 (min 21.1620, max 34.6723)
  - `Validation/PSNR_t1_gd`: last 37.3170 (min 20.5657, max 37.6625)
  - `Validation/PSNR_t1_pre`: last 36.4432 (min 18.2586, max 36.7225)

**Regional loss (final):**
  - `regional/background_loss`: 0.0047
  - `regional/large`: 0.0535
  - `regional/medium`: 0.0444
  - `regional/small`: 0.0444
  - `regional/tiny`: 0.0373
  - `regional/tumor_bg_ratio`: 9.8863
  - `regional/tumor_loss`: 0.0460
  - `regional_bravo/background_loss`: 0.0051
  - `regional_bravo/large`: 0.0594
  - `regional_bravo/medium`: 0.0525
  - `regional_bravo/small`: 0.0446
  - `regional_bravo/tiny`: 0.0406
  - `regional_bravo/tumor_bg_ratio`: 10.0768
  - `regional_bravo/tumor_loss`: 0.0509
  - `regional_flair/background_loss`: 0.0059
  - `regional_flair/large`: 0.0649
  - `regional_flair/medium`: 0.0558
  - `regional_flair/small`: 0.0587
  - `regional_flair/tiny`: 0.0517
  - `regional_flair/tumor_bg_ratio`: 9.9432
  - `regional_flair/tumor_loss`: 0.0586
  - `regional_t1_gd/background_loss`: 0.0035
  - `regional_t1_gd/large`: 0.0552
  - `regional_t1_gd/medium`: 0.0445
  - `regional_t1_gd/small`: 0.0439
  - `regional_t1_gd/tiny`: 0.0322
  - `regional_t1_gd/tumor_bg_ratio`: 12.9829
  - `regional_t1_gd/tumor_loss`: 0.0455
  - `regional_t1_pre/background_loss`: 0.0042
  - `regional_t1_pre/large`: 0.0346
  - `regional_t1_pre/medium`: 0.0249
  - `regional_t1_pre/small`: 0.0305
  - `regional_t1_pre/tiny`: 0.0247
  - `regional_t1_pre/tumor_bg_ratio`: 6.9777
  - `regional_t1_pre/tumor_loss`: 0.0291

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.7673, max 10.1782 @ ep 6
  - `training/grad_norm_d_max`: last 19.9297, max 120.6013 @ ep 0
  - `training/grad_norm_g_avg`: last 0.1762, max 0.6143 @ ep 5
  - `training/grad_norm_g_max`: last 0.6316, max 3.5733 @ ep 10

#### `exp8_6_256x160_20260108-043415`
*started 2026-01-08 04:34 • 125 epochs • 40h57m • 97.5 TFLOPs • peak VRAM 6.9 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0389 → 0.0040 (min 0.0040 @ ep 94)
  - `Loss/L1_val`: 0.0379 → 0.0049 (min 0.0046 @ ep 96)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0965 (min 0.0881, max 0.8950)
  - `Validation/LPIPS_bravo`: last 0.1375 (min 0.1241, max 0.9140)
  - `Validation/LPIPS_flair`: last 0.1145 (min 0.1070, max 0.8911)
  - `Validation/LPIPS_t1_gd`: last 0.0678 (min 0.0611, max 0.8857)
  - `Validation/LPIPS_t1_pre`: last 0.0664 (min 0.0576, max 0.8893)
  - `Validation/MS-SSIM`: last 0.9787 (min 0.4818, max 0.9802)
  - `Validation/MS-SSIM-3D`: last 0.9840 (min 0.3765, max 0.9848)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9812 (min 0.4030, max 0.9819)
  - `Validation/MS-SSIM-3D_flair`: last 0.9775 (min 0.3797, max 0.9790)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9881 (min 0.3795, max 0.9888)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9890 (min 0.3437, max 0.9895)
  - `Validation/MS-SSIM_bravo`: last 0.9756 (min 0.4955, max 0.9770)
  - `Validation/MS-SSIM_flair`: last 0.9700 (min 0.4974, max 0.9726)
  - `Validation/MS-SSIM_t1_gd`: last 0.9842 (min 0.4867, max 0.9854)
  - `Validation/MS-SSIM_t1_pre`: last 0.9850 (min 0.4475, max 0.9858)
  - `Validation/PSNR`: last 35.0847 (min 20.3627, max 35.5049)
  - `Validation/PSNR_bravo`: last 33.7511 (min 21.4500, max 34.1375)
  - `Validation/PSNR_flair`: last 33.7825 (min 21.1674, max 34.4152)
  - `Validation/PSNR_t1_gd`: last 36.9180 (min 20.5708, max 37.2928)
  - `Validation/PSNR_t1_pre`: last 35.8870 (min 18.2626, max 36.2402)

**Regional loss (final):**
  - `regional/background_loss`: 0.0049
  - `regional/large`: 0.0559
  - `regional/medium`: 0.0463
  - `regional/small`: 0.0463
  - `regional/tiny`: 0.0390
  - `regional/tumor_bg_ratio`: 9.8761
  - `regional/tumor_loss`: 0.0480
  - `regional_bravo/background_loss`: 0.0052
  - `regional_bravo/large`: 0.0612
  - `regional_bravo/medium`: 0.0543
  - `regional_bravo/small`: 0.0461
  - `regional_bravo/tiny`: 0.0426
  - `regional_bravo/tumor_bg_ratio`: 10.1405
  - `regional_bravo/tumor_loss`: 0.0527
  - `regional_flair/background_loss`: 0.0061
  - `regional_flair/large`: 0.0696
  - `regional_flair/medium`: 0.0570
  - `regional_flair/small`: 0.0601
  - `regional_flair/tiny`: 0.0534
  - `regional_flair/tumor_bg_ratio`: 9.9428
  - `regional_flair/tumor_loss`: 0.0610
  - `regional_t1_gd/background_loss`: 0.0037
  - `regional_t1_gd/large`: 0.0563
  - `regional_t1_gd/medium`: 0.0472
  - `regional_t1_gd/small`: 0.0460
  - `regional_t1_gd/tiny`: 0.0338
  - `regional_t1_gd/tumor_bg_ratio`: 12.8854
  - `regional_t1_gd/tumor_loss`: 0.0473
  - `regional_t1_pre/background_loss`: 0.0044
  - `regional_t1_pre/large`: 0.0364
  - `regional_t1_pre/medium`: 0.0266
  - `regional_t1_pre/small`: 0.0330
  - `regional_t1_pre/tiny`: 0.0263
  - `regional_t1_pre/tumor_bg_ratio`: 6.9849
  - `regional_t1_pre/tumor_loss`: 0.0310

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.9663, max 10.9626 @ ep 5
  - `training/grad_norm_d_max`: last 21.6842, max 282.4209 @ ep 6
  - `training/grad_norm_g_avg`: last 0.1929, max 0.5955 @ ep 5
  - `training/grad_norm_g_max`: last 0.8759, max 1.9316 @ ep 6

#### `exp8_9_256x160_20260108-043415`
*started 2026-01-08 04:34 • 125 epochs • 35h04m • 282.6 TFLOPs • peak VRAM 6.9 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0398 → 0.0040 (min 0.0040 @ ep 93)
  - `Loss/L1_val`: 0.0378 → 0.0048 (min 0.0045 @ ep 83)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0869 (min 0.0816, max 0.8924)
  - `Validation/LPIPS_bravo`: last 0.1239 (min 0.1146, max 0.9109)
  - `Validation/LPIPS_flair`: last 0.1084 (min 0.1015, max 0.8877)
  - `Validation/LPIPS_t1_gd`: last 0.0594 (min 0.0567, max 0.8835)
  - `Validation/LPIPS_t1_pre`: last 0.0559 (min 0.0535, max 0.8876)
  - `Validation/MS-SSIM`: last 0.9808 (min 0.4786, max 0.9822)
  - `Validation/MS-SSIM-3D`: last 0.9858 (min 0.3755, max 0.9865)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9831 (min 0.4019, max 0.9837)
  - `Validation/MS-SSIM-3D_flair`: last 0.9797 (min 0.3787, max 0.9809)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9897 (min 0.3786, max 0.9902)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9906 (min 0.3429, max 0.9911)
  - `Validation/MS-SSIM_bravo`: last 0.9777 (min 0.4921, max 0.9793)
  - `Validation/MS-SSIM_flair`: last 0.9724 (min 0.4942, max 0.9748)
  - `Validation/MS-SSIM_t1_gd`: last 0.9860 (min 0.4834, max 0.9870)
  - `Validation/MS-SSIM_t1_pre`: last 0.9870 (min 0.4445, max 0.9878)
  - `Validation/PSNR`: last 35.3348 (min 20.3568, max 35.7779)
  - `Validation/PSNR_bravo`: last 33.9625 (min 21.4434, max 34.3889)
  - `Validation/PSNR_flair`: last 34.0042 (min 21.1606, max 34.6441)
  - `Validation/PSNR_t1_gd`: last 37.1348 (min 20.5649, max 37.5707)
  - `Validation/PSNR_t1_pre`: last 36.2376 (min 18.2583, max 36.6974)

**Regional loss (final):**
  - `regional/background_loss`: 0.0047
  - `regional/large`: 0.0532
  - `regional/medium`: 0.0449
  - `regional/small`: 0.0448
  - `regional/tiny`: 0.0376
  - `regional/tumor_bg_ratio`: 9.7312
  - `regional/tumor_loss`: 0.0462
  - `regional_bravo/background_loss`: 0.0051
  - `regional_bravo/large`: 0.0591
  - `regional_bravo/medium`: 0.0533
  - `regional_bravo/small`: 0.0444
  - `regional_bravo/tiny`: 0.0413
  - `regional_bravo/tumor_bg_ratio`: 10.0448
  - `regional_bravo/tumor_loss`: 0.0511
  - `regional_flair/background_loss`: 0.0060
  - `regional_flair/large`: 0.0642
  - `regional_flair/medium`: 0.0557
  - `regional_flair/small`: 0.0592
  - `regional_flair/tiny`: 0.0515
  - `regional_flair/tumor_bg_ratio`: 9.7134
  - `regional_flair/tumor_loss`: 0.0583
  - `regional_t1_gd/background_loss`: 0.0036
  - `regional_t1_gd/large`: 0.0535
  - `regional_t1_gd/medium`: 0.0456
  - `regional_t1_gd/small`: 0.0427
  - `regional_t1_gd/tiny`: 0.0326
  - `regional_t1_gd/tumor_bg_ratio`: 12.5419
  - `regional_t1_gd/tumor_loss`: 0.0451
  - `regional_t1_pre/background_loss`: 0.0043
  - `regional_t1_pre/large`: 0.0359
  - `regional_t1_pre/medium`: 0.0251
  - `regional_t1_pre/small`: 0.0331
  - `regional_t1_pre/tiny`: 0.0249
  - `regional_t1_pre/tumor_bg_ratio`: 7.0282
  - `regional_t1_pre/tumor_loss`: 0.0301

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.3954, max 10.5211 @ ep 6
  - `training/grad_norm_d_max`: last 17.6073, max 278.8748 @ ep 17
  - `training/grad_norm_g_avg`: last 0.1325, max 0.5440 @ ep 6
  - `training/grad_norm_g_max`: last 0.2798, max 2.8726 @ ep 18

#### `exp8_8_256x160_20260108-094220`
*started 2026-01-08 09:42 • 125 epochs • 34h54m • 176.8 TFLOPs • peak VRAM 6.9 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0570 → 0.0043 (min 0.0042 @ ep 98)
  - `Loss/L1_val`: 0.0404 → 0.0049 (min 0.0047 @ ep 98)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0874 (min 0.0808, max 0.9052)
  - `Validation/LPIPS_bravo`: last 0.1225 (min 0.1140, max 0.9214)
  - `Validation/LPIPS_flair`: last 0.1073 (min 0.0999, max 0.8985)
  - `Validation/LPIPS_t1_gd`: last 0.0621 (min 0.0566, max 0.8975)
  - `Validation/LPIPS_t1_pre`: last 0.0576 (min 0.0514, max 0.9036)
  - `Validation/MS-SSIM`: last 0.9802 (min 0.4787, max 0.9814)
  - `Validation/MS-SSIM-3D`: last 0.9853 (min 0.3755, max 0.9860)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9823 (min 0.4018, max 0.9832)
  - `Validation/MS-SSIM-3D_flair`: last 0.9792 (min 0.3787, max 0.9803)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9892 (min 0.3785, max 0.9898)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9903 (min 0.3429, max 0.9908)
  - `Validation/MS-SSIM_bravo`: last 0.9769 (min 0.4922, max 0.9782)
  - `Validation/MS-SSIM_flair`: last 0.9721 (min 0.4943, max 0.9741)
  - `Validation/MS-SSIM_t1_gd`: last 0.9854 (min 0.4835, max 0.9863)
  - `Validation/MS-SSIM_t1_pre`: last 0.9866 (min 0.4446, max 0.9874)
  - `Validation/PSNR`: last 35.1373 (min 20.3591, max 35.5965)
  - `Validation/PSNR_bravo`: last 33.6712 (min 21.4461, max 34.1431)
  - `Validation/PSNR_flair`: last 33.9923 (min 21.1635, max 34.5812)
  - `Validation/PSNR_t1_gd`: last 36.8601 (min 20.5672, max 37.3332)
  - `Validation/PSNR_t1_pre`: last 36.0256 (min 18.2598, max 36.4311)

**Regional loss (final):**
  - `regional/background_loss`: 0.0049
  - `regional/large`: 0.0548
  - `regional/medium`: 0.0459
  - `regional/small`: 0.0464
  - `regional/tiny`: 0.0383
  - `regional/tumor_bg_ratio`: 9.6846
  - `regional/tumor_loss`: 0.0474
  - `regional_bravo/background_loss`: 0.0053
  - `regional_bravo/large`: 0.0618
  - `regional_bravo/medium`: 0.0541
  - `regional_bravo/small`: 0.0461
  - `regional_bravo/tiny`: 0.0413
  - `regional_bravo/tumor_bg_ratio`: 9.9544
  - `regional_bravo/tumor_loss`: 0.0526
  - `regional_flair/background_loss`: 0.0061
  - `regional_flair/large`: 0.0666
  - `regional_flair/medium`: 0.0578
  - `regional_flair/small`: 0.0612
  - `regional_flair/tiny`: 0.0530
  - `regional_flair/tumor_bg_ratio`: 9.8982
  - `regional_flair/tumor_loss`: 0.0604
  - `regional_t1_gd/background_loss`: 0.0038
  - `regional_t1_gd/large`: 0.0542
  - `regional_t1_gd/medium`: 0.0462
  - `regional_t1_gd/small`: 0.0455
  - `regional_t1_gd/tiny`: 0.0333
  - `regional_t1_gd/tumor_bg_ratio`: 12.2937
  - `regional_t1_gd/tumor_loss`: 0.0462
  - `regional_t1_pre/background_loss`: 0.0045
  - `regional_t1_pre/large`: 0.0365
  - `regional_t1_pre/medium`: 0.0257
  - `regional_t1_pre/small`: 0.0327
  - `regional_t1_pre/tiny`: 0.0256
  - `regional_t1_pre/tumor_bg_ratio`: 6.8698
  - `regional_t1_pre/tumor_loss`: 0.0306

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 4.4208, max 10.4797 @ ep 4
  - `training/grad_norm_d_max`: last 12.8727, max 131.5347 @ ep 45
  - `training/grad_norm_g_avg`: last 0.0450, max 0.8784 @ ep 5
  - `training/grad_norm_g_max`: last 0.1398, max 2.5283 @ ep 1

#### `exp8_10_256x160_20260112-121110`
*started 2026-01-12 12:11 • 125 epochs • 64h14m • 150.3 TFLOPs • peak VRAM 24.8 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0426 → 0.0041 (min 0.0041 @ ep 90)
  - `Loss/L1_val`: 0.0389 → 0.0043 (min 0.0042 @ ep 95)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0764 (min 0.0724, max 0.9009)
  - `Validation/LPIPS_bravo`: last 0.1094 (min 0.1037, max 0.9482)
  - `Validation/LPIPS_flair`: last 0.1014 (min 0.0954, max 0.9260)
  - `Validation/LPIPS_t1_gd`: last 0.0533 (min 0.0509, max 0.9170)
  - `Validation/LPIPS_t1_pre`: last 0.0496 (min 0.0464, max 0.9196)
  - `Validation/MS-SSIM`: last 0.9826 (min 0.4747, max 0.9831)
  - `Validation/MS-SSIM-3D`: last 0.9879 (min 0.3757, max 0.9881)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9854 (min 0.4017, max 0.9857)
  - `Validation/MS-SSIM-3D_flair`: last 0.9827 (min 0.3785, max 0.9833)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9915 (min 0.3784, max 0.9917)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9926 (min 0.3428, max 0.9926)
  - `Validation/MS-SSIM_bravo`: last 0.9804 (min 0.4881, max 0.9812)
  - `Validation/MS-SSIM_flair`: last 0.9763 (min 0.4903, max 0.9773)
  - `Validation/MS-SSIM_t1_gd`: last 0.9882 (min 0.4794, max 0.9888)
  - `Validation/MS-SSIM_t1_pre`: last 0.9895 (min 0.4406, max 0.9897)
  - `Validation/PSNR`: last 36.0996 (min 20.3192, max 36.3672)
  - `Validation/PSNR_bravo`: last 34.5125 (min 21.3996, max 34.8171)
  - `Validation/PSNR_flair`: last 34.7646 (min 21.1117, max 35.0728)
  - `Validation/PSNR_t1_gd`: last 37.9171 (min 20.5285, max 38.2298)
  - `Validation/PSNR_t1_pre`: last 37.2044 (min 18.2369, max 37.3661)

**Regional loss (final):**
  - `regional/background_loss`: 0.0043
  - `regional/large`: 0.0495
  - `regional/medium`: 0.0417
  - `regional/small`: 0.0409
  - `regional/tiny`: 0.0353
  - `regional/tumor_bg_ratio`: 9.9298
  - `regional/tumor_loss`: 0.0429
  - `regional_bravo/background_loss`: 0.0047
  - `regional_bravo/large`: 0.0539
  - `regional_bravo/medium`: 0.0500
  - `regional_bravo/small`: 0.0416
  - `regional_bravo/tiny`: 0.0391
  - `regional_bravo/tumor_bg_ratio`: 10.0065
  - `regional_bravo/tumor_loss`: 0.0475
  - `regional_flair/background_loss`: 0.0055
  - `regional_flair/large`: 0.0613
  - `regional_flair/medium`: 0.0522
  - `regional_flair/small`: 0.0537
  - `regional_flair/tiny`: 0.0487
  - `regional_flair/tumor_bg_ratio`: 9.9961
  - `regional_flair/tumor_loss`: 0.0548
  - `regional_t1_gd/background_loss`: 0.0032
  - `regional_t1_gd/large`: 0.0498
  - `regional_t1_gd/medium`: 0.0411
  - `regional_t1_gd/small`: 0.0391
  - `regional_t1_gd/tiny`: 0.0303
  - `regional_t1_gd/tumor_bg_ratio`: 12.7815
  - `regional_t1_gd/tumor_loss`: 0.0415
  - `regional_t1_pre/background_loss`: 0.0038
  - `regional_t1_pre/large`: 0.0330
  - `regional_t1_pre/medium`: 0.0236
  - `regional_t1_pre/small`: 0.0293
  - `regional_t1_pre/tiny`: 0.0229
  - `regional_t1_pre/tumor_bg_ratio`: 7.2970
  - `regional_t1_pre/tumor_loss`: 0.0277

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.6831, max 10.8721 @ ep 4
  - `training/grad_norm_d_max`: last 16.3207, max 719.2972 @ ep 4
  - `training/grad_norm_g_avg`: last 0.0288, max 0.8566 @ ep 4
  - `training/grad_norm_g_max`: last 0.0871, max 6.6241 @ ep 43

#### `exp8_11_256x160_20260112-121110`
*started 2026-01-12 12:11 • 125 epochs • 41h53m • 99.1 TFLOPs • peak VRAM 24.1 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0460 → 0.0044 (min 0.0044 @ ep 88)
  - `Loss/L1_val`: 0.0393 → 0.0046 (min 0.0044 @ ep 95)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0838 (min 0.0786, max 0.9283)
  - `Validation/LPIPS_bravo`: last 0.1211 (min 0.1129, max 0.9693)
  - `Validation/LPIPS_flair`: last 0.1123 (min 0.1052, max 0.9473)
  - `Validation/LPIPS_t1_gd`: last 0.0581 (min 0.0540, max 0.9387)
  - `Validation/LPIPS_t1_pre`: last 0.0532 (min 0.0499, max 0.9419)
  - `Validation/MS-SSIM`: last 0.9817 (min 0.4608, max 0.9823)
  - `Validation/MS-SSIM-3D`: last 0.9874 (min 0.3722, max 0.9876)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9848 (min 0.3976, max 0.9852)
  - `Validation/MS-SSIM-3D_flair`: last 0.9819 (min 0.3746, max 0.9824)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9911 (min 0.3748, max 0.9914)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9920 (min 0.3401, max 0.9922)
  - `Validation/MS-SSIM_bravo`: last 0.9792 (min 0.4737, max 0.9802)
  - `Validation/MS-SSIM_flair`: last 0.9746 (min 0.4765, max 0.9760)
  - `Validation/MS-SSIM_t1_gd`: last 0.9874 (min 0.4652, max 0.9882)
  - `Validation/MS-SSIM_t1_pre`: last 0.9886 (min 0.4275, max 0.9890)
  - `Validation/PSNR`: last 35.7914 (min 20.2915, max 36.1411)
  - `Validation/PSNR_bravo`: last 34.2522 (min 21.3684, max 34.5592)
  - `Validation/PSNR_flair`: last 34.4415 (min 21.0787, max 34.9098)
  - `Validation/PSNR_t1_gd`: last 37.6434 (min 20.5009, max 37.9842)
  - `Validation/PSNR_t1_pre`: last 36.8283 (min 18.2182, max 37.1294)

**Regional loss (final):**
  - `regional/background_loss`: 0.0046
  - `regional/large`: 0.0511
  - `regional/medium`: 0.0429
  - `regional/small`: 0.0421
  - `regional/tiny`: 0.0358
  - `regional/tumor_bg_ratio`: 9.6323
  - `regional/tumor_loss`: 0.0441
  - `regional_bravo/background_loss`: 0.0050
  - `regional_bravo/large`: 0.0566
  - `regional_bravo/medium`: 0.0512
  - `regional_bravo/small`: 0.0415
  - `regional_bravo/tiny`: 0.0393
  - `regional_bravo/tumor_bg_ratio`: 9.7203
  - `regional_bravo/tumor_loss`: 0.0488
  - `regional_flair/background_loss`: 0.0058
  - `regional_flair/large`: 0.0631
  - `regional_flair/medium`: 0.0536
  - `regional_flair/small`: 0.0560
  - `regional_flair/tiny`: 0.0495
  - `regional_flair/tumor_bg_ratio`: 9.7718
  - `regional_flair/tumor_loss`: 0.0564
  - `regional_t1_gd/background_loss`: 0.0034
  - `regional_t1_gd/large`: 0.0522
  - `regional_t1_gd/medium`: 0.0426
  - `regional_t1_gd/small`: 0.0408
  - `regional_t1_gd/tiny`: 0.0306
  - `regional_t1_gd/tumor_bg_ratio`: 12.5020
  - `regional_t1_gd/tumor_loss`: 0.0431
  - `regional_t1_pre/background_loss`: 0.0041
  - `regional_t1_pre/large`: 0.0326
  - `regional_t1_pre/medium`: 0.0243
  - `regional_t1_pre/small`: 0.0299
  - `regional_t1_pre/tiny`: 0.0239
  - `regional_t1_pre/tumor_bg_ratio`: 6.8948
  - `regional_t1_pre/tumor_loss`: 0.0280

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.8839, max 9.7157 @ ep 4
  - `training/grad_norm_d_max`: last 16.3143, max 123.3806 @ ep 6
  - `training/grad_norm_g_avg`: last 0.0414, max 0.5825 @ ep 5
  - `training/grad_norm_g_max`: last 0.1819, max 1.8420 @ ep 11

#### `exp8_12_256x160_20260112-121110`
*started 2026-01-12 12:11 • 125 epochs • 46h12m • 118.9 TFLOPs • peak VRAM 24.3 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0428 → 0.0044 (min 0.0044 @ ep 98)
  - `Loss/L1_val`: 0.0387 → 0.0047 (min 0.0044 @ ep 94)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0822 (min 0.0793, max 0.9162)
  - `Validation/LPIPS_bravo`: last 0.1218 (min 0.1170, max 0.9583)
  - `Validation/LPIPS_flair`: last 0.1066 (min 0.1038, max 0.9352)
  - `Validation/LPIPS_t1_gd`: last 0.0576 (min 0.0557, max 0.9271)
  - `Validation/LPIPS_t1_pre`: last 0.0516 (min 0.0501, max 0.9300)
  - `Validation/MS-SSIM`: last 0.9811 (min 0.4661, max 0.9820)
  - `Validation/MS-SSIM-3D`: last 0.9869 (min 0.3734, max 0.9872)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9843 (min 0.3990, max 0.9847)
  - `Validation/MS-SSIM-3D_flair`: last 0.9812 (min 0.3759, max 0.9819)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9907 (min 0.3760, max 0.9910)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9918 (min 0.3410, max 0.9920)
  - `Validation/MS-SSIM_bravo`: last 0.9786 (min 0.4791, max 0.9796)
  - `Validation/MS-SSIM_flair`: last 0.9737 (min 0.4816, max 0.9757)
  - `Validation/MS-SSIM_t1_gd`: last 0.9870 (min 0.4705, max 0.9877)
  - `Validation/MS-SSIM_t1_pre`: last 0.9884 (min 0.4325, max 0.9888)
  - `Validation/PSNR`: last 35.6839 (min 20.3129, max 36.0392)
  - `Validation/PSNR_bravo`: last 34.1805 (min 21.3929, max 34.5389)
  - `Validation/PSNR_flair`: last 34.3444 (min 21.1054, max 34.8545)
  - `Validation/PSNR_t1_gd`: last 37.4985 (min 20.5218, max 37.8498)
  - `Validation/PSNR_t1_pre`: last 36.7123 (min 18.2313, max 37.0323)

**Regional loss (final):**
  - `regional/background_loss`: 0.0046
  - `regional/large`: 0.0514
  - `regional/medium`: 0.0430
  - `regional/small`: 0.0420
  - `regional/tiny`: 0.0361
  - `regional/tumor_bg_ratio`: 9.5347
  - `regional/tumor_loss`: 0.0442
  - `regional_bravo/background_loss`: 0.0051
  - `regional_bravo/large`: 0.0560
  - `regional_bravo/medium`: 0.0507
  - `regional_bravo/small`: 0.0425
  - `regional_bravo/tiny`: 0.0394
  - `regional_bravo/tumor_bg_ratio`: 9.5777
  - `regional_bravo/tumor_loss`: 0.0487
  - `regional_flair/background_loss`: 0.0059
  - `regional_flair/large`: 0.0649
  - `regional_flair/medium`: 0.0539
  - `regional_flair/small`: 0.0553
  - `regional_flair/tiny`: 0.0498
  - `regional_flair/tumor_bg_ratio`: 9.7353
  - `regional_flair/tumor_loss`: 0.0570
  - `regional_t1_gd/background_loss`: 0.0035
  - `regional_t1_gd/large`: 0.0518
  - `regional_t1_gd/medium`: 0.0420
  - `regional_t1_gd/small`: 0.0404
  - `regional_t1_gd/tiny`: 0.0308
  - `regional_t1_gd/tumor_bg_ratio`: 12.1871
  - `regional_t1_gd/tumor_loss`: 0.0427
  - `regional_t1_pre/background_loss`: 0.0041
  - `regional_t1_pre/large`: 0.0331
  - `regional_t1_pre/medium`: 0.0253
  - `regional_t1_pre/small`: 0.0298
  - `regional_t1_pre/tiny`: 0.0243
  - `regional_t1_pre/tumor_bg_ratio`: 6.9366
  - `regional_t1_pre/tumor_loss`: 0.0285

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 6.0968, max 9.3393 @ ep 4
  - `training/grad_norm_d_max`: last 18.4129, max 88.2451 @ ep 5
  - `training/grad_norm_g_avg`: last 0.0452, max 0.7978 @ ep 2
  - `training/grad_norm_g_max`: last 0.1692, max 2.3304 @ ep 3

#### `exp8_13_256x160_20260112-121110`
*started 2026-01-12 12:11 • 125 epochs • 27h25m • 84.2 TFLOPs • peak VRAM 24.1 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0470 → 0.0051 (min 0.0050 @ ep 99)
  - `Loss/L1_val`: 0.0407 → 0.0053 (min 0.0051 @ ep 102)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0658 (min 0.0644, max 0.9564)
  - `Validation/LPIPS_bravo`: last 0.0891 (min 0.0879, max 1.0213)
  - `Validation/LPIPS_flair`: last 0.0866 (min 0.0849, max 1.0016)
  - `Validation/LPIPS_t1_gd`: last 0.0495 (min 0.0487, max 0.9894)
  - `Validation/LPIPS_t1_pre`: last 0.0451 (min 0.0439, max 0.9907)
  - `Validation/MS-SSIM`: last 0.9766 (min 0.4688, max 0.9773)
  - `Validation/MS-SSIM-3D`: last 0.9826 (min 0.3756, max 0.9831)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9798 (min 0.4015, max 0.9801)
  - `Validation/MS-SSIM-3D_flair`: last 0.9753 (min 0.3784, max 0.9760)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9875 (min 0.3783, max 0.9880)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9886 (min 0.3427, max 0.9889)
  - `Validation/MS-SSIM_bravo`: last 0.9738 (min 0.4820, max 0.9744)
  - `Validation/MS-SSIM_flair`: last 0.9679 (min 0.4841, max 0.9691)
  - `Validation/MS-SSIM_t1_gd`: last 0.9834 (min 0.4733, max 0.9841)
  - `Validation/MS-SSIM_t1_pre`: last 0.9844 (min 0.4345, max 0.9849)
  - `Validation/PSNR`: last 34.7848 (min 20.2388, max 35.0361)
  - `Validation/PSNR_bravo`: last 33.3640 (min 21.3059, max 33.5662)
  - `Validation/PSNR_flair`: last 33.6046 (min 21.0077, max 33.8683)
  - `Validation/PSNR_t1_gd`: last 36.5809 (min 20.4505, max 36.8713)
  - `Validation/PSNR_t1_pre`: last 35.5896 (min 18.1911, max 35.8386)

**Regional loss (final):**
  - `regional/background_loss`: 0.0053
  - `regional/large`: 0.0596
  - `regional/medium`: 0.0501
  - `regional/small`: 0.0485
  - `regional/tiny`: 0.0408
  - `regional/tumor_bg_ratio`: 9.6054
  - `regional/tumor_loss`: 0.0511
  - `regional_bravo/background_loss`: 0.0058
  - `regional_bravo/large`: 0.0659
  - `regional_bravo/medium`: 0.0602
  - `regional_bravo/small`: 0.0491
  - `regional_bravo/tiny`: 0.0446
  - `regional_bravo/tumor_bg_ratio`: 9.8722
  - `regional_bravo/tumor_loss`: 0.0569
  - `regional_flair/background_loss`: 0.0066
  - `regional_flair/large`: 0.0721
  - `regional_flair/medium`: 0.0606
  - `regional_flair/small`: 0.0634
  - `regional_flair/tiny`: 0.0557
  - `regional_flair/tumor_bg_ratio`: 9.7473
  - `regional_flair/tumor_loss`: 0.0640
  - `regional_t1_gd/background_loss`: 0.0041
  - `regional_t1_gd/large`: 0.0627
  - `regional_t1_gd/medium`: 0.0511
  - `regional_t1_gd/small`: 0.0465
  - `regional_t1_gd/tiny`: 0.0351
  - `regional_t1_gd/tumor_bg_ratio`: 12.3910
  - `regional_t1_gd/tumor_loss`: 0.0509
  - `regional_t1_pre/background_loss`: 0.0049
  - `regional_t1_pre/large`: 0.0378
  - `regional_t1_pre/medium`: 0.0286
  - `regional_t1_pre/small`: 0.0350
  - `regional_t1_pre/tiny`: 0.0277
  - `regional_t1_pre/tumor_bg_ratio`: 6.7385
  - `regional_t1_pre/tumor_loss`: 0.0327

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.5667, max 8.6648 @ ep 9
  - `training/grad_norm_d_max`: last 13.4516, max 66.4675 @ ep 21
  - `training/grad_norm_g_avg`: last 0.1675, max 1.9124 @ ep 5
  - `training/grad_norm_g_max`: last 0.6718, max 4.8894 @ ep 42

#### `exp8_14_256x160_20260112-121110`
*started 2026-01-12 12:11 • 125 epochs • 32h32m • 165.2 TFLOPs • peak VRAM 24.8 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0381 → 0.0044 (min 0.0044 @ ep 105)
  - `Loss/L1_val`: 0.0379 → 0.0046 (min 0.0045 @ ep 105)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0515 (min 0.0511, max 0.8477)
  - `Validation/LPIPS_bravo`: last 0.0718 (min 0.0705, max 0.9175)
  - `Validation/LPIPS_flair`: last 0.0687 (min 0.0676, max 0.8962)
  - `Validation/LPIPS_t1_gd`: last 0.0379 (min 0.0367, max 0.8867)
  - `Validation/LPIPS_t1_pre`: last 0.0338 (min 0.0328, max 0.8880)
  - `Validation/MS-SSIM`: last 0.9816 (min 0.4856, max 0.9819)
  - `Validation/MS-SSIM-3D`: last 0.9870 (min 0.3783, max 0.9872)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9844 (min 0.4047, max 0.9846)
  - `Validation/MS-SSIM-3D_flair`: last 0.9813 (min 0.3814, max 0.9815)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9908 (min 0.3810, max 0.9910)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9919 (min 0.3448, max 0.9920)
  - `Validation/MS-SSIM_bravo`: last 0.9789 (min 0.4995, max 0.9794)
  - `Validation/MS-SSIM_flair`: last 0.9747 (min 0.5012, max 0.9751)
  - `Validation/MS-SSIM_t1_gd`: last 0.9873 (min 0.4906, max 0.9877)
  - `Validation/MS-SSIM_t1_pre`: last 0.9886 (min 0.4508, max 0.9888)
  - `Validation/PSNR`: last 35.6982 (min 20.3685, max 35.9205)
  - `Validation/PSNR_bravo`: last 34.0667 (min 21.4562, max 34.4359)
  - `Validation/PSNR_flair`: last 34.4854 (min 21.1735, max 34.7060)
  - `Validation/PSNR_t1_gd`: last 37.4722 (min 20.5769, max 37.7385)
  - `Validation/PSNR_t1_pre`: last 36.7686 (min 18.2674, max 36.9856)

**Regional loss (final):**
  - `regional/background_loss`: 0.0046
  - `regional/large`: 0.0523
  - `regional/medium`: 0.0450
  - `regional/small`: 0.0427
  - `regional/tiny`: 0.0368
  - `regional/tumor_bg_ratio`: 9.8629
  - `regional/tumor_loss`: 0.0454
  - `regional_bravo/background_loss`: 0.0051
  - `regional_bravo/large`: 0.0574
  - `regional_bravo/medium`: 0.0546
  - `regional_bravo/small`: 0.0441
  - `regional_bravo/tiny`: 0.0406
  - `regional_bravo/tumor_bg_ratio`: 9.9620
  - `regional_bravo/tumor_loss`: 0.0508
  - `regional_flair/background_loss`: 0.0058
  - `regional_flair/large`: 0.0654
  - `regional_flair/medium`: 0.0559
  - `regional_flair/small`: 0.0568
  - `regional_flair/tiny`: 0.0512
  - `regional_flair/tumor_bg_ratio`: 10.1287
  - `regional_flair/tumor_loss`: 0.0583
  - `regional_t1_gd/background_loss`: 0.0035
  - `regional_t1_gd/large`: 0.0522
  - `regional_t1_gd/medium`: 0.0444
  - `regional_t1_gd/small`: 0.0395
  - `regional_t1_gd/tiny`: 0.0312
  - `regional_t1_gd/tumor_bg_ratio`: 12.4705
  - `regional_t1_gd/tumor_loss`: 0.0435
  - `regional_t1_pre/background_loss`: 0.0041
  - `regional_t1_pre/large`: 0.0342
  - `regional_t1_pre/medium`: 0.0251
  - `regional_t1_pre/small`: 0.0304
  - `regional_t1_pre/tiny`: 0.0242
  - `regional_t1_pre/tumor_bg_ratio`: 7.1258
  - `regional_t1_pre/tumor_loss`: 0.0290

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 5.2329, max 9.1099 @ ep 3
  - `training/grad_norm_d_max`: last 18.4043, max 72.7005 @ ep 8
  - `training/grad_norm_g_avg`: last 0.5885, max 1.7136 @ ep 4
  - `training/grad_norm_g_max`: last 1.3735, max 7.4037 @ ep 13

### exp14

**exp14 (3D VAE at different compression ratios)** — 4x and 8x downsample
VAEs at 256×160 with varying KL weight: standard (exp14_0, 14_1),
lowres exp14_2, low-KL exp14_3, no-KL (pure autoencoder) exp14_4.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp14_4_vae3d_4x_nokl_20260219-134723` — 35.5430
  2. 🥈 `exp14_0_vae3d_4x_20260214-233543` — 35.1347
  3.  `exp14_3_vae3d_4x_lowkl_20260217-023208` — 34.9596
  4.  `exp14_2_vae3d_8x_lowres_20260215-013754` — 33.4638
  5.  `exp14_1_vae3d_8x_20260214-233543` — 32.6641

#### `exp14_0_vae3d_4x_20260214-233543`
*started 2026-02-14 23:35 • 125 epochs • 115h56m • 277.7 TFLOPs • peak VRAM 49.7 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0896 (min 0.0880, max 1.0464)
  - `Validation/LPIPS_bravo`: last 0.1370 (min 0.1347, max 1.2111)
  - `Validation/LPIPS_flair`: last 0.1310 (min 0.1285, max 1.2055)
  - `Validation/LPIPS_t1_gd`: last 0.0532 (min 0.0529, max 1.1499)
  - `Validation/LPIPS_t1_pre`: last 0.0423 (min 0.0404, max 1.1531)
  - `Validation/MS-SSIM`: last 0.9888 (min 0.8622, max 0.9889)
  - `Validation/MS-SSIM-3D`: last 0.9925 (min 0.8311, max 0.9926)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9900 (min 0.8559, max 0.9904)
  - `Validation/MS-SSIM-3D_flair`: last 0.9884 (min 0.8245, max 0.9886)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9950 (min 0.8523, max 0.9952)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9957 (min 0.7919, max 0.9959)
  - `Validation/MS-SSIM_bravo`: last 0.9860 (min 0.8712, max 0.9869)
  - `Validation/MS-SSIM_flair`: last 0.9845 (min 0.8564, max 0.9846)
  - `Validation/MS-SSIM_t1_gd`: last 0.9929 (min 0.8757, max 0.9930)
  - `Validation/MS-SSIM_t1_pre`: last 0.9945 (min 0.8445, max 0.9947)
  - `Validation/PSNR`: last 36.7397 (min 26.6775, max 36.8445)
  - `Validation/PSNR_bravo`: last 34.9950 (min 27.8032, max 35.1347)
  - `Validation/PSNR_flair`: last 35.4168 (min 26.7743, max 35.5216)
  - `Validation/PSNR_t1_gd`: last 38.4861 (min 27.8608, max 38.6284)
  - `Validation/PSNR_t1_pre`: last 38.0356 (min 24.1882, max 38.2097)

**Regional loss (final):**
  - `regional/background_loss`: 0.0044
  - `regional/large`: 0.0423
  - `regional/medium`: 0.0391
  - `regional/small`: 0.0412
  - `regional/tiny`: 0.0317
  - `regional/tumor_bg_ratio`: 8.9576
  - `regional/tumor_loss`: 0.0391
  - `regional_bravo/background_loss`: 0.0047
  - `regional_bravo/large`: 0.0434
  - `regional_bravo/medium`: 0.0394
  - `regional_bravo/small`: 0.0364
  - `regional_bravo/tiny`: 0.0327
  - `regional_bravo/tumor_bg_ratio`: 8.2930
  - `regional_bravo/tumor_loss`: 0.0388
  - `regional_flair/background_loss`: 0.0054
  - `regional_flair/large`: 0.0581
  - `regional_flair/medium`: 0.0570
  - `regional_flair/small`: 0.0662
  - `regional_flair/tiny`: 0.0467
  - `regional_flair/tumor_bg_ratio`: 10.6087
  - `regional_flair/tumor_loss`: 0.0571
  - `regional_t1_gd/background_loss`: 0.0033
  - `regional_t1_gd/large`: 0.0397
  - `regional_t1_gd/medium`: 0.0348
  - `regional_t1_gd/small`: 0.0346
  - `regional_t1_gd/tiny`: 0.0256
  - `regional_t1_gd/tumor_bg_ratio`: 10.3217
  - `regional_t1_gd/tumor_loss`: 0.0346
  - `regional_t1_pre/background_loss`: 0.0040
  - `regional_t1_pre/large`: 0.0266
  - `regional_t1_pre/medium`: 0.0242
  - `regional_t1_pre/small`: 0.0274
  - `regional_t1_pre/tiny`: 0.0220
  - `regional_t1_pre/tumor_bg_ratio`: 6.2179
  - `regional_t1_pre/tumor_loss`: 0.0252

**LR schedule:**
  - `LR/Discriminator`: peak 0.0001 @ ep 4, final 2.082e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.535e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 63.3080, max 64.1945 @ ep 116
  - `training/grad_norm_d_max`: last 345.8932, max 602.8953 @ ep 113
  - `training/grad_norm_g_avg`: last 4.9006, max 12.2664 @ ep 18
  - `training/grad_norm_g_max`: last 33.6182, max 102.8463 @ ep 47

#### `exp14_1_vae3d_8x_20260214-233543`
*started 2026-02-14 23:35 • 125 epochs • 112h38m • 280.8 TFLOPs • peak VRAM 50.0 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.1611 (min 0.1570, max 0.7388)
  - `Validation/LPIPS_bravo`: last 0.2078 (min 0.2065, max 0.8558)
  - `Validation/LPIPS_flair`: last 0.2098 (min 0.2047, max 0.9513)
  - `Validation/LPIPS_t1_gd`: last 0.1219 (min 0.1131, max 0.8371)
  - `Validation/LPIPS_t1_pre`: last 0.1185 (min 0.1107, max 0.9321)
  - `Validation/MS-SSIM`: last 0.9685 (min 0.8449, max 0.9709)
  - `Validation/MS-SSIM-3D`: last 0.9746 (min 0.8204, max 0.9761)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9700 (min 0.8283, max 0.9722)
  - `Validation/MS-SSIM-3D_flair`: last 0.9644 (min 0.8168, max 0.9656)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9814 (min 0.8464, max 0.9835)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9841 (min 0.7917, max 0.9849)
  - `Validation/MS-SSIM_bravo`: last 0.9639 (min 0.8471, max 0.9672)
  - `Validation/MS-SSIM_flair`: last 0.9580 (min 0.8416, max 0.9603)
  - `Validation/MS-SSIM_t1_gd`: last 0.9765 (min 0.8610, max 0.9794)
  - `Validation/MS-SSIM_t1_pre`: last 0.9797 (min 0.8298, max 0.9808)
  - `Validation/PSNR`: last 33.3348 (min 26.9008, max 33.7015)
  - `Validation/PSNR_bravo`: last 32.3237 (min 27.7882, max 32.6641)
  - `Validation/PSNR_flair`: last 31.9279 (min 26.3828, max 32.2195)
  - `Validation/PSNR_t1_gd`: last 34.8629 (min 28.2342, max 35.3729)
  - `Validation/PSNR_t1_pre`: last 34.2271 (min 24.8610, max 34.6044)

**Regional loss (final):**
  - `regional/background_loss`: 0.0063
  - `regional/large`: 0.0625
  - `regional/medium`: 0.0563
  - `regional/small`: 0.0558
  - `regional/tiny`: 0.0444
  - `regional/tumor_bg_ratio`: 8.9353
  - `regional/tumor_loss`: 0.0559
  - `regional_bravo/background_loss`: 0.0065
  - `regional_bravo/large`: 0.0675
  - `regional_bravo/medium`: 0.0643
  - `regional_bravo/small`: 0.0562
  - `regional_bravo/tiny`: 0.0477
  - `regional_bravo/tumor_bg_ratio`: 9.3252
  - `regional_bravo/tumor_loss`: 0.0605
  - `regional_flair/background_loss`: 0.0077
  - `regional_flair/large`: 0.0821
  - `regional_flair/medium`: 0.0714
  - `regional_flair/small`: 0.0736
  - `regional_flair/tiny`: 0.0583
  - `regional_flair/tumor_bg_ratio`: 9.4163
  - `regional_flair/tumor_loss`: 0.0728
  - `regional_t1_gd/background_loss`: 0.0051
  - `regional_t1_gd/large`: 0.0612
  - `regional_t1_gd/medium`: 0.0569
  - `regional_t1_gd/small`: 0.0540
  - `regional_t1_gd/tiny`: 0.0378
  - `regional_t1_gd/tumor_bg_ratio`: 10.6564
  - `regional_t1_gd/tumor_loss`: 0.0540
  - `regional_t1_pre/background_loss`: 0.0057
  - `regional_t1_pre/large`: 0.0385
  - `regional_t1_pre/medium`: 0.0323
  - `regional_t1_pre/small`: 0.0397
  - `regional_t1_pre/tiny`: 0.0338
  - `regional_t1_pre/tumor_bg_ratio`: 6.2880
  - `regional_t1_pre/tumor_loss`: 0.0361

**LR schedule:**
  - `LR/Discriminator`: peak 0.0001 @ ep 4, final 2.082e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.535e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 131.9605, max 142.6299 @ ep 117
  - `training/grad_norm_d_max`: last 657.7327, max 808.7524 @ ep 99
  - `training/grad_norm_g_avg`: last 8.3402, max 12.2413 @ ep 16
  - `training/grad_norm_g_max`: last 23.2480, max 29.4618 @ ep 20

#### `exp14_2_vae3d_8x_lowres_20260215-013754`
*started 2026-02-15 01:37 • 125 epochs • 66h34m • 139.6 TFLOPs • peak VRAM 25.5 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.1369 (min 0.1334, max 1.0639)
  - `Validation/LPIPS_bravo`: last 0.1533 (min 0.1491, max 1.2036)
  - `Validation/LPIPS_flair`: last 0.1807 (min 0.1769, max 1.2807)
  - `Validation/LPIPS_t1_gd`: last 0.1073 (min 0.1050, max 1.1579)
  - `Validation/LPIPS_t1_pre`: last 0.1335 (min 0.1231, max 1.2712)
  - `Validation/MS-SSIM`: last 0.9734 (min 0.7841, max 0.9744)
  - `Validation/MS-SSIM-3D`: last 0.9766 (min 0.7450, max 0.9769)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9757 (min 0.7416, max 0.9758)
  - `Validation/MS-SSIM-3D_flair`: last 0.9689 (min 0.7271, max 0.9695)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9807 (min 0.7735, max 0.9810)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9802 (min 0.6998, max 0.9807)
  - `Validation/MS-SSIM_bravo`: last 0.9721 (min 0.7922, max 0.9728)
  - `Validation/MS-SSIM_flair`: last 0.9647 (min 0.7792, max 0.9668)
  - `Validation/MS-SSIM_t1_gd`: last 0.9778 (min 0.8025, max 0.9781)
  - `Validation/MS-SSIM_t1_pre`: last 0.9775 (min 0.7253, max 0.9785)
  - `Validation/PSNR`: last 32.9785 (min 23.7405, max 33.4019)
  - `Validation/PSNR_bravo`: last 33.0398 (min 22.7476, max 33.4638)
  - `Validation/PSNR_flair`: last 31.8324 (min 22.7465, max 32.4766)
  - `Validation/PSNR_t1_gd`: last 34.3371 (min 24.0981, max 34.6319)
  - `Validation/PSNR_t1_pre`: last 32.7053 (min 23.8318, max 33.0736)

**Regional loss (final):**
  - `regional/background_loss`: 0.0063
  - `regional/large`: 0.0705
  - `regional/medium`: 0.0641
  - `regional/small`: 0.0632
  - `regional/tiny`: 0.0503
  - `regional/tumor_bg_ratio`: 10.1096
  - `regional/tumor_loss`: 0.0635
  - `regional_bravo/background_loss`: 0.0059
  - `regional_bravo/large`: 0.0676
  - `regional_bravo/medium`: 0.0649
  - `regional_bravo/small`: 0.0622
  - `regional_bravo/tiny`: 0.0522
  - `regional_bravo/tumor_bg_ratio`: 10.6174
  - `regional_bravo/tumor_loss`: 0.0631
  - `regional_flair/background_loss`: 0.0073
  - `regional_flair/large`: 0.0915
  - `regional_flair/medium`: 0.0876
  - `regional_flair/small`: 0.0757
  - `regional_flair/tiny`: 0.0630
  - `regional_flair/tumor_bg_ratio`: 11.3457
  - `regional_flair/tumor_loss`: 0.0824
  - `regional_t1_gd/background_loss`: 0.0052
  - `regional_t1_gd/large`: 0.0679
  - `regional_t1_gd/medium`: 0.0617
  - `regional_t1_gd/small`: 0.0650
  - `regional_t1_gd/tiny`: 0.0464
  - `regional_t1_gd/tumor_bg_ratio`: 11.7505
  - `regional_t1_gd/tumor_loss`: 0.0616
  - `regional_t1_pre/background_loss`: 0.0067
  - `regional_t1_pre/large`: 0.0563
  - `regional_t1_pre/medium`: 0.0425
  - `regional_t1_pre/small`: 0.0490
  - `regional_t1_pre/tiny`: 0.0390
  - `regional_t1_pre/tumor_bg_ratio`: 7.0591
  - `regional_t1_pre/tumor_loss`: 0.0471

**LR schedule:**
  - `LR/Discriminator`: peak 0.0001 @ ep 4, final 1.271e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.134e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 159.6509, max 247.0933 @ ep 96
  - `training/grad_norm_d_max`: last 452.7345, max 983.9886 @ ep 95
  - `training/grad_norm_g_avg`: last 3.5298, max 14.5333 @ ep 8
  - `training/grad_norm_g_max`: last 10.6520, max 27.6926 @ ep 12

#### `exp14_3_vae3d_4x_lowkl_20260217-023208`
*started 2026-02-17 02:32 • 125 epochs • 96h59m • 277.7 TFLOPs • peak VRAM 49.7 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0903 (min 0.0844, max 0.9849)
  - `Validation/LPIPS_bravo`: last 0.1431 (min 0.1293, max 1.1497)
  - `Validation/LPIPS_flair`: last 0.1336 (min 0.1259, max 1.1412)
  - `Validation/LPIPS_t1_gd`: last 0.0501 (min 0.0476, max 1.0917)
  - `Validation/LPIPS_t1_pre`: last 0.0422 (min 0.0395, max 1.0960)
  - `Validation/MS-SSIM`: last 0.9883 (min 0.8808, max 0.9887)
  - `Validation/MS-SSIM-3D`: last 0.9915 (min 0.8518, max 0.9918)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9892 (min 0.8746, max 0.9899)
  - `Validation/MS-SSIM-3D_flair`: last 0.9861 (min 0.8523, max 0.9863)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9946 (min 0.8708, max 0.9951)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9957 (min 0.8081, max 0.9959)
  - `Validation/MS-SSIM_bravo`: last 0.9857 (min 0.8899, max 0.9865)
  - `Validation/MS-SSIM_flair`: last 0.9830 (min 0.8788, max 0.9833)
  - `Validation/MS-SSIM_t1_gd`: last 0.9925 (min 0.8923, max 0.9930)
  - `Validation/MS-SSIM_t1_pre`: last 0.9944 (min 0.8587, max 0.9945)
  - `Validation/PSNR`: last 36.2735 (min 27.0789, max 36.5062)
  - `Validation/PSNR_bravo`: last 34.6646 (min 27.3662, max 34.9596)
  - `Validation/PSNR_flair`: last 34.6618 (min 27.1157, max 34.8201)
  - `Validation/PSNR_t1_gd`: last 37.9945 (min 28.2101, max 38.4092)
  - `Validation/PSNR_t1_pre`: last 37.7091 (min 24.2709, max 37.9816)

**Regional loss (final):**
  - `regional/background_loss`: 0.0046
  - `regional/large`: 0.0445
  - `regional/medium`: 0.0400
  - `regional/small`: 0.0482
  - `regional/tiny`: 0.0348
  - `regional/tumor_bg_ratio`: 9.1742
  - `regional/tumor_loss`: 0.0420
  - `regional_bravo/background_loss`: 0.0049
  - `regional_bravo/large`: 0.0428
  - `regional_bravo/medium`: 0.0394
  - `regional_bravo/small`: 0.0382
  - `regional_bravo/tiny`: 0.0347
  - `regional_bravo/tumor_bg_ratio`: 8.0014
  - `regional_bravo/tumor_loss`: 0.0394
  - `regional_flair/background_loss`: 0.0057
  - `regional_flair/large`: 0.0646
  - `regional_flair/medium`: 0.0587
  - `regional_flair/small`: 0.0896
  - `regional_flair/tiny`: 0.0493
  - `regional_flair/tumor_bg_ratio`: 11.2626
  - `regional_flair/tumor_loss`: 0.0647
  - `regional_t1_gd/background_loss`: 0.0036
  - `regional_t1_gd/large`: 0.0433
  - `regional_t1_gd/medium`: 0.0357
  - `regional_t1_gd/small`: 0.0374
  - `regional_t1_gd/tiny`: 0.0260
  - `regional_t1_gd/tumor_bg_ratio`: 10.2929
  - `regional_t1_gd/tumor_loss`: 0.0366
  - `regional_t1_pre/background_loss`: 0.0041
  - `regional_t1_pre/large`: 0.0263
  - `regional_t1_pre/medium`: 0.0250
  - `regional_t1_pre/small`: 0.0304
  - `regional_t1_pre/tiny`: 0.0291
  - `regional_t1_pre/tumor_bg_ratio`: 6.5715
  - `regional_t1_pre/tumor_loss`: 0.0273

**LR schedule:**
  - `LR/Discriminator`: peak 0.0001 @ ep 4, final 1.829e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.41e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 107.9951, max 113.5925 @ ep 118
  - `training/grad_norm_d_max`: last 1141.1809, max 1450.2699 @ ep 120
  - `training/grad_norm_g_avg`: last 10.1709, max 12.9697 @ ep 14
  - `training/grad_norm_g_max`: last 42.3358, max 62.9517 @ ep 112

#### `exp14_4_vae3d_4x_nokl_20260219-134723`
*started 2026-02-19 13:47 • 125 epochs • 121h53m • 277.7 TFLOPs • peak VRAM 49.7 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0821 (min 0.0779, max 0.9208)
  - `Validation/LPIPS_bravo`: last 0.1301 (min 0.1238, max 1.1102)
  - `Validation/LPIPS_flair`: last 0.1150 (min 0.1107, max 1.1195)
  - `Validation/LPIPS_t1_gd`: last 0.0501 (min 0.0470, max 0.9985)
  - `Validation/LPIPS_t1_pre`: last 0.0399 (min 0.0341, max 1.0144)
  - `Validation/MS-SSIM`: last 0.9893 (min 0.8662, max 0.9898)
  - `Validation/MS-SSIM-3D`: last 0.9937 (min 0.8488, max 0.9941)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9916 (min 0.8639, max 0.9919)
  - `Validation/MS-SSIM-3D_flair`: last 0.9912 (min 0.8276, max 0.9915)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9957 (min 0.8791, max 0.9961)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9971 (min 0.8256, max 0.9973)
  - `Validation/MS-SSIM_bravo`: last 0.9871 (min 0.8726, max 0.9879)
  - `Validation/MS-SSIM_flair`: last 0.9859 (min 0.8507, max 0.9864)
  - `Validation/MS-SSIM_t1_gd`: last 0.9930 (min 0.8850, max 0.9935)
  - `Validation/MS-SSIM_t1_pre`: last 0.9951 (min 0.8550, max 0.9954)
  - `Validation/PSNR`: last 37.2788 (min 26.7749, max 37.4886)
  - `Validation/PSNR_bravo`: last 35.3073 (min 27.6944, max 35.5430)
  - `Validation/PSNR_flair`: last 35.9263 (min 26.2366, max 36.1005)
  - `Validation/PSNR_t1_gd`: last 38.7307 (min 28.2283, max 39.0544)
  - `Validation/PSNR_t1_pre`: last 39.1502 (min 24.9576, max 39.3618)

**Regional loss (final):**
  - `regional/background_loss`: 0.0039
  - `regional/large`: 0.0367
  - `regional/medium`: 0.0321
  - `regional/small`: 0.0303
  - `regional/tiny`: 0.0272
  - `regional/tumor_bg_ratio`: 8.2356
  - `regional/tumor_loss`: 0.0323
  - `regional_bravo/background_loss`: 0.0045
  - `regional_bravo/large`: 0.0420
  - `regional_bravo/medium`: 0.0390
  - `regional_bravo/small`: 0.0321
  - `regional_bravo/tiny`: 0.0311
  - `regional_bravo/tumor_bg_ratio`: 8.2106
  - `regional_bravo/tumor_loss`: 0.0371
  - `regional_flair/background_loss`: 0.0049
  - `regional_flair/large`: 0.0437
  - `regional_flair/medium`: 0.0394
  - `regional_flair/small`: 0.0397
  - `regional_flair/tiny`: 0.0374
  - `regional_flair/tumor_bg_ratio`: 8.1909
  - `regional_flair/tumor_loss`: 0.0405
  - `regional_t1_gd/background_loss`: 0.0030
  - `regional_t1_gd/large`: 0.0383
  - `regional_t1_gd/medium`: 0.0328
  - `regional_t1_gd/small`: 0.0291
  - `regional_t1_gd/tiny`: 0.0231
  - `regional_t1_gd/tumor_bg_ratio`: 10.4898
  - `regional_t1_gd/tumor_loss`: 0.0320
  - `regional_t1_pre/background_loss`: 0.0032
  - `regional_t1_pre/large`: 0.0226
  - `regional_t1_pre/medium`: 0.0173
  - `regional_t1_pre/small`: 0.0203
  - `regional_t1_pre/tiny`: 0.0172
  - `regional_t1_pre/tumor_bg_ratio`: 6.1551
  - `regional_t1_pre/tumor_loss`: 0.0196

**LR schedule:**
  - `LR/Discriminator`: peak 0.0001 @ ep 4, final 2.082e-06
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.535e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 89.4233, max 112.4428 @ ep 105
  - `training/grad_norm_d_max`: last 847.5455, max 1484.9352 @ ep 70
  - `training/grad_norm_g_avg`: last 8.3884, max 13.6612 @ ep 22
  - `training/grad_norm_g_max`: last 36.1970, max 40.8383 @ ep 119

### OTHER

**dcae_exp10_1** — early DC-AE attempt. Per CLAUDE.md the DC-AE branch was
abandoned.

#### `dcae_exp10_1_256x160_20260110-024913`
*started 2026-01-10 02:49 • 91 epochs • 118h56m • 1508.2 TFLOPs • peak VRAM 74.1 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0407 → 0.0096 (min 0.0093 @ ep 83)
  - `Loss/L1_val`: 0.0299 → 0.0101 (min 0.0095 @ ep 83)

**Validation quality:**
  - `Validation/LPIPS`: last 0.4509 (min 0.3522, max 1.3563)
  - `Validation/LPIPS_bravo`: last 0.4886 (min 0.3903, max 1.3766)
  - `Validation/LPIPS_flair`: last 0.4779 (min 0.3799, max 1.3755)
  - `Validation/LPIPS_t1_gd`: last 0.4235 (min 0.2832, max 1.3441)
  - `Validation/LPIPS_t1_pre`: last 0.4136 (min 0.2974, max 1.3289)
  - `Validation/MS-SSIM`: last 0.9351 (min 0.6876, max 0.9351)
  - `Validation/MS-SSIM-3D`: last 0.9351 (min 0.6227, max 0.9351)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9289 (min 0.6370, max 0.9289)
  - `Validation/MS-SSIM-3D_flair`: last 0.9254 (min 0.6122, max 0.9254)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9464 (min 0.6335, max 0.9465)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9397 (min 0.6079, max 0.9397)
  - `Validation/MS-SSIM_bravo`: last 0.9301 (min 0.7009, max 0.9301)
  - `Validation/MS-SSIM_flair`: last 0.9264 (min 0.6815, max 0.9264)
  - `Validation/MS-SSIM_t1_gd`: last 0.9456 (min 0.6947, max 0.9456)
  - `Validation/MS-SSIM_t1_pre`: last 0.9383 (min 0.6734, max 0.9383)
  - `Validation/PSNR`: last 30.9133 (min 23.7266, max 30.9548)
  - `Validation/PSNR_bravo`: last 30.4401 (min 23.5736, max 30.4693)
  - `Validation/PSNR_flair`: last 30.3052 (min 23.4356, max 30.3757)
  - `Validation/PSNR_t1_gd`: last 32.4663 (min 24.4281, max 32.5556)
  - `Validation/PSNR_t1_pre`: last 30.4415 (min 22.1573, max 30.5158)

**Regional loss (final):**
  - `regional/background_loss`: 0.0101
  - `regional/large`: 0.0823
  - `regional/medium`: 0.0753
  - `regional/small`: 0.0670
  - `regional/tiny`: 0.0570
  - `regional/tumor_bg_ratio`: 7.1966
  - `regional/tumor_loss`: 0.0724
  - `regional_bravo/background_loss`: 0.0099
  - `regional_bravo/large`: 0.0808
  - `regional_bravo/medium`: 0.0848
  - `regional_bravo/small`: 0.0657
  - `regional_bravo/tiny`: 0.0588
  - `regional_bravo/tumor_bg_ratio`: 7.5045
  - `regional_bravo/tumor_loss`: 0.0746
  - `regional_flair/background_loss`: 0.0112
  - `regional_flair/large`: 0.1034
  - `regional_flair/medium`: 0.0933
  - `regional_flair/small`: 0.0817
  - `regional_flair/tiny`: 0.0715
  - `regional_flair/tumor_bg_ratio`: 8.0328
  - `regional_flair/tumor_loss`: 0.0901
  - `regional_t1_gd/background_loss`: 0.0084
  - `regional_t1_gd/large`: 0.0791
  - `regional_t1_gd/medium`: 0.0808
  - `regional_t1_gd/small`: 0.0669
  - `regional_t1_gd/tiny`: 0.0508
  - `regional_t1_gd/tumor_bg_ratio`: 8.4989
  - `regional_t1_gd/tumor_loss`: 0.0716
  - `regional_t1_pre/background_loss`: 0.0107
  - `regional_t1_pre/large`: 0.0660
  - `regional_t1_pre/medium`: 0.0424
  - `regional_t1_pre/small`: 0.0539
  - `regional_t1_pre/tiny`: 0.0471
  - `regional_t1_pre/tumor_bg_ratio`: 5.0014
  - `regional_t1_pre/tumor_loss`: 0.0533

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1.059e-05

**Training meta:**
  - `training/grad_norm_avg`: last 61.8988, max 76.5943 @ ep 7
  - `training/grad_norm_max`: last 253.6043, max 272.9560 @ ep 82

---
## compression_3d/seg

*1 runs across 1 experiment families.*

### exp11

**exp11_2 (3D seg VQ-VAE at 256×160)** — the seg-channel compression used
by the 2-stage latent pipeline. BCE + boundary losses (appropriate for
binary masks rather than MSE).

#### `exp11_2_vqvae3d_seg_256x160_20260112-012254`
*started 2026-01-12 01:22 • 125 epochs • 3h02m • 46.3 TFLOPs • peak VRAM 5.8 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 1.8965 → 0.1331 (min 0.1331 @ ep 123)

**Validation quality:**
  - `Validation/Dice`: last 0.8979 (min 1.981e-05, max 0.8992)
  - `Validation/IoU`: last 0.8182 (min 1.981e-05, max 0.8204)

**Regional loss (final):**
  - `regional/dice`: 0.9074
  - `regional/dice_large`: 0.9729
  - `regional/dice_medium`: 0.9819
  - `regional/dice_small`: 0.9492
  - `regional/dice_tiny`: 0.8977
  - `regional/iou`: 0.8495

**LR schedule:**
  - `LR/Generator`: peak 5e-05 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.8176, max 225.5766 @ ep 2
  - `training/grad_norm_max`: last 10.9786, max 3323.4495 @ ep 2

---
## downstream/SegResNet

*18 runs across 7 experiment families.*

### exp1

**exp1 (SegResNet baselines)** — 2D and 3D baseline tumor segmentation
trained on real data only (105 volumes). These set the floor that synthetic
augmentation must beat.

#### `exp1_baseline_2d_20260225-214136`
*started 2026-02-25 21:41 • 501 epochs • 60h28m • 11084.8 TFLOPs • peak VRAM 7.3 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2821 (min 0.1894, max 0.3120)
  - `Validation/IoU`: last 0.2515 (min 0.1583, max 0.2748)
  - `Validation/hd95`: last 44.5005 (min 44.5005, max 87.6190)
  - `Validation/precision`: last 0.4914 (min 0.0709, max 0.6328)
  - `Validation/recall`: last 0.3316 (min 0.1808, max 0.4799)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4401
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0.8710
  - `regional/detection_rate_small`: 0.4729
  - `regional/detection_rate_tiny`: 0.4181
  - `regional/dice`: 0.2821
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.7113
  - `regional/dice_small`: 0.3682
  - `regional/dice_tiny`: 0.2436
  - `regional/false_positives`: 677.0000
  - `regional/iou`: 0.2515
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.6401
  - `regional/iou_small`: 0.3211
  - `regional/iou_tiny`: 0.2192

**Downstream seg metrics:**
  - `Validation/hd95`: last 44.5005
  - `Validation/precision`: last 0.4914 (max 0.6328 @ ep 104)
  - `Validation/recall`: last 0.3316 (max 0.4799 @ ep 12)
  - `regional/dice`: last 0.2821 (max 0.3120 @ ep 6)
  - `regional/dice_large`: last 0 (max 0.8275 @ ep 22)
  - `regional/dice_medium`: last 0.7113 (max 0.7157 @ ep 43)
  - `regional/dice_small`: last 0.3682 (max 0.4125 @ ep 6)
  - `regional/dice_tiny`: last 0.2436 (max 0.2593 @ ep 6)
  - `regional/iou`: last 0.2515 (max 0.2748 @ ep 12)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.7559, max 4.6498 @ ep 25
  - `training/grad_norm_max`: last 4.4680, max 25.2897 @ ep 60

#### `exp1_baseline_3d_20260225-231354`
*started 2026-02-25 23:13 • 500 epochs • 11h46m • 34.4 TFLOPs • peak VRAM 63.4 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.1772 (min 0.0395, max 0.2795)
  - `Validation/IoU`: last 0.1347 (min 0.0362, max 0.2110)
  - `Validation/hd95`: last 19.7776 (min 14.6133, max 69.9498)
  - `Validation/precision`: last 0.6049 (min 0.0165, max 0.6420)
  - `Validation/recall`: last 0.4618 (min 0.0234, max 0.5199)

**Regional loss (final):**
  - `regional/detection_rate`: 0.3621
  - `regional/detection_rate_large`: 1.0000
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.5510
  - `regional/detection_rate_tiny`: 0.3229
  - `regional/dice`: 0.1772
  - `regional/dice_large`: 0.8008
  - `regional/dice_medium`: 0.6717
  - `regional/dice_small`: 0.2763
  - `regional/dice_tiny`: 0.1514
  - `regional/false_positives`: 177.0000
  - `regional/iou`: 0.1347
  - `regional/iou_large`: 0.6762
  - `regional/iou_medium`: 0.5370
  - `regional/iou_small`: 0.1969
  - `regional/iou_tiny`: 0.1157

**Downstream seg metrics:**
  - `Validation/hd95`: last 19.7776 (min 14.6133 @ ep 281)
  - `Validation/precision`: last 0.6049 (max 0.6420 @ ep 124)
  - `Validation/recall`: last 0.4618 (max 0.5199 @ ep 91)
  - `regional/dice`: last 0.1772 (max 0.2795 @ ep 5)
  - `regional/dice_large`: last 0.8008 (max 0.8526 @ ep 91)
  - `regional/dice_medium`: last 0.6717 (max 0.7550 @ ep 99)
  - `regional/dice_small`: last 0.2763 (max 0.4231 @ ep 51)
  - `regional/dice_tiny`: last 0.1514 (max 0.2641 @ ep 5)
  - `regional/iou`: last 0.1347 (max 0.2110 @ ep 5)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3825, max 5.8657 @ ep 103
  - `training/grad_norm_max`: last 4.6121, max 79.6305 @ ep 147

### exp2

**exp2 (SegResNet dual)** — dual-input (T1_pre + T1_gd) 2D and 3D baselines.

#### `exp2_dual_2d_20260226-003835`
*started 2026-02-26 00:38 • 501 epochs • 80h20m • 11098.3 TFLOPs • peak VRAM 7.4 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2718 (min 0.1744, max 0.3073)
  - `Validation/IoU`: last 0.2387 (min 0.1519, max 0.2807)
  - `Validation/hd95`: last 42.9017 (min 42.9017, max 77.6811)
  - `Validation/precision`: last 0.4756 (min 0.1564, max 0.6250)
  - `Validation/recall`: last 0.3135 (min 0.2100, max 0.5051)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4408
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0.7742
  - `regional/detection_rate_small`: 0.4644
  - `regional/detection_rate_tiny`: 0.4243
  - `regional/dice`: 0.2718
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6392
  - `regional/dice_small`: 0.3596
  - `regional/dice_tiny`: 0.2344
  - `regional/false_positives`: 608.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 62.0000
  - `regional/fp_tiny`: 546.0000
  - `regional/iou`: 0.2387
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5790
  - `regional/iou_small`: 0.3131
  - `regional/iou_tiny`: 0.2062

**Downstream seg metrics:**
  - `Validation/hd95`: last 42.9017
  - `Validation/precision`: last 0.4756 (max 0.6250 @ ep 154)
  - `Validation/recall`: last 0.3135 (max 0.5051 @ ep 13)
  - `regional/dice`: last 0.2718 (max 0.3073 @ ep 13)
  - `regional/dice_large`: last 0 (max 0.8296 @ ep 13)
  - `regional/dice_medium`: last 0.6392 (max 0.7395 @ ep 13)
  - `regional/dice_small`: last 0.3596 (max 0.4496 @ ep 6)
  - `regional/dice_tiny`: last 0.2344 (max 0.2420 @ ep 8)
  - `regional/iou`: last 0.2387 (max 0.2807 @ ep 13)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.1451, max 3.8096 @ ep 19
  - `training/grad_norm_max`: last 6.6501, max 49.0106 @ ep 243

#### `exp2_dual_3d_20260226-005507`
*started 2026-02-26 00:55 • 500 epochs • 11h39m • 34.4 TFLOPs • peak VRAM 63.5 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.1605 (min 0.0318, max 0.2565)
  - `Validation/IoU`: last 0.1225 (min 0.0318, max 0.1980)
  - `Validation/hd95`: last 27.7549 (min 19.8073, max 83.2587)
  - `Validation/precision`: last 0.5679 (min 0.0053, max 0.6218)
  - `Validation/recall`: last 0.4469 (min 1.598e-05, max 0.5134)

**Regional loss (final):**
  - `regional/detection_rate`: 0.3432
  - `regional/detection_rate_large`: 1.0000
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.5510
  - `regional/detection_rate_tiny`: 0.3012
  - `regional/dice`: 0.1605
  - `regional/dice_large`: 0.7844
  - `regional/dice_medium`: 0.6472
  - `regional/dice_small`: 0.2946
  - `regional/dice_tiny`: 0.1308
  - `regional/false_positives`: 180.0000
  - `regional/iou`: 0.1225
  - `regional/iou_large`: 0.6531
  - `regional/iou_medium`: 0.5203
  - `regional/iou_small`: 0.2157
  - `regional/iou_tiny`: 0.1000

**Downstream seg metrics:**
  - `Validation/hd95`: last 27.7549 (min 19.8073 @ ep 275)
  - `Validation/precision`: last 0.5679 (max 0.6218 @ ep 141)
  - `Validation/recall`: last 0.4469 (max 0.5134 @ ep 126)
  - `regional/dice`: last 0.1605 (max 0.2565 @ ep 12)
  - `regional/dice_large`: last 0.7844 (max 0.8534 @ ep 126)
  - `regional/dice_medium`: last 0.6472 (max 0.7131 @ ep 126)
  - `regional/dice_small`: last 0.2946 (max 0.4647 @ ep 55)
  - `regional/dice_tiny`: last 0.1308 (max 0.2286 @ ep 12)
  - `regional/iou`: last 0.1225 (max 0.1980 @ ep 12)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.3564, max 9.1588 @ ep 100
  - `training/grad_norm_max`: last 9.5820, max 182.1429 @ ep 227

### exp3

**exp3 (SegResNet + aug)** — real-only baseline with standard augmentation
pipeline (MONAI transforms).

#### `exp3_aug_2d_20260226-102203`
*started 2026-02-26 10:22 • 501 epochs • 54h22m • 11084.8 TFLOPs • peak VRAM 7.3 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2892 (min 0.2387, max 0.3208)
  - `Validation/IoU`: last 0.2585 (min 0.2024, max 0.2734)
  - `Validation/hd95`: last 42.3687 (min 42.3687, max 87.6070)
  - `Validation/precision`: last 0.5084 (min 0.0563, max 0.6267)
  - `Validation/recall`: last 0.3417 (min 0.2414, max 0.4828)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4441
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0.8710
  - `regional/detection_rate_small`: 0.4957
  - `regional/detection_rate_tiny`: 0.4163
  - `regional/dice`: 0.2892
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.7123
  - `regional/dice_small`: 0.3879
  - `regional/dice_tiny`: 0.2469
  - `regional/false_positives`: 581.0000
  - `regional/iou`: 0.2585
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.6441
  - `regional/iou_small`: 0.3400
  - `regional/iou_tiny`: 0.2226

**Downstream seg metrics:**
  - `Validation/hd95`: last 42.3687
  - `Validation/precision`: last 0.5084 (max 0.6267 @ ep 196)
  - `Validation/recall`: last 0.3417 (max 0.4828 @ ep 9)
  - `regional/dice`: last 0.2892 (max 0.3208 @ ep 2)
  - `regional/dice_large`: last 0 (max 0.8326 @ ep 65)
  - `regional/dice_medium`: last 0.7123 (max 0.7305 @ ep 9)
  - `regional/dice_small`: last 0.3879 (max 0.4144 @ ep 9)
  - `regional/dice_tiny`: last 0.2469 (max 0.2881 @ ep 2)
  - `regional/iou`: last 0.2585 (max 0.2734 @ ep 9)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.6775, max 4.0801 @ ep 29
  - `training/grad_norm_max`: last 9.1834, max 32.7583 @ ep 235

#### `exp3_aug_3d_20260226-110711`
*started 2026-02-26 11:07 • 500 epochs • 11h51m • 34.4 TFLOPs • peak VRAM 63.4 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.1836 (min 0.0323, max 0.2743)
  - `Validation/IoU`: last 0.1385 (min 0.0320, max 0.2113)
  - `Validation/hd95`: last 19.7484 (min 13.7579, max 70.5399)
  - `Validation/precision`: last 0.5738 (min 0.0056, max 0.6654)
  - `Validation/recall`: last 0.4831 (min 0.0019, max 0.5132)

**Regional loss (final):**
  - `regional/detection_rate`: 0.3895
  - `regional/detection_rate_large`: 1.0000
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.5918
  - `regional/detection_rate_tiny`: 0.3494
  - `regional/dice`: 0.1836
  - `regional/dice_large`: 0.8284
  - `regional/dice_medium`: 0.6973
  - `regional/dice_small`: 0.3018
  - `regional/dice_tiny`: 0.1551
  - `regional/false_positives`: 212.0000
  - `regional/iou`: 0.1385
  - `regional/iou_large`: 0.7147
  - `regional/iou_medium`: 0.5637
  - `regional/iou_small`: 0.2144
  - `regional/iou_tiny`: 0.1172

**Downstream seg metrics:**
  - `Validation/hd95`: last 19.7484 (min 13.7579 @ ep 134)
  - `Validation/precision`: last 0.5738 (max 0.6654 @ ep 100)
  - `Validation/recall`: last 0.4831 (max 0.5132 @ ep 196)
  - `regional/dice`: last 0.1836 (max 0.2743 @ ep 5)
  - `regional/dice_large`: last 0.8284 (max 0.8496 @ ep 250)
  - `regional/dice_medium`: last 0.6973 (max 0.7414 @ ep 112)
  - `regional/dice_small`: last 0.3018 (max 0.4196 @ ep 105)
  - `regional/dice_tiny`: last 0.1551 (max 0.2643 @ ep 5)
  - `regional/iou`: last 0.1385 (max 0.2113 @ ep 5)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.4585, max 10.1687 @ ep 114
  - `training/grad_norm_max`: last 11.3198, max 252.6839 @ ep 114

### exp4

**exp4 (SegResNet dual + aug)** — dual + augmentation.

#### `exp4_dual_aug_2d_20260226-123633`
*started 2026-02-26 12:36 • 501 epochs • 72h10m • 11098.3 TFLOPs • peak VRAM 7.5 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2780 (min 0.1898, max 0.3129)
  - `Validation/IoU`: last 0.2453 (min 0.1673, max 0.2832)
  - `Validation/hd95`: last 44.9444 (min 44.9444, max 72.5316)
  - `Validation/precision`: last 0.4570 (min 0.1870, max 0.6104)
  - `Validation/recall`: last 0.3136 (min 0.2449, max 0.4969)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4454
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0.7742
  - `regional/detection_rate_small`: 0.4843
  - `regional/detection_rate_tiny`: 0.4243
  - `regional/dice`: 0.2780
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6334
  - `regional/dice_small`: 0.3618
  - `regional/dice_tiny`: 0.2421
  - `regional/false_positives`: 699.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 58.0000
  - `regional/fp_tiny`: 641.0000
  - `regional/iou`: 0.2453
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5704
  - `regional/iou_small`: 0.3136
  - `regional/iou_tiny`: 0.2151

**Downstream seg metrics:**
  - `Validation/hd95`: last 44.9444
  - `Validation/precision`: last 0.4570 (max 0.6104 @ ep 243)
  - `Validation/recall`: last 0.3136 (max 0.4969 @ ep 37)
  - `regional/dice`: last 0.2780 (max 0.3129 @ ep 37)
  - `regional/dice_large`: last 0 (max 0.8107 @ ep 50)
  - `regional/dice_medium`: last 0.6334 (max 0.7325 @ ep 63)
  - `regional/dice_small`: last 0.3618 (max 0.4661 @ ep 37)
  - `regional/dice_tiny`: last 0.2421 (max 0.2471 @ ep 11)
  - `regional/iou`: last 0.2453 (max 0.2832 @ ep 37)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 3.2709, max 3.6162 @ ep 21
  - `training/grad_norm_max`: last 9.8960, max 34.4579 @ ep 135

#### `exp4_dual_aug_3d_20260226-123803`
*started 2026-02-26 12:38 • 501 epochs • 16h15m • 34.4 TFLOPs • peak VRAM 63.7 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2073 (min 0.0323, max 0.2467)
  - `Validation/IoU`: last 0.1602 (min 0.0320, max 0.1972)
  - `Validation/hd95`: last 17.3927 (min 17.3927, max 77.3029)
  - `Validation/precision`: last 0.4419 (min 0.0285, max 0.5986)
  - `Validation/recall`: last 0.3467 (min 0.0030, max 0.5171)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4390
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6154
  - `regional/detection_rate_tiny`: 0.4045
  - `regional/dice`: 0.2073
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6794
  - `regional/dice_small`: 0.3595
  - `regional/dice_tiny`: 0.1776
  - `regional/false_positives`: 160.0000
  - `regional/iou`: 0.1602
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5469
  - `regional/iou_small`: 0.2792
  - `regional/iou_tiny`: 0.1368

**Downstream seg metrics:**
  - `Validation/hd95`: last 17.3927
  - `Validation/precision`: last 0.4419 (max 0.5986 @ ep 184)
  - `Validation/recall`: last 0.3467 (max 0.5171 @ ep 53)
  - `regional/dice`: last 0.2073 (max 0.2467 @ ep 96)
  - `regional/dice_large`: last 0 (max 0.8589 @ ep 53)
  - `regional/dice_medium`: last 0.6794 (max 0.7175 @ ep 53)
  - `regional/dice_small`: last 0.3595 (max 0.4782 @ ep 96)
  - `regional/dice_tiny`: last 0.1776 (max 0.2091 @ ep 3)
  - `regional/iou`: last 0.1602 (max 0.1972 @ ep 3)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 3.4983, max 13.8048 @ ep 121
  - `training/grad_norm_max`: last 12.2217, max 250.4528 @ ep 318

### exp5

**exp5_syn105_3d** — trained on 105 purely-synthetic volumes from
exp1_1_1000 generation. Tests whether synthetic matches real at equal count.

#### `exp5_syn105_3d_20260228-172536`
*started 2026-02-28 17:25 • 501 epochs • 9h26m • 34.4 TFLOPs • peak VRAM 58.9 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.0774 (min 0.0419, max 0.1713)
  - `Validation/IoU`: last 0.0638 (min 0.0377, max 0.1274)
  - `Validation/hd95`: last 58.6474 (min 27.8094, max 70.3134)
  - `Validation/precision`: last 0.6026 (min 0.0247, max 0.6506)
  - `Validation/recall`: last 0.0856 (min 0.0176, max 0.3094)

**Regional loss (final):**
  - `regional/detection_rate`: 0.1843
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0.3333
  - `regional/detection_rate_small`: 0.3269
  - `regional/detection_rate_tiny`: 0.1592
  - `regional/dice`: 0.0774
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.0653
  - `regional/dice_small`: 0.1511
  - `regional/dice_tiny`: 0.0653
  - `regional/false_positives`: 74.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 0
  - `regional/fp_tiny`: 74.0000
  - `regional/iou`: 0.0638
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.0348
  - `regional/iou_small`: 0.1055
  - `regional/iou_tiny`: 0.0572

**Downstream seg metrics:**
  - `Validation/hd95`: last 58.6474 (min 27.8094 @ ep 87)
  - `Validation/precision`: last 0.6026 (max 0.6506 @ ep 169)
  - `Validation/recall`: last 0.0856 (max 0.3094 @ ep 18)
  - `regional/dice`: last 0.0774 (max 0.1713 @ ep 41)
  - `regional/dice_large`: last 0 (max 0.5597 @ ep 18)
  - `regional/dice_medium`: last 0.0653 (max 0.6088 @ ep 53)
  - `regional/dice_small`: last 0.1511 (max 0.2643 @ ep 53)
  - `regional/dice_tiny`: last 0.0653 (max 0.1523 @ ep 10)
  - `regional/iou`: last 0.0638 (max 0.1274 @ ep 41)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3675, max 4.7770 @ ep 77
  - `training/grad_norm_max`: last 2.9989, max 70.8631 @ ep 89

### exp6

**exp6 (synthetic-only)** — trained on ALL synthetic data (exp6_synall_3d)
and 100-epoch variant (exp6_1). Upper bound of synthetic-only downstream.

#### `exp6_1_synall_100ep_3d_20260228-173546`
*started 2026-02-28 17:35 • 101 epochs • 5h41m • 34.6 TFLOPs • peak VRAM 56.4 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.0401 (min 0.0329, max 0.1941)
  - `Validation/IoU`: last 0.0390 (min 0.0325, max 0.1445)
  - `Validation/hd95`: last 81.9833 (min 27.7744, max 86.2779)
  - `Validation/precision`: last 0.1264 (min 0.0305, max 0.6086)
  - `Validation/recall`: last 0.0024 (min 0.0017, max 0.3194)

**Regional loss (final):**
  - `regional/detection_rate`: 0.0921
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0
  - `regional/detection_rate_small`: 0.0385
  - `regional/detection_rate_tiny`: 0.1019
  - `regional/dice`: 0.0401
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.0002275
  - `regional/dice_small`: 0.0125
  - `regional/dice_tiny`: 0.0451
  - `regional/false_positives`: 41.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 0
  - `regional/fp_tiny`: 41.0000
  - `regional/iou`: 0.0390
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.0002275
  - `regional/iou_small`: 0.0081
  - `regional/iou_tiny`: 0.0445

**Downstream seg metrics:**
  - `Validation/hd95`: last 81.9833 (min 27.7744 @ ep 22)
  - `Validation/precision`: last 0.1264 (max 0.6086 @ ep 32)
  - `Validation/recall`: last 0.0024 (max 0.3194 @ ep 8)
  - `regional/dice`: last 0.0401 (max 0.1941 @ ep 3)
  - `regional/dice_large`: last 0 (max 0.5923 @ ep 8)
  - `regional/dice_medium`: last 0.0002275 (max 0.5911 @ ep 19)
  - `regional/dice_small`: last 0.0125 (max 0.2806 @ ep 3)
  - `regional/dice_tiny`: last 0.0451 (max 0.1740 @ ep 3)
  - `regional/iou`: last 0.0390 (max 0.1445 @ ep 3)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.3351, max 4.5469 @ ep 33
  - `training/grad_norm_max`: last 8.1123, max 84.2691 @ ep 20

#### `exp6_synall_3d_20260228-173546`
*started 2026-02-28 17:35 • 501 epochs • 36h08m • 173.1 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.0387 (min 0.0325, max 0.1952)
  - `Validation/IoU`: last 0.0382 (min 0.0322, max 0.1443)
  - `Validation/hd95`: last 90.6604 (min 22.6110, max 95.9263)
  - `Validation/precision`: last 0.0861 (min 0.0239, max 0.5722)
  - `Validation/recall`: last 0.0013 (min 0.0012, max 0.3119)

**Regional loss (final):**
  - `regional/detection_rate`: 0.0840
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 0
  - `regional/detection_rate_small`: 0.0192
  - `regional/detection_rate_tiny`: 0.0955
  - `regional/dice`: 0.0387
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.0002275
  - `regional/dice_small`: 0.0048
  - `regional/dice_tiny`: 0.0447
  - `regional/false_positives`: 47.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 1.0000
  - `regional/fp_tiny`: 46.0000
  - `regional/iou`: 0.0382
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.0002275
  - `regional/iou_small`: 0.0036
  - `regional/iou_tiny`: 0.0443

**Downstream seg metrics:**
  - `Validation/hd95`: last 90.6604 (min 22.6110 @ ep 21)
  - `Validation/precision`: last 0.0861 (max 0.5722 @ ep 33)
  - `Validation/recall`: last 0.0013 (max 0.3119 @ ep 8)
  - `regional/dice`: last 0.0387 (max 0.1952 @ ep 1)
  - `regional/dice_large`: last 0 (max 0.5638 @ ep 9)
  - `regional/dice_medium`: last 0.0002275 (max 0.5883 @ ep 29)
  - `regional/dice_small`: last 0.0048 (max 0.3058 @ ep 8)
  - `regional/dice_tiny`: last 0.0447 (max 0.1779 @ ep 1)
  - `regional/iou`: last 0.0382 (max 0.1443 @ ep 1)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.0132, max 4.1165 @ ep 24
  - `training/grad_norm_max`: last 2.4950, max 94.7111 @ ep 24

### exp7

**exp7 (real + synthetic mix)** — sweep of mix ratios: 25 / 50 / 75 / 105 /
210 / 315 synthetic volumes added to 105 real. Identifies optimal data-mix
for best downstream Dice.

#### `exp7_mixed_3d_20260228-175437`
*started 2026-02-28 17:54 • 501 epochs • 14h08m • 69.4 TFLOPs • peak VRAM 59.3 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2061 (min 0.0897, max 0.2517)
  - `Validation/IoU`: last 0.1588 (min 0.0690, max 0.1905)
  - `Validation/hd95`: last 20.8633 (min 14.6521, max 69.8351)
  - `Validation/precision`: last 0.4799 (min 0.0260, max 0.6331)
  - `Validation/recall`: last 0.3535 (min 0.0984, max 0.5072)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4390
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6731
  - `regional/detection_rate_tiny`: 0.3949
  - `regional/dice`: 0.2061
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6901
  - `regional/dice_small`: 0.3886
  - `regional/dice_tiny`: 0.1712
  - `regional/false_positives`: 184.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 0
  - `regional/fp_tiny`: 184.0000
  - `regional/iou`: 0.1588
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5426
  - `regional/iou_small`: 0.2996
  - `regional/iou_tiny`: 0.1318

**Downstream seg metrics:**
  - `Validation/hd95`: last 20.8633 (min 14.6521 @ ep 151)
  - `Validation/precision`: last 0.4799 (max 0.6331 @ ep 230)
  - `Validation/recall`: last 0.3535 (max 0.5072 @ ep 128)
  - `regional/dice`: last 0.2061 (max 0.2517 @ ep 6)
  - `regional/dice_large`: last 0 (max 0.8480 @ ep 128)
  - `regional/dice_medium`: last 0.6901 (max 0.7320 @ ep 47)
  - `regional/dice_small`: last 0.3886
  - `regional/dice_tiny`: last 0.1712 (max 0.2292 @ ep 6)
  - `regional/iou`: last 0.1588 (max 0.1905 @ ep 6)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3690, max 5.9528 @ ep 72
  - `training/grad_norm_max`: last 4.5149, max 149.1503 @ ep 184

#### `exp7_1_mix105syn_3d_20260228-203554`
*started 2026-02-28 20:35 • 501 epochs • 21h03m • 69.4 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2055 (min 0.0805, max 0.2432)
  - `Validation/IoU`: last 0.1582 (min 0.0627, max 0.1791)
  - `Validation/hd95`: last 22.6215 (min 14.9328, max 70.0119)
  - `Validation/precision`: last 0.4856 (min 0.0240, max 0.6348)
  - `Validation/recall`: last 0.3628 (min 0.1108, max 0.5048)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4363
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6538
  - `regional/detection_rate_tiny`: 0.3949
  - `regional/dice`: 0.2055
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.7194
  - `regional/dice_small`: 0.3917
  - `regional/dice_tiny`: 0.1698
  - `regional/false_positives`: 181.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 0
  - `regional/fp_tiny`: 181.0000
  - `regional/iou`: 0.1582
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5800
  - `regional/iou_small`: 0.3009
  - `regional/iou_tiny`: 0.1305

**Downstream seg metrics:**
  - `Validation/hd95`: last 22.6215 (min 14.9328 @ ep 126)
  - `Validation/precision`: last 0.4856 (max 0.6348 @ ep 68)
  - `Validation/recall`: last 0.3628 (max 0.5048 @ ep 136)
  - `regional/dice`: last 0.2055 (max 0.2432 @ ep 7)
  - `regional/dice_large`: last 0 (max 0.8375 @ ep 80)
  - `regional/dice_medium`: last 0.7194 (max 0.7211 @ ep 210)
  - `regional/dice_small`: last 0.3917 (max 0.3954 @ ep 57)
  - `regional/dice_tiny`: last 0.1698 (max 0.2258 @ ep 7)
  - `regional/iou`: last 0.1582 (max 0.1791 @ ep 7)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3430, max 5.7278 @ ep 63
  - `training/grad_norm_max`: last 6.2022, max 211.3565 @ ep 95

#### `exp7_2_mix210syn_3d_20260228-232359`
*started 2026-02-28 23:23 • 501 epochs • 21h23m • 103.7 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2025 (min 0.0764, max 0.2188)
  - `Validation/IoU`: last 0.1558 (min 0.0605, max 0.1685)
  - `Validation/hd95`: last 19.0601 (min 16.2401, max 69.9722)
  - `Validation/precision`: last 0.4921 (min 0.0300, max 0.6357)
  - `Validation/recall`: last 0.3490 (min 0.1339, max 0.5205)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4363
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6346
  - `regional/detection_rate_tiny`: 0.3981
  - `regional/dice`: 0.2025
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6802
  - `regional/dice_small`: 0.3806
  - `regional/dice_tiny`: 0.1684
  - `regional/false_positives`: 189.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 2.0000
  - `regional/fp_tiny`: 187.0000
  - `regional/iou`: 0.1558
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5390
  - `regional/iou_small`: 0.2943
  - `regional/iou_tiny`: 0.1291

**Downstream seg metrics:**
  - `Validation/hd95`: last 19.0601 (min 16.2401 @ ep 177)
  - `Validation/precision`: last 0.4921 (max 0.6357 @ ep 51)
  - `Validation/recall`: last 0.3490 (max 0.5205 @ ep 61)
  - `regional/dice`: last 0.2025 (max 0.2188 @ ep 124)
  - `regional/dice_large`: last 0 (max 0.8521 @ ep 61)
  - `regional/dice_medium`: last 0.6802 (max 0.7380 @ ep 61)
  - `regional/dice_small`: last 0.3806 (max 0.3855 @ ep 124)
  - `regional/dice_tiny`: last 0.1684 (max 0.1946 @ ep 4)
  - `regional/iou`: last 0.1558 (max 0.1685 @ ep 124)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.4107, max 6.7520 @ ep 62
  - `training/grad_norm_max`: last 5.9795, max 288.7251 @ ep 171

#### `exp7_3_mix315syn_3d_20260301-005001`
*started 2026-03-01 00:50 • 501 epochs • 28h22m • 138.7 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2017 (min 0.1204, max 0.2883)
  - `Validation/IoU`: last 0.1547 (min 0.0948, max 0.2199)
  - `Validation/hd95`: last 22.7552 (min 14.4298, max 70.4787)
  - `Validation/precision`: last 0.4926 (min 0.0306, max 0.6482)
  - `Validation/recall`: last 0.3349 (min 0.1723, max 0.4975)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4417
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.5962
  - `regional/detection_rate_tiny`: 0.4108
  - `regional/dice`: 0.2017
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6609
  - `regional/dice_small`: 0.3537
  - `regional/dice_tiny`: 0.1722
  - `regional/false_positives`: 187.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 1.0000
  - `regional/fp_tiny`: 186.0000
  - `regional/iou`: 0.1547
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5225
  - `regional/iou_small`: 0.2759
  - `regional/iou_tiny`: 0.1310

**Downstream seg metrics:**
  - `Validation/hd95`: last 22.7552 (min 14.4298 @ ep 149)
  - `Validation/precision`: last 0.4926 (max 0.6482 @ ep 118)
  - `Validation/recall`: last 0.3349 (max 0.4975 @ ep 58)
  - `regional/dice`: last 0.2017 (max 0.2883 @ ep 6)
  - `regional/dice_large`: last 0 (max 0.8539 @ ep 58)
  - `regional/dice_medium`: last 0.6609 (max 0.7152 @ ep 104)
  - `regional/dice_small`: last 0.3537 (max 0.3982 @ ep 6)
  - `regional/dice_tiny`: last 0.1722 (max 0.2657 @ ep 6)
  - `regional/iou`: last 0.1547 (max 0.2199 @ ep 6)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3271, max 9.3100 @ ep 152
  - `training/grad_norm_max`: last 5.2657, max 1420.4778 @ ep 152

#### `exp7_4_mix25syn_3d_20260302-013912`
*started 2026-03-02 01:39 • 501 epochs • 18h23m • 42.9 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2018 (min 0.0515, max 0.2603)
  - `Validation/IoU`: last 0.1551 (min 0.0438, max 0.1982)
  - `Validation/hd95`: last 21.4149 (min 14.1540, max 70.3368)
  - `Validation/precision`: last 0.4750 (min 0.0238, max 0.6487)
  - `Validation/recall`: last 0.3491 (min 0.0798, max 0.4970)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4336
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6346
  - `regional/detection_rate_tiny`: 0.3949
  - `regional/dice`: 0.2018
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.7123
  - `regional/dice_small`: 0.3751
  - `regional/dice_tiny`: 0.1682
  - `regional/false_positives`: 184.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 2.0000
  - `regional/fp_tiny`: 182.0000
  - `regional/iou`: 0.1551
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5721
  - `regional/iou_small`: 0.2907
  - `regional/iou_tiny`: 0.1287

**Downstream seg metrics:**
  - `Validation/hd95`: last 21.4149 (min 14.1540 @ ep 278)
  - `Validation/precision`: last 0.4750 (max 0.6487 @ ep 59)
  - `Validation/recall`: last 0.3491 (max 0.4970 @ ep 135)
  - `regional/dice`: last 0.2018 (max 0.2603 @ ep 21)
  - `regional/dice_large`: last 0 (max 0.8393 @ ep 130)
  - `regional/dice_medium`: last 0.7123 (max 0.7355 @ ep 135)
  - `regional/dice_small`: last 0.3751 (max 0.3812 @ ep 54)
  - `regional/dice_tiny`: last 0.1682 (max 0.2372 @ ep 21)
  - `regional/iou`: last 0.1551 (max 0.1982 @ ep 21)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.2977, max 7.1641 @ ep 67
  - `training/grad_norm_max`: last 4.1411, max 129.2387 @ ep 94

#### `exp7_5_mix50syn_3d_20260302-013912`
*started 2026-03-02 01:39 • 501 epochs • 19h02m • 50.9 TFLOPs • peak VRAM 63.6 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2041 (min 0.0566, max 0.2169)
  - `Validation/IoU`: last 0.1568 (min 0.0472, max 0.1659)
  - `Validation/hd95`: last 18.8476 (min 16.0224, max 70.5046)
  - `Validation/precision`: last 0.4776 (min 0.0186, max 0.6312)
  - `Validation/recall`: last 0.3509 (min 0.0497, max 0.5092)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4363
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6346
  - `regional/detection_rate_tiny`: 0.3981
  - `regional/dice`: 0.2041
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.7042
  - `regional/dice_small`: 0.3865
  - `regional/dice_tiny`: 0.1691
  - `regional/false_positives`: 183.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 2.0000
  - `regional/fp_tiny`: 181.0000
  - `regional/iou`: 0.1568
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5602
  - `regional/iou_small`: 0.2996
  - `regional/iou_tiny`: 0.1292

**Downstream seg metrics:**
  - `Validation/hd95`: last 18.8476 (min 16.0224 @ ep 216)
  - `Validation/precision`: last 0.4776 (max 0.6312 @ ep 83)
  - `Validation/recall`: last 0.3509 (max 0.5092 @ ep 80)
  - `regional/dice`: last 0.2041 (max 0.2169 @ ep 105)
  - `regional/dice_large`: last 0 (max 0.8308 @ ep 80)
  - `regional/dice_medium`: last 0.7042 (max 0.7523 @ ep 80)
  - `regional/dice_small`: last 0.3865
  - `regional/dice_tiny`: last 0.1691 (max 0.1871 @ ep 5)
  - `regional/iou`: last 0.1568 (max 0.1659 @ ep 105)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.2001, max 6.4802 @ ep 103
  - `training/grad_norm_max`: last 3.3489, max 182.0441 @ ep 103

#### `exp7_6_mix75syn_3d_20260302-013918`
*started 2026-03-02 01:39 • 501 epochs • 19h53m • 59.5 TFLOPs • peak VRAM 63.4 GB*

**Validation quality:**
  - `Validation/Dice`: last 0.2063 (min 0.0759, max 0.2284)
  - `Validation/IoU`: last 0.1594 (min 0.0595, max 0.1760)
  - `Validation/hd95`: last 20.5135 (min 14.4278, max 70.4759)
  - `Validation/precision`: last 0.4819 (min 0.0301, max 0.6351)
  - `Validation/recall`: last 0.3517 (min 0.1055, max 0.4925)

**Regional loss (final):**
  - `regional/detection_rate`: 0.4363
  - `regional/detection_rate_large`: 0
  - `regional/detection_rate_medium`: 1.0000
  - `regional/detection_rate_small`: 0.6538
  - `regional/detection_rate_tiny`: 0.3949
  - `regional/dice`: 0.2063
  - `regional/dice_large`: 0
  - `regional/dice_medium`: 0.6940
  - `regional/dice_small`: 0.3882
  - `regional/dice_tiny`: 0.1715
  - `regional/false_positives`: 174.0000
  - `regional/fp_large`: 0
  - `regional/fp_medium`: 0
  - `regional/fp_small`: 0
  - `regional/fp_tiny`: 174.0000
  - `regional/iou`: 0.1594
  - `regional/iou_large`: 0
  - `regional/iou_medium`: 0.5491
  - `regional/iou_small`: 0.3014
  - `regional/iou_tiny`: 0.1321

**Downstream seg metrics:**
  - `Validation/hd95`: last 20.5135 (min 14.4278 @ ep 109)
  - `Validation/precision`: last 0.4819 (max 0.6351 @ ep 95)
  - `Validation/recall`: last 0.3517 (max 0.4925 @ ep 178)
  - `regional/dice`: last 0.2063 (max 0.2284 @ ep 68)
  - `regional/dice_large`: last 0 (max 0.8223 @ ep 219)
  - `regional/dice_medium`: last 0.6940 (max 0.7195 @ ep 97)
  - `regional/dice_small`: last 0.3882
  - `regional/dice_tiny`: last 0.1715 (max 0.2017 @ ep 2)
  - `regional/iou`: last 0.1594 (max 0.1760 @ ep 68)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.3238, max 6.5733 @ ep 103
  - `training/grad_norm_max`: last 5.4796, max 169.9493 @ ep 103

---
## downstream/nnunet

*13 runs across 4 experiment families.*

### exp3

**exp3_baseline (nnU-Net, real only)** — nnU-Net ResEnc-L 3D fullres on 105
real volumes, 5-fold CV. The canonical real-only downstream baseline.
nnU-Net writes per-fold TB events under `Dataset501_BrainMet/.../fold_N/tensorboard/`
which our extractor merges with `fold_N::` prefix.

#### `exp3_baseline`
*started ? • 1000 epochs • 49h38m*

**Downstream seg metrics:**
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice`: last 0.1957
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_large`: last 0.8047
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_medium`: last 0.7311
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_small`: last 0.3509
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_tiny`: last 0.1636
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/hd95`: last 22.1812
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/iou`: last 0.1513
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/precision`: last 0.5323
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/recall`: last 0.4375
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/loss`: last -0.8328 (min -0.8443 @ ep 986)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/dice_class_0`: last 0.6465 (max 0.7758 @ ep 322)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/ema_fg_dice`: last 0.6364 (max 0.6814 @ ep 146)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/loss`: last -0.6305 (min -0.7109 @ ep 177)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/mean_fg_dice`: last 0.6465 (max 0.7758 @ ep 322)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice`: last 0.1873
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_large`: last 0.7936
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_medium`: last 0.6939
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_small`: last 0.3389
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_tiny`: last 0.1562
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/hd95`: last 18.9954
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/iou`: last 0.1431
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/precision`: last 0.5498
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/recall`: last 0.4221
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/loss`: last -0.8511 (min -0.8537 @ ep 959)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/dice_class_0`: last 0.7334 (max 0.8237 @ ep 967)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/ema_fg_dice`: last 0.7664 (max 0.7779 @ ep 974)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/loss`: last -0.5851 (min -0.7055 @ ep 967)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/mean_fg_dice`: last 0.7334 (max 0.8237 @ ep 967)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice`: last 0.1911
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_large`: last 0.8076
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_medium`: last 0.6935
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_small`: last 0.3369
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_tiny`: last 0.1608
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/hd95`: last 18.0381
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/iou`: last 0.1458
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/precision`: last 0.5544
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/recall`: last 0.4251
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/loss`: last -0.8474 (min -0.8532 @ ep 971)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/dice_class_0`: last 0.8113 (max 0.8509 @ ep 678)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/ema_fg_dice`: last 0.7962 (max 0.8007 @ ep 997)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/loss`: last -0.7242 (min -0.7300 @ ep 960)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/mean_fg_dice`: last 0.8113 (max 0.8509 @ ep 678)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice`: last 0.1960
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_large`: last 0.8208
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_medium`: last 0.7116
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_small`: last 0.3572
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_tiny`: last 0.1633
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/hd95`: last 20.6236
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/iou`: last 0.1506
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/precision`: last 0.5381
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/recall`: last 0.4367
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/loss`: last -0.8585 (min -0.8617 @ ep 998)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/dice_class_0`: last 0.7166 (max 0.8226 @ ep 460)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/ema_fg_dice`: last 0.6979 (max 0.7286 @ ep 226)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/loss`: last -0.5551 (min -0.6346 @ ep 839)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/mean_fg_dice`: last 0.7166 (max 0.8226 @ ep 460)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice`: last 0.1808
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_large`: last 0.8130
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_medium`: last 0.6910
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_small`: last 0.3302
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_tiny`: last 0.1498
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/hd95`: last 17.3025
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/iou`: last 0.1376
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/precision`: last 0.5518
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/recall`: last 0.4228
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/loss`: last -0.8473 (min -0.8557 @ ep 981)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/dice_class_0`: last 0.8179 (max 0.8278 @ ep 364)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/ema_fg_dice`: last 0.7731 (max 0.7902 @ ep 683)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/loss`: last -0.6828 (min -0.6962 @ ep 194)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/mean_fg_dice`: last 0.8179 (max 0.8278 @ ep 364)
  - `eval_exp3_baseline/tensorboard::test/dice`: last 0.1880
  - `eval_exp3_baseline/tensorboard::test/dice_large`: last 0.8089
  - `eval_exp3_baseline/tensorboard::test/dice_medium`: last 0.7022
  - `eval_exp3_baseline/tensorboard::test/dice_small`: last 0.3386
  - `eval_exp3_baseline/tensorboard::test/dice_tiny`: last 0.1568
  - `eval_exp3_baseline/tensorboard::test/hd95`: last 18.8700
  - `eval_exp3_baseline/tensorboard::test/iou`: last 0.1439
  - `eval_exp3_baseline/tensorboard::test/precision`: last 0.5549
  - `eval_exp3_baseline/tensorboard::test/recall`: last 0.4282

### exp4

**exp4_baseline_dual (nnU-Net, dual input)** — T1_pre + T1_gd input as
separate channels. Note `Dataset502_BrainMet` (distinct from 501).

#### `exp4_baseline_dual`
*started ? • 1000 epochs • 48h55m*

**Downstream seg metrics:**
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice`: last 0.1658
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_large`: last 0.5539
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_medium`: last 0.5625
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_small`: last 0.2938
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_tiny`: last 0.1404
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/hd95`: last 23.1819
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/iou`: last 0.1271
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/precision`: last 0.5709
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/recall`: last 0.3431
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/loss`: last -0.8566 (min -0.8613 @ ep 983)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/dice_class_0`: last 0.4857 (max 0.6478 @ ep 495)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/ema_fg_dice`: last 0.4703 (max 0.5208 @ ep 520)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/loss`: last -0.3829 (min -0.5512 @ ep 114)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/mean_fg_dice`: last 0.4857 (max 0.6478 @ ep 495)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice`: last 0.1755
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_large`: last 0.7821
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_medium`: last 0.6463
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_small`: last 0.3386
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_tiny`: last 0.1433
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/hd95`: last 21.9763
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/iou`: last 0.1351
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/precision`: last 0.5647
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/recall`: last 0.4061
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/loss`: last -0.8659 (min -0.8719 @ ep 985)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/dice_class_0`: last 0.6737 (max 0.7593 @ ep 539)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/ema_fg_dice`: last 0.6789 (max 0.7035 @ ep 315)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/loss`: last -0.4997 (min -0.5873 @ ep 171)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/mean_fg_dice`: last 0.6737 (max 0.7593 @ ep 539)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice`: last 0.1513
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_large`: last 0.7469
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_medium`: last 0.6010
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_small`: last 0.2877
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_tiny`: last 0.1232
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/hd95`: last 25.6785
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/iou`: last 0.1169
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/precision`: last 0.6018
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/recall`: last 0.3729
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/loss`: last -0.8674 (min -0.8740 @ ep 985)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/dice_class_0`: last 0.6642 (max 0.7953 @ ep 92)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/ema_fg_dice`: last 0.6811 (max 0.7154 @ ep 147)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/loss`: last -0.4908 (min -0.6028 @ ep 504)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/mean_fg_dice`: last 0.6642 (max 0.7953 @ ep 92)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice`: last 0.1635
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_large`: last 0.7776
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_medium`: last 0.6132
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_small`: last 0.3095
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_tiny`: last 0.1339
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/hd95`: last 26.4987
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/iou`: last 0.1265
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/precision`: last 0.5766
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/recall`: last 0.3919
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/loss`: last -0.8705 (min -0.8763 @ ep 957)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/dice_class_0`: last 0.6260 (max 0.7552 @ ep 415)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/ema_fg_dice`: last 0.5853 (max 0.6342 @ ep 217)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/loss`: last -0.5325 (min -0.5745 @ ep 481)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/mean_fg_dice`: last 0.6260 (max 0.7552 @ ep 415)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice`: last 0.1526
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_large`: last 0.7115
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_medium`: last 0.6177
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_small`: last 0.2920
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_tiny`: last 0.1240
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/hd95`: last 28.2134
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/iou`: last 0.1178
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/precision`: last 0.5893
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/recall`: last 0.3790
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/loss`: last -0.8709
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/dice_class_0`: last 0.7462 (max 0.8027 @ ep 375)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/ema_fg_dice`: last 0.7293 (max 0.7526 @ ep 563)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/loss`: last -0.5258 (min -0.6374 @ ep 549)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/mean_fg_dice`: last 0.7462 (max 0.8027 @ ep 375)

### exp6

**exp6 (nnU-Net synthetic-only)** — ablations of synthetic-only training.
Variants: `exp6_synthetic`, `exp6_synthetic_105`, `exp6_1_synthetic_105_imagenet`,
`exp6_1_synthetic_105_radimagenet` (last two split by feature-extractor used
in generation-quality filtering).

#### `exp6_1_synthetic_105_imagenet`
*started ? • 1000 epochs • 42h21m*

**Downstream seg metrics:**
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice`: last 0.1612
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_large`: last 0.6771
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_medium`: last 0.6009
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_small`: last 0.2812
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_tiny`: last 0.1358
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/hd95`: last 29.8123
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/iou`: last 0.1221
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/precision`: last 0.5609
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/recall`: last 0.3350
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/loss`: last -0.8641 (min -0.8687 @ ep 987)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/dice_class_0`: last 0.5288 (max 0.6644 @ ep 383)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/ema_fg_dice`: last 0.4139 (max 0.4889 @ ep 91)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/loss`: last -0.4959 (min -0.5802 @ ep 91)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/mean_fg_dice`: last 0.5288 (max 0.6644 @ ep 383)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice`: last 0.1400
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_large`: last 0.6409
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_medium`: last 0.5453
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_small`: last 0.2407
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_tiny`: last 0.1179
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/hd95`: last 27.0481
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/iou`: last 0.1062
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/precision`: last 0.5998
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/recall`: last 0.2940
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/loss`: last -0.8756
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/dice_class_0`: last 0.5524 (max 0.6941 @ ep 137)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/ema_fg_dice`: last 0.5733 (max 0.5899 @ ep 833)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/loss`: last -0.4840 (min -0.5837 @ ep 195)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/mean_fg_dice`: last 0.5524 (max 0.6941 @ ep 137)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice`: last 0.1623
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_large`: last 0.6282
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_medium`: last 0.5729
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_small`: last 0.2727
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_tiny`: last 0.1389
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/hd95`: last 29.5419
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/iou`: last 0.1230
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/precision`: last 0.5559
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/recall`: last 0.3116
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/loss`: last -0.8671 (min -0.8748 @ ep 998)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/dice_class_0`: last 0.6522 (max 0.7464 @ ep 594)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/ema_fg_dice`: last 0.6460 (max 0.6962 @ ep 114)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/loss`: last -0.5999 (min -0.6404 @ ep 594)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/mean_fg_dice`: last 0.6522 (max 0.7464 @ ep 594)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice`: last 0.1624
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_large`: last 0.5107
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_medium`: last 0.5800
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_small`: last 0.2658
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_tiny`: last 0.1403
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/hd95`: last 35.5087
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/iou`: last 0.1221
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/precision`: last 0.5289
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/recall`: last 0.2793
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/loss`: last -0.8708 (min -0.8712 @ ep 968)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/dice_class_0`: last 0.6063 (max 0.7625 @ ep 320)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/ema_fg_dice`: last 0.6395 (max 0.6842 @ ep 306)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/loss`: last -0.4967 (min -0.5783 @ ep 320)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/mean_fg_dice`: last 0.6063 (max 0.7625 @ ep 320)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice`: last 0.1568
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_large`: last 0.6788
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_medium`: last 0.5661
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_small`: last 0.2628
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_tiny`: last 0.1337
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/hd95`: last 28.8319
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/iou`: last 0.1183
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/precision`: last 0.5786
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/recall`: last 0.3193
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/loss`: last -0.8706
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/dice_class_0`: last 0.7316 (max 0.7671 @ ep 85)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/ema_fg_dice`: last 0.7145 (max 0.7190 @ ep 939)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/loss`: last -0.5208 (min -0.6020 @ ep 85)
  - `Dataset501_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/mean_fg_dice`: last 0.7316 (max 0.7671 @ ep 85)
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/dice`: last 0.1533
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/dice_large`: last 0.6245
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/dice_medium`: last 0.5740
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/dice_small`: last 0.2612
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/dice_tiny`: last 0.1300
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/hd95`: last 29.7884
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/iou`: last 0.1159
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/precision`: last 0.5747
  - `eval_exp6_1_synthetic_105_imagenet/tensorboard::test/recall`: last 0.3046

#### `exp6_1_synthetic_105_radimagenet`
*started ? • 1000 epochs • 30h47m*

**Downstream seg metrics:**
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice`: last 0.1468
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_large`: last 0.6830
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_medium`: last 0.5914
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_small`: last 0.2611
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/dice_tiny`: last 0.1221
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/hd95`: last 32.9677
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/iou`: last 0.1119
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/precision`: last 0.5782
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::test/recall`: last 0.3316
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::train/loss`: last -0.8678 (min -0.8736 @ ep 988)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/dice_class_0`: last 0.4395 (max 0.7150 @ ep 201)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/ema_fg_dice`: last 0.4039 (max 0.4831 @ ep 201)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/loss`: last -0.4189 (min -0.5844 @ ep 201)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_0/tensorboard::val/mean_fg_dice`: last 0.4395 (max 0.7150 @ ep 201)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice`: last 0.1791
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_large`: last 0.6252
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_medium`: last 0.6032
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_small`: last 0.3061
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/dice_tiny`: last 0.1533
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/hd95`: last 38.5850
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/iou`: last 0.1368
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/precision`: last 0.5250
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::test/recall`: last 0.3316
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::train/loss`: last -0.8683 (min -0.8760 @ ep 994)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/dice_class_0`: last 0.5812 (max 0.6838 @ ep 393)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/ema_fg_dice`: last 0.5878 (max 0.6010 @ ep 53)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/loss`: last -0.4771 (min -0.5651 @ ep 49)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_1/tensorboard::val/mean_fg_dice`: last 0.5812 (max 0.6838 @ ep 393)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice`: last 0.1751
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_large`: last 0.6712
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_medium`: last 0.5901
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_small`: last 0.3035
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/dice_tiny`: last 0.1489
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/hd95`: last 46.8525
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/iou`: last 0.1332
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/precision`: last 0.5044
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::test/recall`: last 0.3344
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::train/loss`: last -0.8703 (min -0.8733 @ ep 988)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/dice_class_0`: last 0.6675 (max 0.7613 @ ep 248)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/ema_fg_dice`: last 0.6555 (max 0.6912 @ ep 149)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/loss`: last -0.5743 (min -0.6207 @ ep 141)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_2/tensorboard::val/mean_fg_dice`: last 0.6675 (max 0.7613 @ ep 248)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice`: last 0.1503
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_large`: last 0.6797
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_medium`: last 0.5428
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_small`: last 0.2632
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/dice_tiny`: last 0.1265
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/hd95`: last 33.0676
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/iou`: last 0.1154
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/precision`: last 0.5946
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::test/recall`: last 0.3198
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::train/loss`: last -0.8550 (min -0.8700 @ ep 997)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/dice_class_0`: last 0.5893 (max 0.7446 @ ep 299)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/ema_fg_dice`: last 0.5634 (max 0.6610 @ ep 306)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/loss`: last -0.4717 (min -0.5504 @ ep 299)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_3/tensorboard::val/mean_fg_dice`: last 0.5893 (max 0.7446 @ ep 299)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice`: last 0.1302
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_large`: last 0.6186
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_medium`: last 0.4925
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_small`: last 0.2221
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/dice_tiny`: last 0.1100
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/hd95`: last 33.5062
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/iou`: last 0.0994
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/precision`: last 0.6206
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::test/recall`: last 0.2736
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/learning_rate`: last 1.995e-05
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::train/loss`: last -0.8626 (min -0.8677 @ ep 991)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/dice_class_0`: last 0.6581 (max 0.7359 @ ep 136)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/ema_fg_dice`: last 0.6697 (max 0.6865 @ ep 844)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/loss`: last -0.3921 (min -0.5685 @ ep 35)
  - `Dataset502_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/fold_4/tensorboard::val/mean_fg_dice`: last 0.6581 (max 0.7359 @ ep 136)
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/dice`: last 0.1524
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/dice_large`: last 0.6639
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/dice_medium`: last 0.5754
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/dice_small`: last 0.2669
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/dice_tiny`: last 0.1280
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/hd95`: last 27.0872
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/iou`: last 0.1163
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/precision`: last 0.5839 (max 0.5839 @ ep 0)
  - `eval_exp6_1_synthetic_105_radimagenet/tensorboard::test/recall`: last 0.3221

#### `exp6_synthetic`
*started ? • 1000 epochs • 36h23m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1511
  - `test/dice_large`: last 0.5145
  - `test/dice_medium`: last 0.5883
  - `test/dice_small`: last 0.2613
  - `test/dice_tiny`: last 0.1278
  - `test/hd95`: last 26.0261
  - `test/iou`: last 0.1153
  - `test/precision`: last 0.5521
  - `test/recall`: last 0.2855
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.5539 (min -0.6238 @ ep 970)
  - `val/dice_class_0`: last 0.3258 (max 0.6256 @ ep 79)
  - `val/ema_fg_dice`: last 0.3706 (max 0.5068 @ ep 180)
  - `val/loss`: last -0.3369 (min -0.5843 @ ep 155)
  - `val/mean_fg_dice`: last 0.3258 (max 0.6256 @ ep 79)

#### `exp6_synthetic_105`
*started ? • 1000 epochs • 30h44m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1935
  - `test/dice_large`: last 0.7688
  - `test/dice_medium`: last 0.7001
  - `test/dice_small`: last 0.3411
  - `test/dice_tiny`: last 0.1630
  - `test/hd95`: last 23.2820
  - `test/iou`: last 0.1480
  - `test/precision`: last 0.5392
  - `test/recall`: last 0.4127
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.7205 (min -0.7348 @ ep 984)
  - `val/dice_class_0`: last 0.7510 (max 0.8103 @ ep 680)
  - `val/ema_fg_dice`: last 0.7121
  - `val/loss`: last -0.6642 (min -0.7152 @ ep 821)
  - `val/mean_fg_dice`: last 0.7510 (max 0.8103 @ ep 680)

### exp7

**exp7 (nnU-Net real + synthetic)** — mixed training with 25/50/75/105/210/
315/525 synthetic volumes added to 105 real. Parallel sweep to SegResNet exp7.

#### `exp7_mixed_105syn`
*started ? • 1000 epochs • 25h39m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1912
  - `test/dice_large`: last 0.7613
  - `test/dice_medium`: last 0.7064
  - `test/dice_small`: last 0.3471
  - `test/dice_tiny`: last 0.1595
  - `test/hd95`: last 24.2172
  - `test/iou`: last 0.1465
  - `test/precision`: last 0.5340
  - `test/recall`: last 0.4124
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.7271 (min -0.7516 @ ep 989)
  - `val/dice_class_0`: last 0.7122 (max 0.8283 @ ep 158)
  - `val/ema_fg_dice`: last 0.7197 (max 0.7437 @ ep 983)
  - `val/loss`: last -0.6788 (min -0.7226 @ ep 882)
  - `val/mean_fg_dice`: last 0.7122 (max 0.8283 @ ep 158)

#### `exp7_mixed_210syn`
*started ? • 1000 epochs • 47h43m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1963
  - `test/dice_large`: last 0.7621
  - `test/dice_medium`: last 0.7041
  - `test/dice_small`: last 0.3513
  - `test/dice_tiny`: last 0.1649
  - `test/hd95`: last 19.7145
  - `test/iou`: last 0.1502
  - `test/precision`: last 0.5397
  - `test/recall`: last 0.4106
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.6351 (min -0.6851 @ ep 991)
  - `val/dice_class_0`: last 0.6862 (max 0.8156 @ ep 244)
  - `val/ema_fg_dice`: last 0.6972 (max 0.7228 @ ep 968)
  - `val/loss`: last -0.6764 (min -0.7331 @ ep 816)
  - `val/mean_fg_dice`: last 0.6862 (max 0.8156 @ ep 244)

#### `exp7_mixed_25syn`
*started ? • 1000 epochs • 41h52m*

**Downstream seg metrics:**
  - `test/dice`: last 0.2025
  - `test/dice_large`: last 0.8119
  - `test/dice_medium`: last 0.7245
  - `test/dice_small`: last 0.3679
  - `test/dice_tiny`: last 0.1692
  - `test/hd95`: last 21.9009
  - `test/iou`: last 0.1560
  - `test/precision`: last 0.5211
  - `test/recall`: last 0.4499
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.7665 (min -0.7923 @ ep 993)
  - `val/dice_class_0`: last 0.6544 (max 0.8045 @ ep 408)
  - `val/ema_fg_dice`: last 0.6494 (max 0.7098 @ ep 646)
  - `val/loss`: last -0.6302 (min -0.7126 @ ep 737)
  - `val/mean_fg_dice`: last 0.6544 (max 0.8045 @ ep 408)

#### `exp7_mixed_315syn`
*started ? • 1000 epochs • 43h27m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1978
  - `test/dice_large`: last 0.7919
  - `test/dice_medium`: last 0.7173
  - `test/dice_small`: last 0.3576
  - `test/dice_tiny`: last 0.1653
  - `test/hd95`: last 22.9803
  - `test/iou`: last 0.1520
  - `test/precision`: last 0.5358
  - `test/recall`: last 0.4297
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.6308 (min -0.6531 @ ep 991)
  - `val/dice_class_0`: last 0.7327 (max 0.8415 @ ep 328)
  - `val/ema_fg_dice`: last 0.7360 (max 0.7413 @ ep 995)
  - `val/loss`: last -0.6805 (min -0.7305 @ ep 691)
  - `val/mean_fg_dice`: last 0.7327 (max 0.8415 @ ep 328)

#### `exp7_mixed_50syn`
*started ? • 1000 epochs • 43h04m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1901
  - `test/dice_large`: last 0.7831
  - `test/dice_medium`: last 0.7006
  - `test/dice_small`: last 0.3349
  - `test/dice_tiny`: last 0.1599
  - `test/hd95`: last 20.8456
  - `test/iou`: last 0.1452
  - `test/precision`: last 0.5432
  - `test/recall`: last 0.4167
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.7926 (min -0.8054 @ ep 996)
  - `val/dice_class_0`: last 0.7055 (max 0.8130 @ ep 193)
  - `val/ema_fg_dice`: last 0.7207 (max 0.7271 @ ep 995)
  - `val/loss`: last -0.6950 (min -0.7196 @ ep 481)
  - `val/mean_fg_dice`: last 0.7055 (max 0.8130 @ ep 193)

#### `exp7_mixed_525syn`
*started ? • 1000 epochs • 32h25m*

**Downstream seg metrics:**
  - `test/dice`: last 0.2079
  - `test/dice_large`: last 0.8285
  - `test/dice_medium`: last 0.7475
  - `test/dice_small`: last 0.3935
  - `test/dice_tiny`: last 0.1715
  - `test/hd95`: last 18.7076
  - `test/iou`: last 0.1615
  - `test/precision`: last 0.5146
  - `test/recall`: last 0.4707
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.6066 (min -0.6280 @ ep 994)
  - `val/dice_class_0`: last 0.7669 (max 0.8316 @ ep 774)
  - `val/ema_fg_dice`: last 0.7643 (max 0.7758 @ ep 847)
  - `val/loss`: last -0.6956 (min -0.7382 @ ep 816)
  - `val/mean_fg_dice`: last 0.7669 (max 0.8316 @ ep 774)

#### `exp7_mixed_75syn`
*started ? • 1000 epochs • 31h46m*

**Downstream seg metrics:**
  - `test/dice`: last 0.1990
  - `test/dice_large`: last 0.7957
  - `test/dice_medium`: last 0.7148
  - `test/dice_small`: last 0.3524
  - `test/dice_tiny`: last 0.1676
  - `test/hd95`: last 20.5052
  - `test/iou`: last 0.1524
  - `test/precision`: last 0.5321
  - `test/recall`: last 0.4329
  - `train/learning_rate`: last 1.995e-05
  - `train/loss`: last -0.7344 (min -0.7454 @ ep 967)
  - `val/dice_class_0`: last 0.6972 (max 0.8234 @ ep 485)
  - `val/ema_fg_dice`: last 0.7331 (max 0.7583 @ ep 597)
  - `val/loss`: last -0.6547 (min -0.7328 @ ep 549)
  - `val/mean_fg_dice`: last 0.6972 (max 0.8234 @ ep 485)

---
## downstream/old

*4 runs across 2 experiment families.*

### exp1

**Deprecated SegResNet baselines** — superseded by `downstream/SegResNet/exp1`.

#### `exp1_baseline_3d_20260216-171243`
*started 2026-02-16 17:12 • 501 epochs • 11h45m • 34.4 TFLOPs • peak VRAM 63.4 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.1522 (min 0.0330, max 0.2260)
  - `Validation/hd95`: last 27.7328 (min 15.5096, max 70.3754)
  - `Validation/precision`: last 0.4783 (min 0.0072, max 0.6258)
  - `Validation/recall`: last 0.3290 (min 0.0032, max 0.5097)

**Regional loss (final):**
  - `regional_seg/dice`: 0.1975
  - `regional_seg/dice_large`: 0.8420
  - `regional_seg/dice_medium`: 0.4178
  - `regional_seg/dice_small`: 0.3372
  - `regional_seg/dice_tiny`: 0.1606
  - `regional_seg/iou`: 0.1522

**Downstream seg metrics:**
  - `Validation/hd95`: last 27.7328 (min 15.5096 @ ep 134)
  - `Validation/precision`: last 0.4783 (max 0.6258 @ ep 159)
  - `Validation/recall`: last 0.3290 (max 0.5097 @ ep 18)
  - `regional_seg/dice`: last 0.1975 (max 0.2943 @ ep 7)
  - `regional_seg/dice_large`: last 0.8420
  - `regional_seg/dice_medium`: last 0.4178 (max 0.7228 @ ep 63)
  - `regional_seg/dice_small`: last 0.3372 (max 0.3668 @ ep 18)
  - `regional_seg/dice_tiny`: last 0.1606 (max 0.2795 @ ep 7)
  - `regional_seg/iou`: last 0.1522 (max 0.2260 @ ep 7)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.4469, max 4.1082 @ ep 85
  - `training/grad_norm_max`: last 3.0126, max 40.9761 @ ep 84

#### `exp1_baseline_2d_20260217-124658`
*started 2026-02-17 12:46 • 501 epochs • 55h13m • 11084.8 TFLOPs • peak VRAM 7.4 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.2553 (min 0.2164, max 0.3196)
  - `Validation/hd95`: last 45.0810 (min 45.0810, max 88.5183)
  - `Validation/precision`: last 0.4933 (min 0.0367, max 0.6398)
  - `Validation/recall`: last 0.3273 (min 0.2679, max 0.4872)

**Regional loss (final):**
  - `regional_seg/dice`: 0.2866
  - `regional_seg/dice_large`: 0
  - `regional_seg/dice_medium`: 0.6735
  - `regional_seg/dice_small`: 0.3711
  - `regional_seg/dice_tiny`: 0.2496
  - `regional_seg/iou`: 0.2553

**Downstream seg metrics:**
  - `Validation/hd95`: last 45.0810
  - `Validation/precision`: last 0.4933 (max 0.6398 @ ep 184)
  - `Validation/recall`: last 0.3273 (max 0.4872 @ ep 10)
  - `regional_seg/dice`: last 0.2866 (max 0.3815 @ ep 1)
  - `regional_seg/dice_large`: last 0 (max 0.8459 @ ep 77)
  - `regional_seg/dice_medium`: last 0.6735 (max 0.7342 @ ep 10)
  - `regional_seg/dice_small`: last 0.3711 (max 0.4460 @ ep 10)
  - `regional_seg/dice_tiny`: last 0.2496 (max 0.3532 @ ep 1)
  - `regional_seg/iou`: last 0.2553 (max 0.3196 @ ep 1)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.9333, max 4.4529 @ ep 27
  - `training/grad_norm_max`: last 7.3819, max 42.8538 @ ep 47

### exp2

**Deprecated SegResNet dual** — superseded by `downstream/SegResNet/exp2`.

#### `exp2_dual_3d_20260217-012940`
*started 2026-02-17 01:29 • 501 epochs • 10h44m • 34.4 TFLOPs • peak VRAM 63.5 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.1421 (min 0.0330, max 0.1739)
  - `Validation/hd95`: last 22.6062 (min 22.6062, max 77.3705)
  - `Validation/precision`: last 0.4833 (min 0.0578, max 0.6330)
  - `Validation/recall`: last 0.3029 (min 0.0153, max 0.4695)

**Regional loss (final):**
  - `regional_seg/dice`: 0.1855
  - `regional_seg/dice_large`: 0.8749
  - `regional_seg/dice_medium`: 0.3947
  - `regional_seg/dice_small`: 0.3117
  - `regional_seg/dice_tiny`: 0.1517
  - `regional_seg/iou`: 0.1421

**Downstream seg metrics:**
  - `Validation/hd95`: last 22.6062
  - `Validation/precision`: last 0.4833 (max 0.6330 @ ep 288)
  - `Validation/recall`: last 0.3029 (max 0.4695 @ ep 70)
  - `regional_seg/dice`: last 0.1855 (max 0.2269 @ ep 38)
  - `regional_seg/dice_large`: last 0.8749
  - `regional_seg/dice_medium`: last 0.3947 (max 0.6453 @ ep 70)
  - `regional_seg/dice_small`: last 0.3117 (max 0.3569 @ ep 38)
  - `regional_seg/dice_tiny`: last 0.1517 (max 0.1941 @ ep 83)
  - `regional_seg/iou`: last 0.1421 (max 0.1739 @ ep 38)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 1.5574, max 4.3603 @ ep 116
  - `training/grad_norm_max`: last 6.0655, max 127.2468 @ ep 497

#### `exp2_dual_2d_20260217-130537`
*started 2026-02-17 13:05 • 501 epochs • 81h58m • 11098.3 TFLOPs • peak VRAM 7.5 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.2396 (min 0.1915, max 0.2945)
  - `Validation/hd95`: last 41.8133 (min 41.8133, max 80.7358)
  - `Validation/precision`: last 0.4552 (min 0.1261, max 0.6168)
  - `Validation/recall`: last 0.3074 (min 0.2988, max 0.5021)

**Regional loss (final):**
  - `regional_seg/dice`: 0.2733
  - `regional_seg/dice_large`: 0
  - `regional_seg/dice_medium`: 0.6135
  - `regional_seg/dice_small`: 0.3629
  - `regional_seg/dice_tiny`: 0.2361
  - `regional_seg/iou`: 0.2396

**Downstream seg metrics:**
  - `Validation/hd95`: last 41.8133
  - `Validation/precision`: last 0.4552 (max 0.6168 @ ep 199)
  - `Validation/recall`: last 0.3074 (max 0.5021 @ ep 10)
  - `regional_seg/dice`: last 0.2733 (max 0.3303 @ ep 10)
  - `regional_seg/dice_large`: last 0 (max 0.8329 @ ep 79)
  - `regional_seg/dice_medium`: last 0.6135 (max 0.7171 @ ep 22)
  - `regional_seg/dice_small`: last 0.3629 (max 0.4954 @ ep 10)
  - `regional_seg/dice_tiny`: last 0.2361 (max 0.2599 @ ep 10)
  - `regional_seg/iou`: last 0.2396 (max 0.2945 @ ep 10)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 2.3636, max 4.0516 @ ep 29
  - `training/grad_norm_max`: last 5.7289, max 39.7433 @ ep 102

---
## diffusion_2d/bravo

*78 runs across 26 experiment families.*

### exp1

**exp1/exp_1 (2D bravo)** — earliest 2D bravo baseline at 128×128.
`exp_1_rflow_128` is the pre-refactor naming convention; `exp_1_3_continuous_rflow_128`
tested continuous-timestep RFlow.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp1_2_ddpm_128_20251223-233504` — 0.0051
  2. 🥈 `exp_1_3_continuous_rflow_128_20260119-230102` — 0.0066
  3.  `exp_1_rflow_128_20251221-004553` — 0.0067

#### `exp_1_rflow_128_20251221-004553`
*started 2025-12-21 00:45 • 500 epochs • 15h57m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1795 → 0.0037 (min 0.0036 @ ep 477)
  - `Loss/MSE_val`: 0.0099 → 0.0126 (min 0.0067 @ ep 62)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5351 (min 0.5068, max 0.9738)
  - `Validation/MS-SSIM`: last 0.8137 (min 0.7952, max 0.8612)
  - `Validation/PSNR`: last 27.4557 (min 27.3990, max 28.9957)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0270, max 3.3241 @ ep 0
  - `training/grad_norm_max`: last 0.1116, max 10.9716 @ ep 0

#### `exp1_2_ddpm_128_20251223-233504`
*started 2025-12-23 23:35 • 500 epochs • 12h46m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1800 → 0.0035 (min 0.0033 @ ep 474)
  - `Loss/MSE_val`: 0.0079 → 0.0088 (min 0.0051 @ ep 90)

**Validation quality:**
  - `Validation/LPIPS`: last 0.9988 (min 0.9194, max 1.4945)
  - `Validation/MS-SSIM`: last 0.7917 (min 0.7133, max 0.8097)
  - `Validation/PSNR`: last 24.0688 (min 20.9943, max 25.6240)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0265, max 3.1763 @ ep 0
  - `training/grad_norm_max`: last 0.1086, max 11.0511 @ ep 0

#### `exp_1_3_continuous_rflow_128_20260119-230102`
*started 2026-01-19 23:01 • 500 epochs • 16h04m • 3964808.8 TFLOPs • peak VRAM 38.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1787 → 0.0037 (min 0.0036 @ ep 487)
  - `Loss/MSE_val`: 0.0101 → 0.0125 (min 0.0066 @ ep 72)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0006219
  - 0.1-0.2: 0.0010
  - 0.2-0.3: 0.0013
  - 0.3-0.4: 0.0014
  - 0.4-0.5: 0.0016
  - 0.5-0.6: 0.0017
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0031
  - 0.9-1.0: 0.0034

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0476, best 0.0398 @ ep 37
  - `Generation/KID_mean_train`: last 0.0539, best 0.0410 @ ep 37
  - `Generation/KID_std_val`: last 0.0048, best 0.0024 @ ep 37
  - `Generation/KID_std_train`: last 0.0068, best 0.0018 @ ep 8
  - `Generation/CMMD_val`: last 0.2436, best 0.1995 @ ep 129
  - `Generation/CMMD_train`: last 0.2508, best 0.2043 @ ep 65
  - `Generation/extended_KID_mean_val`: last 0.0430, best 0.0370 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.0447, best 0.0393 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.2253, best 0.1778 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.2352, best 0.1852 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.8343 (min 0.8078, max 1.3560)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9044 (min 0.8279, max 0.9080)
  - `Validation/MS-SSIM_bravo`: last 0.8169 (min 0.7964, max 0.8593)
  - `Validation/PSNR_bravo`: last 27.5633 (min 27.5495, max 29.1010)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0129
  - `regional_bravo/medium`: 0.0210
  - `regional_bravo/small`: 0.0147
  - `regional_bravo/tiny`: 0.0093
  - `regional_bravo/tumor_bg_ratio`: 4.9153
  - `regional_bravo/tumor_loss`: 0.0148

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0235, max 3.3121 @ ep 0
  - `training/grad_norm_max`: last 0.0919, max 10.8832 @ ep 0

### exp2

**exp2/exp_2** — 2D bravo with RFlow. Multiple runs across Dec 2025 →
Feb 2026 reflect iterative refactors; tag counts grow (40 → 103)
as metrics were added.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp_2_rflow_128_20260118-211457` — 0.0356
  2. 🥈 `exp_2_rflow_128_20260113-203015` — 0.0416

#### `exp_2_rflow_128_20251221-004554`
*started 2025-12-21 00:45 • 500 epochs • 15h51m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1786 → 0.0038 (min 0.0037 @ ep 444)
  - `Loss/MSE_val`: 0.0096 → 0.0132 (min 0.0067 @ ep 48)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5343 (min 0.4968, max 0.9013)
  - `Validation/MS-SSIM`: last 0.8177 (min 0.8013, max 0.8608)
  - `Validation/PSNR`: last 27.5303 (min 27.5033, max 28.9756)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0284, max 3.3309 @ ep 0
  - `training/grad_norm_max`: last 0.1332, max 10.9497 @ ep 0

#### `exp_2_rflow_128_20251226-230709`
*started 2025-12-26 23:07 • 500 epochs • 16h10m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1784 → 0.0037 (min 0.0037 @ ep 483)
  - `Loss/MSE_val`: 0.0098 → 0.0122 (min 0.0069 @ ep 47)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5367 (min 0.5053, max 1.1180)
  - `Validation/MS-SSIM`: last 0.8138 (min 0.7958, max 0.8610)
  - `Validation/PSNR`: last 27.4755 (min 27.4734, max 29.0029)

**Regional loss (final):**
  - `regional/background_loss`: 0.0033
  - `regional/large`: 0.0130
  - `regional/medium`: 0.0208
  - `regional/small`: 0.0133
  - `regional/tiny`: 0.0092
  - `regional/tumor_bg_ratio`: 4.2947
  - `regional/tumor_loss`: 0.0144

**Training meta:**
  - `training/grad_norm_avg`: last 0.0334, max 3.3610 @ ep 0
  - `training/grad_norm_max`: last 0.1269, max 11.0089 @ ep 0

#### `exp_2_rflow_128_20260107-023421`
*started 2026-01-07 02:34 • 145 epochs • 5h42m • 1149794.6 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1782 → 0.0044 (min 0.0044 @ ep 144)
  - `Loss/MSE_val`: 0.0098 → 0.0097 (min 0.0070 @ ep 31)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 8.821e-05
  - 10-19: 0.0002299
  - 20-29: 0.0004072
  - 30-39: 0.000587
  - 40-49: 0.0007407
  - 50-59: 0.0009471
  - 60-69: 0.0012
  - 70-79: 0.0015
  - 80-89: 0.0021
  - 90-99: 0.0026

**Validation quality:**
  - `Validation/LPIPS`: last 0.6592 (min 0.5108, max 1.1059)
  - `Validation/MS-SSIM`: last 0.8346 (min 0.7956, max 0.8591)
  - `Validation/MS-SSIM-3D`: last 0.9021 (min 0.8344, max 0.9059)
  - `Validation/PSNR`: last 28.5898 (min 27.8702, max 29.2496)

**Regional loss (final):**
  - `regional/background_loss`: 0.0024
  - `regional/large`: 0.0128
  - `regional/medium`: 0.0210
  - `regional/small`: 0.0126
  - `regional/tiny`: 0.0087
  - `regional/tumor_bg_ratio`: 5.9366
  - `regional/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Model`: peak 0.0001 @ ep 5, final 8.196e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0858, max 3.2824 @ ep 0
  - `training/grad_norm_max`: last 0.3672, max 10.8783 @ ep 0

#### `exp_2_rflow_128_20260107-144820`
*started 2026-01-07 14:48 • 500 epochs • 20h27m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1791 → 0.0028 (min 0.0028 @ ep 466)
  - `Loss/MSE_val`: 0.0106 → 0.0135 (min 0.0071 @ ep 50)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 2.212e-05
  - 10-19: 8.344e-05
  - 20-29: 0.0001948
  - 30-39: 0.0003782
  - 40-49: 0.0005866
  - 50-59: 0.0008102
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0019
  - 90-99: 0.0025

**Validation quality:**
  - `Validation/LPIPS`: last 0.5406 (min 0.5170, max 1.0396)
  - `Validation/MS-SSIM`: last 0.8158 (min 0.7931, max 0.8569)
  - `Validation/MS-SSIM-3D`: last 0.9005 (min 0.8317, max 0.9045)
  - `Validation/PSNR`: last 28.0073 (min 27.8039, max 29.2502)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0137
  - `regional/medium`: 0.0225
  - `regional/small`: 0.0143
  - `regional/tiny`: 0.0098
  - `regional/tumor_bg_ratio`: 5.3241
  - `regional/tumor_loss`: 0.0154

**LR schedule:**
  - `LR/Model`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0332, max 3.3104 @ ep 0
  - `training/grad_norm_max`: last 0.1439, max 10.8891 @ ep 0

#### `exp2_bf16mse_rflow_128_20260108-002738`
*started 2026-01-08 00:27 • 107 epochs • 3h55m • 848469.1 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1791 → 0.0049 (min 0.0049 @ ep 106)
  - `Loss/MSE_val`: 0.0103 → 0.0081 (min 0.0069 @ ep 37)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001254
  - 10-19: 0.0002916
  - 20-29: 0.0004476
  - 30-39: 0.0006251
  - 40-49: 0.0007998
  - 50-59: 0.0010
  - 60-69: 0.0013
  - 70-79: 0.0016
  - 80-89: 0.0022
  - 90-99: 0.0026

**Validation quality:**
  - `Validation/LPIPS`: last 0.5241 (min 0.5123, max 0.9755)
  - `Validation/MS-SSIM`: last 0.8436 (min 0.7929, max 0.8568)
  - `Validation/MS-SSIM-3D`: last 0.9008 (min 0.8302, max 0.9051)
  - `Validation/PSNR`: last 28.8631 (min 27.7366, max 29.2340)

**Regional loss (final):**
  - `regional/background_loss`: 0.0020
  - `regional/large`: 0.0102
  - `regional/medium`: 0.0156
  - `regional/small`: 0.0129
  - `regional/tiny`: 0.0093
  - `regional/tumor_bg_ratio`: 5.9744
  - `regional/tumor_loss`: 0.0122

**LR schedule:**
  - `LR/Model`: peak 0.0001 @ ep 5, final 9.017e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0894, max 3.3715 @ ep 0
  - `training/grad_norm_max`: last 0.3524, max 10.9592 @ ep 0

#### `exp_2_rflow_128_20260109-193325`
*started 2026-01-09 19:33 • 500 epochs • 19h53m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1780 → 0.0038 (min 0.0037 @ ep 455)
  - `Loss/MSE_val`: 0.0098 → 0.0126 (min 0.0069 @ ep 41)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001266
  - 10-19: 0.0001988
  - 20-29: 0.0002798
  - 30-39: 0.0004286
  - 40-49: 0.0005856
  - 50-59: 0.0007995
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0020
  - 90-99: 0.0024

**Validation quality:**
  - `Validation/LPIPS`: last 0.5340 (min 0.5066, max 0.8983)
  - `Validation/MS-SSIM`: last 0.8214 (min 0.8032, max 0.8623)
  - `Validation/MS-SSIM-3D`: last 0.9037 (min 0.8325, max 0.9065)
  - `Validation/PSNR`: last 28.2637 (min 28.0644, max 29.3548)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0138
  - `regional/medium`: 0.0204
  - `regional/small`: 0.0140
  - `regional/tiny`: 0.0098
  - `regional/tumor_bg_ratio`: 4.9262
  - `regional/tumor_loss`: 0.0147

**LR schedule:**
  - `LR/Model`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0300, max 3.3549 @ ep 0
  - `training/grad_norm_max`: last 0.1147, max 10.9819 @ ep 0

#### `exp_2_rflow_128_20260113-203015`
*started 2026-01-13 20:30 • 500 epochs • 21h10m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1795 → 0.0038 (min 0.0037 @ ep 478)
  - `Loss/MSE_val`: 0.0099 → 0.0130 (min 0.0068 @ ep 50)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001236
  - 10-19: 0.0002003
  - 20-29: 0.000285
  - 30-39: 0.0004358
  - 40-49: 0.0005932
  - 50-59: 0.0007926
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0018
  - 90-99: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0509, best 0.0368 @ ep 35
  - `Generation/KID_mean_train`: last 0.0570, best 0.0355 @ ep 35
  - `Generation/KID_std_val`: last 0.0056, best 0.0025 @ ep 63
  - `Generation/KID_std_train`: last 0.0071, best 0.0024 @ ep 57
  - `Generation/CMMD_val`: last 0.2456, best 0.1933 @ ep 75
  - `Generation/CMMD_train`: last 0.2536, best 0.1918 @ ep 34
  - `Generation/extended_KID_mean_val`: last 0.0455, best 0.0416 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.0475, best 0.0432 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.2270, best 0.1886 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.2364, best 0.1964 @ ep 124

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5336 (min 0.5036, max 0.9101)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9028 (min 0.8411, max 0.9075)
  - `Validation/MS-SSIM_bravo`: last 0.8191 (min 0.7987, max 0.8615)
  - `Validation/PSNR_bravo`: last 28.1909 (min 27.8937, max 29.3812)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0029
  - `regional_bravo/large`: 0.0143
  - `regional_bravo/medium`: 0.0206
  - `regional_bravo/small`: 0.0147
  - `regional_bravo/tiny`: 0.0094
  - `regional_bravo/tumor_bg_ratio`: 5.1290
  - `regional_bravo/tumor_loss`: 0.0150

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0418, max 3.3453 @ ep 0
  - `training/grad_norm_max`: last 0.1649, max 10.8556 @ ep 0

#### `exp_2_rflow_128_20260118-211457`
*started 2026-01-18 21:14 • 500 epochs • 20h34m • 3964808.8 TFLOPs • peak VRAM 34.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1790 → 0.0037 (min 0.0036 @ ep 451)
  - `Loss/MSE_val`: 0.0097 → 0.0123 (min 0.0066 @ ep 45)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0004806
  - 0.1-0.2: 0.0011
  - 0.2-0.3: 0.0013
  - 0.3-0.4: 0.0013
  - 0.4-0.5: 0.0015
  - 0.5-0.6: 0.0019
  - 0.6-0.7: 0.0020
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0031
  - 0.9-1.0: 0.0033

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0499, best 0.0363 @ ep 40
  - `Generation/KID_mean_train`: last 0.0544, best 0.0369 @ ep 40
  - `Generation/KID_std_val`: last 0.0051, best 0.0018 @ ep 35
  - `Generation/KID_std_train`: last 0.0073, best 0.0025 @ ep 47
  - `Generation/CMMD_val`: last 0.2453, best 0.1991 @ ep 88
  - `Generation/CMMD_train`: last 0.2525, best 0.2032 @ ep 146
  - `Generation/extended_KID_mean_val`: last 0.0394, best 0.0356 @ ep 239
  - `Generation/extended_KID_mean_train`: last 0.0458, best 0.0413 @ ep 239
  - `Generation/extended_CMMD_val`: last 0.2209, best 0.1871 @ ep 159
  - `Generation/extended_CMMD_train`: last 0.2309, best 0.1966 @ ep 159

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.8236 (min 0.8117, max 1.4714)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9054 (min 0.8368, max 0.9078)
  - `Validation/MS-SSIM_bravo`: last 0.8202 (min 0.7983, max 0.8594)
  - `Validation/PSNR_bravo`: last 27.6160 (min 27.5559, max 28.9882)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0131
  - `regional_bravo/medium`: 0.0227
  - `regional_bravo/small`: 0.0136
  - `regional_bravo/tiny`: 0.0091
  - `regional_bravo/tumor_bg_ratio`: 5.0536
  - `regional_bravo/tumor_loss`: 0.0150

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0225, max 3.3350 @ ep 0
  - `training/grad_norm_max`: last 0.0813, max 10.8998 @ ep 0

### exp3

**exp_3** — early 2D baseline variant.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp_3_rflow_128_20251221-004755` — 0.0069
  2. 🥈 `exp_3_1_rflow_128_20251222-135657` — 0.0071

#### `exp_3_rflow_128_20251221-004755`
*started 2025-12-21 00:47 • 500 epochs • 12h41m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1785 → 0.0046 (min 0.0045 @ ep 484)
  - `Loss/MSE_val`: 0.0105 → 0.0104 (min 0.0069 @ ep 85)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5671 (min 0.5227, max 1.2750)
  - `Validation/MS-SSIM`: last 0.8067 (min 0.7852, max 0.8519)
  - `Validation/PSNR`: last 27.4682 (min 27.3063, max 28.8698)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0257, max 3.3639 @ ep 0
  - `training/grad_norm_max`: last 0.0956, max 10.8538 @ ep 0

#### `exp_3_1_rflow_128_20251222-135657`
*started 2025-12-22 13:56 • 500 epochs • 15h42m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1794 → 0.0060 (min 0.0059 @ ep 456)
  - `Loss/MSE_val`: 0.0123 → 0.0077 (min 0.0071 @ ep 77)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5755 (min 0.5545, max 1.1439)
  - `Validation/MS-SSIM`: last 0.8166 (min 0.7702, max 0.8318)
  - `Validation/PSNR`: last 27.9421 (min 26.5438, max 28.3687)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0269, max 3.3370 @ ep 0
  - `training/grad_norm_max`: last 0.0920, max 10.9666 @ ep 0

### exp4

**exp_4** — early 2D baseline variant.

#### `exp_4_rflow_128_20251221-010121`
*started 2025-12-21 01:01 • 500 epochs • 12h36m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1787 → 0.0045 (min 0.0045 @ ep 492)
  - `Loss/MSE_val`: 0.0107 → 0.0107 (min 0.0069 @ ep 99)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5613 (min 0.5230, max 1.0350)
  - `Validation/MS-SSIM`: last 0.8111 (min 0.7725, max 0.8513)
  - `Validation/PSNR`: last 27.5962 (min 26.5571, max 28.8503)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0253, max 3.3504 @ ep 0
  - `training/grad_norm_max`: last 0.0916, max 10.8987 @ ep 0

### exp5

**exp_5** — early 2D baseline variant.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp_5_1_rflow_128_20260112-213706` — 0.0067
  2. 🥈 `exp_5_rflow_128_20251221-010151` — 0.0068

#### `exp_5_rflow_128_20251221-010151`
*started 2025-12-21 01:01 • 500 epochs • 12h40m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1787 → 0.0044 (min 0.0044 @ ep 472)
  - `Loss/MSE_val`: 0.0099 → 0.0112 (min 0.0068 @ ep 168)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5746 (min 0.5260, max 0.9513)
  - `Validation/MS-SSIM`: last 0.8043 (min 0.7935, max 0.8523)
  - `Validation/PSNR`: last 27.4439 (min 27.3659, max 28.8890)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0245, max 3.3392 @ ep 0
  - `training/grad_norm_max`: last 0.0934, max 10.9361 @ ep 0

#### `exp_5_1_rflow_128_20260112-213706`
*started 2026-01-12 21:37 • 500 epochs • 21h50m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1739 → 0.0034 (min 0.0034 @ ep 481)
  - `Loss/MSE_val`: 0.0097 → 0.0134 (min 0.0067 @ ep 40)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001388
  - 10-19: 0.0001977
  - 20-29: 0.0002863
  - 30-39: 0.0004118
  - 40-49: 0.0006038
  - 50-59: 0.0008032
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0019
  - 90-99: 0.0021

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5286 (min 0.4999, max 0.8645)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9039 (min 0.8459, max 0.9068)
  - `Validation/MS-SSIM_bravo`: last 0.8175 (min 0.8062, max 0.8596)
  - `Validation/PSNR_bravo`: last 28.1391 (min 27.9560, max 29.3555)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0155
  - `regional_bravo/medium`: 0.0214
  - `regional_bravo/small`: 0.0137
  - `regional_bravo/tiny`: 0.0096
  - `regional_bravo/tumor_bg_ratio`: 5.0400
  - `regional_bravo/tumor_loss`: 0.0153

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0424, max 3.2801 @ ep 0
  - `training/grad_norm_max`: last 0.1804, max 11.0942 @ ep 0

### exp6

**exp6** — early 2D ablation.

#### `exp6_rflow_256_20251221-173255`
*started 2025-12-21 17:32 • 500 epochs • 33h03m • 7510910.5 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1766 → 0.0042 (min 0.0041 @ ep 482)
  - `Loss/MSE_val`: 0.0088 → 0.0072 (min 0.0056 @ ep 144)

**Validation quality:**
  - `Validation/LPIPS`: last 0.3503 (min 0.3389, max 0.7425)
  - `Validation/MS-SSIM`: last 0.8869 (min 0.8393, max 0.8956)
  - `Validation/PSNR`: last 29.2020 (min 27.6452, max 29.7592)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0211, max 3.3376 @ ep 0
  - `training/grad_norm_max`: last 0.0750, max 10.9253 @ ep 0

### exp7

**exp7** — early 2D ablation.

#### `exp7_rflow_256_20251221-014856`
*started 2025-12-21 01:48 • 500 epochs • 15h31m • 3336546.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1748 → 0.0024 (min 0.0024 @ ep 480)
  - `Loss/MSE_val`: 0.0077 → 0.0125 (min 0.0056 @ ep 41)

**Validation quality:**
  - `Validation/LPIPS`: last 0.3805 (min 0.3270, max 0.6317)
  - `Validation/MS-SSIM`: last 0.8598 (min 0.8570, max 0.9005)
  - `Validation/PSNR`: last 27.8234 (min 27.7495, max 29.6547)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0179, max 3.3173 @ ep 0
  - `training/grad_norm_max`: last 0.0725, max 10.9367 @ ep 0

### exp8

**exp8 (2D sweep)** — 14 variants of 2D bravo with different conditioning
approaches, architectures, or hyperparameters. These were exploratory
runs prior to the 3D pivot.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp8_1b_rflow_128_20251227-121725` — 0.0069
  2. 🥈 `exp8_1_rflow_128_20251221-141917` — 0.0069
  3.  `exp8_8b_rflow_128_20251227-121725` — 0.0069
  4.  `exp8_3_rflow_128_20251221-141947` — 0.0069
  5.  `exp8_2_rflow_128_20251221-141917` — 0.0069
  6.  `exp8_4b_rflow_128_20251227-121725` — 0.0070
  7.  `exp8_9_rflow_128_20251223-233604` — 0.0071
  8.  `exp8_8_rflow_128_20251223-233604` — 0.0072
  9.  `exp8_6_rflow_128_20251223-075531` — 0.0072
  10.  `exp8_10_rflow_128_20251223-233523` — 0.0072
  11.  `exp8_11_rflow_128_20251223-235153` — 0.0073
  12.  `exp8_5_rflow_128_20251223-075531` — 0.0073
  13.  `exp8_7_rflow_128_20251223-080403` — 0.0073
  14.  `exp8_4_rflow_128_20251223-075531` — 0.0073

#### `exp8_1_rflow_128_20251221-141917`
*started 2025-12-21 14:19 • 500 epochs • 12h33m • 1683467.4 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3445 → 0.0046 (min 0.0046 @ ep 492)
  - `Loss/MSE_val`: 0.0152 → 0.0101 (min 0.0069 @ ep 140)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5703 (min 0.5306, max 1.4761)
  - `Validation/MS-SSIM`: last 0.8155 (min 0.7653, max 0.8501)
  - `Validation/PSNR`: last 27.7509 (min 26.2666, max 28.8273)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0283, max 4.4668 @ ep 0
  - `training/grad_norm_max`: last 0.0910, max 7.7577 @ ep 0

#### `exp8_2_rflow_128_20251221-141917`
*started 2025-12-21 14:19 • 500 epochs • 5h59m • 709288.4 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3417 → 0.0050 (min 0.0050 @ ep 499)
  - `Loss/MSE_val`: 0.0141 → 0.0086 (min 0.0069 @ ep 142)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5598 (min 0.5264, max 1.4686)
  - `Validation/MS-SSIM`: last 0.8229 (min 0.7579, max 0.8510)
  - `Validation/PSNR`: last 27.8895 (min 26.2147, max 28.8292)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0284, max 4.4273 @ ep 0
  - `training/grad_norm_max`: last 0.0829, max 7.6999 @ ep 0

#### `exp8_3_rflow_128_20251221-141947`
*started 2025-12-21 14:19 • 500 epochs • 4h50m • 299485.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.5925 → 0.0053 (min 0.0052 @ ep 418)
  - `Loss/MSE_val`: 0.2575 → 0.0079 (min 0.0069 @ ep 145)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5574 (min 0.5373, max 1.8865)
  - `Validation/MS-SSIM`: last 0.8350 (min 0.3687, max 0.8523)
  - `Validation/PSNR`: last 28.3119 (min 14.0322, max 28.8394)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0256, max 4.6531 @ ep 0
  - `training/grad_norm_max`: last 0.0693, max 5.6068 @ ep 0

#### `exp8_4_rflow_128_20251223-075531`
*started 2025-12-23 07:55 • 500 epochs • 6h16m • 793658.1 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3399 → 0.0053 (min 0.0052 @ ep 486)
  - `Loss/MSE_val`: 0.0143 → 0.0108 (min 0.0073 @ ep 94)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6110 (min 0.5572, max 1.4179)
  - `Validation/MS-SSIM`: last 0.7774 (min 0.7476, max 0.8315)
  - `Validation/PSNR`: last 26.7938 (min 26.0427, max 28.3102)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0298, max 4.4088 @ ep 0
  - `training/grad_norm_max`: last 0.0917, max 7.7717 @ ep 0

#### `exp8_5_rflow_128_20251223-075531`
*started 2025-12-23 07:55 • 500 epochs • 6h03m • 438472.5 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3392 → 0.0054 (min 0.0054 @ ep 484)
  - `Loss/MSE_val`: 0.0126 → 0.0118 (min 0.0073 @ ep 104)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6351 (min 0.5662, max 1.5462)
  - `Validation/MS-SSIM`: last 0.7604 (min 0.7586, max 0.8300)
  - `Validation/PSNR`: last 26.4715 (min 26.3762, max 28.3105)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0306, max 4.3666 @ ep 0
  - `training/grad_norm_max`: last 0.1049, max 7.7405 @ ep 0

#### `exp8_6_rflow_128_20251223-075531`
*started 2025-12-23 07:55 • 500 epochs • 7h19m • 1097464.4 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3419 → 0.0051 (min 0.0050 @ ep 481)
  - `Loss/MSE_val`: 0.0138 → 0.0129 (min 0.0072 @ ep 108)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6129 (min 0.5525, max 1.4985)
  - `Validation/MS-SSIM`: last 0.7693 (min 0.7556, max 0.8311)
  - `Validation/PSNR`: last 26.5534 (min 25.9865, max 28.3050)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0293, max 4.4215 @ ep 0
  - `training/grad_norm_max`: last 0.0960, max 7.7199 @ ep 0

#### `exp8_7_rflow_128_20251223-080403`
*started 2025-12-23 08:04 • 500 epochs • 5h48m • 383672.7 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.5920 → 0.0055 (min 0.0054 @ ep 468)
  - `Loss/MSE_val`: 0.2571 → 0.0104 (min 0.0073 @ ep 130)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6035 (min 0.5489, max 1.8878)
  - `Validation/MS-SSIM`: last 0.7928 (min 0.3902, max 0.8323)
  - `Validation/PSNR`: last 27.1972 (min 14.4410, max 28.3323)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0262, max 4.6690 @ ep 0
  - `training/grad_norm_max`: last 0.1198, max 5.5770 @ ep 0

#### `exp8_10_rflow_128_20251223-233523`
*started 2025-12-23 23:35 • 500 epochs • 11h26m • 2623961.5 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1760 → 0.0050 (min 0.0050 @ ep 479)
  - `Loss/MSE_val`: 0.0095 → 0.0140 (min 0.0072 @ ep 125)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6372 (min 0.5478, max 1.0522)
  - `Validation/MS-SSIM`: last 0.7537 (min 0.7500, max 0.8326)
  - `Validation/PSNR`: last 26.3485 (min 26.1733, max 28.3636)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0257, max 3.2959 @ ep 0
  - `training/grad_norm_max`: last 0.0787, max 10.8423 @ ep 0

#### `exp8_8_rflow_128_20251223-233604`
*started 2025-12-23 23:36 • 500 epochs • 15h48m • 2964397.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1765 → 0.0053 (min 0.0052 @ ep 493)
  - `Loss/MSE_val`: 0.0100 → 0.0105 (min 0.0072 @ ep 148)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6040 (min 0.5506, max 1.0066)
  - `Validation/MS-SSIM`: last 0.7811 (min 0.7757, max 0.8321)
  - `Validation/PSNR`: last 26.9215 (min 26.8276, max 28.3265)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0238, max 3.2678 @ ep 0
  - `training/grad_norm_max`: last 0.0987, max 10.8763 @ ep 0

#### `exp8_9_rflow_128_20251223-233604`
*started 2025-12-23 23:36 • 500 epochs • 27h31m • 6650664.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1766 → 0.0057 (min 0.0055 @ ep 477)
  - `Loss/MSE_val`: 0.0103 → 0.0091 (min 0.0071 @ ep 84)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5895 (min 0.5485, max 1.1296)
  - `Validation/MS-SSIM`: last 0.7985 (min 0.7914, max 0.8333)
  - `Validation/PSNR`: last 27.2624 (min 27.2624, max 28.3702)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0254, max 3.2820 @ ep 0
  - `training/grad_norm_max`: last 0.0944, max 10.7877 @ ep 0

#### `exp8_11_rflow_128_20251223-235153`
*started 2025-12-23 23:51 • 500 epochs • 35h31m • 5554341.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1450 → 0.0052 (min 0.0051 @ ep 472)
  - `Loss/MSE_val`: 0.0099 → 0.0097 (min 0.0073 @ ep 105)

**Validation quality:**
  - `Validation/LPIPS`: last 0.6061 (min 0.5481, max 0.9695)
  - `Validation/MS-SSIM`: last 0.7886 (min 0.7853, max 0.8329)
  - `Validation/PSNR`: last 27.0842 (min 27.0091, max 28.3230)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0248, max 2.9996 @ ep 0
  - `training/grad_norm_max`: last 0.1086, max 12.1595 @ ep 0

#### `exp8_1b_rflow_128_20251227-121725`
*started 2025-12-27 12:17 • 500 epochs • 10h50m • 1683467.4 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3415 → 0.0039 (min 0.0038 @ ep 495)
  - `Loss/MSE_val`: 0.0137 → 0.0119 (min 0.0069 @ ep 38)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5297 (min 0.5059, max 1.5554)
  - `Validation/MS-SSIM`: last 0.8209 (min 0.7700, max 0.8613)
  - `Validation/PSNR`: last 27.6221 (min 26.3333, max 29.0391)

**Regional loss (final):**
  - `regional/background_loss`: 0.0032
  - `regional/large`: 0.0134
  - `regional/medium`: 0.0226
  - `regional/small`: 0.0138
  - `regional/tiny`: 0.0088
  - `regional/tumor_bg_ratio`: 4.6726
  - `regional/tumor_loss`: 0.0150

**Training meta:**
  - `training/grad_norm_avg`: last 0.0264, max 4.4219 @ ep 0
  - `training/grad_norm_max`: last 0.0973, max 7.7884 @ ep 0

#### `exp8_4b_rflow_128_20251227-121725`
*started 2025-12-27 12:17 • 500 epochs • 6h40m • 793658.1 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3416 → 0.0032 (min 0.0032 @ ep 480)
  - `Loss/MSE_val`: 0.0130 → 0.0146 (min 0.0070 @ ep 28)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5435 (min 0.5060, max 1.4270)
  - `Validation/MS-SSIM`: last 0.8069 (min 0.7867, max 0.8587)
  - `Validation/PSNR`: last 27.1634 (min 26.7796, max 28.9310)

**Regional loss (final):**
  - `regional/background_loss`: 0.0039
  - `regional/large`: 0.0141
  - `regional/medium`: 0.0206
  - `regional/small`: 0.0149
  - `regional/tiny`: 0.0097
  - `regional/tumor_bg_ratio`: 3.9064
  - `regional/tumor_loss`: 0.0151

**Training meta:**
  - `training/grad_norm_avg`: last 0.0303, max 4.4128 @ ep 0
  - `training/grad_norm_max`: last 0.1072, max 7.6891 @ ep 0

#### `exp8_8b_rflow_128_20251227-121725`
*started 2025-12-27 12:17 • 500 epochs • 14h26m • 2964397.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1758 → 0.0030 (min 0.0029 @ ep 478)
  - `Loss/MSE_val`: 0.0091 → 0.0152 (min 0.0069 @ ep 18)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5498 (min 0.4925, max 0.9804)
  - `Validation/MS-SSIM`: last 0.7956 (min 0.7942, max 0.8626)
  - `Validation/PSNR`: last 26.8820 (min 26.8529, max 28.9533)

**Regional loss (final):**
  - `regional/background_loss`: 0.0042
  - `regional/large`: 0.0151
  - `regional/medium`: 0.0210
  - `regional/small`: 0.0142
  - `regional/tiny`: 0.0099
  - `regional/tumor_bg_ratio`: 3.6366
  - `regional/tumor_loss`: 0.0153

**Training meta:**
  - `training/grad_norm_avg`: last 0.0265, max 3.2704 @ ep 0
  - `training/grad_norm_max`: last 0.0882, max 10.8744 @ ep 0

### exp9

**exp9 (2D sweep)** — 12 runs exploring ScoreAug-era 2D variants.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp9_6_rflow_128_20260109-213406` — 0.0065
  2. 🥈 `exp9_5_rflow_128_20251227-195321` — 0.0066
  3.  `exp9_5_rflow_128_20251222-151019` — 0.0066
  4.  `exp9_7_rflow_128_20260109-213431` — 0.0066
  5.  `exp9_1_rflow_128_20251221-215945` — 0.0066
  6.  `exp9_2_rflow_128_20251221-215945` — 0.0066
  7.  `exp9_1_rflow_128_20251227-150556` — 0.0066
  8.  `exp9_2_rflow_128_20251227-152847` — 0.0067
  9.  `exp9_4_rflow_128_20251227-195226` — 0.0067
  10.  `exp9_4_rflow_128_20251222-145550` — 0.0067
  11.  `exp9_3_rflow_128_20251222-000454` — 0.0068
  12.  `exp9_3_rflow_128_20251227-190557` — 0.0070

#### `exp9_1_rflow_128_20251221-215945`
*started 2025-12-21 21:59 • 500 epochs • 13h34m • 16622812.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1711 → 0.0049 (min 0.0048 @ ep 452)
  - `Loss/MSE_val`: 0.0100 → 0.0089 (min 0.0066 @ ep 95)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5472 (min 0.5023, max 1.2422)
  - `Validation/MS-SSIM`: last 0.8409 (min 0.7850, max 0.8635)
  - `Validation/PSNR`: last 28.2797 (min 27.4429, max 29.0767)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0190, max 3.2960 @ ep 0
  - `training/grad_norm_max`: last 0.0636, max 10.9441 @ ep 0

#### `exp9_2_rflow_128_20251221-215945`
*started 2025-12-21 21:59 • 500 epochs • 15h10m • 16622812.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1751 → 0.0049 (min 0.0047 @ ep 435)
  - `Loss/MSE_val`: 0.0109 → 0.0095 (min 0.0066 @ ep 78)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5239 (min 0.5014, max 1.2187)
  - `Validation/MS-SSIM`: last 0.8349 (min 0.7858, max 0.8638)
  - `Validation/PSNR`: last 28.0354 (min 27.2640, max 29.0980)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0243, max 3.4093 @ ep 0
  - `training/grad_norm_max`: last 0.1210, max 12.0017 @ ep 0

#### `exp9_3_rflow_128_20251222-000454`
*started 2025-12-22 00:04 • 500 epochs • 15h02m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1698 → 0.0051 (min 0.0051 @ ep 493)
  - `Loss/MSE_val`: 0.0100 → 0.0078 (min 0.0068 @ ep 164)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5581 (min 0.5404, max 1.1949)
  - `Validation/MS-SSIM`: last 0.8384 (min 0.7893, max 0.8536)
  - `Validation/PSNR`: last 28.4140 (min 27.4827, max 28.8975)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0188, max 3.2479 @ ep 0
  - `training/grad_norm_max`: last 0.0735, max 10.8537 @ ep 0

#### `exp9_4_rflow_128_20251222-145550`
*started 2025-12-22 14:55 • 500 epochs • 15h18m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1642 → 0.0049 (min 0.0048 @ ep 477)
  - `Loss/MSE_val`: 0.0113 → 0.0090 (min 0.0067 @ ep 110)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5436 (min 0.5071, max 1.6792)
  - `Validation/MS-SSIM`: last 0.8410 (min 0.7611, max 0.8639)
  - `Validation/PSNR`: last 28.2132 (min 26.6521, max 29.0592)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0202, max 3.2625 @ ep 0
  - `training/grad_norm_max`: last 0.1594, max 10.9321 @ ep 0

#### `exp9_5_rflow_128_20251222-151019`
*started 2025-12-22 15:10 • 500 epochs • 12h27m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1454 → 0.0051 (min 0.0050 @ ep 481)
  - `Loss/MSE_val`: 0.0110 → 0.0076 (min 0.0066 @ ep 216)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5185 (min 0.5074, max 1.2566)
  - `Validation/MS-SSIM`: last 0.8513 (min 0.7849, max 0.8643)
  - `Validation/PSNR`: last 28.5814 (min 27.3396, max 29.0838)

**Training meta:**
  - `training/grad_norm_avg`: last 0.0175, max 3.0592 @ ep 0
  - `training/grad_norm_max`: last 0.0734, max 10.8699 @ ep 0

#### `exp9_1_rflow_128_20251227-150556`
*started 2025-12-27 15:05 • 500 epochs • 15h50m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1651 → 0.0050 (min 0.0048 @ ep 488)
  - `Loss/MSE_val`: 0.0111 → 0.0089 (min 0.0066 @ ep 114)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5319 (min 0.5056, max 1.6126)
  - `Validation/MS-SSIM`: last 0.8413 (min 0.7726, max 0.8636)
  - `Validation/PSNR`: last 28.2426 (min 27.0607, max 29.0784)

**Regional loss (final):**
  - `regional/background_loss`: 0.0025
  - `regional/large`: 0.0126
  - `regional/medium`: 0.0196
  - `regional/small`: 0.0128
  - `regional/tiny`: 0.0086
  - `regional/tumor_bg_ratio`: 5.5076
  - `regional/tumor_loss`: 0.0137

**Training meta:**
  - `training/grad_norm_avg`: last 0.0197, max 3.2211 @ ep 0
  - `training/grad_norm_max`: last 0.0704, max 10.9816 @ ep 0

#### `exp9_2_rflow_128_20251227-152847`
*started 2025-12-27 15:28 • 500 epochs • 15h46m • 16622917.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1720 → 0.0048 (min 0.0047 @ ep 455)
  - `Loss/MSE_val`: 0.0109 → 0.0094 (min 0.0067 @ ep 101)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5186 (min 0.4996, max 1.2526)
  - `Validation/MS-SSIM`: last 0.8384 (min 0.7895, max 0.8636)
  - `Validation/PSNR`: last 28.1558 (min 27.5011, max 29.0496)

**Regional loss (final):**
  - `regional/background_loss`: 0.0026
  - `regional/large`: 0.0135
  - `regional/medium`: 0.0195
  - `regional/small`: 0.0128
  - `regional/tiny`: 0.0088
  - `regional/tumor_bg_ratio`: 5.3032
  - `regional/tumor_loss`: 0.0139

**Training meta:**
  - `training/grad_norm_avg`: last 0.0258, max 3.4598 @ ep 0
  - `training/grad_norm_max`: last 0.1015, max 12.3697 @ ep 0

#### `exp9_3_rflow_128_20251227-190557`
*started 2025-12-27 19:05 • 500 epochs • 13h50m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1637 → 0.0062 (min 0.0060 @ ep 478)
  - `Loss/MSE_val`: 0.0114 → 0.0074 (min 0.0070 @ ep 446)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5926 (min 0.5718, max 1.3622)
  - `Validation/MS-SSIM`: last 0.8277 (min 0.7689, max 0.8346)
  - `Validation/PSNR`: last 28.2312 (min 26.9617, max 28.4010)

**Regional loss (final):**
  - `regional/background_loss`: 0.0019
  - `regional/large`: 0.0113
  - `regional/medium`: 0.0192
  - `regional/small`: 0.0129
  - `regional/tiny`: 0.0088
  - `regional/tumor_bg_ratio`: 6.8969
  - `regional/tumor_loss`: 0.0133

**Training meta:**
  - `training/grad_norm_avg`: last 0.0192, max 3.2726 @ ep 0
  - `training/grad_norm_max`: last 0.0935, max 10.8502 @ ep 0

#### `exp9_4_rflow_128_20251227-195226`
*started 2025-12-27 19:52 • 500 epochs • 14h07m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1636 → 0.0048 (min 0.0048 @ ep 484)
  - `Loss/MSE_val`: 0.0111 → 0.0087 (min 0.0067 @ ep 97)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5379 (min 0.5023, max 1.2835)
  - `Validation/MS-SSIM`: last 0.8381 (min 0.7748, max 0.8639)
  - `Validation/PSNR`: last 28.1970 (min 27.0609, max 29.0393)

**Regional loss (final):**
  - `regional/background_loss`: 0.0026
  - `regional/large`: 0.0139
  - `regional/medium`: 0.0184
  - `regional/small`: 0.0125
  - `regional/tiny`: 0.0092
  - `regional/tumor_bg_ratio`: 5.3659
  - `regional/tumor_loss`: 0.0137

**Training meta:**
  - `training/grad_norm_avg`: last 0.0206, max 3.2334 @ ep 0
  - `training/grad_norm_max`: last 0.0799, max 10.8770 @ ep 0

#### `exp9_5_rflow_128_20251227-195321`
*started 2025-12-27 19:53 • 500 epochs • 13h49m • 16622763.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1421 → 0.0051 (min 0.0050 @ ep 493)
  - `Loss/MSE_val`: 0.0112 → 0.0074 (min 0.0066 @ ep 95)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5147 (min 0.5069, max 1.4582)
  - `Validation/MS-SSIM`: last 0.8541 (min 0.7816, max 0.8644)
  - `Validation/PSNR`: last 28.6147 (min 27.2526, max 29.0945)

**Regional loss (final):**
  - `regional/background_loss`: 0.0021
  - `regional/large`: 0.0124
  - `regional/medium`: 0.0176
  - `regional/small`: 0.0113
  - `regional/tiny`: 0.0091
  - `regional/tumor_bg_ratio`: 6.0463
  - `regional/tumor_loss`: 0.0128

**Training meta:**
  - `training/grad_norm_avg`: last 0.0164, max 3.1022 @ ep 0
  - `training/grad_norm_max`: last 0.0684, max 10.8876 @ ep 0

#### `exp9_6_rflow_128_20260109-213406`
*started 2026-01-09 21:34 • 500 epochs • 19h05m • 16622763.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1255 → 0.0050 (min 0.0049 @ ep 484)
  - `Loss/MSE_val`: 0.0141 → 0.0071 (min 0.0065 @ ep 260)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0080
  - 10-19: 0.0076
  - 20-29: 0.0075
  - 30-39: 0.0072
  - 40-49: 0.0073
  - 50-59: 0.0071
  - 60-69: 0.0069
  - 70-79: 0.0068
  - 80-89: 0.0069
  - 90-99: 0.0068

**Validation quality:**
  - `Validation/LPIPS`: last 0.5773 (min 0.5547, max 1.6612)
  - `Validation/MS-SSIM`: last 0.8562 (min 0.7256, max 0.8666)
  - `Validation/MS-SSIM-3D`: last 0.9072 (min 0.8116, max 0.9095)
  - `Validation/PSNR`: last 29.1657 (min 26.1917, max 29.4837)

**Regional loss (final):**
  - `regional/background_loss`: 0.0017
  - `regional/large`: 0.0112
  - `regional/medium`: 0.0153
  - `regional/small`: 0.0118
  - `regional/tiny`: 0.0083
  - `regional/tumor_bg_ratio`: 6.7836
  - `regional/tumor_loss`: 0.0118

**Training meta:**
  - `training/grad_norm_avg`: last 0.0170, max 2.8441 @ ep 0
  - `training/grad_norm_max`: last 0.0729, max 10.7951 @ ep 0

#### `exp9_7_rflow_128_20260109-213431`
*started 2026-01-09 21:34 • 500 epochs • 17h01m • 16622763.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1612 → 0.0051 (min 0.0049 @ ep 449)
  - `Loss/MSE_val`: 0.0111 → 0.0086 (min 0.0066 @ ep 107)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0030
  - 10-19: 0.0032
  - 20-29: 0.0030
  - 30-39: 0.0032
  - 40-49: 0.0034
  - 50-59: 0.0036
  - 60-69: 0.0036
  - 70-79: 0.0036
  - 80-89: 0.0042
  - 90-99: 0.0041

**Validation quality:**
  - `Validation/LPIPS`: last 0.5421 (min 0.4948, max 1.2098)
  - `Validation/MS-SSIM`: last 0.8425 (min 0.7876, max 0.8633)
  - `Validation/MS-SSIM-3D`: last 0.9060 (min 0.8325, max 0.9081)
  - `Validation/PSNR`: last 28.7645 (min 27.3113, max 29.4115)

**Regional loss (final):**
  - `regional/background_loss`: 0.0022
  - `regional/large`: 0.0135
  - `regional/medium`: 0.0197
  - `regional/small`: 0.0127
  - `regional/tiny`: 0.0090
  - `regional/tumor_bg_ratio`: 6.4447
  - `regional/tumor_loss`: 0.0140

**Training meta:**
  - `training/grad_norm_avg`: last 0.0223, max 3.2002 @ ep 0
  - `training/grad_norm_max`: last 0.0799, max 10.8293 @ ep 0

### exp11

**exp11** — 2D restart/phase test.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp11_1_rflow_128_20251228-071134` — 0.0068
  2. 🥈 `exp11_2_rflow_128_20251228-072835` — 0.0069

#### `exp11_1_rflow_128_20251228-071134`
*started 2025-12-28 07:11 • 500 epochs • 29h17m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 1.0062 → 0.0041 (min 0.0041 @ ep 485)
  - `Loss/MSE_val`: 1.0083 → 0.0102 (min 0.0068 @ ep 75)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5769 (min 0.5600, max 1.8846)
  - `Validation/MS-SSIM`: last 0.8259 (min 0.2321, max 0.8607)
  - `Validation/PSNR`: last 27.8408 (min 9.4690, max 28.9879)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0140
  - `regional/medium`: 0.0204
  - `regional/small`: 0.0132
  - `regional/tiny`: 0.0090
  - `regional/tumor_bg_ratio`: 4.8119
  - `regional/tumor_loss`: 0.0144

**Training meta:**
  - `training/grad_norm_avg`: last 0.0092, max 0.3864 @ ep 0
  - `training/grad_norm_max`: last 0.0261, max 15.3761 @ ep 2

#### `exp11_2_rflow_128_20251228-072835`
*started 2025-12-28 07:28 • 500 epochs • 30h37m • 3964808.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2021 → 0.0038 (min 0.0037 @ ep 495)
  - `Loss/MSE_val`: 0.0119 → 0.0115 (min 0.0069 @ ep 48)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5344 (min 0.4909, max 1.3924)
  - `Validation/MS-SSIM`: last 0.8193 (min 0.7637, max 0.8628)
  - `Validation/PSNR`: last 27.6182 (min 26.4736, max 29.0174)

**Regional loss (final):**
  - `regional/background_loss`: 0.0032
  - `regional/large`: 0.0149
  - `regional/medium`: 0.0203
  - `regional/small`: 0.0141
  - `regional/tiny`: 0.0089
  - `regional/tumor_bg_ratio`: 4.6110
  - `regional/tumor_loss`: 0.0148

**Training meta:**
  - `training/grad_norm_avg`: last 0.0238, max 4.1379 @ ep 0
  - `training/grad_norm_max`: last 0.0968, max 10.9587 @ ep 0

### exp12

**exp12 (2D rflow sweep)** — 7 variants with rflow configuration tweaks.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp12_2b_rflow_128_20260118-211448` — 0.0292
  2. 🥈 `exp12_1b_rflow_128_20260118-211433` — 0.0339
  3.  `exp12_1c_rflow_256_20260121-023447` — 0.0374
  4.  `exp12_2c_rflow_256_20260121-023456` — 0.0408

#### `exp12_1_rflow_128_20251229-005658`
*started 2025-12-29 00:56 • 500 epochs • 35h41m • 3664966.8 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2398 → 0.0038 (min 0.0036 @ ep 447)
  - `Loss/MSE_val`: 0.0192 → 0.0120 (min 0.0070 @ ep 50)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5262 (min 0.5020, max 1.4899)
  - `Validation/MS-SSIM`: last 0.8222 (min 0.6961, max 0.8633)
  - `Validation/PSNR`: last 27.6282 (min 25.1455, max 29.0391)

**Regional loss (final):**
  - `regional/background_loss`: 0.0032
  - `regional/large`: 0.0153
  - `regional/medium`: 0.0204
  - `regional/small`: 0.0138
  - `regional/tiny`: 0.0092
  - `regional/tumor_bg_ratio`: 4.6184
  - `regional/tumor_loss`: 0.0149

**Training meta:**
  - `training/grad_norm_avg`: last 0.0214, max 1.4701 @ ep 0
  - `training/grad_norm_max`: last 0.0948, max 5.9976 @ ep 0

#### `exp12_2_rflow_128_20251229-005658`
*started 2025-12-29 00:56 • 500 epochs • 69h19m • 14653303.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1573 → 0.0029 (min 0.0028 @ ep 497)
  - `Loss/MSE_val`: 0.0162 → 0.0133 (min 0.0070 @ ep 58)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5377 (min 0.4964, max 1.3371)
  - `Validation/MS-SSIM`: last 0.8119 (min 0.7324, max 0.8613)
  - `Validation/PSNR`: last 27.3430 (min 26.4363, max 28.9315)

**Regional loss (final):**
  - `regional/background_loss`: 0.0034
  - `regional/large`: 0.0136
  - `regional/medium`: 0.0216
  - `regional/small`: 0.0147
  - `regional/tiny`: 0.0097
  - `regional/tumor_bg_ratio`: 4.4129
  - `regional/tumor_loss`: 0.0152

**Training meta:**
  - `training/grad_norm_avg`: last 0.0185, max 1.3110 @ ep 0
  - `training/grad_norm_max`: last 0.1450, max 7.4135 @ ep 0

#### `exp12_3_rflow_128_20251229-005658`
*started 2025-12-29 00:56 • 3 epochs • 47m46s • 208369.1 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1266 → 0.0114 (min 0.0114 @ ep 2)
  - `Loss/MSE_val`: 0.0160 → 0.0119 (min 0.0119 @ ep 1)

**Validation quality:**
  - `Validation/LPIPS`: last 1.1839 (min 1.1368, max 1.1839)
  - `Validation/MS-SSIM`: last 0.7989 (min 0.7483, max 0.7989)
  - `Validation/PSNR`: last 27.4097 (min 26.7725, max 27.4097)

**Regional loss (final):**
  - `regional/background_loss`: 0.0024
  - `regional/large`: 0.0117
  - `regional/medium`: 0.0143
  - `regional/small`: 0.0103
  - `regional/tiny`: 0.0081
  - `regional/tumor_bg_ratio`: 4.6689
  - `regional/tumor_loss`: 0.0112

**Training meta:**
  - `training/grad_norm_avg`: last 0.8944, max 1.4219 @ ep 0
  - `training/grad_norm_max`: last 2.3760, max 8.6796 @ ep 0

#### `exp12_1b_rflow_128_20260118-211433`
*started 2026-01-18 21:14 • 500 epochs • 36h07m • 3664966.8 TFLOPs • peak VRAM 13.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2551 → 0.0038 (min 0.0037 @ ep 487)
  - `Loss/MSE_val`: 0.0214 → 0.0107 (min 0.0068 @ ep 91)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0003689
  - 0.1-0.2: 0.0008391
  - 0.2-0.3: 0.0011
  - 0.3-0.4: 0.0012
  - 0.4-0.5: 0.0015
  - 0.5-0.6: 0.0017
  - 0.6-0.7: 0.0022
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0030
  - 0.9-1.0: 0.0037

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0422, best 0.0240 @ ep 74
  - `Generation/KID_mean_train`: last 0.0430, best 0.0217 @ ep 74
  - `Generation/KID_std_val`: last 0.0052, best 0.0025 @ ep 64
  - `Generation/KID_std_train`: last 0.0055, best 0.0023 @ ep 64
  - `Generation/CMMD_val`: last 0.2359, best 0.2221 @ ep 214
  - `Generation/CMMD_train`: last 0.2421, best 0.2236 @ ep 75
  - `Generation/extended_KID_mean_val`: last 0.0382, best 0.0339 @ ep 319
  - `Generation/extended_KID_mean_train`: last 0.0368, best 0.0321 @ ep 79
  - `Generation/extended_CMMD_val`: last 0.2093, best 0.2003 @ ep 399
  - `Generation/extended_CMMD_train`: last 0.2179, best 0.2084 @ ep 319

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.8335 (min 0.8284, max 1.7272)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9026 (min 0.7211, max 0.9076)
  - `Validation/MS-SSIM_bravo`: last 0.8265 (min 0.6773, max 0.8622)
  - `Validation/PSNR_bravo`: last 27.7322 (min 25.2418, max 29.0779)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0028
  - `regional_bravo/large`: 0.0135
  - `regional_bravo/medium`: 0.0171
  - `regional_bravo/small`: 0.0125
  - `regional_bravo/tiny`: 0.0087
  - `regional_bravo/tumor_bg_ratio`: 4.6268
  - `regional_bravo/tumor_loss`: 0.0131

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0134, max 1.3518 @ ep 0
  - `training/grad_norm_max`: last 0.0380, max 5.6240 @ ep 0

#### `exp12_2b_rflow_128_20260118-211448`
*started 2026-01-18 21:14 • 410 epochs • 71h47m • 12015708.0 TFLOPs • peak VRAM 27.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1638 → 0.0029 (min 0.0029 @ ep 406)
  - `Loss/MSE_val`: 0.0149 → 0.0115 (min 0.0067 @ ep 64)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0004135
  - 0.1-0.2: 0.0009339
  - 0.2-0.3: 0.0013
  - 0.3-0.4: 0.0015
  - 0.4-0.5: 0.0016
  - 0.5-0.6: 0.0018
  - 0.6-0.7: 0.0021
  - 0.7-0.8: 0.0025
  - 0.8-0.9: 0.0031
  - 0.9-1.0: 0.0032

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0445, best 0.0307 @ ep 177
  - `Generation/KID_mean_train`: last 0.0475, best 0.0303 @ ep 177
  - `Generation/KID_std_val`: last 0.0068, best 0.0027 @ ep 37
  - `Generation/KID_std_train`: last 0.0074, best 0.0026 @ ep 48
  - `Generation/CMMD_val`: last 0.2347, best 0.2208 @ ep 133
  - `Generation/CMMD_train`: last 0.2431, best 0.2209 @ ep 56
  - `Generation/extended_KID_mean_val`: last 0.0340, best 0.0292 @ ep 179
  - `Generation/extended_KID_mean_train`: last 0.0359, best 0.0297 @ ep 179
  - `Generation/extended_CMMD_val`: last 0.2227, best 0.1974 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.2328, best 0.2055 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.8139 (min 0.7817, max 1.6286)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9023 (min 0.7739, max 0.9067)
  - `Validation/MS-SSIM_bravo`: last 0.8121 (min 0.7353, max 0.8599)
  - `Validation/PSNR_bravo`: last 27.4338 (min 26.4939, max 29.0048)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0143
  - `regional_bravo/medium`: 0.0186
  - `regional_bravo/small`: 0.0137
  - `regional_bravo/tiny`: 0.0090
  - `regional_bravo/tumor_bg_ratio`: 4.6430
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 9.029e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0124, max 1.2065 @ ep 0
  - `training/grad_norm_max`: last 0.0425, max 7.1172 @ ep 0

#### `exp12_1c_rflow_256_20260121-023447`
*started 2026-01-21 02:34 • 500 epochs • 63h59m • 3675713.8 TFLOPs • peak VRAM 15.1 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.3138 → 0.0037 (min 0.0037 @ ep 484)
  - `Loss/MSE_val`: 0.0249 → 0.0083 (min 0.0056 @ ep 86)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0370
  - 0.1-0.2: 0.0216
  - 0.2-0.3: 0.0151
  - 0.3-0.4: 0.0100
  - 0.4-0.5: 0.0071
  - 0.5-0.6: 0.0051
  - 0.6-0.7: 0.0040
  - 0.7-0.8: 0.0036
  - 0.8-0.9: 0.0034
  - 0.9-1.0: 0.0038

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0465, best 0.0346 @ ep 132
  - `Generation/KID_mean_train`: last 0.0442, best 0.0299 @ ep 68
  - `Generation/KID_std_val`: last 0.0045, best 0.0029 @ ep 95
  - `Generation/KID_std_train`: last 0.0047, best 0.0023 @ ep 68
  - `Generation/CMMD_val`: last 0.2150, best 0.2022 @ ep 387
  - `Generation/CMMD_train`: last 0.2205, best 0.2077 @ ep 387
  - `Generation/extended_KID_mean_val`: last 0.0444, best 0.0374 @ ep 299
  - `Generation/extended_KID_mean_train`: last 0.0477, best 0.0381 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.1912, best 0.1865 @ ep 474
  - `Generation/extended_CMMD_train`: last 0.2020, best 0.1952 @ ep 299

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6493 (min 0.6311, max 1.7688)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9070 (min 0.6833, max 0.9239)
  - `Validation/MS-SSIM_bravo`: last 0.8732 (min 0.6828, max 0.9025)
  - `Validation/PSNR_bravo`: last 28.3562 (min 24.0934, max 29.6836)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.7565
  - `Generation_Diversity/extended_MSSSIM`: 0.3539
  - `Generation_Diversity/LPIPS`: 0.7458
  - `Generation_Diversity/MSSSIM`: 0.3415

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0022
  - `regional_bravo/large`: 0.0148
  - `regional_bravo/medium`: 0.0208
  - `regional_bravo/small`: 0.0169
  - `regional_bravo/tiny`: 0.0100
  - `regional_bravo/tumor_bg_ratio`: 7.2258
  - `regional_bravo/tumor_loss`: 0.0160

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0079, max 1.5114 @ ep 0
  - `training/grad_norm_max`: last 0.0235, max 7.2712 @ ep 1

#### `exp12_2c_rflow_256_20260121-023456`
*started 2026-01-21 02:34 • 500 epochs • 93h33m • 14691498.0 TFLOPs • peak VRAM 30.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2066 → 0.0025 (min 0.0025 @ ep 463)
  - `Loss/MSE_val`: 0.0168 → 0.0117 (min 0.0057 @ ep 58)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0752
  - 0.1-0.2: 0.0443
  - 0.2-0.3: 0.0202
  - 0.3-0.4: 0.0123
  - 0.4-0.5: 0.0080
  - 0.5-0.6: 0.0054
  - 0.6-0.7: 0.0043
  - 0.7-0.8: 0.0037
  - 0.8-0.9: 0.0036
  - 0.9-1.0: 0.0038

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0567, best 0.0386 @ ep 66
  - `Generation/KID_mean_train`: last 0.0588, best 0.0334 @ ep 66
  - `Generation/KID_std_val`: last 0.0052, best 0.0030 @ ep 53
  - `Generation/KID_std_train`: last 0.0080, best 0.0028 @ ep 66
  - `Generation/CMMD_val`: last 0.2002, best 0.1871 @ ep 178
  - `Generation/CMMD_train`: last 0.2151, best 0.1931 @ ep 178
  - `Generation/extended_KID_mean_val`: last 0.0443, best 0.0408 @ ep 274
  - `Generation/extended_KID_mean_train`: last 0.0450, best 0.0415 @ ep 299
  - `Generation/extended_CMMD_val`: last 0.1948, best 0.1782 @ ep 199
  - `Generation/extended_CMMD_train`: last 0.2100, best 0.1893 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5831 (min 0.5713, max 1.7598)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9008 (min 0.7473, max 0.9223)
  - `Validation/MS-SSIM_bravo`: last 0.8603 (min 0.7630, max 0.9002)
  - `Validation/PSNR_bravo`: last 27.8488 (min 25.9432, max 29.6219)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.7865
  - `Generation_Diversity/extended_MSSSIM`: 0.3679
  - `Generation_Diversity/LPIPS`: 0.8049
  - `Generation_Diversity/MSSSIM`: 0.3597

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0178
  - `regional_bravo/medium`: 0.0192
  - `regional_bravo/small`: 0.0177
  - `regional_bravo/tiny`: 0.0111
  - `regional_bravo/tumor_bg_ratio`: 6.3499
  - `regional_bravo/tumor_loss`: 0.0166

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0076, max 1.1155 @ ep 0
  - `training/grad_norm_max`: last 0.0199, max 5.3373 @ ep 0

### exp13

**exp13** — 2D variant.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp13_1_ema_simple_rflow_128_20260109-213446` — 0.0070
  2. 🥈 `exp13_2_ema_slow_rflow_128_20260109-213501` — 0.0070

#### `exp13_1_ema_simple_rflow_128_20260109-213446`
*started 2026-01-09 21:34 • 500 epochs • 23h28m • 3964808.8 TFLOPs • peak VRAM 35.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1788 → 0.0060 (min 0.0058 @ ep 477)
  - `Loss/MSE_val`: 0.0728 → 0.0078 (min 0.0070 @ ep 182)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001666
  - 10-19: 0.0003135
  - 20-29: 0.0004605
  - 30-39: 0.0006734
  - 40-49: 0.0009098
  - 50-59: 0.0013
  - 60-69: 0.0018
  - 70-79: 0.0025
  - 80-89: 0.0034
  - 90-99: 0.0042

**Validation quality:**
  - `Validation/LPIPS`: last 0.5802 (min 0.5489, max 1.8504)
  - `Validation/MS-SSIM`: last 0.8113 (min 0.5260, max 0.8338)
  - `Validation/MS-SSIM-3D`: last 0.8904 (min 0.6383, max 0.8931)
  - `Validation/PSNR`: last 28.1378 (min 19.3841, max 28.7070)

**Regional loss (final):**
  - `regional/background_loss`: 0.0023
  - `regional/large`: 0.0161
  - `regional/medium`: 0.0284
  - `regional/small`: 0.0150
  - `regional/tiny`: 0.0111
  - `regional/tumor_bg_ratio`: 7.7648
  - `regional/tumor_loss`: 0.0181

**Training meta:**
  - `training/grad_norm_avg`: last 0.0276, max 3.2946 @ ep 0
  - `training/grad_norm_max`: last 0.0918, max 10.9388 @ ep 0

#### `exp13_2_ema_slow_rflow_128_20260109-213501`
*started 2026-01-09 21:35 • 500 epochs • 18h00m • 3964808.8 TFLOPs • peak VRAM 35.5 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1791 → 0.0060 (min 0.0059 @ ep 469)
  - `Loss/MSE_val`: 0.0723 → 0.0077 (min 0.0070 @ ep 231)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001673
  - 10-19: 0.0003149
  - 20-29: 0.0004637
  - 30-39: 0.0006817
  - 40-49: 0.0009339
  - 50-59: 0.0013
  - 60-69: 0.0018
  - 70-79: 0.0026
  - 80-89: 0.0035
  - 90-99: 0.0042

**Validation quality:**
  - `Validation/LPIPS`: last 0.5694 (min 0.5494, max 1.8413)
  - `Validation/MS-SSIM`: last 0.8238 (min 0.5376, max 0.8354)
  - `Validation/MS-SSIM-3D`: last 0.8909 (min 0.6460, max 0.8926)
  - `Validation/PSNR`: last 28.3955 (min 19.3634, max 28.7102)

**Regional loss (final):**
  - `regional/background_loss`: 0.0021
  - `regional/large`: 0.0140
  - `regional/medium`: 0.0290
  - `regional/small`: 0.0173
  - `regional/tiny`: 0.0106
  - `regional/tumor_bg_ratio`: 8.7354
  - `regional/tumor_loss`: 0.0182

**Training meta:**
  - `training/grad_norm_avg`: last 0.0266, max 3.3511 @ ep 0
  - `training/grad_norm_max`: last 0.0761, max 10.9819 @ ep 0

### exp14

**exp14 (2D sweep)** — 3 variants.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp14_2_rflow_128_20260109-213543` — 0.0069
  2. 🥈 `exp14_1_rflow_128_20260109-213501` — 0.0069
  3.  `exp14_3_rflow_128_20260109-213545` — 0.0070

#### `exp14_1_rflow_128_20260109-213501`
*started 2026-01-09 21:35 • 500 epochs • 41h42m • 3664966.8 TFLOPs • peak VRAM 13.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2635 → 0.0048 (min 0.0047 @ ep 485)
  - `Loss/MSE_val`: 0.0218 → 0.0096 (min 0.0069 @ ep 112)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.000182
  - 10-19: 0.0003088
  - 20-29: 0.0004156
  - 30-39: 0.000516
  - 40-49: 0.0006825
  - 50-59: 0.0008594
  - 60-69: 0.0011
  - 70-79: 0.0015
  - 80-89: 0.0020
  - 90-99: 0.0025

**Validation quality:**
  - `Validation/LPIPS`: last 0.5300 (min 0.5029, max 1.7379)
  - `Validation/MS-SSIM`: last 0.8282 (min 0.6827, max 0.8636)
  - `Validation/MS-SSIM-3D`: last 0.9028 (min 0.7223, max 0.9079)
  - `Validation/PSNR`: last 28.3245 (min 25.7055, max 29.3638)

**Regional loss (final):**
  - `regional/background_loss`: 0.0028
  - `regional/large`: 0.0172
  - `regional/medium`: 0.0153
  - `regional/small`: 0.0130
  - `regional/tiny`: 0.0085
  - `regional/tumor_bg_ratio`: 4.9166
  - `regional/tumor_loss`: 0.0136

**Training meta:**
  - `training/grad_norm_avg`: last 0.0175, max 1.2151 @ ep 0
  - `training/grad_norm_max`: last 0.0484, max 5.7345 @ ep 0

#### `exp14_2_rflow_128_20260109-213543`
*started 2026-01-09 21:35 • 500 epochs • 41h06m • 3664966.8 TFLOPs • peak VRAM 13.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2745 → 0.0051 (min 0.0050 @ ep 476)
  - `Loss/MSE_val`: 0.0215 → 0.0090 (min 0.0069 @ ep 83)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001712
  - 10-19: 0.0003337
  - 20-29: 0.0004489
  - 30-39: 0.0005821
  - 40-49: 0.0007156
  - 50-59: 0.0009135
  - 60-69: 0.0012
  - 70-79: 0.0015
  - 80-89: 0.0020
  - 90-99: 0.0025

**Validation quality:**
  - `Validation/LPIPS`: last 0.5284 (min 0.5050, max 1.7538)
  - `Validation/MS-SSIM`: last 0.8308 (min 0.6730, max 0.8625)
  - `Validation/MS-SSIM-3D`: last 0.9044 (min 0.7129, max 0.9075)
  - `Validation/PSNR`: last 28.3837 (min 25.6139, max 29.3424)

**Regional loss (final):**
  - `regional/background_loss`: 0.0028
  - `regional/large`: 0.0156
  - `regional/medium`: 0.0177
  - `regional/small`: 0.0127
  - `regional/tiny`: 0.0089
  - `regional/tumor_bg_ratio`: 4.9561
  - `regional/tumor_loss`: 0.0139

**Training meta:**
  - `training/grad_norm_avg`: last 0.0183, max 1.1920 @ ep 0
  - `training/grad_norm_max`: last 0.0485, max 5.6268 @ ep 0

#### `exp14_3_rflow_128_20260109-213545`
*started 2026-01-09 21:35 • 500 epochs • 41h36m • 3664966.8 TFLOPs • peak VRAM 13.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2772 → 0.0054 (min 0.0052 @ ep 439)
  - `Loss/MSE_val`: 0.0225 → 0.0085 (min 0.0070 @ ep 114)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001989
  - 10-19: 0.000349
  - 20-29: 0.0004834
  - 30-39: 0.000636
  - 40-49: 0.0007575
  - 50-59: 0.0009359
  - 60-69: 0.0012
  - 70-79: 0.0015
  - 80-89: 0.0021
  - 90-99: 0.0023

**Validation quality:**
  - `Validation/LPIPS`: last 0.5288 (min 0.5126, max 1.7983)
  - `Validation/MS-SSIM`: last 0.8350 (min 0.6716, max 0.8635)
  - `Validation/MS-SSIM-3D`: last 0.9062 (min 0.7160, max 0.9075)
  - `Validation/PSNR`: last 28.4983 (min 25.3084, max 29.3617)

**Regional loss (final):**
  - `regional/background_loss`: 0.0027
  - `regional/large`: 0.0196
  - `regional/medium`: 0.0193
  - `regional/small`: 0.0141
  - `regional/tiny`: 0.0089
  - `regional/tumor_bg_ratio`: 5.6976
  - `regional/tumor_loss`: 0.0156

**Training meta:**
  - `training/grad_norm_avg`: last 0.0190, max 1.1731 @ ep 0
  - `training/grad_norm_max`: last 0.0725, max 5.3250 @ ep 0

### exp15

**exp15 (2D weight decay / architecture)** — weight decay and other
regularizers at 128. Per memory: WD 0.05 → FID 43.45 at 128, best 2D result
in that configuration.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp15_3_wd01_rflow_128_20260111-015029` — 0.0068
  2. 🥈 `exp15_1_wd001_rflow_128_20260111-015002` — 0.0068
  3.  `exp15_2_wd005_rflow_128_20260111-015031` — 0.0068

#### `exp15_1_wd001_rflow_128_20260111-015002`
*started 2026-01-11 01:50 • 500 epochs • 19h54m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1783 → 0.0037 (min 0.0037 @ ep 481)
  - `Loss/MSE_val`: 0.0097 → 0.0125 (min 0.0068 @ ep 51)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001282
  - 10-19: 0.0001911
  - 20-29: 0.0002838
  - 30-39: 0.0004171
  - 40-49: 0.0005855
  - 50-59: 0.0007909
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0018
  - 90-99: 0.0023

**Validation quality:**
  - `Validation/LPIPS`: last 0.5481 (min 0.5023, max 0.9595)
  - `Validation/MS-SSIM`: last 0.8138 (min 0.7977, max 0.8597)
  - `Validation/MS-SSIM-3D`: last 0.9044 (min 0.8297, max 0.9066)
  - `Validation/PSNR`: last 28.0537 (min 28.0515, max 29.3248)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0148
  - `regional/medium`: 0.0211
  - `regional/small`: 0.0139
  - `regional/tiny`: 0.0093
  - `regional/tumor_bg_ratio`: 5.1444
  - `regional/tumor_loss`: 0.0150

**Training meta:**
  - `training/grad_norm_avg`: last 0.0327, max 3.3381 @ ep 0
  - `training/grad_norm_max`: last 0.1161, max 10.9703 @ ep 0

#### `exp15_3_wd01_rflow_128_20260111-015029`
*started 2026-01-11 01:50 • 500 epochs • 19h56m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1780 → 0.0040 (min 0.0038 @ ep 483)
  - `Loss/MSE_val`: 0.0104 → 0.0125 (min 0.0068 @ ep 53)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.000136
  - 10-19: 0.0002138
  - 20-29: 0.0003125
  - 30-39: 0.0004415
  - 40-49: 0.0006098
  - 50-59: 0.000818
  - 60-69: 0.0010
  - 70-79: 0.0013
  - 80-89: 0.0021
  - 90-99: 0.0023

**Validation quality:**
  - `Validation/LPIPS`: last 0.5438 (min 0.5086, max 0.8631)
  - `Validation/MS-SSIM`: last 0.8224 (min 0.7976, max 0.8596)
  - `Validation/MS-SSIM-3D`: last 0.9050 (min 0.8349, max 0.9066)
  - `Validation/PSNR`: last 28.2811 (min 27.9401, max 29.3258)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0129
  - `regional/medium`: 0.0208
  - `regional/small`: 0.0131
  - `regional/tiny`: 0.0091
  - `regional/tumor_bg_ratio`: 4.9832
  - `regional/tumor_loss`: 0.0143

**Training meta:**
  - `training/grad_norm_avg`: last 0.0384, max 3.3451 @ ep 0
  - `training/grad_norm_max`: last 0.1221, max 10.9428 @ ep 0

#### `exp15_2_wd005_rflow_128_20260111-015031`
*started 2026-01-11 01:50 • 500 epochs • 19h46m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1792 → 0.0037 (min 0.0037 @ ep 493)
  - `Loss/MSE_val`: 0.0095 → 0.0124 (min 0.0068 @ ep 38)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001283
  - 10-19: 0.0002008
  - 20-29: 0.0002828
  - 30-39: 0.0004144
  - 40-49: 0.0006024
  - 50-59: 0.0007904
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0020
  - 90-99: 0.0024

**Validation quality:**
  - `Validation/LPIPS`: last 0.5422 (min 0.5050, max 1.0160)
  - `Validation/MS-SSIM`: last 0.8178 (min 0.8024, max 0.8602)
  - `Validation/MS-SSIM-3D`: last 0.9037 (min 0.8409, max 0.9072)
  - `Validation/PSNR`: last 28.2081 (min 27.8210, max 29.3181)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0135
  - `regional/medium`: 0.0228
  - `regional/small`: 0.0149
  - `regional/tiny`: 0.0095
  - `regional/tumor_bg_ratio`: 5.2003
  - `regional/tumor_loss`: 0.0155

**Training meta:**
  - `training/grad_norm_avg`: last 0.0223, max 3.3621 @ ep 0
  - `training/grad_norm_max`: last 0.0862, max 10.9150 @ ep 0

### exp16

**exp16 (2D batch-size sweep)** — 4 variants including `bs32` which came
back empty (crash). These test whether larger batch helps at 128.

**Family ranking by `Loss/MSE_val` (val MSE ↓):**
  1. 🥇 `exp16_bs8_rflow_128_20260111-190651` — 0.0066
  2. 🥈 `exp16_bs24_rflow_128_20260111-190650` — 0.0067
  3.  `exp16_bs4_rflow_128_20260111-190651` — 0.0068

#### `exp16_bs24_rflow_128_20260111-190650`
*started 2026-01-11 19:06 • 500 epochs • 8h47m • 3967069.2 TFLOPs • peak VRAM 37.0 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2633 → 0.0038 (min 0.0037 @ ep 497)
  - `Loss/MSE_val`: 0.0104 → 0.0124 (min 0.0067 @ ep 52)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001305
  - 10-19: 0.0001928
  - 20-29: 0.0002854
  - 30-39: 0.0004181
  - 40-49: 0.0006229
  - 50-59: 0.0008118
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0020
  - 90-99: 0.0022

**Validation quality:**
  - `Validation/LPIPS`: last 0.6012 (min 0.5885, max 1.3006)
  - `Validation/MS-SSIM`: last 0.8189 (min 0.7916, max 0.8605)
  - `Validation/MS-SSIM-3D`: last 0.9043 (min 0.8358, max 0.9066)
  - `Validation/PSNR`: last 28.1691 (min 27.7939, max 29.3735)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0145
  - `regional/medium`: 0.0214
  - `regional/small`: 0.0133
  - `regional/tiny`: 0.0088
  - `regional/tumor_bg_ratio`: 5.0601
  - `regional/tumor_loss`: 0.0148

**Training meta:**
  - `training/grad_norm_avg`: last 0.0293, max 4.8468 @ ep 0
  - `training/grad_norm_max`: last 0.1322, max 10.8214 @ ep 0

#### `exp16_bs4_rflow_128_20260111-190651`
*started 2026-01-11 19:06 • 500 epochs • 29h01m • 3961418.2 TFLOPs • peak VRAM 33.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0525 → 0.0038 (min 0.0037 @ ep 471)
  - `Loss/MSE_val`: 0.0089 → 0.0131 (min 0.0068 @ ep 54)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001304
  - 10-19: 0.0001916
  - 20-29: 0.0002802
  - 30-39: 0.0004217
  - 40-49: 0.0006099
  - 50-59: 0.0008089
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0019
  - 90-99: 0.0022

**Validation quality:**
  - `Validation/LPIPS`: last 0.5343 (min 0.5031, max 0.9616)
  - `Validation/MS-SSIM`: last 0.8145 (min 0.8063, max 0.8604)
  - `Validation/MS-SSIM-3D`: last 0.9041 (min 0.8497, max 0.9063)
  - `Validation/PSNR`: last 28.3860 (min 28.2050, max 29.5447)

**Regional loss (final):**
  - `regional/background_loss`: 0.0031
  - `regional/large`: 0.0148
  - `regional/medium`: 0.0230
  - `regional/small`: 0.0140
  - `regional/tiny`: 0.0093
  - `regional/tumor_bg_ratio`: 5.0271
  - `regional/tumor_loss`: 0.0156

**Training meta:**
  - `training/grad_norm_avg`: last 0.0274, max 1.1149 @ ep 0
  - `training/grad_norm_max`: last 0.0958, max 11.0471 @ ep 0

#### `exp16_bs8_rflow_128_20260111-190651`
*started 2026-01-11 19:06 • 500 epochs • 22h52m • 3962548.5 TFLOPs • peak VRAM 33.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0953 → 0.0037 (min 0.0037 @ ep 493)
  - `Loss/MSE_val`: 0.0092 → 0.0128 (min 0.0066 @ ep 43)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001262
  - 10-19: 0.000192
  - 20-29: 0.0002825
  - 30-39: 0.0004201
  - 40-49: 0.0006006
  - 50-59: 0.0007992
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0019
  - 90-99: 0.0021

**Validation quality:**
  - `Validation/LPIPS`: last 0.5364 (min 0.4987, max 1.0702)
  - `Validation/MS-SSIM`: last 0.8170 (min 0.8106, max 0.8617)
  - `Validation/MS-SSIM-3D`: last 0.9047 (min 0.8460, max 0.9067)
  - `Validation/PSNR`: last 28.2901 (min 28.1671, max 29.4828)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0130
  - `regional/medium`: 0.0202
  - `regional/small`: 0.0154
  - `regional/tiny`: 0.0091
  - `regional/tumor_bg_ratio`: 4.8719
  - `regional/tumor_loss`: 0.0147

**Training meta:**
  - `training/grad_norm_avg`: last 0.0286, max 1.8773 @ ep 0
  - `training/grad_norm_max`: last 0.1005, max 10.9585 @ ep 0

#### `exp16_bs32_rflow_128_20260111-205709`
*started 2026-01-11 20:57*

**Empty run** — no scalar tags written.

### exp17

**exp17 (2D variants)** — 3 runs.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp17_3_sit_s_droppath02_rflow_128_20260112-213706` — 29.4092
  2. 🥈 `exp17_2_sit_s_droppath015_rflow_128_20260112-213706` — 29.3989
  3.  `exp17_1_sit_s_droppath01_rflow_128_20260112-213706` — 29.3806

#### `exp17_1_sit_s_droppath01_rflow_128_20260112-213706`
*started 2026-01-12 21:37 • 500 epochs • 44h13m • 3664966.8 TFLOPs • peak VRAM 12.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2448 → 0.0044 (min 0.0043 @ ep 489)
  - `Loss/MSE_val`: 0.0219 → 0.0094 (min 0.0069 @ ep 90)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001654
  - 10-19: 0.0002681
  - 20-29: 0.0003637
  - 30-39: 0.0004878
  - 40-49: 0.0006374
  - 50-59: 0.0008559
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0020
  - 90-99: 0.0025

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5251 (min 0.4967, max 1.6166)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9020 (min 0.7240, max 0.9060)
  - `Validation/MS-SSIM_bravo`: last 0.8308 (min 0.6907, max 0.8637)
  - `Validation/PSNR_bravo`: last 28.4478 (min 25.7222, max 29.3806)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0163
  - `regional_bravo/medium`: 0.0188
  - `regional_bravo/small`: 0.0135
  - `regional_bravo/tiny`: 0.0088
  - `regional_bravo/tumor_bg_ratio`: 5.5381
  - `regional_bravo/tumor_loss`: 0.0145

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0249, max 1.4251 @ ep 0
  - `training/grad_norm_max`: last 0.0726, max 6.0640 @ ep 0

#### `exp17_2_sit_s_droppath015_rflow_128_20260112-213706`
*started 2026-01-12 21:37 • 500 epochs • 43h03m • 3664966.8 TFLOPs • peak VRAM 12.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2469 → 0.0046 (min 0.0045 @ ep 476)
  - `Loss/MSE_val`: 0.0224 → 0.0091 (min 0.0069 @ ep 80)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001728
  - 10-19: 0.0002908
  - 20-29: 0.0003843
  - 30-39: 0.0005027
  - 40-49: 0.0006591
  - 50-59: 0.000844
  - 60-69: 0.0012
  - 70-79: 0.0015
  - 80-89: 0.0019
  - 90-99: 0.0023

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5299 (min 0.4974, max 1.6021)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9021 (min 0.7217, max 0.9065)
  - `Validation/MS-SSIM_bravo`: last 0.8352 (min 0.6895, max 0.8649)
  - `Validation/PSNR_bravo`: last 28.5421 (min 25.8558, max 29.3989)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0191
  - `regional_bravo/medium`: 0.0182
  - `regional_bravo/small`: 0.0136
  - `regional_bravo/tiny`: 0.0094
  - `regional_bravo/tumor_bg_ratio`: 5.9357
  - `regional_bravo/tumor_loss`: 0.0152

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0173, max 1.4850 @ ep 0
  - `training/grad_norm_max`: last 0.0665, max 5.9472 @ ep 0

#### `exp17_3_sit_s_droppath02_rflow_128_20260112-213706`
*started 2026-01-12 21:37 • 500 epochs • 43h42m • 3664966.8 TFLOPs • peak VRAM 12.2 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2447 → 0.0049 (min 0.0047 @ ep 479)
  - `Loss/MSE_val`: 0.0199 → 0.0089 (min 0.0069 @ ep 124)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001666
  - 10-19: 0.0003101
  - 20-29: 0.0004222
  - 30-39: 0.0005398
  - 40-49: 0.0006894
  - 50-59: 0.0009053
  - 60-69: 0.0012
  - 70-79: 0.0016
  - 80-89: 0.0020
  - 90-99: 0.0024

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5117 (min 0.4959, max 1.8743)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9025 (min 0.2762, max 0.9061)
  - `Validation/MS-SSIM_bravo`: last 0.8370 (min 0.2215, max 0.8639)
  - `Validation/PSNR_bravo`: last 28.6030 (min 9.1220, max 29.4092)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0024
  - `regional_bravo/large`: 0.0126
  - `regional_bravo/medium`: 0.0176
  - `regional_bravo/small`: 0.0126
  - `regional_bravo/tiny`: 0.0086
  - `regional_bravo/tumor_bg_ratio`: 5.3491
  - `regional_bravo/tumor_loss`: 0.0131

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0174, max 4065.1072 @ ep 10
  - `training/grad_norm_max`: last 0.0568, max 2.964e+06 @ ep 11

### exp18

**exp18** — single 2D run.

#### `exp18_1_unet_sda_rflow_128_20260113-204351`
*started 2026-01-13 20:43 • 500 epochs • 27h42m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1827 → 0.0057 (min 0.0055 @ ep 440)
  - `Loss/MSE_val`: 0.0103 → 0.0072 (min 0.0067 @ ep 113)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001683
  - 10-19: 0.0003379
  - 20-29: 0.0005074
  - 30-39: 0.0006649
  - 40-49: 0.0008494
  - 50-59: 0.0010
  - 60-69: 0.0013
  - 70-79: 0.0017
  - 80-89: 0.0021
  - 90-99: 0.0027

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0449, best 0.0255 @ ep 337
  - `Generation/KID_mean_train`: last 0.0482, best 0.0281 @ ep 337
  - `Generation/KID_std_val`: last 0.0047, best 0.0023 @ ep 160
  - `Generation/KID_std_train`: last 0.0057, best 0.0020 @ ep 87
  - `Generation/CMMD_val`: last 0.2178, best 0.1815 @ ep 22
  - `Generation/CMMD_train`: last 0.2220, best 0.1744 @ ep 22
  - `Generation/extended_KID_mean_val`: last 0.0405, best 0.0333 @ ep 199
  - `Generation/extended_KID_mean_train`: last 0.0432, best 0.0347 @ ep 199
  - `Generation/extended_CMMD_val`: last 0.1996, best 0.1970 @ ep 349
  - `Generation/extended_CMMD_train`: last 0.2049, best 0.2009 @ ep 199

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5122 (min 0.5015, max 1.0541)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9054 (min 0.8182, max 0.9057)
  - `Validation/MS-SSIM_bravo`: last 0.8513 (min 0.7815, max 0.8619)
  - `Validation/PSNR_bravo`: last 29.0121 (min 27.2461, max 29.3664)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0020
  - `regional_bravo/large`: 0.0140
  - `regional_bravo/medium`: 0.0190
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0095
  - `regional_bravo/tumor_bg_ratio`: 7.0705
  - `regional_bravo/tumor_loss`: 0.0140

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0803, max 3.5456 @ ep 0
  - `training/grad_norm_max`: last 0.5135, max 199.5333 @ ep 292

### exp19

**exp19** — single 2D ablation.

#### `exp19_1_unet_constant_lr_rflow_128_20260112-230233`
*started 2026-01-12 23:02 • 500 epochs • 9h31m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1784 → 0.0039 (min 0.0039 @ ep 492)
  - `Loss/MSE_val`: 0.0100 → 0.0100 (min 0.0067 @ ep 47)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001304
  - 10-19: 0.0002101
  - 20-29: 0.0002931
  - 30-39: 0.0004373
  - 40-49: 0.0006013
  - 50-59: 0.0007931
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0018
  - 90-99: 0.0024

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5901 (min 0.5819, max 1.3040)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9043 (min 0.8371, max 0.9073)
  - `Validation/MS-SSIM_bravo`: last 0.8298 (min 0.8021, max 0.8600)
  - `Validation/PSNR_bravo`: last 28.5713 (min 27.9417, max 29.3715)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0025
  - `regional_bravo/large`: 0.0139
  - `regional_bravo/medium`: 0.0195
  - `regional_bravo/small`: 0.0124
  - `regional_bravo/tiny`: 0.0086
  - `regional_bravo/tumor_bg_ratio`: 5.5295
  - `regional_bravo/tumor_loss`: 0.0138

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 0.0001

**Training meta:**
  - `training/grad_norm_avg`: last 0.0259, max 3.3202 @ ep 0
  - `training/grad_norm_max`: last 0.1042, max 10.9744 @ ep 0

### exp20

**exp20** — 2D variant.

#### `exp20_1_unet_grad_noise_rflow_128_20260112-230233`
*started 2026-01-12 23:02 • 500 epochs • 9h43m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2203 → 0.0052 (min 0.0051 @ ep 429)
  - `Loss/MSE_val`: 0.0169 → 0.0083 (min 0.0066 @ ep 119)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.000173
  - 10-19: 0.0003247
  - 20-29: 0.0004631
  - 30-39: 0.0006117
  - 40-49: 0.0007532
  - 50-59: 0.000953
  - 60-69: 0.0012
  - 70-79: 0.0015
  - 80-89: 0.0020
  - 90-99: 0.0026

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6252 (min 0.6160, max 1.5349)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9050 (min 0.7498, max 0.9070)
  - `Validation/MS-SSIM_bravo`: last 0.8397 (min 0.6562, max 0.8646)
  - `Validation/PSNR_bravo`: last 28.7606 (min 24.4641, max 29.4591)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0023
  - `regional_bravo/large`: 0.0131
  - `regional_bravo/medium`: 0.0184
  - `regional_bravo/small`: 0.0119
  - `regional_bravo/tiny`: 0.0089
  - `regional_bravo/tumor_bg_ratio`: 5.7381
  - `regional_bravo/tumor_loss`: 0.0133

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0351, max 3.6219 @ ep 0
  - `training/grad_norm_max`: last 0.1305, max 9.8039 @ ep 0

### exp21

**exp21** — curriculum training at 128.

#### `exp21_1_unet_curriculum_rflow_128_20260112-230233`
*started 2026-01-12 23:02 • 500 epochs • 9h25m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.2332 → 0.0170 (min 0.0148 @ ep 482)
  - `Loss/MSE_val`: 0.0267 → 0.0101 (min 0.0067 @ ep 68)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 5.991e-05
  - 10-19: 0.0001668
  - 20-29: 0.0002979
  - 30-39: 0.0004785
  - 40-49: 0.0006631
  - 50-59: 0.0008676
  - 60-69: 0.0011
  - 70-79: 0.0015
  - 80-89: 0.0019
  - 90-99: 0.0025

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6008 (min 0.5909, max 1.5210)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9056 (min 0.7001, max 0.9073)
  - `Validation/MS-SSIM_bravo`: last 0.8311 (min 0.5804, max 0.8596)
  - `Validation/PSNR_bravo`: last 28.4652 (min 21.1806, max 29.3641)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0126
  - `regional_bravo/medium`: 0.0219
  - `regional_bravo/small`: 0.0135
  - `regional_bravo/tiny`: 0.0090
  - `regional_bravo/tumor_bg_ratio`: 5.7186
  - `regional_bravo/tumor_loss`: 0.0146

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0370, max 3.4551 @ ep 0
  - `training/grad_norm_max`: last 0.1492, max 10.4346 @ ep 0

### exp22

**exp22** — single 2D run.

#### `exp22_1_unet_timestep_jitter_rflow_128_20260113-204425`
*started 2026-01-13 20:44 • 500 epochs • 21h37m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1827 → 0.0066 (min 0.0051 @ ep 498)
  - `Loss/MSE_val`: 0.0105 → 0.0124 (min 0.0068 @ ep 57)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 9.421e-05
  - 10-19: 0.0001882
  - 20-29: 0.0002711
  - 30-39: 0.0004215
  - 40-49: 0.000607
  - 50-59: 0.0008303
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0018
  - 90-99: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0478, best 0.0313 @ ep 24
  - `Generation/KID_mean_train`: last 0.0553, best 0.0329 @ ep 24
  - `Generation/KID_std_val`: last 0.0062, best 0.0025 @ ep 24
  - `Generation/KID_std_train`: last 0.0070, best 0.0025 @ ep 24
  - `Generation/CMMD_val`: last 0.2391, best 0.1791 @ ep 54
  - `Generation/CMMD_train`: last 0.2490, best 0.1754 @ ep 54
  - `Generation/extended_KID_mean_val`: last 0.0405, best 0.0289 @ ep 24
  - `Generation/extended_KID_mean_train`: last 0.0448, best 0.0305 @ ep 24
  - `Generation/extended_CMMD_val`: last 0.2214, best 0.1696 @ ep 49
  - `Generation/extended_CMMD_train`: last 0.2306, best 0.1740 @ ep 49

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5285 (min 0.5004, max 0.9408)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9032 (min 0.8381, max 0.9068)
  - `Validation/MS-SSIM_bravo`: last 0.8194 (min 0.8003, max 0.8596)
  - `Validation/PSNR_bravo`: last 28.2014 (min 27.7965, max 29.2602)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0132
  - `regional_bravo/medium`: 0.0191
  - `regional_bravo/small`: 0.0138
  - `regional_bravo/tiny`: 0.0094
  - `regional_bravo/tumor_bg_ratio`: 4.7106
  - `regional_bravo/tumor_loss`: 0.0141

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0474, max 3.3678 @ ep 0
  - `training/grad_norm_max`: last 0.1907, max 10.9124 @ ep 0

### exp23

**exp23 (2D noise-aug)** — 2D variant of the ScoreAug noise-aug framework.

#### `exp23_1_unet_noise_aug_rflow_128_20260113-204454`
*started 2026-01-13 20:44 • 500 epochs • 21h45m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1797 → 0.0038 (min 0.0037 @ ep 471)
  - `Loss/MSE_val`: 0.0100 → 0.0133 (min 0.0068 @ ep 30)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001246
  - 10-19: 0.0001959
  - 20-29: 0.0002806
  - 30-39: 0.000419
  - 40-49: 0.0006047
  - 50-59: 0.0008208
  - 60-69: 0.0011
  - 70-79: 0.0014
  - 80-89: 0.0019
  - 90-99: 0.0025

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0452, best 0.0319 @ ep 53
  - `Generation/KID_mean_train`: last 0.0518, best 0.0337 @ ep 53
  - `Generation/KID_std_val`: last 0.0051, best 0.0018 @ ep 39
  - `Generation/KID_std_train`: last 0.0058, best 0.0021 @ ep 38
  - `Generation/CMMD_val`: last 0.2424, best 0.1821 @ ep 48
  - `Generation/CMMD_train`: last 0.2507, best 0.1800 @ ep 48
  - `Generation/extended_KID_mean_val`: last 0.0441, best 0.0416 @ ep 399
  - `Generation/extended_KID_mean_train`: last 0.0479, best 0.0455 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.2253, best 0.1987 @ ep 124
  - `Generation/extended_CMMD_train`: last 0.2353, best 0.2036 @ ep 124

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5359 (min 0.5061, max 0.9313)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9031 (min 0.8416, max 0.9071)
  - `Validation/MS-SSIM_bravo`: last 0.8186 (min 0.7784, max 0.8608)
  - `Validation/PSNR_bravo`: last 28.2054 (min 27.4091, max 29.3355)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0029
  - `regional_bravo/large`: 0.0140
  - `regional_bravo/medium`: 0.0199
  - `regional_bravo/small`: 0.0141
  - `regional_bravo/tiny`: 0.0094
  - `regional_bravo/tumor_bg_ratio`: 4.9819
  - `regional_bravo/tumor_loss`: 0.0146

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0428, max 3.3959 @ ep 0
  - `training/grad_norm_max`: last 0.1739, max 10.9379 @ ep 0

### exp24

**exp24** — single 2D run.

#### `exp24_1_unet_feature_perturb_rflow_128_20260113-204448`
*started 2026-01-13 20:44 • 500 epochs • 9h48m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1793 → 0.0038 (min 0.0037 @ ep 482)
  - `Loss/MSE_val`: 0.0100 → 0.0133 (min 0.0067 @ ep 49)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001327
  - 10-19: 0.0001949
  - 20-29: 0.0002756
  - 30-39: 0.0004189
  - 40-49: 0.0005942
  - 50-59: 0.0008007
  - 60-69: 0.0010
  - 70-79: 0.0014
  - 80-89: 0.0020
  - 90-99: 0.0023

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0500, best 0.0342 @ ep 37
  - `Generation/KID_mean_train`: last 0.0576, best 0.0313 @ ep 37
  - `Generation/KID_std_val`: last 0.0052, best 0.0024 @ ep 33
  - `Generation/KID_std_train`: last 0.0077, best 0.0025 @ ep 78
  - `Generation/CMMD_val`: last 0.2443, best 0.1980 @ ep 25
  - `Generation/CMMD_train`: last 0.2518, best 0.1991 @ ep 25
  - `Generation/extended_KID_mean_val`: last 0.0424, best 0.0299 @ ep 249
  - `Generation/extended_KID_mean_train`: last 0.0468, best 0.0333 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.2251, best 0.1926 @ ep 74
  - `Generation/extended_CMMD_train`: last 0.2339, best 0.1978 @ ep 74

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5951 (min 0.5818, max 1.2457)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9032 (min 0.8396, max 0.9079)
  - `Validation/MS-SSIM_bravo`: last 0.8235 (min 0.7977, max 0.8621)
  - `Validation/PSNR_bravo`: last 28.3350 (min 27.9927, max 29.4063)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0029
  - `regional_bravo/large`: 0.0138
  - `regional_bravo/medium`: 0.0219
  - `regional_bravo/small`: 0.0141
  - `regional_bravo/tiny`: 0.0089
  - `regional_bravo/tumor_bg_ratio`: 5.1840
  - `regional_bravo/tumor_loss`: 0.0150

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0261, max 3.3463 @ ep 0
  - `training/grad_norm_max`: last 0.1020, max 10.9618 @ ep 0

### exp25

**exp25** — single 2D run.

#### `exp25_1_unet_self_cond_rflow_128_20260113-204448`
*started 2026-01-13 20:44 • 500 epochs • 9h46m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1795 → 0.0038 (min 0.0037 @ ep 485)
  - `Loss/MSE_val`: 0.0097 → 0.0123 (min 0.0067 @ ep 38)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001229
  - 10-19: 0.000191
  - 20-29: 0.0002846
  - 30-39: 0.0004217
  - 40-49: 0.00059
  - 50-59: 0.0008069
  - 60-69: 0.0011
  - 70-79: 0.0015
  - 80-89: 0.0019
  - 90-99: 0.0022

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0506, best 0.0331 @ ep 25
  - `Generation/KID_mean_train`: last 0.0564, best 0.0305 @ ep 18
  - `Generation/KID_std_val`: last 0.0067, best 0.0024 @ ep 45
  - `Generation/KID_std_train`: last 0.0062, best 0.0024 @ ep 10
  - `Generation/CMMD_val`: last 0.2453, best 0.1606 @ ep 19
  - `Generation/CMMD_train`: last 0.2537, best 0.1557 @ ep 19
  - `Generation/extended_KID_mean_val`: last 0.0439, best 0.0361 @ ep 249
  - `Generation/extended_KID_mean_train`: last 0.0472, best 0.0399 @ ep 249
  - `Generation/extended_CMMD_val`: last 0.2234, best 0.1922 @ ep 174
  - `Generation/extended_CMMD_train`: last 0.2328, best 0.2006 @ ep 174

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.6053 (min 0.5897, max 1.3266)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9039 (min 0.8417, max 0.9064)
  - `Validation/MS-SSIM_bravo`: last 0.8207 (min 0.7958, max 0.8607)
  - `Validation/PSNR_bravo`: last 28.2927 (min 27.8450, max 29.3450)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0029
  - `regional_bravo/large`: 0.0129
  - `regional_bravo/medium`: 0.0203
  - `regional_bravo/small`: 0.0139
  - `regional_bravo/tiny`: 0.0093
  - `regional_bravo/tumor_bg_ratio`: 4.9402
  - `regional_bravo/tumor_loss`: 0.0144

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0271, max 3.3393 @ ep 0
  - `training/grad_norm_max`: last 0.0989, max 10.9328 @ ep 0

### exp26

**exp26** — single 2D run.

#### `exp26_1_rflow_128_20260115-213438`
*started 2026-01-15 21:34 • 500 epochs • 18h32m • 3964808.8 TFLOPs • peak VRAM 35.4 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.1787 → 0.0038 (min 0.0037 @ ep 489)
  - `Loss/MSE_val`: 0.0097 → 0.0137 (min 0.0069 @ ep 57)

**Per-timestep MSE (final value per bucket):**
  - 0000-0010: 0.0005328
  - 0010-0020: 0.0011
  - 0020-0030: 0.0013
  - 0030-0040: 0.0016
  - 0040-0050: 0.0016
  - 0050-0060: 0.0017
  - 0060-0070: 0.0021
  - 0070-0080: 0.0026
  - 0080-0090: 0.0030
  - 0090-0100: 0.0036

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0508, best 0.0382 @ ep 48
  - `Generation/KID_mean_train`: last 0.0558, best 0.0371 @ ep 48
  - `Generation/KID_std_val`: last 0.0065, best 0.0025 @ ep 48
  - `Generation/KID_std_train`: last 0.0070, best 0.0022 @ ep 48
  - `Generation/CMMD_val`: last 0.2427, best 0.1803 @ ep 49
  - `Generation/CMMD_train`: last 0.2530, best 0.1812 @ ep 49
  - `Generation/extended_KID_mean_val`: last 0.0417, best 0.0374 @ ep 324
  - `Generation/extended_KID_mean_train`: last 0.0497, best 0.0425 @ ep 74
  - `Generation/extended_CMMD_val`: last 0.2246, best 0.1672 @ ep 74
  - `Generation/extended_CMMD_train`: last 0.2349, best 0.1708 @ ep 74

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.5444 (min 0.5082, max 0.9491)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9039 (min 0.8312, max 0.9069)
  - `Validation/MS-SSIM_bravo`: last 0.8193 (min 0.7892, max 0.8608)
  - `Validation/PSNR_bravo`: last 27.5579 (min 27.4411, max 28.9809)

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0137
  - `regional_bravo/medium`: 0.0214
  - `regional_bravo/small`: 0.0136
  - `regional_bravo/tiny`: 0.0096
  - `regional_bravo/tumor_bg_ratio`: 4.9233
  - `regional_bravo/tumor_loss`: 0.0148

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 5, final 1.001e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0433, max 3.3485 @ ep 0
  - `training/grad_norm_max`: last 0.1614, max 10.9581 @ ep 0

### exp27

**exp27** — single 2D run.

#### `exp27_rflow_plateau_rflow_128_20260122-132228`
*started 2026-01-22 13:22 • 375 epochs • 22h14m • 2973606.8 TFLOPs • peak VRAM 34.9 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0308 → 0.0052 (min 0.0051 @ ep 369)
  - `Loss/MSE_val`: 0.0092 → 0.0075 (min 0.0065 @ ep 55)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.0435
  - 0.1-0.2: 0.0189
  - 0.2-0.3: 0.0112
  - 0.3-0.4: 0.0077
  - 0.4-0.5: 0.0060
  - 0.5-0.6: 0.0049
  - 0.6-0.7: 0.0045
  - 0.7-0.8: 0.0041
  - 0.8-0.9: 0.0040
  - 0.9-1.0: 0.0039

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.0631, best 0.0367 @ ep 19
  - `Generation/KID_mean_train`: last 0.0651, best 0.0363 @ ep 19
  - `Generation/KID_std_val`: last 0.0040, best 0.0023 @ ep 19
  - `Generation/KID_std_train`: last 0.0055, best 0.0026 @ ep 19
  - `Generation/CMMD_val`: last 0.2499, best 0.1979 @ ep 52
  - `Generation/CMMD_train`: last 0.2524, best 0.2024 @ ep 52
  - `Generation/extended_KID_mean_val`: last 0.0583, best 0.0504 @ ep 349
  - `Generation/extended_KID_mean_train`: last 0.0613, best 0.0525 @ ep 349
  - `Generation/extended_CMMD_val`: last 0.1986, best 0.1923 @ ep 349
  - `Generation/extended_CMMD_train`: last 0.2067, best 0.2010 @ ep 349

**Validation quality:**
  - `Validation/LPIPS_bravo`: last 0.9509 (min 0.9287, max 1.2618)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9065 (min 0.8353, max 0.9077)
  - `Validation/MS-SSIM_bravo`: last 0.8471 (min 0.7938, max 0.8623)
  - `Validation/PSNR_bravo`: last 28.4994 (min 27.7842, max 29.0907)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 0.7770
  - `Generation_Diversity/extended_MSSSIM`: 0.4221
  - `Generation_Diversity/LPIPS`: 0.7422
  - `Generation_Diversity/MSSSIM`: 0.4040

**Regional loss (final):**
  - `regional_bravo/background_loss`: 0.0021
  - `regional_bravo/large`: 0.0116
  - `regional_bravo/medium`: 0.0163
  - `regional_bravo/small`: 0.0117
  - `regional_bravo/tiny`: 0.0081
  - `regional_bravo/tumor_bg_ratio`: 5.8863
  - `regional_bravo/tumor_loss`: 0.0121

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 0, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0412, max 0.5239 @ ep 0
  - `training/grad_norm_max`: last 0.1751, max 9.7674 @ ep 0

---
## diffusion_2d/multi

*12 runs across 1 experiment families.*

### exp10

**exp10 (2D multi-modality)** — 12 runs exploring dual/triple modality
generation at 2D. Captures the transition from single-mode bravo to
multi-modality conditioning that informed the 3D dual (1v2) and
triple (1v3) variants.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp10_3_rflow_128_20251227-145449` — 28.8983
  2. 🥈 `exp10_3_rflow_128_20251226-142311` — 28.8656
  3.  `exp10_2_rflow_128_20251226-120701` — 28.8530
  4.  `exp10_2_rflow_128_20251227-144605` — 28.8401
  5.  `exp10_10_rflow_128_20260111-001830` — 28.8011
  6.  `exp10_1_rflow_128_20251226-120701` — 28.7595
  7.  `exp10_9_rflow_128_20260111-001830` — 28.6413

#### `exp10_1_rflow_128_20251226-120701`
*started 2025-12-26 12:07 • 125 epochs • 14h21m • 16622807.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0525 → 0.0049 (min 0.0049 @ ep 124)
  - `Loss/MSE_val`: 0.0091 → 0.0102 (min 0.0075 @ ep 11)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5428 (min 0.4903, max 0.8195)
  - `Validation/LPIPS_bravo`: last 0.5642 (min 0.5280, max 0.6106)
  - `Validation/LPIPS_flair`: last 0.6430 (min 0.6035, max 0.6801)
  - `Validation/LPIPS_t1_gd`: last 0.4433 (min 0.3965, max 0.4819)
  - `Validation/LPIPS_t1_pre`: last 0.5058 (min 0.4557, max 0.5354)
  - `Validation/MS-SSIM`: last 0.8351 (min 0.8235, max 0.8594)
  - `Validation/MS-SSIM_bravo`: last 0.8347 (min 0.8316, max 0.8526)
  - `Validation/MS-SSIM_flair`: last 0.8151 (min 0.8151, max 0.8394)
  - `Validation/MS-SSIM_t1_gd`: last 0.8439 (min 0.8431, max 0.8687)
  - `Validation/MS-SSIM_t1_pre`: last 0.8438 (min 0.8438, max 0.8722)
  - `Validation/PSNR`: last 27.4405 (min 27.2889, max 28.2574)
  - `Validation/PSNR_bravo`: last 28.1159 (min 28.1159, max 28.7595)
  - `Validation/PSNR_flair`: last 26.9994 (min 26.9762, max 27.6959)
  - `Validation/PSNR_t1_gd`: last 28.2155 (min 28.2069, max 29.1687)
  - `Validation/PSNR_t1_pre`: last 26.4471 (min 26.4471, max 27.6322)

**Regional loss (final):**
  - `regional/background_loss`: 0.0042
  - `regional/large`: 0.0143
  - `regional/medium`: 0.0149
  - `regional/small`: 0.0169
  - `regional/tiny`: 0.0118
  - `regional/tumor_bg_ratio`: 3.5651
  - `regional/tumor_loss`: 0.0149
  - `regional_bravo/background_loss`: 0.0036
  - `regional_bravo/large`: 0.0123
  - `regional_bravo/medium`: 0.0139
  - `regional_bravo/small`: 0.0144
  - `regional_bravo/tiny`: 0.0111
  - `regional_bravo/tumor_bg_ratio`: 3.7432
  - `regional_bravo/tumor_loss`: 0.0133
  - `regional_flair/background_loss`: 0.0042
  - `regional_flair/large`: 0.0226
  - `regional_flair/medium`: 0.0239
  - `regional_flair/small`: 0.0239
  - `regional_flair/tiny`: 0.0170
  - `regional_flair/tumor_bg_ratio`: 5.3429
  - `regional_flair/tumor_loss`: 0.0226
  - `regional_t1_gd/background_loss`: 0.0037
  - `regional_t1_gd/large`: 0.0123
  - `regional_t1_gd/medium`: 0.0160
  - `regional_t1_gd/small`: 0.0151
  - `regional_t1_gd/tiny`: 0.0111
  - `regional_t1_gd/tumor_bg_ratio`: 3.8717
  - `regional_t1_gd/tumor_loss`: 0.0142
  - `regional_t1_pre/background_loss`: 0.0055
  - `regional_t1_pre/large`: 0.0080
  - `regional_t1_pre/medium`: 0.0076
  - `regional_t1_pre/small`: 0.0101
  - `regional_t1_pre/tiny`: 0.0092
  - `regional_t1_pre/tumor_bg_ratio`: 1.5624
  - `regional_t1_pre/tumor_loss`: 0.0086

**Training meta:**
  - `training/grad_norm_avg`: last 0.0292, max 1.0784 @ ep 0
  - `training/grad_norm_max`: last 0.1038, max 10.8761 @ ep 0

#### `exp10_2_rflow_128_20251226-120701`
*started 2025-12-26 12:07 • 125 epochs • 14h22m • 16622770.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0485 → 0.0057 (min 0.0057 @ ep 124)
  - `Loss/MSE_val`: 0.0099 → 0.0081 (min 0.0072 @ ep 36)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5205 (min 0.5007, max 0.9245)
  - `Validation/LPIPS_bravo`: last 0.5520 (min 0.5342, max 0.7251)
  - `Validation/LPIPS_flair`: last 0.6199 (min 0.6056, max 0.8239)
  - `Validation/LPIPS_t1_gd`: last 0.4172 (min 0.4022, max 0.5938)
  - `Validation/LPIPS_t1_pre`: last 0.4833 (min 0.4506, max 0.6810)
  - `Validation/MS-SSIM`: last 0.8517 (min 0.8113, max 0.8628)
  - `Validation/MS-SSIM_bravo`: last 0.8471 (min 0.8417, max 0.8571)
  - `Validation/MS-SSIM_flair`: last 0.8365 (min 0.8259, max 0.8471)
  - `Validation/MS-SSIM_t1_gd`: last 0.8626 (min 0.8516, max 0.8716)
  - `Validation/MS-SSIM_t1_pre`: last 0.8613 (min 0.8568, max 0.8762)
  - `Validation/PSNR`: last 27.9273 (min 27.2679, max 28.3475)
  - `Validation/PSNR_bravo`: last 28.4963 (min 28.4473, max 28.8530)
  - `Validation/PSNR_flair`: last 27.4993 (min 27.4174, max 27.8091)
  - `Validation/PSNR_t1_gd`: last 28.7453 (min 28.7453, max 29.2740)
  - `Validation/PSNR_t1_pre`: last 27.1160 (min 27.1160, max 27.6918)

**Regional loss (final):**
  - `regional/background_loss`: 0.0033
  - `regional/large`: 0.0134
  - `regional/medium`: 0.0138
  - `regional/small`: 0.0132
  - `regional/tiny`: 0.0119
  - `regional/tumor_bg_ratio`: 3.9967
  - `regional/tumor_loss`: 0.0133
  - `regional_bravo/background_loss`: 0.0028
  - `regional_bravo/large`: 0.0130
  - `regional_bravo/medium`: 0.0141
  - `regional_bravo/small`: 0.0115
  - `regional_bravo/tiny`: 0.0115
  - `regional_bravo/tumor_bg_ratio`: 4.4848
  - `regional_bravo/tumor_loss`: 0.0128
  - `regional_flair/background_loss`: 0.0033
  - `regional_flair/large`: 0.0203
  - `regional_flair/medium`: 0.0195
  - `regional_flair/small`: 0.0207
  - `regional_flair/tiny`: 0.0159
  - `regional_flair/tumor_bg_ratio`: 5.8659
  - `regional_flair/tumor_loss`: 0.0195
  - `regional_t1_gd/background_loss`: 0.0028
  - `regional_t1_gd/large`: 0.0129
  - `regional_t1_gd/medium`: 0.0156
  - `regional_t1_gd/small`: 0.0122
  - `regional_t1_gd/tiny`: 0.0105
  - `regional_t1_gd/tumor_bg_ratio`: 4.7119
  - `regional_t1_gd/tumor_loss`: 0.0133
  - `regional_t1_pre/background_loss`: 0.0040
  - `regional_t1_pre/large`: 0.0083
  - `regional_t1_pre/medium`: 0.0072
  - `regional_t1_pre/small`: 0.0096
  - `regional_t1_pre/tiny`: 0.0096
  - `regional_t1_pre/tumor_bg_ratio`: 2.0934
  - `regional_t1_pre/tumor_loss`: 0.0085

**Training meta:**
  - `training/grad_norm_avg`: last 0.0208, max 1.0381 @ ep 0
  - `training/grad_norm_max`: last 0.0823, max 10.8679 @ ep 0

#### `exp10_3_rflow_128_20251226-142311`
*started 2025-12-26 14:23 • 125 epochs • 12h53m • 16622770.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0447 → 0.0058 (min 0.0058 @ ep 114)
  - `Loss/MSE_val`: 0.0099 → 0.0076 (min 0.0071 @ ep 35)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5202 (min 0.5021, max 0.9807)
  - `Validation/LPIPS_bravo`: last 0.5526 (min 0.5465, max 0.8089)
  - `Validation/LPIPS_flair`: last 0.6307 (min 0.6190, max 0.8794)
  - `Validation/LPIPS_t1_gd`: last 0.4180 (min 0.4141, max 0.7000)
  - `Validation/LPIPS_t1_pre`: last 0.4800 (min 0.4772, max 0.7325)
  - `Validation/MS-SSIM`: last 0.8608 (min 0.8025, max 0.8655)
  - `Validation/MS-SSIM_bravo`: last 0.8546 (min 0.8324, max 0.8608)
  - `Validation/MS-SSIM_flair`: last 0.8374 (min 0.8183, max 0.8458)
  - `Validation/MS-SSIM_t1_gd`: last 0.8695 (min 0.8495, max 0.8728)
  - `Validation/MS-SSIM_t1_pre`: last 0.8744 (min 0.8538, max 0.8792)
  - `Validation/PSNR`: last 28.2477 (min 27.0028, max 28.4191)
  - `Validation/PSNR_bravo`: last 28.7125 (min 28.1214, max 28.8656)
  - `Validation/PSNR_flair`: last 27.5709 (min 27.1452, max 27.8453)
  - `Validation/PSNR_t1_gd`: last 29.1006 (min 28.4596, max 29.2716)
  - `Validation/PSNR_t1_pre`: last 27.5351 (min 27.0703, max 27.7764)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0123
  - `regional/medium`: 0.0134
  - `regional/small`: 0.0136
  - `regional/tiny`: 0.0117
  - `regional/tumor_bg_ratio`: 4.5394
  - `regional/tumor_loss`: 0.0130
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0117
  - `regional_bravo/medium`: 0.0143
  - `regional_bravo/small`: 0.0130
  - `regional_bravo/tiny`: 0.0121
  - `regional_bravo/tumor_bg_ratio`: 4.9365
  - `regional_bravo/tumor_loss`: 0.0130
  - `regional_flair/background_loss`: 0.0032
  - `regional_flair/large`: 0.0179
  - `regional_flair/medium`: 0.0227
  - `regional_flair/small`: 0.0189
  - `regional_flair/tiny`: 0.0166
  - `regional_flair/tumor_bg_ratio`: 6.0871
  - `regional_flair/tumor_loss`: 0.0197
  - `regional_t1_gd/background_loss`: 0.0024
  - `regional_t1_gd/large`: 0.0106
  - `regional_t1_gd/medium`: 0.0131
  - `regional_t1_gd/small`: 0.0125
  - `regional_t1_gd/tiny`: 0.0114
  - `regional_t1_gd/tumor_bg_ratio`: 5.0568
  - `regional_t1_gd/tumor_loss`: 0.0121
  - `regional_t1_pre/background_loss`: 0.0034
  - `regional_t1_pre/large`: 0.0078
  - `regional_t1_pre/medium`: 0.0052
  - `regional_t1_pre/small`: 0.0096
  - `regional_t1_pre/tiny`: 0.0091
  - `regional_t1_pre/tumor_bg_ratio`: 2.1957
  - `regional_t1_pre/tumor_loss`: 0.0075

**Training meta:**
  - `training/grad_norm_avg`: last 0.0188, max 1.0282 @ ep 0
  - `training/grad_norm_max`: last 0.0887, max 10.9346 @ ep 0

#### `exp10_2_rflow_128_20251227-144605`
*started 2025-12-27 14:46 • 125 epochs • 12h53m • 16622770.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0493 → 0.0057 (min 0.0056 @ ep 117)
  - `Loss/MSE_val`: 0.0103 → 0.0081 (min 0.0073 @ ep 32)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5314 (min 0.4936, max 0.9871)
  - `Validation/LPIPS_bravo`: last 0.5524 (min 0.5322, max 0.6887)
  - `Validation/LPIPS_flair`: last 0.6285 (min 0.6001, max 0.7845)
  - `Validation/LPIPS_t1_gd`: last 0.4370 (min 0.4024, max 0.5510)
  - `Validation/LPIPS_t1_pre`: last 0.4861 (min 0.4592, max 0.6799)
  - `Validation/MS-SSIM`: last 0.8504 (min 0.7977, max 0.8629)
  - `Validation/MS-SSIM_bravo`: last 0.8482 (min 0.8415, max 0.8578)
  - `Validation/MS-SSIM_flair`: last 0.8324 (min 0.8265, max 0.8464)
  - `Validation/MS-SSIM_t1_gd`: last 0.8554 (min 0.8548, max 0.8711)
  - `Validation/MS-SSIM_t1_pre`: last 0.8649 (min 0.8528, max 0.8781)
  - `Validation/PSNR`: last 27.8867 (min 26.9589, max 28.3322)
  - `Validation/PSNR_bravo`: last 28.5158 (min 28.4848, max 28.8401)
  - `Validation/PSNR_flair`: last 27.3701 (min 27.3701, max 27.8540)
  - `Validation/PSNR_t1_gd`: last 28.6021 (min 28.6021, max 29.2260)
  - `Validation/PSNR_t1_pre`: last 27.2054 (min 27.1341, max 27.7124)

**Regional loss (final):**
  - `regional/background_loss`: 0.0034
  - `regional/large`: 0.0124
  - `regional/medium`: 0.0128
  - `regional/small`: 0.0139
  - `regional/tiny`: 0.0118
  - `regional/tumor_bg_ratio`: 3.7515
  - `regional/tumor_loss`: 0.0129
  - `regional_bravo/background_loss`: 0.0030
  - `regional_bravo/large`: 0.0106
  - `regional_bravo/medium`: 0.0129
  - `regional_bravo/small`: 0.0123
  - `regional_bravo/tiny`: 0.0115
  - `regional_bravo/tumor_bg_ratio`: 4.0531
  - `regional_bravo/tumor_loss`: 0.0120
  - `regional_flair/background_loss`: 0.0034
  - `regional_flair/large`: 0.0137
  - `regional_flair/medium`: 0.0212
  - `regional_flair/small`: 0.0185
  - `regional_flair/tiny`: 0.0144
  - `regional_flair/tumor_bg_ratio`: 5.2470
  - `regional_flair/tumor_loss`: 0.0178
  - `regional_t1_gd/background_loss`: 0.0029
  - `regional_t1_gd/large`: 0.0112
  - `regional_t1_gd/medium`: 0.0129
  - `regional_t1_gd/small`: 0.0129
  - `regional_t1_gd/tiny`: 0.0106
  - `regional_t1_gd/tumor_bg_ratio`: 4.1689
  - `regional_t1_gd/tumor_loss`: 0.0122
  - `regional_t1_pre/background_loss`: 0.0040
  - `regional_t1_pre/large`: 0.0082
  - `regional_t1_pre/medium`: 0.0065
  - `regional_t1_pre/small`: 0.0088
  - `regional_t1_pre/tiny`: 0.0085
  - `regional_t1_pre/tumor_bg_ratio`: 1.9293
  - `regional_t1_pre/tumor_loss`: 0.0078

**Training meta:**
  - `training/grad_norm_avg`: last 0.0217, max 1.0551 @ ep 0
  - `training/grad_norm_max`: last 0.1218, max 10.9131 @ ep 0

#### `exp10_3_rflow_128_20251227-145449`
*started 2025-12-27 14:54 • 125 epochs • 15h48m • 16622770.0 TFLOPs*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0440 → 0.0058 (min 0.0058 @ ep 115)
  - `Loss/MSE_val`: 0.0099 → 0.0074 (min 0.0072 @ ep 50)

**Validation quality:**
  - `Validation/LPIPS`: last 0.5011 (min 0.4944, max 1.0893)
  - `Validation/LPIPS_bravo`: last 0.5424 (min 0.5264, max 0.8429)
  - `Validation/LPIPS_flair`: last 0.6119 (min 0.5930, max 0.9507)
  - `Validation/LPIPS_t1_gd`: last 0.4073 (min 0.3947, max 0.7245)
  - `Validation/LPIPS_t1_pre`: last 0.4719 (min 0.4574, max 0.8018)
  - `Validation/MS-SSIM`: last 0.8592 (min 0.8002, max 0.8639)
  - `Validation/MS-SSIM_bravo`: last 0.8557 (min 0.8376, max 0.8607)
  - `Validation/MS-SSIM_flair`: last 0.8426 (min 0.8220, max 0.8465)
  - `Validation/MS-SSIM_t1_gd`: last 0.8679 (min 0.8492, max 0.8740)
  - `Validation/MS-SSIM_t1_pre`: last 0.8734 (min 0.8544, max 0.8787)
  - `Validation/PSNR`: last 28.1922 (min 26.9712, max 28.3857)
  - `Validation/PSNR_bravo`: last 28.7419 (min 28.3858, max 28.8983)
  - `Validation/PSNR_flair`: last 27.7135 (min 27.2695, max 27.8279)
  - `Validation/PSNR_t1_gd`: last 29.0342 (min 28.5792, max 29.2938)
  - `Validation/PSNR_t1_pre`: last 27.4955 (min 26.9880, max 27.6868)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0122
  - `regional/medium`: 0.0134
  - `regional/small`: 0.0139
  - `regional/tiny`: 0.0126
  - `regional/tumor_bg_ratio`: 4.5614
  - `regional/tumor_loss`: 0.0132
  - `regional_bravo/background_loss`: 0.0026
  - `regional_bravo/large`: 0.0128
  - `regional_bravo/medium`: 0.0131
  - `regional_bravo/small`: 0.0121
  - `regional_bravo/tiny`: 0.0124
  - `regional_bravo/tumor_bg_ratio`: 4.7810
  - `regional_bravo/tumor_loss`: 0.0127
  - `regional_flair/background_loss`: 0.0030
  - `regional_flair/large`: 0.0157
  - `regional_flair/medium`: 0.0210
  - `regional_flair/small`: 0.0203
  - `regional_flair/tiny`: 0.0155
  - `regional_flair/tumor_bg_ratio`: 6.2339
  - `regional_flair/tumor_loss`: 0.0188
  - `regional_t1_gd/background_loss`: 0.0025
  - `regional_t1_gd/large`: 0.0118
  - `regional_t1_gd/medium`: 0.0153
  - `regional_t1_gd/small`: 0.0126
  - `regional_t1_gd/tiny`: 0.0115
  - `regional_t1_gd/tumor_bg_ratio`: 5.2448
  - `regional_t1_gd/tumor_loss`: 0.0132
  - `regional_t1_pre/background_loss`: 0.0034
  - `regional_t1_pre/large`: 0.0072
  - `regional_t1_pre/medium`: 0.0059
  - `regional_t1_pre/small`: 0.0109
  - `regional_t1_pre/tiny`: 0.0097
  - `regional_t1_pre/tumor_bg_ratio`: 2.3871
  - `regional_t1_pre/tumor_loss`: 0.0081

**Training meta:**
  - `training/grad_norm_avg`: last 0.0191, max 1.0138 @ ep 0
  - `training/grad_norm_max`: last 0.0970, max 10.8838 @ ep 0

#### `exp10_4_rflow_128_20260110-014655`
*started 2026-01-10 01:46 • 125 epochs • 15h46m • 16622864.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0522 → 0.0045 (min 0.0045 @ ep 120)
  - `Loss/MSE_val`: 0.0090 → 0.0116 (min 0.0074 @ ep 11)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001452
  - 10-19: 0.0002557
  - 20-29: 0.0003606
  - 30-39: 0.000489
  - 40-49: 0.0006743
  - 50-59: 0.000923
  - 60-69: 0.0012
  - 70-79: 0.0017
  - 80-89: 0.0025
  - 90-99: 0.0032

**Validation quality:**
  - `Validation/LPIPS`: last 0.5158 (min 0.4621, max 0.8144)
  - `Validation/MS-SSIM`: last 0.8274 (min 0.8256, max 0.8649)
  - `Validation/PSNR`: last 26.9810 (min 26.9548, max 28.3741)

**Regional loss (final):**
  - `regional/background_loss`: 0.0035
  - `regional/large`: 0.0173
  - `regional/medium`: 0.0213
  - `regional/small`: 0.0185
  - `regional/tiny`: 0.0107
  - `regional/tumor_bg_ratio`: 4.9388
  - `regional/tumor_loss`: 0.0172

**Training meta:**
  - `training/grad_norm_avg`: last 0.0268, max 1.0880 @ ep 0
  - `training/grad_norm_max`: last 0.0988, max 10.8247 @ ep 0

#### `exp10_5_rflow_128_20260110-014655`
*started 2026-01-10 01:46 • 125 epochs • 15h27m • 16622892.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0449 → 0.0057 (min 0.0056 @ ep 121)
  - `Loss/MSE_val`: 0.0098 → 0.0076 (min 0.0070 @ ep 55)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0073
  - 10-19: 0.0070
  - 20-29: 0.0072
  - 30-39: 0.0072
  - 40-49: 0.0073
  - 50-59: 0.0073
  - 60-69: 0.0073
  - 70-79: 0.0072
  - 80-89: 0.0079
  - 90-99: 0.0075

**Validation quality:**
  - `Validation/LPIPS`: last 0.4844 (min 0.4749, max 0.9492)
  - `Validation/MS-SSIM`: last 0.8626 (min 0.8009, max 0.8694)
  - `Validation/PSNR`: last 28.2099 (min 26.8970, max 28.4921)

**Regional loss (final):**
  - `regional/background_loss`: 0.0020
  - `regional/large`: 0.0118
  - `regional/medium`: 0.0143
  - `regional/small`: 0.0130
  - `regional/tiny`: 0.0099
  - `regional/tumor_bg_ratio`: 6.2752
  - `regional/tumor_loss`: 0.0124

**Training meta:**
  - `training/grad_norm_avg`: last 0.0177, max 1.0102 @ ep 0
  - `training/grad_norm_max`: last 0.0741, max 10.8207 @ ep 0

#### `exp10_6_rflow_128_20260110-113040`
*started 2026-01-10 11:30 • 34 epochs • 56h32m • 4388436.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0523 → 0.0061 (min 0.0061 @ ep 33)
  - `Loss/MSE_val`: 0.0094 → 0.0078 (min 0.0073 @ ep 17)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001665
  - 10-19: 0.0003519
  - 20-29: 0.0005402
  - 30-39: 0.0007199
  - 40-49: 0.0009342
  - 50-59: 0.0012
  - 60-69: 0.0016
  - 70-79: 0.0022
  - 80-89: 0.0030
  - 90-99: 0.0041

**Validation quality:**
  - `Validation/LPIPS`: last 0.5134 (min 0.4730, max 0.7290)
  - `Validation/MS-SSIM`: last 0.8540 (min 0.8311, max 0.8627)
  - `Validation/PSNR`: last 28.1290 (min 27.6073, max 28.3725)

**Regional loss (final):**
  - `regional/background_loss`: 0.0019
  - `regional/large`: 0.0116
  - `regional/medium`: 0.0135
  - `regional/small`: 0.0122
  - `regional/tiny`: 0.0088
  - `regional/tumor_bg_ratio`: 6.0206
  - `regional/tumor_loss`: 0.0116

**Training meta:**
  - `training/grad_norm_avg`: last 0.0851, max 1.0978 @ ep 0
  - `training/grad_norm_max`: last 0.4002, max 10.8693 @ ep 0

#### `exp10_7_rflow_128_20260110-113040`
*started 2026-01-10 11:30 • 125 epochs • 15h36m • 16622752.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0522 → 0.0048 (min 0.0048 @ ep 123)
  - `Loss/MSE_val`: 0.0094 → 0.0104 (min 0.0075 @ ep 22)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0001448
  - 10-19: 0.0002634
  - 20-29: 0.0003779
  - 30-39: 0.000544
  - 40-49: 0.0007338
  - 50-59: 0.0009945
  - 60-69: 0.0013
  - 70-79: 0.0019
  - 80-89: 0.0027
  - 90-99: 0.0035

**Validation quality:**
  - `Validation/LPIPS`: last 0.5106 (min 0.4910, max 0.8864)
  - `Validation/MS-SSIM`: last 0.8330 (min 0.8265, max 0.8622)
  - `Validation/PSNR`: last 27.3833 (min 27.3674, max 28.3348)

**Regional loss (final):**
  - `regional/background_loss`: 0.0028
  - `regional/large`: 0.0136
  - `regional/medium`: 0.0156
  - `regional/small`: 0.0133
  - `regional/tiny`: 0.0091
  - `regional/tumor_bg_ratio`: 4.7269
  - `regional/tumor_loss`: 0.0130

**Training meta:**
  - `training/grad_norm_avg`: last 0.0292, max 1.0809 @ ep 0
  - `training/grad_norm_max`: last 0.1323, max 10.8864 @ ep 0

#### `exp10_8_rflow_128_20260110-113040`
*started 2026-01-10 11:30 • 125 epochs • 15h39m • 16622886.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0522 → 0.0048 (min 0.0048 @ ep 117)
  - `Loss/MSE_val`: 0.0091 → 0.0103 (min 0.0075 @ ep 24)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.000147
  - 10-19: 0.0002626
  - 20-29: 0.0003825
  - 30-39: 0.0005353
  - 40-49: 0.0007234
  - 50-59: 0.0009972
  - 60-69: 0.0014
  - 70-79: 0.0019
  - 80-89: 0.0027
  - 90-99: 0.0037

**Validation quality:**
  - `Validation/LPIPS`: last 0.5148 (min 0.4878, max 0.8026)
  - `Validation/MS-SSIM`: last 0.8337 (min 0.8252, max 0.8598)
  - `Validation/PSNR`: last 27.3351 (min 27.3093, max 28.2902)

**Regional loss (final):**
  - `regional/background_loss`: 0.0028
  - `regional/large`: 0.0137
  - `regional/medium`: 0.0146
  - `regional/small`: 0.0133
  - `regional/tiny`: 0.0092
  - `regional/tumor_bg_ratio`: 4.5307
  - `regional/tumor_loss`: 0.0128

**Training meta:**
  - `training/grad_norm_avg`: last 0.0276, max 1.0793 @ ep 0
  - `training/grad_norm_max`: last 0.0963, max 10.9342 @ ep 0

#### `exp10_10_rflow_128_20260111-001830`
*started 2026-01-11 00:18 • 125 epochs • 17h39m • 16623096.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0441 → 0.0057 (min 0.0056 @ ep 120)
  - `Loss/MSE_val`: 0.0099 → 0.0075 (min 0.0070 @ ep 57)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.0075
  - 10-19: 0.0073
  - 20-29: 0.0074
  - 30-39: 0.0075
  - 40-49: 0.0074
  - 50-59: 0.0075
  - 60-69: 0.0074
  - 70-79: 0.0074
  - 80-89: 0.0080
  - 90-99: 0.0076

**Validation quality:**
  - `Validation/LPIPS`: last 0.4821 (min 0.4696, max 0.9637)
  - `Validation/LPIPS_bravo`: last 0.5743 (min 0.5529, max 0.9695)
  - `Validation/LPIPS_flair`: last 0.6795 (min 0.6680, max 1.0559)
  - `Validation/LPIPS_t1_gd`: last 0.4128 (min 0.3847, max 0.8592)
  - `Validation/LPIPS_t1_pre`: last 0.4774 (min 0.4696, max 0.9659)
  - `Validation/MS-SSIM`: last 0.8639 (min 0.8070, max 0.8694)
  - `Validation/MS-SSIM_bravo`: last 0.8487 (min 0.8056, max 0.8576)
  - `Validation/MS-SSIM_flair`: last 0.8311 (min 0.7856, max 0.8390)
  - `Validation/MS-SSIM_t1_gd`: last 0.8674 (min 0.8155, max 0.8774)
  - `Validation/MS-SSIM_t1_pre`: last 0.8742 (min 0.8076, max 0.8788)
  - `Validation/PSNR`: last 28.2621 (min 27.1456, max 28.5027)
  - `Validation/PSNR_bravo`: last 28.4670 (min 27.7333, max 28.8011)
  - `Validation/PSNR_flair`: last 27.4000 (min 26.7253, max 27.7206)
  - `Validation/PSNR_t1_gd`: last 28.9394 (min 27.7862, max 29.3717)
  - `Validation/PSNR_t1_pre`: last 27.5022 (min 26.0962, max 27.7981)

**Regional loss (final):**
  - `regional/background_loss`: 0.0019
  - `regional/large`: 0.0138
  - `regional/medium`: 0.0138
  - `regional/small`: 0.0126
  - `regional/tiny`: 0.0100
  - `regional/tumor_bg_ratio`: 6.5075
  - `regional/tumor_loss`: 0.0126
  - `regional_bravo/background_loss`: 0.0018
  - `regional_bravo/large`: 0.0129
  - `regional_bravo/medium`: 0.0166
  - `regional_bravo/small`: 0.0094
  - `regional_bravo/tiny`: 0.0082
  - `regional_bravo/tumor_bg_ratio`: 6.5613
  - `regional_bravo/tumor_loss`: 0.0119
  - `regional_flair/background_loss`: 0.0022
  - `regional_flair/large`: 0.0285
  - `regional_flair/medium`: 0.0296
  - `regional_flair/small`: 0.0201
  - `regional_flair/tiny`: 0.0120
  - `regional_flair/tumor_bg_ratio`: 10.1551
  - `regional_flair/tumor_loss`: 0.0228
  - `regional_t1_gd/background_loss`: 0.0017
  - `regional_t1_gd/large`: 0.0112
  - `regional_t1_gd/medium`: 0.0154
  - `regional_t1_gd/small`: 0.0103
  - `regional_t1_gd/tiny`: 0.0074
  - `regional_t1_gd/tumor_bg_ratio`: 6.5200
  - `regional_t1_gd/tumor_loss`: 0.0112
  - `regional_t1_pre/background_loss`: 0.0022
  - `regional_t1_pre/large`: 0.0073
  - `regional_t1_pre/medium`: 0.0062
  - `regional_t1_pre/small`: 0.0089
  - `regional_t1_pre/tiny`: 0.0080
  - `regional_t1_pre/tumor_bg_ratio`: 3.4316
  - `regional_t1_pre/tumor_loss`: 0.0076

**Training meta:**
  - `training/grad_norm_avg`: last 0.0193, max 1.0174 @ ep 0
  - `training/grad_norm_max`: last 0.0790, max 10.8601 @ ep 0

#### `exp10_9_rflow_128_20260111-001830`
*started 2026-01-11 00:18 • 125 epochs • 19h11m • 16623284.0 TFLOPs • peak VRAM 35.7 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.0523 → 0.0045 (min 0.0045 @ ep 124)
  - `Loss/MSE_val`: 0.0091 → 0.0115 (min 0.0074 @ ep 16)

**Per-timestep MSE (final value per bucket):**
  - 0-9: 0.000143
  - 10-19: 0.0002597
  - 20-29: 0.0003649
  - 30-39: 0.000498
  - 40-49: 0.0006736
  - 50-59: 0.0009024
  - 60-69: 0.0013
  - 70-79: 0.0017
  - 80-89: 0.0025
  - 90-99: 0.0031

**Validation quality:**
  - `Validation/LPIPS`: last 0.4862 (min 0.4634, max 0.8330)
  - `Validation/LPIPS_bravo`: last 0.5787 (min 0.5397, max 0.8822)
  - `Validation/LPIPS_flair`: last 0.7007 (min 0.6465, max 0.9780)
  - `Validation/LPIPS_t1_gd`: last 0.4215 (min 0.3741, max 0.7523)
  - `Validation/LPIPS_t1_pre`: last 0.5029 (min 0.4427, max 0.8294)
  - `Validation/MS-SSIM`: last 0.8263 (min 0.8263, max 0.8628)
  - `Validation/MS-SSIM_bravo`: last 0.8157 (min 0.8139, max 0.8494)
  - `Validation/MS-SSIM_flair`: last 0.7878 (min 0.7863, max 0.8358)
  - `Validation/MS-SSIM_t1_gd`: last 0.8328 (min 0.8328, max 0.8711)
  - `Validation/MS-SSIM_t1_pre`: last 0.8275 (min 0.8255, max 0.8740)
  - `Validation/PSNR`: last 26.9297 (min 26.9297, max 28.3695)
  - `Validation/PSNR_bravo`: last 27.4708 (min 27.4708, max 28.6413)
  - `Validation/PSNR_flair`: last 26.2669 (min 26.2669, max 27.5739)
  - `Validation/PSNR_t1_gd`: last 27.6864 (min 27.6864, max 29.1405)
  - `Validation/PSNR_t1_pre`: last 25.7597 (min 25.7233, max 27.5920)

**Regional loss (final):**
  - `regional/background_loss`: 0.0035
  - `regional/large`: 0.0159
  - `regional/medium`: 0.0188
  - `regional/small`: 0.0169
  - `regional/tiny`: 0.0104
  - `regional/tumor_bg_ratio`: 4.5174
  - `regional/tumor_loss`: 0.0157
  - `regional_bravo/background_loss`: 0.0029
  - `regional_bravo/large`: 0.0149
  - `regional_bravo/medium`: 0.0184
  - `regional_bravo/small`: 0.0128
  - `regional_bravo/tiny`: 0.0081
  - `regional_bravo/tumor_bg_ratio`: 4.6840
  - `regional_bravo/tumor_loss`: 0.0137
  - `regional_flair/background_loss`: 0.0036
  - `regional_flair/large`: 0.0329
  - `regional_flair/medium`: 0.0331
  - `regional_flair/small`: 0.0229
  - `regional_flair/tiny`: 0.0123
  - `regional_flair/tumor_bg_ratio`: 7.1095
  - `regional_flair/tumor_loss`: 0.0255
  - `regional_t1_gd/background_loss`: 0.0030
  - `regional_t1_gd/large`: 0.0142
  - `regional_t1_gd/medium`: 0.0180
  - `regional_t1_gd/small`: 0.0139
  - `regional_t1_gd/tiny`: 0.0085
  - `regional_t1_gd/tumor_bg_ratio`: 4.5782
  - `regional_t1_gd/tumor_loss`: 0.0139
  - `regional_t1_pre/background_loss`: 0.0048
  - `regional_t1_pre/large`: 0.0103
  - `regional_t1_pre/medium`: 0.0098
  - `regional_t1_pre/small`: 0.0130
  - `regional_t1_pre/tiny`: 0.0100
  - `regional_t1_pre/tumor_bg_ratio`: 2.2742
  - `regional_t1_pre/tumor_loss`: 0.0108

**Training meta:**
  - `training/grad_norm_avg`: last 0.0329, max 1.0936 @ ep 0
  - `training/grad_norm_max`: last 0.1513, max 10.9764 @ ep 0

---
## diffusion_2d/restoration

*2 runs across 1 experiment families.*

### exp33

**exp33 (2D restoration)** — 2D IR-SDE restoration, two variants.
Analogous to the 3D exp33 restoration sweep but at 2D resolution.

**Family ranking by `Generation/extended_KID_mean_val` (ext KID ↓):**
  1. 🥇 `exp33_4_resfusion_restoration_20260415-195458` — 0.4237
  2. 🥈 `exp33_1_irsde_restoration_20260415-194624` — 0.4265

#### `exp33_1_irsde_restoration_20260415-194624`
*started 2026-04-15 19:46 • 417 epochs • 19h23m • 1091043.0 TFLOPs • peak VRAM 11.5 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0021 → 0.0002522 (min 0.0002494 @ ep 401)
  - `Loss/L1_val`: 0.0041 → 0.0060 (min 0.0041 @ ep 0)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 0.8268
  - 0.1-0.2: 0.4437
  - 0.2-0.3: 0.4728
  - 0.3-0.4: 0.4495
  - 0.4-0.5: 0.4390
  - 0.5-0.6: 0.4296
  - 0.6-0.7: 0.4118
  - 0.7-0.8: 0.4047
  - 0.8-0.9: 0.4027
  - 0.9-1.0: 0.3996

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.4778, best 0.4749 @ ep 405
  - `Generation/KID_mean_train`: last 0.4886, best 0.4858 @ ep 405
  - `Generation/KID_std_val`: last 0.0061, best 0.0049 @ ep 128
  - `Generation/KID_std_train`: last 0.0252, best 0.0177 @ ep 163
  - `Generation/CMMD_val`: last 0.6323, best 0.6221 @ ep 22
  - `Generation/CMMD_train`: last 0.6499, best 0.6390 @ ep 9
  - `Generation/extended_KID_mean_val`: last 0.4308, best 0.4265 @ ep 359
  - `Generation/extended_KID_mean_train`: last 0.4382, best 0.4382 @ ep 399
  - `Generation/extended_CMMD_val`: last 0.6191, best 0.6122 @ ep 79
  - `Generation/extended_CMMD_train`: last 0.6387, best 0.6296 @ ep 79

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.4655 (min 1.3283, max 1.5913)
  - `Validation/MS-SSIM_restoration`: last 0.5385 (min 0.4207, max 0.5853)
  - `Validation/PSNR_restoration`: last 21.2633 (min 17.7103, max 24.7868)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.1844
  - `Generation_Diversity/extended_MSSSIM`: 0.2955
  - `Generation_Diversity/LPIPS`: 0.6730
  - `Generation_Diversity/MSSSIM`: 0.3348

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.1138
  - `regional_restoration/large`: 0.1017
  - `regional_restoration/medium`: 0.1382
  - `regional_restoration/small`: 0.0905
  - `regional_restoration/tiny`: 0.0975
  - `regional_restoration/tumor_bg_ratio`: 0.9009
  - `regional_restoration/tumor_loss`: 0.1025

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 4.786e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0074, max 0.0375 @ ep 1
  - `training/grad_norm_max`: last 0.0291, max 0.0846 @ ep 1

#### `exp33_4_resfusion_restoration_20260415-195458`
*started 2026-04-15 19:54 • 445 epochs • 11h50m • 207412.6 TFLOPs • peak VRAM 4.3 GB*

**Loss dynamics:**
  - `Loss/MSE_train`: 0.8287 → 0.0203 (min 0.0191 @ ep 347)
  - `Loss/MSE_val`: 1.0017 → 0.0516 (min 0.0446 @ ep 424)

**Per-timestep MSE (final value per bucket):**
  - 0.0-0.1: 1.7470
  - 0.1-0.2: 1.2734
  - 0.2-0.3: 0.9144
  - 0.3-0.4: 0.6883

**Generation metrics:**
  - `Generation/KID_mean_val`: last 0.4267, best 0.4183 @ ep 325
  - `Generation/KID_mean_train`: last 0.4467, best 0.4306 @ ep 348
  - `Generation/KID_std_val`: last 0.0079, best 0.0057 @ ep 428
  - `Generation/KID_std_train`: last 0.0254, best 0.0186 @ ep 18
  - `Generation/CMMD_val`: last 0.6330, best 0.6046 @ ep 3
  - `Generation/CMMD_train`: last 0.6575, best 0.6254 @ ep 3
  - `Generation/extended_KID_mean_val`: last 0.4237, best 0.4237 @ ep 439
  - `Generation/extended_KID_mean_train`: last 0.4395, best 0.4395 @ ep 439
  - `Generation/extended_CMMD_val`: last 0.6204, best 0.6161 @ ep 39
  - `Generation/extended_CMMD_train`: last 0.6459, best 0.6407 @ ep 39

**Validation quality:**
  - `Validation/LPIPS_restoration`: last 1.3564 (min 1.1244, max 1.6959)
  - `Validation/MS-SSIM_restoration`: last 0.5428 (min 0.3809, max 0.6393)
  - `Validation/PSNR_restoration`: last 23.0734 (min 16.0208, max 27.4536)

**Diversity (extended):**
  - `Generation_Diversity/extended_LPIPS`: 1.1389
  - `Generation_Diversity/extended_MSSSIM`: 0.2545
  - `Generation_Diversity/LPIPS`: 1.1575
  - `Generation_Diversity/MSSSIM`: 0.2391

**Regional loss (final):**
  - `regional_restoration/background_loss`: 0.0446
  - `regional_restoration/large`: 0.0969
  - `regional_restoration/medium`: 0.0929
  - `regional_restoration/small`: 0.0702
  - `regional_restoration/tiny`: 0.0709
  - `regional_restoration/tumor_bg_ratio`: 1.8667
  - `regional_restoration/tumor_loss`: 0.0833

**LR schedule:**
  - `LR/Generator`: peak 0.00011 @ ep 4, final 4.66e-05

**Training meta:**
  - `training/grad_norm_avg`: last 0.0312, max 4.9008 @ ep 0
  - `training/grad_norm_max`: last 0.0730, max 6.8457 @ ep 1

---
## compression_2d/multi_modality

*13 runs across 5 experiment families.*

### exp3

**exp3 (2D VAE baseline)** — early 2D multi-modality VAE at 256×256.

#### `exp3_256_20251226-120701`
*started 2025-12-26 12:07 • 125 epochs • 33h36m • 385012.6 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0116 → 0.0021 (min 0.0021 @ ep 123)
  - `Loss/L1_val`: 0.0106 → 0.0029 (min 0.0028 @ ep 70)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0164 (min 0.0162, max 0.7447)
  - `Validation/LPIPS_bravo`: last 0.0240 (min 0.0237, max 0.8098)
  - `Validation/LPIPS_flair`: last 0.0249 (min 0.0243, max 0.8001)
  - `Validation/LPIPS_t1_gd`: last 0.0095 (min 0.0092, max 0.6962)
  - `Validation/LPIPS_t1_pre`: last 0.0070 (min 0.0069, max 0.6730)
  - `Validation/MS-SSIM`: last 0.9951 (min 0.9575, max 0.9954)
  - `Validation/MS-SSIM_bravo`: last 0.9933 (min 0.9508, max 0.9938)
  - `Validation/MS-SSIM_flair`: last 0.9922 (min 0.9489, max 0.9929)
  - `Validation/MS-SSIM_t1_gd`: last 0.9973 (min 0.9643, max 0.9975)
  - `Validation/MS-SSIM_t1_pre`: last 0.9977 (min 0.9659, max 0.9978)
  - `Validation/PSNR`: last 39.8271 (min 30.9997, max 40.1674)
  - `Validation/PSNR_bravo`: last 38.0219 (min 30.6043, max 38.4306)
  - `Validation/PSNR_flair`: last 37.7397 (min 30.2323, max 38.2160)
  - `Validation/PSNR_t1_gd`: last 43.5541 (min 32.2998, max 43.7828)
  - `Validation/PSNR_t1_pre`: last 42.6715 (min 31.0972, max 42.8882)

**Regional loss (final):**
  - `regional/background_loss`: 0.0033
  - `regional/large`: 0.0268
  - `regional/medium`: 0.0242
  - `regional/small`: 0.0214
  - `regional/tiny`: 0.0189
  - `regional/tumor_bg_ratio`: 6.8347
  - `regional/tumor_loss`: 0.0227
  - `regional_bravo/background_loss`: 0.0042
  - `regional_bravo/large`: 0.0357
  - `regional_bravo/medium`: 0.0313
  - `regional_bravo/small`: 0.0255
  - `regional_bravo/tiny`: 0.0231
  - `regional_bravo/tumor_bg_ratio`: 6.7644
  - `regional_bravo/tumor_loss`: 0.0287
  - `regional_flair/background_loss`: 0.0043
  - `regional_flair/large`: 0.0322
  - `regional_flair/medium`: 0.0315
  - `regional_flair/small`: 0.0291
  - `regional_flair/tiny`: 0.0271
  - `regional_flair/tumor_bg_ratio`: 6.9244
  - `regional_flair/tumor_loss`: 0.0300
  - `regional_t1_gd/background_loss`: 0.0023
  - `regional_t1_gd/large`: 0.0253
  - `regional_t1_gd/medium`: 0.0207
  - `regional_t1_gd/small`: 0.0185
  - `regional_t1_gd/tiny`: 0.0143
  - `regional_t1_gd/tumor_bg_ratio`: 8.5623
  - `regional_t1_gd/tumor_loss`: 0.0195
  - `regional_t1_pre/background_loss`: 0.0025
  - `regional_t1_pre/large`: 0.0139
  - `regional_t1_pre/medium`: 0.0131
  - `regional_t1_pre/small`: 0.0126
  - `regional_t1_pre/tiny`: 0.0112
  - `regional_t1_pre/tumor_bg_ratio`: 5.1727
  - `regional_t1_pre/tumor_loss`: 0.0127

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 0.8518, max 15.3029 @ ep 0
  - `training/grad_norm_d_max`: last 3.8801, max 386.4064 @ ep 7
  - `training/grad_norm_g_avg`: last 0.0765, max 7.1030 @ ep 1
  - `training/grad_norm_g_max`: last 0.3468, max 282.5897 @ ep 0

### exp4

**exp4** — 2D VAE variant.

#### `exp4_256_20251226-120701`
*started 2025-12-26 12:07 • 125 epochs • 38h26m • 421119.6 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0103 → 0.0012 (min 0.0012 @ ep 124)
  - `Loss/L1_val`: 0.0063 → 0.0013 (min 0.0013 @ ep 107)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0035 (min 0.0031, max 0.3519)
  - `Validation/LPIPS_bravo`: last 0.0054 (min 0.0050, max 0.4425)
  - `Validation/LPIPS_flair`: last 0.0054 (min 0.0048, max 0.3803)
  - `Validation/LPIPS_t1_gd`: last 0.0019 (min 0.0017, max 0.3302)
  - `Validation/LPIPS_t1_pre`: last 0.0014 (min 0.0012, max 0.2816)
  - `Validation/MS-SSIM`: last 0.9992 (min 0.9836, max 0.9992)
  - `Validation/MS-SSIM_bravo`: last 0.9989 (min 0.9775, max 0.9989)
  - `Validation/MS-SSIM_flair`: last 0.9989 (min 0.9792, max 0.9989)
  - `Validation/MS-SSIM_t1_gd`: last 0.9996 (min 0.9879, max 0.9996)
  - `Validation/MS-SSIM_t1_pre`: last 0.9996 (min 0.9897, max 0.9996)
  - `Validation/PSNR`: last 46.8062 (min 34.3325, max 46.8109)
  - `Validation/PSNR_bravo`: last 45.1774 (min 33.0707, max 45.1902)
  - `Validation/PSNR_flair`: last 44.6956 (min 33.3070, max 44.7099)
  - `Validation/PSNR_t1_gd`: last 50.5390 (min 36.2836, max 50.5390)
  - `Validation/PSNR_t1_pre`: last 49.1589 (min 35.4071, max 49.1694)

**Regional loss (final):**
  - `regional/background_loss`: 0.0015
  - `regional/large`: 0.0108
  - `regional/medium`: 0.0098
  - `regional/small`: 0.0086
  - `regional/tiny`: 0.0075
  - `regional/tumor_bg_ratio`: 6.0634
  - `regional/tumor_loss`: 0.0091
  - `regional_bravo/background_loss`: 0.0020
  - `regional_bravo/large`: 0.0142
  - `regional_bravo/medium`: 0.0133
  - `regional_bravo/small`: 0.0103
  - `regional_bravo/tiny`: 0.0093
  - `regional_bravo/tumor_bg_ratio`: 5.9611
  - `regional_bravo/tumor_loss`: 0.0118
  - `regional_flair/background_loss`: 0.0019
  - `regional_flair/large`: 0.0139
  - `regional_flair/medium`: 0.0120
  - `regional_flair/small`: 0.0108
  - `regional_flair/tiny`: 0.0101
  - `regional_flair/tumor_bg_ratio`: 6.2442
  - `regional_flair/tumor_loss`: 0.0116
  - `regional_t1_gd/background_loss`: 0.0010
  - `regional_t1_gd/large`: 0.0091
  - `regional_t1_gd/medium`: 0.0080
  - `regional_t1_gd/small`: 0.0071
  - `regional_t1_gd/tiny`: 0.0057
  - `regional_t1_gd/tumor_bg_ratio`: 7.1515
  - `regional_t1_gd/tumor_loss`: 0.0075
  - `regional_t1_pre/background_loss`: 0.0011
  - `regional_t1_pre/large`: 0.0061
  - `regional_t1_pre/medium`: 0.0058
  - `regional_t1_pre/small`: 0.0059
  - `regional_t1_pre/tiny`: 0.0050
  - `regional_t1_pre/tumor_bg_ratio`: 4.9438
  - `regional_t1_pre/tumor_loss`: 0.0057

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 0.0106, max 5.9917 @ ep 2
  - `training/grad_norm_d_max`: last 0.0834, max 216.4629 @ ep 2
  - `training/grad_norm_g_avg`: last 0.0668, max 7.4929 @ ep 2
  - `training/grad_norm_g_max`: last 0.2329, max 73.2941 @ ep 0

### exp5

**exp5** — 2D VAE variant.

#### `exp5_256_20251226-230709`
*started 2025-12-26 23:07 • 125 epochs • 32h56m*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0179 → 0.0064 (min 0.0063 @ ep 56)
  - `Loss/L1_val`: 0.0128 → 0.0068 (min 0.0059 @ ep 46)

**Validation quality:**
  - `Validation/LPIPS`: last 0.1717 (min 0.1598, max 0.8073)
  - `Validation/LPIPS_bravo`: last 0.2175 (min 0.2116, max 0.8494)
  - `Validation/LPIPS_flair`: last 0.2050 (min 0.2000, max 0.8457)
  - `Validation/LPIPS_t1_gd`: last 0.1455 (min 0.0948, max 0.7686)
  - `Validation/LPIPS_t1_pre`: last 0.1188 (min 0.0871, max 0.7651)
  - `Validation/MS-SSIM`: last 0.9758 (min 0.9133, max 0.9803)
  - `Validation/MS-SSIM_bravo`: last 0.9698 (min 0.9041, max 0.9746)
  - `Validation/MS-SSIM_flair`: last 0.9695 (min 0.9048, max 0.9744)
  - `Validation/MS-SSIM_t1_gd`: last 0.9805 (min 0.9256, max 0.9855)
  - `Validation/MS-SSIM_t1_pre`: last 0.9832 (min 0.9189, max 0.9867)
  - `Validation/PSNR`: last 33.0470 (min 29.4808, max 34.1458)
  - `Validation/PSNR_bravo`: last 31.9957 (min 29.3146, max 32.8892)
  - `Validation/PSNR_flair`: last 32.0510 (min 28.7720, max 33.0796)
  - `Validation/PSNR_t1_gd`: last 34.6056 (min 31.0524, max 36.1639)
  - `Validation/PSNR_t1_pre`: last 34.1014 (min 29.1062, max 35.3093)

**Regional loss (final):**
  - `regional/background_loss`: 0.0076
  - `regional/large`: 0.0546
  - `regional/medium`: 0.0480
  - `regional/small`: 0.0467
  - `regional/tiny`: 0.0403
  - `regional/tumor_bg_ratio`: 6.1993
  - `regional/tumor_loss`: 0.0472
  - `regional_bravo/background_loss`: 0.0085
  - `regional_bravo/large`: 0.0624
  - `regional_bravo/medium`: 0.0548
  - `regional_bravo/small`: 0.0494
  - `regional_bravo/tiny`: 0.0439
  - `regional_bravo/tumor_bg_ratio`: 6.1455
  - `regional_bravo/tumor_loss`: 0.0523
  - `regional_flair/background_loss`: 0.0088
  - `regional_flair/large`: 0.0614
  - `regional_flair/medium`: 0.0576
  - `regional_flair/small`: 0.0577
  - `regional_flair/tiny`: 0.0510
  - `regional_flair/tumor_bg_ratio`: 6.4416
  - `regional_flair/tumor_loss`: 0.0568
  - `regional_t1_gd/background_loss`: 0.0063
  - `regional_t1_gd/large`: 0.0578
  - `regional_t1_gd/medium`: 0.0480
  - `regional_t1_gd/small`: 0.0462
  - `regional_t1_gd/tiny`: 0.0360
  - `regional_t1_gd/tumor_bg_ratio`: 7.4422
  - `regional_t1_gd/tumor_loss`: 0.0466
  - `regional_t1_pre/background_loss`: 0.0069
  - `regional_t1_pre/large`: 0.0369
  - `regional_t1_pre/medium`: 0.0318
  - `regional_t1_pre/small`: 0.0336
  - `regional_t1_pre/tiny`: 0.0304
  - `regional_t1_pre/tumor_bg_ratio`: 4.8037
  - `regional_t1_pre/tumor_loss`: 0.0329

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 19.9477, max 54.7356 @ ep 92
  - `training/grad_norm_d_max`: last 236.0000, max 384.0000 @ ep 2
  - `training/grad_norm_g_avg`: last 3.6694, max 8.7263 @ ep 1
  - `training/grad_norm_g_max`: last 17.5000, max 108.0000 @ ep 0

### exp6

**exp6 (2D VAE refined)** — two variants (base + 6_1) at 256×256.

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp6_1_256_20251228-190109` — 40.3707
  2. 🥈 `exp6_256_20251228-190109` — 37.4540

#### `exp6_1_256_20251228-190109`
*started 2025-12-28 19:01 • 125 epochs • 9h03m • 27030.0 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0126 → 0.0024 (min 0.0023 @ ep 107)
  - `Loss/L1_val`: 0.0058 → 0.0026 (min 0.0024 @ ep 85)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0078 (min 0.0077, max 0.1939)
  - `Validation/LPIPS_bravo`: last 0.0125 (min 0.0124, max 0.3137)
  - `Validation/LPIPS_flair`: last 0.0106 (min 0.0104, max 0.2522)
  - `Validation/LPIPS_t1_gd`: last 0.0046 (min 0.0045, max 0.1138)
  - `Validation/LPIPS_t1_pre`: last 0.0036 (min 0.0036, max 0.0957)
  - `Validation/MS-SSIM`: last 0.9975 (min 0.9891, max 0.9977)
  - `Validation/MS-SSIM_bravo`: last 0.9964 (min 0.9856, max 0.9967)
  - `Validation/MS-SSIM_flair`: last 0.9964 (min 0.9863, max 0.9967)
  - `Validation/MS-SSIM_t1_gd`: last 0.9985 (min 0.9922, max 0.9986)
  - `Validation/MS-SSIM_t1_pre`: last 0.9986 (min 0.9924, max 0.9988)
  - `Validation/PSNR`: last 41.6809 (min 35.2185, max 42.1174)
  - `Validation/PSNR_bravo`: last 39.9673 (min 33.8186, max 40.3707)
  - `Validation/PSNR_flair`: last 39.9004 (min 34.3341, max 40.3732)
  - `Validation/PSNR_t1_gd`: last 45.1063 (min 37.4684, max 45.5626)
  - `Validation/PSNR_t1_pre`: last 43.8835 (min 36.1746, max 44.5393)

**Regional loss (final):**
  - `regional/background_loss`: 0.0029
  - `regional/large`: 0.0192
  - `regional/medium`: 0.0193
  - `regional/small`: 0.0167
  - `regional/tiny`: 0.0147
  - `regional/tumor_bg_ratio`: 6.0919
  - `regional/tumor_loss`: 0.0175
  - `regional_bravo/background_loss`: 0.0037
  - `regional_bravo/large`: 0.0240
  - `regional_bravo/medium`: 0.0244
  - `regional_bravo/small`: 0.0194
  - `regional_bravo/tiny`: 0.0180
  - `regional_bravo/tumor_bg_ratio`: 5.8446
  - `regional_bravo/tumor_loss`: 0.0215
  - `regional_flair/background_loss`: 0.0035
  - `regional_flair/large`: 0.0246
  - `regional_flair/medium`: 0.0243
  - `regional_flair/small`: 0.0217
  - `regional_flair/tiny`: 0.0199
  - `regional_flair/tumor_bg_ratio`: 6.4590
  - `regional_flair/tumor_loss`: 0.0226
  - `regional_t1_gd/background_loss`: 0.0020
  - `regional_t1_gd/large`: 0.0166
  - `regional_t1_gd/medium`: 0.0164
  - `regional_t1_gd/small`: 0.0144
  - `regional_t1_gd/tiny`: 0.0113
  - `regional_t1_gd/tumor_bg_ratio`: 7.1938
  - `regional_t1_gd/tumor_loss`: 0.0147
  - `regional_t1_pre/background_loss`: 0.0022
  - `regional_t1_pre/large`: 0.0115
  - `regional_t1_pre/medium`: 0.0119
  - `regional_t1_pre/small`: 0.0112
  - `regional_t1_pre/tiny`: 0.0096
  - `regional_t1_pre/tumor_bg_ratio`: 4.9351
  - `regional_t1_pre/tumor_loss`: 0.0111

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 1.5619, max 22.2413 @ ep 3
  - `training/grad_norm_d_max`: last 5.4676, max 184.0850 @ ep 4
  - `training/grad_norm_g_avg`: last 0.1286, max 0.9928 @ ep 107
  - `training/grad_norm_g_max`: last 0.7691, max 3.2549 @ ep 1

#### `exp6_256_20251228-190109`
*started 2025-12-28 19:01 • 125 epochs • 9h27m • 30851.3 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0183 → 0.0031 (min 0.0031 @ ep 107)
  - `Loss/L1_val`: 0.0076 → 0.0034 (min 0.0033 @ ep 107)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0527 (min 0.0338, max 0.2934)
  - `Validation/LPIPS_bravo`: last 0.0741 (min 0.0480, max 0.4023)
  - `Validation/LPIPS_flair`: last 0.0691 (min 0.0453, max 0.3844)
  - `Validation/LPIPS_t1_gd`: last 0.0373 (min 0.0228, max 0.1961)
  - `Validation/LPIPS_t1_pre`: last 0.0305 (min 0.0191, max 0.1911)
  - `Validation/MS-SSIM`: last 0.9935 (min 0.9667, max 0.9936)
  - `Validation/MS-SSIM_bravo`: last 0.9915 (min 0.9587, max 0.9915)
  - `Validation/MS-SSIM_flair`: last 0.9903 (min 0.9616, max 0.9903)
  - `Validation/MS-SSIM_t1_gd`: last 0.9960 (min 0.9742, max 0.9960)
  - `Validation/MS-SSIM_t1_pre`: last 0.9965 (min 0.9724, max 0.9965)
  - `Validation/PSNR`: last 39.0083 (min 32.5247, max 39.0226)
  - `Validation/PSNR_bravo`: last 37.4249 (min 31.5283, max 37.4540)
  - `Validation/PSNR_flair`: last 37.2506 (min 31.7685, max 37.2899)
  - `Validation/PSNR_t1_gd`: last 42.0336 (min 34.4703, max 42.0553)
  - `Validation/PSNR_t1_pre`: last 41.0991 (min 32.8524, max 41.1236)

**Regional loss (final):**
  - `regional/background_loss`: 0.0038
  - `regional/large`: 0.0333
  - `regional/medium`: 0.0284
  - `regional/small`: 0.0251
  - `regional/tiny`: 0.0215
  - `regional/tumor_bg_ratio`: 7.0156
  - `regional/tumor_loss`: 0.0269
  - `regional_bravo/background_loss`: 0.0046
  - `regional_bravo/large`: 0.0412
  - `regional_bravo/medium`: 0.0349
  - `regional_bravo/small`: 0.0279
  - `regional_bravo/tiny`: 0.0255
  - `regional_bravo/tumor_bg_ratio`: 6.9535
  - `regional_bravo/tumor_loss`: 0.0322
  - `regional_flair/background_loss`: 0.0048
  - `regional_flair/large`: 0.0390
  - `regional_flair/medium`: 0.0358
  - `regional_flair/small`: 0.0333
  - `regional_flair/tiny`: 0.0295
  - `regional_flair/tumor_bg_ratio`: 7.1528
  - `regional_flair/tumor_loss`: 0.0343
  - `regional_t1_gd/background_loss`: 0.0028
  - `regional_t1_gd/large`: 0.0354
  - `regional_t1_gd/medium`: 0.0266
  - `regional_t1_gd/small`: 0.0232
  - `regional_t1_gd/tiny`: 0.0176
  - `regional_t1_gd/tumor_bg_ratio`: 9.0370
  - `regional_t1_gd/tumor_loss`: 0.0254
  - `regional_t1_pre/background_loss`: 0.0031
  - `regional_t1_pre/large`: 0.0175
  - `regional_t1_pre/medium`: 0.0165
  - `regional_t1_pre/small`: 0.0158
  - `regional_t1_pre/tiny`: 0.0136
  - `regional_t1_pre/tumor_bg_ratio`: 5.0785
  - `regional_t1_pre/tumor_loss`: 0.0158

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 0.3457, max 19.8534 @ ep 1
  - `training/grad_norm_d_max`: last 11.6364, max 494.2174 @ ep 1
  - `training/grad_norm_g_avg`: last 0.1041, max 0.8774 @ ep 107
  - `training/grad_norm_g_max`: last 0.5851, max 3.2941 @ ep 89

### exp9

**exp9 (2D VAE + DC-AE sweep)** — sweep of compression ratios (f32, f64, f128)
and architectures. VAE variants first; DC-AE variants added later
(`exp9_dcae_*`). Per CLAUDE.md the DC-AE branch was abandoned,
so `exp9_dcae_*` results are not load-bearing for current work.
Also includes `exp9_1_f32_phase3` (multi-phase training) and
`exp9_4_f128_lpips` (LPIPS aux loss at f128).

**Family ranking by `Validation/PSNR_bravo` (PSNR ↑):**
  1. 🥇 `exp9_4_f128_lpips_20260111-145520` — 38.9554
  2. 🥈 `exp9_f128_20260106-165531` — 38.6998
  3.  `exp9_dcae_f128_1_5_20260221-082421` — 35.9491
  4.  `exp9_dcae_f64_1_5_20260221-025729` — 35.4299
  5.  `exp9_dcae_f32_1_5_20260221-010858` — 33.9084
  6.  `exp9_1_f32_phase3_20260111-013053` — 33.6927

#### `exp9_f32_20251231-175024`
*started 2025-12-31 17:50 • 125 epochs • 63h56m*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0220 → 0.0065 (min 0.0065 @ ep 124)
  - `Loss/L1_val`: 0.0078 → 0.0050 (min 0.0050 @ ep 124)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0257 (min 0.0247, max 0.1145)
  - `Validation/MS-SSIM`: last 0.9837 (min 0.9625, max 0.9837)
  - `Validation/PSNR`: last 35.0051 (min 32.0815, max 35.0088)

**Regional loss (final):**
  - `regional/background_loss`: 0.0056
  - `regional/large`: 0.0421
  - `regional/medium`: 0.0398
  - `regional/small`: 0.0356
  - `regional/tiny`: 0.0314
  - `regional/tumor_bg_ratio`: 6.6577
  - `regional/tumor_loss`: 0.0372

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_g_avg`: last 0.4499, max 9.8685 @ ep 0
  - `training/grad_norm_g_max`: last 0.8399, max 498.5181 @ ep 1

#### `exp9_f64_20251231-175024`
*started 2025-12-31 17:50 • 125 epochs • 64h06m*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0248 → 0.0045 (min 0.0045 @ ep 124)
  - `Loss/L1_val`: 0.0081 → 0.0030 (min 0.0030 @ ep 124)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0071 (min 0.0071, max 0.1517)
  - `Validation/MS-SSIM`: last 0.9962 (min 0.9636, max 0.9962)
  - `Validation/PSNR`: last 39.9799 (min 32.0560, max 39.9799)

**Regional loss (final):**
  - `regional/background_loss`: 0.0032
  - `regional/large`: 0.0212
  - `regional/medium`: 0.0212
  - `regional/small`: 0.0188
  - `regional/tiny`: 0.0168
  - `regional/tumor_bg_ratio`: 6.0168
  - `regional/tumor_loss`: 0.0195

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_g_avg`: last 0.4552, max 10.5213 @ ep 1
  - `training/grad_norm_g_max`: last 0.8164, max 748.9611 @ ep 1

#### `exp9_f128_20260106-165531`
*started 2026-01-06 16:55 • 120 epochs • 71h04m • 1881.0 TFLOPs • peak VRAM 27.5 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0244 → 0.0042 (min 0.0042 @ ep 119)
  - `Loss/L1_val`: 0.0150 → 0.0027 (min 0.0027 @ ep 117)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0056 (min 0.0056, max 0.5869)
  - `Validation/LPIPS_bravo`: last 0.0098 (min 0.0098, max 0.6446)
  - `Validation/LPIPS_flair`: last 0.0073 (min 0.0073, max 0.6409)
  - `Validation/LPIPS_t1_gd`: last 0.0029 (min 0.0029, max 0.5406)
  - `Validation/LPIPS_t1_pre`: last 0.0025 (min 0.0025, max 0.5216)
  - `Validation/MS-SSIM`: last 0.9970 (min 0.8911, max 0.9970)
  - `Validation/MS-SSIM_bravo`: last 0.9953 (min 0.8892, max 0.9953)
  - `Validation/MS-SSIM_flair`: last 0.9956 (min 0.8852, max 0.9956)
  - `Validation/MS-SSIM_t1_gd`: last 0.9984 (min 0.9013, max 0.9984)
  - `Validation/MS-SSIM_t1_pre`: last 0.9987 (min 0.8887, max 0.9987)
  - `Validation/PSNR`: last 40.7560 (min 28.6011, max 40.7560)
  - `Validation/PSNR_bravo`: last 38.6975 (min 28.9424, max 38.6998)
  - `Validation/PSNR_flair`: last 39.2132 (min 28.0701, max 39.2149)
  - `Validation/PSNR_t1_gd`: last 44.0730 (min 29.9333, max 44.0730)
  - `Validation/PSNR_t1_pre`: last 43.3493 (min 27.7551, max 43.3493)

**Regional loss (final):**
  - `regional/background_loss`: 0.0030
  - `regional/large`: 0.0191
  - `regional/medium`: 0.0192
  - `regional/small`: 0.0168
  - `regional/tiny`: 0.0152
  - `regional/tumor_bg_ratio`: 5.9326
  - `regional/tumor_loss`: 0.0176
  - `regional_bravo/background_loss`: 0.0040
  - `regional_bravo/large`: 0.0248
  - `regional_bravo/medium`: 0.0246
  - `regional_bravo/small`: 0.0200
  - `regional_bravo/tiny`: 0.0190
  - `regional_bravo/tumor_bg_ratio`: 5.5312
  - `regional_bravo/tumor_loss`: 0.0221
  - `regional_flair/background_loss`: 0.0036
  - `regional_flair/large`: 0.0242
  - `regional_flair/medium`: 0.0241
  - `regional_flair/small`: 0.0215
  - `regional_flair/tiny`: 0.0213
  - `regional_flair/tumor_bg_ratio`: 6.2716
  - `regional_flair/tumor_loss`: 0.0228
  - `regional_t1_gd/background_loss`: 0.0021
  - `regional_t1_gd/large`: 0.0168
  - `regional_t1_gd/medium`: 0.0166
  - `regional_t1_gd/small`: 0.0146
  - `regional_t1_gd/tiny`: 0.0113
  - `regional_t1_gd/tumor_bg_ratio`: 7.2407
  - `regional_t1_gd/tumor_loss`: 0.0149
  - `regional_t1_pre/background_loss`: 0.0022
  - `regional_t1_pre/large`: 0.0107
  - `regional_t1_pre/medium`: 0.0114
  - `regional_t1_pre/small`: 0.0111
  - `regional_t1_pre/tiny`: 0.0095
  - `regional_t1_pre/tumor_bg_ratio`: 4.8756
  - `regional_t1_pre/tumor_loss`: 0.0107

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1.423e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.4516, max 9.6281 @ ep 0
  - `training/grad_norm_max`: last 0.9580, max 261.2724 @ ep 0

#### `exp9_1_f32_phase3_20260111-013053`
*started 2026-01-11 01:30 • 175 epochs • 14h22m • 2677.9 TFLOPs • peak VRAM 7.1 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0066 → 0.0081 (min 0.0066 @ ep 124)
  - `Loss/L1_val`: 0.0051 → 0.0064 (min 0.0051 @ ep 124)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0432 (min 0.0266, max 0.0819)
  - `Validation/LPIPS_bravo`: last 0.0814 (min 0.0406, max 0.1654)
  - `Validation/LPIPS_flair`: last 0.0756 (min 0.0357, max 0.1507)
  - `Validation/LPIPS_t1_gd`: last 0.0454 (min 0.0162, max 0.1160)
  - `Validation/LPIPS_t1_pre`: last 0.0456 (min 0.0156, max 0.1190)
  - `Validation/MS-SSIM`: last 0.9806 (min 0.9776, max 0.9828)
  - `Validation/MS-SSIM_bravo`: last 0.9758 (min 0.9719, max 0.9786)
  - `Validation/MS-SSIM_flair`: last 0.9744 (min 0.9710, max 0.9772)
  - `Validation/MS-SSIM_t1_gd`: last 0.9874 (min 0.9844, max 0.9894)
  - `Validation/MS-SSIM_t1_pre`: last 0.9878 (min 0.9855, max 0.9899)
  - `Validation/PSNR`: last 34.0297 (min 33.3676, max 35.0522)
  - `Validation/PSNR_bravo`: last 32.8329 (min 32.3072, max 33.6927)
  - `Validation/PSNR_flair`: last 32.5437 (min 31.9872, max 33.5618)
  - `Validation/PSNR_t1_gd`: last 36.6381 (min 35.7005, max 37.7924)
  - `Validation/PSNR_t1_pre`: last 35.2302 (min 34.3733, max 36.5707)

**Regional loss (final):**
  - `regional/background_loss`: 0.0071
  - `regional/large`: 0.0470
  - `regional/medium`: 0.0460
  - `regional/small`: 0.0408
  - `regional/tiny`: 0.0357
  - `regional/tumor_bg_ratio`: 6.0156
  - `regional/tumor_loss`: 0.0424
  - `regional_bravo/background_loss`: 0.0083
  - `regional_bravo/large`: 0.0544
  - `regional_bravo/medium`: 0.0532
  - `regional_bravo/small`: 0.0443
  - `regional_bravo/tiny`: 0.0403
  - `regional_bravo/tumor_bg_ratio`: 5.8347
  - `regional_bravo/tumor_loss`: 0.0481
  - `regional_flair/background_loss`: 0.0082
  - `regional_flair/large`: 0.0587
  - `regional_flair/medium`: 0.0601
  - `regional_flair/small`: 0.0523
  - `regional_flair/tiny`: 0.0470
  - `regional_flair/tumor_bg_ratio`: 6.7034
  - `regional_flair/tumor_loss`: 0.0547
  - `regional_t1_gd/background_loss`: 0.0055
  - `regional_t1_gd/large`: 0.0461
  - `regional_t1_gd/medium`: 0.0436
  - `regional_t1_gd/small`: 0.0389
  - `regional_t1_gd/tiny`: 0.0293
  - `regional_t1_gd/tumor_bg_ratio`: 7.2433
  - `regional_t1_gd/tumor_loss`: 0.0395
  - `regional_t1_pre/background_loss`: 0.0064
  - `regional_t1_pre/large`: 0.0282
  - `regional_t1_pre/medium`: 0.0270
  - `regional_t1_pre/small`: 0.0278
  - `regional_t1_pre/tiny`: 0.0262
  - `regional_t1_pre/tumor_bg_ratio`: 4.2828
  - `regional_t1_pre/tumor_loss`: 0.0272

**LR schedule:**
  - `LR/Discriminator`: peak 5.4e-06 @ ep 124, final 4.572e-06
  - `LR/Generator`: peak 3.894e-05 @ ep 174, final 3.894e-05

**Training meta:**
  - `training/grad_norm_d_avg`: last 12.7262, max 20.2692 @ ep 163
  - `training/grad_norm_d_max`: last 258.4632, max 817.0479 @ ep 159
  - `training/grad_norm_g_avg`: last 1.2381, max 1.9194 @ ep 163
  - `training/grad_norm_g_max`: last 10.2626, max 11.6237 @ ep 171

#### `exp9_4_f128_lpips_20260111-145520`
*started 2026-01-11 14:55 • 109 epochs • 71h09m • 1708.6 TFLOPs • peak VRAM 30.0 GB*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0190 → 0.0045 (min 0.0045 @ ep 108)
  - `Loss/L1_val`: 0.0107 → 0.0026 (min 0.0026 @ ep 108)

**Validation quality:**
  - `Validation/LPIPS`: last 0.0239 (min 0.0233, max 1.1583)
  - `Validation/LPIPS_bravo`: last 0.0399 (min 0.0393, max 1.1318)
  - `Validation/LPIPS_flair`: last 0.0349 (min 0.0340, max 1.1038)
  - `Validation/LPIPS_t1_gd`: last 0.0130 (min 0.0124, max 1.1066)
  - `Validation/LPIPS_t1_pre`: last 0.0094 (min 0.0089, max 1.1147)
  - `Validation/MS-SSIM`: last 0.9964 (min 0.3994, max 0.9965)
  - `Validation/MS-SSIM_bravo`: last 0.9953 (min 0.4182, max 0.9953)
  - `Validation/MS-SSIM_flair`: last 0.9957 (min 0.4187, max 0.9957)
  - `Validation/MS-SSIM_t1_gd`: last 0.9984 (min 0.4064, max 0.9984)
  - `Validation/MS-SSIM_t1_pre`: last 0.9987 (min 0.3544, max 0.9987)
  - `Validation/PSNR`: last 41.0880 (min 19.0479, max 41.1108)
  - `Validation/PSNR_bravo`: last 38.9489 (min 20.4992, max 38.9554)
  - `Validation/PSNR_flair`: last 39.6835 (min 19.4944, max 39.6835)
  - `Validation/PSNR_t1_gd`: last 44.4581 (min 19.5642, max 44.4599)
  - `Validation/PSNR_t1_pre`: last 43.6547 (min 17.2599, max 43.6653)

**Regional loss (final):**
  - `regional/background_loss`: 0.0028
  - `regional/large`: 0.0191
  - `regional/medium`: 0.0193
  - `regional/small`: 0.0168
  - `regional/tiny`: 0.0152
  - `regional/tumor_bg_ratio`: 6.2142
  - `regional/tumor_loss`: 0.0176
  - `regional_bravo/background_loss`: 0.0038
  - `regional_bravo/large`: 0.0250
  - `regional_bravo/medium`: 0.0250
  - `regional_bravo/small`: 0.0202
  - `regional_bravo/tiny`: 0.0188
  - `regional_bravo/tumor_bg_ratio`: 5.8469
  - `regional_bravo/tumor_loss`: 0.0223
  - `regional_flair/background_loss`: 0.0034
  - `regional_flair/large`: 0.0236
  - `regional_flair/medium`: 0.0239
  - `regional_flair/small`: 0.0214
  - `regional_flair/tiny`: 0.0211
  - `regional_flair/tumor_bg_ratio`: 6.5745
  - `regional_flair/tumor_loss`: 0.0225
  - `regional_t1_gd/background_loss`: 0.0020
  - `regional_t1_gd/large`: 0.0167
  - `regional_t1_gd/medium`: 0.0167
  - `regional_t1_gd/small`: 0.0146
  - `regional_t1_gd/tiny`: 0.0113
  - `regional_t1_gd/tumor_bg_ratio`: 7.5125
  - `regional_t1_gd/tumor_loss`: 0.0149
  - `regional_t1_pre/background_loss`: 0.0021
  - `regional_t1_pre/large`: 0.0109
  - `regional_t1_pre/medium`: 0.0116
  - `regional_t1_pre/small`: 0.0110
  - `regional_t1_pre/tiny`: 0.0094
  - `regional_t1_pre/tumor_bg_ratio`: 5.0810
  - `regional_t1_pre/tumor_loss`: 0.0108

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 5.279e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1259, max 4.8839 @ ep 0
  - `training/grad_norm_max`: last 0.6066, max 49.8144 @ ep 1

#### `exp9_dcae_f32_1_5_20260221-010858`
*started 2026-02-21 01:08 • 125 epochs • 80h57m • 1120002.4 TFLOPs • peak VRAM 74.4 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0329 (min 0.0309, max 1.0568)
  - `Validation/LPIPS_bravo`: last 0.0466 (min 0.0442, max 1.1696)
  - `Validation/LPIPS_flair`: last 0.0443 (min 0.0409, max 1.1451)
  - `Validation/LPIPS_t1_gd`: last 0.0217 (min 0.0206, max 1.1325)
  - `Validation/LPIPS_t1_pre`: last 0.0217 (min 0.0206, max 1.1270)
  - `Validation/MS-SSIM`: last 0.9763 (min 0.3799, max 0.9763)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9781 (min 0.3891, max 0.9781)
  - `Validation/MS-SSIM-3D_flair`: last 0.9765 (min 0.3686, max 0.9766)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9885 (min 0.3664, max 0.9885)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9892 (min 0.3300, max 0.9892)
  - `Validation/MS-SSIM_bravo`: last 0.9713 (min 0.3972, max 0.9713)
  - `Validation/MS-SSIM_flair`: last 0.9689 (min 0.3986, max 0.9689)
  - `Validation/MS-SSIM_t1_gd`: last 0.9845 (min 0.3861, max 0.9845)
  - `Validation/MS-SSIM_t1_pre`: last 0.9847 (min 0.3367, max 0.9847)
  - `Validation/PSNR`: last 35.7551 (min 20.8673, max 35.7561)
  - `Validation/PSNR_bravo`: last 33.9033 (min 21.8288, max 33.9084)
  - `Validation/PSNR_flair`: last 34.6108 (min 21.5618, max 34.6150)
  - `Validation/PSNR_t1_gd`: last 37.7625 (min 21.1056, max 37.7672)
  - `Validation/PSNR_t1_pre`: last 36.7336 (min 18.9776, max 36.7340)

**Regional loss (final):**
  - `regional/background_loss`: 0.0064
  - `regional/large`: 0.0527
  - `regional/medium`: 0.0472
  - `regional/small`: 0.0420
  - `regional/tiny`: 0.0366
  - `regional/tumor_bg_ratio`: 6.9958
  - `regional/tumor_loss`: 0.0445
  - `regional_bravo/background_loss`: 0.0074
  - `regional_bravo/large`: 0.0616
  - `regional_bravo/medium`: 0.0555
  - `regional_bravo/small`: 0.0463
  - `regional_bravo/tiny`: 0.0412
  - `regional_bravo/tumor_bg_ratio`: 6.9200
  - `regional_bravo/tumor_loss`: 0.0510
  - `regional_flair/background_loss`: 0.0077
  - `regional_flair/large`: 0.0610
  - `regional_flair/medium`: 0.0583
  - `regional_flair/small`: 0.0525
  - `regional_flair/tiny`: 0.0490
  - `regional_flair/tumor_bg_ratio`: 7.1905
  - `regional_flair/tumor_loss`: 0.0552
  - `regional_t1_gd/background_loss`: 0.0047
  - `regional_t1_gd/large`: 0.0556
  - `regional_t1_gd/medium`: 0.0474
  - `regional_t1_gd/small`: 0.0411
  - `regional_t1_gd/tiny`: 0.0302
  - `regional_t1_gd/tumor_bg_ratio`: 9.1532
  - `regional_t1_gd/tumor_loss`: 0.0433
  - `regional_t1_pre/background_loss`: 0.0057
  - `regional_t1_pre/large`: 0.0325
  - `regional_t1_pre/medium`: 0.0278
  - `regional_t1_pre/small`: 0.0282
  - `regional_t1_pre/tiny`: 0.0259
  - `regional_t1_pre/tumor_bg_ratio`: 5.0159
  - `regional_t1_pre/tumor_loss`: 0.0284

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1.423e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.7293, max 10.9589 @ ep 1
  - `training/grad_norm_max`: last 1.1485, max 44.5634 @ ep 1

#### `exp9_dcae_f64_1_5_20260221-025729`
*started 2026-02-21 02:57 • 125 epochs • 93h23m • 1120300.9 TFLOPs • peak VRAM 74.4 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0218 (min 0.0215, max 0.2968)
  - `Validation/LPIPS_bravo`: last 0.0362 (min 0.0356, max 0.4438)
  - `Validation/LPIPS_flair`: last 0.0310 (min 0.0305, max 0.4408)
  - `Validation/LPIPS_t1_gd`: last 0.0113 (min 0.0112, max 0.3492)
  - `Validation/LPIPS_t1_pre`: last 0.0104 (min 0.0103, max 0.3251)
  - `Validation/MS-SSIM`: last 0.9869 (min 0.8479, max 0.9869)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9876 (min 0.8483, max 0.9877)
  - `Validation/MS-SSIM-3D_flair`: last 0.9872 (min 0.8250, max 0.9872)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9947 (min 0.8619, max 0.9947)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9955 (min 0.8014, max 0.9955)
  - `Validation/MS-SSIM_bravo`: last 0.9832 (min 0.8554, max 0.9832)
  - `Validation/MS-SSIM_flair`: last 0.9822 (min 0.8362, max 0.9822)
  - `Validation/MS-SSIM_t1_gd`: last 0.9924 (min 0.8709, max 0.9924)
  - `Validation/MS-SSIM_t1_pre`: last 0.9933 (min 0.8276, max 0.9933)
  - `Validation/PSNR`: last 37.7762 (min 27.6178, max 37.7817)
  - `Validation/PSNR_bravo`: last 35.4219 (min 28.4477, max 35.4299)
  - `Validation/PSNR_flair`: last 36.3540 (min 27.2477, max 36.3599)
  - `Validation/PSNR_t1_gd`: last 39.9595 (min 28.9912, max 39.9639)
  - `Validation/PSNR_t1_pre`: last 39.3684 (min 25.7811, max 39.3750)

**Regional loss (final):**
  - `regional/background_loss`: 0.0049
  - `regional/large`: 0.0387
  - `regional/medium`: 0.0361
  - `regional/small`: 0.0324
  - `regional/tiny`: 0.0277
  - `regional/tumor_bg_ratio`: 6.8891
  - `regional/tumor_loss`: 0.0337
  - `regional_bravo/background_loss`: 0.0060
  - `regional_bravo/large`: 0.0464
  - `regional_bravo/medium`: 0.0437
  - `regional_bravo/small`: 0.0366
  - `regional_bravo/tiny`: 0.0326
  - `regional_bravo/tumor_bg_ratio`: 6.6115
  - `regional_bravo/tumor_loss`: 0.0398
  - `regional_flair/background_loss`: 0.0061
  - `regional_flair/large`: 0.0468
  - `regional_flair/medium`: 0.0448
  - `regional_flair/small`: 0.0416
  - `regional_flair/tiny`: 0.0380
  - `regional_flair/tumor_bg_ratio`: 7.0521
  - `regional_flair/tumor_loss`: 0.0428
  - `regional_t1_gd/background_loss`: 0.0035
  - `regional_t1_gd/large`: 0.0385
  - `regional_t1_gd/medium`: 0.0344
  - `regional_t1_gd/small`: 0.0309
  - `regional_t1_gd/tiny`: 0.0221
  - `regional_t1_gd/tumor_bg_ratio`: 8.9639
  - `regional_t1_gd/tumor_loss`: 0.0314
  - `regional_t1_pre/background_loss`: 0.0039
  - `regional_t1_pre/large`: 0.0230
  - `regional_t1_pre/medium`: 0.0214
  - `regional_t1_pre/small`: 0.0208
  - `regional_t1_pre/tiny`: 0.0181
  - `regional_t1_pre/tumor_bg_ratio`: 5.2633
  - `regional_t1_pre/tumor_loss`: 0.0208

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1.609e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.7441, max 9.1752 @ ep 1
  - `training/grad_norm_max`: last 1.4304, max 72.3782 @ ep 3

#### `exp9_dcae_f128_1_5_20260221-082421`
*started 2026-02-21 08:24 • 125 epochs • 121h04m • 1121494.8 TFLOPs • peak VRAM 74.4 GB*

**Validation quality:**
  - `Validation/LPIPS`: last 0.0178 (min 0.0177, max 0.3278)
  - `Validation/LPIPS_bravo`: last 0.0290 (min 0.0288, max 0.4572)
  - `Validation/LPIPS_flair`: last 0.0257 (min 0.0252, max 0.4399)
  - `Validation/LPIPS_t1_gd`: last 0.0091 (min 0.0091, max 0.3825)
  - `Validation/LPIPS_t1_pre`: last 0.0083 (min 0.0083, max 0.3508)
  - `Validation/MS-SSIM`: last 0.9887 (min 0.8589, max 0.9887)
  - `Validation/MS-SSIM-3D_bravo`: last 0.9896 (min 0.8537, max 0.9896)
  - `Validation/MS-SSIM-3D_flair`: last 0.9890 (min 0.8431, max 0.9890)
  - `Validation/MS-SSIM-3D_t1_gd`: last 0.9955 (min 0.8668, max 0.9955)
  - `Validation/MS-SSIM-3D_t1_pre`: last 0.9962 (min 0.8441, max 0.9962)
  - `Validation/MS-SSIM_bravo`: last 0.9855 (min 0.8593, max 0.9856)
  - `Validation/MS-SSIM_flair`: last 0.9845 (min 0.8504, max 0.9845)
  - `Validation/MS-SSIM_t1_gd`: last 0.9935 (min 0.8718, max 0.9936)
  - `Validation/MS-SSIM_t1_pre`: last 0.9943 (min 0.8538, max 0.9943)
  - `Validation/PSNR`: last 38.3338 (min 28.5770, max 38.3367)
  - `Validation/PSNR_bravo`: last 35.9452 (min 28.8950, max 35.9491)
  - `Validation/PSNR_flair`: last 36.8375 (min 28.4025, max 36.8430)
  - `Validation/PSNR_t1_gd`: last 40.5753 (min 29.6029, max 40.5820)
  - `Validation/PSNR_t1_pre`: last 39.9778 (min 27.3902, max 39.9875)

**Regional loss (final):**
  - `regional/background_loss`: 0.0046
  - `regional/large`: 0.0352
  - `regional/medium`: 0.0341
  - `regional/small`: 0.0301
  - `regional/tiny`: 0.0257
  - `regional/tumor_bg_ratio`: 6.8229
  - `regional/tumor_loss`: 0.0313
  - `regional_bravo/background_loss`: 0.0057
  - `regional_bravo/large`: 0.0426
  - `regional_bravo/medium`: 0.0420
  - `regional_bravo/small`: 0.0338
  - `regional_bravo/tiny`: 0.0304
  - `regional_bravo/tumor_bg_ratio`: 6.5208
  - `regional_bravo/tumor_loss`: 0.0373
  - `regional_flair/background_loss`: 0.0057
  - `regional_flair/large`: 0.0441
  - `regional_flair/medium`: 0.0421
  - `regional_flair/small`: 0.0389
  - `regional_flair/tiny`: 0.0358
  - `regional_flair/tumor_bg_ratio`: 7.0205
  - `regional_flair/tumor_loss`: 0.0402
  - `regional_t1_gd/background_loss`: 0.0033
  - `regional_t1_gd/large`: 0.0339
  - `regional_t1_gd/medium`: 0.0321
  - `regional_t1_gd/small`: 0.0279
  - `regional_t1_gd/tiny`: 0.0202
  - `regional_t1_gd/tumor_bg_ratio`: 8.7599
  - `regional_t1_gd/tumor_loss`: 0.0286
  - `regional_t1_pre/background_loss`: 0.0036
  - `regional_t1_pre/large`: 0.0202
  - `regional_t1_pre/medium`: 0.0199
  - `regional_t1_pre/small`: 0.0197
  - `regional_t1_pre/tiny`: 0.0166
  - `regional_t1_pre/tumor_bg_ratio`: 5.2454
  - `regional_t1_pre/tumor_loss`: 0.0191

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1.829e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.7468, max 14.0159 @ ep 1
  - `training/grad_norm_max`: last 1.4615, max 47.7535 @ ep 2

---
## compression_2d/seg

*4 runs across 2 experiment families.*

### exp11

**exp11_1 (2D seg VQ-VAE)** — seg-channel compression at 2D.

#### `exp11_1_seg_compress_20260111-145520`
*started 2026-01-11 14:55 • 125 epochs • 4h29m • 114.9 TFLOPs • peak VRAM 27.2 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.9901 (min 0.4900, max 0.9902)

**Regional loss (final):**
  - `regional_seg/dice`: 0.9917
  - `regional_seg/dice_large`: 0.9993
  - `regional_seg/dice_medium`: 0.9986
  - `regional_seg/dice_small`: 0.9981
  - `regional_seg/dice_tiny`: 0.9896
  - `regional_seg/iou`: 0.9863

**Downstream seg metrics:**
  - `regional_seg/dice`: last 0.9917 (max 0.9925 @ ep 115)
  - `regional_seg/dice_large`: last 0.9993 (max 0.9994 @ ep 121)
  - `regional_seg/dice_medium`: last 0.9986 (max 0.9991 @ ep 106)
  - `regional_seg/dice_small`: last 0.9981 (max 0.9985 @ ep 90)
  - `regional_seg/dice_tiny`: last 0.9896 (max 0.9907 @ ep 115)
  - `regional_seg/iou`: last 0.9863 (max 0.9877 @ ep 115)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.4153, max 17.3214 @ ep 0
  - `training/grad_norm_max`: last 0.9926, max 268.6725 @ ep 0

### exp13

**exp13 (2D seg compression ratios)** — f32 / f64 / f128 seg compression
variants, parallel to bravo exp9 sweep.

#### `exp13_2_seg_f64_20260201-024544`
*started 2026-02-01 02:45 • 500 epochs • 17h45m • 461.7 TFLOPs • peak VRAM 29.2 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.9990 (min 0.5003, max 0.9991)

**Regional loss (final):**
  - `regional_seg_seg/dice`: 0.9990
  - `regional_seg_seg/dice_large`: 1.0000
  - `regional_seg_seg/dice_medium`: 0.9999
  - `regional_seg_seg/dice_small`: 0.9998
  - `regional_seg_seg/dice_tiny`: 0.9987
  - `regional_seg_seg/iou`: 0.9982

**Downstream seg metrics:**
  - `regional_seg_seg/dice`: last 0.9990 (max 0.9994 @ ep 412)
  - `regional_seg_seg/dice_large`: last 1.0000 (max 1.0000 @ ep 311)
  - `regional_seg_seg/dice_medium`: last 0.9999 (max 0.9999 @ ep 395)
  - `regional_seg_seg/dice_small`: last 0.9998 (max 0.9999 @ ep 371)
  - `regional_seg_seg/dice_tiny`: last 0.9987 (max 0.9992 @ ep 412)
  - `regional_seg_seg/iou`: last 0.9982 (max 0.9989 @ ep 412)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0917, max 15.6637 @ ep 0
  - `training/grad_norm_max`: last 0.3037, max 157.5781 @ ep 0

#### `exp13_1_seg_f32_20260201-024545`
*started 2026-02-01 02:45 • 500 epochs • 17h46m • 459.4 TFLOPs • peak VRAM 29.2 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.9986 (min 0.4852, max 0.9988)

**Regional loss (final):**
  - `regional_seg_seg/dice`: 0.9991
  - `regional_seg_seg/dice_large`: 1.0000
  - `regional_seg_seg/dice_medium`: 0.9998
  - `regional_seg_seg/dice_small`: 0.9998
  - `regional_seg_seg/dice_tiny`: 0.9989
  - `regional_seg_seg/iou`: 0.9984

**Downstream seg metrics:**
  - `regional_seg_seg/dice`: last 0.9991 (max 0.9993 @ ep 429)
  - `regional_seg_seg/dice_large`: last 1.0000 (max 1.0000 @ ep 310)
  - `regional_seg_seg/dice_medium`: last 0.9998 (max 0.9999 @ ep 284)
  - `regional_seg_seg/dice_small`: last 0.9998 (max 0.9999 @ ep 334)
  - `regional_seg_seg/dice_tiny`: last 0.9989 (max 0.9992 @ ep 429)
  - `regional_seg_seg/iou`: last 0.9984 (max 0.9988 @ ep 429)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.1377, max 23.2674 @ ep 0
  - `training/grad_norm_max`: last 0.3759, max 191.1124 @ ep 0

#### `exp13_3_seg_f128_20260201-025617`
*started 2026-02-01 02:56 • 500 epochs • 17h58m • 470.6 TFLOPs • peak VRAM 29.3 GB*

**Validation quality:**
  - `Validation/IoU`: last 0.9990 (min 0.5868, max 0.9990)

**Regional loss (final):**
  - `regional_seg_seg/dice`: 0.9991
  - `regional_seg_seg/dice_large`: 1.0000
  - `regional_seg_seg/dice_medium`: 0.9998
  - `regional_seg_seg/dice_small`: 0.9998
  - `regional_seg_seg/dice_tiny`: 0.9989
  - `regional_seg_seg/iou`: 0.9985

**Downstream seg metrics:**
  - `regional_seg_seg/dice`: last 0.9991 (max 0.9994 @ ep 469)
  - `regional_seg_seg/dice_large`: last 1.0000 (max 1.0000 @ ep 265)
  - `regional_seg_seg/dice_medium`: last 0.9998 (max 0.9999 @ ep 436)
  - `regional_seg_seg/dice_small`: last 0.9998 (max 0.9999 @ ep 421)
  - `regional_seg_seg/dice_tiny`: last 0.9989 (max 0.9992 @ ep 469)
  - `regional_seg_seg/iou`: last 0.9985 (max 0.9989 @ ep 469)

**LR schedule:**
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_avg`: last 0.0720, max 19.1875 @ ep 0
  - `training/grad_norm_max`: last 0.1809, max 209.5683 @ ep 0

---
## compression_2d/progressive

*2 runs across 1 experiment families.*

### exp1

**exp1 (progressive growing)** — progressive-growing 2D compression with
two restart timestamps. Exploratory, not load-bearing for 3D work.

#### `exp1_20251215-020451`
*started 2025-12-15 02:04 • 100 epochs • 20h59m*

#### `exp1_20251221-234218`
*started 2025-12-21 23:42 • 200 epochs • 24h13m*

---
## compression_2d/bravo

*1 runs across 1 experiment families.*

### exp2

**exp2_256 (2D bravo compression)** — single 2D bravo VAE at 256×256.
Bypassed — for the 2D diffusion pipeline bravo is generated in pixel-space
directly (see diffusion_2d/bravo), not via latent compression.

#### `exp2_256_20251223-233235`
*started 2025-12-23 23:32 • 100 epochs • 2h50m • 77046.3 TFLOPs*

**Loss dynamics:**
  - `Loss/L1_train`: 0.0046 → 0.0046 (min 0.0046 @ ep 0)
  - `Loss/L1_val`: 0.0028 → 0.0031 (min 0.0028 @ ep 0)

**Validation quality:**
  - `Validation/MS-SSIM`: last 0.9960 (min 0.9941, max 0.9965)
  - `Validation/PSNR`: last 39.7431 (min 38.7025, max 40.3501)

**LR schedule:**
  - `LR/Discriminator`: peak 0.0005 @ ep 4, final 1e-06
  - `LR/Generator`: peak 0.0001 @ ep 4, final 1e-06

**Training meta:**
  - `training/grad_norm_d_avg`: last 9.4862, max 11.4302 @ ep 49
  - `training/grad_norm_d_max`: last 23.5241, max 121.0596 @ ep 1
  - `training/grad_norm_g_avg`: last 0.0866, max 0.3372 @ ep 31
  - `training/grad_norm_g_max`: last 0.4746, max 1.7446 @ ep 73

---
## diffusion_3d/old-discard

*50 runs across 10 deprecated experiment families. These are early or abandoned 3D-diffusion runs superseded by current canonical runs in other categories. Listed in bulk with one-line descriptions; no per-run metric breakdown.*

### exp1 (7 runs — deprecated)

early pixel-bravo 256×160 runs and CFG-debugging runs; superseded by the current `exp1` family in diffusion_3d/bravo. Includes chained-generation experiments (`exp1_chained`, `exp1_chained_chain`).

<details><summary>Run list</summary>

- `exp1_1_pixel_bravo_rflow_256x160_20260119-212710` — 20260119-212710
- `exp1_1_pixel_bravo_rflow_256x160_20260126-183654` — 20260126-183654
- `exp1_pixel_bravo_rflow_128x160_20260126-184743` — 20260126-184743
- `exp1_debugging_cfg_rflow_128x160_20260128-025709` — 20260128-025709
- `exp1_1_pixel_bravo_rflow_256x160_20260129-233435` — 20260129-233435
- `exp1_chained_chain_20260213-015507` — 20260213-015507
- `exp1_chained_20260213-031857` — 20260213-031857

</details>

### exp4 (2 runs — deprecated)

early SDA (shifted data augmentation) 128×160 variants; superseded by the current `exp4` in diffusion_3d/bravo (now also small) and by the SDA results that led to ScoreAug being preferred.

<details><summary>Run list</summary>

- `exp4_pixel_bravo_sda_rflow_128x160_20260122-130906` — 20260122-130906
- `exp4_1_pixel_bravo_sda_rflow_128x160_20260127-043308` — 20260127-043308

</details>

### exp5 (3 runs — deprecated)

early ScoreAug 128×160 variants including compose mode — superseded by `exp23` in diffusion_3d/bravo.

<details><summary>Run list</summary>

- `exp5_1_pixel_bravo_scoreaug_rflow_128x160_20260123-024740` — 20260123-024740
- `exp5_2_pixel_bravo_scoreaug_compose_rflow_128x160_20260123-024808` — 20260123-024808
- `exp5_3_pixel_bravo_scoreaug_rflow_128x160_20260126-214534` — 20260126-214534

</details>

### exp6 (1 runs — deprecated)

ControlNet stage-1 at 128×160 — superseded by `exp6a/b/b_1` in diffusion_3d/bravo at 256×160.

<details><summary>Run list</summary>

- `exp6a_pixel_bravo_controlnet_stage1_rflow_128x160_20260126-181256` — 20260126-181256

</details>

### exp7 (8 runs — deprecated)

SiT size sweep at 128 and 256 (patch 8 / 16) — early large-model tests, superseded by the current exp7 SiT runs in diffusion_3d/bravo.

<details><summary>Run list</summary>

- `exp7_1_sit_s_256_patch16_rflow_256x160_20260127-051957` — 20260127-051957
- `exp7_sit_s_128_patch8_rflow_128x160_20260127-051957` — 20260127-051957
- `exp7_1_sit_b_256_patch16_rflow_256x160_20260127-133520` — 20260127-133520
- `exp7_1_sit_l_256_patch16_rflow_256x160_20260127-182806` — 20260127-182806
- `exp7_sit_l_128_patch8_rflow_128x160_20260127-194155` — 20260127-194155
- `exp7_sit_b_128_patch8_rflow_128x160_20260127-194653` — 20260127-194653
- `exp7_1_sit_xl_256_patch16_rflow_256x160_20260127-212438` — 20260127-212438
- `exp7_sit_xl_128_patch8_rflow_128x160_20260128-000337` — 20260128-000337

</details>

### exp9 (8 runs — deprecated)

LDM 4x / 8x bravo at 256×160 — first-wave latent-diffusion runs; superseded by `exp9_1` in diffusion_3d/bravo_latent.

<details><summary>Run list</summary>

- `exp9_ldm_4x_bravo_rflow_256x160_20260201-155324` — 20260201-155324
- `exp9_ldm_8x_bravo_rflow_256x160_20260201-155324` — 20260201-155324
- `exp9_0_ldm_8x_bravo_small_rflow_256x160_20260201-193834` — 20260201-193834
- `exp9_1_ldm_4x_bravo_rflow_256x160_20260211-124914` — 20260211-124914
- `exp9_ldm_8x_bravo_rflow_256x160_20260213-025408` — 20260213-025408
- `exp9_ldm_4x_bravo_rflow_256x160_20260213-025522` — 20260213-025522
- `exp9_0_ldm_8x_bravo_small_20260221-215044` — 20260221-215044
- `exp9_1_ldm_4x_bravo_20260221-224417` — 20260221-224417

</details>

### exp10 (3 runs — deprecated)

SiT-B on DC-AE latents (8x / 4x / 2x) — DC-AE branch abandoned per CLAUDE.md. Two runs produced no TB events (extraction-empty).

<details><summary>Run list</summary>

- `exp10_1_sit_b_p1_dcae_8x8x32_rflow_256x160_20260202-120117` — 20260202-120117
- `exp10_2_sit_b_p1_dcae_4x4x128_rflow_256x160_20260213-024933` — 20260213-024933 (empty)
- `exp10_3_sit_b_p1_dcae_2x2x512_rflow_256x160_20260213-025208` — 20260213-025208 (empty)

</details>

### exp11 (3 runs — deprecated)

S2D bravo at 128 / 256 — stopped in favor of pixel-space baseline.

<details><summary>Run list</summary>

- `exp11_s2d_bravo_rflow_128x160_20260210-001225` — 20260210-001225
- `exp11_1_s2d_bravo_rflow_256x160_20260210-001236` — 20260210-001236
- `exp11_2_20260214-203209` — 20260214-203209

</details>

### exp12 (11 runs — deprecated)

wavelet bravo at 128 / 256 — superseded by `exp26_1 WDM` in diffusion_3d/bravo.

<details><summary>Run list</summary>

- `exp12_1_wavelet_bravo_rflow_256x160_20260210-001255` — 20260210-001255
- `exp12_wavelet_bravo_rflow_128x160_20260210-001311` — 20260210-001311
- `exp12_1_wavelet_bravo_rflow_256x160_20260213-033451` — 20260213-033451
- `exp12_2_20260214-163133` — 20260214-163133
- `exp12_4_20260215-175624` — 20260215-175624
- `exp12_3_20260215-175625` — 20260215-175625
- `exp12_5_20260216-022108` — 20260216-022108
- `exp12_6_20260216-161013` — 20260216-161013
- `exp12b_2_20260221-215953` — 20260221-215953
- `exp12b_3_20260221-224417` — 20260221-224417
- `exp12_2_20260221-225423` — 20260221-225423

</details>

### exp13 (4 runs — deprecated)

DiT on latents (4x / 8x) — superseded by `exp13_1` in diffusion_3d/bravo_latent.

<details><summary>Run list</summary>

- `exp13_dit_8x_bravo_20260215-024826` — 20260215-024826
- `exp13_dit_4x_bravo_20260215-183526` — 20260215-183526
- `exp13_dit_4x_bravo_20260221-234320` — 20260221-234320
- `exp13_dit_8x_bravo_20260221-234320` — 20260221-234320

</details>

---

## diffrs/tensorboard

*0 runs — the `runs_tb/diffrs/tensorboard/` directory is empty.* DiffRS
post-hoc rejection-sampling experiments were not logged to a separate TB
root; any related metrics live within the base diffusion run's
`Generation/*` tags when DiffRS is evaluated.

---

## Global summary

### Runs by scale

- **Extracted:** 391 runs, 29,253 scalar series, 13 MB JSON.
- **Empty runs (3):** `diffusion_2d/bravo/exp16_bs32_rflow_128_20260111-205709`
  (batch-32 2D run — likely OOM before any event was written),
  `diffusion_3d/old-discard/exp10_2_sit_b_p1_dcae_4x4x128_rflow_256x160_20260213-024933`
  and `diffusion_3d/old-discard/exp10_3_sit_b_p1_dcae_2x2x512_rflow_256x160_20260213-025208`
  (DC-AE SiT variants that never produced TB events).
- **Runs with NaN scalars (3):** `diffusion_3d/old-discard/exp1_debugging_cfg_rflow_128x160_20260128-025709`
  and two early `diffusion_3d/seg/exp2_pixel_seg_sizebin_rflow_128x160` runs
  — all NaN in `Generation/CMMD_*` tags (CMMD feature extractor failed on
  very-early checkpoints; later checkpoints finite).

### Load-bearing recipes (as of April 2026)

| Use case | Best run | Evidence |
|---|---|---|
| Pixel-space 3D bravo (best anatomy) | `exp1_1_1000_pixel_bravo_20260402-121556` | 100% PCA pass, post-hoc FID 19.12 @ Euler-27 (memory/finding_exp1_1_1000_best.md) |
| Pixel-space 3D bravo (best in-training FID) | `exp23_1_pixel_bravo_scoreaug_safe_20260402-122902` / `exp23_pixel_bravo_scoreaug_256_20260309-220559` | FID 20.38 post-hoc; but ~20% broken brain shapes, not recommended for downstream |
| Latent 3D bravo (most efficient) | `exp22_2_ldm_4x_dit_l` | FID 47.41 in 3.53 h training |
| 3D VAE (latent ckpt source) | `compression_3d/multi_modality/exp14_*` | per-exp-14 KL/adv variants |
| 3D seg generation (canonical) | `exp14_1_pixel_seg` | unconditional seg used by the 2-stage pipeline (size-bin validation must be OFF for this model) |
| Mean-blur fine-tune (mid/high-t HF recovery) | `exp37_3_pixel_bravo_lpips_ffl_short` | +27% vs baseline at high-freq band per memory/project_phase1_spectrum_finding.md |
| Downstream (nnU-Net real-only baseline) | `exp3_baseline` (nnU-Net) | 5-fold CV, canonical real-only floor |

### Regenerating this document

1. Re-run the extractor if `runs_tb/` has new runs:
   ```
   python scripts/extract_tb.py --out runs_tb_extracted.json
   ```
2. Re-emit any category section:
   ```
   python /tmp/emit_category.py diffusion_3d bravo > /tmp/new.md
   ```
3. Replace the corresponding `## <category>/<mode>` section in
   `EXPERIMENT_SUMMARIES.md`.

All family intros and memory cross-refs live in `/tmp/emit_category.py`'s
`FAMILY_INTRO` dict; extend there when adding new experiments. The old-discard
descriptions live in `/tmp/emit_old_discard.py`.

### Caveats

- Per-epoch scalars use 0-indexed steps; the "epochs reached" reported is
  `last_step + 1`. A run that stopped at step 499 is reported as 500 epochs.
- Wall-time is per-tag span (wall_time of last event − wall_time of first
  event). A run may have multiple events files from resumes; the wall span
  covers the union but not necessarily the cumulative compute time.
- `Generation/KID_*` at in-training resolution uses only 1 volume and 10
  Euler steps, so values are noisy — use `Generation/extended_*` (4 vols,
  25 steps, every-50-epochs) for ranking.
- Post-hoc FID is not in TB events; best post-hoc numbers come from the
  user's eval scripts and are cited from `memory/project_3d_experiment_findings.md`.
- The extractor caps per-scalar point count at 20,000 — for very long
  runs with per-step tags this truncates; min/max/last are still correct.
