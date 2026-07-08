# Proven Techniques for 3D Brain MRI Generation

Last updated: May 5, 2026.

Techniques that have shown measurable improvement over baseline (exp1/exp1_1: 270M UNet, RFlow, pixel-space bravo, no tricks).

---

## Training Improvements

| Technique | Experiment | Evidence | Config |
|-----------|-----------|----------|--------|
| **Adjusted offset noise** | exp1l / exp1l_1 | Improved generation metrics over baseline | `training.offset_noise.enabled=true training.offset_noise.strength=0.1 training.offset_noise.adjusted=true` |
| **Post-hoc EMA** (Karras EDM2) | exp1o_1 | Improved generation metrics; allows post-training sweep of optimal EMA decay | `training.use_ema=true training.ema.mode=post_hoc 'training.ema.sigma_rels=[0.05,0.28]'` |
| **Uniform timestep sampling** | exp1p_1 | On par or better than logit-normal; simpler | `strategy.sample_method=uniform` |
| **Offset noise** (standard) | exp1k | Positive — fixes inter-volume intensity variance collapse | `training.offset_noise.enabled=true training.offset_noise.strength=0.1` |
| **ScoreAug** | exp23 / exp5_1 | Only technique that scales to 1000+ epochs in **pixel space**. Post-hoc FID 20.38 (27 steps), RadImageNet FID 0.659 (48 steps). **Hurts latent space** — see below. Note: omega encoding had a paper-conformance bug fixed in April 2026 — see `@docs/scoreaug_omega.md` and the exp23 translation-leak diagnosis | `training.score_aug.enabled=true` |
| **LPIPS at low t** (`perceptual_max_timestep=250`) | exp32_2 family | Reduces mean-blur in pixel-space outputs by adding LPIPS only for t < 0.25. Effective on top of exp1_1 (improves vessel-scale energy at low t). Currently being applied to the 17M tiny UNet (exp20_6 → exp32_2_1000_exp20_6) since exp20_6 was vindicated by Tier 1 | `training.perceptual_weight=0.1 training.perceptual_max_timestep=250` |
| **LPIPS at high t** (`perceptual_t_schedule=[t_on,t_full,t_off]`) | exp37 family | Targets HF deficit at the noisier end of the trajectory. Opposite schedule of exp32_2 — used to test whether LPIPS guidance at high t recovers vessel-scale energy that mean-collapse pruned out | `training.perceptual_t_schedule=[0.05,0.20,0.70] training.perceptual_weight=0.4` |
| **Weight decay** (0.05) | exp1s (128x128) | Good 128x128 in-training FID (43.45). Not tested at 256x256. KID trajectory flat | `training.optimizer.weight_decay=0.05` |

## Multi-Modality

| Technique | Experiment | Evidence | Config |
|-----------|-----------|----------|--------|
| **Dual mode** (T1pre+T1gd) | exp1v2 / exp1v2_1 | FID 32.80 (independent norm) / 24.30 (joint norm) at 128x128. Joint normalization preserves cross-modality intensity relationships | `mode=dual` / `mode=dual mode.joint_normalization=true` |

## Conditioning

| Technique | Experiment | Evidence | Config |
|-----------|-----------|----------|--------|
| **ControlNet** (two-stage) | exp6a/exp6b | Early promising results at 128x128x160; seg conditioning via zero-conv residuals without modifying UNet | `controlnet.enabled=true controlnet.freeze_unet=true pretrained_checkpoint=<stage1_ckpt>` |

## Inference Improvements

| Technique | Evidence | Config |
|-----------|----------|--------|
| **Timestep shift** | Improved generation quality at inference; ratio 2.0 optimal (FID 49.25 vs 50.18@1.5, 54.78@1.0) | `shift_ratio=2.0` (generate.py); also `shift_ratio_seg` / `shift_ratio_bravo` for per-model overrides |
| **Euler 27-32 steps** | Optimal for ImageNet FID (exp1_1: 19.12@27, v2: 20.84@32, exp23: 20.38@27). RadImageNet optimal at ~48-79 steps. Higher-order solvers are worse | `num_steps=27` (generate.py); also `num_steps_seg` / `num_steps_bravo` |
| **Per-experiment optima search** (Tier 1) | Default step count is wrong for many models. Mamba L (exp34_1_1000) optimal at 14 steps (FID 35.77, ~half the steps of UNet); 17M tiny UNet (exp20_6) optimal at 76 steps (FID 30.79, beats the larger 67M UNets in the same exp20 family — exp20_4=43.54, exp20_5=37.12, exp20_7=39.12 — though still well above exp1_1 270M @ FID 19.12). TB extended-eval misclassified both as mediocre. | `find_optimal_steps --metric fid,fid_radimagenet,pca` |
| **HandoffWrapper** (two-stage low-t/high-t) | Inference-time composition for low-t fine-tunes (exp48 family). High-t base handles t > handoff_t, fine-tune handles t ≤ handoff_t. Enables specialized fine-tunes without sacrificing high-t performance | `generate.py`: Hydra keys `image_model_high_t=<base> image_model=<ft> handoff_t=0.25`. `find_optimal_steps.py`: CLI `--high-t-checkpoint <base> --low-t-checkpoint <ft> --handoff-t 0.25` |

## Regularization (Completed)

| Technique | Experiment | Result | Config |
|-----------|-----------|--------|--------|
| **Weight decay** (0.05) | exp1s (128x128) | **Positive at 128x128** — FID 43.45, KID 0.032. Not tested at 256x256. Flat KID trajectory | `training.optimizer.weight_decay=0.05` |
| **Attention dropout** (0.1) | exp1r (128x128) | **Mixed** — good latest FID (49.32) but terrible best-ckpt FID (82.21). Train/inference dropout gap | `model.dropout_cattn=0.1` |

## Not Helpful / Inconclusive

| Technique | Experiment | Finding |
|-----------|-----------|---------|
| **Standard EMA** | exp8 | Best 128x128 val loss but not positive on generation metrics |
| **[-1,1] rescaling** | exp1b / exp1b_1 | MSE 3x higher (expected), many gradient spikes; TBD if generation quality differs |
| **Higher-order ODE solvers** | eval_ode_solvers | All worse than Euler 25 — RK4, dopri5, heun2 all degrade quality |
| **More Euler steps (ImageNet FID)** | eval_ode_solvers | ImageNet FID degrades past 27 steps. RadImageNet FID improves up to ~48 steps |
| **FreeU** | exp1_1 post-hoc | Marginal: best config (b=1.0, s=0.9) FID=19.87 vs baseline 19.12 |
| **CFG-Zero*** | exp1n post-hoc | Best scale=1.0 (FID 44.79). All scales >1.0 much worse (155+). Not helpful |
| **Stacking techniques** | exp24 (1000ep) | Combined ScoreAug+AdjOffset+PHEMA+UniformT (FID 62.87) ≈ ScoreAug alone (FID 62.57). Added 742 gradient spikes vs 139 |
| **DiffRS** | Part 8 eval | Negative — discriminator-based rejection sampling did not improve quality |
| **Restart Sampling** | Part 8 eval | No gain over Euler 25 |
| **Gaussian normalization** | Part 14 eval | Catastrophic — breaks generation |
| **ScoreAug in latent space** | exp27/28/28_2 | Hurts LDM — DiT-L FID 57→47 (worse), MAISI UNet FID 80→51 (worse). Augmentations too destructive for compact latent representations |
| **Diffusion Mixup in latent** | exp28_1 | Negative — FID 98.56 vs exp28's 79.91. Cross-batch interpolation with batch_size=1 in latent space is not useful |
| **ScoreAug v2 (stronger)** | exp28_2 | No improvement over v1 — FID 80.17 vs 79.91. Stronger augmentation doesn't help in latent space |
| **WDM beyond 500 epochs** | exp26_1 (1000ep) vs exp19_2 (500ep, same WDM 270M architecture extended) | WDM overfits — FID 77.28 at 1000ep (exp26_1) vs 67.32 at 500ep (exp19_2). KID was already flat at 500ep |
| **WDM + ScoreAug** | exp26 | Gen metrics unavailable (crash), training MSE ~0.35 suggests partial collapse |
| **Triple mode** | exp1v3/1v3_1 | Much harder than dual — FID 65-67 vs 25-33. Joint norm doesn't help triple (unlike dual) |

