# Diffusion Training Tricks — Complete Inventory

Comprehensive catalog of every diffusion training trick, technique, and method.
Covers what's implemented in this codebase and what exists in the literature.

**Project context**: 3D brain MRI (BrainMetShare-3), 105 training volumes, RFlow strategy,
primarily UNet architecture, single A100/H100 GPU, pixel-space best so far.

---

## IMPLEMENTED (67 entries, 66 functional + 1 dead config)

### Loss Functions & Weighting

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 1 | MSE Loss | Default L2 loss between prediction and target | All diffusion experiments (default loss) | DDPM (Ho et al., 2020) |
| 2 | Pseudo-Huber Loss | `sqrt(||error||^2 + c^2) - c` — robust to outlier velocity predictions, `c = 0.00054 * sqrt(d)` | RFlow velocity training. 3D: exp1g/exp1g_1 | Lee et al., NeurIPS 2024 |
| 3 | LPIPS-Huber Loss | `(1-t)*Huber + LPIPS(x0, x0_hat)` — Huber fades at high noise, LPIPS provides perceptual signal | RFlow bravo mode (LPIPS needs perceptual content). 3D: exp1h/exp1h_1 | Lee et al., NeurIPS 2024 |
| 4 | Perceptual Loss (RadImageNet ResNet50) | Feature-space loss via MONAI PerceptualLoss, optional additive term | Compression trainers (VAE/VQ-VAE/DC-AE), bravo diffusion with perceptual_weight>0 | Johnson et al., 2016; RadImageNet |
| 5 | Min-SNR-gamma (DDPM) | Reweights per-sample loss by `min(SNR, gamma)/SNR` — prevents high-noise timesteps from dominating | DDPM strategy only | Hang et al., ICCV 2023 |
| 6 | RFlow Min-SNR-gamma | Same idea using RFlow's SNR formula `((1-t)/t)^2` | RFlow strategy. 3D: exp1e/exp1e_1 | Strategy config `snr_gamma` |
| 7 | Region-Weighted Loss | Per-pixel loss weighting by tumor size (RANO-BM thresholds) | Seg-conditioned modes, clinical evaluation. 2D: exp26_1 | Custom (clinical relevance) |
| 8 | FP32 Loss Computation | Cast to float32 before loss to prevent BF16 underflow | All experiments (always active, bugfix) | Internal fix |

### Noise & Timestep Manipulation

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 9 | Logit-Normal Timestep Sampling | Biases sampling toward mid-range (harder) timesteps. **Default for all RFlow experiments** (`strategy.sample_method=logit-normal`) | RFlow strategy (all experiments use this by default). 3D: exp1p_1 tests uniform as ablation | SD3 (Esser et al., 2024) |
| 10 | Timestep Transform | Resolution-based timestep transformation | RFlow strategy (configurable via `use_timestep_transform`) | OpenSora |
| 11 | Curriculum Timestep Scheduling | Progressively shifts from easy (low noise) to hard (high noise) over warmup | All strategies. 2D: exp21_1 | Standard curriculum |
| 12 | Timestep Jitter | Adds small Gaussian noise to sampled timesteps | All strategies. 2D: exp22_1 | Custom |
| 13 | Offset Noise (standard) | Adds spatially-constant noise `noise + strength * randn(B,C,1,...,1)` | Pixel-space bravo — fixes brightness variance collapse. 3D: exp1k/exp1k_1 | CrossLabs blog, 2023 |
| 14 | Adjusted Offset Noise | Generation also starts from `N(strength*xi, I)` to match training distribution | Pixel-space bravo — fixes train/generation distribution mismatch. 3D: exp1l/exp1l_1 | Kutsuna, 2024 |
| 15 | Noise Augmentation | Perturbs noise vector, renormalized to maintain variance | All strategies. 2D: exp23_1 | Custom |
| 16 | Continuous Timesteps | Float timesteps in [0, T] — RFlow default | RFlow strategy (all RFlow experiments) | Liu, 2022 |
| 17 | Discrete Timesteps | Integer timesteps in [0, 999] — DDPM standard | DDPM strategy, WDM experiments | Ho, 2020 |

### Data Augmentation for Diffusion

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 18 | ScoreAug | D4 symmetries on noisy data with omega conditioning | All modes. 2D: exp9_x series. 3D: exp5_1/exp5_2/exp5_3, exp1i/exp1i_1 | ScoreAug, 2025 |
| 19 | ScoreAug v2 (Structured) | Non-destructive (stackable) + destructive (pick one) split | Same as ScoreAug. 2D: exp9_7 | Extended ScoreAug |
| 20 | ScoreAug Fixed Patterns | 16 deterministic masks (checkerboard, grid, coarse, patch dropout) | Same as ScoreAug. Adds spatial dropout diversity | Custom |
| 21 | ~~ScoreAug Brightness~~ | **Dead config**: `brightness` key exists in ScoreAugConfig and was set in exp9_2, but no transform code applies it — ScoreAugTransform only supports rotation/flip/translation/cutout/patterns. No-op. | N/A (never functional) | — |
| 22 | ScoreAug Mode Intensity Scaling | Mode-specific scale factor, forces model to use conditioning | Multi-modality mode only (2D). 2D: exp10_x series | Custom (2D only) |
| 23 | SDA (Shifted Data Augmentation) | Transforms clean data with shifted timesteps to prevent leakage | All modes. 3D: exp4. 2D: exp18_1 | IEEE Access 2025 |
| 24 | Batch Augmentations | Mixup, CutMix, Mosaic, Copy-Paste (compression training) | Compression trainers only (VAE/VQ-VAE/DC-AE) | Various |
| 25 | Data Rescaling [-1, 1] | Rescales [0,1] to [-1,1] for pixel-space diffusion | Pixel-space experiments. 3D: exp1b | Standard |
| 26 | Conditioning Dropout (CFG training) | Randomly drops conditioning with prob p to enable CFG at inference | Seg-conditioned modes (default 0.15, exp2c/2d/2e use 0.25), ControlNet (default 0.15). Bravo/bravo_seg_cond have it disabled (0.0) by default | Ho & Salimans, 2022 |

### Optimizer & Regularization

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 27 | AdamW | Adam with decoupled weight decay | All experiments (default optimizer) | Loshchilov & Hutter, 2019 |
| 28 | Gradient Clipping | Max gradient norm clipping (default 1.0) | All experiments. Critical for LDM stability (exp9_1 used 0.5) | Standard |
| 29 | Gradient Accumulation | Accumulate over N micro-batches for larger effective batch | Large models / small batch. 3D: exp1j/exp1j_1 | Standard |
| 30 | SAM / ASAM | Seeks flat minima via 2x forward-backward | Generalization on small datasets. 2D: exp11_1 (SAM), exp11_2 (ASAM) | Foret 2020, Kwon 2021 |
| 31 | Gradient Noise Injection | Decaying Gaussian noise on gradients: `sigma / (1+step)^decay` | Implicit regularization. 2D: exp20_1 | Neelakantan et al., 2015 |
| 32 | Gradient Spike Detection | Skips optimizer step on anomalous gradient spikes | All experiments (always active). Critical for LDM — exp9 had 26K gradient spikes | Custom |
| 33 | Feature Perturbation | Gaussian noise on intermediate features (continuous dropout) | Regularization. 2D: exp24_1 | Custom |
| 34 | Self-Conditioning via Consistency | Two-pass: second pass consistency loss | All modes. 2D: exp25_1 | Custom |

### LR Scheduling

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 35 | Linear Warmup | Ramps LR from ~0 to target over warmup epochs | All experiments (default 5-10 epochs warmup) | Standard |
| 36 | Cosine Annealing | Cosine decay from peak LR to eta_min | Default LR schedule for all experiments | Loshchilov & Hutter, 2017 |
| 37 | Constant LR | Warmup then constant | Alternative to cosine. 2D: exp19_1 | Standard |
| 38 | ReduceLROnPlateau | Reduces LR when val loss plateaus | Alternative schedule. 2D: exp27 | Standard |

### Architecture-Level

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 39 | EMA | Slowly-updated weight copy for generation | All architectures. 3D: exp8 (best 128x128 result). 2D: exp13_1/exp13_2 | Polyak averaging |
| 67 | Post-hoc EMA Reconstruction | Store periodic snapshots during training, reconstruct arbitrary EMA profiles post-hoc | 3D: exp1o/exp1o_1 (sigma_rels=[0.05, 0.28]). Promising — allows finding optimal EMA decay without retraining | Karras EDM2, 2024 |
| 40 | EDM Preconditioning | Skip-scaling: `v = c_skip*x_t + c_out*F(c_in*input, t)` | RFlow strategy (sigma_data config). 3D: exp1f/exp1f_1 | Karras EDM, NeurIPS 2022 |
| 41 | ControlNet (Stage 1 + 2) | Freeze UNet, train control branch | Conditional generation (seg→bravo). 3D: exp6a (stage 1), exp6b (stage 2) | Zhang et al., 2023 |
| 42 | Mode Embedding | Conditions on which modality is being generated | Multi-modality mode only. 2D: exp10_x series | Custom |
| 43 | Size Bin Embedding (FiLM) | Conditions seg generation on tumor size histograms | Seg-conditioned modes. 3D: exp2/exp2_1/exp2b/exp2b_1/exp2c | Custom |
| 44 | Auxiliary Bin Prediction Loss | Extra head predicts size bins from bottleneck | Seg-conditioned modes. 3D: exp2d/exp2d_1/exp2e | Custom |
| 45 | Drop Path / Stochastic Depth | Randomly drops transformer blocks | DiT/SiT/HDiT only. 2D: exp17_1/exp17_2/exp17_3. 3D: exp7 series | Huang et al., 2016 |
| 46 | QK-Norm | Normalizes Q/K vectors in attention to prevent entropy collapse | DiT/HDiT/UViT (default=true). Prevents BF16 attention divergence | Dehghani 2023; SD3/FLUX |
| 47 | torch.compile + Compiled Fused Forward | JIT compilation for ~10% speedup | All architectures. Most 3D experiments use `use_compile=true` | PyTorch 2.0 |
| 48 | Mixed Precision (BF16/FP32) | Autocast BF16 forward, FP32 master weights | All experiments (always active) | Standard |
| 49 | Pure BF16 Weights | Store model in BF16 (saves 50% memory, VAE only) | Compression trainers only. 2D: exp5 | Custom |
| 50 | Gradient Checkpointing | Recompute activations during backward (~20% slower) | Large models / high-res. LDM UNets, ControlNet, 256x256 experiments | Standard |
| 51 | DDP | Multi-GPU synchronized gradients | Multi-GPU runs (infrastructure, rarely used on IDUN) | PyTorch DDP |

### Diffusion Spaces

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 52 | Latent Diffusion (LDM) | Train in compressed VAE/VQ-VAE/DC-AE latent space | LDM experiments. 3D: exp9/exp9_1 (VQ-VAE 4x/8x), exp10 (DC-AE), exp13 (DiT LDM), exp21/exp22 | Rombach et al., LDM (CVPR 2022) |
| 53 | Wavelet-Space Diffusion (WDM) | Diffuse in DWT wavelet coefficient space — separate subbands | Wavelet experiments. 3D: exp12_2/12_3/12_4 (WDM model), exp19_x series (all collapsed) | Friedrich et al., WDM (MICCAI 2024) |
| 54 | S2D (Stride-to-Depth) Space | Reshape spatial dims into channels via strided decomposition | S2D experiments. 3D: exp11/exp11_1 (both collapsed to mean prediction) | Custom |
| 55 | DC-AE 1.5 Augmented Diffusion | Channel masking: randomly mask subset of latent channels | DC-AE latent diffusion. 2D: exp9_1/9_2/9_3 (structured latent) | DC-AE 1.5 paper |
| 56 | Slicewise Encoding (3D) | Encode 3D volumes slice-by-slice through 2D VAE | 3D LDM with 2D compression models (VQ-VAE, DC-AE) | Custom |

### Generation / Sampling

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 57 | Classifier-Free Guidance (CFG) | `pred = uncond + scale*(cond - uncond)` | Bravo mode (seg mask guidance), bravo_seg_cond (latent seg guidance) | Ho & Salimans, 2022 |
| 58 | Dynamic CFG | Linear interpolation of CFG scale over denoising steps | Same as CFG — inference-time variant | Custom |
| 59 | DiffRS (Rejection Sampling) | Post-hoc discriminator rejects bad intermediates | Post-hoc quality filtering. Evaluated but failed — discriminator memorized 105 volumes | DiffRS paper |
| 60 | Restart Sampling | Forward noise + backward ODE to contract discretization errors | Inference-time. Evaluated but no improvement over Euler/25 | NeurIPS 2023 |
| 61 | Multiple ODE Solvers | Euler, midpoint, heun, RK4, dopri5 via torchdiffeq | Inference-time solver selection. Euler/25 optimal; all higher-order solvers worse | Various |
| 65 | Time-Shifted Sampler | Adjusts step schedule based on resolution — shifts more steps to mid-range timesteps | Inference-only, drop-in replacement for step schedule. Shift ~1.5 shows improvement. Needs further investigation | ICLR 2024 |
| 66 | CFG-Zero* | Zero-init first K unconditional steps in CFG — designed for flow-matching velocity prediction | Seg-conditioned inference (exp2c/2d/2e). `cfg_zero_init_steps=1` default | Fan et al., 2025 |

### Prediction Targets

| # | Trick | What It Does | Relevant To | Reference |
|---|-------|-------------|-------------|-----------|
| 62 | Epsilon Prediction | Model predicts noise | DDPM experiments. 2D: exp1_2, exp8_10/8_11 | DDPM (Ho 2020) |
| 63 | Velocity Prediction | Model predicts `v = x_0 - noise` (RFlow standard) | All RFlow experiments (primary strategy) | Liu, 2022 |
| 64 | x0 (Sample) Prediction | Model directly predicts clean image | WDM experiments (DDPM x0). 3D: exp12_4/12_5/12_6, exp19_0/19_2/19_3/19_5 | Various |

---

## NOT IMPLEMENTED — Ranked by FID Impact (Codebase-Informed)

Sorted by expected generation quality (FID) improvement potential, then training efficiency.
Informed by **89 2D experiments** and **~25 3D experiments** in this codebase.

Constraints: 105 training volumes, RFlow, primarily UNet 270M, 3D brain MRI, single GPU, Euler/25 optimal.

**Key evidence from 2D experiments (baseline FID=36.32):**
Every training trick tested in 2D made FID **worse** or unchanged:
timestep jitter +0.84, noise augmentation +0.35, self-conditioning +1.10,
feature perturbation +1.33, regional weighting +0.13, SDA +4.05.
Only architecture change (SiT: 33.43) and resolution (256x256) improved FID.
3D: Min-SNR didn't help (exp1e), EDM preconditioning didn't help (exp1f).

### HIGH — Novel Axis, Strong Potential

| # | Trick | What It Does | FID Rationale | Effort | Reference |
|---|-------|-------------|---------------|--------|-----------|
| 110 | **FreeU (Skip Connection Reweighting)** | Inference-only: amplifies UNet backbone features (factor b>1) and attenuates high-frequency skip connections (spectral filter factor s<1). Two scalars per decoder level | **Zero training cost**. Purely inference-time quality improvement. Orthogonal to time-shifted sampler (#65). For 3D: apply 3D FFT instead of 2D for spectral filtering | Trivial (~10 lines in sampling, no retraining) | Si et al., CVPR 2024. [arXiv:2309.11497](https://arxiv.org/abs/2309.11497) |
| 111 | **UNet Dropout** | Standard dropout (0.1–0.2) on intermediate UNet residual blocks | With 270M params / 105 samples = 2.5M params per sample — extreme capacity/data ratio. Dropout directly addresses this. EDM2 recommends dropout for overfitting models. **Caveat**: `model.drop_rate` config only wired to DiT/HDiT/UViT, NOT UNet. MONAI's `DiffusionModelUNet` ResBlocks have no dropout — only `dropout_cattn` for attention layers. Needs monkey-patching or subclassing to insert `nn.Dropout3d` into ResBlocks | Medium (modify MONAI ResBlocks or post-init hook) | EDM2 (Karras, CVPR 2024) |
| 112 | **2D-to-3D Weight Inflation** | Pretrain 2D RFlow on ~16K slices extracted from 105 volumes, then inflate 2D conv weights → 3D (repeat/average along depth axis) and fine-tune in 3D. Both 2D and 3D training pipelines already exist | Reported 14.7 FID improvement in medical 3D generation. Addresses fundamental data scarcity by leveraging 150x more 2D training data from the same volumes. "Average" inflation preserves activation magnitudes best | Medium (inflation function ~100 lines + 2D training run) | Liu et al., 2022. [arXiv:2208.03934](https://arxiv.org/abs/2208.03934) |
| 113 | **Public Brain MRI Pretraining** | Pretrain single-channel 3D RFlow on public datasets (IXI ~600, OASIS ~400, HCP ~1000+ volumes), then adapt input channels and fine-tune on 105 multi-channel volumes | Goes from 105 → 2000+ volumes for learning brain anatomy. Same architecture, just different data phases. Channel adaptation: duplicate/zero-init input conv for 2-channel bravo input | Medium (data preprocessing + pretraining + channel adaptation) | Standard transfer learning |

### MEDIUM — Worth Trying (Low Risk or Untested Axis)

| # | Trick | What It Does | FID Rationale | Effort | Reference |
|---|-------|-------------|---------------|--------|-----------|
| 68 | **Progressive Resolution Training** | Pre-train at 128x128, fine-tune at 256x256 | No FID improvement expected (same final quality). **Efficiency only**: 128x128 trains at 70s/epoch vs 224s/epoch — could halve total wall time for 256x256 experiments. UNet conv weights are resolution-independent. Most valuable if running many 256x256 experiments | Medium | SD3 |
| 114 | **TADA (Timestep-Aware Data Augmentation)** | Modulates augmentation strength based on timestep — stronger at mid-range (no distribution shift), weaker at extremes. Refines ScoreAug (#18) which currently applies uniform augmentation | ScoreAug helped 3D (exp23) — TADA makes it more principled. Distribution shifts from augmentation only affect specific timestep intervals, not all. Low code change to existing ScoreAug pipeline | Low (modify ScoreAug to scale by timestep) | Gong et al., NeurIPS 2023. [OpenReview:U6Mb3CRuj8](https://openreview.net/forum?id=U6Mb3CRuj8) |
| 115 | **Patch Diffusion with Coordinate Conditioning** | Train on random 3D patches (64³–96³) with 3 concatenated coordinate channels (normalized x,y,z position). Generate full volumes at inference | Multiplies effective dataset by orders of magnitude — each volume = hundreds of unique patches. Also reduces VRAM → larger batch sizes. PatchDDM demonstrated this for 3D medical on single 40GB GPU | Medium (coordinate channels + crop logic + inference stitching) | Wang et al., NeurIPS 2023. [arXiv:2304.12526](https://arxiv.org/abs/2304.12526); PatchDDM, MIDL 2024 |
| 116 | **Elastic/Affine 3D Deformation Augmentation** | Beyond D4 symmetry (8x discrete), add continuous deformations: random affine + elastic B-spline warps on noisy x_t (equivariantly applied to target) | D4 gives 8x augmentation. Elastic gives infinite continuous augmentation. Brain anatomy has natural smooth variability that elastic deformations approximate. MONAI has `RandAffine` and `RandElasticDeformation` ready | Low-Medium (MONAI transforms exist) | Standard medical imaging augmentation |
| 117 | **Diffusion Mixup** | Interpolate pairs of training volumes in pixel space before adding noise: `x_mix = λ*x_a + (1-λ)*x_b`. Model denoises from interpolated examples on the convex hull of data manifold | Creates genuinely unseen training samples via convex interpolation. Different from CutMix/Mosaic (#24, compression only). Score smoothing analysis (2025) shows diffusion models already implicitly interpolate — making it explicit strengthens anti-memorization | Low (~10 lines) | On the Interpolation Effect of Score Smoothing, 2025. [arXiv:2502.19499](https://arxiv.org/html/2502.19499) |
| 118 | **Noise Consistency Regularization** | Two auxiliary losses: (a) velocity predictions for augmented versions of same sample must be consistent, (b) predictions robust to noise-level modulation. Regularizes velocity field smoothness | Directly regularizes velocity field to be smooth — the core issue with small datasets. Does not require pretrained model. Complementary to DeltaFM (#109) which pushes different samples apart | Medium (~50 lines, one extra forward pass) | CVPR 2025 Workshop. [arXiv:2506.06483](https://arxiv.org/abs/2506.06483) |
| 119 | **APT: Adaptive Personalized Training** | Per-timestep-bin overfitting detection with adaptive augmentation strength + representation stabilization (regularizes mean/variance of intermediate feature maps) | Novel: different timestep ranges overfit at different rates. Monitors and corrects this per-bin. Representation stabilization prevents feature drift with small datasets | Medium-High | CVPR 2025. [arXiv:2507.02687](https://arxiv.org/abs/2507.02687) |

### LOW — Unlikely FID Improvement

**Empirically disfavored by our experiments:**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 69 | Laplace Timestep Distribution | Heavy-tailed alternative to logit-normal for timestep sampling | RFlow training | Paper's 26.6% FID gain is vs **uniform** sampling — we already use logit-normal (#9). Timestep jitter (also a timestep-axis change) made 2D FID **worse** (+0.84). Trivial to sweep but expectations should be low. **Note**: exp1p_1 tests uniform vs logit-normal — if logit-normal wins convincingly, Laplace/Cauchy are unlikely to add more | Hang et al., ICCV 2025 |
| 70 | Cauchy Timestep Distribution | Even heavier-tailed mid-range concentration | RFlow training | Same as Laplace. If exp1p_1 shows logit-normal >> uniform, the distribution choice matters but Cauchy is untested and risky | Hang et al., ICCV 2025 |
| 71 | Input Perturbation | Small Gaussian noise on x_t during training | All modes training | Conceptually similar to noise augmentation (exp23_1), which made 2D FID worse (+0.35). Paper is DDPM/DDIM specific, not validated for RFlow | Ning et al., ICML 2023 |
| 72 | Reflow (Multi-Round) | Generate (x0, x1) pairs, retrain on straighter trajectories | Post-training | FID degrades at 50+ steps (27.5→30.25→43.39) suggesting curved trajectories, but quality ceiling is data-limited (105 volumes). Large effort (generate pairs + retrain) for uncertain gain. Better step scheduling (#65) addresses the same problem more cheaply | Liu, 2022 |
| 73 | Guidance Interval | Apply CFG only during a window of timesteps | Seg-conditioned inference | Minor tuning knob for seg-conditioned experiments only (exp2c/2d/2e). Over-guidance less of a problem for medical images than natural images | Kynkaanniemi et al., NeurIPS 2024 |
| 74 | CFG Rescale | Rescales CFG output to prevent over-saturation | Seg-conditioned inference | Minor tuning. Over-saturation less of a problem for medical images (constrained [0,1] range) than natural images | Lin et al., WACV 2024 |
| 75 | Adaptive Non-Uniform Timestep Sampling | Tracks per-timestep gradient variance to optimize allocation | RFlow training | Complex version of Laplace/Cauchy (#69/#70). If those don't help, this won't either. If they do, this is unnecessary | Kim et al., CVPR 2025 |

**Novel but higher effort or uncertain benefit:**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 109 | Contrastive Flow Matching (DeltaFM) | Adds contrastive loss pushing velocity predictions of different samples apart within a batch. Prevents conditional flow collapse | All modes with discrete conditioning | **Designed for discrete conditioning** (class labels, text via AdaLN/cross-attention) where conditional flows overlap. Bravo uses input concatenation — flow overlap problem doesn't apply. 8.9 FID gain is class-conditioned ImageNet. Could act as weak velocity regularizer unconditionally but core benefit doesn't transfer | Stoica et al., ICCV 2025. [arXiv:2506.05350](https://arxiv.org/abs/2506.05350) |
| 120 | NoiseCutMix | Cut-and-paste patches of noise/velocity estimates from different samples during training | All modes training | Novel 2025 technique. Could help diversity but unproven for flow matching. Interaction with ScoreAug unclear | [arXiv:2509.00378](https://arxiv.org/html/2509.00378v1) |
| 121 | R-Drop for Diffusion | Two forward passes with different dropout masks on same (x_t, t), KL-divergence loss between velocity predictions. Explicit consistency regularizer | All modes (requires dropout > 0, see #111) | Novel application — never been applied to diffusion models. Cheap (one extra forward pass) but requires dropout first. Could be thesis-novel contribution | Liang et al., NeurIPS 2021. [arXiv:2106.14448](https://arxiv.org/abs/2106.14448) |
| 122 | Temporal Pair Consistency (TPC-FM) | Sample two timesteps along same probability path, enforce velocity consistency between them | Flow matching training | Self-supervised regularization that reduces gradient variance. ~2x compute per step but faster convergence. Designed for flow matching | [arXiv:2602.04908](https://arxiv.org/abs/2602.04908) |
| 123 | SIMS (Self-Improving Diffusion) | Generate synthetic samples, retrain using both real + synthetic — but extrapolates away from synthetic manifold to prevent collapse | Post-training | Bootstraps 105 → larger dataset without model collapse. But DiffRS discriminator memorized 105 volumes (#59) — same risk applies to the quality signal | [arXiv:2408.16333](https://arxiv.org/abs/2408.16333) |
| 124 | Anti-Memorization Guidance (AMG) | Inference-time: detects memorization via attention patterns at high timesteps and steers sampling away from memorized outputs | Bravo inference | Post-hoc, no retraining. But adds inference complexity and unclear benefit for medical data where anatomical similarity is expected | Chen et al., CVPR 2024 |
| 125 | MULAN (Learned Adaptive Noise) | Per-voxel noise schedule that varies across spatial locations. Different noise rate for tumor vs background vs ventricles | All modes training | Theoretically ideal for medical images with heterogeneous information density. But major refactor of forward process, incompatible with standard RFlow formulation | NeurIPS 2024. [arXiv:2312.13236](https://arxiv.org/abs/2312.13236) |

**Theoretical basis but blocked by constraints:**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 76 | Saddle-Free Guidance | Label-free guidance via Hessian curvature, no extra training | Bravo inference | Training-free, SOTA unconditional FID on ImageNet. But Hessian computation adds ~2x inference cost, and benefit for medical images unclear — already have spatial conditioning | arXiv 2025 |
| 77 | Contrastive Loss (CDL) | Self-supervised loss regularizer — improves denoiser in OOD regions | All modes training | Better OOD denoising → better generation quality. But unclear negative pair definition for medical data where all samples are same anatomy | Wu et al., NeurIPS 2024 |
| 78 | Soft Min-SNR | Softer Min-SNR variant | Any architecture | Marginal benefit over regular Min-SNR (#5/#6) already implemented. 3D exp1e showed Min-SNR didn't improve val loss | Crowson, HDiT 2024 |
| 79 | Pyramid Noise | Multi-scale noise at different resolutions | All pixel-space experiments | Offset noise (#13/#14) already handles low-frequency variation. Brain MRI has relatively consistent large-scale anatomy | Whitaker, W&B report 2023 |
| 80 | P2 Weighting | Downweights loss at extreme noise levels | All strategies | Superseded by Min-SNR-gamma (#5/#6) already implemented | Choi et al., CVPR 2022 |

**Architecture-specific (not primary architecture):**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 81 | Magnitude-Preserving Layers (EDM2) | Redesigns all layers to preserve magnitudes | UNet and DiT | Massive architecture rewrite. Pixel UNet training is already stable (all exp1 runs converge) | Karras, CVPR 2024 |
| 82 | RMSNorm | Replaces LayerNorm in attention | DiT/HDiT/UViT | Only matters for DiT at scale, not primary architecture. UNet uses GroupNorm | SD3/FLUX; EDM2 |
| 83 | MaskDiT | Mask 50% patches in DiT, MAE aux loss | DiT models only | DiT not primary architecture. DiT VRAM already low (6GB), FLOPs reduction not needed | Zheng, TMLR 2024 |
| 84 | 2D RoPE | Rotary position encoding for attention | DiT/SiT/HDiT | Fixed resolution (128/256) makes learned embeddings sufficient | FLUX |

**Domain mismatch (no medical 3D vision foundation model):**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 85 | REPA | Aligns diffusion hidden states with DINOv2 — 17.5x speedup | DiT/SiT models only | DINOv2 trained on natural images, not brain MRI. Feature alignment meaningless for medical 3D | Yu et al., ICLR 2025 |
| 86 | REG | Learnable token aligned with repr encoder — 63x faster | DiT/SiT models only | Same DINOv2 problem as REPA | Wu et al., 2025 |
| 87 | REPA-E | Joint VAE + diffusion with REPA | LDM + DiT pipeline | Depends on REPA (broken for medical) + LDM (fragile in exp9/exp10) | Leng et al., ICCV 2025 |
| 88 | VA-VAE | Aligns VAE latent space with vision foundation model | LDM pipeline | Same DINOv2 problem — no medical 3D vision foundation model exists | Yao et al., 2025 |

**Wrong bottleneck (speed/capacity not the limiting factor):**

| # | Trick | What It Does | Would Apply To | Why Not | Reference |
|---|-------|-------------|----------------|---------|-----------|
| 89 | Autoguidance | Smaller/weaker model as guidance | Bravo inference | Requires training two models. Diversity is data-limited (105 volumes), not guidance-limited | Karras et al., 2024 |
| 90 | Adversarial/GAN Loss | Discriminator loss for sharper outputs | All modes | DiffRS discriminator memorized 105 volumes instead of learning density ratios. GAN training = same problem | Xiao et al., ICLR 2022 |
| 91 | FP8 Training | 8-bit floating point on H100 | All models on H100 | Quality ceiling is data-limited (105 volumes), not memory/speed-limited. 270M UNet fits in BF16 | Micikevicius 2022 |
| 92 | Consistency Distillation | Distill to 1-step model | Post-training | Speed not the bottleneck — Euler/25 is 18s/volume | Song et al., 2023 |
| 93 | Consistency Training | Train consistency model from scratch | Alternative to diffusion | Different paradigm. No evidence of benefit for small medical datasets | Song 2023; sCM 2024 |
| 94 | Progressive Distillation | Halve inference steps iteratively | Post-training | Speed not the bottleneck — 25 steps already fast enough | Salimans & Ho, 2022 |

### NOT APPLICABLE — Wrong Strategy/Domain

These don't apply to our setup (RFlow, unconditional/seg-conditioned, no text, 3D medical):

| # | Trick | Would Apply To | Why Not Applicable |
|---|-------|----------------|---------------------|
| 95 | CFG++ | DDPM/DDIM inference | Manifold-constrained guidance designed for discrete score-based diffusion (DDPM/DDIM). Rewrites the denoising update formula — incompatible with RFlow's continuous ODE/velocity fields | Chung et al., ICLR 2025 |
| 96 | Trailing Timestep Selection | DDPM inference schedule | RFlow already uses uniform linspace with dt=40. The "trailing" fix addresses DDPM's cosine schedule where the last step removes near-zero noise — in RFlow every step does equal work (linear interpolation) |
| 97 | Zero Terminal SNR | DDPM noise schedule | RFlow already has zero terminal SNR by construction: at t̃=1, x_t = pure noise. This is a DDPM-only fix |
| 98 | Learned Noise Schedule (VDM) | DDPM β schedule | RFlow has fixed linear forward process x_t = (1-t)*x_0 + t*eps — no β schedule to learn |
| 99 | Sigmoid Noise Schedule | DDPM β schedule | Same — DDPM β schedule concept, not applicable to RFlow |
| 100 | Shifted Cosine Schedule | DDPM β schedule for high-res | Same — DDPM β schedule concept, not applicable to RFlow |
| 101 | Multi-Aspect Ratio Training | Variable aspect ratio datasets | Medical images have fixed 256x256 acquisition protocol |
| 102 | Micro-Conditioning | Text-conditioned models | Requires text conditioning (crop coords, aesthetic score). No text captions for brain MRI |
| 103 | MMDiT | Text+image multimodal | Requires text modality for bidirectional image-text flow. No text in our setup |
| 104 | MoE (Mixture of Experts) | Large-scale training | Dataset too small (105 volumes), 270M UNet already adequate. More capacity = more overfitting |
| 105 | Score Distillation (SDS) | 3D NeRF / scene generation | For 3D scene generation from text — completely different domain than medical image synthesis |
| 106 | Negative Guidance | Text-conditioned models | Requires text conditioning to specify "negative prompts" |
| 107 | V-Prediction for DDPM | DDPM strategy | DDPM-only reparametrization `v = α_t*ε - σ_t*x_0`. RFlow already uses velocity prediction `v = x_0 - ε` |
| 108 | RL Fine-tuning / RLHF | Post-training quality improvement | Out of scope for thesis. Requires reward model, RL infrastructure, and is an open research problem for medical imaging |

---

## Summary

| Category | Count |
|----------|-------|
| Implemented | 67 (66 functional + 1 dead config) |
| Not implemented — HIGH (novel axis, strong potential) | 4 (#110–#113) |
| Not implemented — MEDIUM (worth trying) | 7 (#68, #114–#119) |
| Not implemented — LOW (unlikely FID improvement) | 33 (#69–#80, #109, #120–#125, existing #81–#94) |
| Not applicable (wrong strategy/domain) | 14 (#95–#108) |
| **Total** | **125** |
