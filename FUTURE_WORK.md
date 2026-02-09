# Future Work

Potential improvements and experiments for the diffusion-based medical image synthesis project.

---

## Implemented Features

Features that have been completed and are available in the codebase.

### Training & Infrastructure
- [x] **Separate training pipelines** - Different configs for seg vs bravo via Hydra YAML configs
- [x] **Min-SNR weighting** - Reweight loss across timesteps to prevent high-noise step domination
- [x] **EMA weights** - Slowly-updated parameter copy for higher quality samples
- [x] **LR Finder** - Automatic optimal learning rate search (`lr_finder.py`)
- [x] **Warmup Cosine Scheduler** - Linear warmup followed by cosine annealing
- [x] **Track loss by diffusion step** - Timestep-binned loss analysis (10 bins, normalized [0,1])
- [x] **SAM/ASAM optimizer** - Sharpness-Aware Minimization for flat minima
- [x] **Pure BF16 weights** - Model weights stored in BF16 for ~50% memory savings
- [x] **torch.compile** - Fused forward pass optimization
- [x] **Multi-GPU DDP** - Distributed training support
- [x] **Gradient checkpointing** - Memory reduction for 3D training

### Conditioning & Guidance
- [x] **Classifier-free guidance (CFG)** - Condition dropout training, interpolate cond/uncond at inference
- [x] **Controllable mask generation** - `seg_conditioned` mode with tumor size bin conditioning
- [x] **ControlNet** - Pixel-resolution conditioning for latent diffusion (two-stage training)
- [x] **Mode embedding** - Multi-modality conditioning (bravo, flair, t1_pre, t1_gd) via FiLM

### Architectures
- [x] **DiT (Diffusion Transformer)** - Scalable Interpolant Transformer (S/B/L variants)
- [x] **Latent diffusion** - VAE/DC-AE compressed latent space diffusion (8x-128x compression)
- [x] **Wavelet diffusion** - 3D Haar wavelet frequency decomposition space
- [x] **Space-to-depth** - Lossless 2x2x2 spatial-to-channel rearrangement (3D)
- [x] **DC-AE** - Deep Compression Autoencoder (32x/64x/128x spatial compression)
- [x] **VQ-VAE** - Vector-quantized discrete latent space
- [x] **DC-AE 1.5** - Structured latent space with channel masking

### Regularization
- [x] **ScoreAug / v2** - Augments noisy data (rotation, flip, translation, cutout, learned patterns)
- [x] **SDA** - Shifted Data Augmentation (clean augmentation with timestep compensation)
- [x] **Curriculum timesteps** - Train from easier to harder denoising tasks over warmup period
- [x] **Regional loss weighting** - Upweight small tumor regions (RANO-BM clinical thresholds)

### Evaluation
- [x] **Generation metrics** - KID, CMMD, FID tracking during training (ResNet50 + BiomedCLIP)
- [x] **Downstream segmentation** - SegResNet trainer with per-tumor-size Dice evaluation
- [x] **3D volume generation** - Unified 2D/3D via `spatial_dims` parameter

---

## Remaining Ideas

### Conditioning Methods

Current implementation uses **channel concatenation** (`[noise, mask]` → UNet). Multiple conditioning methods can be combined since they operate at different levels.

| Category | Method | Mechanism | Status |
|----------|--------|-----------|--------|
| **Input** | Channel Concat | `[noise, cond]` as input | Done |
| **Normalization** | SPADE | Spatial γ(x,y), β(x,y) from mask | Not started |
| **Normalization** | FiLM | Feature-wise γ * x + β | Done (mode embed) |
| **Architecture** | ControlNet | Parallel trainable encoder | Done |
| **Inference** | CFG | Interpolate cond/uncond | Done |

Methods are **orthogonal** - they work at different levels:

| Step | Method | Benefit |
|------|--------|---------|
| 1 | Channel Concat | Baseline, simple |
| 2 | + CFG | Control strength at inference |
| 3 | + SPADE | Stronger mask adherence |

- [ ] **SPADE conditioning** - Spatial adaptive normalization for stronger mask adherence

---

### Computational Performance

- [ ] **NVIDIA 2:4 structured sparsity** - Up to 2x speedup for linear layers (A100/H100, BF16 weights ready)
- [ ] **Network pruning** - Magnitude/gradient-based pruning (diffusion models tolerate significant pruning)
- [ ] **RFlow reflow distillation** - Train on straightened trajectories for better quality
- [ ] **DDIM / DPM-Solver++** - Deterministic samplers for faster inference
- [ ] **Learned step scheduling** - Optimal step sizes learned rather than heuristic

---

### Quality Improvements

- [ ] **Alternative noise schedules** - Laplace, Cauchy schedules (concentrate noise at mid-range timesteps)
- [ ] **Frequency loss (FFT)** - Preserve high-frequency details that MSE blurs
- [ ] **MMDiT** - Separate transformer branches for conditioning and noisy inputs

---

### Reward-Based Fine-Tuning

- [ ] **DRaFT (Direct Reward Fine-Tuning)** - Train distortion classifier, backprop through it to reduce failure rate
- [ ] **Distortion detector** - Binary classifier (good/distorted) on generated samples as prerequisite for DRaFT
- [ ] **Distortion dataset** - Label ~500 images for DRaFT classifier training

---

### Data-Centric Approaches

#### Quality Gates for Synthetic Data
When generating 15,000 synthetic images, ~6-10% are distorted. A quality gate would:
1. Run every generated image through a distortion classifier
2. Only keep images that pass (confidence > threshold)
3. Generate extra to compensate for rejected samples

Simpler than DRaFT (no fine-tuning needed), gives immediate improvement to downstream segmentation.

- [ ] **Implement quality gates** - Auto-reject distorted samples before downstream use

#### Hard Example Mining
Segmentation models fail more on certain cases (small tumors, multiple lesions, tumors near the edge):
1. Run segmentation model on real test set, identify failure cases
2. Find mask characteristics that correlate with failures (size, location, count)
3. Bias mask generation toward these hard cases

- [ ] **Hard example mining** - Generate more of the cases segmentation model fails on

#### Per-Patient Failure Analysis
The 6-10% distortion rate is an average. Is it random or concentrated in specific patients?
- Generate 100 images per patient, measure distortion rate per patient
- If some patients have 30% distortion while others have 2%, indicates data quality issues or anatomical edge cases

- [ ] **Per-patient failure analysis** - Identify if certain anatomies cause more distortions

---

### Medical-Specific Techniques

#### Anatomy-Aware Loss
Current loss treats all pixels equally, but tumor in ventricles is anatomically impossible:
1. Use brain parcellation (FreeSurfer or atlas-based)
2. Penalize high tumor probability in anatomically implausible regions
3. Can be soft (weighted loss) or hard (mask out impossible regions)

- [ ] **Anatomy-aware loss** - Penalize anatomically implausible generations

#### Brain Atlas Registration
MNI152 is the standard brain template:
1. Register each generated brain to MNI template
2. Measure registration quality (mutual information)
3. Poor registration = anatomically abnormal generation

- [ ] **Brain atlas registration** - Check if generated brains align to MNI template

#### Cycle Consistency
Validates both seg and bravo models together:

**Forward cycle**: real_mask → generated_image → predicted_mask ≈ real_mask
**Backward cycle**: real_image → predicted_mask → generated_image ≈ real_image

- [ ] **Cycle consistency check** - Validate seg and bravo models are coherent

---

### Inference-Time Improvements

#### Best-of-N Sampling
At inference, instead of generating 1 image per mask:
1. Generate N=4 images from same mask (different noise seeds)
2. Run distortion classifier on all 4
3. Keep the one with lowest distortion probability

Costs 4x compute but requires no retraining. Could reduce distortion rate from 6% to <1%.

- [ ] **Best-of-N sampling** - Generate multiple, pick best by classifier (immediate win)

---

### Interpretability & Analysis

- [ ] **Attention visualization** - Understand where model/DiT focuses during generation
- [ ] **Failure mode taxonomy** - Categorize distortion types systematically (anatomical, textural, boundary, intensity, artifacts, conditioning failure)
- [ ] **Timestep distortion analysis** - Find which denoising steps introduce failures

---

### Uncertainty Quantification

#### Monte Carlo Sampling
Generate same mask→image 10 times with different noise:
```python
images = [generate(mask, seed=i) for i in range(10)]
mean_image = torch.stack(images).mean(0)
variance_map = torch.stack(images).var(0)
```

High variance regions = model is uncertain. Valuable for medical applications.

- [ ] **Monte Carlo uncertainty** - Generate multiple samples, compute variance map
- [ ] **Ensemble models** - Train multiple models, use disagreement as uncertainty

---

### Validation

- [ ] **Combine synthetic + traditional augmentation** - Maximize data diversity
- [ ] **Dual T1 evaluation** - Quantitative metrics for paired pre/post-contrast generation
- [ ] **Clinical expert evaluation** - Blinded radiologist studies

---

## Priority Ranking

### Quick Wins (1-2 days)
1. **Best-of-N sampling** - Immediate quality improvement, no retraining
2. **Quality gates** - Auto-reject distorted samples

### Medium Effort (3-5 days)
3. **Failure mode taxonomy** - Essential thesis content
4. **Label distortion dataset** - Prerequisite for DRaFT, ~500 images
5. **SPADE conditioning** - Stronger mask adherence

### Thesis-Worthy (1-2 weeks)
6. **DRaFT for distortion reduction** - Novel application to medical imaging
7. **Cycle consistency analysis** - Validates entire pipeline
8. **Monte Carlo uncertainty** - Medical application value
