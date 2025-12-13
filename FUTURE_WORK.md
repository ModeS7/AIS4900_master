# Future Work

Potential improvements and experiments for the diffusion-based medical image synthesis project.

---

## Immediate Extensions

- [x] **Separate training pipelines** - Different configs for seg vs bravo via Hydra YAML configs
- [ ] **Classifier-free guidance** - Train with random mask dropout, interpolate conditional/unconditional at inference
- [ ] **Controllable mask generation** - Cross-attention conditioning for tumor count/size/location
- [x] **Min-SNR weighting** - Reweight loss across timesteps to prevent high-noise step domination (IMPLEMENTED)
- [x] **EMA weights** - Maintain slowly-updated parameter copy for higher quality samples (IMPLEMENTED)
- [x] **LR Finder** - Automatically find optimal learning rate (IMPLEMENTED in trainer)
- [x] **Warmup Cosine Scheduler** - Linear warmup followed by cosine annealing (IMPLEMENTED)

---

## Conditioning Methods

Current implementation uses **channel concatenation** (`[noise, mask]` → UNet). Multiple conditioning methods can be combined since they operate at different levels.

### Conditioning Taxonomy

| Category | Method | Mechanism | Use Case |
|----------|--------|-----------|----------|
| **Input** | Channel Concat ✅ | `[noise, cond]` as input | Dense spatial (current) |
| **Normalization** | SPADE | Spatial γ(x,y), β(x,y) from mask | Seg→image |
| | AdaIN | Global γ, β from condition | Style transfer |
| | FiLM | Feature-wise γ * x + β | General modulation |
| **Attention** | Cross-Attention | Q=image, K/V=condition | Text prompts |
| | Reference Attention | Attend to reference features | Image-guided |
| | IP-Adapter | Image embeddings → cross-attn | Image prompts |
| **Architecture** | ControlNet | Parallel trainable encoder | Add control to pretrained |
| | T2I-Adapter | Small adapter networks | Lightweight control |
| **Inference** | CFG | Interpolate cond/uncond | Control strength |
| | Classifier Guidance | Classifier gradients | Class-conditional |

### Combining Methods

Methods are **orthogonal** - they work at different levels:

```
Input: [noise, mask]     ← Channel Concat (WHAT goes in)
         │
    ┌────▼────┐
    │  Conv   │
    │ + SPADE │           ← SPADE (HOW features processed)
    └────┬────┘
         │
    [prediction]
         │
    CFG inference         ← CFG (HOW MUCH to follow condition)
```

| Step | Method | Benefit |
|------|--------|---------|
| 1 ✅ | Channel Concat | Baseline, simple |
| 2 | + CFG | Control strength at inference |
| 3 | + SPADE | Stronger mask adherence |

---

## Code/Architecture TODOs

- [ ] **Add network layer** - Increase model capacity (currently 3 levels: 128, 256, 256)
- [ ] **RFlow 100-step training** - Compare quality vs 1000 steps for faster inference
- [ ] **Fix test segmentation masks** - Dataset preprocessing issue
- [ ] **Distortion dataset** - Label ~500 images for DRaFT classifier training
- [ ] **Distortion classifier** - Train as DRaFT reward model
- [ ] **Noise schedule comparison** - Systematic cosine/linear/sigmoid test

---

## Computational Performance

- [ ] **Pure BF16 + structured sparsity** - Redesign noise scheduler for bfloat16 precision, enable 2:4 sparsity
- [ ] **Network pruning** - Magnitude/gradient-based pruning (diffusion models tolerate significant pruning)
- [ ] **Adaptive step sizing for RFlow** - Larger steps early, smaller steps for fine details
- [ ] **RFlow reflow distillation** - Train on straightened trajectories for better quality
- [ ] **Train RFlow on fewer steps** - 100 steps instead of 1000 (straight trajectories may allow this)
- [ ] **DDIM / DPM-Solver++** - Deterministic samplers for faster inference
- [ ] **Learned step scheduling** - Optimal step sizes learned rather than heuristic

---

## Quality Improvements

- [ ] **Track loss by diffusion step** - Analyze which timesteps are hardest to learn
- [ ] **Curriculum learning** - Train sequentially from easier to harder denoising tasks
- [ ] **Network scaling** - More channels/layers (current: 128, 256, 256)
- [ ] **Add network layers** - Increase model capacity
- [ ] **Alternative noise schedules** - Laplace, Cauchy schedules (concentrate noise at mid-range timesteps)
- [ ] **Progressive resolution training** - Train 64→128→256, may reduce distortion rate
- [ ] **Frequency loss (FFT)** - Preserve high-frequency details that MSE blurs

---

## Architectural Alternatives

- [ ] **Diffusion Transformers (DiT)** - Superior scaling, global receptive field for anatomical relationships
- [ ] **MMDiT** - Separate transformer branches for conditioning and noisy inputs
- [ ] **SiT** - Scalable interpolant transformer
- [ ] **Mamba / State Space Models** - Linear complexity, enable higher resolutions or 3D

---

## Latent Diffusion Revisited

- [ ] **Gradual compression** - 4x or 8x instead of 64x spatial compression
- [ ] **Multi-scale VAE** - Gradual compression through multiple stages
- [ ] **Wavelet Diffusion Model** - Wavelet space instead of pixel/latent space

---

## Reward-Based Fine-Tuning

- [ ] **DRaFT (Direct Reward Fine-Tuning)** - Train distortion classifier, backprop through it to reduce failure rate
- [ ] **Distortion detector** - Binary classifier (good/distorted) on generated samples as prerequisite for DRaFT

---

## Data-Centric Approaches

### Quality Gates for Synthetic Data
When generating 15,000 synthetic images, ~6-10% are distorted. A quality gate would:
1. Run every generated image through a distortion classifier
2. Only keep images that pass (confidence > threshold)
3. Generate extra to compensate for rejected samples

Simpler than DRaFT (no fine-tuning needed), gives immediate improvement to downstream segmentation.

- [ ] **Implement quality gates** - Auto-reject distorted samples before downstream use

### Hard Example Mining
Segmentation models fail more on certain cases (small tumors, multiple lesions, tumors near the edge):
1. Run segmentation model on real test set, identify failure cases
2. Find mask characteristics that correlate with failures (size, location, count)
3. Bias mask generation toward these hard cases

- [ ] **Hard example mining** - Generate more of the cases segmentation model fails on

### Per-Patient Failure Analysis
The 6-10% distortion rate is an average. Is it random or concentrated in specific patients?
- Generate 100 images per patient, measure distortion rate per patient
- If some patients have 30% distortion while others have 2%, indicates data quality issues or anatomical edge cases

- [ ] **Per-patient failure analysis** - Identify if certain anatomies cause more distortions

---

## Medical-Specific Techniques

### Anatomy-Aware Loss
Current loss treats all pixels equally, but tumor in ventricles is anatomically impossible:
1. Use brain parcellation (FreeSurfer or atlas-based)
2. Penalize high tumor probability in anatomically implausible regions
3. Can be soft (weighted loss) or hard (mask out impossible regions)

- [ ] **Anatomy-aware loss** - Penalize anatomically implausible generations

### Brain Atlas Registration
MNI152 is the standard brain template:
1. Register each generated brain to MNI template
2. Measure registration quality (mutual information)
3. Poor registration = anatomically abnormal generation

Gives quantitative "anatomical plausibility" score without expert annotation.

- [ ] **Brain atlas registration** - Check if generated brains align to MNI template

### Cycle Consistency
Validates both seg and bravo models together:

**Forward cycle**: real_mask → generated_image → predicted_mask ≈ real_mask
**Backward cycle**: real_image → predicted_mask → generated_image ≈ real_image

Large cycle errors indicate models aren't learning consistent mappings.

- [ ] **Cycle consistency check** - Validate seg and bravo models are coherent

---

## Inference-Time Improvements

### Best-of-N Sampling
At inference, instead of generating 1 image per mask:
1. Generate N=4 images from same mask (different noise seeds)
2. Run distortion classifier on all 4
3. Keep the one with lowest distortion probability

Costs 4x compute but requires no retraining. Could reduce distortion rate from 6% to <1%.

- [ ] **Best-of-N sampling** - Generate multiple, pick best by classifier (immediate win)

---

## Interpretability & Analysis

### Attention Visualization
U-Net has attention layers - visualize what they attend to:
1. Extract attention maps during generation
2. For a given output region (e.g., tumor), which input regions have high attention?
3. Does attention focus on conditioning mask appropriately?

- [ ] **Attention visualization** - Understand where model focuses during generation

### Failure Mode Taxonomy
The 6-10% distortions aren't all the same. Systematically categorize them:
- **Anatomical**: Brain shape wrong, ventricles missing/duplicated
- **Textural**: Unrealistic tissue texture, blurring
- **Boundary**: Tumor edges poorly defined
- **Intensity**: Wrong contrast, too bright/dark
- **Artifacts**: Checkerboard patterns, stripes
- **Conditioning failure**: Tumor in wrong location vs mask

Label 100 distorted samples, compute frequencies. Shows what to fix first.

- [ ] **Failure mode taxonomy** - Categorize distortion types systematically

### Timestep Analysis
At which denoising step do distortions appear?
1. Save intermediate outputs at t=1000, 750, 500, 250, 100, 50, 0
2. Run distortion classifier on each
3. Plot distortion probability vs timestep

If distortions appear early (high t) = global structure issues. Late (low t) = fine detail issues.

- [ ] **Timestep distortion analysis** - Find which denoising steps introduce failures

---

## Uncertainty Quantification

### Monte Carlo Sampling
Generate same mask→image 10 times with different noise:
```python
images = [generate(mask, seed=i) for i in range(10)]
mean_image = torch.stack(images).mean(0)
variance_map = torch.stack(images).var(0)
```

High variance regions = model is uncertain. Valuable for medical applications.

- [ ] **Monte Carlo uncertainty** - Generate multiple samples, compute variance map

### Ensemble Disagreement
Train 3 models with different random seeds:
1. Generate from all 3 models
2. Compute pixel-wise disagreement
3. High disagreement = epistemic uncertainty

More expensive but captures model uncertainty rather than sampling uncertainty.

- [ ] **Ensemble models** - Train multiple models, use disagreement as uncertainty

---

## Validation

- [ ] **Downstream segmentation evaluation** - Train nnU-Net/MedNeXt on real+synthetic, measure improvement
- [ ] **Combine synthetic + traditional augmentation** - Maximize data diversity
- [ ] **Dual T1 evaluation** - Quantitative metrics for paired pre/post-contrast generation
- [ ] **Clinical expert evaluation** - Blinded radiologist studies

---

## 3D Volume Generation

- [ ] **3D LDM (MAISI)** - Fine-tune pretrained 3D medical image synthesis models
- [ ] **Spatial consistency** - Ensure inter-slice coherence for volumetric analysis

---

## Priority Ranking (Vacation Projects)

### Quick Wins (1-2 days)
1. **Best-of-N sampling** - Immediate quality improvement, 1 hour to implement
2. **EMA weights** - Simple addition, often +5-10% quality
3. **RFlow 100 steps** - May match DDPM quality with 10x faster inference

### Medium Effort (3-5 days)
4. **Failure mode taxonomy** - Essential thesis content
5. **Label distortion dataset** - Prerequisite for DRaFT, ~500 images
6. **Classifier-free guidance** - Standard technique, good thesis contribution

### Thesis-Worthy (1-2 weeks)
7. **DRaFT for distortion reduction** - Novel application to medical imaging
8. **Cycle consistency analysis** - Validates entire pipeline
9. **Downstream segmentation study** - Ultimate validation of synthetic data utility
