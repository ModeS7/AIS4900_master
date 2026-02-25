# MedGen Project

## CRITICAL: Definition of Done (READ FIRST)

Before marking ANY task complete, STOP and verify:

1. **All requirements addressed** - Re-read the original request. Did you do everything asked?
2. **All files modified** - If request mentions multiple files, did you touch all of them?
3. **Both 2D AND 3D** - If modifying a trainer/loader, did you update both variants?
4. **Config + Code** - If adding a feature, did you add BOTH the config option AND the implementation?
5. **Syntax check passed** - Run: `python3 -m py_compile <modified_files>`
6. **No partial implementations** - If you hit a blocker, say so. Don't pretend it's done.

**If unsure about any requirement, ASK before implementing.**

---

## Communication Style

- Be honest, not agreeable
- If I'm wrong, tell me directly
- If you don't know, say "I don't know"
- Just implement what I ask - OR push back with honest reasons
- No flattery, no false reassurance

---

## Key Terminology (Don't Confuse)

| Term | Meaning |
|------|---------|
| **Mode** | WHAT to generate: seg, bravo, dual, multi, seg_conditioned, seg_conditioned_input, bravo_seg_cond |
| **Strategy** | HOW to denoise: ddpm, rflow (continuous timesteps by default) |
| **Architecture** | UNet, DiT/SiT, HDiT (hierarchical), UViT (skip-connection ViT) |
| **VAE dual** | 2 channels (t1_pre, t1_gd) - NO seg |
| **Diffusion dual** | 3 channels (t1_pre, t1_gd, seg) - HAS seg |
| **seg_conditioned** | Generate seg masks conditioned on tumor sizes (FiLM embedding) |
| **seg_conditioned_input** | Generate seg masks with size bins as channel-concat input |
| **bravo_seg_cond** | Latent diffusion: generate BRAVO latents conditioned on VQ-VAE seg latents |
| **DiffRS** | Diffusion Rejection Sampling - post-hoc discriminator for quality filtering |
| `train.py` | Diffusion (2D default, use `model.spatial_dims=3` for 3D) |
| `train_compression.py` | Unified compression training (VAE/VQ-VAE/DC-AE, use `--config-name=` to select) |
| **Continuous timesteps** | RFlow with `use_discrete_timesteps: false` - floats in [0, 1000] |
| **Discrete timesteps** | DDPM - integers in [0, 999] |
| **Voxel spacing (NIfTI)** | `compute_voxel_size()` returns `(x, y, z)` = `(0.9375, 0.9375, 1.0)` for affine matrices |
| **Voxel spacing (3D bins)** | Config `voxel_spacing` is `[D, H, W]` = `[1.0, 0.9375, 0.9375]` for `compute_feret_diameter_3d` |

---

## Ask Before Assuming

Stop and ask if unclear:
1. Which script? (2D vs 3D)
2. Which mode? (seg, bravo, dual, multi)
3. Pixel vs latent space?

---

## Quick Commands

```bash
# === DIFFUSION (UNet) ===
python -m medgen.scripts.train mode=bravo strategy=rflow                    # 2D
python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3  # 3D

# === DIFFUSION (DiT/HDiT/UViT) ===
python -m medgen.scripts.train model=dit model.variant=S mode=bravo strategy=rflow   # DiT
python -m medgen.scripts.train model=hdit_3d model.variant=S mode=bravo strategy=rflow model.spatial_dims=3  # HDiT
python -m medgen.scripts.train model=uvit_3d model.variant=S mode=bravo strategy=rflow model.spatial_dims=3  # UViT

# === VAE ===
python -m medgen.scripts.train_compression --config-name=vae mode=multi_modality
python -m medgen.scripts.train_compression --config-name=vae_3d mode=multi_modality

# === VQ-VAE ===
python -m medgen.scripts.train_compression --config-name=vqvae mode=multi_modality
python -m medgen.scripts.train_compression --config-name=vqvae_3d mode=multi_modality

# === DC-AE ===
python -m medgen.scripts.train_compression --config-name=dcae mode=multi_modality          # 32x
python -m medgen.scripts.train_compression --config-name=dcae dcae=f64 mode=multi_modality # 64x
python -m medgen.scripts.train_compression --config-name=dcae_3d mode=multi_modality

# === LATENT DIFFUSION ===
python -m medgen.scripts.train mode=bravo strategy=rflow \
    latent.enabled=true \
    latent.compression_checkpoint=runs/compression_2d/.../checkpoint_best.pt

# === SYNTAX CHECK ===
python3 -m py_compile src/medgen/**/*.py
```

For full command reference, see `@docs/commands.md`

---

## Before SLURM Submit (MANDATORY)

Run validation before every cluster submission:

```bash
./misc/validate_before_submit.sh IDUN/train/your_job.slurm
```

This catches:
- Syntax errors
- Import failures
- Config resolution issues
- Runtime errors (1-batch dry run)

**DO NOT skip this step.** It prevents 80% of fix commits.

---

## Detailed Documentation

| Doc | Contents |
|-----|----------|
| `@docs/architecture.md` | File locations, trainer hierarchy, config structure, TensorBoard metrics |
| `@docs/common-pitfalls.md` | 82 known issues, bug fixes, and gotchas (numbered 1-82, #43 skipped) |
| `@docs/commands.md` | Full command reference with all options |
| `@docs/eval-ode-solvers.md` | ODE solver evaluation results (Euler/25 optimal for RFlow) |
| `@docs/experiment_results.md` | Comprehensive 2D experiment results and metrics |
| `@docs/experiment_results_3d.md` | 3D experiment results (pixel, latent, compression) |
| `@docs/profiling_results.md` | VRAM profiling for DiT, UNet, HDiT, UViT |
| `@papers/PAPERS.md` | Reference papers (VAE, DDPM, RFlow, DC-AE, etc.) |

---

## Code Patterns

**Trainer hierarchy:**
```
BaseTrainer
├── DiffusionTrainerBase (abstract)
│   └── DiffusionTrainer (unified 2D/3D via spatial_dims parameter)
├── BaseCompressionTrainer
│   ├── VAETrainer (unified 2D/3D via .create_3d() factory)
│   ├── VQVAETrainer (unified 2D/3D via .create_3d() factory)
│   └── DCAETrainer (unified 2D/3D via .create_3d() factory)
└── SegmentationTrainer (downstream, unified 2D/3D)
```

**Key rules:**
- Always `.float()` before loss computation (BF16 precision bug)
- Save/restore RNG state around validation code
- Mode embedding requires homogeneous batches (GroupedBatchSampler)
- RFlow continuous timesteps: Generation must scale [0,1] → [0, num_train_timesteps] for model input
- Timestep jitter: Must normalize to [0,1] first, then scale back after clamping
- RFlow Euler integration: Use ADDITION (x + dt*v) - velocity points toward clean data (v = x_0 - x_1)

**Unified Metrics (`src/medgen/metrics/unified.py`) - MANDATORY:**
- NEVER add visualization/metrics methods to trainers
- ALWAYS use unified metrics: `log_worst_batch()`, `log_reconstruction_figure()`, `log_denoising_trajectory()`, etc.
- If missing functionality: extend `unified.py`, make it work for 2D/3D and diffusion/autoencoder
- NEVER duplicate code between 2D/3D trainers

**Batch data handling (`src/medgen/diffusion/batch_data.py`):**
- Use `BatchData.from_raw(data)` for standardized batch unpacking (tensor batches)
- Handles: Tensor, 2-tuple (images/labels or seg/size_bins), 3-tuple, dict formats
- For numpy arrays from raw datasets, convert to tensors first

---

## Core Principles

- **DRY** - Don't repeat yourself
- **KISS** - Keep it simple
- **SRP** - Single responsibility per function
- **Fail Fast** - Raise errors early, never suppress failures
