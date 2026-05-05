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

## Do Not Assume

Don't assume anything that wasn't explicitly stated. If something is unclear or ambiguous, **ask before implementing**.

---

## Key Terminology (Don't Confuse)

| Term | Meaning |
|------|---------|
| **Mode** | WHAT to generate: seg, bravo, dual, triple, multi, multi_modality, seg_conditioned{,_3d}, seg_conditioned_input{,_3d}, bravo_seg_cond, seg_compression, restoration |
| **Strategy** | HOW to denoise: ddpm, rflow (continuous timesteps by default), bridge, irsde, resfusion |
| **Architecture** | UNet, DiT/SiT, HDiT (hierarchical), UViT (skip-connection ViT), Mamba (LaMamba-Diff, pixel-space only), WDM (Wavelet Diffusion Model, 3D-only) |
| **VAE dual** | 2 channels (t1_pre, t1_gd) - NO seg |
| **Diffusion dual** | 3 channels (t1_pre, t1_gd, seg) - HAS seg |
| **Diffusion triple** | 4 channels (t1_pre, t1_gd, flair, seg) - HAS seg |
| **seg_conditioned** | Generate seg masks conditioned on tumor sizes (FiLM embedding); 3D variant in `seg_conditioned_3d.yaml` |
| **seg_conditioned_input** | Generate seg masks with size bins as channel-concat input; 3D variant in `seg_conditioned_input_3d.yaml` |
| **bravo_seg_cond** | Latent diffusion: generate BRAVO latents conditioned on VQ-VAE seg latents |
| **multi_modality** | VAE/VQ-VAE/DC-AE pre-training on mixed slices from all modalities (single-channel, non-conditional) |
| **seg_compression** | DC-AE compression of segmentation masks (BCE+Dice+Boundary loss, Dice/IoU metrics) |
| **DiffRS** | Diffusion Rejection Sampling - post-hoc discriminator for quality filtering |
| **HandoffWrapper** | Two-stage inference: seed/base model for t > handoff_t, fine-tuned model for t ≤ handoff_t (e.g. exp48 low-t fine-tunes). Configured at generation time, not via `configs/model/` |
| **bridge strategy** | Diffusion Bridge Model for paired restoration (Zhang et al. 2025, arXiv:2504.15267); γ_max=0.125 for 3D brain MRI |
| **irsde strategy** | IR-SDE mean-reverting SDE (Luo et al., ICML 2023); L1 loss, posterior sampling at inference |
| **resfusion strategy** | Resfusion residual noise diffusion (Shi et al., NeurIPS 2024); short T=12 schedule, ~5–12 reverse steps |
| **WDM** | Wavelet Diffusion Model (Friedrich et al. 2024, arXiv:2402.19043); 3D-only, requires `strategy.prediction_type=sample` (x₀) and `wavelet=default`. Config: `model=wdm_3d` |
| `train.py` | Diffusion (2D default, use `model.spatial_dims=3` for 3D) |
| `train_compression.py` | Unified compression training (VAE/VQ-VAE/DC-AE, use `--config-name=` to select) |
| **Continuous timesteps** | RFlow with `use_discrete_timesteps: false` - floats in [0, 1000] |
| **Discrete timesteps** | DDPM - integers in [0, 999] |
| **RFlow t convention** | MONAI: t=0 → clean, t=T (t̃=1) → noise. `x_t = (1-t̃)*x₀ + t̃*ε`. REVERSED from original RF paper (Liu: t=0→noise). |
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

# === DIFFUSION (Mamba / LaMamba-Diff, pixel-space only) ===
python -m medgen.scripts.train model=mamba model.variant=S mode=bravo strategy=rflow   # 2D
python -m medgen.scripts.train model=mamba_3d model.variant=S mode=bravo strategy=rflow model.spatial_dims=3  # 3D

# === DIFFUSION (WDM, wavelet-domain, 3D only) ===
python -m medgen.scripts.train model=wdm_3d wavelet=default mode=bravo \
    strategy=ddpm strategy.prediction_type=sample model.spatial_dims=3

# === RESTORATION (RFlow / IR-SDE / Bridge / Resfusion) ===
# All four strategies work with mode=restoration per configs/mode/restoration.yaml
python -m medgen.scripts.train mode=restoration strategy=rflow model.spatial_dims=3
python -m medgen.scripts.train mode=restoration strategy=irsde model.spatial_dims=3
python -m medgen.scripts.train mode=restoration strategy=bridge model.spatial_dims=3
python -m medgen.scripts.train mode=restoration strategy=resfusion model.spatial_dims=3

# === RESTORE GENERATED VOLUMES (post-hoc) ===
python -m medgen.scripts.restore_volumes \
    --restoration-model runs/diffusion_3d/.../checkpoint_best.pt \
    --input-dir <path-to-generated-volumes> \
    --output-dir <restored-output-dir>

# === GENERATE WITH HANDOFF (two-stage low-t/high-t) ===
# generate.py: Hydra config keys (image_model = low-t, image_model_high_t = base)
python -m medgen.scripts.generate mode=bravo \
    image_model=<fine-tuned-low-t-model.pt> \
    image_model_high_t=<base-model.pt> \
    handoff_t=0.25

# find_optimal_steps.py: CLI flags
python -m medgen.scripts.find_optimal_steps \
    --high-t-checkpoint <base-model.pt> \
    --low-t-checkpoint <fine-tuned-low-t-model.pt> \
    --handoff-t 0.25 \
    --data-root ~/MedicalDataSets/brainmetshare-3 --output-dir eval_handoff

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
| `@docs/common-pitfalls.md` | 87 known issues, bug fixes, and gotchas (numbered 1-88 with #43 intentionally skipped → 87 actual entries) |
| `@docs/commands.md` | Full command reference with all options |
| `@docs/eval-ode-solvers.md` | ODE solver evaluation results (Euler/25 optimal for RFlow) |
| `@docs/experiment_results.md` | Comprehensive 2D experiment results and metrics |
| `@docs/experiment_results_3d.md` | 3D experiment results (pixel, latent, compression, downstream) |
| `@docs/profiling_results.md` | VRAM profiling for DiT, UNet, HDiT, UViT |
| `@docs/proven_techniques.md` | Confirmed positive/negative techniques for 3D brain MRI generation |
| `@docs/future_work_v2.md` | 125 diffusion tricks inventory (67 implemented, 58+ not) |
| `@docs/scoreaug_omega.md` | ScoreAug omega-vector encoding spec (paper conformance, layout, identity-as-zeros invariant) |
| `@docs/notes_for_report.txt` | Historical design notes (Dec 2025) — VAE features, ScoreAug, DiT, VQ-VAE |
| `@papers/PAPERS.md` | Reference papers (VAE, DDPM, RFlow, DC-AE, etc.) — **check here FIRST before web search.** Note: directory is in `.gitignore`, so the file is local-only |

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

**Loss schedules (in `training/default.yaml`, applied via `_compute_t_schedule_weight()`):**
- `training.perceptual_weight: 0.0` (default; >0 enables) + `training.perceptual_max_timestep: null` (default; legacy ramp — set integer N to apply LPIPS only when t < N) — LPIPS at low t (legacy mode); exp32_2 family uses `perceptual_weight=0.1 perceptual_max_timestep=250`
- `training.perceptual_t_schedule: null` (default; set `[t_on, t_full, t_off]`) — piecewise-linear schedule in normalized [0,1] units (zero below t_on, ramps to 1 at t_full, drops to 0 at t_off)
- `training.focal_frequency_weight: 0.0` (default) + `training.focal_frequency_t_schedule: null` (default) — focal frequency loss with same piecewise schedule (slice-wise FFT for 3D)
- All schedules disabled by default; enable per-experiment in SLURM overrides

**Two-stage inference (`src/medgen/models/handoff.py`):**
- `HandoffWrapper` combines two checkpoints: high-t (seed/base) and low-t (fine-tuned)
- Used at inference only. `generate.py` reads Hydra keys `image_model` (low-t), `image_model_high_t` (base), `handoff_t` (default 0.25 from `configs/generate.yaml`). `find_optimal_steps.py` uses argparse flags `--high-t-checkpoint`/`--low-t-checkpoint`/`--handoff-t`.
- No `configs/model/handoff.yaml` — wrapper is constructed at runtime from the two model paths

---

## Core Principles

- **DRY** - Don't repeat yourself
- **KISS** - Keep it simple
- **SRP** - Single responsibility per function
- **Fail Fast** - Raise errors early, never suppress failures
