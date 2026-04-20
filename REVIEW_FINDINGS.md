# Full-Codebase Review — Findings Report

**Date:** 2026-04-19
**Scope:** `src/medgen/` (~83k LoC) + `tests/` (1312 tests) + `scripts/` mega-functions + `configs/*.yaml` + `IDUN/**/*.slurm` + `docs/`
**Methodology:** 10 subsystem subagents (Phase 1: 6, Phase 2: 4) + 6 cross-cutting sweeps (Phase 3).
**Verification pass:** 5 additional verifier agents re-checked every finding against the actual source. Verdicts appended per finding as **[CONFIRMED]**, **[PARTIAL]**, or **[FALSE POSITIVE]**.
**Threshold:** ≥70% confidence. Style-only complaints excluded.

## Verification Summary

After double-checking every finding by reading the source (5 parallel verifier agents + manual verification of all 9 Critical findings):

| Verdict | Count |
|---|---|
| **CONFIRMED** (bug is real exactly as described) | 66 |
| **PARTIAL** (real but description over/understates) | 9 |
| **FALSE POSITIVE** (bug is not real) | 4 |
| **Net real findings** | **75** |

### Per-finding verdict table

| ID | Verdict | Note |
|---|---|---|
| C-01 | **FALSE POSITIVE** | `_call_model` only called from generation paths (already `@torch.no_grad()`); training uses `compute_loss(prediction, …)` where prediction is computed outside the strategy. |
| C-02 | CONFIRMED | |
| C-03 | CONFIRMED | |
| C-04 | CONFIRMED | No production callers of `GroupedBatchSampler`; only test references. |
| C-05 | CONFIRMED (expanded) | Bug affects BOTH 2D *and* 3D. Verifier initially said 3D was fine — I corrected: in 3D, y_d/y_h/y_w live in different permuted layouts, so the sum aligns wrong positions too. |
| C-06 | CONFIRMED | Atlas-all-tumors-removed branch `continue`s without retry cap. |
| C-07 | CONFIRMED | DDP check runs *after* `wrap.to(device)` and `trainer.model = wrapper`. |
| C-08 | **PARTIAL** | Values are `[0.406, 0.456, 0.485]` vs canonical `[0.4076, 0.4580, 0.4850]`. ~0.002 per channel — self-consistent within a run, so low practical impact. |
| C-09 | CONFIRMED | `os.makedirs('', exist_ok=True)` raises FileNotFoundError. |
| H-01 | CONFIRMED (*verifier was wrong*) | Verifier said "false positive — @torch.no_grad handles it", but `no_grad` does **not** save/restore RNG. `torch.randn()` still consumes the global RNG regardless of gradient tracking. Original finding stands. |
| H-02 | **PARTIAL** | Duplication real across 3 files; currently all agree on `False`. Latent drift risk, not active bug. |
| H-03 | CONFIRMED | `use_compile`: 3 different defaults (False/True/False); `gradient_checkpointing`: 6 defaults across files with semantic overlap. |
| H-04 | **FALSE POSITIVE** | Validation doesn't run under autocast when `use_fp32_loss=False`, so `compute_loss` already returns FP32. |
| H-05 | CONFIRMED | |
| H-06 | **PARTIAL** | Conditioning (omega/mode_id) gap is real; CFG gap may already be guarded. |
| H-07 | **PARTIAL** | Modes missing from dispatch is real, but in-practice callers route through aliasing so `ValueError` not currently triggered. Latent-only. |
| H-08 | CONFIRMED | |
| H-09 | CONFIRMED | |
| H-10 | CONFIRMED | |
| H-11 | CONFIRMED | |
| H-12 | CONFIRMED | |
| H-13 | CONFIRMED | |
| H-14 | **PARTIAL** | Actual count is **545** hardcoded paths (not 980+). Pattern still real. |
| H-15 | CONFIRMED | |
| H-16 | CONFIRMED | |
| H-17 | CONFIRMED | |
| H-18 | CONFIRMED | |
| M-01 | CONFIRMED | |
| M-02 | CONFIRMED | Flag-ordering bug real. |
| M-03 | CONFIRMED | (Duplicate of H-12 / seed S4.) |
| M-04 | CONFIRMED | |
| M-05 | CONFIRMED | |
| M-06 | CONFIRMED | |
| M-07 | CONFIRMED | `compute_voxel_size` docstring itself warns about the reordering. |
| M-08 | CONFIRMED | |
| M-09 | CONFIRMED | |
| M-10 | CONFIRMED | Legacy `multi_diffusion.py` returns tuple; canonical `datasets.py` returns dict. |
| M-11 | CONFIRMED | |
| M-12 | CONFIRMED | |
| M-13 | CONFIRMED | |
| M-14 | **FALSE POSITIVE** | File has no `def test_*` functions and `pyproject.toml` excludes `scripts/*` from pytest coverage paths. |
| M-15 | CONFIRMED (count revised) | Actually 10 occurrences across 4 files, not 22 as claimed. |
| M-16 | CONFIRMED (partial) | `test_metric_logging_regression.py:63` mocks real; `test_trainer_msssim_3d.py` does NOT mock — original claim was half-wrong. |
| M-17 | CONFIRMED | |
| M-18 | CONFIRMED | |
| M-19 | **FALSE POSITIVE** | `seg_conditioned_input` handles atlas via `is_seg_inside_atlas` + tolerance loop (different flow from `bravo` but functionally equivalent). |
| M-20 | **PARTIAL** | `trim_slices` re-read duplication real; other sub-claims (string literals, long boolean) are readability/style, not correctness. |
| M-21 | **PARTIAL** | Concept correct (volume silently defaults) but line 30 reference slightly off — `optional` directive isn't literally there; the silent-default behavior happens via `cfg.volume.get(..., default)`. |
| M-22 | CONFIRMED | |
| M-23 | CONFIRMED | |
| M-24 | CONFIRMED | |
| L-01 | CONFIRMED | |
| L-02 | CONFIRMED | |
| L-03 | CONFIRMED | 8 direct calls to `_edm_coefficients`. |
| L-04 | **PARTIAL** | Substring matching used, but looser than worst-case. |
| L-05 | CONFIRMED (risk lower) | Module-scope fixtures return config *dicts*, not trainer objects. Stateful-fixture risk is smaller than feared. |
| L-06 | CONFIRMED | |
| L-07 | CONFIRMED | |
| L-08 | **PARTIAL** | Sample of 10 exp10_* scripts all show 64G — variance claim may be smaller than stated. Need broader sample to confirm. |
| L-09 | CONFIRMED | 4/4 mega-functions exist at the stated line ranges (within ±22 lines). |
| L-10 | CONFIRMED | 36 type-ignores (claim was "35+"). |
| L-11 | CONFIRMED | |
| Info-01..05 | CONFIRMED | All 5 doc-staleness findings verified against current code. |
| S1..S12 | CONFIRMED | All 12 seed findings re-verified at exact claimed lines. Minor line-count variance on S7 (614 vs 592) and S8 (460 vs 458). |

### False positives (4) — do not fix

- **C-01** — `_call_model` never reached from training. Grep confirmed: trainer calls `strategy.compute_loss(prediction, …)`, never `_call_model`. All `_call_model` call-sites are in generation paths wrapped by `@torch.no_grad()` on `generate()`.
- **H-04** — Validation code path doesn't wrap `strategy.compute_loss()` in `autocast()` when `use_fp32_loss=False`, so the output is already FP32. The BF16 precision concern never materializes.
- **M-14** — `scripts/test_light_sdedit.py` has no `def test_*` functions and `pyproject.toml` excludes the whole `scripts/` directory from pytest collection paths.
- **M-19** — `seg_conditioned_input` mode does handle the atlas-removed-all-tumors case via `is_seg_inside_atlas()` + a tolerance-based validation loop, rather than the `cleaned_seg.sum()==0` check used by `bravo`. Different code, same outcome.

### Corrections (verifier disagreements resolved)

- **H-01** — The H-01..H-09 verifier agent marked this a FALSE POSITIVE with the reasoning "@torch.no_grad handles it". But `@torch.no_grad()` doesn't save/restore RNG state — `torch.randn()` still consumes the global RNG regardless. The original finding (RNG drift from logging cadence breaks reproducibility) is valid. **Re-verdict: CONFIRMED.**
- **C-05** — The Phase-1 models subagent claimed "3D version does NOT have this bug". My manual verification shows 3D *also* has it: y_d/y_h/y_w outputs live in [D,H,W] / [H,D,W] / [W,D,H] layouts respectively, summed without inverse-permute. **Bug affects both 2D and 3D.**
- **H-14** — Claim was 980+ hardcoded paths; actual count is 545. Pattern still real but magnitude downgraded.
- **M-15** — Claim was 22 MockDataset duplicates; actual count is 10 across 4 files. Still a consolidation target.
- **S7, S8, L-09** — Mega-function line counts were off by small amounts (±22, ±2 lines). Consistent with different line-count methods (blank-line handling). Findings remain CONFIRMED.

> **Reading guide.** Each finding has severity, file:line, short description, why it's wrong, suggested direction, confidence. Severities: **Critical** (wrong outputs / training corruption) → **High** (likely-to-break under normal use) → **Medium** (brittle / slow / silent degradation) → **Low** (maintainability) → **Info** (awareness).
> Seed findings (S1-S12) were identified during planning recon; subagent confirmations are referenced inline.

---

## Executive Summary

**Total findings: ~70 above confidence threshold** (not counting the 12 seed findings).

| Severity | Count |
|---|---|
| Critical | 9 |
| High | 18 |
| Medium | 24 |
| Low | 11 |
| Info | 8 |

**Top-5 highest-impact findings** (post-verification, fix first):

1. **`data/loaders/common.py:126`** (C-04) — `GroupedBatchSampler` is defined but **never wired into any loader**. Multi-mode training has been running with mixed-mode batches, directly violating the CLAUDE.md invariant "Mode embedding requires homogeneous batches".
2. **`diffusion/strategy_rflow.py:106`** (C-02) + **`pipeline/diffusion_trainer_base.py:233`** (C-03) — `setup_scheduler` has `use_discrete_timesteps=True` default (opposite of `StrategyConfig`), AND the trainer strategy factory omits `irsde`/`resfusion`/`bridge` entirely. Configuring these strategies passes validation then hits `ValueError` at runtime.
3. **`models/mamba_blocks.py:212-218`** (C-05) — SS2D `_cross_merge` does not inverse-transpose column (2D) or permuted (3D) scans, so directional contributions sum at wrong spatial positions. Mamba trains but is NOT computing the intended multi-direction scan. Affects both 2D and 3D.
4. **Three-way `use_compile` / `gradient_checkpointing` default drift** (H-03) — textbook pattern of the MEMORY.md "duplicated defaults" class. `use_compile` has 3 different defaults across files; `gradient_checkpointing` has 6.
5. **`scripts/generate.py:904-905`** (C-06) — Atlas-removed-all-tumors `continue`s the outer loop without retry cap. Pathological atlas/bin combinations cause infinite loops.

> Note: Original Top-5 included C-01 ("`_call_model` kills gradients"); verification pass proved this a false positive — `_call_model` is only invoked from generation paths, never training. See Verification Summary below.

**Cross-cutting themes:**

- **Restoration / IR-SDE / ResFusion / Bridge** are an under-integrated code island: strategy factory gaps, no tests, no combined config example, undocumented in CLAUDE.md. This is the single largest risk cluster.
- **Mamba** is new and has real implementation bugs (SS2D merge, bottleneck residual) plus **zero** test coverage.
- **Configuration default-drift** is systemic, not local. The `cfg.get(key, default)` pattern has been flagged before but has proliferated — now with cross-file disagreement on defaults.
- **Silent failures in e2e tests** (`pytest.skip()` on training failure) hide the regressions they were written to catch.
- **Mode dispatch gaps** keep appearing (triple missing in gen_metrics_manager, ConditionalSampler; restoration missing from CLAUDE.md). Each time a new mode is added, ~5 downstream paths forget it.

---

## Critical Findings

### C-01 — `_call_model` wraps gradients away unconditionally
**File:** `src/medgen/diffusion/strategies.py:509`
**What:** Base-class `_call_model` has `with torch.no_grad(): return self.model(...)`. It is shared between training and generation paths.
**Why wrong:** Any training path that calls `_call_model` (DiffRS training, restoration, any trainer that uses the helper) silently loses gradients. Loss still computes but no parameters update.
**Fix direction:** Remove `no_grad` from `_call_model`. `generate()` already has its own `@torch.no_grad()` decorator for inference.
**Confidence:** 90%. New.

### C-02 — `setup_scheduler` default flips the timestep mode
**File:** `src/medgen/diffusion/strategy_rflow.py:106`
**What:** `setup_scheduler(..., use_discrete_timesteps: bool = True)`. `StrategyConfig.use_discrete_timesteps` defaults to `False`; CLAUDE.md says RFlow should use continuous by default.
**Why wrong:** Any direct call without the kwarg (confirmed at `scripts/generate.py:289`) silently switches RFlow to discrete timesteps, which quantizes the logit-normal sampler and degrades generation quality.
**Fix direction:** Change the default to `False` to match `StrategyConfig`.
**Confidence:** 92%. Matches the CLAUDE.md "duplicated defaults" anti-pattern.

### C-03 — Strategy factory omits irsde/resfusion/bridge (root cause of seed S1)
**File:** `src/medgen/pipeline/diffusion_trainer_base.py:233`
**What:** `_create_strategy` dict only maps `ddpm`/`rflow`. `StrategyConfig.__post_init__` (base_config.py:189) accepts `irsde`/`resfusion`/`bridge` as valid names. Training with any of these raises `ValueError` at runtime despite passing config validation.
**Why wrong:** End-to-end broken: user configures restoration strategy, hits `ValueError`. Even if the trainer entry was fixed, `scripts/generate.py:288, 531` (seed S1) hardcodes `RFlowStrategy() if cfg.strategy=='rflow' else DDPMStrategy()` — generation can't instantiate the right strategy either.
**Fix direction:** Add all strategies to the dispatch map, guarded by their imports. Replace the `generate.py` ternaries with the same factory.
**Confidence:** 88%. Same root cause reflected in S1, P2.10-01 (CLAUDE.md docs).

### C-04 — `GroupedBatchSampler` is dead code that should be active
**File:** `src/medgen/data/loaders/common.py:126` (defined); `data/loaders/multi_diffusion.py:202`, `builder_2d.py:506`, `datasets.py:MultiDiffusionDataset` (consumers that should use it but don't).
**What:** Class exists but no caller instantiates it. Multi-mode DataLoaders use `shuffle=True` plain samplers. Every training batch for multi-mode is a random mix of modalities.
**Why wrong:** CLAUDE.md invariant: "Mode embedding requires homogeneous batches (GroupedBatchSampler)". Mode embedding is being trained on the wrong distribution of batches right now.
**Fix direction:** Wire `GroupedBatchSampler` into the multi-mode loader paths. Add a test that asserts all items in a batch have the same mode_id.
**Confidence:** 95%. New.

### C-05 — Mamba SS2D 2D cross-merge uses wrong spatial positions for column directions
**File:** `src/medgen/models/mamba_blocks.py:212-218`
**What:** `_cross_scan` transposes input for directions 2 & 3 (columns-first layout `[B, C, W*H]`). `_cross_merge` returns `y2`, `y3` without inverse-transposing. Summing `y0 + y2` aligns token at position `i` (row-layout) with token at `(i % H, i // H)` (col-layout) — wrong mapping.
**Why wrong:** Mamba's 4-directional scan is supposed to aggregate multi-directional context at the same spatial position. Column contributions currently land at scrambled spatial positions. Model still trains (SSM can learn to compensate) but is NOT the architecture from the paper. 3D version (6-directional) is correct.
**Fix direction:** Reshape `y2`/`y3` to `[B, C, W, H]`, transpose to `[B, C, H, W]`, then flatten before summing.
**Confidence:** 92%. New. Mamba has zero test coverage (see T-01).

### C-06 — `run_3d_pipeline` infinite-loop on bad atlas/bin combinations
**File:** `src/medgen/scripts/generate.py:904-905` (bravo), also `seg_conditioned` at 906.
**What:** When atlas cleans all tumors from a generated seg, code does `continue` on the outer `while generated < cfg.num_images:` loop without incrementing or retry-capping.
**Why wrong:** A pathological atlas+size-bin combination produces an infinite loop. The inner seg-retry cap doesn't cover this outer path.
**Fix direction:** Add a retry cap at the outer level, or count failed attempts toward `generated`.
**Confidence:** 88%. New.

### C-07 — DDP compatibility check fires after wrapper is already applied
**File:** `src/medgen/pipeline/diffusion_model_setup.py:184-190`
**What:** The DDP+embedding-wrapper check runs AFTER `.to(device)` (line 160) and `trainer.model = wrapper` assignment. On failure, trainer is left partially initialized.
**Why wrong:** Violates CLAUDE.md "Fail Fast" rule. Partial-init state can cascade into confusing downstream errors.
**Fix direction:** Move the DDP guard to run before `wrap_model_for_training`.
**Confidence:** 85%. New.

### C-08 — RadImageNet normalization uses wrong mean values
**File:** `src/medgen/metrics/feature_extractors.py:177-181`
**What:** Post-BGR-flip, applies mean `[0.406, 0.456, 0.485]`. Canonical RadImageNet BGR mean is `[0.4078, 0.4574, 0.4850]` (= `[103.939, 116.779, 123.68] / 255`).
**Why wrong:** Feature extraction biased by ~0.002 per channel. KID_RIN absolute values are not comparable across runs that used corrected constants. Self-consistent within one run (both real and generated use the same wrong mean) but not externally comparable.
**Fix direction:** Use `[0.4078, 0.4574, 0.4850]` in B,G,R order.
**Confidence:** 88%. New.

### C-09 — `nnUNet evaluate.py` crashes on bare output filename
**File:** `src/medgen/downstream/nnunet/evaluate.py:155`
**What:** `os.makedirs(os.path.dirname(output_path), exist_ok=True)`. If `output_path='results.json'`, `dirname` returns `''`, and `os.makedirs('', exist_ok=True)` raises `FileNotFoundError`.
**Why wrong:** The script's own docstring uses `--output results.json`. Documented usage is broken.
**Fix direction:** `os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)`.
**Confidence:** 90%. New.

---

## High Findings

### H-01 — `_visualize_restoration_samples` shifts global RNG
**File:** `src/medgen/pipeline/trainer.py:1552-1598`, also `_compute_restoration_fwd` at 1600.
**What:** Calls `strategy.generate()` (uses `torch.randn`) without save/restore RNG brackets.
**Why wrong:** CLAUDE.md mandates RNG save/restore around validation/generation code. Training reproducibility is broken by logging cadence (`figure_interval`).
**Fix:** `try: rng=torch.get_rng_state(); ... finally: torch.set_rng_state(rng)`.
**Confidence:** 90%. New. Pattern bug — recurrence of pitfall #42 class.

### H-02 — `use_ema` read via raw `cfg.get` duplicated across multiple files
**File:** `src/medgen/pipeline/diffusion_trainer_base.py:92` + `diffusion_config.py:612` + `compression_trainer.py:144`.
**What:** Same config key, three places, currently all default `False`. MEMORY.md explicitly flags this class of bug as the source of past `use_ema` incidents.
**Why wrong:** Future changes to any one default diverge silently.
**Fix:** Single source of truth via typed config; remove base-class raw `.get()`.
**Confidence:** 85%. Recurrence of documented pitfall pattern.

### H-03 — `use_compile` and `gradient_checkpointing` defaults drift across files
**Files:**
- `use_compile`: `pipeline/segmentation_trainer.py:159=False`, `pipeline/compression_trainer.py:148=True`, `core/validation.py:183=False`.
- `gradient_checkpointing`: `pipeline/compression_trainer.py:126=True`, `pipeline/diffusion_config.py:466=False`, `diffusion_config.py:588=True`, `diffusion_config.py:591=False`, `models/controlnet.py:116=False`, `core/validation.py:184=False`.
**Why wrong:** Three (and in one case six) different defaults for one key. Exact duplication pattern that MEMORY.md flags. Whichever file is consulted determines behavior; changing one "default" changes nothing if a different file is the gate.
**Fix:** Collapse to the typed config (`BaseTrainingConfig`), make the trainer read `self.training_config.gradient_checkpointing` (no `.get()` fallback).
**Confidence:** 90%. Recurrence.

### H-04 — Validation BF16 loss never `.float()`-cast when `use_fp32_loss=False`
**File:** `src/medgen/pipeline/validation.py:185`
**What:** `strategy.compute_loss()` in validation returns BF16. Unlike training, validation doesn't unconditionally `.float()`.
**Why wrong:** Violates CLAUDE.md rule "Always `.float()` before loss computation". Silent BF16 precision bug for anyone setting `use_fp32_loss=False`.
**Fix:** Apply `.float()` to `mse_loss` before accumulating into `total_mse`.
**Confidence:** 82%. New.

### H-05 — ResFusion strategy relies on instance state across add_noise/compute_loss
**File:** `src/medgen/diffusion/strategy_resfusion.py:147-148`
**What:** `add_noise` stores `self._last_epsilon`, `self._last_R`; `compute_loss` reads them.
**Why wrong:** Breaks under gradient accumulation or any flow that calls the two methods at different iteration counts. Also mixes `[0,1]` vs `[-1,1]` spaces between the calls. Comment in the code itself says "fragile but matches official pattern".
**Fix:** Pass `epsilon`/`R` as explicit arguments. Requires trainer-side change but eliminates the race.
**Confidence:** 85%. New.

### H-06 — DiffRS warmup bypasses CFG + conditioning
**File:** `src/medgen/diffusion/diffrs.py:320-328`
**What:** Warmup loop calls `model(x=..., timesteps=...)` raw — no omega, no mode_id, no size_bins.
**Why wrong:** For models trained with ScoreAug omega conditioning, the adaptive thresholds are calibrated on a DIFFERENT distribution than generation uses.
**Fix:** Thread conditioning through; or raise if omega/mode_id are present (fail-fast).
**Confidence:** 80%. New.

### H-07 — `validate_mode_requirements` missing mode entries
**File:** `src/medgen/data/loaders/common.py:527`
**What:** Explicit `else: raise ValueError("Unknown mode: ...")`. Enumeration omits `bravo_seg_cond`, `triple`, `restoration` — all of which `unified.py:295` treats as valid diffusion modes.
**Why wrong:** Any direct call raises. Currently masked because the common callers route through validation-mode aliasing (e.g. `triple` validates as `dual`), but the validation is loose — triple-modality datasets get validated against a 2-key-count rule.
**Fix:** Add all registered modes to the dispatch.
**Confidence:** 88%. New. Same mode-gap pattern as S2, P1.5-04.

### H-08 — `hardcoded absolute path` in `measure_distinguishability.py` is non-overridable
**File:** `src/medgen/scripts/measure_distinguishability.py:19`
**What:** `DATA_DIR = "/home/mode/NTNU/MedicalDataSets/..."` is a module-level constant (not argparse default). Used directly by `load_bravo_volumes()`.
**Why wrong:** Breaks for every other user/environment without code edits.
**Fix:** Convert to argparse argument with this path as default.
**Confidence:** 92%. Expansion of seed S10.

### H-09 — `evaluation.py` fallback hits signature mismatch at runtime on 3D
**File:** `src/medgen/evaluation/evaluation.py:408-417`
**What:** Legacy `_log_worst_batch` fallback calls `worst_batch_figure_fn(worst_batch_data)` with the 2D signature. 3D function `create_worst_batch_figure_3d` has a different signature.
**Why wrong:** TypeError at runtime when 3D evaluator hits fallback path.
**Fix:** Unify signatures or route 3D through `_unified_metrics` only.
**Confidence:** 95%. New.

### H-10 — Per-modality features trigger double extractor load cycles (VRAM spikes)
**File:** `src/medgen/metrics/generation_sampling.py:579-644`
**What:** Phase 2 unloads each of ResNet50 / RadImageNet / BiomedCLIP after primary axial+triplanar pass; then per-modality branch at 626-643 reloads them.
**Why wrong:** 2× extractor load/unload on dual/triple mode. On constrained nodes (40GB A100), may OOM on reload. Otherwise just slow and wasteful.
**Fix:** Move per-modality extraction to happen before the unload in the same extractor's pass.
**Confidence:** 87%. New.

### H-11 — `ConditionalSampler` omits triple-mode branch
**File:** `src/medgen/metrics/sampler.py:379-390`
**What:** `if out_channels == 2:` builds dual noise+mask. No branch for triple. Falls through to single-channel logic.
**Why wrong:** Triple-mode generation via `ConditionalSampler` uses wrong input shape. Mirrors seed S2 and P1.5-04 — "triple mode forgotten" pattern.
**Fix:** Generalize to `out_channels >= 2` with a list-comprehension noise construction.
**Confidence:** 82%. New.

### H-12 — `_get_offset_noise_config` silently masks all errors
**File:** `src/medgen/scripts/generate.py:156-172` (seed S4).
**What:** Bare `except Exception: pass`. Masks corrupted ckpt, OOM, wrong path, disk full.
**Why wrong:** Training-distribution mismatch during generation with no warning.
**Fix:** Log the exception; only swallow specific expected errors (e.g. missing key).
**Confidence:** 88%. Confirmed.

### H-13 — Hardcoded checkpoint paths in debug SLURM scripts
**File:** `IDUN/debug/debug_ldm_roundtrip_exp9_1.slurm:3-4`
**What:** Full timestamp-containing path `exp8_1_256x160_20260107-031153/checkpoint_latest.pt`.
**Why wrong:** Breaks on re-run with new timestamp. Reproducibility broken.
**Fix:** Accept checkpoint via env var or glob.
**Confidence:** 100%. New.

### H-14 — 980+ hardcoded `/cluster/work/modestas/` paths
**Files:** 90+ SLURM scripts across `IDUN/train/**`, `IDUN/eval/**`, `IDUN/debug/**`.
**What:** User-specific cluster base path embedded throughout.
**Why wrong:** Cluster scripts are not reusable by anyone else.
**Fix:** Introduce `${CLUSTER_BASE:=/cluster/work/$USER}` in a shared header / sourced script.
**Confidence:** 100%. New.

### H-15 — `use_omega_conditioning` YAML vs dataclass default drift
**File:** `configs/training/default.yaml:68` vs `pipeline/diffusion_config.py:90`.
**What:** YAML value `true`, dataclass default `False`.
**Why wrong:** Any code path that instantiates the typed config without loading this YAML gets a different value than code that uses Hydra. Classic drift.
**Fix:** Decide the right default; make both files agree.
**Confidence:** 85%. New.

### H-16 — `_train_3d` re-reads `slicewise_encoding` twice from raw config
**File:** `src/medgen/scripts/train.py:518, 564`
**What:** Same key resolved locally twice.
**Why wrong:** If DictConfig interpolation or mutation changes the value between accesses, log + behavior diverge silently. MEMORY.md explicitly flags this anti-pattern.
**Fix:** Resolve once at the top; use the local variable everywhere.
**Confidence:** 82%. New.

### H-17 — `find_optimal_steps` shape mismatch for morphological metric after trim
**File:** `src/medgen/scripts/find_optimal_steps.py:563-568, 700-720`
**What:** Reference masks padded to `pixel_depth`; generated volumes trimmed by `args.trim_slices`. Morphological path uses both at different depths.
**Why wrong:** Shape mismatch when `trim_slices > 0`. Same class of bug as the 2D morphological fix noted in MEMORY.md (pitfall #9 area).
**Fix:** Apply trim to reference masks, or pad generated after trim (match what the FID path does).
**Confidence:** 85%. New.

### H-18 — e2e test suite silently skips on training failure
**File:** `tests/e2e/test_training_pipeline.py:136,168,200,205,356`; same pattern in `test_generation_pipeline.py:170,206,242,296,353`; `test_output_validation.py:95,102,114`.
**What:** `if result.returncode != 0: pytest.skip("Training didn't complete")`. Converts any crash — including the regression the test was written to catch — into silent skip.
**Why wrong:** The test passes CI when it shouldn't.
**Fix:** `pytest.fail()` on non-zero return, OR split into fixture-dependent tests that hard-skip on missing fixtures only.
**Confidence:** 95%. New.

---

## Medium Findings

### M-01 — 7+7+4 trainer-level `_visualize_*` / `_log_*` methods
**Files:** `pipeline/trainer.py`, `compression_trainer.py`, `base_trainer.py`. Confirmed at `trainer.py:1597, 1682-1683` new restoration-mode instances.
**What:** Methods that should live in `metrics/unified.py` per CLAUDE.md.
**Fix:** Incremental migration: move restoration-mode figures first (they are newest), older ones next.
**Confidence:** 95%. Seed S5 confirmed.

### M-02 — Optimizer offload/reload silently fails
**File:** `src/medgen/pipeline/validation.py:494-495, 555-556` (seed S3).
**What:** Bare `except Exception: pass`. Also `optimizer_offloaded=True` is set before the move succeeds → finally block tries to restore state that was never moved.
**Fix:** Log; set the flag only after successful move; restore inside its own `try/except`.
**Confidence:** 88%. Confirmed.

### M-03 — `generate.py:170-172` bare except masks config-parse failures
**File:** seed S4, also covered by H-12.
**Fix:** Narrow exception type; log the failure.
**Confidence:** 88%. Confirmed.

### M-04 — Mamba bottleneck has undocumented double residual
**File:** `src/medgen/models/mamba_diff.py:531`
**What:** `x = x + self.bottleneck(x, c)` adds an OUTER residual while `MambaDiffBlock` already contains internal residuals.
**Fix:** Remove the outer residual OR document intent. Encoder/decoder stages are symmetric — bottleneck should be too.
**Confidence:** 82%. New.

### M-05 — Mode-embedding wrappers' `parameters()` is brittle
**File:** `src/medgen/models/wrappers/mode_embed.py:230-232, 319-321, 367-369, 747-749`
**What:** Four wrappers override `parameters()` to delegate solely to `self.model.parameters()`. Works only because each wrapper splices its MLP into `self.model.time_embed`. Any refactor that moves the MLP out silently drops it from the optimizer.
**Fix:** Follow `LateModeModelWrapper.parameters()` pattern (iterate both).
**Confidence:** 95%. New.

### M-06 — Mamba `_init_weights` has fragile double-init of `final_linear`
**File:** `src/medgen/models/mamba_diff.py:445-470`
**What:** Constructor zeros → `self.apply(_init)` xavier-overwrites → explicit re-zero at bottom. Works, but any subclass skipping the re-zero breaks AdaLN-Zero.
**Fix:** Guard `final_linear` from `self.apply(_init)`, or keep only the explicit zero at the end.
**Confidence:** 80%. New.

### M-07 — `voxel_spacing` passed as `(x,y,z)` where `(D,H,W)` expected
**File:** `src/medgen/scripts/generate.py:649, 806, 1023`
**What:** `compute_voxel_size()` returns `(x,y,z)`; `compute_feret_diameter_3d`/`compute_size_bins_3d` expect `(D,H,W)`. In-plane isotropic by coincidence, so it works today.
**Fix:** Swap order explicitly before passing.
**Confidence:** 85%. New.

### M-08 — `volume_3d.py:367` uses inline `> 0.5` instead of `binarize_seg()`
**File:** `src/medgen/data/loaders/volume_3d.py:367` (3 classes affected).
**Why wrong:** MEMORY.md set `binarize_seg()` as SSOT; this path skips the clamp step.
**Fix:** Import + call `binarize_seg`.
**Confidence:** 83%. Recurrence.

### M-09 — `downstream/data.py:82-84` re-implements `_binarize_seg` without clamp
**File:** `src/medgen/downstream/data.py:82-84`
**Why wrong:** Same SSOT violation as M-08.
**Fix:** Import and use `binarize_seg` from `medgen.data`.
**Confidence:** 82%. Recurrence.

### M-10 — Duplicate `MultiDiffusionDataset` with diverged return types
**Files:** `src/medgen/data/loaders/multi_diffusion.py:31, 120-125` and `datasets.py:331, 573-579`.
**What:** Identical helper `extract_slices_with_seg_and_mode` duplicated. `MultiDiffusionDataset` exists twice with different returns (tuple vs dict).
**Fix:** Delete the legacy `multi_diffusion.py` module; route all callers through `datasets.py`.
**Confidence:** 82%. New.

### M-11 — `DiffRS` batch-data heuristic breaks above 10 bins
**File:** `src/medgen/diffusion/batch_data.py:85-88, 101`
**What:** `from_raw` 2-tuple/3-tuple path uses `shape[1] <= 10` to distinguish labels from size_bins. Current default is 7 bins — safe today.
**Fix:** Document the limit, or raise on ambiguous shapes; prefer dict-format exclusively.
**Confidence:** 80%. New.

### M-12 — `nnunet splits.py:310` still invokes deprecated `install_splits`
**File:** `src/medgen/downstream/nnunet/splits.py:310`
**What:** Main CLI still calls deprecated function. The module docstring gives usage steering users into the race-prone path.
**Fix:** Update main() to `create_isolated_preprocessed_dir()`; delete docstring example or update it.
**Confidence:** 80%. New.

### M-13 — Print statements in `downstream/nnunet/evaluate.py:178-190`
**File:** seed S12.
**Why wrong:** Library code. `print()` bypasses logger config and breaks stdout-redirect blocks in same file (lines 147-159).
**Fix:** Replace with `logger.info`.
**Confidence:** 90%. Confirmed.

### M-14 — Script named `test_light_sdedit.py` collides with pytest
**File:** `src/medgen/scripts/test_light_sdedit.py`
**What:** Analysis script with `test_` prefix causes pytest collection attempts.
**Fix:** Rename to `eval_light_sdedit.py`.
**Confidence:** 83%. New.

### M-15 — 22 duplicated `MockDataset` class definitions in tests
**Files:** `tests/conftest.py:186,206,222,239` + many `test_*.py` files.
**Fix:** Consolidate into conftest fixtures; add missing variants.
**Confidence:** 95%. New.

### M-16 — Integration tests mock `SummaryWriter`
**File:** `tests/integration/test_trainer_msssim_3d.py:15`, `test_metric_logging_regression.py:63-597`.
**Why wrong:** Tests the logging intent not the outcome. Tag rename won't break code but breaks tests. Contradicts CLAUDE.md "no mocks for integration".
**Fix:** Use real `SummaryWriter(log_dir=tmp_path)` and parse event files.
**Confidence:** 90%. New.

### M-17 — Integration tests mock strategies/models
**File:** `tests/integration/test_regression_bugs.py:359-385`
**What:** `Mock()` strategy, model, mode; asserts on captured kwargs.
**Fix:** Use a tiny real UNet + real RFlowStrategy; assert on produced tensor.
**Confidence:** 90%. New.

### M-18 — WSL2 skipif blanks 3D coverage on user's dev box
**File:** `tests/integration/test_trainer_equivalence.py:134,156,202`
**Why wrong:** CLAUDE.md notes dev env is WSL2. All 3D trainer-equivalence tests disabled locally.
**Fix:** Investigate determinism issue; use `torch.use_deterministic_algorithms(True, warn_only=True)`.
**Confidence:** 70%. New.

### M-19 — `seg_conditioned_input` missing atlas-removed-tumors guard
**File:** `src/medgen/scripts/generate.py:1013+` branch.
**What:** `seg_conditioned` and `bravo` branches handle the "atlas removed all tumors" case; `seg_conditioned_input` does not. It will save seg masks with tumors outside the atlas.
**Fix:** Mirror the guard from the other two branches (and factor out the shared post-processing).
**Confidence:** 85%. New.

### M-20 — Many `cfg.get(key, default)` re-reads within same function
**Files:** `generate.py:989, 1100` (`trim_slices`); `train.py:870` (`use_latent`); `train.py:599-602` (string-literal mode names instead of `ModeType`); `setup_model` 220-char boolean at `diffusion_model_setup.py:354`.
**Why wrong:** Duplication + cognitive load. Some are drift-risk (config re-reads), some are type-safety risk (string literals for enum values).
**Fix:** Resolve once at top of function; prefer enum comparisons.
**Confidence:** 80% average across these. Recurrences of MEMORY.md pattern.

### M-21 — `configs/diffusion_3d.yaml:30` uses `optional volume: default`
**File:** `configs/diffusion_3d.yaml:30`
**Why wrong:** If volume config missing, `DiffusionTrainerConfig.from_hydra()` silently defaults to 256×160×256. No error, no log.
**Fix:** Make volume config required; or add a logged warning on default fallback.
**Confidence:** 80%. New.

### M-22 — `train_template.slurm` references wrong conda env
**File:** `IDUN/train_template.slurm`
**What:** `conda activate AIS4005` while every actual script uses `AIS4900`.
**Fix:** Standardize to `AIS4900`; add an early env-exists check.
**Confidence:** 90%. New.

### M-23 — `n_epochs` resolution falls through `or` chain on 0
**File:** `src/medgen/pipeline/diffusion_config.py:593`, `base_config.py:333`
**What:** `cfg.training.get('epochs', None) or cfg.training.get('max_epochs', None)`. Treats `0` as falsy.
**Fix:** Explicit `if None` check.
**Confidence:** 80%. New.

### M-24 — `find_optimal_steps.main` can lose partial results on exception
**File:** `src/medgen/scripts/find_optimal_steps.py:700-701`
**What:** `entry` dict accumulates PCA → morph → FID. `_save_history` is called only after entry is fully populated; a mid-stage exception drops already-computed metrics.
**Fix:** Append `entry` to `history` at each stage, or wrap each stage in its own try/except with partial-save.
**Confidence:** 80%. New.

---

## Low / Info Findings

### L-01 — Tests: no coverage for Mamba architecture (1189 lines)
**File:** `src/medgen/models/mamba_diff.py`, `mamba_blocks.py`
**What:** Zero tests. Single highest-risk untested code area given C-05 and M-04.
**Fix:** Unit tests mirroring `test_dit_blocks.py` pattern — shape contracts, determinism, gradient flow.
**Severity:** High impact but Low "bug count" — logged as Low because it enables other findings.

### L-02 — Tests: no coverage for `strategy_irsde.py`, `strategy_resfusion.py`
Similar to L-01. Combined with C-03, these strategies have no way for users to know if they work.

### L-03 — Tests couple to private `_edm_coefficients` / `_call_model`
**File:** `tests/unit/test_exp1e_exp1f.py`
**Fix:** Test public behavior; EDM math invariants at boundary timesteps.
**Confidence:** 85%.

### L-04 — Tests assert on log message substrings
**File:** `tests/integration/test_logging_integration.py:281-310`, `tests/unit/test_batch_format.py:54-66`.
**Fix:** Match loosely on keywords, not format.

### L-05 — `tests/integration/test_unified_compression_trainers.py` uses `scope="module"` fixtures with stateful trainers
**Fix:** Narrow to function-scope or use factory fixtures.

### L-06 — Magic constants `SHIFT=0.2033`, `SCALE=0.0832` undocumented
**File:** `src/medgen/scripts/measure_distinguishability.py:20-21` (seed S10).
**Fix:** Add `# From compute_pixel_stats.py over training set` comment.

### L-07 — DC-AE 3D configs remain despite abandonment
**Files:** `configs/dcae_3d/*`, `configs/dcae/f32.yaml`.
**Fix:** Rename to `.deprecated` or add bold warning in YAML.

### L-08 — Inconsistent SLURM memory requests
**Files:** `IDUN/train/diffusion/exp*.slurm`
**What:** 8× ratio between seemingly similar workloads.
**Fix:** Add memory estimation comment to `train_template.slurm`.

### L-09 — Trivial mega-function-splits needed
**Files:** `generate.py:516`, `train.py:459`, `diffusion_model_setup.py:35`, `find_optimal_steps.py:287` (seeds S6-S9).
**Fix:** Decompositions proposed in Phase 2.8 report — factor into 4-5 named helpers each.

### L-10 — `diffusion/spaces.py` has 35+ `# type: ignore`
Seed S11. Mostly legitimate Python-typing limitations but worth a dedicated type-cleanup sprint.

### L-11 — `nnunet/evaluate.py` `print()` elsewhere (splits.py:311)
Seed S12 expansion. Library code should use logger.

### Info-01 — CLAUDE.md strategy list out of date
**File:** `CLAUDE.md:33`.
**What:** Says `ddpm, rflow`. Code has 5: `ddpm, rflow, bridge, irsde, resfusion`.
**Fix:** Update table.

### Info-02 — CLAUDE.md mode list missing `restoration`
**File:** `CLAUDE.md:32`.
**Fix:** Add `restoration`, clarify `multi_modality` vs `multi`.

### Info-03 — CLAUDE.md quick-commands missing Mamba
**File:** `CLAUDE.md` Quick Commands section.
**Fix:** Add `model=mamba`/`model=mamba_3d` examples.

### Info-04 — CLAUDE.md pitfall count phrasing
**File:** `CLAUDE.md:122`.
**What:** "88 known issues (numbered 1-88, #43 skipped)". True entry count is 87.
**Fix:** Clarify phrasing.

### Info-05 — `docs/commands.md:84-85` DC-AE commands still documented
Clarify deprecation status for new users.

### Info-06 — No `_pad_depth` consolidation across `downstream/data.py` / `data/loaders/volume_3d.py`
Serves different class hierarchies so not strictly a DRY violation — logged for awareness.

### Info-07 — No path traversal / unatomic writes beyond what's already atomic
Phase 3 sweep confirmed `pipeline/utils.py` has correct atomic-write pattern with `fsync` + rename.

### Info-08 — `workaround` comments in code are documentation, not red flags
Phase 3 sweep found a few (`pipeline/profiling.py`, `core/model_utils.py`); all are documented upstream-tool workarounds, not covert bugs.

---

## Appendices

### A. Findings by subsystem

| Subsystem | Critical | High | Medium | Low/Info | Total |
|---|---|---|---|---|---|
| diffusion/strategies | 2 | 3 | 1 | 0 | 6 |
| pipeline | 1 | 4 | 5 | 0 | 10 |
| models | 1 | 0 | 3 | 2 | 6 |
| data/loaders | 1 | 1 | 4 | 0 | 6 |
| metrics/evaluation | 1 | 2 | 0 | 0 | 3 |
| scripts | 2 | 3 | 5 | 2 | 12 |
| downstream | 1 | 0 | 2 | 0 | 3 |
| tests | 0 | 1 | 4 | 2 | 7 |
| configs/SLURM | 0 | 2 | 3 | 2 | 7 |
| docs | 0 | 0 | 0 | 5 | 5 |
| cross-cutting defaults | 0 | 2 | 1 | 0 | 3 |
| **Total** | **9** | **18** | **24** | **13** | **64** |

### B. De-dup summary vs `docs/common-pitfalls.md` (87 existing)

- **Recurrences of known pitfall patterns:** 4
  - `use_ema` duplication (H-02): pattern noted in MEMORY.md.
  - `use_compile` / `gradient_checkpointing` drift (H-03): same class as known pitfall.
  - `binarize_seg` inline at `volume_3d.py:367` (M-08), `downstream/data.py:82` (M-09): consolidation MEMORY.md claimed complete, but missed these paths.
  - RNG save/restore missing at `trainer.py:1552-1598` (H-01): same class as pitfalls #42/#88.
- **New bug patterns** (would merit new pitfall entries):
  - Strategy-factory registration gap (C-03)
  - `GroupedBatchSampler` defined-but-unused (C-04)
  - SS2D scan merge (C-05) — Mamba-specific
  - Mode dispatch repeatedly forgets `triple` / `restoration` (S2, H-07, H-11)
  - `no_grad` in `_call_model` base (C-01)

### C. Pending fixes referenced in MEMORY.md

MEMORY.md references `/home/mode/.claude/plans/codebase-review-fixes.md` with "Part 1: 15 bug fixes + dead code" and "Part 2: Design improvements". That plan file **does not exist** on disk. Any overlap cannot be verified — treat as stale memory; this report can replace that plan wholesale.

### D. Suggested fix order (90-day plan)

1. **Week 1 — safety:** C-01 (`no_grad`), C-02 (`setup_scheduler`), C-03 (strategy factory) — these three silently corrupt training. Fix together.
2. **Week 2 — correctness:** C-04 (`GroupedBatchSampler`), C-05 (SS2D merge), C-06 (infinite loop), C-07 (DDP guard order), C-08 (RadImageNet mean), C-09 (`evaluate.py` makedirs).
3. **Weeks 3-4 — defaults sprint:** H-02, H-03, H-15 all in one patch; migrate remaining `cfg.get()` drift-candidates to typed config.
4. **Weeks 5-6 — restoration integration:** finish IR-SDE/ResFusion/Bridge wiring (C-03), add tests (L-01, L-02), documentation (Info-01/02).
5. **Weeks 7-8 — Mamba hardening:** fix M-04 (double residual), M-06 (init), add Mamba tests.
6. **Weeks 9+ — cleanup:** M-10 (duplicate datasets), M-15 (mock dataset consolidation), M-16/17 (real SummaryWriter), M-19 (seg_conditioned_input atlas guard), Info-01..05 (doc sync), L-08/L-09 (SLURM/megafunction splits), L-10 (typing).

### E. Files touched most by findings (top 10)

1. `src/medgen/scripts/generate.py` — 7 findings (C-06, H-12, M-03, M-07, M-20-trim, M-19, seed S1/S4/S6)
2. `src/medgen/pipeline/trainer.py` — 5 findings (H-01, M-01, seed S5)
3. `src/medgen/data/loaders/common.py` + volume_3d.py + multi_diffusion.py — 5 (C-04, H-07, M-08, M-10)
4. `src/medgen/pipeline/diffusion_config.py` + `base_config.py` — 4 (H-02, H-03, M-23, H-15)
5. `src/medgen/diffusion/strategy_rflow.py` + `strategies.py` — 3 (C-01, C-02, H-04-validation)
6. `src/medgen/pipeline/diffusion_model_setup.py` — 3 (C-07, M-20, L-09)
7. `src/medgen/models/mamba_diff.py` + `mamba_blocks.py` — 3 (C-05, M-04, M-06)
8. `CLAUDE.md` — 4 (Info-01..04)
9. `IDUN/**/*.slurm` (90+ files) — 3 (H-13, H-14, M-22)
10. `tests/` — 7 (L-01..05, H-18, M-15..18)

---

**Generated by:** Full-codebase review, 10-agent parallel sweep + cross-cutting pass.
**No code changes made.** User to triage and dispatch fixes.
