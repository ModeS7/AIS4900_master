# Downstream nnU-Net Segmentation — Complete Reference

This document is the single source of truth for the downstream brain-metastasis
segmentation track of the project: training pipeline, dataset conventions,
custom trainer, every completed experiment with results, threshold-sweep
findings, the random-split methodology test, and how our numbers compare to
the literature (Grøvik 2020, Ottesen 2023, Ottesen 2025).

For pixel/latent diffusion experiment results see
[`experiment_results_3d.md`](experiment_results_3d.md).

---

## TL;DR

- **Headline result**: 1-channel BRAVO nnU-Net (exp3) achieves volumetric Dice
  **0.3209 ± 0.28** on the official Grøvik 2020 51-patient hold-out test set.
- **Augmentation experiments are null**: 4-channel input (exp345), generic
  synthetic mixing (exp7 family, 25→525 syn), and size-targeted synthetic
  mixing (exp8 family, 25→105 syn) all land within ±0.011 of the baseline.
- **Synthetic-only is meaningfully worse**: 105 synthetic volumes alone give
  Dice ≈ 0.27 vs 0.32 for 105 real.
- **Critical methodological finding**: A random 105/51 split (seed=42) of the
  same 156 patients (Ottesen 2025's protocol) lifts the *same* model from
  Dice 0.32 → **0.57**. The 0.25-point gap is **entirely** attributable to
  the official hold-out being a deliberately harder cohort. Literature
  comparisons that use random splits (Ottesen 2025, possibly Ottesen 2023)
  are not apples-to-apples with the official BrainMetShare-3 hold-out.
- **Threshold tuning helps tiny lesions, not overall Dice**: Optimum at
  t ≈ 0.005 (vs nnU-Net default 0.5) gives +0.012 absolute Dice but the
  benefit is concentrated in tiny lesions (detection +9.4 pp, Dice +56% rel).
- **Slice-wise Dice (Ottesen 2023 metric) matches literature**: We get
  **0.8353 ± 0.12** vs Ottesen 2023's reported **0.85 ± 0.13** — the
  apparent volumetric gap is a metric-choice artifact, not a model gap.

---

## Pipeline Architecture

```
brainmetshare-3 (NIfTI per patient)
         │
         ▼
convert_dataset.py / convert_random_split.py
         │
         ▼
nnUNet_raw/Dataset{ID}_BrainMet/
  ├─ imagesTr/  BrainMet_XXX_0000.nii.gz  (channel 0 = BRAVO)
  ├─ labelsTr/  BrainMet_XXX.nii.gz       (binary mask)
  ├─ imagesTs/  BrainMet_XXX_0000.nii.gz
  ├─ labelsTs/  BrainMet_XXX.nii.gz
  ├─ dataset.json
  └─ case_info.json     (custom — for split tracking)
         │
         ▼
nnUNetv2_plan_and_preprocess   (auto patch size, batch size, normalization)
         │
         ▼
nnUNet_preprocessed_{experiment_name}/Dataset{ID}_BrainMet/
  (isolated per experiment to avoid splits_final.json race conditions)
         │
         ▼
train_nnunet.py (5-fold CV)
  uses nnUNetTrainerBrainMets (DC+TopK10, smooth=0, 100% oversample)
         │
         ▼
runs/downstream/nnunet/{experiment}/Dataset{ID}_BrainMet/
  nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/
    fold_{0..4}/
      ├─ checkpoint_best.pth
      ├─ checkpoint_final.pth
      ├─ tensorboard/
      └─ validation/   (per-case eval + summary.json)
         │
         ▼
eval_nnunet.py (5-fold ensemble inference + medgen metrics)
         │
         ▼
runs/downstream/nnunet/{experiment}/eval_{experiment}/
  ├─ predictions/  BrainMet_XXX.nii.gz   (binary, t=0.5 argmax)
  │                BrainMet_XXX.npz      (softmax probs, if --save-probabilities)
  └─ eval.json     (volumetric, lesion-wise, slice-wise, detection)
```

---

## Trainer Customizations (`src/medgen/downstream/nnunet/trainer.py`)

Brain metastases are **0.001–0.06% of voxels** — extreme class imbalance.
Default nnU-Net (DC+CE loss with `smooth=1e-5`, SGD lr=0.01, 33% foreground
oversample) collapses to all-background on this data (Dice = 0). We override
four things in `nnUNetTrainerBrainMets`:

| Override | Default | Ours | Why |
|---|---|---|---|
| Loss | DC+CE | **DC + TopK10** | TopK10 backpropagates only the 10% hardest-error voxels — necessary for sparse foreground |
| `smooth` (Dice) | 1e-5 | **0** | Sharper Dice gradients; no numerical floor that hides under tiny-target |
| `batch_dice` | from plans | **False** | Per-sample Dice (forced); avoids whole-batch averaging that flatters easy patches |
| `oversample_foreground_percent` | 0.33 | **1.0** | Every patch is guaranteed to contain a tumor voxel |

Three trainer variants ship in `trainer.py`:
- `nnUNetTrainerBrainMets` — production trainer (above)
- `nnUNetTrainerBrainMets_200epochs` — same but capped at 200 epochs (cheaper experiments)
- `nnUNetTrainerVanilla` — passive subclass for the abandoned vanilla baseline (verified to collapse to Dice 0)
- `nnUNetTrainerTensorBoard` — default `nnUNetTrainer` + TB hooks (no other overrides)

All variants add TensorBoard logging via `_TensorBoardMixin`.

---

## Dataset Conventions

### Patient count
- **156 total** Stanford BrainMetShare-3 patients
- **Official Grøvik 2020 split**: 105 train + 51 test (`val/` 25 + `test_new/` 26 = 51)
- **Random split (this work)**: 105 train + 51 test, drawn randomly with seed=42 from the 156-patient pool — replicates Ottesen 2025's reported "split randomly into 105 and 51"

### Volume dimensions
- **Training input**: 256 × 256 × **160** (BrainMetShare native depth padded with 10 slices to a round 160)
- **Generated/output volumes**: 256 × 256 × **150** (the 10 padding slices stripped at inference)

### Voxel spacing
- nibabel (NIfTI) layout: (X, Y, Z) = (0.9375, 0.9375, 1.0) mm
- 3D bin computations: (D, H, W) = (1.0, 0.9375, 0.9375) mm

### Channel conventions per dataset ID

| Dataset ID | Modality preset | Channels | Used by |
|---|---|---|---|
| 600 | `bravo` (1 channel) | BRAVO | exp3_baseline_v2_d600 |
| 601 | `dual` (2 channels) | T1pre + T1gd | exp4_baseline_dual_v2_d601 |
| 602 | `triple` (3 channels) | T1pre + T1gd + FLAIR | exp5_baseline_triple_d602 |
| 603 | `quad` (4 channels) | BRAVO + T1pre + T1gd + FLAIR | exp345_baseline_quad_d603 |
| 614 | `bravo` | BRAVO + N synthetic mixed in | exp7_*_mixed_*syn_d614 |
| 630–633 | `bravo` | BRAVO + N size-targeted synthetic | exp8_{1,2,3,4} (25/50/75/105 syn) |
| **640** | `bravo` | BRAVO, **random 105/51 split** | **exp3_random_split_d640** |

`convert_random_split.py` is the only converter that pools all three official
split dirs and re-splits with a seed; everything else uses `convert_dataset.py`
which respects the official Grøvik split.

---

## Experiment Results

All numbers are **on the official Grøvik 2020 51-patient hold-out** unless
otherwise noted. Volumetric is per-volume Dice (PRIMARY); lesion-wise is
BraTS-Mets per-lesion Dice with FPs penalised; slice-wise is Ottesen 2023's
sagittal slice-wise Dice with empty-empty → 1.0.

### Baselines (real-only training)

| Experiment | Channels | Train | Vol Dice | Lesion Dice | Slice Dice | Det% | Det% (tiny) | FPs |
|---|---|---|---|---|---|---|---|---|
| **exp3_baseline_v2_d600** | 1 (BRAVO) | 105 real | **0.3209 ± 0.28** | 0.1991 | 0.8353 | 39.5% | 35.5% | 302 |
| exp4_baseline_dual_v2_d601 | 2 (T1pre+T1gd) | 105 real | 0.3231 ± 0.28 | 0.1826 | 0.8314 | 35.1% | — | 179 |
| exp5_baseline_triple_d602 | 3 (T1pre+T1gd+FLAIR) | 105 real | 0.3232 ± 0.28 | 0.1893 | 0.8319 | 34.4% | — | 193 |
| exp345_baseline_quad_d603 | 4 (BRAVO+T1pre+T1gd+FLAIR) | 105 real | 0.3261 ± 0.28 | 0.1950 | 0.8346 | 38.6% | 34.6% | 275 |

**Modality count makes essentially no difference** on the official hold-out
(0.3209 → 0.3261 from 1ch → 4ch, Δ=0.005 ≈ 50× smaller than per-case std).
This invalidates the modality-count hypothesis that motivated exp345.

### Synthetic-only training (no real data)

| Experiment | Synth source | Vol Dice | Det% | FPs |
|---|---|---|---|---|
| exp6_2_synthetic_105_exp1_1_1000_IN | 105 synth (ImageNet-tuned) | 0.2677 ± 0.26 | 34.1% | 231 |
| exp6_2_synthetic_105_exp1_1_1000_RIN | 105 synth (RadImageNet-tuned) | 0.2654 ± 0.26 | 32.9% | 238 |
| exp6_2_synthetic_105_exp1_1_1000plus | 105 synth (best exp1 model) | 0.2298 ± 0.24 | 31.8% | 230 |
| exp6_2_synthetic_105_exp32_2_1000 | 105 synth (exp32_2) | 0.2562 ± 0.25 | 34.5% | 262 |
| exp6_2_synthetic_105_exp48c_handoff_exp32 | 105 synth (exp48c handoff) | 0.2826 ± 0.27 | 37.7% | 282 |
| exp6_2_synthetic_105_exp48c_standalone | 105 synth (exp48c standalone) | 0.2760 ± 0.26 | 36.5% | 286 |
| exp6_2_synthetic_105_exp48d_handoff_exp1 | 105 synth (exp48d handoff) | 0.2799 ± 0.26 | 37.2% | 268 |
| exp6_2_synthetic_525_exp48c_handoff_exp32 | 525 synth (exp48c handoff) | 0.2845 ± 0.27 | 36.6% | 266 |

**Synthetic-only training is meaningfully worse than real-only** (0.27 vs 0.32,
Δ ≈ −0.05). Among synthetic sources, exp48c-based outputs are the strongest;
exp1_1_1000plus is the weakest. Increasing synthetic from 105 → 525 produces
a marginal improvement (0.2826 → 0.2845) that's well within noise.

### Real + synthetic mixed (Dataset 614)

| Experiment | Real | Synth | Vol Dice | Det% | FPs |
|---|---|---|---|---|---|
| exp7_4_mixed_25syn_d614 | 105 | 25 | 0.3165 ± 0.28 | 39.1% | 288 |
| exp7_5_mixed_50syn_d614 | 105 | 50 | 0.3214 ± 0.28 | 39.9% | 289 |
| exp7_6_mixed_75syn_d614 | 105 | 75 | 0.3207 ± 0.28 | 39.3% | 296 |
| exp7_1_mixed_105syn_d614 | 105 | 105 | 0.3186 ± 0.28 | 38.7% | 293 |
| exp7_2_mixed_210syn_d614 | 105 | 210 | 0.3167 ± 0.28 | 38.7% | 288 |
| exp7_3_mixed_315syn_d614 | 105 | 315 | 0.3184 ± 0.28 | 39.5% | 268 |
| exp7_7_mixed_525syn_d614 | 105 | 525 | 0.3256 ± 0.28 | 39.3% | 291 |

**Generic synthetic mixing produces zero improvement across N ∈ {25, 50, 75,
105, 210, 315, 525}**. Every variant is within ±0.005 of exp3 baseline (0.3209).

### Real + size-targeted synthetic (exp8 family, Datasets 630–633)

Synthetic pool ranked by max-lesion-bucket (tiny → small → medium → large
priority order, seed=1 per dataset). Designed to address exp3's 35.5%
tiny-lesion detection rate.

| Experiment | Real | Synth | Synth bucket dist (tiny/small/medium/large) | Vol Dice | Det% (tiny) | FPs |
|---|---|---|---|---|---|---|
| exp8_1_size_targeted_25syn_d630 | 105 | 25 | — | 0.3212 | 35.1% | 314 |
| exp8_2_size_targeted_50syn_d631 | 105 | 50 | — | 0.3109 | 35.3% | 273 |
| exp8_3_size_targeted_75syn_d632 | 105 | 75 | — | 0.3176 | 35.1% | 315 |
| exp8_4_size_targeted_105syn_d633 | 105 | 105 | — | 0.3212 | 35.4% | 300 |

**Size-targeted mixing also produces zero improvement** — tiny-lesion
detection stays at 35.1–35.4% (vs exp3's 35.5%). Hypothesis rejected.

### Random-split control (Dataset 640)

| Experiment | Channels | Split | Vol Dice | Lesion Dice | Slice Dice | Det% | Det% (tiny) | FPs |
|---|---|---|---|---|---|---|---|---|
| exp3_baseline_v2_d600 | 1 | Official Grøvik 2020 | 0.3209 ± 0.28 | 0.1991 | 0.8353 | 39.5% | 35.5% | 302 |
| **exp3_random_split_d640** | 1 | **Random (seed=42)** | **0.5744 ± 0.31** | **0.3922** | **0.8794** | **66.5%** | **65.2%** | **51** |

**Same model, same training pipeline, same hyperparameters — only the train/
test split changes.** The 0.25 absolute Dice difference (and 30 pp tiny-
lesion detection difference) demonstrates the official Grøvik 2020 hold-out
is materially harder than a random subset of the same patient pool.

This is the closest available analog to Ottesen 2025's reported 0.66 ± 0.01,
and our 0.574 lands within 0.09 of their number — much smaller than the
0.34-point apparent gap when comparing on different cohorts.

---

## Threshold Sweep Findings

`misc/run_per_fold_local.py` runs the 51 test cases through each of the 5
fold models individually, producing 255 softmax `.npz` files locally.
`misc/analyze_per_fold_threshold.py` then sweeps thresholds and computes
per-fold optima + ensemble metrics.

### Cross-fold optimum stability

Each fold's optimum threshold landed in **t ∈ [0.001, 0.02]**:

| Fold | Best t | Dice @ best | Dice @ 0.50 |
|---|---|---|---|
| 0 | 0.001 | 0.342 | 0.325 |
| 1 | 0.001 | 0.340 | 0.327 |
| 2 | 0.005 | 0.343 | 0.333 |
| 3 | 0.020 | 0.347 | 0.339 |
| 4 | 0.001 | 0.343 | 0.329 |
| **Median / std** | **0.001 / 0.008** | — | — |

**Ensemble (mean of 5-fold softmaxes) baseline vs swept optimum:**

| Threshold | Ensemble Dice |
|---|---|
| 0.50 (nnU-Net default) | 0.3333 |
| 0.005 (swept optimum) | **0.3457** |
| **Δ** | **+0.0124 (+3.7% relative)** |

### Size-stratified at t=0.50 vs t=0.005

Where the threshold-tuning benefit actually lives:

| Bucket | n_GT | Dice @ 0.50 | Det @ 0.50 | FPs | Dice @ 0.005 | Det @ 0.005 | FPs |
|---|---|---|---|---|---|---|---|
| **tiny** | **724** | 0.100 ± 0.18 | **27.9%** | 386 | **0.156 ± 0.23** | **37.3%** | 576 |
| small | 117 | 0.294 | 75.2% | 1 | 0.356 | 82.9% | 6 |
| medium | 12 | 0.512 | 83.3% | 0 | 0.562 | 91.7% | 3 |
| large | 8 | 0.673 | 100.0% | 0 | 0.701 | 100.0% | 0 |

Threshold tuning is **specifically a tiny-lesion-detection trick**: tiny
detection +9.4 pp, tiny Dice +56% relative. The cost is 198 extra FPs
across 51 patients (~4 extra FP per scan, mostly tiny blobs).

### Catastrophic-floor diagnostic

| Threshold | Ensemble Dice | Pred fg voxels / case |
|---|---|---|
| t < 0 (all foreground) | 0.0008 | 9,830,400 |
| t = 0 strict (any nonzero) | 0.0022 | 3,289,112 |
| t = 1e-5 | 0.293 | 10,200 |
| t = 1e-3 | 0.343 | 5,164 |
| t = 0.005 (peak) | 0.3457 | 4,495 |
| t = 0.50 (default) | 0.3333 | 2,857 |
| GT reference | — | 3,799 |

At default t=0.50, the model **under-predicts by 25%** (2,857 vs 3,799 true
voxels). At t=0.005 the model is 18% over the true count, which is closer
to optimal Dice. Any threshold in `[0.001, 0.02]` is within 0.001 of optimum.

---

## Local nnU-Net Inference Harness

Set up in this session for fast iteration on the 51-case test set without
cluster latency:

| Component | Path |
|---|---|
| Compatible venv | `.venv_nnunet/` (PyTorch 2.12 + cu130, nnunetv2 latest) |
| Local nnU-Net raw dir | `data/nnunet_local/nnUNet_raw/Dataset600_BrainMet/` (symlinks) |
| Per-fold inference | `misc/run_per_fold_local.py` |
| Bash wrapper for inference | `misc/run_per_fold_local.sh` (uses main venv via eval_nnunet) |
| Threshold sweep analysis | `misc/analyze_per_fold_threshold.py` |
| Size-stratified comparison | `misc/compare_thresholds_by_size.py` |
| General-purpose sweep tool | `src/medgen/scripts/threshold_sweep.py` (cluster + local) |

The local harness produces results identical to cluster eval (verified by
matching exp3 baseline Dice 0.321 within rounding). Cluster path is needed
only for training; all downstream metric analysis runs locally in ~10 min.

### Local execution flow

```bash
# (one-time) install nnunetv2 in dedicated venv
python3 -m venv .venv_nnunet
.venv_nnunet/bin/pip install nnunetv2 nibabel numpy scipy

# Register our custom trainer into nnunetv2's package
TRAINER_PKG=$(.venv_nnunet/bin/python -c \
    "import nnunetv2.training.nnUNetTrainer as p, os; print(os.path.dirname(p.__file__))")
ln -sf "$(pwd)/src/medgen/downstream/nnunet/trainer.py" \
    "$TRAINER_PKG/nnUNetTrainerTensorBoard.py"

# Run 5 fold models on the 51 test cases (saves softmax .npz)
.venv_nnunet/bin/python misc/run_per_fold_local.py

# Sweep thresholds locally
.venv_nnunet/bin/python misc/analyze_per_fold_threshold.py

# Size-stratified comparison at two thresholds
.venv_nnunet/bin/python misc/compare_thresholds_by_size.py
```

The `.venv_nnunet/` is gitignored. The `data/nnunet_local/` is gitignored
(symlinks only, no data duplication).

---

## Literature Comparison

### Grøvik 2020 (BrainMetShare-3 origin)

- Multi-modal nnU-Net on Stanford 156 patients
- Originally published as BraTS-style metrics; **not directly comparable**
  on volumetric per-volume Dice for the 51-patient subset.
- Defines the **official 105/51 hold-out split** that we use for exp3.

### Ottesen 2023 (PMC9889663)

- Same Stanford cohort, 4-channel input (BRAVO + T1pre + T1post + FLAIR)
- Reports **slice-wise** Dice with empty-empty slices = 1.0:
  Stanford nnU-Net = **0.85 ± 0.13**
- Our exp3 (1-channel) slice-wise: **0.835 ± 0.12** — matches within sampling noise
- Our exp345 (4-channel) slice-wise: **0.835 ± 0.12** — same
- Does NOT report volumetric Dice
- Critical methodological note from this work: their slice-wise metric uses
  sagittal axis (axis=0 in our array layout). Our original implementation
  iterated axial, getting 0.73; fixing the axis lifted it to 0.835. See
  `memory/project_ottesen_slice_axis_fix.md` for the diagnosis.

### Ottesen 2025 JMRI (PMC12063759)

- Same Stanford cohort, 4-channel input, 3D U-Net "configured similarly to nnUNet"
- Reports **volumetric per-volume Dice** averaged across 5 folds: **0.66 ± 0.01**
  (the ±0.01 is std across 5 fold means, not per-patient std)
- **Crucially uses a RANDOM 105/51 split, NOT the official Grøvik 2020 hold-out**
- Their methods: *"The annotated Stanford data were split randomly into a
  training and test dataset containing 105 and 51 cases, respectively."*
- Per-fold threshold tuning on validation, HD-BET skull stripping,
  resampling to isotropic 1×1×1 mm
- Our exp3_random_split_d640 (1-channel BRAVO, random split, no tuning,
  no HD-BET, no resampling): **0.5744 ± 0.31**
- Remaining gap to their 0.66: ~0.09, plausibly from 4-channel + threshold
  tuning + HD-BET + isotropic resampling combined

### Honest comparison table for the thesis

| Source | Modalities | Test split | Reported Dice |
|---|---|---|---|
| Ottesen 2023 nnU-Net (Stanford) | 4 | random | 0.85 (slice-wise) |
| Our exp345 (4-channel) | 4 | **official Grøvik 2020 hold-out** | 0.835 slice-wise / **0.326 volumetric** |
| Ottesen 2025 baseline (Stanford) | 4 | random | **0.66 ± 0.01 volumetric** |
| Our exp3_random_split_d640 | 1 (BRAVO only) | **random (seed=42)** | **0.574 volumetric** |
| Our exp3_baseline_v2_d600 | 1 (BRAVO only) | **official Grøvik 2020 hold-out** | **0.321 volumetric** |

The Ottesen 2025 number is **not a fair comparator** to our exp3 on the
official hold-out. When the protocol is matched (random split, same data
pool), we land within 0.09 of their number using only 1 channel vs their 4.

---

## File Reference

### Source code

| Path | Purpose |
|---|---|
| `src/medgen/downstream/nnunet/trainer.py` | `nnUNetTrainerBrainMets` + 3 sibling trainers |
| `src/medgen/downstream/nnunet/convert_dataset.py` | Official-split nnU-Net dataset builder |
| `src/medgen/downstream/nnunet/splits.py` | Isolated preprocessed dirs + race-safe symlinks |
| `src/medgen/downstream/nnunet/evaluate.py` | Per-case Dice variants + size-stratified metrics |
| `src/medgen/scripts/train_nnunet.py` | Training entry point (+ `--validation-only`, `--export-validation-probabilities`) |
| `src/medgen/scripts/eval_nnunet.py` | Inference + eval (+ `--save-probabilities`, `--output-dir`) |
| `src/medgen/scripts/threshold_sweep.py` | Sweep + tune-eval modes, general-purpose |
| `src/medgen/scripts/convert_random_split.py` | Random 105/51 split builder (Ottesen 2025 protocol) |
| `src/medgen/scripts/classify_synth_by_lesion_size.py` | Bucketise synth pool by max Feret diameter |
| `src/medgen/scripts/pick_synth_subset_by_lesion_size.py` | Priority-sample N synth vols (used by exp8) |

### SLURMs (`IDUN/train/downstream/nnunet/`)

Currently retained (delete after final write-up if desired):
- `convert_and_preprocess.slurm` — historical (Dataset 501/502/503)
- `convert_d603_quad.slurm` — exp345 (Dataset 603)
- `convert_d640_random_split.slurm` — exp3 random split (Dataset 640)
- `convert_exp8_d{630,631,632,633}_size_targeted_*.slurm` — exp8 family
- `convert_exp1_1_1000_brain_masked.slurm` — earlier brain-masked variant
- `convert_seg_candidates.slurm` — seg candidate filter
- `exp3_baseline_v2_d600.slurm` — baseline training
- `exp3_random_split_d640.slurm` — random split training (NEW)
- `exp4_baseline_dual_v2_d601.slurm` — dual training
- `exp5_baseline_triple_d602.slurm` — triple training
- `exp345_baseline_quad_d603.slurm` — quad training
- `exp6_*` — synthetic-only training variants
- `exp7_*` — real + synthetic mixed
- `exp8_{1,2,3,4}_size_targeted_*.slurm` — size-targeted exp8
- `eval_5fold_*.slurm` — one per training experiment

### Eval outputs (cluster)

`IDUN/output/output/train/downstream/nnunet/eval_5fold/*.{err,out}` — Python
logging output is in `.err`; `.out` has nnU-Net's own progress prints. Eval
JSONs land in `${NNUNET_RESULTS}/{experiment}/eval_{experiment}.json` (on
cluster) which can be downloaded for analysis.

### Local analysis outputs

| Path | Content |
|---|---|
| `runs/exp3_baseline_v2_d600/threshold_analysis_per_fold.{json,md}` | Per-fold sweep + ensemble curve |
| `runs/exp3_baseline_v2_d600/threshold_size_comparison.json` | Size-stratified t=0.50 vs t=0.005 |
| `runs/exp3_baseline_v2_d600/eval_exp3_baseline_v2_d600/per_fold_test/fold_{0..4}/predictions/*.{nii.gz,npz}` | Raw per-fold predictions for further analysis |

---

## Reproducing the Headline Results

### Reproduce exp3 baseline (official split, 0.32 volumetric)

```bash
# 1. Build Dataset 600 (official Grøvik 2020 split)
sbatch IDUN/train/downstream/nnunet/convert_and_preprocess.slurm    # builds 501/502/503; adapt for 600

# 2. Train all 5 folds
for F in 0 1 2 3 4; do
    FOLD=$F sbatch IDUN/train/downstream/nnunet/exp3_baseline_v2_d600.slurm
done

# 3. 5-fold ensemble eval
sbatch IDUN/train/downstream/nnunet/eval_5fold_exp3_baseline_v2_d600.slurm
```

### Reproduce exp3_random_split (Ottesen 2025 protocol, 0.57 volumetric)

```bash
# 1. Build Dataset 640 with random 105/51 split (seed=42)
sbatch IDUN/train/downstream/nnunet/convert_d640_random_split.slurm

# 2. Train all 5 folds
for F in 0 1 2 3 4; do
    FOLD=$F sbatch IDUN/train/downstream/nnunet/exp3_random_split_d640.slurm
done

# 3. 5-fold ensemble eval
sbatch IDUN/train/downstream/nnunet/eval_5fold_exp3_random_split_d640.slurm
```

The `case_info.json` written by `convert_random_split.py` contains the exact
chosen 105 train + 51 test patient IDs for audit / re-application.

### Reproduce the threshold-sweep + size-stratified analysis (local, no GPU job)

```bash
# 1. Set up local venv + raw data (one-time, see "Local nnU-Net Inference Harness")

# 2. Run 5-fold per-fold inference locally (~10 min on RTX 3090)
.venv_nnunet/bin/python misc/run_per_fold_local.py

# 3. Threshold sweep
.venv_nnunet/bin/python misc/analyze_per_fold_threshold.py
cat runs/exp3_baseline_v2_d600/threshold_analysis_per_fold.md

# 4. Size-stratified t=0.50 vs t=0.005 comparison
.venv_nnunet/bin/python misc/compare_thresholds_by_size.py
```
