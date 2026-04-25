# IDUN/output/ — SLURM job log summaries

Generated from `IDUN/output/**/*.out` (and matched `.err`) via
`scripts/extract_idun_logs.py` → `idun_logs_extracted.json`. Covers
**1,291 SLURM jobs** spanning Dec 2025 – Apr 2026 across 10 subdirectories.

This complements [`EXPERIMENT_SUMMARIES.md`](EXPERIMENT_SUMMARIES.md) (which
covers per-epoch TB metrics from `runs_tb/`) by surfacing data only present
in stdout/stderr:

- **End-of-training test metrics** — MSE / MS-SSIM / PSNR / LPIPS / FID /
  KID ± std / CMMD, computed against the held-out test set after training.
- **Eval-script outputs** — find_optimal_steps, eval_steps_pca,
  diagnose_mean_blur, calibrate_degradation, light_sdedit, analyze_generation_spectrum,
  compare_*, etc. These don't go to TB.
- **Generation-pipeline runs** — sample counts produced, retry counts due
  to seg-validation failures, output directories on the cluster.
- **OOM / crash fingerprints** — jobs that never wrote a TB event are
  invisible in `EXPERIMENT_SUMMARIES.md`; here they get logged.
- **Resume / chain segments** — checkpoint-resume hints + chain-segment IDs.

Per-job entries are formatted as:

```
#### `<job_tag>_<jobid>`  [status]
*Job <jobid> • <node> • <gpu> • wall time*

(metric blocks per applicable category)
```

Status values: `completed`, `oom_killed`, `crashed`, `truncated`
(ran but didn't reach a clean end), `chained` (TB server resubmits),
`empty` (zero-byte log).

---

## Index by subdir

| Subdir | Jobs | Treatment |
|---|---:|---|
| [eval/](#eval) | 230 | per-job, grouped by eval script |
| [generate/](#generate) | 44 | per-job, grouped by generation campaign |
| [profiling/](#profiling) | 22 | per-job VRAM profiling sweeps |
| [debug/](#debug) | 62 | nnU-Net diagnostics, label checks |
| [train/diffusion_3d/](#traindiffusion_3d) | 1,388 | per-experiment with end-of-training test metrics; cross-ref to TB |
| [train/compression/](#traincompression) | 202 | VAE/VQ-VAE/DC-AE training |
| [train/diffusion/](#traindiffusion-2d) | 187 | 2D diffusion (historical) |
| [train/downstream/](#traindownstream) | 300 | nnU-Net + SegResNet training |
| [tensorboard/](#tensorboard) | 140 | TB-server chain logs (bulk) |
| [test/](#test) | 6 | bulk |

---
## eval/

*115 eval-script jobs across 45 distinct scripts.*

### find_optimal_steps (40 jobs)

Golden-section search for the optimal number of Euler steps. Per memory: Euler/25 is optimal for RFlow (FID 27.50 in-training); post-hoc evaluation found 27 steps best. Each job runs the search for one specific model/checkpoint.

#### `find_optimal_steps_24061899`  [✅ completed]
*job 24061899 • idun-06-04 • A100 • 3KB log*

**Search results:** **best_steps=23** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 7 evals
**Best eval (steps=23):** FID 26.632 • KID 0.0225 ± 0.0057 • CMMD 0.1154

#### `find_optimal_steps_exp14_1_24061926`  [✅ completed]
*job 24061926 • idun-06-04 • A100 • 3KB log*

**Search results:** **best_steps=37** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 7 evals
**Best eval (steps=37):** FID 1.743 • KID 0.0002 ± 0.0007 • CMMD 0.0270

#### `find_optimal_steps_exp9_1_24063405`  [✅ completed]
*job 24063405 • idun-06-04 • A100 • 4KB log*

**Search results:** **best_steps=87** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 9 evals
**Best eval (steps=87):** FID 290.430 • KID 0.3536 ± 0.0421 • CMMD 0.6106

#### `find_optimal_steps_exp12b_2_24063406`  [✅ completed]
*job 24063406 • idun-06-04 • A100 • 4KB log*

**Search results:** **best_steps=55** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 9 evals
**Best eval (steps=55):** FID 121.569 • KID 0.1332 ± 0.0183 • CMMD 0.4555

#### `find_optimal_steps_exp13_dit_4x_24063433`  [✅ completed]
*job 24063433 • idun-06-04 • A100 • 5KB log*

**Search results:** **best_steps=10** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 10 evals
**Best eval (steps=10):** FID 335.444 • KID 0.3964 ± 0.0389 • CMMD 0.7503

#### `find_optimal_steps_exp13_dit_4x_24063575`  [⚠️ truncated]
*job 24063575 • idun-06-01 • A100 • 1KB log*


#### `find_optimal_steps_exp1_1b_1c_24067817`  [✅ completed]
*job 24067817 • idun-07-05 • A100 • 16KB log*

**Search results:** **best_steps=50** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 48 evals
**Best eval (steps=50):** FID 141.977 • KID 0.1607 ± 0.0260 • CMMD 0.4944

#### `find_optimal_steps_exp16_hdit_24082353`  [✅ completed]
*job 24082353 • idun-07-09 • A100 • 4KB log*

**Search results:** **best_steps=15** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 7 evals
**Best eval (steps=15):** FID 92.838 • KID 0.1032 ± 0.0220 • CMMD 0.3925

#### `find_optimal_steps_exp19_2_24092964`  [✅ completed]
*job 24092964 • idun-06-03 • A100 • 6KB log*

**Search results:** **best_steps=10** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 10 evals
**Best eval (steps=10):** FID 114.654 • KID 0.1178 ± 0.0124 • CMMD 0.3626 • FID_radimagenet 16.3076

#### `find_optimal_steps_exp1_1_24093079`  [✅ completed]
*job 24093079 • idun-06-01 • A100 • 15KB log*

**Search results:** **best_steps=15** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 30 evals
**Best eval (steps=15):** FID 20.162 • KID 0.0153 ± 0.0067 • CMMD 0.1101 • FID_radimagenet 2.6250

#### `find_optimal_steps_exp1_1b_1c_24121580`  [✅ completed]
*job 24121580 • idun-07-07 • A100 • 23KB log*

**Search results:** **best_steps=50** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 48 evals
**Best eval (steps=50):** FID 142.357 • KID 0.1485 ± 0.0251 • CMMD 0.4974 • FID_radimagenet 6.1581

#### `find_optimal_steps_exp1_1_24127050`  [✅ completed]
*job 24127050 • idun-08-01 • H100 • 15KB log*

**Search results:** **best_steps=23** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 30 evals
**Best eval (steps=23):** FID 23.850 • KID 0.0182 ± 0.0060 • CMMD 0.1236 • FID_radimagenet 0.9273

#### `find_optimal_steps_exp1_1_1000_24128889`  [✅ completed]
*job 24128889 • idun-07-09 • A100 • 16KB log*

**Search results:** **best_steps=27** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 32 evals
**Best eval (steps=27):** FID 19.120 • KID 0.0132 ± 0.0050 • CMMD 0.0899 • FID_radimagenet 0.9097

#### `find_optimal_steps_exp6a_1_24154049`  [❌ crashed]
*job 24154049 • idun-07-09 • A100 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...

real	1m0.994s
user	1m30.489s
sys	0m11.959s
cat: /cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp6a_1_fid_rin/search_results.json: No such file or directory
```

#### `find_optimal_steps_exp23_24157595`  [✅ completed]
*job 24157595 • idun-07-09 • A100 • 16KB log*

**Search results:** **best_steps=27** • metric=`fid` • range=[10.0, 50.0] • 25 volumes • 32 evals
**Best eval (steps=27):** FID 20.375 • KID 0.0145 ± 0.0056 • CMMD 0.1104 • FID_radimagenet 0.9289

#### `find_optimal_steps_exp14_1_morph_24235409`  [✅ completed]
*job 24235409 • idun-07-04 • A100 • 5KB log*

**Search results:** **best_steps=25** • metric=`morphological` • range=[15.0, 60.0] • 25 volumes

#### `find_optimal_steps_exp23_24247668`  [✅ completed]
*job 24247668 • idun-07-10 • A100 • 18KB log*

**Search results:** **best_steps=23** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 36 evals
**Best eval (steps=23):** FID 20.832 • KID 0.0148 ± 0.0049 • CMMD 0.1132 • FID_radimagenet 1.1358

#### `find_optimal_steps_exp22_2_24272519`  [✅ completed]
*job 24272519 • idun-01-03 • H100 • 20KB log*

**Search results:** **best_steps=93** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 36 evals
**Best eval (steps=93):** FID 40.912 • KID 0.0373 ± 0.0079 • CMMD 0.2417 • FID_radimagenet 2.6945

#### `find_optimal_steps_exp1_1_1000_v2_24278428`  [❌ crashed]
*job 24278428 • idun-07-09 • A100 • 21KB log*

**Search results:** **best_steps=32** • metric=`fid` • range=[10.0, 100.0] • 25 volumes • 36 evals
**Best eval (steps=32):** FID 20.842 • KID 0.0144 ± 0.0050 • CMMD 0.1008 • FID_radimagenet 0.8640
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
UnboundLocalError: cannot access local variable 'brain_pca' where it is not associated with a value

real	2m20.865s
user	0m11.315s
sys	0m6.898s
```

#### `find_optimal_steps_exp21_2_24294141`  [✅ completed]
*job 24294141 • idun-01-05 • H100 • 12KB log*

**Search results:** **best_steps=75** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp26_1_24295416`  [✅ completed]
*job 24295416 • idun-01-05 • H100 • 10KB log*

**Search results:** **best_steps=77** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp32_1_1000_24326296`  [✅ completed]
*job 24326296 • idun-01-05 • H100 • 12KB log*

**Search results:** **best_steps=55** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp32_2_1000_24326399`  [✅ completed]
*job 24326399 • idun-06-07 • A100 • 14KB log*

**Search results:** **best_steps=80** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp32_3_1000_24327643`  [✅ completed]
*job 24327643 • idun-01-04 • H100 • 13KB log*

**Search results:** **best_steps=56** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp37_1_24327733`  [✅ completed]
*job 24327733 • idun-08-01 • H100 • 14KB log*

**Search results:** **best_steps=79** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp37_2_24327735`  [✅ completed]
*job 24327735 • idun-08-01 • H100 • 13KB log*

**Search results:** **best_steps=79** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp37_3_24327736`  [✅ completed]
*job 24327736 • idun-08-01 • H100 • 13KB log*

**Search results:** **best_steps=80** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp1v2_2_1000_24334546`  [❌ crashed]
*job 24334546 • idun-08-01 • H100 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/diffusion/strategy_rflow.py", line 875, in generate
    parsed = self._parse_model_i
```

#### `find_optimal_steps_exp1v2_2_1000_24334554`  [✅ completed]
*job 24334554 • idun-09-02 • H200 • 13KB log*

**Search results:** **best_steps=69** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp34_0_mamba_24335868`  [✅ completed]
*job 24335868 • idun-01-03 • H100 • 10KB log*

**Search results:** **best_steps=11** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp34_1_mamba_l_24336079`  [✅ completed]
*job 24336079 • idun-01-03 • H100 • 9KB log*

**Search results:** **best_steps=14** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp38_24336343`  [✅ completed]
*job 24336343 • idun-09-02 • H200 • 13KB log*

**Search results:** **best_steps=80** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp40_24336344`  [✅ completed]
*job 24336344 • idun-01-03 • H100 • 13KB log*

**Search results:** **best_steps=83** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp35_1_24340433`  [✅ completed]
*job 24340433 • idun-07-08 • A100 • 12KB log*

**Search results:** **best_steps=56** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp35_2_24340434`  [✅ completed]
*job 24340434 • idun-09-02 • H200 • 13KB log*

**Search results:** **best_steps=57** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp35_3_24340435`  [✅ completed]
*job 24340435 • idun-08-01 • H100 • 13KB log*

**Search results:** **best_steps=60** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp36_1_24340436`  [✅ completed]
*job 24340436 • idun-06-01 • A100 • 13KB log*

**Search results:** **best_steps=79** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp36_2_24340437`  [✅ completed]
*job 24340437 • idun-08-01 • H100 • 12KB log*

**Search results:** **best_steps=56** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp36_3_24340438`  [✅ completed]
*job 24340438 • idun-09-02 • H200 • 13KB log*

**Search results:** **best_steps=80** • range=[10.0, 100.0] • 25 volumes

#### `find_optimal_steps_exp36_4_24340439`  [❌ crashed]
*job 24340439 • idun-06-06 • A100 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
FileNotFoundError: [Errno 2] No such file or directory: '/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp36_4/search_results.json'

real	330m56.788s
user	313m14.359s
sys	42m3.871s
```

### phema_sweep (9 jobs)

Post-hoc EMA sigma sweep — finds optimal sigma_rel for the PostHocEMA wrapper. Per memory: post-hoc EMA helps (exp1o).

#### `phema_sweep_exp14_2_24303613`  [⚠️ truncated]
*job 24303613 • A100 • 1KB log*


#### `phema_sweep_exp1o_1_24303614`  [⚠️ truncated]
*job 24303614 • A100 • 1KB log*


#### `phema_sweep_exp24_24303615`  [⚠️ truncated]
*job 24303615 • A100 • 1KB log*


#### `phema_sweep_exp25_24303616`  [⚠️ truncated]
*job 24303616 • A100 • 1KB log*


#### `phema_sweep_exp30_24303617`  [⚠️ truncated]
*job 24303617 • A100 • 1KB log*


#### `phema_sweep_exp31_0_24303618`  [⚠️ truncated]
*job 24303618 • A100 • 1KB log*


#### `phema_sweep_exp31_1_24303620`  [⚠️ truncated]
*job 24303620 • H100 • 1KB log*


#### `phema_sweep_exp31_2_24303621`  [⚠️ truncated]
*job 24303621 • A100 • 1KB log*


#### `phema_sweep_exp1o_1_24316696`  [⚠️ truncated]
*job 24316696 • idun-07-09 • A100 • 2KB log*


### diagnose_mean_blur (8 jobs)

Stochastic diversity diagnostic — measures pred_std + HF energy across 8 noise seeds × {0.02, 0.20, 0.50, 0.80} of t for a fixed x₀. Cited in `memory/project_mean_blur_diagnostic.md`: model collapses to deterministic posterior mean; HF deficit grows with t. Motivated exp37 high-t fine-tunes.

#### `diagnose_mean_blur_24318695`  [✅ completed]
*job 24318695 • idun-07-09 • A100 • 1KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_20260417-165922`

#### `diagnose_mean_blur_exp37_1_24322604`  [✅ completed]
*job 24322604 • idun-06-05 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_1_20260418-184514`

#### `diagnose_mean_blur_exp37_2_24322605`  [✅ completed]
*job 24322605 • idun-06-04 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_2_20260418-195122`

#### `diagnose_mean_blur_exp37_1_24324716`  [✅ completed]
*job 24324716 • idun-06-04 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_1_20260419-165541`

#### `diagnose_mean_blur_exp37_2_24324717`  [✅ completed]
*job 24324717 • idun-06-04 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_2_20260419-172130`

#### `diagnose_mean_blur_exp37_1_24326292`  [✅ completed]
*job 24326292 • idun-06-07 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_1_20260420-155002`

#### `diagnose_mean_blur_exp37_2_24326293`  [✅ completed]
*job 24326293 • idun-06-04 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_2_20260420-155020`

#### `diagnose_mean_blur_exp37_3_24327641`  [✅ completed]
*job 24327641 • idun-07-10 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_3_20260420-215325`

### eval_time_shift (5 jobs)

Time-shift parameter optimization (ω in t̃ = (ω·t)/(1 + (ω-1)·t)). Per memory: time-shift 2.0 with Euler-27 is the canonical generation recipe. Uses golden-section search via `--search`.

#### `eval_time_shift_exp1_1_24121738`  [✅ completed]
*job 24121738 • idun-01-05 • H100 • 5KB log*

**Search results:** metric=`fid` • 25 volumes

#### `eval_time_shift_exp23_24232410`  [✅ completed]
*job 24232410 • idun-07-09 • A100 • 21KB log*

**Search results:** metric=`fid` • 25 volumes

#### `eval_time_shift_exp14_1_24235912`  [❌ crashed]
*job 24235912 • idun-07-10 • A100 • 23KB log*

**Search results:** metric=`morphological` • 25 volumes
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
[2026-03-26 02:03:05,339][__main__][INFO] - Euler steps: 37
[2026-03-26 02:03:05,340][__main__][INFO] - Ratios to test: [1.0, 1.5, 2.0, 3.0, 4.0, 6.84]
[2026-03-26 02:03:05,340][__main__][INFO] - Volumes per eval: 25
[2026-03-26 02:03:05,340][__main__][INFO] - Metric: fid (vs 'all')
[2026-0
```

#### `eval_time_shift_exp1_1_1000_v2_24282370`  [✅ completed]
*job 24282370 • idun-01-05 • H100 • 22KB log*

**Search results:** metric=`fid` • 25 volumes

#### `eval_time_shift_exp1v2_2_1000_24336012`  [✅ completed]
*job 24336012 • idun-09-02 • H200 • 21KB log*

**Search results:** metric=`fid` • 25 volumes

### trajectory_emergence_dense (5 jobs)

Dense sampling of the diffusion trajectory to track when anatomical structures emerge across denoising steps.

#### `trajectory_emergence_dense_24327899`  [⚠️ truncated]
*job 24327899 • idun-07-09 • A100 • 1KB log*


#### `trajectory_emergence_dense_24329894`  [⚠️ truncated]
*job 24329894 • idun-01-03 • H100 • 1KB log*


#### `trajectory_emergence_dense_24330511`  [⚠️ truncated]
*job 24330511 • idun-08-01 • H100 • 1KB log*


#### `trajectory_emergence_dense_24334574`  [⚠️ truncated]
*job 24334574 • idun-09-02 • H200 • 1KB log*


#### `trajectory_emergence_dense_24336188`  [✅ completed]
*job 24336188 • idun-09-02 • H200 • 3KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/trajectory_emergence_dense_exp1_1_1000`

### eval_bin_adherence (2 jobs)

Tests whether seg models produce tumors matching the requested size-bin distribution — diagnostic for size_bin conditioning.

#### `eval_bin_adherence_24062831`  [⚠️ truncated]
*job 24062831 • idun-08-01 • H100 • 7KB log*


#### `eval_bin_adherence_exp2e_24107330`  [⚠️ truncated]
*job 24107330 • idun-07-04 • A100 • 3KB log*


### eval_diffrs (2 jobs)

Evaluates Diffusion Rejection Sampling (DiffRS) post-hoc discriminator for quality filtering.

#### `eval_diffrs_24060507`  [⚠️ truncated]
*job 24060507 • idun-06-01 • A100 • chain 0/20 • 11KB log*


#### `eval_diffrs_24061335`  [⚠️ truncated]
*job 24061335 • idun-06-06 • A100 • chain 0/20 • 11KB log*


### eval_pca (2 jobs)

Computes brain/seg PCA error for one model.

#### `eval_pca_exp1_128_variants_24260946`  [⚠️ truncated]
*job 24260946 • idun-07-08 • A100 • 5KB log*


#### `eval_pca_exp1_variants_24262218`  [⚠️ truncated]
*job 24262218 • idun-07-08 • A100 • 4KB log*


### eval_pca_all_models (2 jobs)

Computes brain & seg PCA reconstruction error across multiple models, normalized to 3× max real error threshold. Per memory/pca_reference_values.md: exp1_1_1000 reaches 0.0094 mean (100% pass); exp23 0.0168 (80% pass).

#### `eval_pca_all_models_24271554`  [❌ crashed]
*job 24271554 • idun-01-03 • H100 • 3KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
real	0m18.605s
user	0m12.196s
sys	0m6.133s
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorf
```

#### `eval_pca_all_models_24271679`  [❌ crashed]
*job 24271679 • idun-01-05 • H100 • 3KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
real	0m17.168s
user	0m11.916s
sys	0m5.692s
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorf
```

### eval_steps_pca (2 jobs)

Golden-section search for the step count that minimizes PCA brain-shape error (anatomical fidelity proxy, distinct from FID-based step search).

#### `eval_steps_pca_exp23_24254819`  [⚠️ truncated]
*job 24254819 • idun-07-09 • A100 • 1KB log*


#### `eval_steps_pca_exp23_24255588`  [✅ completed]
*job 24255588 • idun-06-03 • A100 • 13KB log*

**Search results:** **best_steps=76** • 25 volumes • PCA pass 16.0%

### fid_vqvae_rt (2 jobs)

Round-trip FID: encodes real volumes through VQ-VAE → decodes → measures FID against original. Tests reconstruction-quality of the latent space.

#### `fid_vqvae_rt_24336431`  [⚠️ truncated]
*job 24336431 • idun-09-02 • H200 • 3KB log*


#### `fid_vqvae_rt_24341774`  [⚠️ truncated]
*job 24341774 • idun-07-08 • A100 • 3KB log*


### measure_latent_std (2 jobs)

Measures per-channel std of VAE latents — used to validate normalization before latent diffusion training.

#### `measure_latent_std_24061519`  [⚠️ truncated]
*job 24061519 • idun-09-18 • A100 • 11KB log*


#### `measure_latent_std_24076974`  [❌ crashed]
*job 24076974 • idun-06-02 • A100 • 11KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/measure_latent_std.py", line 286, in main
    measure_latent_stats(cfg)
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/data/loaders/compression_detection.py", line 34, in _validate_checkpoint_path
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
FileNotFoundError: Checkpoint not found: runs/compression_3d/multi_modality/exp8_1_vqvae3d_8x_20260202-002436/checkpoint_best.pt

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

### vqvae_probe (2 jobs)

Probes VQ-VAE codebook usage / per-channel statistics for compression diagnostics.

#### `vqvae_probe_24336407`  [⚠️ truncated]
*job 24336407 • idun-09-02 • H200 • 2KB log*


#### `vqvae_probe_24336417`  [⚠️ truncated]
*job 24336417 • idun-09-02 • H200 • 2KB log*


### 24316092 (1 job)

#### `24316092`  [✅ completed]
**

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/degradation_comparison_20260416-154014`

### 24316154 (1 job)

#### `24316154`  [✅ completed]
**

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/fft_comparison_20260416-160826`

### 24316197 (1 job)

#### `24316197_24316197`  [✅ completed]
*job 24316197 • idun-06-07 • A100 • 3KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/light_sdedit_20260416-202900`

### 24316685 (1 job)

#### `24316685_24316685`  [❌ crashed]
*job 24316685 • idun-01-04 • H100 • 1KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/light_sdedit_20260416-214626`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
ValueError: `num_inference_steps`: 1600 should be at least 1, and cannot be larger than `self.num_train_timesteps`: 1000 as the unet model trained with this scheduler can only handle maximal 1000 timesteps.

real	0m23.201s
user	0m21.339s
sys	0m3.214s
```

### calibrate_degradation (1 job)

Calibrates IR-SDE / SDEdit degradation strength t₀ for the restoration network. Per memory/project_irsde_restoration.md: best t₀ = 0.50.

#### `calibrate_degradation_24298808`  [✅ completed]
*job 24298808 • idun-01-04 • H100 • 2KB log*

**Search results:** **best_t0=0.5**

### eval_compare (1 job)

Generic comparison script.

#### `eval_compare_exp32_24298183`  [✅ completed]
*job 24298183 • idun-07-10 • A100 • 30KB log*

**Search results:** 25 volumes

### eval_compression_fid (1 job)

FID of compression-roundtrip volumes vs originals.

#### `eval_compression_fid_exp8_1_24063454`  [✅ completed]
*job 24063454 • idun-06-04 • A100 • 8KB log*

**Search results:** 51 volumes

### eval_pca_dual_triple (1 job)

PCA error for dual/triple-mode models.

#### `eval_pca_dual_triple_24260933`  [⚠️ truncated]
*job 24260933 • idun-06-07 • A100 • 2KB log*


### eval_restart (1 job)

Restart-sampling evaluation (Restart Sampling, Xu et al.).

#### `eval_restart_24061646`  [⚠️ truncated]
*job 24061646 • idun-06-06 • A100 • chain 0/20 • 18KB log*


### eval_threshold (1 job)

Threshold sweep for binary masking thresholds.

#### `eval_threshold_exp14_1_24286663`  [✅ completed]
*job 24286663 • idun-01-05 • H100 • 6KB log*

**Search results:** 25 volumes

### fid_eq (1 job)

FID equalization across feature extractors.

#### `fid_eq_24337754`  [⚠️ truncated]
*job 24337754 • idun-07-10 • A100 • 2KB log*


### find_optimal_cfg (1 job)

Searches optimal classifier-free guidance scale. Per memory: best CFG scale = 1.0 (no guidance); >1.0 is dramatically worse.

#### `find_optimal_cfg_exp1n_24154081`  [✅ completed]
*job 24154081 • idun-07-09 • A100 • 18KB log*

**Search results:** metric=`fid` • 25 volumes

### find_optimal_freeu (1 job)

Searches optimal FreeU coefficients.

#### `find_optimal_freeu_exp1_1_1000_24151369`  [✅ completed]
*job 24151369 • idun-01-03 • H100 • 33KB log*

**Search results:** metric=`fid` • 25 volumes

### hybrid_generation (1 job)

Hybrid generation modes (e.g. real seg + generated bravo).

#### `hybrid_generation_24327875`  [⚠️ truncated]
*job 24327875 • idun-07-10 • A100 • 4KB log*


### light_sdedit (1 job)

SDEdit pair generation for restoration training data. Outputs go to `runs/eval/light_sdedit_*/` (consumed by exp33 IR-SDE training).

#### `light_sdedit_24316709`  [✅ completed]
*job 24316709 • idun-01-04 • H100 • 3KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/light_sdedit_20260416-215901`

### measure_vqvae_latent_std (1 job)

Per-channel latent std measurement (alias of measure_latent_std).

#### `measure_vqvae_latent_std_24061933`  [⚠️ truncated]
*job 24061933 • idun-06-04 • A100 • 40KB log*


### pca_ablation_clean (1 job)

Ablation on PCA model components / threshold.

#### `pca_ablation_clean_24325185`  [⚠️ truncated]
*job 24325185 • idun-03-02 • 3KB log*


### precompute_vqvae_pairs (1 job)

Pre-encodes volumes through VQ-VAE for faster latent-diffusion training.

#### `precompute_vqvae_pairs_24341605`  [⚠️ truncated]
*job 24341605 • idun-09-02 • H200 • 2KB log*


### pregen_restoration_pairs (1 job)

Pre-generates input/target pairs for IR-SDE restoration training.

#### `pregen_restoration_pairs_24334548`  [⚠️ truncated]
*job 24334548 • idun-09-02 • H200 • 2KB log*


### restore (1 job)

Single restoration pipeline test (post-hoc deblur).

#### `restore_exp41_24336345`  [⚠️ truncated]
*job 24336345 • idun-01-03 • H100 • 3KB log*


### sdedit (1 job)

Single SDEdit run.

#### `sdedit_24341708`  [💥 oom_killed]
*job 24341708 • idun-08-01 • H100 • 1KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/dif
```

### spectral_eq (1 job)

Spectral equalization eval — analyzes radial 3D power spectrum vs real, see memory/project_phase1_spectrum_finding.md.

#### `spectral_eq_24341707`  [⚠️ truncated]
*job 24341707 • idun-08-01 • H100 • 2KB log*


### stochastic_sampling (1 job)

Compares deterministic Euler vs stochastic ancestral sampling at varying step counts. Cited in mean-blur experiments.

#### `stochastic_sampling_24327897`  [❌ crashed]
*job 24327897 • idun-07-10 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/stochastic_sampling_20260421-044525`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
PermissionError: [Errno 13] Permission denied: '/home/mode'

real	69m48.480s
user	67m10.709s
sys	2m20.948s
```

### stochastic_sampling_sweep (1 job)

Sweep across stochastic-sampling parameters (sigma_min, sigma_max, etc.).

#### `stochastic_sampling_sweep_24330605`  [❌ crashed]
*job 24330605 • idun-07-10 • A100 • 3KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/stochastic_sampling_sweep_20260422-005501`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
PermissionError: [Errno 13] Permission denied: '/home/mode'

real	115m57.657s
user	110m36.900s
sys	4m13.181s
```

### test_pca_wdm_ldm (1 job)

Tests PCA brain-shape metrics on WDM and LDM model outputs.

#### `test_pca_wdm_ldm_24271505`  [⚠️ truncated]
*job 24271505 • A100 • 1KB log*


### timestep_response (1 job)

Probes model output sensitivity at specific timesteps.

#### `timestep_response_24327898`  [❌ crashed]
*job 24327898 • idun-07-10 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/timestep_response_20260421-044525`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
PermissionError: [Errno 13] Permission denied: '/home/mode'

real	115m22.681s
user	104m46.535s
sys	10m11.750s
```

### train_diffrs (1 job)

Trains the DiffRS discriminator.

#### `train_diffrs_24059814`  [⚠️ truncated]
*job 24059814 • idun-06-01 • A100 • epoch 100/100 • 9KB log*

**Best metrics:** `loss`=0.3634
**Trainer best loss:** 0.363400

### trajectory_emergence_all (1 job)

All-model variant of trajectory emergence.

#### `trajectory_emergence_all_24327870`  [⚠️ truncated]
*job 24327870 • idun-07-08 • A100 • 10KB log*


### velocity_breakdown (1 job)

Decomposes RFlow velocity into x₀ vs x₁ contributions; complement of diagnose_mean_blur.

#### `velocity_breakdown_24316982`  [✅ completed]
*job 24316982 • idun-07-09 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/velocity_breakdown_20260417-043545`

### velocity_divergence_all (1 job)

Computes velocity-field divergence statistics across multiple models.

#### `velocity_divergence_all_24327871`  [✅ completed]
*job 24327871 • idun-07-10 • A100 • 2KB log*

**Results dir:** `/cluster/work/modestas/AIS4900_master/runs/eval/velocity_divergence_20260421-022058`

### verify_ldm_pipeline (1 job)

Sanity-checks the latent-diffusion encode→diffuse→decode pipeline.

#### `verify_ldm_pipeline_24063571`  [⚠️ truncated]
*job 24063571 • idun-06-04 • A100 • 2KB log*


### verify_wdm_pipeline (1 job)

Sanity-checks the wavelet-diffusion encode→diffuse→decode pipeline.

#### `verify_wdm_pipeline_24063567`  [⚠️ truncated]
*job 24063567 • idun-06-04 • A100 • 1KB log*


---
## generate/

*22 generation-pipeline jobs across 5 groups.*

### compare_* (2 jobs)

Multi-model comparison batch generation — 10 volumes per model for side-by-side evaluation. `compare_exp37_*` is the data backing memory/project_phase1_spectrum_finding.md.

#### `compare_exp1_1_vs_exp37_24326455`  [✅ completed]
*job 24326455 • idun-06-01 • A100 • 28KB log*

**Generation summary (model: completed/total):**
  - `exp1_1_1000`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp1_1_1000`
  - `exp32_1_1000`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp32_1_1000`
  - `exp32_2_1000`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp32_2_1000`
  - `exp32_3_1000`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp32_3_1000`
  - `exp37_1`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp37_1`
  - `exp37_2`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp37_2`
  - `exp37_3`: 10/10 → `/cluster/work/modestas/MedicalDataSets/generated/compare_exp37_20260420-165132/exp37_3`

#### `compare_imagenet_optima_24329950`  [✅ completed]
*job 24329950 • idun-07-08 • A100 • 27KB log*

**Saved samples:**
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp32_1_1000`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp32_2_1000`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp32_3_1000`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp37_1`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp37_2`
  - 10 samples → `/cluster/work/modestas/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp37_3`

### exp1_* (1 jobs)

Early generation tests at 128/256 (pre-pipeline).

#### `exp1_gen_3d_128_23980653`  [✅ completed]
*job 23980653 • idun-07-10 • A100 • 5KB log*

**Saved samples:**
  - 100 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_3d_128_100_baseline`

### exp1_1* (6 jobs)

Generation runs with the canonical best model `exp1_1_1000`. Includes 525-volume train-set generation for downstream nnU-Net, real-seg conditioning tests, and ImageNet/RadImageNet variants.

#### `exp1_1_gen_3d_256_23980654`  [✅ completed]
*job 23980654 • idun-01-03 • H100 • 4KB log*

**Saved samples:**
  - 100 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_3d_256_100_baseline`

#### `exp1_1_gen_3d_256_24083168`  [⚠️ truncated]
*job 24083168 • idun-06-01 • A100 • 8KB log*


#### `exp1_1_gen_3d_256_24085731`  [⚠️ truncated]
*job 24085731 • idun-06-05 • A100 • 45KB log*


#### `exp1_1_gen_3d_256_24090257`  [✅ completed]
*job 24090257 • idun-01-03 • H100 • 54KB log*

**Saved samples:**
  - 525 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_3d_256_525_baseline`

#### `exp1_1_gen_3d_256_24111089`  [❌ crashed]
*job 24111089 • idun-07-07 • A100 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/generate.py", line 984, in main
    run_3d_pipeline(cfg, output_dir)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	2m45.861s
user	0m7.657s
sys	0m2.961s
```

#### `exp1_1_gen_3d_256_24245630`  [❌ crashed]
*job 24245630 • idun-07-09 • A100 • chain 0/10 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/generate.py", line 1065, in main
    run_3d_pipeline(cfg, output_dir)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m18.693s
user	0m8.142s
sys	0m3.122s
```

### gen_degradation_pairs* (1 jobs)

Pre-generated degradation pairs for IR-SDE restoration training (see also eval/light_sdedit, eval/pregen_restoration_pairs).

#### `gen_degradation_pairs_24298904`  [⚠️ truncated]
*job 24298904 • idun-01-04 • H100 • chain 0/5 • 1KB log*


### other* (12 jobs)

#### `gen_exp1_1_1000_bravo_24255600`  [✅ completed]
*job 24255600 • idun-06-03 • A100 • chain 0/10 • 57KB log*

**Saved samples:**
  - 525 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_1000_bravo_525`

#### `gen_exp23_bravo_imagenet_24255601`  [⚠️ truncated]
*job 24255601 • idun-07-10 • A100 • chain 0/10 • 94KB log*


#### `gen_exp23_bravo_real_seg_24255603`  [🔗 chained]
*job 24255603 • idun-06-07 • A100 • chain 0/10 • 307KB log*


#### `gen_exp23_bravo_real_seg_24260625`  [✅ completed]
*job 24260625 • idun-07-08 • A100 • chain 1/10 • 70KB log*

**Saved samples:**
  - 329 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp23_bravo_real_seg`

#### `gen_exp1_1_imagenet_24287152`  [✅ completed]
*job 24287152 • idun-01-05 • H100 • chain 0/10 • 79KB log*

**Saved samples:**
  - 525 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_imagenet_525`

#### `gen_exp1_1_radimagenet_24287154`  [✅ completed]
*job 24287154 • idun-06-04 • A100 • chain 0/10 • 83KB log*

**Saved samples:**
  - 525 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525`

#### `gen_exp1_1_1000_bravo_real_seg_test1_nofilter_24323132`  [✅ completed]
*job 24323132 • idun-01-03 • H100 • chain 0/5 • 14KB log*

**Saved samples:**
  - 150 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_1000_bravo_real_seg_test1_nofilter`

#### `gen_exp23_bravo_real_seg_test1_nofilter_24323134`  [✅ completed]
*job 24323134 • idun-01-03 • H100 • chain 0/5 • 36KB log*

**Saved samples:**
  - 150 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp23_bravo_real_seg_test1_nofilter`

#### `gen_exp1_1_1000_bravo_real_seg_test1_nofilter_24324749`  [✅ completed]
*job 24324749 • idun-01-03 • H100 • chain 0/5 • 15KB log*

**Saved samples:**
  - 150 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1_1_1000_bravo_real_seg_test1_nofilter`

#### `gen_exp23_bravo_real_seg_test1_nofilter_24324750`  [✅ completed]
*job 24324750 • idun-06-04 • A100 • chain 0/5 • 36KB log*

**Saved samples:**
  - 150 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp23_bravo_real_seg_test1_nofilter`

#### `gen_exp1v2_2_1000_dual_525_24336350`  [❌ crashed]
*job 24336350 • idun-09-02 • H200 • chain 0/10 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/generate.py", line 1208, in main
    run_3d_pipeline(cfg, output_dir)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m10.283s
user	0m5.293s
sys	0m1.449s
```

#### `gen_exp1v2_2_1000_dual_525_24336371`  [✅ completed]
*job 24336371 • idun-01-03 • H100 • chain 0/10 • 100KB log*

**Saved samples:**
  - 525 samples → `/cluster/work/modestas/MedicalDataSets/generated/exp1v2_2_1000_dual_525`

---
## profiling/

*11 jobs.*

VRAM profiling sweeps for model architectures (DiT, UNet, HDiT, UViT, Mamba). Logs the FLOPs, peak VRAM, and forward/backward times across configurations. Cited in `docs/profiling_results.md`.

#### `profile_latent_3d_unet_23996563`  [⚠️ truncated]
*job 23996563 • idun-06-01 • A100 • 7KB log*


#### `profile_s2d_3d_24026871`  [⚠️ truncated]
*job 24026871 • idun-01-04 • H100 • 31KB log*


#### `profile_dit_memory_24042653`  [⚠️ truncated]
*job 24042653 • idun-06-04 • A100 • 39KB log*


#### `profile_pixel_3d_unet_24056063`  [⚠️ truncated]
*job 24056063 • idun-07-08 • A100 • 12KB log*


#### `sweep_vae3d_24056474`  [⚠️ truncated]
*job 24056474 • idun-06-07 • A100 • 4KB log*


#### `profile_hdit_uvit_24058718`  [⚠️ truncated]
*job 24058718 • idun-07-08 • A100 • 28KB log*


#### `profile_pixel_3d_unet_v2_24090875`  [⚠️ truncated]
*job 24090875 • idun-07-08 • A100 • 24KB log*


#### `profile_small_24152233`  [⚠️ truncated]
*job 24152233 • idun-07-10 • A100 • 20KB log*


#### `profile_small_24154175`  [⚠️ truncated]
*job 24154175 • idun-07-09 • A100 • 19KB log*


#### `profile_mamba_24314337`  [⚠️ truncated]
*job 24314337 • idun-06-01 • A100 • 66KB log*


#### `profile_mamba_24315093`  [⚠️ truncated]
*job 24315093 • idun-06-07 • A100 • 56KB log*


---
## debug/

*31 jobs.*

nnU-Net diagnostics, label-distribution checks, and pipeline-debug scripts. Most logs are short (one-shot diagnostic prints) or smoke-tests that verify a code path before committing.

#### `24111081`  [⚠️ truncated]
*9KB log*


#### `24115214`  [⚠️ truncated]
*5KB log*


#### `24115216`  [⚠️ truncated]
*6KB log*


#### `test_old_23885933`  [⚠️ truncated]
*job 23885933 • A100 • epoch 93/500 • 21KB log*


#### `test_new_23885934`  [❌ crashed]
*job 23885934 • A100 • epoch 57/500 • 23KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/clus
```

#### `bisect_bdb9b94_23885935`  [❌ crashed]
*job 23885935 • A100 • epoch 30/500 • 15KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/tmp/bisect_bdb9b94_23885935/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/
```

#### `bisect_8ed1be3_23885936`  [⚠️ truncated]
*job 23885936 • A100 • epoch 60/500 • 17KB log*


#### `test_head_bf16mse_23885955`  [❌ crashed]
*job 23885955 • A100 • epoch 65/500 • 24KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/clus
```

#### `bisect_58fd733_23886666`  [❌ crashed]
*job 23886666 • A100 • epoch 59/500 • 23KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/tmp/bisect_58fd733_23886666/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/
```

#### `bisect_ce119d5_23886667`  [⚠️ truncated]
*job 23886667 • A100 • epoch 105/500 • 24KB log*


#### `bisect_9d1217d_23886843`  [❌ crashed]
*job 23886843 • A100 • epoch 59/500 • 22KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/tmp/bisect_9d1217d_23886843/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/
```

#### `bisect_d9caa07_23886844`  [⚠️ truncated]
*job 23886844 • A100 • 1KB log*


#### `bisect_f8468d2_23886845`  [❌ crashed]
*job 23886845 • A100 • epoch 66/500 • 23KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/tmp/bisect_f8468d2_23886845/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/
```

#### `bisect_baadcdd_23887991`  [⚠️ truncated]
*job 23887991 • A100 • epoch 103/500 • 24KB log*


#### `bisect_c4ccc65_23887992`  [⚠️ truncated]
*job 23887992 • A100 • epoch 88/500 • 21KB log*


#### `bisect_6240069_23887997`  [⚠️ truncated]
*job 23887997 • A100 • epoch 95/500 • 22KB log*


#### `test_new_23888145`  [⚠️ truncated]
*job 23888145 • A100 • epoch 64/500 • 18KB log*


#### `test_new_23888150`  [⚠️ truncated]
*job 23888150 • A100 • epoch 76/500 • 20KB log*


#### `test_new_23888188`  [⚠️ truncated]
*job 23888188 • A100 • epoch 75/500 • 20KB log*


#### `test_baadcdd_baseline_23888206`  [⚠️ truncated]
*job 23888206 • A100 • epoch 93/500 • 22KB log*


#### `test_head_baseline_23888207`  [⚠️ truncated]
*job 23888207 • A100 • epoch 52/500 • 16KB log*


#### `test_no_msssim_23888213`  [⚠️ truncated]
*job 23888213 • A100 • epoch 107/500 • 25KB log*


#### `test_no_compile_no_msssim_23888214`  [⚠️ truncated]
*job 23888214 • A100 • epoch 86/500 • 21KB log*


#### `test_no_compile_23888215`  [⚠️ truncated]
*job 23888215 • A100 • epoch 63/500 • 18KB log*


#### `test_head_baseline_23888301`  [⚠️ truncated]
*job 23888301 • A100 • epoch 77/500 • 20KB log*


#### `exp1_memory_profile_23962252`  [💥 oom_killed]
*job 23962252 • idun-06-04 • A100 • epoch 2/10 • 9KB log*

**Traceback excerpt:**
```
ation:  32%|███▏      | 8/25 [00:03<00:06,  2.47it/s]
Validation:  36%|███▌      | 9/25 [00:04<00:08,  1.79it/s]
Validation:  40%|████      | 10/25 [00:05<00:08,  1.79it/s]
...
Epoch 3:   3%|▎         | 3/105 [00:02<01:22,  1.23it/s, loss=0.4570]
Epoch 3:   3%|▎         | 3/105 [00:03<01:22,  1.23it/s, loss=0.4536]
Epoch 3:   4%|▍         | 4/105 [00:03<01:15,  1.33it/s, loss=0.4536]
Epoch 3:   4%|▍         | 4/105 [00:03<01:15,  1.33it/s, loss=0.4596]
Epoch 3:   5%|▍         | 5/105 [00:03<01:11,  1.39it/s, loss=0.4596][2026-01-18T20:48:23.608] error: *** JOB 23962252 ON idun-06-04 CANCELLED AT 2026-01-18T20:48:23 DUE to SIGNAL Terminated ***
```

#### `exp1_memory_profile_23962262`  [💥 oom_killed]
*job 23962262 • idun-06-04 • A100 • epoch 1/10 • 7KB log*

**Traceback excerpt:**
```
:39,  1.51it/s, loss=0.7120]
Epoch 2:  44%|████▍     | 46/105 [00:32<00:39,  1.51it/s, loss=0.7007]
Epoch 2:  45%|████▍     | 47/105 [00:32<00:38,  1.51it/s, loss=0.7007]
...
Epoch 2:  54%|█████▍    | 57/105 [00:39<00:31,  1.51it/s, loss=0.6503]
Epoch 2:  55%|█████▌    | 58/105 [00:39<00:31,  1.51it/s, loss=0.6503][2026-01-18T20:54:01.508] error: *** JOB 23962262 ON idun-06-04 CANCELLED AT 2026-01-18T20:54:01 DUE to SIGNAL Terminated ***

Epoch 2:  55%|█████▌    | 58/105 [00:40<00:31,  1.51it/s, loss=0.6489]
Epoch 2:  56%|█████▌    | 59/105 [00:40<00:30,  1.51it/s, loss=0.6489]
```

#### `exp1_memory_profile_23962271`  [⚠️ truncated]
*job 23962271 • idun-06-04 • A100 • epoch 3/10 • 11KB log*


#### `debug_conditioning_24062921`  [⚠️ truncated]
*job 24062921 • idun-08-01 • H100 • 3KB log*


#### `debug_conditioning_24062934`  [⚠️ truncated]
*job 24062934 • idun-08-01 • H100 • 6KB log*


#### `debug_ldm_roundtrip_exp9_1_24063460`  [❌ crashed]
*job 24063460 • idun-06-04 • A100 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...

real	0m23.100s
user	0m11.324s
sys	0m9.336s
cat: /cluster/work/modestas/MedicalDataSets/debug_ldm_roundtrip_exp9_1/roundtrip_results.json: No such file or directory
```

---
## train/diffusion_3d/

*695 training jobs across 12 experiment families. Many experiments span multiple SLURM jobs (chain-resumes); they are listed chronologically by job ID within each family. Cross-reference to `EXPERIMENT_SUMMARIES.md §diffusion_3d/bravo` for the per-epoch TB curves of the corresponding `runs_tb/` entries.*

### exp1 (13 jobs)

*Status: ❌ crashed=5 ✅ completed=4 ⚠️ truncated=2 💥 oom_killed=2*

Long-running baseline + variants (1a..1v3). Includes the canonical best model `exp1_1_1000` (FID 19.12 post-hoc) and the dual/triple mode variants exp1v2/exp1v3. Many runs are chain-resumes (the 1000-epoch runs span multiple SLURM jobs).

#### `exp1_pixel_bravo_23969657`  [⚠️ truncated]
*job 23969657 • idun-06-04 • A100 • epoch 500/500 • 204KB log*


#### `exp1_1_pixel_bravo_23969658`  [⚠️ truncated]
*job 23969658 • idun-06-05 • A100 • epoch 500/500 • 207KB log*


#### `exp1_1_pixel_bravo_23982866`  [💥 oom_killed]
*job 23982866 • idun-06-02 • A100 • epoch 500/500 • 2211KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1906m18.389s
user	2198m4.836s
sys	228m21.742s
```

#### `exp1_pixel_bravo_23982867`  [❌ crashed]
*job 23982867 • idun-06-01 • A100 • epoch 500/500 • 4068KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	881m54.783s
user	1079m50.263s
sys	204m0.064s
```

#### `exp1_debugging_cfg_23984445`  [❌ crashed]
*job 23984445 • idun-06-01 • A100 • epoch 500/500 • 268KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	617m21.303s
user	926m29.906s
s
```

#### `exp1_debugging_cfg_23985115`  [❌ crashed]
*job 23985115 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	605m56.941s
user	935m6.195s
sy
```

#### `exp1_pixel_bravo_23985116`  [❌ crashed]
*job 23985116 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	609m29.011s
user	931m57.274s
s
```

#### `exp1_1_pixel_bravo_23989010`  [💥 oom_killed]
*job 23989010 • idun-06-07 • A100 • epoch 500/500 • 505KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1884m48.852s
user	2223m3.903s
```

#### `exp1_pixel_bravo_23989011`  [❌ crashed]
*job 23989011 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	601m45.956s
user	939m36.336s
s
```

#### `exp1e_pixel_bravo_snr_gamma_24072105`  [✅ completed]
*job 24072105 • idun-06-05 • A100 • 16.95h training • epoch 500/500 • 87KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003829 • MS-SSIM 0.9470 • PSNR 32.96 dB • LPIPS 0.6404 • FID 92.73 • KID 0.1051 ± 0.0058 • CMMD 0.2918
  - **latest** ckpt (26 samples): MSE 0.009645 • MS-SSIM 0.9034 • PSNR 29.97 dB • LPIPS 0.4710 • FID 68.73 • KID 0.0715 ± 0.0052 • CMMD 0.1557

#### `exp1f_pixel_bravo_edm_precond_24072106`  [✅ completed]
*job 24072106 • idun-06-01 • A100 • 16.8h training • epoch 500/500 • 189KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01936 • MS-SSIM 0.9049 • PSNR 30.64 dB • LPIPS 1.2412 • FID 219.37 • KID 0.2865 ± 0.0107 • CMMD 0.6236
  - **latest** ckpt (26 samples): MSE 0.0177 • MS-SSIM 0.9009 • PSNR 30.73 dB • LPIPS 1.3032 • FID 220.17 • KID 0.2913 ± 0.0124 • CMMD 0.6248

#### `exp1e_1_pixel_bravo_snr_gamma_24072109`  [✅ completed]
*job 24072109 • idun-06-01 • A100 • 49.2h training • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002907 • MS-SSIM 0.9563 • PSNR 33.09 dB • LPIPS 0.5720 • FID 114.24 • KID 0.1324 ± 0.0101 • CMMD 0.2861
  - **latest** ckpt (26 samples): MSE 0.004712 • MS-SSIM 0.9606 • PSNR 33.57 dB • LPIPS 0.4926 • FID 108.41 • KID 0.1254 ± 0.0082 • CMMD 0.2294

#### `exp1f_1_pixel_bravo_edm_precond_24072110`  [✅ completed]
*job 24072110 • idun-06-02 • A100 • 49.43h training • epoch 500/500 • 180KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003571 • MS-SSIM 0.9392 • PSNR 31.55 dB • LPIPS 0.5694 • FID 211.48 • KID 0.3090 ± 0.0184 • CMMD 0.4421
  - **latest** ckpt (26 samples): MSE 0.003059 • MS-SSIM 0.9476 • PSNR 32.02 dB • LPIPS 0.5285 • FID 196.32 • KID 0.2765 ± 0.0181 • CMMD 0.4081

### exp2 (12 jobs)

*Status: ⚠️ truncated=6 💥 oom_killed=3 ❌ crashed=3*

#### `exp2_pixel_seg_sizebin_23972121`  [⚠️ truncated]
*job 23972121 • idun-06-05 • A100 • epoch 500/500 • 92KB log*


#### `exp2_1_pixel_seg_sizebin_23973544`  [💥 oom_killed]
*job 23973544 • idun-07-10 • A100 • epoch 500/500 • 441KB log*

**Traceback excerpt:**
```

real	2381m38.860s
user	2500m26.158s
sys	810m42.434s
```

#### `exp2_pixel_seg_sizebin_23982949`  [⚠️ truncated]
*job 23982949 • idun-07-10 • A100 • epoch 500/500 • 162KB log*


#### `exp2_1_pixel_seg_sizebin_23982950`  [💥 oom_killed]
*job 23982950 • idun-06-03 • A100 • epoch 500/500 • 511KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)

real	2411m48.584s
user	2240m33.632s
sys	957m5.454s
```

#### `exp2b_1_pixel_seg_input_cond_23982974`  [❌ crashed]
*job 23982974 • idun-07-08 • A100 • 1KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 95, in main
    validate_config(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m26.032s
user	0m7.485s
sys	0m3.812s
```

#### `exp2b_pixel_seg_input_cond_23982975`  [❌ crashed]
*job 23982975 • idun-07-08 • A100 • 1KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 95, in main
    validate_config(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m26.031s
user	0m7.500s
sys	0m3.780s
```

#### `exp2_pixel_seg_sizebin_23996720`  [⚠️ truncated]
*job 23996720 • idun-07-10 • A100 • epoch 443/500 • 84KB log*


#### `exp2_1_pixel_seg_sizebin_23996728`  [⚠️ truncated]
*job 23996728 • idun-06-03 • A100 • epoch 500/500 • 97KB log*


#### `exp2b_pixel_seg_input_cond_23996776`  [⚠️ truncated]
*job 23996776 • idun-06-07 • A100 • epoch 500/500 • 89KB log*


#### `exp2b_1_pixel_seg_input_cond_23997358`  [⚠️ truncated]
*job 23997358 • idun-07-10 • A100 • epoch 500/500 • 95KB log*


#### `exp2c_pixel_seg_improved_24031953`  [❌ crashed]
*job 24031953 • idun-07-10 • A100 • epoch 500/500 • 283KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1165, in train
    logger.error(f"Checkpoint save failed at epoch {epoch}: {
```

#### `exp2c_1_pixel_seg_improved_24039864`  [💥 oom_killed]
*job 24039864 • idun-06-01 • A100 • 2.66h training • epoch 25/500 • 27KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.31 GiB. GPU 0 has a total capacity of 79.25 GiB of which 24.65 GiB is free. Including non-PyTorch memory, this process has 54.59 GiB memory in use. Of the allocated memory 23.57 GiB is allocated by PyTorch, with 9.93 GiB allocated in private pools (e.g., CUDA Graphs), and 30.41 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation fo
```

### exp4 (4 jobs)

*Status: ❌ crashed=3 ⚠️ truncated=1*

SDA at 128×160 — superseded.

#### `exp4_pixel_bravo_sda_23973532`  [❌ crashed]
*job 23973532 • idun-07-10 • A100 • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1210m48.286s
user	1508m58.309s
sys	396m37.989s
```

#### `exp4_1_pixel_bravo_sda_23982869`  [❌ crashed]
*job 23982869 • idun-07-10 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1347m16.527s
user	1744m28.560s
```

#### `exp4_pixel_bravo_sda_23987602`  [⚠️ truncated]
*job 23987602 • idun-07-09 • A100 • epoch 400/500 • 131KB log*


#### `exp4_pixel_bravo_sda_23989012`  [❌ crashed]
*job 23989012 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1055m3.000s
user	1059m39.083s
```

### exp5 (5 jobs)

*Status: ❌ crashed=5*

Early ScoreAug at 128×160 — superseded by exp23.

#### `exp5_1_pixel_bravo_scoreaug_23973838`  [❌ crashed]
*job 23973838 • idun-07-09 • A100 • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	921m51.327s
user	1320m17.354s
sys	289m1.255s
```

#### `exp5_2_pixel_bravo_scoreaug_compose_23973839`  [❌ crashed]
*job 23973839 • idun-01-04 • H100 • epoch 500/500 • 88KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	598m1.317s
user	874m37.874s
sys	186m10.506s
```

#### `exp5_3_pixel_bravo_scoreaug_23982868`  [❌ crashed]
*job 23982868 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	929m40.615s
user	1072m45.500s
sys	352m31.078s
```

#### `exp5_1_pixel_bravo_scoreaug_23987603`  [❌ crashed]
*job 23987603 • idun-07-08 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	881m39.960s
user	1470m22.326s
```

#### `exp5_1_pixel_bravo_scoreaug_23989013`  [❌ crashed]
*job 23989013 • idun-07-08 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	888m54.571s
user	1451m38.134s
```

### exp6 (3 jobs)

*Status: ❌ crashed=3*

ControlNet — exp6a stage-1, exp6b stage-2.

#### `exp6a_pixel_bravo_controlnet_stage1_23982865`  [❌ crashed]
*job 23982865 • idun-06-01 • A100 • epoch 500/500 • 3712KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	693m48.232s
user	941m24.993s
sys	152m50.417s
```

#### `exp6a_pixel_bravo_controlnet_stage1_23987604`  [❌ crashed]
*job 23987604 • idun-06-03 • A100 • epoch 500/500 • 272KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	597m41.009s
user	938m30.300s
s
```

#### `exp6a_pixel_bravo_controlnet_stage1_23989014`  [❌ crashed]
*job 23989014 • idun-07-08 • A100 • epoch 500/500 • 272KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	644m34.121s
user	1342m46.552s
```

### exp7 (14 jobs)

*Status: ❌ crashed=12 ⚠️ truncated=2*

SiT/DiT pixel-space scaling sweep at 128/256.

#### `exp7_1_sit_s_256_patch16_23982976`  [⚠️ truncated]
*job 23982976 • idun-07-08 • A100 • epoch 262/500 • 93KB log*


#### `exp7_sit_s_128_patch8_23982977`  [⚠️ truncated]
*job 23982977 • idun-07-08 • A100 • epoch 410/500 • 136KB log*


#### `exp7_1_sit_b_256_patch16_23983255`  [❌ crashed]
*job 23983255 • idun-06-01 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W127 19:39:30.867309723 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	366m3.066s
user	697m33.444s
sys	264m1.699s
```

#### `exp7_1_sit_l_256_patch16_23983256`  [❌ crashed]
*job 23983256 • idun-06-03 • A100 • epoch 500/500 • 162KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 01:34:45.924049829 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	428m21.419s
user	712m22.139s
sys	270m0.198s
```

#### `exp7_sit_l_128_patch8_23983257`  [❌ crashed]
*job 23983257 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 02:54:36.218818035 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	434m0.343s
user	763m29.757s
sys	210m3.392s
```

#### `exp7_sit_b_128_patch8_23983258`  [❌ crashed]
*job 23983258 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 00:00:34.501732707 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	253m52.604s
user	615m9.011s
sys	166m44.680s
```

#### `exp7_1_sit_xl_256_patch16_23984266`  [❌ crashed]
*job 23984266 • idun-07-08 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 08:53:45.936101779 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	691m0.114s
user	1274m12.928s
sys	361m44.731s
```

#### `exp7_sit_xl_128_patch8_23984267`  [❌ crashed]
*job 23984267 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 09:58:18.543157453 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	596m32.800s
user	866m4.917s
sys	213m36.524s
```

#### `exp7_sit_s_128_patch8_23987609`  [❌ crashed]
*job 23987609 • idun-06-07 • A100 • epoch 500/500 • 164KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 10:27:49.365873948 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	183m46.292s
user	540m28.016s
sys	153m24.189s
```

#### `exp7_sit_b_128_patch8_23987610`  [❌ crashed]
*job 23987610 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 11:22:36.724677251 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	213m50.673s
user	558m0.143s
sys	168m17.943s
```

#### `exp7_sit_l_128_patch8_23987611`  [❌ crashed]
*job 23987611 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 14:38:52.998854119 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	359m49.537s
user	682m45.118s
sys	193m18.033s
```

#### `exp7_sit_xl_128_patch8_23987612`  [❌ crashed]
*job 23987612 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 17:30:46.373138228 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	526m20.493s
user	816m12.553s
sys	204m15.002s
```

#### `exp7_sit_b_128_patch8_2000_23989812`  [❌ crashed]
*job 23989812 • idun-07-10 • A100 • epoch 2000/2000 • 603KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W131 08:37:36.884972023 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	1143m45.130s
user	3518m58.784s
sys	820m35.176s
```

#### `exp7_sit_b_128_patch8_23989813`  [❌ crashed]
*job 23989813 • idun-07-09 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W130 19:42:46.954818782 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	290m8.472s
user	880m58.693s
sys	199m32.531s
```

### exp8 (1 jobs)

*Status: ❌ crashed=1*

EMA baseline at 128×160.

#### `exp8_pixel_bravo_ema_23991162`  [❌ crashed]
*job 23991162 • idun-07-10 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	666m49.122s
user	1310m56.385s
sys	220m6.420s
```

### exp9 (11 jobs)

*Status: ⚠️ truncated=8 ❌ crashed=1 ✅ completed=1 💥 oom_killed=1*

First LDM bravo runs (4x/8x latent) — superseded by exp22 in bravo_latent.

#### `exp9_ldm_8x_bravo_23997506`  [⚠️ truncated]
*job 23997506 • idun-06-03 • A100 • epoch 500/500 • 92KB log*


#### `exp9_ldm_4x_bravo_23997507`  [⚠️ truncated]
*job 23997507 • idun-06-03 • A100 • epoch 500/500 • 93KB log*


#### `exp9_0_ldm_8x_bravo_small_23997680`  [⚠️ truncated]
*job 23997680 • idun-01-04 • H100 • epoch 100/100 • 30KB log*


#### `exp9_0_ldm_8x_bravo_small_23997808`  [❌ crashed]
*job 23997808 • idun-06-07 • A100 • epoch 100/100 • 31KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[W201 21:15:53.130025056 AllocatorCo
```

#### `exp9_1_ldm_4x_bravo_24036417`  [⚠️ truncated]
*job 24036417 • idun-07-08 • A100 • epoch 354/500 • 69KB log*


#### `exp9_ldm_8x_bravo_24039867`  [⚠️ truncated]
*job 24039867 • idun-06-01 • A100 • epoch 335/500 • 5284KB log*


#### `exp9_ldm_4x_bravo_24039868`  [⚠️ truncated]
*job 24039868 • idun-01-04 • H100 • epoch 302/500 • 4452KB log*


#### `exp9_0_ldm_8x_bravo_small_24061951`  [⚠️ truncated]
*job 24061951 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 38KB log*


#### `exp9_0_ldm_8x_bravo_small_24062097`  [⚠️ truncated]
*job 24062097 • idun-06-02 • A100 • epoch 170/500 • chain 0/20 • 50KB log*


#### `exp9_0_ldm_8x_bravo_small_24062563`  [✅ completed]
*job 24062563 • idun-06-01 • A100 • 7.58h training • epoch 500/500 • chain 0/20 • 119KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 3.232 • MS-SSIM 0.9997 • PSNR 55.21 dB • LPIPS 0.0014

#### `exp9_ldm_8x_bravo_24063647`  [💥 oom_killed]
*job 24063647 • idun-06-03 • A100 • 38.24h training • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/serialization.py", line 1864, in restore_location
    return default_restore_location(storage, str(map_location))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/serializat
```

### exp10 (5 jobs)

*Status: ❌ crashed=4 💥 oom_killed=1*

DC-AE SiT runs — abandoned (DC-AE branch was dropped). 3 OOM cases here.

#### `exp10_1_sit_dcae_8x8_23998878`  [💥 oom_killed]
*job 23998878 • idun-06-02 • A100 • epoch 500/500 • 97KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	1900m19.037s
user	1326m19.930s
sys	578m11.133s
[2026-02-03T19:39:34.813] error: Detected 1 oom_kill event in StepId=23998878.batch. Some of the step tasks have been OOM Killed.
```

#### `exp10_2_sit_dcae_4x4_23998879`  [❌ crashed]
*job 23998879 • idun-06-02 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (2560) at non-singleton dimension 1

Set the environment variable HY
```

#### `exp10_3_sit_dcae_2x2_23998880`  [❌ crashed]
*job 23998880 • idun-06-02 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (640) at non-singleton dimension 1

Set the environment variable HYD
```

#### `exp10_2_sit_dcae_4x4_24039865`  [❌ crashed]
*job 24039865 • idun-06-01 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (2560) at non-singleton dimension 1

Set the environment variable HYDRA_FULL_E
```

#### `exp10_3_sit_dcae_2x2_24039866`  [❌ crashed]
*job 24039866 • idun-06-01 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (640) at non-singleton dimension 1

Set the environment variable HYDRA_FULL_ER
```

### exp11 (4 jobs)

*Status: ❌ crashed=2 💥 oom_killed=2*

S2D bravo at 128/256 — abandoned.

#### `exp11_s2d_pixel_bravo_24031954`  [❌ crashed]
*job 24031954 • idun-07-10 • A100 • epoch 500/500 • 227KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1186, in train
    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_ms
```

#### `exp11_1_s2d_pixel_bravo_24031955`  [❌ crashed]
*job 24031955 • idun-06-02 • A100 • epoch 64/500 • 64KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 99, in add_event
    self.event_writer.add_event(event)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 117, in add_event
    self._async_writer.write(event.SerializeToString())
  File "/cluster/home/modestas/
```

#### `exp11_1_s2d_pixel_bravo_24039873`  [💥 oom_killed]
*job 24039873 • idun-01-03 • H100 • epoch 3/500 • 8KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	16m41.222s
user	11m6.118s
sys	4m22.979s
[2026-02-13T03:50:42.532] error: Detected 1 oom_kill event in StepId=24039873.batch. Some of the step tasks have been OOM Killed.
```

#### `exp11_s2d_pixel_bravo_24039874`  [💥 oom_killed]
*job 24039874 • idun-09-16 • A100 • epoch 4/500 • 15KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	18m34.137s
user	16m40.842s
sys	1m27.822s
[2026-02-13T00:29:46.878] error: Detected 1 oom_kill event in StepId=24039874.batch. Some of the step tasks have been OOM Killed.
```

### exp12 (4 jobs)

*Status: 💥 oom_killed=3 ❌ crashed=1*

Wavelet bravo at 128/256 — superseded by exp26 WDM.

#### `exp12_wavelet_pixel_bravo_24031956`  [💥 oom_killed]
*job 24031956 • idun-07-07 • A100 • epoch 49/500 • 55KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/unified.py", line 975, in log_generated_samples
    log_generated_samples(self, samples, epoch, tag, nrow, num_slices)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/unified_visualization.py", line 164, in log_generated_samples
    _log_generated_samples_3d(metrics, samples, epoch, tag, num_slices)
  File "/cluster/work/m
```

#### `exp12_1_wavelet_pixel_bravo_24031957`  [❌ crashed]
*job 24031957 • idun-06-07 • A100 • epoch 500/500 • 242KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1186, in train
    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_ms
```

#### `exp12_1_wavelet_pixel_bravo_24039875`  [💥 oom_killed]
*job 24039875 • idun-01-03 • H100 • epoch 210/500 • 182KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	1006m44.493s
user	674m26.029s
sys	384m3.159s
[2026-02-13T20:20:46.722] error: Detected 1 oom_kill event in StepId=24039875.batch. Some of the step tasks have been OOM Killed.
```

#### `exp12_wavelet_pixel_bravo_24039876`  [💥 oom_killed]
*job 24039876 • idun-09-16 • A100 • epoch 10/500 • 17KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	45m17.717s
user	40m47.783s
sys	3m50.149s
[2026-02-13T00:57:01.381] error: Detected 1 oom_kill event in StepId=24039876.batch. Some of the step tasks have been OOM Killed.
```

### OTHER (619 jobs)

*Status: 🔗 chained=414 ✅ completed=115 ❌ crashed=35 💥 oom_killed=32 ⚠️ truncated=23*

#### `24039895_24039895`  [🔗 chained]
*job 24039895 • idun-09-16 • A100 • epoch 7/100 • chain 0/20 • 11KB log*


#### `24039896_24039896`  [🔗 chained]
*job 24039896 • idun-09-16 • A100 • epoch 15/100 • chain 1/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039899_24039899`  [🔗 chained]
*job 24039899 • idun-09-16 • A100 • epoch 21/100 • chain 2/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039900_24039900`  [🔗 chained]
*job 24039900 • idun-09-16 • A100 • epoch 29/100 • chain 3/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039902_24039902`  [🔗 chained]
*job 24039902 • idun-09-16 • A100 • epoch 37/100 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039904_24039904`  [🔗 chained]
*job 24039904 • idun-09-16 • A100 • epoch 44/100 • chain 5/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039907_24039907`  [🔗 chained]
*job 24039907 • idun-07-07 • A100 • epoch 52/100 • chain 6/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039908_24039908`  [🔗 chained]
*job 24039908 • idun-09-16 • A100 • epoch 60/100 • chain 7/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039909_24039909`  [🔗 chained]
*job 24039909 • idun-09-16 • A100 • epoch 68/100 • chain 8/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039911_24039911`  [🔗 chained]
*job 24039911 • idun-09-16 • A100 • epoch 75/100 • chain 9/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039915_24039915`  [🔗 chained]
*job 24039915 • idun-09-16 • A100 • epoch 83/100 • chain 10/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039922_24039922`  [🔗 chained]
*job 24039922 • idun-07-07 • A100 • epoch 91/100 • chain 11/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039936_24039936`  [🔗 chained]
*job 24039936 • idun-09-16 • A100 • epoch 99/100 • chain 12/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039971_24039971`  [✅ completed]
*job 24039971 • idun-09-16 • A100 • 0.11h training • epoch 100/100 • chain 13/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24042396_24042396`  [🔗 chained]
*job 24042396 • idun-09-16 • A100 • epoch 405/500 • chain 0/20 • 69KB log*


#### `24042475_24042475`  [🔗 chained]
*job 24042475 • idun-07-05 • A100 • epoch 489/500 • chain 0/20 • 110KB log*


#### `24042655_24042655`  [🔗 chained]
*job 24042655 • idun-06-07 • A100 • epoch 374/500 • chain 0/20 • 127KB log*


#### `24042656_24042656`  [🔗 chained]
*job 24042656 • idun-06-07 • A100 • epoch 18/500 • chain 0/20 • 11KB log*


#### `24042660_24042660`  [🔗 chained]
*job 24042660 • idun-07-09 • A100 • epoch 488/500 • chain 0/20 • 83KB log*


#### `24042663_24042663`  [❌ crashed]
*job 24042663 • idun-06-07 • A100 • 2.05h training • epoch 500/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260214-163133/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042671_24042671`  [❌ crashed]
*job 24042671 • idun-06-07 • A100 • 0.26h training • epoch 500/500 • chain 1/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp11_2_20260214-203209/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042824_24042824`  [❌ crashed]
*job 24042824 • idun-06-04 • A100 • 4.17h training • epoch 500/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_20260215-023826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surf
```

#### `24042825_24042825`  [🔗 chained]
*job 24042825 • idun-06-07 • A100 • epoch 35/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24042844_24042844`  [❌ crashed]
*job 24042844 • idun-06-07 • A100 • 0.27h training • epoch 500/500 • chain 1/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp13_dit_8x_bravo_20260215-024826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042958_24042958`  [✅ completed]
*job 24042958 • idun-06-07 • A100 • 10.88h training • epoch 500/500 • chain 0/20 • 88KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001044 • MS-SSIM 0.8980 • PSNR 30.31 dB • LPIPS 0.6941 • FID 76.67 • KID 0.0576 ± 0.0089 • CMMD 0.3962
  - **latest** ckpt (26 samples): MSE 0.000989 • MS-SSIM 0.9043 • PSNR 30.64 dB • LPIPS 0.6711 • FID 78.37 • KID 0.0591 ± 0.0077 • CMMD 0.3767

#### `24042959_24042959`  [🔗 chained]
*job 24042959 • idun-07-09 • A100 • epoch 483/500 • chain 0/20 • 110KB log*


#### `24042968_24042968`  [❌ crashed]
*job 24042968 • idun-07-08 • A100 • 11.86h training • epoch 500/500 • chain 0/20 • 89KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.09429 • MS-SSIM 0.9455 • PSNR 32.14 dB • LPIPS 0.3282
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W216 06:31:15.521623016 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	715m47.475s
user	697m25.791s
sys	121m3.781s
```

#### `24043083_24043083`  [🔗 chained]
*job 24043083 • idun-07-10 • A100 • epoch 234/500 • chain 0/20 • 54KB log*


#### `24043086_24043086`  [🔗 chained]
*job 24043086 • idun-07-10 • A100 • epoch 358/500 • chain 0/20 • 64KB log*


#### `24043087_24043087`  [🔗 chained]
*job 24043087 • idun-06-07 • A100 • epoch 52/500 • chain 2/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24043097_24043097`  [✅ completed]
*job 24043097 • idun-06-04 • A100 • 0.39h training • epoch 500/500 • chain 1/20 • 15KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.06197 • MS-SSIM 0.6490 • PSNR 23.16 dB • LPIPS 1.7949 • FID 277.37 • KID 0.3363 ± 0.0188 • CMMD 0.6074
  - **latest** ckpt (26 samples): MSE 0.06664 • MS-SSIM 0.7233 • PSNR 25.53 dB • LPIPS 1.7187 • FID 282.13 • KID 0.3426 ± 0.0202 • CMMD 0.6064
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_3_20260215-175625/checkpoint_latest.pt`

#### `24043682_24043682`  [🔗 chained]
*job 24043682 • idun-07-10 • A100 • epoch 464/500 • chain 1/20 • 558KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_3_sit_s_128_patch4_20260216-020438/checkpoint_latest.pt`

#### `24043701_24043701`  [💥 oom_killed]
*job 24043701 • idun-07-10 • A100 • 4.95h training • epoch 500/500 • chain 1/20 • 352KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001115 • MS-SSIM 0.8986 • PSNR 30.10 dB • LPIPS 0.7983 • FID 71.11 • KID 0.0443 ± 0.0050 • CMMD 0.3321
  - **latest** ckpt (26 samples): MSE 0.001632 • MS-SSIM 0.8403 • PSNR 28.19 dB • LPIPS 0.7647
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_5_20260216-022108/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	305m34.660s
user	257m48.428s
sys	43m24.249s
[2026-02-16T19:24:47.568] error: Detected 2 oom_kill events in StepId=24043701.batch. Some of the step tasks have been OOM Killed.
```

#### `24043721_24043721`  [🔗 chained]
*job 24043721 • idun-06-07 • A100 • epoch 70/500 • chain 3/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24044737_24044737`  [🔗 chained]
*job 24044737 • idun-09-16 • A100 • epoch 199/500 • chain 0/20 • 483KB log*


#### `24045125_24045125`  [🔗 chained]
*job 24045125 • idun-07-09 • A100 • epoch 115/500 • chain 0/20 • 127KB log*


#### `24045647_24045647`  [✅ completed]
*job 24045647 • idun-01-03 • H100 • 1.1h training • epoch 500/500 • chain 2/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002765 • MS-SSIM 0.9342 • PSNR 32.06 dB • LPIPS 0.6456 • FID 126.78 • KID 0.1514 ± 0.0092 • CMMD 0.4015
  - **latest** ckpt (26 samples): MSE 0.00399 • MS-SSIM 0.9511 • PSNR 32.83 dB • LPIPS 0.5543 • FID 115.82 • KID 0.1340 ± 0.0066 • CMMD 0.3700
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_3_sit_s_128_patch4_20260216-020438/checkpoint_latest.pt`

#### `24045747_24045747`  [🔗 chained]
*job 24045747 • idun-06-07 • A100 • epoch 86/500 • chain 4/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24045963_24045963`  [🔗 chained]
*job 24045963 • idun-09-16 • A100 • epoch 397/500 • chain 1/20 • 197KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_6_20260216-161013/checkpoint_latest.pt`

#### `24047474_24047474`  [🔗 chained]
*job 24047474 • idun-01-04 • H100 • epoch 311/500 • chain 1/20 • 47KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt`

#### `24047482_24047482`  [✅ completed]
*job 24047482 • idun-07-06 • A100 • 6.2h training • epoch 500/500 • chain 2/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00105 • MS-SSIM 0.8980 • PSNR 30.19 dB • LPIPS 0.6533 • FID 59.49 • KID 0.0385 ± 0.0059 • CMMD 0.3065
  - **latest** ckpt (26 samples): MSE 0.001116 • MS-SSIM 0.8908 • PSNR 30.03 dB • LPIPS 0.6292 • FID 60.42 • KID 0.0388 ± 0.0039 • CMMD 0.3124
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_6_20260216-161013/checkpoint_latest.pt`

#### `24051502_24051502`  [🔗 chained]
*job 24051502 • idun-07-10 • A100 • epoch 100/500 • chain 5/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24052178_24052178`  [✅ completed]
*job 24052178 • idun-01-04 • H100 • 11.49h training • epoch 500/500 • chain 2/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt`

#### `24052709_24052709`  [🔗 chained]
*job 24052709 • idun-07-08 • A100 • epoch 114/500 • chain 6/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24053109_24053109`  [🔗 chained]
*job 24053109 • idun-07-10 • A100 • epoch 313/2000 • chain 0/20 • 104KB log*


#### `24053112_24053112`  [⚠️ truncated]
*job 24053112 • idun-07-08 • A100 • epoch 390/2000 • chain 0/20 • 108KB log*


#### `24055687_24055687`  [🔗 chained]
*job 24055687 • idun-01-04 • H100 • epoch 490/500 • chain 0/20 • 107KB log*


#### `24055689_24055689`  [🔗 chained]
*job 24055689 • idun-01-04 • H100 • epoch 176/500 • chain 0/20 • 51KB log*


#### `24055690_24055690`  [🔗 chained]
*job 24055690 • idun-07-04 • A100 • epoch 291/500 • chain 0/20 • 65KB log*


#### `24056427_24056427`  [✅ completed]
*job 24056427 • idun-07-09 • A100 • 3.77h training • epoch 500/500 • chain 0/20 • 146KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01576 • MS-SSIM 0.7086 • PSNR 27.10 dB • LPIPS 1.7616 • FID 273.99 • KID 0.3872 ± 0.0076 • CMMD 0.6156
  - **latest** ckpt (26 samples): MSE 0.01507 • MS-SSIM 0.7001 • PSNR 26.98 dB • LPIPS 1.7743 • FID 264.66 • KID 0.3702 ± 0.0086 • CMMD 0.6171

#### `24056428_24056428`  [✅ completed]
*job 24056428 • idun-07-10 • A100 • 12.66h training • epoch 500/500 • chain 0/20 • 117KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007365 • MS-SSIM 0.9459 • PSNR 32.78 dB • LPIPS 0.6684 • FID 132.46 • KID 0.1628 ± 0.0091 • CMMD 0.4628
  - **latest** ckpt (26 samples): MSE 0.005478 • MS-SSIM 0.9252 • PSNR 31.31 dB • LPIPS 0.4983 • FID 98.14 • KID 0.1178 ± 0.0069 • CMMD 0.2885

#### `24056484_24056484`  [🔗 chained]
*job 24056484 • idun-06-04 • A100 • epoch 131/500 • chain 7/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24056491_24056491`  [🔗 chained]
*job 24056491 • idun-06-04 • A100 • epoch 687/2000 • chain 1/20 • 102KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24056742_24056742`  [✅ completed]
*job 24056742 • idun-07-04 • A100 • 8.55h training • epoch 500/500 • chain 1/20 • 45KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_pixel_seg_improved_20260218-183124/checkpoint_latest.pt`

#### `24056743_24056743`  [🔗 chained]
*job 24056743 • idun-07-08 • A100 • epoch 281/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24056744_24056744`  [✅ completed]
*job 24056744 • idun-07-08 • A100 • 0.39h training • epoch 500/500 • chain 1/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_pixel_seg_20260218-183155/checkpoint_latest.pt`

#### `24060136_24060136`  [🔗 chained]
*job 24060136 • idun-06-05 • A100 • epoch 149/500 • chain 8/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24060137_24060137`  [🔗 chained]
*job 24060137 • idun-07-10 • A100 • epoch 997/2000 • chain 2/20 • 93KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24060138_24060138`  [🔗 chained]
*job 24060138 • idun-06-05 • A100 • epoch 1/500 • chain 0/20 • 4KB log*


#### `24060139_24060139`  [🔗 chained]
*job 24060139 • idun-06-06 • A100 • epoch 25/500 • chain 0/20 • 14KB log*


#### `24060169_24060169`  [❌ crashed]
*job 24060169 • idun-07-08 • A100 • epoch 385/500 • chain 2/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 690, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/base_trainer.py", line 794, in train
    self._handle_checkpoints(epoch, merged_metrics)
  Fil
```

#### `24060846_24060846`  [🔗 chained]
*job 24060846 • idun-07-10 • A100 • epoch 1175/2000 • chain 3/20 • 86KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24060847_24060847`  [🔗 chained]
*job 24060847 • idun-06-06 • A100 • epoch 166/500 • chain 9/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24060855_24060855`  [🔗 chained]
*job 24060855 • idun-07-08 • A100 • epoch 1/500 • chain 1/20 • 4KB log*


#### `24060856_24060856`  [🔗 chained]
*job 24060856 • idun-07-08 • A100 • epoch 25/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24061044_24061044`  [🔗 chained]
*job 24061044 • idun-06-01 • A100 • epoch 426/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24061288_24061288`  [🔗 chained]
*job 24061288 • idun-06-01 • A100 • epoch 163/500 • chain 0/20 • 57KB log*


#### `24061290_24061290`  [🔗 chained]
*job 24061290 • idun-01-03 • H100 • epoch 162/500 • chain 0/20 • 53KB log*


#### `24061291_24061291`  [✅ completed]
*job 24061291 • idun-06-04 • A100 • 10.58h training • epoch 500/500 • chain 0/20 • 144KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003976 • MS-SSIM 0.9285 • PSNR 31.00 dB • LPIPS 0.7093 • FID 161.63 • KID 0.2054 ± 0.0134 • CMMD 0.3920
  - **latest** ckpt (26 samples): MSE 0.005496 • MS-SSIM 0.9285 • PSNR 31.13 dB • LPIPS 0.7534 • FID 167.24 • KID 0.2119 ± 0.0118 • CMMD 0.3988

#### `24061292_24061292`  [🔗 chained]
*job 24061292 • idun-06-03 • A100 • epoch 404/500 • chain 0/20 • 120KB log*


#### `24061494_24061494`  [🔗 chained]
*job 24061494 • idun-07-09 • A100 • epoch 159/500 • chain 0/20 • 52KB log*


#### `24061806_24061806`  [🔗 chained]
*job 24061806 • idun-07-08 • A100 • epoch 1488/2000 • chain 4/20 • 66KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24061807_24061807`  [🔗 chained]
*job 24061807 • idun-06-03 • A100 • epoch 183/500 • chain 10/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24061811_24061811`  [🔗 chained]
*job 24061811 • idun-06-03 • A100 • epoch 2/500 • chain 2/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_0_hdit_s_p2_256_20260219-205831/checkpoint_latest.pt`

#### `24061813_24061813`  [🔗 chained]
*job 24061813 • idun-06-01 • A100 • epoch 50/500 • chain 2/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24061859_24061859`  [✅ completed]
*job 24061859 • idun-06-04 • A100 • 7.87h training • epoch 500/500 • chain 4/20 • 27KB log*

**Final test metrics:**
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24061920_24061920`  [🔗 chained]
*job 24061920 • idun-08-01 • H100 • epoch 407/500 • chain 1/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_l_256_patch8_20260220-131107/checkpoint_latest.pt`

#### `24061922_24061922`  [🔗 chained]
*job 24061922 • idun-08-01 • H100 • epoch 322/500 • chain 1/20 • 43KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24061931_24061931`  [✅ completed]
*job 24061931 • idun-08-01 • H100 • 1.94h training • epoch 500/500 • chain 1/20 • 27KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.005235 • MS-SSIM 0.9316 • PSNR 31.53 dB • LPIPS 1.3710 • FID 154.84 • KID 0.1896 ± 0.0116 • CMMD 0.3878
  - **latest** ckpt (26 samples): MSE 0.01005 • MS-SSIM 0.9220 • PSNR 30.64 dB • LPIPS 0.5395 • FID 152.29 • KID 0.1937 ± 0.0107 • CMMD 0.3421
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_3_hdit_xl_p8_256_20260220-141245/checkpoint_latest.pt`

#### `24061961_24061961`  [🔗 chained]
*job 24061961 • idun-06-03 • A100 • epoch 342/500 • chain 1/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp18_uvit_l_p8_256_20260220-170256/checkpoint_latest.pt`

#### `24062040_24062040`  [🔗 chained]
*job 24062040 • idun-07-10 • A100 • epoch 1798/2000 • chain 5/20 • 60KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24062070_24062070`  [❌ crashed]
*job 24062070 • idun-06-01 • A100 • 11.36h training • epoch 199/500 • chain 11/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
       ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'PixelSpace' object has no attribute 'needs_decode'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[W222 00:43:50.819646877 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA
```

#### `24062071_24062071`  [❌ crashed]
*job 24062071 • idun-06-01 • A100 • 3.2h training • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_0_hdit_s_p2_256_20260219-205831/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W221 16:39:03.520194663 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	199m51.125s
user	147m16.880s
sys	51m7.182s
```

#### `24062089_24062089`  [🔗 chained]
*job 24062089 • idun-07-08 • A100 • epoch 73/500 • chain 3/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24062511_24062511`  [🔗 chained]
*job 24062511 • idun-08-01 • H100 • epoch 458/500 • chain 0/20 • 87KB log*


#### `24062564_24062564`  [🔗 chained]
*job 24062564 • idun-06-01 • A100 • epoch 351/500 • chain 0/20 • 94KB log*


#### `24062597_24062597`  [❌ crashed]
*job 24062597 • idun-06-03 • A100 • 8.65h training • epoch 500/500 • chain 0/20 • 101KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 18.29 • MS-SSIM 0.9987 • PSNR 50.68 dB • LPIPS 0.0052
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 08:28:53.147816293 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	525m30.097s
user	749m21.804s
sys	91m41.577s
```

#### `24062598_24062598`  [❌ crashed]
*job 24062598 • idun-06-03 • A100 • 9.85h training • epoch 500/500 • chain 0/20 • 94KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.642 • MS-SSIM 0.9997 • PSNR 55.49 dB • LPIPS 0.0014
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 09:39:54.723334518 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	596m32.186s
user	809m43.339s
sys	86m35.424s
```

#### `24062820_24062820`  [✅ completed]
*job 24062820 • idun-06-01 • A100 • 6.91h training • epoch 500/500 • chain 2/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00473 • MS-SSIM 0.9466 • PSNR 32.39 dB • LPIPS 0.4509 • FID 132.38 • KID 0.1440 ± 0.0095 • CMMD 0.3918
  - **latest** ckpt (26 samples): MSE 0.003225 • MS-SSIM 0.9466 • PSNR 32.17 dB • LPIPS 0.4568 • FID 131.99 • KID 0.1460 ± 0.0081 • CMMD 0.3862
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_l_256_patch8_20260220-131107/checkpoint_latest.pt`

#### `24062821_24062821`  [🔗 chained]
*job 24062821 • idun-07-08 • A100 • epoch 411/500 • chain 2/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24062848_24062848`  [🔗 chained]
*job 24062848 • idun-07-06 • A100 • epoch 436/500 • chain 0/20 • 75KB log*


#### `24062849_24062849`  [🔗 chained]
*job 24062849 • idun-09-18 • A100 • epoch 384/500 • chain 0/20 • 76KB log*


#### `24062850_24062850`  [🔗 chained]
*job 24062850 • idun-09-18 • A100 • epoch 375/500 • chain 0/20 • 76KB log*


#### `24062935_24062935`  [✅ completed]
*job 24062935 • idun-06-01 • A100 • 10.2h training • epoch 500/500 • chain 2/20 • 41KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.008652 • MS-SSIM 0.9488 • PSNR 32.94 dB • LPIPS 0.9663 • FID 149.15 • KID 0.1802 ± 0.0114 • CMMD 0.3810
  - **latest** ckpt (26 samples): MSE 0.003476 • MS-SSIM 0.9502 • PSNR 32.67 dB • LPIPS 0.7485 • FID 147.86 • KID 0.1877 ± 0.0125 • CMMD 0.3879
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp18_uvit_l_p8_256_20260220-170256/checkpoint_latest.pt`

#### `24062965_24062965`  [✅ completed]
*job 24062965 • idun-06-05 • A100 • 6.49h training • epoch 2000/2000 • chain 6/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004644 • MS-SSIM 0.9529 • PSNR 33.03 dB • LPIPS 0.5121 • FID 157.82 • KID 0.2049 ± 0.0107 • CMMD 0.3550
  - **latest** ckpt (26 samples): MSE 0.00343 • MS-SSIM 0.9479 • PSNR 32.27 dB • LPIPS 0.4261 • FID 129.36 • KID 0.1599 ± 0.0091 • CMMD 0.2721
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24062966_24062966`  [🔗 chained]
*job 24062966 • idun-07-10 • A100 • epoch 94/500 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24062979_24062979`  [✅ completed]
*job 24062979 • idun-06-03 • A100 • 1.53h training • epoch 500/500 • chain 1/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007356 • MS-SSIM 0.9663 • PSNR 34.59 dB • LPIPS 0.5287 • FID 84.13 • KID 0.0849 ± 0.0169 • CMMD 0.3426
  - **latest** ckpt (26 samples): MSE 0.02154 • MS-SSIM 0.9387 • PSNR 32.16 dB • LPIPS 0.4058 • FID 90.56 • KID 0.0990 ± 0.0173 • CMMD 0.3006
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_20260221-172156/checkpoint_latest.pt`

#### `24063022_24063022`  [✅ completed]
*job 24063022 • idun-07-05 • A100 • 1.77h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.377 • MS-SSIM 0.9278 • PSNR 33.50 dB • LPIPS 0.5723 • FID 112.82 • KID 0.1046 ± 0.0091 • CMMD 0.4834
  - **latest** ckpt (26 samples): MSE 0.5867 • MS-SSIM 0.9093 • PSNR 32.33 dB • LPIPS 0.5931 • FID 98.08 • KID 0.0847 ± 0.0069 • CMMD 0.4365
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_20260221-215953/checkpoint_latest.pt`

#### `24063070_24063070`  [❌ crashed]
*job 24063070 • idun-06-02 • A100 • 5.06h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.936 • MS-SSIM 0.9998 • PSNR 57.96 dB • LPIPS 0.0010
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260221-224417/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 18:24:08.692785036 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	312m1.101s
user	277m36.075s
sys	109m44.250s
```

#### `24063071_24063071`  [✅ completed]
*job 24063071 • idun-09-18 • A100 • 3.59h training • epoch 500/500 • chain 1/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.6159 • MS-SSIM 0.8946 • PSNR 34.96 dB • LPIPS 1.4661 • FID 305.22 • KID 0.4231 ± 0.0133 • CMMD 0.6040
  - **latest** ckpt (26 samples): MSE 0.6088 • MS-SSIM 0.9065 • PSNR 35.86 dB • LPIPS 1.4473 • FID 302.24 • KID 0.4122 ± 0.0114 • CMMD 0.6139
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_3_20260221-224417/checkpoint_latest.pt`

#### `24063079_24063079`  [✅ completed]
*job 24063079 • idun-09-18 • A100 • 4.05h training • epoch 500/500 • chain 1/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004118 • MS-SSIM 0.8509 • PSNR 30.23 dB • LPIPS 1.2873 • FID 149.50 • KID 0.1519 ± 0.0122 • CMMD 0.5199
  - **latest** ckpt (26 samples): MSE 0.003497 • MS-SSIM 0.8730 • PSNR 31.02 dB • LPIPS 1.2673 • FID 146.14 • KID 0.1463 ± 0.0126 • CMMD 0.5326
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260221-225423/checkpoint_latest.pt`

#### `24063241_24063241`  [🔗 chained]
*job 24063241 • idun-07-10 • A100 • epoch 499/500 • chain 3/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24063375_24063375`  [🔗 chained]
*job 24063375 • idun-06-01 • A100 • epoch 119/500 • chain 5/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24063576_24063576`  [🔗 chained]
*job 24063576 • idun-06-02 • A100 • epoch 106/500 • chain 0/20 • 28KB log*


#### `24063579_24063579`  [❌ crashed]
*job 24063579 • idun-06-01 • A100 • 8.04h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2911 • MS-SSIM 0.9970 • PSNR 46.86 dB • LPIPS 0.0183
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W223 07:11:28.639423918 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	487m4.098s
user	605m26.713s
sys	151m24.145s
```

#### `24063615_24063615`  [🔗 chained]
*job 24063615 • idun-07-06 • A100 • epoch 434/500 • chain 0/20 • 85KB log*


#### `24063616_24063616`  [🔗 chained]
*job 24063616 • idun-09-16 • A100 • epoch 381/500 • chain 0/20 • 66KB log*


#### `24063622_24063622`  [🔗 chained]
*job 24063622 • idun-06-03 • A100 • epoch 349/500 • chain 0/20 • 79KB log*


#### `24063623_24063623`  [🔗 chained]
*job 24063623 • idun-06-01 • A100 • epoch 121/500 • chain 0/20 • 33KB log*


#### `24063648_24063648`  [❌ crashed]
*job 24063648 • idun-01-04 • H100 • 5.8h training • epoch 500/500 • chain 0/20 • 87KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.377 • MS-SSIM 0.9954 • PSNR 43.53 dB • LPIPS 0.0190
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W223 10:19:21.504942614 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	353m44.036s
user	439m55.989s
sys	89m39.309s
```

#### `24063682_24063682`  [✅ completed]
*job 24063682 • idun-01-04 • H100 • 0.27h training • epoch 500/500 • chain 4/20 • 9KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003828 • MS-SSIM 0.9372 • PSNR 31.91 dB • LPIPS 0.5120 • FID 152.97 • KID 0.1881 ± 0.0139 • CMMD 0.3809
  - **latest** ckpt (26 samples): MSE 0.009043 • MS-SSIM 0.9507 • PSNR 33.08 dB • LPIPS 0.4149 • FID 157.10 • KID 0.1974 ± 0.0134 • CMMD 0.3905
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24063686_24063686`  [🔗 chained]
*job 24063686 • idun-01-04 • H100 • epoch 133/500 • chain 6/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24063887_24063887`  [🔗 chained]
*job 24063887 • idun-06-03 • A100 • epoch 433/500 • chain 1/20 • 58KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260222-225625/checkpoint_latest.pt`

#### `24063940_24063940`  [✅ completed]
*job 24063940 • idun-07-07 • A100 • 1.8h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004617 • MS-SSIM 0.8374 • PSNR 30.01 dB • LPIPS 1.2790 • FID 256.54 • KID 0.3115 ± 0.0133 • CMMD 0.5483
  - **latest** ckpt (26 samples): MSE 0.0045 • MS-SSIM 0.8429 • PSNR 29.95 dB • LPIPS 1.2748 • FID 243.49 • KID 0.2878 ± 0.0158 • CMMD 0.5594
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260222-233635/checkpoint_latest.pt`

#### `24063965_24063965`  [✅ completed]
*job 24063965 • idun-09-16 • A100 • 3.73h training • epoch 500/500 • chain 1/20 • 28KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.5198 • MS-SSIM 0.8889 • PSNR 31.44 dB • LPIPS 0.7354 • FID 152.36 • KID 0.1398 ± 0.0127 • CMMD 0.5845
  - **latest** ckpt (26 samples): MSE 0.4083 • MS-SSIM 0.9164 • PSNR 32.83 dB • LPIPS 0.5358 • FID 154.74 • KID 0.1432 ± 0.0128 • CMMD 0.5774
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_20260222-235610/checkpoint_latest.pt`

#### `24063991_24063991`  [✅ completed]
*job 24063991 • idun-06-03 • A100 • 5.46h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.07605 • MS-SSIM 0.9849 • PSNR 38.78 dB • LPIPS 0.2314 • FID 113.97 • KID 0.0997 ± 0.0059 • CMMD 0.5071
  - **latest** ckpt (26 samples): MSE 0.1444 • MS-SSIM 0.9730 • PSNR 35.92 dB • LPIPS 0.2615 • FID 143.60 • KID 0.1613 ± 0.0069 • CMMD 0.4406
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_20260223-002347/checkpoint_latest.pt`

#### `24064017_24064017`  [🔗 chained]
*job 24064017 • idun-01-03 • H100 • epoch 297/500 • chain 1/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24065812_24065812`  [🔗 chained]
*job 24065812 • idun-06-03 • A100 • epoch 157/500 • chain 7/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24067712_24067712`  [🔗 chained]
*job 24067712 • idun-06-05 • A100 • epoch 112/500 • chain 0/20 • 85KB log*


#### `24067738_24067738`  [✅ completed]
*job 24067738 • idun-06-02 • A100 • 2.27h training • epoch 500/500 • chain 2/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2609 • MS-SSIM 0.9968 • PSNR 46.45 dB • LPIPS 0.0203 • FID 232.16 • KID 0.2637 ± 0.0267 • CMMD 0.5815
  - **latest** ckpt (26 samples): MSE 0.8212 • MS-SSIM 0.9923 • PSNR 44.12 dB • LPIPS 0.0453 • FID 182.09 • KID 0.2008 ± 0.0301 • CMMD 0.4864
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260222-225625/checkpoint_latest.pt`

#### `24067773_24067773`  [🔗 chained]
*job 24067773 • idun-01-03 • H100 • epoch 473/500 • chain 2/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24067802_24067802`  [🔗 chained]
*job 24067802 • idun-06-01 • A100 • epoch 190/500 • chain 0/20 • 36KB log*


#### `24067803_24067803`  [🔗 chained]
*job 24067803 • idun-06-02 • A100 • epoch 190/500 • chain 0/20 • 35KB log*


#### `24067804_24067804`  [🔗 chained]
*job 24067804 • idun-06-02 • A100 • epoch 199/500 • chain 0/20 • 39KB log*


#### `24067805_24067805`  [🔗 chained]
*job 24067805 • idun-06-02 • A100 • epoch 289/500 • chain 0/20 • 51KB log*


#### `24067806_24067806`  [🔗 chained]
*job 24067806 • idun-01-03 • H100 • epoch 257/500 • chain 0/20 • 45KB log*


#### `24067807_24067807`  [🔗 chained]
*job 24067807 • idun-07-08 • A100 • epoch 174/500 • chain 0/20 • 34KB log*


#### `24067808_24067808`  [⚠️ truncated]
*job 24067808 • idun-07-09 • A100 • chain 0/20 • 1KB log*


#### `24070764_24070764`  [🔗 chained]
*job 24070764 • idun-07-08 • A100 • epoch 178/500 • chain 8/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24070899_24070899`  [🔗 chained]
*job 24070899 • idun-06-01 • A100 • epoch 224/500 • chain 1/20 • 71KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_1_pixel_seg_aux_bin_20260224-000544/checkpoint_latest.pt`

#### `24070923_24070923`  [🔗 chained]
*job 24070923 • idun-06-02 • A100 • epoch 190/500 • chain 0/20 • 36KB log*


#### `24070925_24070925`  [✅ completed]
*job 24070925 • idun-06-02 • A100 • 2.72h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.07344 • MS-SSIM 0.9920 • PSNR 41.04 dB • LPIPS 0.1385 • FID 295.98 • KID 0.4118 ± 0.0165 • CMMD 0.6002
  - **latest** ckpt (26 samples): MSE 0.06805 • MS-SSIM 0.9903 • PSNR 39.89 dB • LPIPS 0.1525 • FID 306.39 • KID 0.4247 ± 0.0167 • CMMD 0.6081
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24071122_24071122`  [🔗 chained]
*job 24071122 • idun-07-09 • A100 • epoch 360/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_0_20260224-015924/checkpoint_latest.pt`

#### `24071123_24071123`  [🔗 chained]
*job 24071123 • idun-01-03 • H100 • epoch 465/500 • chain 1/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_paper_20260224-015924/checkpoint_latest.pt`

#### `24071215_24071215`  [❌ crashed]
*job 24071215 • idun-06-02 • A100 • 0.01h training • chain 0/20 • 4KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	2m23.684s
user	1m31.990s
sys	0m31.447s
```

#### `24071216_24071216`  [✅ completed]
*job 24071216 • idun-06-02 • A100 • 10.59h training • epoch 500/500 • chain 0/20 • 122KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.0756 • MS-SSIM 0.9939 • PSNR 41.11 dB • LPIPS 0.1051 • FID 117.07 • KID 0.1017 ± 0.0049 • CMMD 0.5555
  - **latest** ckpt (26 samples): MSE 0.1331 • MS-SSIM 0.9800 • PSNR 38.07 dB • LPIPS 0.1776 • FID 95.12 • KID 0.0724 ± 0.0054 • CMMD 0.4502

#### `24071218_24071218`  [🔗 chained]
*job 24071218 • idun-06-05 • A100 • epoch 395/500 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_1_20260224-022448/checkpoint_latest.pt`

#### `24071219_24071219`  [✅ completed]
*job 24071219 • idun-07-08 • A100 • 10.67h training • epoch 500/500 • chain 1/20 • 42KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3851 • MS-SSIM 0.9582 • PSNR 35.94 dB • LPIPS 0.2392 • FID 68.92 • KID 0.0453 ± 0.0052 • CMMD 0.2383
  - **latest** ckpt (26 samples): MSE 0.2968 • MS-SSIM 0.9651 • PSNR 36.67 dB • LPIPS 0.1924 • FID 67.32 • KID 0.0437 ± 0.0051 • CMMD 0.2351
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_2_20260224-022548/checkpoint_latest.pt`

#### `24071365_24071365`  [🔗 chained]
*job 24071365 • idun-07-08 • A100 • epoch 424/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_3_20260224-034637/checkpoint_latest.pt`

#### `24071590_24071590`  [🔗 chained]
*job 24071590 • idun-06-05 • A100 • epoch 333/500 • chain 0/20 • 248KB log*


#### `24072508_24072508`  [✅ completed]
*job 24072508 • idun-07-08 • A100 • 8.76h training • epoch 500/500 • chain 0/20 • 90KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.518 • MS-SSIM 0.9918 • PSNR 34.89 dB • LPIPS 0.0665 • FID 75.13 • KID 0.0712 ± 0.0129 • CMMD 0.3318
  - **latest** ckpt (26 samples): MSE 1.65 • MS-SSIM 0.9926 • PSNR 35.68 dB • LPIPS 0.0623 • FID 59.94 • KID 0.0506 ± 0.0098 • CMMD 0.3079

#### `24072517_24072517`  [🔗 chained]
*job 24072517 • idun-09-16 • A100 • epoch 245/500 • chain 0/20 • 188KB log*


#### `24075917_24075917`  [🔗 chained]
*job 24075917 • idun-06-02 • A100 • epoch 372/500 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_4_20260224-092339/checkpoint_latest.pt`

#### `24076104_24076104`  [🔗 chained]
*job 24076104 • idun-06-01 • A100 • epoch 49/500 • chain 0/20 • 20KB log*


#### `24076153_24076153`  [🔗 chained]
*job 24076153 • idun-07-08 • A100 • epoch 327/500 • chain 2/20 • 57KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_1_pixel_seg_aux_bin_20260224-000544/checkpoint_latest.pt`

#### `24076158_24076158`  [🔗 chained]
*job 24076158 • idun-06-05 • A100 • epoch 378/500 • chain 1/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_5_20260224-142309/checkpoint_latest.pt`

#### `24076159_24076159`  [✅ completed]
*job 24076159 • idun-07-09 • A100 • 9.7h training • epoch 500/500 • chain 2/20 • 32KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3549 • MS-SSIM 0.9663 • PSNR 36.29 dB • LPIPS 0.2495 • FID 128.82 • KID 0.1176 ± 0.0095 • CMMD 0.4297
  - **latest** ckpt (26 samples): MSE 0.2181 • MS-SSIM 0.9787 • PSNR 38.65 dB • LPIPS 0.1673 • FID 131.75 • KID 0.1200 ± 0.0138 • CMMD 0.4457
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_0_20260224-015924/checkpoint_latest.pt`

#### `24076160_24076160`  [✅ completed]
*job 24076160 • idun-06-02 • A100 • 2.26h training • epoch 500/500 • chain 2/20 • 14KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00351 • MS-SSIM 0.9264 • PSNR 31.45 dB • LPIPS 0.6003 • FID 87.01 • KID 0.0544 ± 0.0052 • CMMD 0.3535
  - **latest** ckpt (26 samples): MSE 0.003405 • MS-SSIM 0.9307 • PSNR 31.99 dB • LPIPS 0.4647 • FID 81.22 • KID 0.0446 ± 0.0047 • CMMD 0.2846
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_paper_20260224-015924/checkpoint_latest.pt`

#### `24076418_24076418`  [✅ completed]
*job 24076418 • idun-06-01 • A100 • 6.35h training • epoch 500/500 • chain 2/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2198 • MS-SSIM 0.9956 • PSNR 44.83 dB • LPIPS 0.0690 • FID 235.69 • KID 0.2532 ± 0.0148 • CMMD 0.5038
  - **latest** ckpt (26 samples): MSE 0.2166 • MS-SSIM 0.9960 • PSNR 46.04 dB • LPIPS 0.0603 • FID 235.27 • KID 0.2484 ± 0.0164 • CMMD 0.4904
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_1_20260224-022448/checkpoint_latest.pt`

#### `24076724_24076724`  [✅ completed]
*job 24076724 • idun-07-09 • A100 • 5.37h training • epoch 500/500 • chain 2/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3201 • MS-SSIM 0.9623 • PSNR 35.46 dB • LPIPS 0.2349 • FID 107.45 • KID 0.0788 ± 0.0065 • CMMD 0.3993
  - **latest** ckpt (26 samples): MSE 0.2795 • MS-SSIM 0.9703 • PSNR 37.63 dB • LPIPS 0.2013 • FID 123.96 • KID 0.0999 ± 0.0085 • CMMD 0.3737
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_3_20260224-034637/checkpoint_latest.pt`

#### `24076868_24076868`  [⚠️ truncated]
*job 24076868 • idun-06-05 • A100 • epoch 438/500 • chain 1/20 • 79KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_pixel_seg_aux_bin_20260225-005914/checkpoint_latest.pt`

#### `24077301_24077301`  [⚠️ truncated]
*job 24077301 • idun-06-05 • A100 • epoch 306/500 • chain 1/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_pixel_seg_aux_bin_20260225-040703/checkpoint_latest.pt`

#### `24077858_24077858`  [⚠️ truncated]
*job 24077858 • idun-06-01 • A100 • chain 0/20 • 1KB log*


#### `24081617_24081617`  [⚠️ truncated]
*job 24081617 • idun-06-01 • A100 • epoch 55/500 • chain 1/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_4_hdit_s_p4_256_20260225-111541/checkpoint_latest.pt`

#### `24082354_24082354`  [✅ completed]
*job 24082354 • idun-07-09 • A100 • 10.57h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.341 • MS-SSIM 0.9922 • PSNR 44.11 dB • LPIPS 0.1116 • FID 176.51 • KID 0.1888 ± 0.0271 • CMMD 0.3985
  - **latest** ckpt (26 samples): MSE 1.618 • MS-SSIM 0.9948 • PSNR 46.36 dB • LPIPS 0.0723 • FID 164.34 • KID 0.1784 ± 0.0280 • CMMD 0.3946

#### `24082355_24082355`  [✅ completed]
*job 24082355 • idun-06-05 • A100 • 9.68h training • epoch 500/500 • chain 0/20 • 92KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.1575 • MS-SSIM 0.9951 • PSNR 47.77 dB • LPIPS 0.0237 • FID 99.68 • KID 0.1043 ± 0.0163 • CMMD 0.2696
  - **latest** ckpt (26 samples): MSE 0.1553 • MS-SSIM 0.9952 • PSNR 49.70 dB • LPIPS 0.0239 • FID 94.77 • KID 0.0935 ± 0.0143 • CMMD 0.2922

#### `24082412_24082412`  [🔗 chained]
*job 24082412 • idun-06-01 • A100 • epoch 172/500 • chain 0/20 • 34KB log*


#### `24082413_24082413`  [✅ completed]
*job 24082413 • idun-06-05 • A100 • 10.59h training • epoch 500/500 • chain 0/20 • 85KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00288 • MS-SSIM 0.9420 • PSNR 32.85 dB • LPIPS 0.7587 • FID 76.43 • KID 0.0919 ± 0.0099 • CMMD 0.2239
  - **latest** ckpt (26 samples): MSE 0.004557 • MS-SSIM 0.9171 • PSNR 30.69 dB • LPIPS 0.5204 • FID 36.85 • KID 0.0272 ± 0.0038 • CMMD 0.1809

#### `24082489_24082489`  [✅ completed]
*job 24082489 • idun-01-03 • H100 • 5.36h training • epoch 500/500 • chain 2/20 • 37KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.05143 • MS-SSIM 0.9517 • PSNR 35.14 dB • LPIPS 0.3715 • FID 83.03 • KID 0.0615 ± 0.0057 • CMMD 0.2622
  - **latest** ckpt (26 samples): MSE 0.04286 • MS-SSIM 0.9619 • PSNR 37.02 dB • LPIPS 0.2952 • FID 92.77 • KID 0.0722 ± 0.0048 • CMMD 0.2548
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_5_20260224-142309/checkpoint_latest.pt`

#### `24082612_24082612`  [🔗 chained]
*job 24082612 • idun-06-05 • A100 • epoch 333/500 • chain 0/20 • 228KB log*


#### `24085100_24085100`  [🔗 chained]
*job 24085100 • idun-06-05 • A100 • epoch 169/500 • chain 0/20 • 41KB log*


#### `24085101_24085101`  [✅ completed]
*job 24085101 • idun-06-07 • A100 • 10.9h training • epoch 500/500 • chain 0/20 • 108KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006735 • MS-SSIM 0.9265 • PSNR 31.31 dB • LPIPS 0.4632 • FID 54.48 • KID 0.0541 ± 0.0064 • CMMD 0.1649
  - **latest** ckpt (26 samples): MSE 0.007308 • MS-SSIM 0.9148 • PSNR 30.60 dB • LPIPS 0.4801 • FID 54.94 • KID 0.0557 ± 0.0061 • CMMD 0.1632

#### `24085346_24085346`  [🔗 chained]
*job 24085346 • idun-06-01 • A100 • epoch 343/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1g_1_pixel_bravo_pseudo_huber_20260226-034319/checkpoint_latest.pt`

#### `24089369_24089369`  [❌ crashed]
*job 24089369 • idun-06-01 • A100 • 5.95h training • epoch 500/500 • chain 1/20 • 99KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2e_pixel_seg_multilevel_aux_20260226-130811/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	361m26.664s
user	384m7.568s
sys	108m7.598s
```

#### `24089584_24089584`  [🔗 chained]
*job 24089584 • idun-06-05 • A100 • epoch 336/500 • chain 1/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1h_1_pixel_bravo_lpips_huber_20260226-202457/checkpoint_latest.pt`

#### `24089829_24089829`  [✅ completed]
*job 24089829 • idun-06-01 • A100 • 10.99h training • epoch 500/500 • chain 2/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002612 • MS-SSIM 0.9526 • PSNR 32.69 dB • LPIPS 0.5583 • FID 84.71 • KID 0.0898 ± 0.0092 • CMMD 0.1824
  - **latest** ckpt (26 samples): MSE 0.004478 • MS-SSIM 0.9384 • PSNR 31.62 dB • LPIPS 0.4834 • FID 68.30 • KID 0.0688 ± 0.0035 • CMMD 0.1143
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1g_1_pixel_bravo_pseudo_huber_20260226-034319/checkpoint_latest.pt`

#### `24093199_24093199`  [✅ completed]
*job 24093199 • idun-06-01 • A100 • 11.6h training • epoch 500/500 • chain 2/20 • 43KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003908 • MS-SSIM 0.9571 • PSNR 33.05 dB • LPIPS 0.4572 • FID 71.87 • KID 0.0755 ± 0.0064 • CMMD 0.1446
  - **latest** ckpt (26 samples): MSE 0.004614 • MS-SSIM 0.9480 • PSNR 32.18 dB • LPIPS 0.4736 • FID 63.96 • KID 0.0636 ± 0.0099 • CMMD 0.1391
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1h_1_pixel_bravo_lpips_huber_20260226-202457/checkpoint_latest.pt`

#### `24096039_24096039`  [🔗 chained]
*job 24096039 • idun-08-01 • H100 • epoch 141/500 • chain 0/20 • 34KB log*


#### `24096042_24096042`  [🔗 chained]
*job 24096042 • idun-08-01 • H100 • epoch 173/500 • chain 0/20 • 43KB log*


#### `24096044_24096044`  [🔗 chained]
*job 24096044 • idun-01-04 • H100 • epoch 263/500 • chain 0/20 • 48KB log*


#### `24096047_24096047`  [🔗 chained]
*job 24096047 • idun-01-04 • H100 • epoch 127/1000 • chain 0/40 • 29KB log*


#### `24096390_24096390`  [🔗 chained]
*job 24096390 • idun-07-08 • A100 • epoch 226/500 • chain 1/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`

#### `24096391_24096391`  [🔗 chained]
*job 24096391 • idun-01-04 • H100 • epoch 345/500 • chain 1/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24096393_24096393`  [✅ completed]
*job 24096393 • idun-01-04 • H100 • 10.85h training • epoch 500/500 • chain 1/20 • 45KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00313 • MS-SSIM 0.9567 • PSNR 33.10 dB • LPIPS 0.6425 • FID 69.23 • KID 0.0621 ± 0.0089 • CMMD 0.2253
  - **latest** ckpt (26 samples): MSE 0.003924 • MS-SSIM 0.9598 • PSNR 33.34 dB • LPIPS 0.5712 • FID 73.83 • KID 0.0689 ± 0.0090 • CMMD 0.2072
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1j_1_pixel_bravo_grad_accum_20260301-015744/checkpoint_latest.pt`

#### `24096397_24096397`  [🔗 chained]
*job 24096397 • idun-01-03 • H100 • epoch 304/1000 • chain 1/40 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24097015_24097015`  [🔗 chained]
*job 24097015 • idun-06-01 • A100 • epoch 121/500 • chain 0/20 • 29KB log*


#### `24097351_24097351`  [🔗 chained]
*job 24097351 • idun-07-10 • A100 • epoch 455/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24097352_24097352`  [🔗 chained]
*job 24097352 • idun-07-10 • A100 • epoch 314/500 • chain 2/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`

#### `24097361_24097361`  [🔗 chained]
*job 24097361 • idun-07-10 • A100 • epoch 417/1000 • chain 2/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24097410_24097410`  [🔗 chained]
*job 24097410 • idun-06-01 • A100 • epoch 241/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`

#### `24099367_24099367`  [✅ completed]
*job 24099367 • idun-07-10 • A100 • 4.86h training • epoch 500/500 • chain 3/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00299 • MS-SSIM 0.9592 • PSNR 33.46 dB • LPIPS 0.8082 • FID 139.99 • KID 0.1811 ± 0.0143 • CMMD 0.2642
  - **latest** ckpt (26 samples): MSE 0.002944 • MS-SSIM 0.9523 • PSNR 33.08 dB • LPIPS 0.9080 • FID 142.26 • KID 0.1851 ± 0.0154 • CMMD 0.2886
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24099368_24099368`  [💥 oom_killed]
*job 24099368 • idun-06-06 • A100 • 0.14h training • epoch 315/500 • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24099393_24099393`  [❌ crashed]
*job 24099393 • idun-08-01 • H100 • 9.02h training • epoch 551/1000 • chain 3/40 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	543m41.949s
user	537m23.007s
sys	174m52.959s
```

#### `24101988_24101988`  [❌ crashed]
*job 24101988 • idun-01-04 • H100 • 8.23h training • epoch 362/500 • chain 2/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	496m47.857s
user	484m37.372s
sys	166m12.672s
```

#### `24102847_24102847`  [🔗 chained]
*job 24102847 • idun-06-06 • A100 • epoch 109/500 • chain 0/20 • 37KB log*


#### `24102848_24102848`  [🔗 chained]
*job 24102848 • idun-01-04 • H100 • epoch 142/500 • chain 0/20 • 45KB log*


#### `24104590_24104590`  [💥 oom_killed]
*job 24104590 • idun-06-05 • A100 • 0.14h training • epoch 316/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24104596_24104596`  [✅ completed]
*job 24104596 • idun-01-04 • H100 • 9.33h training • epoch 500/500 • chain 0/20 • 33KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003122 • MS-SSIM 0.9579 • PSNR 33.24 dB • LPIPS 0.5616 • FID 91.46 • KID 0.0862 ± 0.0060 • CMMD 0.2347
  - **latest** ckpt (26 samples): MSE 0.003422 • MS-SSIM 0.9529 • PSNR 32.68 dB • LPIPS 0.5209 • FID 51.17 • KID 0.0332 ± 0.0055 • CMMD 0.1934
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`

#### `24104601_24104601`  [❌ crashed]
*job 24104601 • idun-01-04 • H100 • 6.1h training • epoch 642/1000 • chain 0/40 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	368m13.195s
user	363m32.088s
sys	122m31.923s
```

#### `24105243_24105243`  [💥 oom_killed]
*job 24105243 • idun-06-06 • A100 • 0.15h training • epoch 317/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24105272_24105272`  [🔗 chained]
*job 24105272 • idun-07-09 • A100 • epoch 232/500 • chain 1/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24105273_24105273`  [❌ crashed]
*job 24105273 • idun-07-09 • A100 • 3.35h training • epoch 136/500 • chain 1/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	204m25.835s
user	181m0.160s
sys	66m59.130s
```

#### `24107156_24107156`  [🔗 chained]
*job 24107156 • idun-07-09 • A100 • epoch 753/1000 • chain 0/40 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24107157_24107157`  [💥 oom_killed]
*job 24107157 • idun-06-05 • A100 • 0.15h training • epoch 318/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24107165_24107165`  [🔗 chained]
*job 24107165 • idun-06-03 • A100 • epoch 244/500 • chain 0/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24107260_24107260`  [❌ crashed]
*job 24107260 • idun-06-07 • A100 • 6.16h training • epoch 175/500 • chain 0/20 • 44KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 117, in add_event
    self._async_writer.write(event.SerializeToString())
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 171, in write
    self._check_worker_status()
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.p
```

#### `24107261_24107261`  [❌ crashed]
*job 24107261 • idun-06-03 • A100 • 10.03h training • epoch 500/500 • chain 0/20 • 99KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.414 • MS-SSIM 0.9899 • PSNR 32.97 dB • LPIPS 0.0874 • FID 58.19 • KID 0.0499 ± 0.0144 • CMMD 0.1982
  - **latest** ckpt (26 samples): MSE 3.565 • MS-SSIM 0.9776 • PSNR 31.07 dB • LPIPS 0.1821 • FID 50.89 • KID 0.0413 ± 0.0102 • CMMD 0.1740
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 851, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/base_trainer.py", line 794, in train
    self._handle_checkpoints(epoch, merged_metrics)
  Fil
```

#### `24107262_24107262`  [✅ completed]
*job 24107262 • idun-06-05 • A100 • 6.56h training • epoch 500/500 • chain 0/20 • 118KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.894 • MS-SSIM 0.9833 • PSNR 31.18 dB • LPIPS 0.1372 • FID 134.56 • KID 0.1355 ± 0.0266 • CMMD 0.3688
  - **latest** ckpt (26 samples): MSE 3.571 • MS-SSIM 0.9749 • PSNR 30.19 dB • LPIPS 0.1784 • FID 83.90 • KID 0.0852 ± 0.0181 • CMMD 0.2561

#### `24107263_24107263`  [🔗 chained]
*job 24107263 • idun-06-03 • A100 • epoch 405/500 • chain 0/20 • 76KB log*


#### `24107264_24107264`  [🔗 chained]
*job 24107264 • idun-07-10 • A100 • epoch 149/500 • chain 0/20 • 41KB log*


#### `24107272_24107272`  [🔗 chained]
*job 24107272 • idun-07-09 • A100 • epoch 674/2000 • chain 0/40 • 120KB log*


#### `24107429_24107429`  [🔗 chained]
*job 24107429 • idun-01-03 • H100 • epoch 375/500 • chain 2/20 • 64KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24110352_24110352`  [🔗 chained]
*job 24110352 • idun-06-02 • A100 • epoch 875/1000 • chain 1/40 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24110741_24110741`  [🔗 chained]
*job 24110741 • idun-07-09 • A100 • epoch 277/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24111006_24111006`  [✅ completed]
*job 24111006 • idun-06-05 • A100 • 2.74h training • epoch 500/500 • chain 1/20 • 27KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.339 • MS-SSIM 0.9887 • PSNR 33.10 dB • LPIPS 0.0876 • FID 73.74 • KID 0.0679 ± 0.0094 • CMMD 0.3877
  - **latest** ckpt (26 samples): MSE 2.407 • MS-SSIM 0.9863 • PSNR 33.45 dB • LPIPS 0.1075 • FID 48.99 • KID 0.0376 ± 0.0075 • CMMD 0.2661
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_1_ldm_4x_dit_b_20260304-100512/checkpoint_latest.pt`

#### `24111008_24111008`  [🔗 chained]
*job 24111008 • idun-07-10 • A100 • epoch 297/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24111018_24111018`  [🔗 chained]
*job 24111018 • idun-07-09 • A100 • epoch 1348/2000 • chain 1/40 • 112KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_3_ldm_4x_dit_s_long_20260304-102347/checkpoint_latest.pt`

#### `24111052_24111052`  [🔗 chained]
*job 24111052 • idun-07-09 • A100 • epoch 398/500 • chain 3/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24111102_24111102`  [🔗 chained]
*job 24111102 • idun-07-09 • A100 • epoch 988/1000 • chain 2/40 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24111271_24111271`  [🔗 chained]
*job 24111271 • idun-07-08 • A100 • epoch 376/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24111387_24111387`  [🔗 chained]
*job 24111387 • idun-06-01 • A100 • epoch 457/500 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24111388_24111388`  [✅ completed]
*job 24111388 • idun-07-10 • A100 • 11.21h training • epoch 2000/2000 • chain 2/40 • 114KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.48 • MS-SSIM 0.9923 • PSNR 34.59 dB • LPIPS 0.0613 • FID 68.87 • KID 0.0630 ± 0.0120 • CMMD 0.3208
  - **latest** ckpt (26 samples): MSE 4.891 • MS-SSIM 0.9686 • PSNR 28.41 dB • LPIPS 0.2423 • FID 61.14 • KID 0.0521 ± 0.0087 • CMMD 0.2833
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_3_ldm_4x_dit_s_long_20260304-102347/checkpoint_latest.pt`

#### `24111664_24111664`  [🔗 chained]
*job 24111664 • idun-07-10 • A100 • epoch 489/500 • chain 4/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24113841_24113841`  [✅ completed]
*job 24113841 • idun-06-07 • A100 • 9.93h training • epoch 500/500 • chain 0/20 • 85KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.5202 • MS-SSIM 0.9262 • PSNR 32.45 dB • LPIPS 0.3218 • FID 162.98 • KID 0.1618 ± 0.0090 • CMMD 0.4848
  - **latest** ckpt (26 samples): MSE 0.3158 • MS-SSIM 0.9507 • PSNR 33.98 dB • LPIPS 0.1938 • FID 110.88 • KID 0.0938 ± 0.0080 • CMMD 0.4088

#### `24114000_24114000`  [✅ completed]
*job 24114000 • idun-06-02 • A100 • 1.33h training • epoch 1000/1000 • chain 3/40 • 11KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002886 • MS-SSIM 0.9560 • PSNR 33.12 dB • LPIPS 0.7589 • FID 111.86 • KID 0.0921 ± 0.0067 • CMMD 0.3259
  - **latest** ckpt (26 samples): MSE 0.008323 • MS-SSIM 0.9237 • PSNR 30.16 dB • LPIPS 0.3856 • FID 49.53 • KID 0.0352 ± 0.0069 • CMMD 0.1495
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24114364_24114364`  [💥 oom_killed]
*job 24114364 • idun-07-09 • A100 • 11.43h training • epoch 500/500 • chain 0/20 • 96KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004095 • MS-SSIM 0.9497 • PSNR 33.44 dB • LPIPS 0.7918 • FID 78.30 • KID 0.0887 ± 0.0060 • CMMD 0.2198
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	690m37.038s
user	524m6.108s
sys	150m11.531s
[2026-03-06T23:06:23.424] error: Detected 1 oom_kill event in StepId=24114364.batch. Some of the step tasks have been OOM Killed.
```

#### `24114365_24114365`  [🔗 chained]
*job 24114365 • idun-06-01 • A100 • epoch 120/500 • chain 0/20 • 29KB log*


#### `24114366_24114366`  [✅ completed]
*job 24114366 • idun-06-01 • A100 • 10.89h training • epoch 500/500 • chain 0/20 • 102KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003311 • MS-SSIM 0.9459 • PSNR 32.94 dB • LPIPS 0.8903 • FID 82.85 • KID 0.0975 ± 0.0079 • CMMD 0.2354
  - **latest** ckpt (26 samples): MSE 0.007188 • MS-SSIM 0.9322 • PSNR 31.70 dB • LPIPS 0.5438 • FID 70.01 • KID 0.0820 ± 0.0069 • CMMD 0.1784

#### `24115304_24115304`  [🔗 chained]
*job 24115304 • idun-06-04 • A100 • epoch 484/500 • chain 3/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24115341_24115341`  [✅ completed]
*job 24115341 • idun-07-10 • A100 • 3.53h training • epoch 500/500 • chain 3/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.307 • MS-SSIM 0.9921 • PSNR 34.30 dB • LPIPS 0.0661 • FID 75.79 • KID 0.0675 ± 0.0125 • CMMD 0.3399
  - **latest** ckpt (26 samples): MSE 2.396 • MS-SSIM 0.9899 • PSNR 33.95 dB • LPIPS 0.0837 • FID 47.41 • KID 0.0355 ± 0.0075 • CMMD 0.2515
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24115380_24115380`  [✅ completed]
*job 24115380 • idun-07-10 • A100 • 1.51h training • epoch 500/500 • chain 5/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004045 • MS-SSIM 0.9579 • PSNR 33.47 dB • LPIPS 0.6612 • FID 129.77 • KID 0.1583 ± 0.0140 • CMMD 0.2623
  - **latest** ckpt (26 samples): MSE 0.006105 • MS-SSIM 0.9321 • PSNR 31.11 dB • LPIPS 0.4891 • FID 98.00 • KID 0.1100 ± 0.0114 • CMMD 0.1903
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24118770_24118770`  [🔗 chained]
*job 24118770 • idun-08-01 • H100 • epoch 260/500 • chain 0/20 • 60KB log*


#### `24118771_24118771`  [💥 oom_killed]
*job 24118771 • idun-06-07 • A100 • 10.62h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01449 • MS-SSIM 0.9058 • PSNR 32.83 dB • LPIPS 1.3048 • FID 118.72 • KID 0.1116 ± 0.0045 • CMMD 0.4160
  - **latest** ckpt (26 samples): MSE 1.003 • MS-SSIM 0.3960 • PSNR 15.75 dB • LPIPS 1.8775
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	642m20.340s
user	448m34.979s
sys	189m9.467s
[2026-03-07T09:49:58.272] error: Detected 1 oom_kill event in StepId=24118771.batch. Some of the step tasks have been OOM Killed.
```

#### `24121041_24121041`  [🔗 chained]
*job 24121041 • idun-06-07 • A100 • epoch 240/500 • chain 1/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`

#### `24121176_24121176`  [✅ completed]
*job 24121176 • idun-08-01 • H100 • 1.28h training • epoch 500/500 • chain 4/20 • 13KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003615 • MS-SSIM 0.9542 • PSNR 32.83 dB • LPIPS 0.6113 • FID 127.91 • KID 0.1477 ± 0.0090 • CMMD 0.3290
  - **latest** ckpt (26 samples): MSE 0.005463 • MS-SSIM 0.9336 • PSNR 30.99 dB • LPIPS 0.4606 • FID 99.62 • KID 0.1141 ± 0.0091 • CMMD 0.1601
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24121342_24121342`  [✅ completed]
*job 24121342 • idun-01-03 • H100 • 10.96h training • epoch 500/500 • chain 1/20 • 67KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006567 • MS-SSIM 0.9635 • PSNR 34.07 dB • LPIPS 0.5974 • FID 134.31 • KID 0.1252 ± 0.0093 • CMMD 0.3261
  - **latest** ckpt (26 samples): MSE 0.007464 • MS-SSIM 0.9716 • PSNR 34.76 dB • LPIPS 0.4379 • FID 139.64 • KID 0.1382 ± 0.0106 • CMMD 0.2737
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1m_1_pixel_bravo_global_norm_20260306-215802/checkpoint_latest.pt`

#### `24121607_24121607`  [🔗 chained]
*job 24121607 • idun-06-07 • A100 • epoch 359/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`

#### `24121732_24121732`  [💥 oom_killed]
*job 24121732 • idun-01-05 • H100 • epoch 125/1000 • chain 0/20 • 130KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[2026-03-08T16:09:23.152] error: *** JOB 24121732 ON idun-01-05 CANCELLED AT 2026-03-08T16:09:23 DUE to SIGNAL Terminated ***
```

#### `24122196_24122196`  [💥 oom_killed]
*job 24122196 • idun-06-07 • A100 • epoch 392/500 • chain 3/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[2026-03-08T16:11:27.682] error: *** JOB 24122196 ON idun-06-07 CANCELLED AT 2026-03-08T16:11:27 DUE to SIGNAL Terminated ***
```

#### `24122599_24122599`  [❌ crashed]
*job 24122599 • idun-07-09 • A100 • 0.5h training • epoch 3/3 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.06258 • MS-SSIM 0.6368 • PSNR 22.54 dB • LPIPS 1.7175 • FID 365.45 • KID 0.4820 ± 0.0135 • CMMD 0.6928
  - **latest** ckpt (26 samples): MSE 0.062 • MS-SSIM 0.5090 • PSNR 19.90 dB • LPIPS 1.7322 • FID 364.20 • KID 0.4810 ± 0.0143 • CMMD 0.6919
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<string>", line 13, in <module>
AttributeError: module 'medgen.pipeline.validation' has no attribute 'run_validation'
...
  covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

real	46m25.144s
user	47m20.104s
sys	13m24.493s
```

#### `24122694_24122694`  [❌ crashed]
*job 24122694 • idun-01-05 • H100 • 10.07h training • epoch 146/1000 • chain 0/20 • 140KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 381, in add_scalar
    self._get_file_writer().add_summary(summary, global_step, walltime)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 115, in add_summary
    self.add_event(event, global_step, walltime)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", lin
```

#### `24122695_24122695`  [❌ crashed]
*job 24122695 • idun-07-08 • A100 • 3.21h training • epoch 29/500 • chain 0/20 • 31KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24122698_24122698`  [❌ crashed]
*job 24122698 • idun-01-05 • H100 • 2.33h training • epoch 33/500 • chain 0/20 • 36KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24126433_24126433`  [🔗 chained]
*job 24126433 • idun-01-05 • H100 • epoch 169/500 • chain 0/20 • 39KB log*


#### `24126434_24126434`  [🔗 chained]
*job 24126434 • idun-06-02 • A100 • epoch 117/500 • chain 0/20 • 97KB log*


#### `24126436_24126436`  [🔗 chained]
*job 24126436 • idun-07-10 • A100 • epoch 110/1000 • chain 0/20 • 29KB log*


#### `24126648_24126648`  [🔗 chained]
*job 24126648 • idun-06-02 • A100 • epoch 113/500 • chain 0/20 • 30KB log*


#### `24127045_24127045`  [🔗 chained]
*job 24127045 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 31KB log*


#### `24127090_24127090`  [🔗 chained]
*job 24127090 • idun-06-07 • A100 • epoch 231/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24127098_24127098`  [🔗 chained]
*job 24127098 • idun-07-09 • A100 • epoch 277/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`

#### `24127181_24127181`  [🔗 chained]
*job 24127181 • idun-08-01 • H100 • epoch 282/1000 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24128564_24128564`  [🔗 chained]
*job 24128564 • idun-06-01 • A100 • epoch 228/500 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24129682_24129682`  [❌ crashed]
*job 24129682 • idun-01-03 • H100 • epoch 185/500 • chain 0/20 • 914KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/validation.py", line 513, in compute_validation_losses
    gen_results = trainer._gen_metrics.compute_epoch_metrics(
...
    pred_cond = self._call_model(
                ^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/diffusion/strategy_rflow.py", line 195, in _call_model
    return super()._call_model(model, model_input, timesteps, omega, mode_id, size_bins)
           ^^^^^^^^^
```

#### `24130511_24130511`  [🔗 chained]
*job 24130511 • idun-06-02 • A100 • epoch 236/500 • chain 1/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24131281_24131281`  [🔗 chained]
*job 24131281 • idun-07-09 • A100 • epoch 341/500 • chain 2/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24131353_24131353`  [💥 oom_killed]
*job 24131353 • idun-08-01 • H100 • epoch 432/500 • chain 2/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	650m29.237s
user	486m9.064s
sys	173m15.159s
[2026-03-11T20:38:27.894] error: Detected 1 oom_kill event in StepId=24131353.batch. Some of the step tasks have been OOM Killed.
```

#### `24131377_24131377`  [🔗 chained]
*job 24131377 • idun-06-01 • A100 • epoch 401/1000 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24131382_24131382`  [🔗 chained]
*job 24131382 • idun-07-10 • A100 • epoch 336/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24133692_24133692`  [❌ crashed]
*job 24133692 • idun-06-06 • A100 • 0.1h training • chain 1/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24137988_24137988`  [🔗 chained]
*job 24137988 • idun-08-01 • H100 • epoch 172/500 • chain 0/20 • 39KB log*


#### `24139725_24139725`  [❌ crashed]
*job 24139725 • idun-08-01 • H100 • 5.98h training • epoch 322/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet, chunk_sz, orig_d).cpu())
                              ^^^^^^^^
```

#### `24139869_24139869`  [❌ crashed]
*job 24139869 • idun-01-05 • H100 • 4.61h training • epoch 407/500 • chain 3/20 • 15KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet, chunk_sz, orig_d).cpu())
                              ^^^^^^^^
```

#### `24139886_24139886`  [🔗 chained]
*job 24139886 • idun-06-01 • A100 • epoch 521/1000 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24139898_24139898`  [💥 oom_killed]
*job 24139898 • idun-07-08 • A100 • epoch 339/500 • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	22m1.456s
user	14m18.872s
sys	6m11.402s
[2026-03-12T01:40:51.118] error: Detected 1 oom_kill event in StepId=24139898.batch. Some of the step tasks have been OOM Killed.
```

#### `24139926_24139926`  [❌ crashed]
*job 24139926 • idun-07-10 • A100 • 1.0h training • epoch 193/500 • chain 2/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 563, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24139930_24139930`  [❌ crashed]
*job 24139930 • idun-01-03 • H100 • 0.96h training • epoch 444/500 • chain 3/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 563, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24139968_24139968`  [🔗 chained]
*job 24139968 • idun-07-08 • A100 • epoch 110/500 • chain 0/20 • 50KB log*


#### `24140632_24140632`  [🔗 chained]
*job 24140632 • idun-06-07 • A100 • epoch 288/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`

#### `24142845_24142845`  [🔗 chained]
*job 24142845 • idun-06-01 • A100 • epoch 640/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24144753_24144753`  [🔗 chained]
*job 24144753 • idun-07-08 • A100 • epoch 219/500 • chain 1/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24147112_24147112`  [🔗 chained]
*job 24147112 • idun-06-07 • A100 • epoch 404/500 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`

#### `24147147_24147147`  [🔗 chained]
*job 24147147 • idun-01-03 • H100 • epoch 499/500 • chain 0/20 • 81KB log*


#### `24147155_24147155`  [💥 oom_killed]
*job 24147155 • idun-08-01 • H100 • epoch 343/500 • chain 4/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	21m33.700s
user	15m35.729s
sys	5m54.117s
[2026-03-12T21:34:07.271] error: Detected 1 oom_kill event in StepId=24147155.batch. Some of the step tasks have been OOM Killed.
```

#### `24147160_24147160`  [🔗 chained]
*job 24147160 • idun-08-01 • H100 • epoch 369/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`

#### `24147165_24147165`  [✅ completed]
*job 24147165 • idun-06-05 • A100 • 9.36h training • epoch 500/500 • chain 4/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002761 • MS-SSIM 0.9476 • PSNR 32.69 dB • LPIPS 0.8929 • FID 165.65 • KID 0.2147 ± 0.0149 • CMMD 0.3337
  - **latest** ckpt (26 samples): MSE 0.004248 • MS-SSIM 0.9355 • PSNR 31.18 dB • LPIPS 0.4949 • FID 132.83 • KID 0.1739 ± 0.0152 • CMMD 0.2477
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24147168_24147168`  [✅ completed]
*job 24147168 • idun-06-06 • A100 • 5.81h training • epoch 500/500 • chain 4/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002532 • MS-SSIM 0.9477 • PSNR 32.36 dB • LPIPS 0.8944 • FID 92.49 • KID 0.1024 ± 0.0139 • CMMD 0.2773
  - **latest** ckpt (26 samples): MSE 0.002621 • MS-SSIM 0.9487 • PSNR 32.23 dB • LPIPS 0.6694 • FID 72.52 • KID 0.0704 ± 0.0123 • CMMD 0.2260
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`

#### `24147171_24147171`  [🔗 chained]
*job 24147171 • idun-07-09 • A100 • epoch 431/500 • chain 3/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24147343_24147343`  [✅ completed]
*job 24147343 • idun-01-05 • H100 • 11.01h training • epoch 500/500 • chain 5/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002897 • MS-SSIM 0.9469 • PSNR 32.68 dB • LPIPS 0.8119 • FID 80.55 • KID 0.0889 ± 0.0123 • CMMD 0.2226
  - **latest** ckpt (26 samples): MSE 0.003812 • MS-SSIM 0.9590 • PSNR 33.36 dB • LPIPS 0.5329 • FID 62.64 • KID 0.0537 ± 0.0092 • CMMD 0.1898
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24147348_24147348`  [🔗 chained]
*job 24147348 • idun-07-09 • A100 • epoch 750/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24147766_24147766`  [🔗 chained]
*job 24147766 • idun-06-05 • A100 • epoch 337/500 • chain 2/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24148199_24148199`  [💥 oom_killed]
*job 24148199 • idun-01-05 • H100 • 6.62h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003326 • MS-SSIM 0.9508 • PSNR 32.67 dB • LPIPS 0.8495
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	398m53.534s
user	300m44.359s
sys	103m48.996s
[2026-03-13T14:49:52.774] error: Detected 2 oom_kill events in StepId=24148199.batch. Some of the step tasks have been OOM Killed.
```

#### `24148244_24148244`  [✅ completed]
*job 24148244 • idun-06-07 • A100 • 0.05h training • epoch 500/500 • chain 1/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006818 • MS-SSIM 0.9335 • PSNR 31.67 dB • LPIPS 0.5036 • FID 52.98 • KID 0.0520 ± 0.0066 • CMMD 0.1799
  - **latest** ckpt (26 samples): MSE 0.005155 • MS-SSIM 0.9279 • PSNR 31.26 dB • LPIPS 0.5373 • FID 49.62 • KID 0.0470 ± 0.0072 • CMMD 0.1624
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_pixel_bravo_controlnet_stage2_20260312-210758/checkpoint_latest.pt`

#### `24148265_24148265`  [💥 oom_killed]
*job 24148265 • idun-01-03 • H100 • 8.77h training • epoch 500/500 • chain 4/20 • 31KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003448 • MS-SSIM 0.9577 • PSNR 33.57 dB • LPIPS 0.5714
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	530m12.740s
user	401m55.684s
sys	135m11.556s
[2026-03-13T18:23:08.744] error: Detected 1 oom_kill event in StepId=24148265.batch. Some of the step tasks have been OOM Killed.
```

#### `24150904_24150904`  [🔗 chained]
*job 24150904 • idun-01-05 • H100 • epoch 437/500 • chain 0/20 • 79KB log*


#### `24150929_24150929`  [✅ completed]
*job 24150929 • idun-01-03 • H100 • 4.78h training • epoch 500/500 • chain 4/20 • 25KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.009399 • MS-SSIM 0.9676 • PSNR 34.99 dB • LPIPS 0.5624 • FID 116.55 • KID 0.1107 ± 0.0096 • CMMD 0.2171
  - **latest** ckpt (26 samples): MSE 0.01513 • MS-SSIM 0.9345 • PSNR 31.27 dB • LPIPS 0.4882 • FID 72.00 • KID 0.0689 ± 0.0078 • CMMD 0.1667
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24150939_24150939`  [🔗 chained]
*job 24150939 • idun-08-01 • H100 • epoch 410/500 • chain 0/20 • 74KB log*


#### `24150962_24150962`  [🔗 chained]
*job 24150962 • idun-01-05 • H100 • epoch 925/1000 • chain 6/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24151901_24151901`  [🔗 chained]
*job 24151901 • idun-01-03 • H100 • epoch 189/500 • chain 0/20 • 36KB log*


#### `24152108_24152108`  [🔗 chained]
*job 24152108 • idun-07-09 • A100 • epoch 447/500 • chain 3/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24154205_24154205`  [✅ completed]
*job 24154205 • idun-01-05 • H100 • 1.75h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.005689 • MS-SSIM 0.9552 • PSNR 33.69 dB • LPIPS 0.5217 • FID 54.60 • KID 0.0509 ± 0.0069 • CMMD 0.1899
  - **latest** ckpt (26 samples): MSE 0.004787 • MS-SSIM 0.9195 • PSNR 30.86 dB • LPIPS 0.5152 • FID 43.45 • KID 0.0323 ± 0.0055 • CMMD 0.1681
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_pixel_bravo_weight_decay_20260313-141301/checkpoint_latest.pt`

#### `24154213_24154213`  [✅ completed]
*job 24154213 • idun-01-05 • H100 • 2.48h training • epoch 500/500 • chain 1/20 • 26KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002354 • MS-SSIM 0.9408 • PSNR 32.63 dB • LPIPS 0.7802 • FID 82.21 • KID 0.0887 ± 0.0077 • CMMD 0.2667
  - **latest** ckpt (26 samples): MSE 0.008438 • MS-SSIM 0.9187 • PSNR 30.55 dB • LPIPS 0.4523 • FID 49.32 • KID 0.0466 ± 0.0045 • CMMD 0.1629
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1r_pixel_bravo_attn_dropout_20260313-142342/checkpoint_latest.pt`

#### `24154217_24154217`  [🔗 chained]
*job 24154217 • idun-07-09 • A100 • epoch 258/500 • chain 0/20 • 54KB log*


#### `24154218_24154218`  [🔗 chained]
*job 24154218 • idun-07-09 • A100 • epoch 187/500 • chain 0/20 • 40KB log*


#### `24154219_24154219`  [🔗 chained]
*job 24154219 • idun-01-05 • H100 • epoch 425/500 • chain 0/20 • 109KB log*


#### `24154238_24154238`  [❌ crashed]
*job 24154238 • idun-07-09 • A100 • 10.82h training • epoch 239/500 • chain 0/20 • 74KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24154257_24154257`  [✅ completed]
*job 24154257 • idun-01-05 • H100 • 5.15h training • epoch 1000/1000 • chain 7/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002195 • MS-SSIM 0.9553 • PSNR 33.04 dB • LPIPS 0.5915 • FID 72.39 • KID 0.0579 ± 0.0061 • CMMD 0.1936
  - **latest** ckpt (26 samples): MSE 0.003152 • MS-SSIM 0.9569 • PSNR 33.34 dB • LPIPS 0.5307 • FID 62.57 • KID 0.0531 ± 0.0091 • CMMD 0.1850
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24154353_24154353`  [🔗 chained]
*job 24154353 • idun-07-09 • A100 • epoch 292/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24154591_24154591`  [✅ completed]
*job 24154591 • idun-07-08 • A100 • 5.81h training • epoch 500/500 • chain 4/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003009 • MS-SSIM 0.9484 • PSNR 32.84 dB • LPIPS 0.8018 • FID 62.00 • KID 0.0508 ± 0.0097 • CMMD 0.2228
  - **latest** ckpt (26 samples): MSE 0.00416 • MS-SSIM 0.9495 • PSNR 33.16 dB • LPIPS 0.6704 • FID 58.85 • KID 0.0464 ± 0.0108 • CMMD 0.1977
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24156955_24156955`  [✅ completed]
*job 24156955 • idun-06-02 • A100 • 9.96h training • epoch 500/500 • chain 1/20 • 56KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003971 • MS-SSIM 0.9517 • PSNR 32.92 dB • LPIPS 0.6851 • FID 95.85 • KID 0.0944 ± 0.0045 • CMMD 0.2430
  - **latest** ckpt (26 samples): MSE 0.004889 • MS-SSIM 0.9298 • PSNR 30.77 dB • LPIPS 0.5877 • FID 99.86 • KID 0.1104 ± 0.0092 • CMMD 0.2452
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_4_pixel_bravo_small_67m_20260314-064756/checkpoint_latest.pt`

#### `24156956_24156956`  [🔗 chained]
*job 24156956 • idun-01-05 • H100 • epoch 490/500 • chain 1/20 • 59KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_5_pixel_bravo_mid_152m_20260314-064756/checkpoint_latest.pt`

#### `24156991_24156991`  [✅ completed]
*job 24156991 • idun-06-07 • A100 • 2.98h training • epoch 500/500 • chain 1/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00329 • MS-SSIM 0.9580 • PSNR 33.06 dB • LPIPS 0.5910 • FID 112.87 • KID 0.1394 ± 0.0134 • CMMD 0.2344
  - **latest** ckpt (26 samples): MSE 0.004855 • MS-SSIM 0.9556 • PSNR 32.90 dB • LPIPS 0.5191 • FID 94.75 • KID 0.1017 ± 0.0100 • CMMD 0.2306
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_6_pixel_bravo_tiny_20m_20260314-072201/checkpoint_latest.pt`

#### `24157008_24157008`  [✅ completed]
*job 24157008 • idun-06-06 • A100 • 10.42h training • epoch 500/500 • chain 0/20 • 58KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003105 • MS-SSIM 0.9525 • PSNR 32.63 dB • LPIPS 0.6132 • FID 112.64 • KID 0.1244 ± 0.0135 • CMMD 0.3127
  - **latest** ckpt (26 samples): MSE 0.004998 • MS-SSIM 0.9585 • PSNR 33.26 dB • LPIPS 0.5157 • FID 92.16 • KID 0.0977 ± 0.0087 • CMMD 0.2342
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_7_pixel_bravo_67m_no_attn_20260314-072441/checkpoint_latest.pt`

#### `24158373_24158373`  [🔗 chained]
*job 24158373 • idun-08-01 • H100 • epoch 477/500 • chain 2/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24158696_24158696`  [🔗 chained]
*job 24158696 • idun-01-05 • H100 • epoch 283/2000 • chain 0/40 • 129KB log*


#### `24158706_24158706`  [🔗 chained]
*job 24158706 • idun-01-05 • H100 • epoch 172/1000 • chain 0/20 • 51KB log*


#### `24158726_24158726`  [✅ completed]
*job 24158726 • idun-01-05 • H100 • 0.44h training • epoch 500/500 • chain 2/20 • 11KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003212 • MS-SSIM 0.9569 • PSNR 33.13 dB • LPIPS 0.6350 • FID 95.00 • KID 0.0895 ± 0.0118 • CMMD 0.3304
  - **latest** ckpt (26 samples): MSE 0.004167 • MS-SSIM 0.9329 • PSNR 30.95 dB • LPIPS 0.5574 • FID 72.11 • KID 0.0706 ± 0.0097 • CMMD 0.1885
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_5_pixel_bravo_mid_152m_20260314-064756/checkpoint_latest.pt`

#### `24158771_24158771`  [✅ completed]
*job 24158771 • idun-01-05 • H100 • 1.45h training • epoch 500/500 • chain 3/20 • 15KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002549 • MS-SSIM 0.9522 • PSNR 32.99 dB • LPIPS 0.6602 • FID 76.11 • KID 0.0749 ± 0.0102 • CMMD 0.2481
  - **latest** ckpt (26 samples): MSE 0.002899 • MS-SSIM 0.9573 • PSNR 33.40 dB • LPIPS 0.5920 • FID 82.82 • KID 0.0896 ± 0.0093 • CMMD 0.2242
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24160233_24160233`  [🔗 chained]
*job 24160233 • idun-06-02 • A100 • epoch 491/2000 • chain 1/40 • 96KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24160234_24160234`  [🔗 chained]
*job 24160234 • idun-06-02 • A100 • epoch 291/1000 • chain 1/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24167549_24167549`  [🔗 chained]
*job 24167549 • idun-08-01 • H100 • epoch 744/2000 • chain 2/40 • 119KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24167550_24167550`  [🔗 chained]
*job 24167550 • idun-07-09 • A100 • epoch 400/1000 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24188995_24188995`  [🔗 chained]
*job 24188995 • idun-01-03 • H100 • epoch 1031/2000 • chain 3/40 • 138KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24189005_24189005`  [🔗 chained]
*job 24189005 • idun-01-05 • H100 • epoch 572/1000 • chain 3/20 • 55KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24193972_24193972`  [🔗 chained]
*job 24193972 • idun-08-01 • H100 • epoch 1314/2000 • chain 4/40 • 142KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24193976_24193976`  [🔗 chained]
*job 24193976 • idun-01-05 • H100 • epoch 743/1000 • chain 4/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24199806_24199806`  [🔗 chained]
*job 24199806 • idun-01-05 • H100 • epoch 1599/2000 • chain 5/40 • 138KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24200116_24200116`  [🔗 chained]
*job 24200116 • idun-06-05 • A100 • epoch 860/1000 • chain 5/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24204011_24204011`  [🔗 chained]
*job 24204011 • idun-08-01 • H100 • epoch 1887/2000 • chain 6/40 • 135KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24204272_24204272`  [✅ completed]
*job 24204272 • idun-08-01 • H100 • 9.63h training • epoch 1000/1000 • chain 6/20 • 54KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01121 • MS-SSIM 0.9504 • PSNR 34.24 dB • LPIPS 0.7429 • FID 65.46 • KID 0.0542 ± 0.0107 • CMMD 0.2549
  - **latest** ckpt (26 samples): MSE 0.005543 • MS-SSIM 0.9404 • PSNR 32.70 dB • LPIPS 0.8810 • FID 62.87 • KID 0.0526 ± 0.0093 • CMMD 0.2594
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24206608_24206608`  [✅ completed]
*job 24206608 • idun-07-08 • A100 • 7.14h training • epoch 2000/2000 • chain 7/40 • 62KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.008342 • MS-SSIM 0.9470 • PSNR 33.54 dB • LPIPS 0.7312 • FID 72.95 • KID 0.0579 ± 0.0051 • CMMD 0.3016
  - **latest** ckpt (26 samples): MSE 0.02007 • MS-SSIM 0.9580 • PSNR 34.20 dB • LPIPS 0.6224 • FID 73.49 • KID 0.0565 ± 0.0070 • CMMD 0.3082
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24232399_24232399`  [🔗 chained]
*job 24232399 • idun-06-05 • A100 • epoch 281/1000 • chain 0/20 • 52KB log*


#### `24232401_24232401`  [🔗 chained]
*job 24232401 • idun-06-05 • A100 • epoch 580/1000 • chain 0/20 • 103KB log*


#### `24233992_24233992`  [🔗 chained]
*job 24233992 • idun-06-05 • A100 • epoch 558/1000 • chain 1/20 • 52KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`

#### `24234000_24234000`  [✅ completed]
*job 24234000 • idun-06-05 • A100 • 8.81h training • epoch 1000/1000 • chain 1/20 • 82KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.112 • MS-SSIM 0.9830 • PSNR 31.39 dB • LPIPS 0.1553 • FID 131.66 • KID 0.1368 ± 0.0237 • CMMD 0.3866
  - **latest** ckpt (26 samples): MSE 3.005 • MS-SSIM 0.9895 • PSNR 33.44 dB • LPIPS 0.0970 • FID 79.91 • KID 0.0786 ± 0.0173 • CMMD 0.2612
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_ldm_4x_unet_maisi_scoreaug_20260325-013810/checkpoint_latest.pt`

#### `24235208_24235208`  [🔗 chained]
*job 24235208 • idun-07-08 • A100 • epoch 136/1000 • chain 0/20 • 120KB log*


#### `24235533_24235533`  [🔗 chained]
*job 24235533 • idun-06-05 • A100 • epoch 569/1000 • chain 0/20 • 108KB log*


#### `24235992_24235992`  [🔗 chained]
*job 24235992 • idun-06-05 • A100 • epoch 838/1000 • chain 2/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`

#### `24236459_24236459`  [🔗 chained]
*job 24236459 • idun-07-10 • A100 • epoch 270/1000 • chain 1/20 • 151KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24236678_24236678`  [✅ completed]
*job 24236678 • idun-06-05 • A100 • 8.97h training • epoch 1000/1000 • chain 1/20 • 86KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.104 • MS-SSIM 0.9868 • PSNR 32.68 dB • LPIPS 0.1136 • FID 168.88 • KID 0.1857 ± 0.0279 • CMMD 0.3986
  - **latest** ckpt (26 samples): MSE 2.499 • MS-SSIM 0.9812 • PSNR 32.74 dB • LPIPS 0.1553 • FID 98.56 • KID 0.0960 ± 0.0211 • CMMD 0.3725
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_1_ldm_4x_unet_maisi_scoreaug_mixup_20260325-211501/checkpoint_latest.pt`

#### `24238367_24238367`  [💥 oom_killed]
*job 24238367 • idun-06-07 • A100 • 6.95h training • epoch 1000/1000 • chain 3/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3704 • MS-SSIM 0.9661 • PSNR 36.27 dB • LPIPS 0.2521
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`
**Traceback excerpt:**
```
nown)"))'), '(Request ID: 08de47ff-3be5-4f0c-aa43-3fd382d13d69)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json
Retrying in 2s [Retry 2/5].
'(MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json (Caused by NameResolutionError("HTTPSConnection(host=\'huggingface.co\', port=443): Failed to resolve \'huggingface.co\' ([Errno -2] Name or service not known)"))'), '(Request ID: ed818f74-6c73-426e-991c-05c7d142f3f4)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json
...

real	423m24.548s
user	308m18.299s
sys	107m46.389s
[2026-03-26T21:36:56.874] error: Detected 1 oom_kill event in StepId=24238367.batch. Some of the step tasks have been OOM Killed.
```

#### `24238431_24238431`  [🔗 chained]
*job 24238431 • idun-06-05 • A100 • epoch 568/1000 • chain 0/20 • 102KB log*


#### `24239579_24239579`  [🔗 chained]
*job 24239579 • idun-07-10 • A100 • epoch 259/500 • chain 0/20 • 85KB log*


#### `24239580_24239580`  [🔗 chained]
*job 24239580 • idun-07-09 • A100 • epoch 268/500 • chain 0/20 • 95KB log*


#### `24240420_24240420`  [🔗 chained]
*job 24240420 • idun-07-10 • A100 • epoch 399/1000 • chain 2/20 • 398KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24241623_24241623`  [🔗 chained]
*job 24241623 • idun-06-07 • A100 • epoch 282/1000 • chain 0/20 • 74KB log*


#### `24241654_24241654`  [🔗 chained]
*job 24241654 • idun-06-03 • A100 • epoch 319/500 • chain 0/20 • 94KB log*


#### `24241655_24241655`  [🔗 chained]
*job 24241655 • idun-06-06 • A100 • epoch 318/500 • chain 0/20 • 90KB log*


#### `24241672_24241672`  [✅ completed]
*job 24241672 • idun-07-10 • A100 • 10.82h training • epoch 1000/1000 • chain 1/20 • 92KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.24 • MS-SSIM 0.9905 • PSNR 35.24 dB • LPIPS 0.0970 • FID 93.32 • KID 0.1057 ± 0.0250 • CMMD 0.2819
  - **latest** ckpt (26 samples): MSE 2.342 • MS-SSIM 0.9836 • PSNR 33.51 dB • LPIPS 0.1540 • FID 80.17 • KID 0.0776 ± 0.0167 • CMMD 0.2945
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_2_ldm_4x_unet_maisi_scoreaug_v2_20260326-144538/checkpoint_latest.pt`

#### `24241848_24241848`  [✅ completed]
*job 24241848 • idun-07-10 • A100 • 11.1h training • epoch 500/500 • chain 1/20 • 79KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002548 • MS-SSIM 0.9370 • PSNR 32.47 dB • LPIPS 0.9223 • FID 79.68 • KID 0.0887 ± 0.0104 • CMMD 0.3276
  - **latest** ckpt (26 samples): MSE 0.00376 • MS-SSIM 0.9475 • PSNR 32.93 dB • LPIPS 0.6537 • FID 65.37 • KID 0.0642 ± 0.0063 • CMMD 0.2506
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_pixel_triple_20260326-164117/checkpoint_latest.pt`

#### `24241849_24241849`  [✅ completed]
*job 24241849 • idun-06-04 • A100 • 9.13h training • epoch 500/500 • chain 1/20 • 70KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002441 • MS-SSIM 0.9439 • PSNR 33.44 dB • LPIPS 0.7884 • FID 32.80 • KID 0.0216 ± 0.0022 • CMMD 0.1715
  - **latest** ckpt (26 samples): MSE 0.003058 • MS-SSIM 0.9418 • PSNR 32.39 dB • LPIPS 0.6395 • FID 35.91 • KID 0.0299 ± 0.0044 • CMMD 0.1756
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_pixel_dual_20260326-164147/checkpoint_latest.pt`

#### `24241903_24241903`  [🔗 chained]
*job 24241903 • idun-07-10 • A100 • epoch 531/1000 • chain 3/20 • 283KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24244849_24244849`  [🔗 chained]
*job 24244849 • idun-06-07 • A100 • epoch 563/1000 • chain 1/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24244953_24244953`  [✅ completed]
*job 24244953 • idun-01-03 • H100 • 5.13h training • epoch 500/500 • chain 1/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004596 • MS-SSIM 0.9559 • PSNR 32.84 dB • LPIPS 0.5267 • FID 24.30 • KID 0.0097 ± 0.0027 • CMMD 0.2267
  - **latest** ckpt (26 samples): MSE 0.003629 • MS-SSIM 0.9261 • PSNR 30.64 dB • LPIPS 0.6767 • FID 44.54 • KID 0.0376 ± 0.0048 • CMMD 0.1839
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_1_pixel_dual_joint_norm_20260327-031630/checkpoint_latest.pt`

#### `24245009_24245009`  [✅ completed]
*job 24245009 • idun-06-04 • A100 • 6.84h training • epoch 500/500 • chain 1/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006714 • MS-SSIM 0.9434 • PSNR 32.16 dB • LPIPS 0.7475 • FID 80.54 • KID 0.0890 ± 0.0050 • CMMD 0.3843
  - **latest** ckpt (26 samples): MSE 0.007807 • MS-SSIM 0.9412 • PSNR 32.04 dB • LPIPS 0.6518 • FID 66.57 • KID 0.0735 ± 0.0056 • CMMD 0.3145
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_1_pixel_triple_joint_norm_20260327-033611/checkpoint_latest.pt`

#### `24245657_24245657`  [🔗 chained]
*job 24245657 • idun-06-07 • A100 • epoch 684/1000 • chain 4/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24247235_24247235`  [🔗 chained]
*job 24247235 • idun-07-10 • A100 • epoch 806/1000 • chain 2/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24248097_24248097`  [🔗 chained]
*job 24248097 • idun-07-09 • A100 • epoch 821/1000 • chain 5/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24250164_24250164`  [✅ completed]
*job 24250164 • idun-06-02 • A100 • 8.16h training • epoch 1000/1000 • chain 3/20 • 41KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.395 • MS-SSIM 0.9514 • PSNR 34.81 dB • LPIPS 0.2612 • FID 79.03 • KID 0.0557 ± 0.0060 • CMMD 0.2320
  - **latest** ckpt (26 samples): MSE 0.4457 • MS-SSIM 0.9430 • PSNR 34.30 dB • LPIPS 0.2602 • FID 77.28 • KID 0.0511 ± 0.0049 • CMMD 0.2174
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24250963_24250963`  [🔗 chained]
*job 24250963 • idun-07-10 • A100 • epoch 959/1000 • chain 6/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24253509_24253509`  [✅ completed]
*job 24253509 • idun-06-03 • A100 • 3.2h training • epoch 1000/1000 • chain 7/20 • 20KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.288 • MS-SSIM 0.9864 • PSNR 33.30 dB • LPIPS 0.1150 • FID 60.02 • KID 0.0493 ± 0.0080 • CMMD 0.2899
  - **latest** ckpt (26 samples): MSE 3.105 • MS-SSIM 0.9863 • PSNR 33.96 dB • LPIPS 0.1160 • FID 57.24 • KID 0.0477 ± 0.0109 • CMMD 0.2514
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24262210_24262210`  [🔗 chained]
*job 24262210 • idun-06-03 • A100 • epoch 119/1000 • chain 0/40 • 30KB log*


#### `24262211_24262211`  [🔗 chained]
*job 24262211 • idun-06-07 • A100 • epoch 116/1000 • chain 0/20 • 30KB log*


#### `24262212_24262212`  [🔗 chained]
*job 24262212 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 29KB log*


#### `24269769_24269769`  [🔗 chained]
*job 24269769 • idun-01-05 • H100 • epoch 292/1000 • chain 1/40 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24269836_24269836`  [🔗 chained]
*job 24269836 • idun-01-04 • H100 • epoch 289/1000 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24269857_24269857`  [🔗 chained]
*job 24269857 • idun-06-03 • A100 • epoch 236/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24269897_24269897`  [🔗 chained]
*job 24269897 • idun-06-02 • A100 • epoch 113/500 • chain 0/20 • 58KB log*


#### `24269898_24269898`  [🔗 chained]
*job 24269898 • idun-06-01 • A100 • epoch 87/1000 • chain 0/20 • 22KB log*


#### `24270064_24270064`  [🔗 chained]
*job 24270064 • idun-07-10 • A100 • epoch 401/1000 • chain 2/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24270897_24270897`  [🔗 chained]
*job 24270897 • idun-07-08 • A100 • epoch 399/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24270899_24270899`  [🔗 chained]
*job 24270899 • idun-06-03 • A100 • epoch 353/500 • chain 2/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24271085_24271085`  [🔗 chained]
*job 24271085 • idun-01-05 • H100 • epoch 279/500 • chain 1/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24271087_24271087`  [🔗 chained]
*job 24271087 • idun-07-10 • A100 • epoch 197/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24271626_24271626`  [🔗 chained]
*job 24271626 • idun-07-10 • A100 • epoch 511/1000 • chain 3/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24271654_24271654`  [🔗 chained]
*job 24271654 • idun-07-09 • A100 • epoch 509/1000 • chain 3/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24271661_24271661`  [🔗 chained]
*job 24271661 • idun-07-09 • A100 • epoch 463/500 • chain 3/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24271681_24271681`  [🔗 chained]
*job 24271681 • idun-01-05 • H100 • epoch 172/1000 • chain 0/20 • 42KB log*


#### `24271682_24271682`  [🔗 chained]
*job 24271682 • idun-01-05 • H100 • epoch 170/1000 • chain 0/20 • 39KB log*


#### `24271719_24271719`  [🔗 chained]
*job 24271719 • idun-01-05 • H100 • epoch 444/500 • chain 2/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24271730_24271730`  [🔗 chained]
*job 24271730 • idun-07-08 • A100 • epoch 307/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24272484_24272484`  [🔗 chained]
*job 24272484 • idun-01-04 • H100 • epoch 686/1000 • chain 4/40 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24272485_24272485`  [🔗 chained]
*job 24272485 • idun-07-09 • A100 • epoch 619/1000 • chain 4/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24272487_24272487`  [🔗 chained]
*job 24272487 • idun-01-04 • H100 • epoch 126/500 • chain 0/20 • 33KB log*


#### `24272492_24272492`  [✅ completed]
*job 24272492 • idun-07-09 • A100 • 4.08h training • epoch 500/500 • chain 4/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003614 • MS-SSIM 0.9596 • PSNR 33.64 dB • LPIPS 0.5065 • FID 70.23 • KID 0.0559 ± 0.0078 • CMMD 0.2434
  - **latest** ckpt (26 samples): MSE 0.003257 • MS-SSIM 0.9574 • PSNR 33.31 dB • LPIPS 0.5367 • FID 72.30 • KID 0.0599 ± 0.0094 • CMMD 0.2741
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24272501_24272501`  [🔗 chained]
*job 24272501 • idun-01-04 • H100 • epoch 436/500 • chain 0/20 • 76KB log*


#### `24272522_24272522`  [🔗 chained]
*job 24272522 • idun-06-02 • A100 • epoch 292/1000 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24272524_24272524`  [🔗 chained]
*job 24272524 • idun-07-08 • A100 • epoch 280/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24272595_24272595`  [✅ completed]
*job 24272595 • idun-01-03 • H100 • 4.12h training • epoch 500/500 • chain 3/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002772 • MS-SSIM 0.9539 • PSNR 33.49 dB • LPIPS 0.9445 • FID 137.93 • KID 0.1472 ± 0.0065 • CMMD 0.4062
  - **latest** ckpt (26 samples): MSE 0.00264 • MS-SSIM 0.9622 • PSNR 34.13 dB • LPIPS 0.7044 • FID 88.85 • KID 0.0839 ± 0.0070 • CMMD 0.3084
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24272609_24272609`  [🔗 chained]
*job 24272609 • idun-01-05 • H100 • epoch 478/1000 • chain 3/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24274934_24274934`  [🔗 chained]
*job 24274934 • idun-06-02 • A100 • epoch 339/500 • chain 0/20 • 63KB log*


#### `24274953_24274953`  [🔗 chained]
*job 24274953 • idun-01-04 • H100 • epoch 861/1000 • chain 5/40 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24274955_24274955`  [🔗 chained]
*job 24274955 • idun-01-04 • H100 • epoch 795/1000 • chain 5/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24274957_24274957`  [🔗 chained]
*job 24274957 • idun-01-03 • H100 • epoch 297/500 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24274974_24274974`  [✅ completed]
*job 24274974 • idun-01-03 • H100 • 1.75h training • epoch 500/500 • chain 1/20 • 20KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00252 • MS-SSIM 0.9334 • PSNR 32.49 dB • LPIPS 0.8540 • FID 67.67 • KID 0.0723 ± 0.0077 • CMMD 0.2184
  - **latest** ckpt (26 samples): MSE 0.005569 • MS-SSIM 0.9198 • PSNR 30.74 dB • LPIPS 0.5132 • FID 35.56 • KID 0.0267 ± 0.0043 • CMMD 0.1265
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_02_pixel_bravo_weight_decay_128_20260404-144426/checkpoint_latest.pt`

#### `24274999_24274999`  [🔗 chained]
*job 24274999 • idun-01-04 • H100 • epoch 419/1000 • chain 2/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24275061_24275061`  [🔗 chained]
*job 24275061 • idun-01-03 • H100 • epoch 452/1000 • chain 2/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24275088_24275088`  [🔗 chained]
*job 24275088 • idun-01-05 • H100 • epoch 651/1000 • chain 4/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24275414_24275414`  [💥 oom_killed]
*job 24275414 • idun-06-01 • A100 • 5.76h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002792 • MS-SSIM 0.9370 • PSNR 32.57 dB • LPIPS 0.7480 • FID 57.25 • KID 0.0544 ± 0.0068 • CMMD 0.2225
  - **latest** ckpt (26 samples): MSE 0.003154 • MS-SSIM 0.9360 • PSNR 32.23 dB • LPIPS 0.5959
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_pixel_bravo_mixup_128_20260405-042205/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	354m51.968s
user	268m14.444s
sys	88m7.109s
[2026-04-05T23:56:29.313] error: Detected 1 oom_kill event in StepId=24275414.batch. Some of the step tasks have been OOM Killed.
```

#### `24275450_24275450`  [🔗 chained]
*job 24275450 • idun-07-08 • A100 • epoch 973/1000 • chain 6/40 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24275452_24275452`  [🔗 chained]
*job 24275452 • idun-08-01 • H100 • epoch 968/1000 • chain 6/20 • 45KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24275454_24275454`  [🔗 chained]
*job 24275454 • idun-08-01 • H100 • epoch 467/500 • chain 2/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24275460_24275460`  [🔗 chained]
*job 24275460 • idun-08-01 • H100 • epoch 593/1000 • chain 3/20 • 43KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24275551_24275551`  [🔗 chained]
*job 24275551 • idun-06-02 • A100 • epoch 286/1000 • chain 0/20 • 143KB log*


#### `24275552_24275552`  [🔗 chained]
*job 24275552 • idun-06-02 • A100 • epoch 206/2000 • chain 0/40 • 80KB log*


#### `24276244_24276244`  [❌ crashed]
*job 24276244 • idun-01-04 • H100 • 1.65h training • epoch 476/1000 • chain 3/20 • 13KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	102m1.530s
user	99m34.169s
sys	31m57.012s
```

#### `24276638_24276638`  [🔗 chained]
*job 24276638 • idun-01-05 • H100 • epoch 798/1000 • chain 5/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24276704_24276704`  [🔗 chained]
*job 24276704 • idun-01-03 • H100 • epoch 170/1000 • chain 0/20 • 55KB log*


#### `24276705_24276705`  [💥 oom_killed]
*job 24276705 • idun-07-09 • A100 • 3.93h training • epoch 500/500 • chain 2/20 • 26KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003494 • MS-SSIM 0.9385 • PSNR 32.02 dB • LPIPS 0.5698 • FID 60.61 • KID 0.0529 ± 0.0058 • CMMD 0.2814
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_pixel_bravo_mixup_128_20260405-042205/checkpoint_latest.pt`
**Traceback excerpt:**
```
gface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: d377e735-8d9c-4089-a2a8-b9f890ab774b)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/resolve/main/config.json
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a71fa5e2-6f1b-4e69-90d4-77e6649a000b)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_model.safetensors
...

real	244m33.104s
user	201m17.104s
sys	44m17.548s
[2026-04-06T09:39:20.032] error: Detected 1 oom_kill event in StepId=24276705.batch. Some of the step tasks have been OOM Killed.
```

#### `24276706_24276706`  [🔗 chained]
*job 24276706 • idun-01-04 • H100 • epoch 624/1000 • chain 4/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24276792_24276792`  [💥 oom_killed]
*job 24276792 • idun-06-02 • A100 • 2.73h training • epoch 1000/1000 • chain 7/40 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	168m32.841s
user	110m34.444s
sys	61m55.793s
[2026-04-06T12:00:16.518] error: Detected 2 oom_kill events in StepId=24276792.batch. Some of the step tasks have been OOM Killed.
```

#### `24276842_24276842`  [🔗 chained]
*job 24276842 • idun-07-09 • A100 • epoch 702/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24276843_24276843`  [💥 oom_killed]
*job 24276843 • idun-06-03 • A100 • 3.26h training • epoch 1000/1000 • chain 7/20 • 16KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004321 • MS-SSIM 0.9474 • PSNR 32.40 dB • LPIPS 0.4816 • FID 62.30 • KID 0.0478 ± 0.0073 • CMMD 0.1836
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	208m40.817s
user	138m34.530s
sys	72m0.487s
[2026-04-06T12:44:07.615] error: Detected 1 oom_kill event in StepId=24276843.batch. Some of the step tasks have been OOM Killed.
```

#### `24276844_24276844`  [✅ completed]
*job 24276844 • idun-06-03 • A100 • 3.54h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001808 • MS-SSIM 0.9573 • PSNR 33.72 dB • LPIPS 0.7451 • FID 54.31 • KID 0.0364 ± 0.0054 • CMMD 0.2795
  - **latest** ckpt (26 samples): MSE 0.003421 • MS-SSIM 0.9681 • PSNR 35.07 dB • LPIPS 0.5920 • FID 50.32 • KID 0.0377 ± 0.0059 • CMMD 0.2169
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24276863_24276863`  [🔗 chained]
*job 24276863 • idun-01-04 • H100 • epoch 690/1000 • chain 1/20 • 190KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24276866_24276866`  [🔗 chained]
*job 24276866 • idun-06-03 • A100 • epoch 414/2000 • chain 1/40 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24277127_24277127`  [🔗 chained]
*job 24277127 • idun-01-05 • H100 • epoch 971/1000 • chain 6/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24277645_24277645`  [🔗 chained]
*job 24277645 • idun-01-05 • H100 • epoch 340/1000 • chain 1/20 • 52KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24277739_24277739`  [🔗 chained]
*job 24277739 • idun-08-01 • H100 • epoch 794/1000 • chain 5/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24277906_24277906`  [🔗 chained]
*job 24277906 • idun-07-10 • A100 • epoch 812/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24277912_24277912`  [🔗 chained]
*job 24277912 • idun-07-10 • A100 • epoch 950/1000 • chain 2/20 • 126KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24277915_24277915`  [🔗 chained]
*job 24277915 • idun-01-04 • H100 • epoch 700/2000 • chain 2/40 • 108KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24278291_24278291`  [💥 oom_killed]
*job 24278291 • idun-01-05 • H100 • 2.08h training • epoch 1000/1000 • chain 7/20 • 14KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005083 • MS-SSIM 0.9184 • PSNR 29.88 dB • LPIPS 0.3982 • FID 50.69 • KID 0.0349 ± 0.0090 • CMMD 0.1653
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	137m4.532s
user	99m36.552s
sys	34m58.079s
[2026-04-07T08:16:15.786] error: Detected 1 oom_kill event in StepId=24278291.batch. Some of the step tasks have been OOM Killed.
```

#### `24278590_24278590`  [🔗 chained]
*job 24278590 • idun-06-03 • A100 • epoch 457/1000 • chain 2/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24278628_24278628`  [🔗 chained]
*job 24278628 • idun-06-03 • A100 • epoch 912/1000 • chain 6/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24278984_24278984`  [🔗 chained]
*job 24278984 • idun-01-03 • H100 • epoch 985/1000 • chain 6/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24278999_24278999`  [✅ completed]
*job 24278999 • idun-08-01 • H100 • 2.8h training • epoch 1000/1000 • chain 3/20 • 940KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01737 • MS-SSIM 0.9371 • PSNR 32.40 dB • LPIPS 0.7213 • FID 65.04 • KID 0.0546 ± 0.0078 • CMMD 0.2479
  - **latest** ckpt (26 samples): MSE 0.01111 • MS-SSIM 0.9415 • PSNR 32.92 dB • LPIPS 0.7381 • FID 71.03 • KID 0.0647 ± 0.0134 • CMMD 0.2529
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24279000_24279000`  [🔗 chained]
*job 24279000 • idun-08-01 • H100 • epoch 867/2000 • chain 3/40 • 2842KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24281999_24281999`  [🔗 chained]
*job 24281999 • idun-06-03 • A100 • epoch 575/1000 • chain 3/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24282106_24282106`  [💥 oom_killed]
*job 24282106 • idun-06-01 • A100 • 8.92h training • epoch 1000/1000 • chain 7/20 • 27KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004809 • MS-SSIM 0.9382 • PSNR 31.44 dB • LPIPS 0.4736 • FID 90.82 • KID 0.0889 ± 0.0092 • CMMD 0.3366
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`
**Traceback excerpt:**
```
ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 52358bf0-35d4-4416-a838-21ad86b32d6a)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_pytorch_model.bin
...

real	548m47.841s
user	345m55.073s
sys	207m41.298s
[2026-04-08T10:57:49.988] error: Detected 1 oom_kill event in StepId=24282106.batch. Some of the step tasks have been OOM Killed.
```

#### `24282576_24282576`  [✅ completed]
*job 24282576 • idun-06-07 • A100 • 1.55h training • epoch 1000/1000 • chain 7/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002987 • MS-SSIM 0.9508 • PSNR 32.59 dB • LPIPS 0.5341 • FID 66.73 • KID 0.0529 ± 0.0091 • CMMD 0.2423
  - **latest** ckpt (26 samples): MSE 0.00505 • MS-SSIM 0.9355 • PSNR 31.50 dB • LPIPS 0.4936 • FID 87.29 • KID 0.0819 ± 0.0114 • CMMD 0.3418
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24282670_24282670`  [🔗 chained]
*job 24282670 • idun-07-09 • A100 • epoch 1052/2000 • chain 4/40 • 76KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24282692_24282692`  [🔗 chained]
*job 24282692 • idun-06-07 • A100 • epoch 284/1000 • chain 0/20 • 168KB log*


#### `24284984_24284984`  [🔗 chained]
*job 24284984 • idun-06-07 • A100 • epoch 693/1000 • chain 4/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24285861_24285861`  [🔗 chained]
*job 24285861 • idun-06-04 • A100 • epoch 1252/2000 • chain 5/40 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24285928_24285928`  [🔗 chained]
*job 24285928 • idun-01-05 • H100 • epoch 804/1000 • chain 1/20 • 240KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_2_pixel_seg_67m_20260408-035801/checkpoint_latest.pt`

#### `24287503_24287503`  [🔗 chained]
*job 24287503 • idun-06-07 • A100 • epoch 810/1000 • chain 5/20 • 47KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24287525_24287525`  [🔗 chained]
*job 24287525 • idun-01-05 • H100 • epoch 1530/2000 • chain 6/40 • 126KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24287534_24287534`  [✅ completed]
*job 24287534 • idun-01-05 • H100 • 4.57h training • epoch 1000/1000 • chain 2/20 • 101KB log*

**Final test metrics:**
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_2_pixel_seg_67m_20260408-035801/checkpoint_latest.pt`

#### `24294144_24294144`  [💥 oom_killed]
*job 24294144 • idun-01-05 • H100 • 7.49h training • epoch 100/100 • chain 0/5 • 27KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005135 • MS-SSIM 0.9175 • PSNR 29.78 dB • LPIPS 0.4468 • FID 43.07 • KID 0.0256 ± 0.0066 • CMMD 0.1705
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	459m12.736s
user	355m48.694s
sys	114m53.337s
[2026-04-10T03:02:49.296] error: Detected 1 oom_kill event in StepId=24294144.batch. Some of the step tasks have been OOM Killed.
```

#### `24294145_24294145`  [✅ completed]
*job 24294145 • idun-01-04 • H100 • 9.83h training • epoch 100/100 • chain 0/5 • 29KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005167 • MS-SSIM 0.9229 • PSNR 30.53 dB • LPIPS 0.4719 • FID 90.71 • KID 0.1037 ± 0.0085 • CMMD 0.1817
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24294359_24294359`  [🔗 chained]
*job 24294359 • idun-07-08 • A100 • epoch 917/1000 • chain 6/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24294389_24294389`  [🔗 chained]
*job 24294389 • idun-06-05 • A100 • epoch 1737/2000 • chain 7/40 • 83KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24295517_24295517`  [✅ completed]
*job 24295517 • idun-07-08 • A100 • 9.23h training • epoch 1000/1000 • chain 7/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004286 • MS-SSIM 0.9386 • PSNR 32.64 dB • LPIPS 1.0223 • FID 81.96 • KID 0.0797 ± 0.0124 • CMMD 0.2929
  - **latest** ckpt (26 samples): MSE 0.004319 • MS-SSIM 0.9463 • PSNR 32.87 dB • LPIPS 0.7149 • FID 59.22 • KID 0.0509 ± 0.0116 • CMMD 0.2082
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24295987_24295987`  [🔗 chained]
*job 24295987 • idun-06-02 • A100 • epoch 1945/2000 • chain 8/40 • 84KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24297432_24297432`  [🔗 chained]
*job 24297432 • idun-06-02 • A100 • epoch 86/1000 • chain 0/20 • 24KB log*


#### `24297433_24297433`  [🔗 chained]
*job 24297433 • idun-06-04 • A100 • epoch 110/1000 • chain 0/20 • 33KB log*


#### `24297472_24297472`  [✅ completed]
*job 24297472 • idun-01-04 • H100 • 2.42h training • epoch 2000/2000 • chain 9/40 • 32KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006804 • MS-SSIM 0.9354 • PSNR 32.48 dB • LPIPS 0.9823 • FID 77.24 • KID 0.0716 ± 0.0134 • CMMD 0.2938
  - **latest** ckpt (26 samples): MSE 0.04487 • MS-SSIM 0.9130 • PSNR 31.89 dB • LPIPS 0.6392 • FID 48.08 • KID 0.0264 ± 0.0049 • CMMD 0.1795
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24298186_24298186`  [🔗 chained]
*job 24298186 • idun-07-10 • A100 • epoch 213/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24298187_24298187`  [🔗 chained]
*job 24298187 • idun-06-07 • A100 • epoch 171/1000 • chain 1/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24298538_24298538`  [✅ completed]
*job 24298538 • idun-06-01 • A100 • 11.51h training • epoch 100/100 • chain 0/5 • 94KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.006176 • MS-SSIM 0.9249 • PSNR 30.35 dB • LPIPS 0.4021 • FID 53.12 • KID 0.0379 ± 0.0059 • CMMD 0.1696
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24298633_24298633`  [🔗 chained]
*job 24298633 • idun-06-01 • A100 • epoch 119/1000 • chain 0/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24299033_24299033`  [🔗 chained]
*job 24299033 • idun-07-10 • A100 • epoch 316/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24299035_24299035`  [🔗 chained]
*job 24299035 • idun-06-07 • A100 • epoch 256/1000 • chain 2/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24299399_24299399`  [🔗 chained]
*job 24299399 • idun-01-03 • H100 • epoch 294/1000 • chain 1/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24299653_24299653`  [🔗 chained]
*job 24299653 • idun-01-03 • H100 • epoch 170/1000 • chain 0/20 • 71KB log*


#### `24299684_24299684`  [🔗 chained]
*job 24299684 • idun-06-05 • A100 • epoch 117/1000 • chain 0/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24299685_24299685`  [🔗 chained]
*job 24299685 • idun-07-08 • A100 • epoch 108/1000 • chain 0/20 • 97KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24301332_24301332`  [🔗 chained]
*job 24301332 • idun-01-04 • H100 • epoch 487/1000 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24301334_24301334`  [🔗 chained]
*job 24301334 • idun-01-03 • H100 • epoch 383/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24303660_24303660`  [🔗 chained]
*job 24303660 • idun-07-10 • A100 • epoch 404/1000 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24303687_24303687`  [🔗 chained]
*job 24303687 • idun-01-04 • H100 • epoch 90/500 • chain 0/20 • 21KB log*


#### `24303688_24303688`  [🔗 chained]
*job 24303688 • idun-07-10 • A100 • epoch 59/500 • chain 0/20 • 17KB log*


#### `24303775_24303775`  [🔗 chained]
*job 24303775 • idun-06-05 • A100 • epoch 114/500 • chain 0/20 • 31KB log*


#### `24303782_24303782`  [🔗 chained]
*job 24303782 • idun-06-06 • A100 • epoch 282/1000 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24303825_24303825`  [🔗 chained]
*job 24303825 • idun-06-05 • A100 • epoch 235/1000 • chain 1/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24303833_24303833`  [🔗 chained]
*job 24303833 • idun-07-10 • A100 • epoch 217/1000 • chain 1/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24304029_24304029`  [🔗 chained]
*job 24304029 • idun-01-04 • H100 • epoch 657/1000 • chain 4/20 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24304062_24304062`  [🔗 chained]
*job 24304062 • idun-01-04 • H100 • epoch 510/1000 • chain 4/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24304474_24304474`  [🔗 chained]
*job 24304474 • idun-06-06 • A100 • epoch 523/1000 • chain 3/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24304514_24304514`  [🔗 chained]
*job 24304514 • idun-06-05 • A100 • epoch 167/500 • chain 1/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24305858_24305858`  [🔗 chained]
*job 24305858 • idun-01-04 • H100 • epoch 149/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24305987_24305987`  [🔗 chained]
*job 24305987 • idun-01-04 • H100 • epoch 236/500 • chain 1/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24305993_24305993`  [🔗 chained]
*job 24305993 • idun-01-04 • H100 • epoch 453/1000 • chain 2/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24306093_24306093`  [🔗 chained]
*job 24306093 • idun-01-04 • H100 • epoch 408/1000 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24306132_24306132`  [🔗 chained]
*job 24306132 • idun-01-04 • H100 • epoch 389/1000 • chain 2/20 • 151KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24308606_24308606`  [🔗 chained]
*job 24308606 • idun-07-09 • A100 • epoch 760/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24308625_24308625`  [🔗 chained]
*job 24308625 • idun-07-09 • A100 • epoch 590/1000 • chain 5/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24308833_24308833`  [🔗 chained]
*job 24308833 • idun-07-09 • A100 • epoch 634/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24308908_24308908`  [🔗 chained]
*job 24308908 • idun-06-06 • A100 • epoch 243/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24308951_24308951`  [🔗 chained]
*job 24308951 • idun-07-10 • A100 • epoch 75/500 • chain 0/20 • 21KB log*


#### `24309326_24309326`  [🔗 chained]
*job 24309326 • idun-01-04 • H100 • epoch 624/1000 • chain 3/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24309327_24309327`  [🔗 chained]
*job 24309327 • idun-07-09 • A100 • epoch 345/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24309328_24309328`  [🔗 chained]
*job 24309328 • idun-06-01 • A100 • epoch 224/500 • chain 2/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24309382_24309382`  [🔗 chained]
*job 24309382 • idun-07-08 • A100 • epoch 518/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24309383_24309383`  [🔗 chained]
*job 24309383 • idun-06-04 • A100 • epoch 507/1000 • chain 3/20 • 107KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24310828_24310828`  [🔗 chained]
*job 24310828 • idun-07-08 • A100 • epoch 669/1000 • chain 6/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24310829_24310829`  [🔗 chained]
*job 24310829 • idun-06-05 • A100 • epoch 872/1000 • chain 6/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24310900_24310900`  [🔗 chained]
*job 24310900 • idun-06-06 • A100 • epoch 752/1000 • chain 5/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24311036_24311036`  [🔗 chained]
*job 24311036 • idun-06-05 • A100 • epoch 120/1000 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24311411_24311411`  [🔗 chained]
*job 24311411 • idun-06-06 • A100 • epoch 324/500 • chain 3/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24311744_24311744`  [🔗 chained]
*job 24311744 • idun-01-04 • H100 • epoch 186/500 • chain 1/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630/checkpoint_latest.pt`

#### `24313294_24313294`  [🔗 chained]
*job 24313294 • idun-07-10 • A100 • epoch 75/500 • chain 0/20 • 19KB log*


#### `24313642_24313642`  [🔗 chained]
*job 24313642 • idun-06-07 • A100 • epoch 736/1000 • chain 4/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24313728_24313728`  [🔗 chained]
*job 24313728 • idun-06-05 • A100 • epoch 458/500 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24313866_24313866`  [🔗 chained]
*job 24313866 • idun-01-04 • H100 • epoch 298/500 • chain 3/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24313917_24313917`  [🔗 chained]
*job 24313917 • idun-07-10 • A100 • epoch 628/1000 • chain 4/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24313978_24313978`  [🔗 chained]
*job 24313978 • idun-06-04 • A100 • epoch 624/1000 • chain 4/20 • 107KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24313980_24313980`  [🔗 chained]
*job 24313980 • idun-06-04 • A100 • epoch 753/1000 • chain 7/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24314133_24314133`  [🔗 chained]
*job 24314133 • idun-07-09 • A100 • epoch 976/1000 • chain 7/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24314507_24314507`  [🔗 chained]
*job 24314507 • idun-07-10 • A100 • epoch 863/1000 • chain 6/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24314543_24314543`  [🔗 chained]
*job 24314543 • idun-07-10 • A100 • epoch 231/1000 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24314556_24314556`  [🔗 chained]
*job 24314556 • idun-07-10 • A100 • epoch 380/500 • chain 4/20 • 18KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24314691_24314691`  [🔗 chained]
*job 24314691 • idun-07-10 • A100 • epoch 265/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630/checkpoint_latest.pt`

#### `24314954_24314954`  [🔗 chained]
*job 24314954 • idun-06-07 • A100 • epoch 115/500 • chain 0/20 • 49KB log*


#### `24314955_24314955`  [🔗 chained]
*job 24314955 • idun-06-07 • A100 • epoch 277/500 • chain 0/20 • 107KB log*


#### `24315002_24315002`  [🔗 chained]
*job 24315002 • idun-06-07 • A100 • epoch 195/500 • chain 1/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2b_rflow_bridge_restoration_17m_20260415-122714/checkpoint_latest.pt`

#### `24315035_24315035`  [💥 oom_killed]
*job 24315035 • idun-07-09 • A100 • 6.81h training • epoch 24/500 • chain 0/20 • 8KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
```

#### `24315037_24315037`  [💥 oom_killed]
*job 24315037 • idun-07-09 • A100 • 6.86h training • epoch 24/500 • chain 0/20 • 8KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
```

#### `24315038_24315038`  [💥 oom_killed]
*job 24315038 • idun-01-04 • H100 • 5.6h training • epoch 24/500 • chain 0/20 • 9KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24315064_24315064`  [⚠️ truncated]
*job 24315064 • idun-07-09 • A100 • epoch 59/500 • chain 0/20 • 15KB log*


#### `24315065_24315065`  [⚠️ truncated]
*job 24315065 • idun-07-09 • A100 • epoch 49/500 • chain 0/20 • 20KB log*


#### `24315083_24315083`  [⚠️ truncated]
*job 24315083 • idun-06-01 • A100 • epoch 74/500 • chain 0/20 • 18KB log*


#### `24315089_24315089`  [🔗 chained]
*job 24315089 • idun-01-04 • H100 • epoch 907/1000 • chain 5/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24315090_24315090`  [✅ completed]
*job 24315090 • idun-01-04 • H100 • 4.26h training • epoch 500/500 • chain 4/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004296 • MS-SSIM 0.9514 • PSNR 33.26 dB • LPIPS 1.1520 • FID 143.44 • KID 0.1907 ± 0.0132 • CMMD 0.2380
  - **latest** ckpt (26 samples): MSE 0.004361 • MS-SSIM 0.9520 • PSNR 33.21 dB • LPIPS 1.2162 • FID 140.32 • KID 0.1903 ± 0.0110 • CMMD 0.2237
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24315096_24315096`  [⚠️ truncated]
*job 24315096 • idun-06-05 • A100 • epoch 322/500 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24315097_24315097`  [🔗 chained]
*job 24315097 • idun-07-10 • A100 • epoch 739/1000 • chain 5/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24315098_24315098`  [🔗 chained]
*job 24315098 • idun-06-07 • A100 • epoch 741/1000 • chain 5/20 • 100KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24315099_24315099`  [🔗 chained]
*job 24315099 • idun-06-01 • A100 • epoch 839/1000 • chain 8/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24315101_24315101`  [✅ completed]
*job 24315101 • idun-06-07 • A100 • 2.58h training • epoch 1000/1000 • chain 8/20 • 14KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002314 • MS-SSIM 0.9713 • PSNR 35.45 dB • LPIPS 0.5361 • FID 67.76 • KID 0.0583 ± 0.0039 • CMMD 0.2307
  - **latest** ckpt (26 samples): MSE 0.005111 • MS-SSIM 0.9439 • PSNR 32.01 dB • LPIPS 0.3772 • FID 41.15 • KID 0.0260 ± 0.0042 • CMMD 0.1980
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24315535_24315535`  [🔗 chained]
*job 24315535 • idun-07-09 • A100 • epoch 341/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24315536_24315536`  [🔗 chained]
*job 24315536 • idun-07-09 • A100 • epoch 973/1000 • chain 7/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24315690_24315690`  [🔗 chained]
*job 24315690 • idun-07-09 • A100 • epoch 221/500 • chain 1/20 • 42KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24315691_24315691`  [✅ completed]
*job 24315691 • idun-07-09 • A100 • 11.0h training • epoch 500/500 • chain 1/20 • 74KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007023 • MS-SSIM 0.9283 • PSNR 30.95 dB • LPIPS 0.8128 • FID 195.38 • KID 0.2653 ± 0.0182 • CMMD 0.4760
  - **latest** ckpt (26 samples): MSE 0.0241 • MS-SSIM 0.9151 • PSNR 29.89 dB • LPIPS 0.4458 • FID 178.97 • KID 0.2269 ± 0.0159 • CMMD 0.4156
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_s_p4_20260416-005333/checkpoint_latest.pt`

#### `24315963_24315963`  [🔗 chained]
*job 24315963 • idun-06-06 • A100 • epoch 116/1000 • chain 0/20 • 48KB log*


#### `24315964_24315964`  [🔗 chained]
*job 24315964 • idun-06-01 • A100 • epoch 277/1000 • chain 0/20 • 116KB log*


#### `24316690_24316690`  [✅ completed]
*job 24316690 • idun-01-03 • H100 • 6.43h training • epoch 1000/1000 • chain 6/20 • 30KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002472 • MS-SSIM 0.9638 • PSNR 34.62 dB • LPIPS 0.7534 • FID 108.28 • KID 0.1273 ± 0.0080 • CMMD 0.2461
  - **latest** ckpt (26 samples): MSE 0.00512 • MS-SSIM 0.9314 • PSNR 30.77 dB • LPIPS 0.4605 • FID 65.81 • KID 0.0652 ± 0.0071 • CMMD 0.1940
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24316691_24316691`  [✅ completed]
*job 24316691 • idun-01-04 • H100 • 11.56h training • epoch 500/500 • chain 0/20 • 191KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.05237 • MS-SSIM 0.4655 • PSNR 21.96 dB • LPIPS 1.7800 • FID 346.41 • KID 0.4726 ± 0.0159 • CMMD 0.6346
  - **latest** ckpt (26 samples): MSE 0.0537 • MS-SSIM 0.4721 • PSNR 22.17 dB • LPIPS 1.7801 • FID 349.33 • KID 0.4697 ± 0.0127 • CMMD 0.6382

#### `24316692_24316692`  [🔗 chained]
*job 24316692 • idun-07-09 • A100 • epoch 148/500 • chain 0/20 • 41KB log*


#### `24316927_24316927`  [🔗 chained]
*job 24316927 • idun-07-09 • A100 • epoch 849/1000 • chain 6/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24316958_24316958`  [🔗 chained]
*job 24316958 • idun-01-03 • H100 • epoch 914/1000 • chain 6/20 • 143KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24316963_24316963`  [🔗 chained]
*job 24316963 • idun-01-03 • H100 • epoch 965/1000 • chain 9/20 • 30KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24316991_24316991`  [🔗 chained]
*job 24316991 • idun-06-05 • A100 • epoch 459/1000 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24316992_24316992`  [💥 oom_killed]
*job 24316992 • idun-06-05 • A100 • 2.71h training • epoch 1000/1000 • chain 8/20 • 15KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.007728 • MS-SSIM 0.9022 • PSNR 28.98 dB • LPIPS 0.3477 • FID 44.08 • KID 0.0172 ± 0.0039 • CMMD 0.1499
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
/var/slurm_spool/job24316992/slurm_script: line 91: 139790 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=1000 training.learning_rate=1e-4 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	174m47.036s
user	113m35.289s
sys	64m20.278s
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/multiprocessing/resource_sharer.py", line 138, in _serve
[2026-04-17T09:06:11.316] error: Detected 1 oom_kill event in StepId=24316992.batch. Some of the step tasks have been OOM Killed.
```

#### `24316998_24316998`  [🔗 chained]
*job 24316998 • idun-06-05 • A100 • epoch 336/500 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24317029_24317029`  [🔗 chained]
*job 24317029 • idun-06-06 • A100 • epoch 231/1000 • chain 1/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_l_p4_gc_20260416-203245/checkpoint_latest.pt`

#### `24317062_24317062`  [🔗 chained]
*job 24317062 • idun-06-01 • A100 • epoch 558/1000 • chain 1/20 • 88KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_s_p4_20260416-210331/checkpoint_latest.pt`

#### `24318602_24318602`  [🔗 chained]
*job 24318602 • idun-01-04 • H100 • epoch 376/500 • chain 1/20 • 68KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260417-031657/checkpoint_latest.pt`

#### `24318666_24318666`  [🔗 chained]
*job 24318666 • idun-06-04 • A100 • epoch 965/1000 • chain 7/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24318712_24318712`  [💥 oom_killed]
*job 24318712 • idun-07-08 • A100 • 9.49h training • epoch 1000/1000 • chain 7/20 • 79KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005326 • MS-SSIM 0.9108 • PSNR 29.46 dB • LPIPS 0.4418 • FID 44.29 • KID 0.0225 ± 0.0053 • CMMD 0.1562
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24318712/slurm_script: line 93: 1545972 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=1000 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.perceptual_weight=0.1 training.perceptual_max_timestep=250 training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	585m57.251s
user	424m46.393s
sys	165m35.549s
[2026-04-18T09:15:19.614] error: Detected 1 oom_kill event in StepId=24318712.batch. Some of the step tasks have been OOM Killed.
```

#### `24318753_24318753`  [✅ completed]
*job 24318753 • idun-07-08 • A100 • 5.21h training • epoch 1000/1000 • chain 10/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001339 • MS-SSIM 0.9875 • PSNR 37.40 dB • LPIPS 0.2908 • FID 46.57 • KID 0.0262 ± 0.0049 • CMMD 0.1679
  - **latest** ckpt (26 samples): MSE 0.001936 • MS-SSIM 0.9890 • PSNR 37.72 dB • LPIPS 0.2615 • FID 47.07 • KID 0.0265 ± 0.0064 • CMMD 0.1642
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24318757_24318757`  [🔗 chained]
*job 24318757 • idun-07-08 • A100 • epoch 570/1000 • chain 4/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24318809_24318809`  [🔗 chained]
*job 24318809 • idun-07-08 • A100 • epoch 441/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24318831_24318831`  [🔗 chained]
*job 24318831 • idun-06-06 • A100 • epoch 116/500 • chain 0/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24318832_24318832`  [🔗 chained]
*job 24318832 • idun-06-05 • A100 • epoch 114/500 • chain 0/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24318948_24318948`  [🔗 chained]
*job 24318948 • idun-01-04 • H100 • epoch 417/1000 • chain 2/20 • 82KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_l_p4_gc_20260416-203245/checkpoint_latest.pt`

#### `24318972_24318972`  [🔗 chained]
*job 24318972 • idun-07-08 • A100 • epoch 802/1000 • chain 2/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_s_p4_20260416-210331/checkpoint_latest.pt`

#### `24319138_24319138`  [✅ completed]
*job 24319138 • idun-06-01 • A100 • 9.11h training • epoch 500/500 • chain 2/20 • 80KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006321 • MS-SSIM 0.9198 • PSNR 30.76 dB • LPIPS 1.4442 • FID 229.45 • KID 0.2909 ± 0.0148 • CMMD 0.5112
  - **latest** ckpt (26 samples): MSE 0.006731 • MS-SSIM 0.9150 • PSNR 30.61 dB • LPIPS 1.4622 • FID 236.27 • KID 0.3020 ± 0.0156 • CMMD 0.4990
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260417-031657/checkpoint_latest.pt`

#### `24319180_24319180`  [✅ completed]
*job 24319180 • idun-06-05 • A100 • 3.6h training • epoch 1000/1000 • chain 8/20 • 19KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.00767 • MS-SSIM 0.9081 • PSNR 29.34 dB • LPIPS 0.4096 • FID 46.48 • KID 0.0275 ± 0.0078 • CMMD 0.1760
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24319336_24319336`  [✅ completed]
*job 24319336 • idun-01-03 • H100 • 3.86h training • epoch 500/500 • chain 4/20 • 66KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004804 • MS-SSIM 0.9343 • PSNR 31.55 dB • LPIPS 1.4403 • FID 220.82 • KID 0.3037 ± 0.0172 • CMMD 0.5052
  - **latest** ckpt (26 samples): MSE 0.0052 • MS-SSIM 0.9106 • PSNR 29.34 dB • LPIPS 0.3282 • FID 161.57 • KID 0.2066 ± 0.0159 • CMMD 0.3652
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24319337_24319337`  [🔗 chained]
*job 24319337 • idun-07-08 • A100 • epoch 680/1000 • chain 5/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24319346_24319346`  [🔗 chained]
*job 24319346 • idun-07-08 • A100 • epoch 223/500 • chain 1/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24319503_24319503`  [🔗 chained]
*job 24319503 • idun-06-07 • A100 • epoch 229/500 • chain 1/20 • 71KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24323474_24323474`  [🔗 chained]
*job 24323474 • idun-01-03 • H100 • epoch 391/500 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24323475_24323475`  [🔗 chained]
*job 24323475 • idun-06-04 • A100 • epoch 798/1000 • chain 6/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24323478_24323478`  [🔗 chained]
*job 24323478 • idun-06-04 • A100 • epoch 345/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24324706_24324706`  [🔗 chained]
*job 24324706 • idun-06-07 • A100 • epoch 114/1000 • chain 0/20 • 45KB log*


#### `24324707_24324707`  [🔗 chained]
*job 24324707 • idun-06-07 • A100 • epoch 274/1000 • chain 0/20 • 104KB log*


#### `24324708_24324708`  [🔗 chained]
*job 24324708 • idun-06-01 • A100 • epoch 162/500 • chain 0/20 • 45KB log*


#### `24324709_24324709`  [🔗 chained]
*job 24324709 • idun-06-01 • A100 • epoch 413/500 • chain 0/20 • 123KB log*


#### `24325059_24325059`  [✅ completed]
*job 24325059 • idun-06-04 • A100 • 11.19h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005893 • MS-SSIM 0.9270 • PSNR 30.59 dB • LPIPS 0.3724 • FID 69.58 • KID 0.0637 ± 0.0078 • CMMD 0.1682
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24325060_24325060`  [🔗 chained]
*job 24325060 • idun-07-08 • A100 • epoch 908/1000 • chain 7/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24325084_24325084`  [🔗 chained]
*job 24325084 • idun-06-04 • A100 • epoch 460/500 • chain 3/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24325324_24325324`  [💥 oom_killed]
*job 24325324 • idun-08-01 • H100 • 11.04h training • epoch 150/150 • chain 0/20 • 38KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004514 • MS-SSIM 0.9310 • PSNR 30.63 dB • LPIPS 0.4476 • FID 96.02 • KID 0.1156 ± 0.0143 • CMMD 0.2001
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24325324/slurm_script: line 101: 458587 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=150 training.learning_rate=2e-5 training.warmup_epochs=10 training.eta_min=1e-7 training.gradient_clip_norm=0.5 training.perceptual_weight=0.5 'training.perceptual_t_schedule=[0.05,0.20,0.70]' training.focal_frequency_weight=0.7 'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	674m32.944s
user	518m29.499s
sys	169m25.572s
[2026-04-20T12:16:12.543] error: Detected 2 oom_kill events in StepId=24325324.batch. Some of the step tasks have been OOM Killed.
```

#### `24325358_24325358`  [🔗 chained]
*job 24325358 • idun-01-03 • H100 • epoch 299/1000 • chain 1/20 • 66KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24325359_24325359`  [🔗 chained]
*job 24325359 • idun-06-07 • A100 • epoch 548/1000 • chain 1/20 • 83KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24325360_24325360`  [🔗 chained]
*job 24325360 • idun-06-07 • A100 • epoch 324/500 • chain 1/20 • 50KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/checkpoint_latest.pt`

#### `24325363_24325363`  [✅ completed]
*job 24325363 • idun-07-08 • A100 • 3.14h training • epoch 500/500 • chain 1/20 • 33KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.03708 • MS-SSIM 0.4959 • PSNR 22.78 dB • LPIPS 1.7538 • FID 334.70 • KID 0.4441 ± 0.0148 • CMMD 0.6253
  - **latest** ckpt (26 samples): MSE 0.04212 • MS-SSIM 0.5510 • PSNR 23.96 dB • LPIPS 1.7220 • FID 349.43 • KID 0.4688 ± 0.0136 • CMMD 0.6237
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_3_mamba_5s_p4_20260419-162046/checkpoint_latest.pt`

#### `24325380_24325380`  [✅ completed]
*job 24325380 • idun-06-01 • A100 • 9.15h training • epoch 1000/1000 • chain 8/20 • 25KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.01025 • MS-SSIM 0.9068 • PSNR 29.28 dB • LPIPS 0.3977 • FID 48.22 • KID 0.0284 ± 0.0058 • CMMD 0.1632
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24325381_24325381`  [✅ completed]
*job 24325381 • idun-06-01 • A100 • 4.11h training • epoch 500/500 • chain 4/20 • 18KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005817 • MS-SSIM 0.9255 • PSNR 30.55 dB • LPIPS 0.4031 • FID 68.98 • KID 0.0631 ± 0.0085 • CMMD 0.1734
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24326102_24326102`  [🔗 chained]
*job 24326102 • idun-01-03 • H100 • epoch 484/1000 • chain 2/20 • 58KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24326194_24326194`  [💥 oom_killed]
*job 24326194 • idun-01-05 • H100 • chain 2/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/models/mamba_diff.py", line 528, in forward
    x = self.encoder_stages[i](x, c)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*a
```

#### `24326398_24326398`  [🔗 chained]
*job 24326398 • idun-08-01 • H100 • epoch 940/1000 • chain 0/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24327639_24327639`  [❌ crashed]
*job 24327639 • idun-07-08 • A100 • epoch 352/500 • chain 0/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
         ^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/regional_losses.json'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[2026-04-21T02:02:40.652] error: *** JOB 24327639 ON idun-07-08 CANCELLED AT 2026-04-21T02:02:40 DUE to SIGNAL Terminated ***
```

#### `24327729_24327729`  [🔗 chained]
*job 24327729 • idun-01-03 • H100 • epoch 173/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327730_24327730`  [🔗 chained]
*job 24327730 • idun-01-03 • H100 • epoch 171/500 • chain 0/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327732_24327732`  [🔗 chained]
*job 24327732 • idun-08-01 • H100 • epoch 167/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327859_24327859`  [🔗 chained]
*job 24327859 • idun-08-01 • H100 • epoch 142/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327860_24327860`  [🔗 chained]
*job 24327860 • idun-01-03 • H100 • epoch 173/500 • chain 0/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327861_24327861`  [🔗 chained]
*job 24327861 • idun-07-09 • A100 • epoch 109/500 • chain 0/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327864_24327864`  [🔗 chained]
*job 24327864 • idun-07-10 • A100 • epoch 109/500 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327887_24327887`  [🔗 chained]
*job 24327887 • idun-07-10 • A100 • epoch 590/1000 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24327896_24327896`  [✅ completed]
*job 24327896 • idun-07-09 • A100 • 2.97h training • epoch 1000/1000 • chain 1/20 • 24KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.01837 • MS-SSIM 0.9197 • PSNR 30.02 dB • LPIPS 0.3322 • FID 144.59 • KID 0.1791 ± 0.0106 • CMMD 0.3520
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24328879_24328879`  [🔗 chained]
*job 24328879 • idun-07-10 • A100 • epoch 281/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24329911_24329911`  [🔗 chained]
*job 24329911 • idun-01-03 • H100 • epoch 343/500 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24330209_24330209`  [🔗 chained]
*job 24330209 • idun-07-09 • A100 • epoch 274/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24330471_24330471`  [🔗 chained]
*job 24330471 • idun-07-08 • A100 • epoch 251/500 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24330559_24330559`  [🔗 chained]
*job 24330559 • idun-01-03 • H100 • epoch 345/500 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_2_scoreaug_structure_20260421-105310/checkpoint_latest.pt`

#### `24330580_24330580`  [🔗 chained]
*job 24330580 • idun-01-03 • H100 • epoch 282/500 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`

#### `24330583_24330583`  [🔗 chained]
*job 24330583 • idun-06-05 • A100 • epoch 704/1000 • chain 4/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24330584_24330584`  [🔗 chained]
*job 24330584 • idun-08-01 • H100 • epoch 250/500 • chain 1/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24330608_24330608`  [🔗 chained]
*job 24330608 • idun-06-05 • A100 • epoch 399/500 • chain 2/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24330613_24330613`  [💥 oom_killed]
*job 24330613 • idun-06-01 • A100 • 6.98h training • epoch 60/60 • chain 0/20 • 20KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.006013 • MS-SSIM 0.9343 • PSNR 31.37 dB • LPIPS 0.3986 • FID 90.80 • KID 0.1022 ± 0.0114 • CMMD 0.1978
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24330613/slurm_script: line 109: 3109653 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=60 training.learning_rate=1e-5 training.warmup_epochs=3 training.eta_min=1e-7 training.gradient_clip_norm=0.5 training.perceptual_weight=0.5 'training.perceptual_t_schedule=[0.05,0.20,0.70]' training.focal_frequency_weight=0.7 'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' 'training.mse_t_schedule=[0.05,0.15,1.0]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	428m20.930s
user	277m17.674s
sys	156m56.871s
[2026-04-22T16:29:00.437] error: Detected 1 oom_kill event in StepId=24330613.batch. Some of the step tasks have been OOM Killed.
```

#### `24330636_24330636`  [🔗 chained]
*job 24330636 • idun-06-05 • A100 • epoch 456/500 • chain 2/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24330682_24330682`  [🔗 chained]
*job 24330682 • idun-06-01 • A100 • epoch 390/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24332507_24332507`  [🔗 chained]
*job 24332507 • idun-07-09 • A100 • epoch 361/500 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24332650_24332650`  [🔗 chained]
*job 24332650 • idun-06-05 • A100 • epoch 99/100 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt`

#### `24333401_24333401`  [🔗 chained]
*job 24333401 • idun-09-02 • H200 • epoch 462/500 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`

#### `24333402_24333402`  [💥 oom_killed]
*job 24333402 • idun-09-02 • H200 • 10.27h training • epoch 500/500 • chain 2/20 • 40KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007519 • MS-SSIM 0.9338 • PSNR 31.05 dB • LPIPS 0.3894 • FID 54.39 • KID 0.0362 ± 0.0069 • CMMD 0.1816
  - **latest** ckpt (26 samples): MSE 0.007733 • MS-SSIM 0.9241 • PSNR 30.26 dB • LPIPS 0.4485
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_2_scoreaug_structure_20260421-105310/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24333402/slurm_script: line 96: 19389 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=500 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.score_aug.enabled=true 'training.scoreaug_t_schedule=[0.70,0.85,1.0]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	626m19.223s
user	459m20.353s
sys	174m16.112s
[2026-04-23T08:59:57.217] error: Detected 1 oom_kill event in StepId=24333402.batch. Some of the step tasks have been OOM Killed.
```

#### `24333414_24333414`  [🔗 chained]
*job 24333414 • idun-09-02 • H200 • epoch 913/1000 • chain 5/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24333461_24333461`  [🔗 chained]
*job 24333461 • idun-09-02 • H200 • epoch 429/500 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24334406_24334406`  [✅ completed]
*job 24334406 • idun-09-02 • H200 • 6.72h training • epoch 500/500 • chain 3/20 • 28KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003164 • MS-SSIM 0.9553 • PSNR 33.20 dB • LPIPS 0.5584 • FID 68.42 • KID 0.0458 ± 0.0054 • CMMD 0.2512
  - **latest** ckpt (26 samples): MSE 0.002815 • MS-SSIM 0.9637 • PSNR 33.73 dB • LPIPS 0.4990 • FID 70.27 • KID 0.0545 ± 0.0071 • CMMD 0.2545
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24334407_24334407`  [✅ completed]
*job 24334407 • idun-09-02 • H200 • 7.41h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003855 • MS-SSIM 0.9374 • PSNR 31.45 dB • LPIPS 0.4567 • FID 50.17 • KID 0.0347 ± 0.0062 • CMMD 0.1746
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24334408_24334408`  [✅ completed]
*job 24334408 • idun-09-02 • H200 • 2.94h training • epoch 500/500 • chain 3/20 • 19KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004374 • MS-SSIM 0.9423 • PSNR 31.97 dB • LPIPS 0.4540 • FID 57.25 • KID 0.0397 ± 0.0082 • CMMD 0.1670
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24334547_24334547`  [✅ completed]
*job 24334547 • idun-08-01 • H100 • 8.96h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334549_24334549`  [✅ completed]
*job 24334549 • idun-09-02 • H200 • 8.23h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334588_24334588`  [✅ completed]
*job 24334588 • idun-09-02 • H200 • 7.66h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334610_24334610`  [✅ completed]
*job 24334610 • idun-09-02 • H200 • 9.24h training • epoch 500/500 • chain 3/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003243 • MS-SSIM 0.9334 • PSNR 32.08 dB • LPIPS 0.6479 • FID 125.83 • KID 0.1346 ± 0.0108 • CMMD 0.4178
  - **latest** ckpt (26 samples): MSE 0.004973 • MS-SSIM 0.9414 • PSNR 31.65 dB • LPIPS 0.4449 • FID 45.36 • KID 0.0261 ± 0.0048 • CMMD 0.1611
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24334832_24334832`  [✅ completed]
*job 24334832 • idun-09-02 • H200 • 0.11h training • epoch 100/100 • chain 1/20 • 10KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003924 • MS-SSIM 0.9236 • PSNR 30.26 dB • LPIPS 0.3856 • FID 81.49 • KID 0.0880 ± 0.0097 • CMMD 0.1803
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp40_patchgan_20260422-220017/checkpoint_latest.pt`

#### `24334911_24334911`  [💥 oom_killed]
*job 24334911 • idun-09-02 • H200 • 2.54h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003958 • MS-SSIM 0.9559 • PSNR 33.28 dB • LPIPS 0.4711 • FID 60.03 • KID 0.0448 ± 0.0114 • CMMD 0.1609
  - **latest** ckpt (26 samples): MSE 0.004998 • MS-SSIM 0.9469 • PSNR 32.46 dB • LPIPS 0.4508
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24334911/slurm_script: line 95: 622642 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=500 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.score_aug.enabled=true training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	161m16.088s
user	118m51.179s
sys	43m51.984s
[2026-04-23T14:29:31.120] error: Detected 1 oom_kill event in StepId=24334911.batch. Some of the step tasks have been OOM Killed.
```

#### `24334912_24334912`  [✅ completed]
*job 24334912 • idun-09-02 • H200 • 4.73h training • epoch 500/500 • chain 3/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002311 • MS-SSIM 0.9599 • PSNR 33.31 dB • LPIPS 0.5531 • FID 70.99 • KID 0.0525 ± 0.0082 • CMMD 0.2511
  - **latest** ckpt (26 samples): MSE 0.002523 • MS-SSIM 0.9560 • PSNR 33.11 dB • LPIPS 0.5534 • FID 73.28 • KID 0.0540 ± 0.0050 • CMMD 0.2858
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24334913_24334913`  [✅ completed]
*job 24334913 • idun-09-02 • H200 • 4.97h training • epoch 1000/1000 • chain 6/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.008194 • MS-SSIM 0.9118 • PSNR 29.21 dB • LPIPS 0.2896 • FID 148.60 • KID 0.1831 ± 0.0131 • CMMD 0.3263
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24336018_24336018`  [🔗 chained]
*job 24336018 • idun-01-03 • H100 • epoch 171/1000 • chain 0/20 • 41KB log*


#### `24336117_24336117`  [✅ completed]
*job 24336117 • idun-09-02 • H200 • 6.86h training • epoch 80/80 • chain 0/20 • 18KB log*


#### `24336401_24336401`  [⚠️ truncated]
*job 24336401 • idun-01-03 • H100 • chain 0/20 • 2KB log*


#### `24336457_24336457`  [🔗 chained]
*job 24336457 • idun-08-01 • H100 • epoch 344/1000 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24337305_24337305`  [⚠️ truncated]
*job 24337305 • idun-07-08 • A100 • chain 0/20 • 1KB log*


#### `24338909_24338909`  [🔗 chained]
*job 24338909 • idun-09-02 • H200 • epoch 525/1000 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24341731_24341731`  [⚠️ truncated]
*job 24341731 • idun-09-02 • H200 • chain 0/20 • 1KB log*


#### `24341732_24341732`  [✅ completed]
*job 24341732 • idun-09-02 • H200 • 9.38h training • epoch 100/100 • chain 0/20 • 20KB log*


#### `24341733_24341733`  [⚠️ truncated]
*job 24341733 • idun-08-01 • H100 • chain 0/20 • 1KB log*


#### `24341756_24341756`  [🔗 chained]
*job 24341756 • idun-08-01 • H100 • epoch 166/500 • chain 0/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341757_24341757`  [🔗 chained]
*job 24341757 • idun-09-02 • H200 • epoch 178/500 • chain 0/20 • 160KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341758_24341758`  [🔗 chained]
*job 24341758 • idun-06-06 • A100 • epoch 113/1000 • chain 0/20 • 100KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24341759_24341759`  [🔗 chained]
*job 24341759 • idun-08-01 • H100 • epoch 127/1000 • chain 0/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24341775_24341775`  [🔗 chained]
*job 24341775 • idun-07-08 • A100 • epoch 108/500 • chain 0/20 • 105KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341776_24341776`  [🔗 chained]
*job 24341776 • idun-07-10 • A100 • epoch 107/500 • chain 0/20 • 96KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341777_24341777`  [🔗 chained]
*job 24341777 • idun-09-02 • H200 • epoch 180/500 • chain 0/20 • 139KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341778_24341778`  [🔗 chained]
*job 24341778 • idun-06-04 • A100 • epoch 117/500 • chain 0/20 • 102KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341791_24341791`  [🔗 chained]
*job 24341791 • idun-06-04 • A100 • epoch 645/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24342024_24342024`  [⚠️ truncated]
*job 24342024 • idun-07-08 • A100 • epoch 200/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp45_stack_lpips_scoreaug_20260425-031502/checkpoint_latest.pt`

#### `24342052_24342052`  [⚠️ truncated]
*job 24342052 • idun-09-02 • H200 • epoch 230/500 • chain 1/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_extended_pixel_bravo_lpips_lowt_20260425-034324/checkpoint_latest.pt`

#### `24342065_24342065`  [⚠️ truncated]
*job 24342065 • idun-06-04 • A100 • epoch 30/500 • chain 0/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342066_24342066`  [❌ crashed]
*job 24342066 • idun-06-06 • A100 • chain 0/20 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m20.443s
user	0m6.265s
sys	0m4.810s
```

#### `24342067_24342067`  [⚠️ truncated]
*job 24342067 • idun-06-06 • A100 • epoch 27/500 • chain 0/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342068_24342068`  [⚠️ truncated]
*job 24342068 • idun-01-04 • H100 • epoch 24/500 • chain 0/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342079_24342079`  [⚠️ truncated]
*job 24342079 • idun-07-08 • A100 • epoch 124/1000 • chain 1/20 • 13KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp46_mamba_l_lpips_lowt_20260425-042726/checkpoint_latest.pt`

#### `24342089_24342089`  [⚠️ truncated]
*job 24342089 • idun-07-10 • A100 • epoch 138/1000 • chain 1/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp46b_mamba_l_lpips_scoreaug_20260425-043934/checkpoint_latest.pt`

#### `24342186_24342186`  [⚠️ truncated]
*job 24342186 • idun-09-02 • H200 • epoch 127/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp47a_lpips_strong_20260425-055252/checkpoint_latest.pt`

#### `24342187_24342187`  [⚠️ truncated]
*job 24342187 • idun-06-04 • A100 • epoch 115/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp47b_huber_lpips_lowt_20260425-055554/checkpoint_latest.pt`

---
## train/compression/

*101 jobs across 14 families.*

### exp1 (1 jobs)

*Status: ⚠️ truncated=1*

#### `exp1_progressive_23870900`  [⚠️ truncated]
*job 23870900 • idun-08-01 • H100 • epoch 200/500 • 70KB log*


### exp2 (1 jobs)

*Status: ⚠️ truncated=1*

#### `exp2_finetune_23872206`  [⚠️ truncated]
*job 23872206 • idun-01-03 • H100 • epoch 100/100 • 27KB log*


### exp3 (1 jobs)

*Status: 💥 oom_killed=1*

Early 2D VAE.

#### `exp3_multimodal_256_23873171`  [💥 oom_killed]
*job 23873171 • idun-06-07 • A100 • epoch 125/125 • 29KB log*

**Traceback excerpt:**
```
/var/slurm_spool/job23873171/slurm_script: line 43: 366145 Killed                  python -m medgen.scripts.train_vae paths=cluster mode=multi_modality model.image_size=256 training.epochs=125 training.batch_size=16 training.augment=false training.name=exp3_

real	2041m46.094s
user	2063m57.315s
sys	68m32.569s
[2025-12-27T22:07:52.154] error: Detected 1 oom_kill event in StepId=23873171.batch. Some of the step tasks have been OOM Killed.
```

### exp4 (2 jobs)

*Status: 💥 oom_killed=1 ⚠️ truncated=1*

Early 2D VAE.

#### `exp4_multimodal_64lat_23873172`  [💥 oom_killed]
*job 23873172 • idun-06-07 • A100 • epoch 125/125 • 30KB log*

**Traceback excerpt:**
```
/var/slurm_spool/job23873172/slurm_script: line 46: 366144 Killed                  python -m medgen.scripts.train_vae paths=cluster mode=multi_modality model.image_size=256 'vae.channels=[64,128,256]' 'vae.attention_levels=[false,false,true]' training.epochs=125 training.batch_size=16 training.augment=false training.name=exp4_

real	2333m39.643s
user	2360m57.997s
sys	62m30.072s
[2025-12-28T02:59:45.741] error: Detected 1 oom_kill event in StepId=23873172.batch. Some of the step tasks have been OOM Killed.
```

#### `exp4_multimodal_64lat_23886882`  [⚠️ truncated]
*job 23886882 • idun-06-02 • A100 • 1KB log*


### exp5 (1 jobs)

*Status: 💥 oom_killed=1*

Early 2D VAE.

#### `exp5_multimodal_bf16_23873346`  [💥 oom_killed]
*job 23873346 • idun-06-02 • A100 • epoch 125/125 • 29KB log*

**Traceback excerpt:**
```
/var/slurm_spool/job23873346/slurm_script: line 46: 4005158 Killed                  python -m medgen.scripts.train_vae paths=cluster mode=multi_modality model.image_size=256 training.epochs=125 training.batch_size=16 training.augment=false 'training.precision.dtype=bf16' 'training.precision.pure_weights=true' training.name=exp5_

real	2001m29.302s
user	1994m56.178s
sys	90m46.768s
[2025-12-28T08:27:15.651] error: Detected 1 oom_kill event in StepId=23873346.batch. Some of the step tasks have been OOM Killed.
```

### exp6 (2 jobs)

*Status: ❌ crashed=2*

Refined 2D VAE.

#### `exp6_vqvae_23874585`  [❌ crashed]
*job 23874585 • idun-06-01 • A100 • epoch 125/125 • 32KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_vqvae.py", line 213, in main
    trainer.evaluate_test_set(test_loader, checkpoint_name="best")
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	583m28.735s
user	765m57.472s
sys	153m50.961s
```

#### `exp6_1_vqvae_4x_23874586`  [❌ crashed]
*job 23874586 • idun-06-01 • A100 • epoch 125/125 • 32KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_vqvae.py", line 213, in main
    trainer.evaluate_test_set(test_loader, checkpoint_name="best")
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	559m7.780s
user	735m18.848s
sys	149m33.222s
```

### exp7 (3 jobs)

*Status: ⚠️ truncated=2 ✅ completed=1*

Early 3D VAE multi-modality runs.

#### `exp7_vae3d_multimodal_23877660`  [⚠️ truncated]
*job 23877660 • idun-06-04 • A100 • epoch 125/125 • 30KB log*


#### `exp7_vae3d_multimodal_23879377`  [⚠️ truncated]
*job 23879377 • idun-06-05 • A100 • epoch 125/125 • 2544KB log*


#### `exp7_vae3d_multimodal_23883787`  [✅ completed]
*job 23883787 • idun-06-04 • A100 • 44.39h training • epoch 125/125 • 38KB log*


### exp8 (15 jobs)

*Status: ✅ completed=13 ⚠️ truncated=1 ❌ crashed=1*

3D VAE 256×160 sweep — 12 variants of KL/adv weight, depth.

#### `exp8_vqvae3d_4x_23877682`  [⚠️ truncated]
*job 23877682 • idun-06-05 • A100 • epoch 125/125 • 32KB log*


#### `exp8_vqvae3d_4x_23878983`  [❌ crashed]
*job 23878983 • idun-06-04 • A100 • epoch 125/125 • 2523KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_vqvae_3d.py", line 221, in main
    trainer.evaluate_test(test_loader, checkpoint_name="best")
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	4562m34.960s
user	4659m6.643s
sys	233m33.743s
```

#### `exp8_1_vqvae3d_lat4_23884814`  [✅ completed]
*job 23884814 • idun-06-04 • A100 • 34.97h training • epoch 125/125 • 42KB log*


#### `exp8_2_vqvae3d_lat8_23884815`  [✅ completed]
*job 23884815 • idun-06-02 • A100 • 33.51h training • epoch 125/125 • 41KB log*


#### `exp8_3_vqvae3d_lat16_23884816`  [✅ completed]
*job 23884816 • idun-06-02 • A100 • 32.94h training • epoch 125/125 • 41KB log*


#### `exp8_5_8x_lat4_23885962`  [✅ completed]
*job 23885962 • idun-07-09 • A100 • 40.38h training • epoch 125/125 • 41KB log*


#### `exp8_7_8x_lat16_23885965`  [✅ completed]
*job 23885965 • idun-07-10 • A100 • 40.59h training • epoch 125/125 • 41KB log*


#### `exp8_6_8x_lat8_23885967`  [✅ completed]
*job 23885967 • idun-07-08 • A100 • 41.28h training • epoch 125/125 • 41KB log*


#### `exp8_9_8x_lat64_23885968`  [✅ completed]
*job 23885968 • idun-06-06 • A100 • 35.36h training • epoch 125/125 • 41KB log*


#### `exp8_8_8x_lat32_23885970`  [✅ completed]
*job 23885970 • idun-06-05 • A100 • 35.19h training • epoch 125/125 • 41KB log*


#### `exp8_10_vqvae3d_wide_23889692`  [✅ completed]
*job 23889692 • idun-06-05 • A100 • 65.32h training • epoch 125/125 • 60KB log*


#### `exp8_11_vqvae3d_cb1024_23889693`  [✅ completed]
*job 23889693 • idun-07-08 • A100 • 42.24h training • epoch 125/125 • 61KB log*


#### `exp8_12_vqvae3d_deeper_23889696`  [✅ completed]
*job 23889696 • idun-07-10 • A100 • 46.59h training • epoch 125/125 • 60KB log*


#### `exp8_13_vqvae3d_highperc_23889697`  [✅ completed]
*job 23889697 • idun-01-03 • H100 • 27.65h training • epoch 125/125 • 60KB log*


#### `exp8_14_vqvae3d_combined_23889698`  [✅ completed]
*job 23889698 • idun-01-03 • H100 • 32.8h training • epoch 125/125 • 61KB log*


### exp9 (5 jobs)

*Status: ❌ crashed=2 ⚠️ truncated=2 ✅ completed=1*

2D VAE/DC-AE compression-ratio sweep.

#### `exp9_1_dcae_f32_23879375`  [❌ crashed]
*job 23879375 • idun-06-07 • A100 • epoch 125/125 • 24KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_dcae.py", line 137, in main
    trainer.evaluate_test(test_loader, checkpoint_name="best")
...
[W103 10:29:31.617334552 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	3880m4.957s
user	3547m4.940s
sys	900m10.503s
```

#### `exp9_2_dcae_f64_23879376`  [❌ crashed]
*job 23879376 • idun-06-04 • A100 • epoch 125/125 • 24KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_dcae.py", line 137, in main
    trainer.evaluate_test(test_loader, checkpoint_name="best")
...
[W103 10:39:44.775423041 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	3890m7.386s
user	3575m20.108s
sys	900m9.235s
```

#### `exp9_3_dcae_f128_23884433`  [⚠️ truncated]
*job 23884433 • idun-06-04 • A100 • epoch 120/125 • 77KB log*


#### `exp9_1_dcae_f32_phase3_23888825`  [✅ completed]
*job 23888825 • idun-06-02 • A100 • 14.66h training • epoch 175/175 • 37KB log*


#### `exp9_4_dcae_f128_lpips_23888949`  [⚠️ truncated]
*job 23888949 • idun-07-09 • A100 • epoch 109/125 • 71KB log*


### exp10 (1 jobs)

*Status: ⚠️ truncated=1*

#### `exp10_1_dcae3d_default_23888483`  [⚠️ truncated]
*job 23888483 • idun-06-04 • A100 • epoch 90/125 • 30KB log*


### exp11 (2 jobs)

*Status: ✅ completed=2*

Seg-channel compression (BCE+boundary loss).

#### `exp11_1_dcae_seg_compression_23888948`  [✅ completed]
*job 23888948 • idun-07-09 • A100 • 4.55h training • epoch 125/125 • 41KB log*


#### `exp11_2_vqvae3d_seg_23889173`  [✅ completed]
*job 23889173 • idun-06-07 • A100 • 3.06h training • epoch 125/125 • 59KB log*


### exp12 (3 jobs)

*Status: ⚠️ truncated=3*

#### `exp12_1_eval_maisi_vae_23919197`  [⚠️ truncated]
*job 23919197 • idun-01-03 • H100 • 5KB log*


#### `exp12_1_eval_maisi_vae_23919199`  [⚠️ truncated]
*job 23919199 • idun-01-03 • H100 • 5KB log*


#### `exp12_1_eval_maisi_vae_23919200`  [⚠️ truncated]
*job 23919200 • idun-01-03 • H100 • 5KB log*


### exp13 (3 jobs)

*Status: ✅ completed=3*

2D seg compression ratios (f32/f64/f128).

#### `exp13_1_dcae_seg_f32_23996772`  [✅ completed]
*job 23996772 • idun-07-08 • A100 • 17.83h training • epoch 500/500 • 83KB log*


#### `exp13_2_dcae_seg_f64_23996773`  [✅ completed]
*job 23996773 • idun-07-09 • A100 • 17.82h training • epoch 500/500 • 83KB log*


#### `exp13_3_dcae_seg_f128_23996774`  [✅ completed]
*job 23996774 • idun-07-09 • A100 • 18.03h training • epoch 500/500 • 82KB log*


### OTHER (61 jobs)

*Status: 🔗 chained=53 ✅ completed=7 💥 oom_killed=1*

#### `24042621_24042621`  [🔗 chained]
*job 24042621 • idun-06-04 • A100 • epoch 16/125 • chain 0/20 • 17KB log*


#### `24042622_24042622`  [🔗 chained]
*job 24042622 • idun-06-04 • A100 • epoch 15/125 • chain 0/20 • 17KB log*


#### `24042650_24042650`  [🔗 chained]
*job 24042650 • idun-06-04 • A100 • epoch 28/125 • chain 0/20 • 25KB log*


#### `24042713_24042713`  [🔗 chained]
*job 24042713 • idun-06-07 • A100 • epoch 31/125 • chain 1/20 • 20KB log*


#### `24042714_24042714`  [🔗 chained]
*job 24042714 • idun-07-09 • A100 • epoch 28/125 • chain 1/20 • 17KB log*


#### `24042802_24042802`  [🔗 chained]
*job 24042802 • idun-06-04 • A100 • epoch 55/125 • chain 1/20 • 24KB log*


#### `24043055_24043055`  [🔗 chained]
*job 24043055 • idun-06-07 • A100 • epoch 42/125 • chain 2/20 • 20KB log*


#### `24043056_24043056`  [🔗 chained]
*job 24043056 • idun-07-09 • A100 • epoch 44/125 • chain 2/20 • 26KB log*


#### `24043081_24043081`  [🔗 chained]
*job 24043081 • idun-06-04 • A100 • epoch 82/125 • chain 2/20 • 22KB log*


#### `24043346_24043346`  [🔗 chained]
*job 24043346 • idun-06-07 • A100 • epoch 59/125 • chain 3/20 • 31KB log*


#### `24043347_24043347`  [🔗 chained]
*job 24043347 • idun-06-07 • A100 • epoch 56/125 • chain 3/20 • 22KB log*


#### `24043608_24043608`  [🔗 chained]
*job 24043608 • idun-06-04 • A100 • epoch 109/125 • chain 3/20 • 21KB log*


#### `24045076_24045076`  [🔗 chained]
*job 24045076 • idun-06-07 • A100 • epoch 16/125 • chain 0/20 • 18KB log*


#### `24045485_24045485`  [🔗 chained]
*job 24045485 • idun-07-08 • A100 • epoch 69/125 • chain 4/20 • 26KB log*


#### `24045486_24045486`  [🔗 chained]
*job 24045486 • idun-06-04 • A100 • epoch 74/125 • chain 4/20 • 39KB log*


#### `24045605_24045605`  [💥 oom_killed]
*job 24045605 • idun-07-10 • A100 • 7.59h training • epoch 125/125 • chain 4/20 • 18KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_compression.py", line 650, in main
    _train_3d(cfg, trainer_config)
...
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/evaluation/evaluation_3d.py", line 173, in _compute_batch_metrics
    reconstructed = self.forward_fn(model, images)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/vae_t
```

#### `24047301_24047301`  [🔗 chained]
*job 24047301 • idun-01-04 • H100 • epoch 32/125 • chain 1/20 • 22KB log*


#### `24047745_24047745`  [🔗 chained]
*job 24047745 • idun-07-10 • A100 • epoch 82/125 • chain 5/20 • 27KB log*


#### `24051155_24051155`  [🔗 chained]
*job 24051155 • idun-06-04 • A100 • epoch 89/125 • chain 5/20 • 47KB log*


#### `24052177_24052177`  [🔗 chained]
*job 24052177 • idun-01-04 • H100 • epoch 55/125 • chain 2/20 • 39KB log*


#### `24052205_24052205`  [🔗 chained]
*job 24052205 • idun-07-10 • A100 • epoch 95/125 • chain 6/20 • 30KB log*


#### `24052521_24052521`  [🔗 chained]
*job 24052521 • idun-06-04 • A100 • epoch 104/125 • chain 6/20 • 43KB log*


#### `24055961_24055961`  [🔗 chained]
*job 24055961 • idun-01-04 • H100 • epoch 78/125 • chain 3/20 • 47KB log*


#### `24056005_24056005`  [🔗 chained]
*job 24056005 • idun-01-04 • H100 • epoch 118/125 • chain 7/20 • 29KB log*


#### `24056359_24056359`  [🔗 chained]
*job 24056359 • idun-06-01 • A100 • epoch 119/125 • chain 7/20 • 28KB log*


#### `24056849_24056849`  [🔗 chained]
*job 24056849 • idun-07-08 • A100 • epoch 92/125 • chain 4/20 • 36KB log*


#### `24056940_24056940`  [✅ completed]
*job 24056940 • idun-06-06 • A100 • 6.11h training • epoch 125/125 • chain 8/20 • 17KB log*


#### `24057127_24057127`  [🔗 chained]
*job 24057127 • idun-06-01 • A100 • epoch 16/125 • chain 0/20 • 18KB log*


#### `24057193_24057193`  [✅ completed]
*job 24057193 • idun-06-02 • A100 • 5.21h training • epoch 125/125 • chain 8/20 • 20KB log*


#### `24060420_24060420`  [🔗 chained]
*job 24060420 • idun-06-04 • A100 • epoch 107/125 • chain 5/20 • 34KB log*


#### `24060703_24060703`  [🔗 chained]
*job 24060703 • idun-06-02 • A100 • epoch 31/125 • chain 1/20 • 18KB log*


#### `24061242_24061242`  [🔗 chained]
*job 24061242 • idun-06-04 • A100 • epoch 122/125 • chain 6/20 • 30KB log*


#### `24061658_24061658`  [🔗 chained]
*job 24061658 • idun-07-10 • A100 • epoch 44/125 • chain 2/20 • 24KB log*


#### `24061895_24061895`  [✅ completed]
*job 24061895 • idun-07-08 • A100 • 3.23h training • epoch 125/125 • chain 7/20 • 19KB log*


#### `24061916_24061916`  [🔗 chained]
*job 24061916 • idun-06-01 • A100 • epoch 18/125 • chain 0/20 • 50KB log*


#### `24061917_24061917`  [🔗 chained]
*job 24061917 • idun-01-03 • H100 • epoch 32/125 • chain 0/20 • 103KB log*


#### `24061918_24061918`  [🔗 chained]
*job 24061918 • idun-08-01 • H100 • epoch 31/125 • chain 0/20 • 59KB log*


#### `24061980_24061980`  [🔗 chained]
*job 24061980 • idun-06-03 • A100 • epoch 59/125 • chain 3/20 • 30KB log*


#### `24062116_24062116`  [🔗 chained]
*job 24062116 • idun-01-03 • H100 • epoch 49/125 • chain 1/20 • 71KB log*


#### `24062477_24062477`  [🔗 chained]
*job 24062477 • idun-06-02 • A100 • epoch 47/125 • chain 1/20 • 28KB log*


#### `24062818_24062818`  [🔗 chained]
*job 24062818 • idun-06-02 • A100 • epoch 47/125 • chain 1/20 • 34KB log*


#### `24062936_24062936`  [🔗 chained]
*job 24062936 • idun-06-02 • A100 • epoch 74/125 • chain 4/20 • 33KB log*


#### `24062975_24062975`  [🔗 chained]
*job 24062975 • idun-06-03 • A100 • epoch 64/125 • chain 2/20 • 24KB log*


#### `24062976_24062976`  [🔗 chained]
*job 24062976 • idun-06-05 • A100 • epoch 63/125 • chain 2/20 • 26KB log*


#### `24063145_24063145`  [🔗 chained]
*job 24063145 • idun-06-05 • A100 • epoch 63/125 • chain 2/20 • 28KB log*


#### `24063323_24063323`  [🔗 chained]
*job 24063323 • idun-06-05 • A100 • epoch 89/125 • chain 5/20 • 36KB log*


#### `24063458_24063458`  [🔗 chained]
*job 24063458 • idun-06-03 • A100 • epoch 79/125 • chain 3/20 • 23KB log*


#### `24063472_24063472`  [🔗 chained]
*job 24063472 • idun-06-05 • A100 • epoch 79/125 • chain 3/20 • 26KB log*


#### `24063643_24063643`  [🔗 chained]
*job 24063643 • idun-06-05 • A100 • epoch 79/125 • chain 3/20 • 28KB log*


#### `24063684_24063684`  [🔗 chained]
*job 24063684 • idun-06-05 • A100 • epoch 104/125 • chain 6/20 • 37KB log*


#### `24063706_24063706`  [🔗 chained]
*job 24063706 • idun-01-04 • H100 • epoch 109/125 • chain 4/20 • 35KB log*


#### `24063720_24063720`  [🔗 chained]
*job 24063720 • idun-06-05 • A100 • epoch 95/125 • chain 4/20 • 26KB log*


#### `24064055_24064055`  [🔗 chained]
*job 24064055 • idun-06-02 • A100 • epoch 87/125 • chain 4/20 • 20KB log*


#### `24065736_24065736`  [🔗 chained]
*job 24065736 • idun-06-02 • A100 • epoch 119/125 • chain 7/20 • 29KB log*


#### `24066193_24066193`  [✅ completed]
*job 24066193 • idun-06-01 • A100 • 11.27h training • epoch 125/125 • chain 5/20 • 26KB log*


#### `24066243_24066243`  [🔗 chained]
*job 24066243 • idun-06-05 • A100 • epoch 110/125 • chain 5/20 • 24KB log*


#### `24067813_24067813`  [🔗 chained]
*job 24067813 • idun-06-03 • A100 • epoch 101/125 • chain 5/20 • 25KB log*


#### `24070763_24070763`  [✅ completed]
*job 24070763 • idun-06-02 • A100 • 5.2h training • epoch 125/125 • chain 8/20 • 21KB log*


#### `24070842_24070842`  [✅ completed]
*job 24070842 • idun-06-02 • A100 • 10.74h training • epoch 125/125 • chain 6/20 • 26KB log*


#### `24075914_24075914`  [🔗 chained]
*job 24075914 • idun-07-08 • A100 • epoch 115/125 • chain 6/20 • 26KB log*


#### `24079124_24079124`  [✅ completed]
*job 24079124 • idun-07-09 • A100 • 8.09h training • epoch 125/125 • chain 7/20 • 24KB log*


---
## train/diffusion/

*788 jobs across 28 families.*

### exp1 (16 jobs)

*Status: ❌ crashed=6 ✅ completed=6 ⚠️ truncated=2 💥 oom_killed=2*

Early 2D bravo baselines.

#### `exp1_rflow_23870661`  [✅ completed]
*job 23870661 • idun-09-16 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006722 • MS-SSIM 0.8566 • PSNR 28.79 dB • LPIPS 0.5664
  - **latest** ckpt (3339 samples): MSE 0.01152 • MS-SSIM 0.8248 • PSNR 27.60 dB • LPIPS 0.5114

#### `exp1_2_ddpm_23872208`  [❌ crashed]
*job 23872208 • idun-06-03 • A100 • epoch 500/500 • 106KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.005586 • MS-SSIM 0.7926 • PSNR 23.12 dB • LPIPS 1.1494
  - **latest** ckpt (3339 samples): MSE 0.007744 • MS-SSIM 0.8061 • PSNR 24.64 dB • LPIPS 0.9856
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 347, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
[2025-12-24 12:27:14,317][medgen.pipeline.trainer][INFO] -   MS-SSIM: 0.7926
[2025-12-24 12:27:14,317][medgen.pipeline.trainer][INFO] -   PSNR:    23.12 dB
[2025-12-24 12:27:14,317][medgen.pipeline.trainer][INFO] -   LPIPS:   1.1494
[2025-12-24 12:27:14,332][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/bravo/exp1_2_ddpm_128_20251223-233504/test_results_best.json
[2025-12-24 12:27:15,403][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900_master/runs/
```

#### `exp1_pixel_bravo_23969657`  [⚠️ truncated]
*job 23969657 • idun-06-04 • A100 • epoch 500/500 • 204KB log*


#### `exp1_1_pixel_bravo_23969658`  [⚠️ truncated]
*job 23969658 • idun-06-05 • A100 • epoch 500/500 • 207KB log*


#### `exp1_3_continuous_23969703`  [✅ completed]
*job 23969703 • idun-06-06 • A100 • epoch 500/500 • 90KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006545 • MS-SSIM 0.8599 • PSNR 29.11 dB • LPIPS 1.0135 • FID 53.74 • KID 0.0686 ± 0.0106 • CMMD 0.2642
  - **latest** ckpt (3339 samples): MSE 0.01175 • MS-SSIM 0.8307 • PSNR 28.31 dB • LPIPS 0.8584 • FID 35.98 • KID 0.0420 ± 0.0085 • CMMD 0.2268

#### `exp1_1_pixel_bravo_23982866`  [💥 oom_killed]
*job 23982866 • idun-06-02 • A100 • epoch 500/500 • 2211KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1906m18.389s
user	2198m4.836s
sys	228m21.742s
```

#### `exp1_pixel_bravo_23982867`  [❌ crashed]
*job 23982867 • idun-06-01 • A100 • epoch 500/500 • 4068KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	881m54.783s
user	1079m50.263s
sys	204m0.064s
```

#### `exp1_debugging_cfg_23984445`  [❌ crashed]
*job 23984445 • idun-06-01 • A100 • epoch 500/500 • 268KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	617m21.303s
user	926m29.906s
s
```

#### `exp1_debugging_cfg_23985115`  [❌ crashed]
*job 23985115 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	605m56.941s
user	935m6.195s
sy
```

#### `exp1_pixel_bravo_23985116`  [❌ crashed]
*job 23985116 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	609m29.011s
user	931m57.274s
s
```

#### `exp1_1_pixel_bravo_23989010`  [💥 oom_killed]
*job 23989010 • idun-06-07 • A100 • epoch 500/500 • 505KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1884m48.852s
user	2223m3.903s
```

#### `exp1_pixel_bravo_23989011`  [❌ crashed]
*job 23989011 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	601m45.956s
user	939m36.336s
s
```

#### `exp1e_pixel_bravo_snr_gamma_24072105`  [✅ completed]
*job 24072105 • idun-06-05 • A100 • 16.95h training • epoch 500/500 • 87KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003829 • MS-SSIM 0.9470 • PSNR 32.96 dB • LPIPS 0.6404 • FID 92.73 • KID 0.1051 ± 0.0058 • CMMD 0.2918
  - **latest** ckpt (26 samples): MSE 0.009645 • MS-SSIM 0.9034 • PSNR 29.97 dB • LPIPS 0.4710 • FID 68.73 • KID 0.0715 ± 0.0052 • CMMD 0.1557

#### `exp1f_pixel_bravo_edm_precond_24072106`  [✅ completed]
*job 24072106 • idun-06-01 • A100 • 16.8h training • epoch 500/500 • 189KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01936 • MS-SSIM 0.9049 • PSNR 30.64 dB • LPIPS 1.2412 • FID 219.37 • KID 0.2865 ± 0.0107 • CMMD 0.6236
  - **latest** ckpt (26 samples): MSE 0.0177 • MS-SSIM 0.9009 • PSNR 30.73 dB • LPIPS 1.3032 • FID 220.17 • KID 0.2913 ± 0.0124 • CMMD 0.6248

#### `exp1e_1_pixel_bravo_snr_gamma_24072109`  [✅ completed]
*job 24072109 • idun-06-01 • A100 • 49.2h training • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002907 • MS-SSIM 0.9563 • PSNR 33.09 dB • LPIPS 0.5720 • FID 114.24 • KID 0.1324 ± 0.0101 • CMMD 0.2861
  - **latest** ckpt (26 samples): MSE 0.004712 • MS-SSIM 0.9606 • PSNR 33.57 dB • LPIPS 0.4926 • FID 108.41 • KID 0.1254 ± 0.0082 • CMMD 0.2294

#### `exp1f_1_pixel_bravo_edm_precond_24072110`  [✅ completed]
*job 24072110 • idun-06-02 • A100 • 49.43h training • epoch 500/500 • 180KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003571 • MS-SSIM 0.9392 • PSNR 31.55 dB • LPIPS 0.5694 • FID 211.48 • KID 0.3090 ± 0.0184 • CMMD 0.4421
  - **latest** ckpt (26 samples): MSE 0.003059 • MS-SSIM 0.9476 • PSNR 32.02 dB • LPIPS 0.5285 • FID 196.32 • KID 0.2765 ± 0.0181 • CMMD 0.4081

### exp2 (19 jobs)

*Status: ⚠️ truncated=6 ✅ completed=5 ❌ crashed=4 💥 oom_killed=4*

2D bravo with RFlow.

#### `exp2_rflow_100steps_23870662`  [✅ completed]
*job 23870662 • idun-09-16 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006831 • MS-SSIM 0.8652 • PSNR 29.00 dB • LPIPS 0.5237
  - **latest** ckpt (3339 samples): MSE 0.01191 • MS-SSIM 0.8280 • PSNR 27.65 dB • LPIPS 0.5058

#### `exp2_rflow_100steps_23873347`  [✅ completed]
*job 23873347 • idun-09-16 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006646 • MS-SSIM 0.8651 • PSNR 29.01 dB • LPIPS 0.5266
  - **latest** ckpt (3339 samples): MSE 0.01223 • MS-SSIM 0.8299 • PSNR 27.68 dB • LPIPS 0.5051

#### `exp2_rflow_100steps_23885460`  [💥 oom_killed]
*job 23885460 • idun-09-16 • A100 • epoch 500/500 • 151KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006715 • MS-SSIM 0.8624 • PSNR 29.24 dB • LPIPS 0.5791
  - **latest** ckpt (3339 samples): MSE 0.01214 • MS-SSIM 0.8268 • PSNR 28.06 dB • LPIPS 0.5092
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/clus
```

#### `exp2_bf16mse_23885956`  [❌ crashed]
*job 23885956 • idun-07-08 • A100 • epoch 107/500 • 37KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 368, in generate_samples
    samples, intermediates = self._generate_with_intermediate_steps(
...
            ^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/clus
```

#### `exp2_rflow_100steps_23888424`  [✅ completed]
*job 23888424 • idun-07-04 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006626 • MS-SSIM 0.8668 • PSNR 29.27 dB • LPIPS 0.5089
  - **latest** ckpt (3339 samples): MSE 0.01235 • MS-SSIM 0.8335 • PSNR 28.41 dB • LPIPS 0.5019

#### `exp2_rflow_100steps_23914222`  [✅ completed]
*job 23914222 • idun-07-06 • A100 • epoch 500/500 • 100KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006661 • MS-SSIM 0.8665 • PSNR 29.43 dB • LPIPS 0.5270 • FID 44.81 • KID 0.0561 ± 0.0097 • CMMD 0.2026
  - **latest** ckpt (3339 samples): MSE 0.01208 • MS-SSIM 0.8305 • PSNR 28.22 dB • LPIPS 0.5030 • FID 36.42 • KID 0.0442 ± 0.0118 • CMMD 0.2304

#### `exp2_rflow_100steps_23962279`  [✅ completed]
*job 23962279 • idun-09-18 • A100 • epoch 500/500 • 99KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006275 • MS-SSIM 0.8625 • PSNR 29.31 dB • LPIPS 1.1101 • FID 55.05 • KID 0.0698 ± 0.0097 • CMMD 0.2408
  - **latest** ckpt (3339 samples): MSE 0.01186 • MS-SSIM 0.8304 • PSNR 28.30 dB • LPIPS 0.8261 • FID 36.32 • KID 0.0431 ± 0.0095 • CMMD 0.2234

#### `exp2_pixel_seg_sizebin_23972121`  [⚠️ truncated]
*job 23972121 • idun-06-05 • A100 • epoch 500/500 • 92KB log*


#### `exp2_1_pixel_seg_sizebin_23973544`  [💥 oom_killed]
*job 23973544 • idun-07-10 • A100 • epoch 500/500 • 441KB log*

**Traceback excerpt:**
```

real	2381m38.860s
user	2500m26.158s
sys	810m42.434s
```

#### `exp2_pixel_seg_sizebin_23982949`  [⚠️ truncated]
*job 23982949 • idun-07-10 • A100 • epoch 500/500 • 162KB log*


#### `exp2_1_pixel_seg_sizebin_23982950`  [💥 oom_killed]
*job 23982950 • idun-06-03 • A100 • epoch 500/500 • 511KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)

real	2411m48.584s
user	2240m33.632s
sys	957m5.454s
```

#### `exp2b_1_pixel_seg_input_cond_23982974`  [❌ crashed]
*job 23982974 • idun-07-08 • A100 • 1KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 95, in main
    validate_config(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m26.032s
user	0m7.485s
sys	0m3.812s
```

#### `exp2b_pixel_seg_input_cond_23982975`  [❌ crashed]
*job 23982975 • idun-07-08 • A100 • 1KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 95, in main
    validate_config(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m26.031s
user	0m7.500s
sys	0m3.780s
```

#### `exp2_pixel_seg_sizebin_23996720`  [⚠️ truncated]
*job 23996720 • idun-07-10 • A100 • epoch 443/500 • 84KB log*


#### `exp2_1_pixel_seg_sizebin_23996728`  [⚠️ truncated]
*job 23996728 • idun-06-03 • A100 • epoch 500/500 • 97KB log*


#### `exp2b_pixel_seg_input_cond_23996776`  [⚠️ truncated]
*job 23996776 • idun-06-07 • A100 • epoch 500/500 • 89KB log*


#### `exp2b_1_pixel_seg_input_cond_23997358`  [⚠️ truncated]
*job 23997358 • idun-07-10 • A100 • epoch 500/500 • 95KB log*


#### `exp2c_pixel_seg_improved_24031953`  [❌ crashed]
*job 24031953 • idun-07-10 • A100 • epoch 500/500 • 283KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1165, in train
    logger.error(f"Checkpoint save failed at epoch {epoch}: {
```

#### `exp2c_1_pixel_seg_improved_24039864`  [💥 oom_killed]
*job 24039864 • idun-06-01 • A100 • 2.66h training • epoch 25/500 • 27KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.31 GiB. GPU 0 has a total capacity of 79.25 GiB of which 24.65 GiB is free. Including non-PyTorch memory, this process has 54.59 GiB memory in use. Of the allocated memory 23.57 GiB is allocated by PyTorch, with 9.93 GiB allocated in private pools (e.g., CUDA Graphs), and 30.41 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation fo
```

### exp3 (2 jobs)

*Status: ✅ completed=2*

#### `exp3_rflow_100+a_23870663`  [✅ completed]
*job 23870663 • idun-06-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00675 • MS-SSIM 0.8546 • PSNR 28.83 dB • LPIPS 0.5090
  - **latest** ckpt (3339 samples): MSE 0.009425 • MS-SSIM 0.8234 • PSNR 27.82 dB • LPIPS 0.5301

#### `exp3_1_rflow_100+a_23871160`  [✅ completed]
*job 23871160 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007163 • MS-SSIM 0.8367 • PSNR 28.30 dB • LPIPS 0.5414
  - **latest** ckpt (3339 samples): MSE 0.007427 • MS-SSIM 0.8284 • PSNR 28.11 dB • LPIPS 0.5497

### exp4 (5 jobs)

*Status: ❌ crashed=3 ✅ completed=1 ⚠️ truncated=1*

#### `exp4_rflow_100+a+ema_23870665`  [✅ completed]
*job 23870665 • idun-06-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006668 • MS-SSIM 0.8505 • PSNR 28.77 dB • LPIPS 0.5668
  - **latest** ckpt (3339 samples): MSE 0.009299 • MS-SSIM 0.8254 • PSNR 27.86 dB • LPIPS 0.5289

#### `exp4_pixel_bravo_sda_23973532`  [❌ crashed]
*job 23973532 • idun-07-10 • A100 • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1210m48.286s
user	1508m58.309s
sys	396m37.989s
```

#### `exp4_1_pixel_bravo_sda_23982869`  [❌ crashed]
*job 23982869 • idun-07-10 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1347m16.527s
user	1744m28.560s
```

#### `exp4_pixel_bravo_sda_23987602`  [⚠️ truncated]
*job 23987602 • idun-07-09 • A100 • epoch 400/500 • 131KB log*


#### `exp4_pixel_bravo_sda_23989012`  [❌ crashed]
*job 23989012 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	1055m3.000s
user	1059m39.083s
```

### exp5 (7 jobs)

*Status: ❌ crashed=5 ✅ completed=2*

#### `exp5_rflow_100+a+ema+m_snr_23870666`  [✅ completed]
*job 23870666 • idun-06-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006866 • MS-SSIM 0.8520 • PSNR 28.79 dB • LPIPS 0.5413
  - **latest** ckpt (3339 samples): MSE 0.009993 • MS-SSIM 0.8214 • PSNR 27.72 dB • LPIPS 0.5324

#### `exp5_1_rflow_100_m_snr_23898317`  [✅ completed]
*job 23898317 • idun-09-18 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006789 • MS-SSIM 0.8677 • PSNR 29.39 dB • LPIPS 0.5155
  - **latest** ckpt (3339 samples): MSE 0.01191 • MS-SSIM 0.8263 • PSNR 28.20 dB • LPIPS 0.5058

#### `exp5_1_pixel_bravo_scoreaug_23973838`  [❌ crashed]
*job 23973838 • idun-07-09 • A100 • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	921m51.327s
user	1320m17.354s
sys	289m1.255s
```

#### `exp5_2_pixel_bravo_scoreaug_compose_23973839`  [❌ crashed]
*job 23973839 • idun-01-04 • H100 • epoch 500/500 • 88KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	598m1.317s
user	874m37.874s
sys	186m10.506s
```

#### `exp5_3_pixel_bravo_scoreaug_23982868`  [❌ crashed]
*job 23982868 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	929m40.615s
user	1072m45.500s
sys	352m31.078s
```

#### `exp5_1_pixel_bravo_scoreaug_23987603`  [❌ crashed]
*job 23987603 • idun-07-08 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	881m39.960s
user	1470m22.326s
```

#### `exp5_1_pixel_bravo_scoreaug_23989013`  [❌ crashed]
*job 23989013 • idun-07-08 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	888m54.571s
user	1451m38.134s
```

### exp6 (4 jobs)

*Status: ❌ crashed=3 💥 oom_killed=1*

#### `exp6_rflow_256_23870730`  [💥 oom_killed]
*job 23870730 • idun-06-03 • A100 • epoch 500/500 • 99KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 191, in main
    trainer.evaluate_test_set(test_loader, checkpoint_name="best")
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.
```

#### `exp6a_pixel_bravo_controlnet_stage1_23982865`  [❌ crashed]
*job 23982865 • idun-06-01 • A100 • epoch 500/500 • 3712KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	693m48.232s
user	941m24.993s
sys	152m50.417s
```

#### `exp6a_pixel_bravo_controlnet_stage1_23987604`  [❌ crashed]
*job 23987604 • idun-06-03 • A100 • epoch 500/500 • 272KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	597m41.009s
user	938m30.300s
s
```

#### `exp6a_pixel_bravo_controlnet_stage1_23989014`  [❌ crashed]
*job 23989014 • idun-07-08 • A100 • epoch 500/500 • 272KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	644m34.121s
user	1342m46.552s
```

### exp7 (15 jobs)

*Status: ❌ crashed=13 ⚠️ truncated=2*

#### `exp7_rflow_256_4lvl_23870677`  [❌ crashed]
*job 23870677 • idun-06-03 • A100 • epoch 500/500 • 98KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 168, in main
    # Setup model
...
srun: error: idun-06-03: task 0: Exited with exit code 1

real	941m46.577s
user	0m0.028s
sys	0m0.115s
```

#### `exp7_1_sit_s_256_patch16_23982976`  [⚠️ truncated]
*job 23982976 • idun-07-08 • A100 • epoch 262/500 • 93KB log*


#### `exp7_sit_s_128_patch8_23982977`  [⚠️ truncated]
*job 23982977 • idun-07-08 • A100 • epoch 410/500 • 136KB log*


#### `exp7_1_sit_b_256_patch16_23983255`  [❌ crashed]
*job 23983255 • idun-06-01 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W127 19:39:30.867309723 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	366m3.066s
user	697m33.444s
sys	264m1.699s
```

#### `exp7_1_sit_l_256_patch16_23983256`  [❌ crashed]
*job 23983256 • idun-06-03 • A100 • epoch 500/500 • 162KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 01:34:45.924049829 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	428m21.419s
user	712m22.139s
sys	270m0.198s
```

#### `exp7_sit_l_128_patch8_23983257`  [❌ crashed]
*job 23983257 • idun-06-01 • A100 • epoch 500/500 • 161KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 02:54:36.218818035 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	434m0.343s
user	763m29.757s
sys	210m3.392s
```

#### `exp7_sit_b_128_patch8_23983258`  [❌ crashed]
*job 23983258 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 00:00:34.501732707 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	253m52.604s
user	615m9.011s
sys	166m44.680s
```

#### `exp7_1_sit_xl_256_patch16_23984266`  [❌ crashed]
*job 23984266 • idun-07-08 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 08:53:45.936101779 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	691m0.114s
user	1274m12.928s
sys	361m44.731s
```

#### `exp7_sit_xl_128_patch8_23984267`  [❌ crashed]
*job 23984267 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W128 09:58:18.543157453 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	596m32.800s
user	866m4.917s
sys	213m36.524s
```

#### `exp7_sit_s_128_patch8_23987609`  [❌ crashed]
*job 23987609 • idun-06-07 • A100 • epoch 500/500 • 164KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 10:27:49.365873948 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	183m46.292s
user	540m28.016s
sys	153m24.189s
```

#### `exp7_sit_b_128_patch8_23987610`  [❌ crashed]
*job 23987610 • idun-06-01 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 11:22:36.724677251 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	213m50.673s
user	558m0.143s
sys	168m17.943s
```

#### `exp7_sit_l_128_patch8_23987611`  [❌ crashed]
*job 23987611 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 14:38:52.998854119 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	359m49.537s
user	682m45.118s
sys	193m18.033s
```

#### `exp7_sit_xl_128_patch8_23987612`  [❌ crashed]
*job 23987612 • idun-06-03 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W129 17:30:46.373138228 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	526m20.493s
user	816m12.553s
sys	204m15.002s
```

#### `exp7_sit_b_128_patch8_2000_23989812`  [❌ crashed]
*job 23989812 • idun-07-10 • A100 • epoch 2000/2000 • 603KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W131 08:37:36.884972023 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	1143m45.130s
user	3518m58.784s
sys	820m35.176s
```

#### `exp7_sit_b_128_patch8_23989813`  [❌ crashed]
*job 23989813 • idun-07-09 • A100 • epoch 500/500 • 160KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
[W130 19:42:46.954818782 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	290m8.472s
user	880m58.693s
sys	199m32.531s
```

### exp8 (15 jobs)

*Status: ✅ completed=14 ❌ crashed=1*

#### `exp8_1_small_23870740`  [✅ completed]
*job 23870740 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006828 • MS-SSIM 0.8516 • PSNR 28.79 dB • LPIPS 0.5660
  - **latest** ckpt (3339 samples): MSE 0.008598 • MS-SSIM 0.8263 • PSNR 27.98 dB • LPIPS 0.5342

#### `exp8_2_minimal_23870741`  [✅ completed]
*job 23870741 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006867 • MS-SSIM 0.8544 • PSNR 28.79 dB • LPIPS 0.5217
  - **latest** ckpt (3339 samples): MSE 0.007796 • MS-SSIM 0.8375 • PSNR 28.23 dB • LPIPS 0.5173

#### `exp8_3_tiny_23870742`  [✅ completed]
*job 23870742 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007012 • MS-SSIM 0.8485 • PSNR 28.72 dB • LPIPS 0.5888
  - **latest** ckpt (3339 samples): MSE 0.007118 • MS-SSIM 0.8476 • PSNR 28.58 dB • LPIPS 0.5181

#### `exp8_4_lean4lvl_23871467`  [✅ completed]
*job 23871467 • idun-06-07 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006958 • MS-SSIM 0.8328 • PSNR 28.25 dB • LPIPS 0.5658
  - **latest** ckpt (3339 samples): MSE 0.01017 • MS-SSIM 0.7924 • PSNR 27.04 dB • LPIPS 0.5829

#### `exp8_5_lean5lvl_23871468`  [✅ completed]
*job 23871468 • idun-07-05 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007207 • MS-SSIM 0.8280 • PSNR 28.11 dB • LPIPS 0.5768
  - **latest** ckpt (3339 samples): MSE 0.01155 • MS-SSIM 0.7776 • PSNR 26.71 dB • LPIPS 0.6112

#### `exp8_6_bottleneck_23871469`  [✅ completed]
*job 23871469 • idun-06-02 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007114 • MS-SSIM 0.8300 • PSNR 28.15 dB • LPIPS 0.5466
  - **latest** ckpt (3339 samples): MSE 0.01167 • MS-SSIM 0.7845 • PSNR 26.85 dB • LPIPS 0.5861

#### `exp8_7_deep_narrow_23871471`  [✅ completed]
*job 23871471 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00714 • MS-SSIM 0.8302 • PSNR 28.15 dB • LPIPS 0.6114
  - **latest** ckpt (3339 samples): MSE 0.009373 • MS-SSIM 0.8071 • PSNR 27.41 dB • LPIPS 0.5638

#### `exp8_10_ddpm_23872209`  [✅ completed]
*job 23872209 • idun-06-03 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007354 • MS-SSIM 0.8334 • PSNR 28.28 dB • LPIPS 0.5339
  - **latest** ckpt (3339 samples): MSE 0.01249 • MS-SSIM 0.7720 • PSNR 26.45 dB • LPIPS 0.6072

#### `exp8_8_adm_23872211`  [✅ completed]
*job 23872211 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007202 • MS-SSIM 0.8339 • PSNR 28.37 dB • LPIPS 0.5617
  - **latest** ckpt (3339 samples): MSE 0.009441 • MS-SSIM 0.7987 • PSNR 27.19 dB • LPIPS 0.5755

#### `exp8_9_edm_23872212`  [✅ completed]
*job 23872212 • idun-07-06 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007223 • MS-SSIM 0.8377 • PSNR 28.42 dB • LPIPS 0.5434
  - **latest** ckpt (3339 samples): MSE 0.008168 • MS-SSIM 0.8131 • PSNR 27.54 dB • LPIPS 0.5612

#### `exp8_11_iddpm_23872224`  [✅ completed]
*job 23872224 • idun-06-03 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007077 • MS-SSIM 0.8318 • PSNR 28.21 dB • LPIPS 0.6026
  - **latest** ckpt (3339 samples): MSE 0.009124 • MS-SSIM 0.7995 • PSNR 27.23 dB • LPIPS 0.5716

#### `exp8_1b_small_23873470`  [✅ completed]
*job 23873470 • idun-06-02 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006923 • MS-SSIM 0.8658 • PSNR 29.05 dB • LPIPS 0.5175
  - **latest** ckpt (3339 samples): MSE 0.01137 • MS-SSIM 0.8311 • PSNR 27.80 dB • LPIPS 0.5021

#### `exp8_4b_lean4lvl_23873471`  [✅ completed]
*job 23873471 • idun-07-09 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006915 • MS-SSIM 0.8645 • PSNR 29.04 dB • LPIPS 0.5391
  - **latest** ckpt (3339 samples): MSE 0.01425 • MS-SSIM 0.8167 • PSNR 27.25 dB • LPIPS 0.5268

#### `exp8_8b_adm_23873472`  [✅ completed]
*job 23873472 • idun-07-09 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006604 • MS-SSIM 0.8663 • PSNR 28.95 dB • LPIPS 0.5224
  - **latest** ckpt (3339 samples): MSE 0.01526 • MS-SSIM 0.8132 • PSNR 27.11 dB • LPIPS 0.5096

#### `exp8_pixel_bravo_ema_23991162`  [❌ crashed]
*job 23991162 • idun-07-10 • A100 • epoch 500/500 • 163KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 105, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	666m49.122s
user	1310m56.385s
sys	220m6.420s
```

### exp9 (23 jobs)

*Status: ✅ completed=13 ⚠️ truncated=8 ❌ crashed=1 💥 oom_killed=1*

#### `exp9_1_scoreaug_23870865`  [✅ completed]
*job 23870865 • idun-07-10 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006509 • MS-SSIM 0.8649 • PSNR 28.99 dB • LPIPS 0.5016
  - **latest** ckpt (3339 samples): MSE 0.008091 • MS-SSIM 0.8493 • PSNR 28.38 dB • LPIPS 0.4907

#### `exp9_2_scoreaug_full_23870866`  [✅ completed]
*job 23870866 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006604 • MS-SSIM 0.8688 • PSNR 29.07 dB • LPIPS 0.4944
  - **latest** ckpt (3339 samples): MSE 0.008531 • MS-SSIM 0.8467 • PSNR 28.27 dB • LPIPS 0.4906

#### `exp9_3_scoreaug_combined_23870912`  [✅ completed]
*job 23870912 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006784 • MS-SSIM 0.8539 • PSNR 28.80 dB • LPIPS 0.5633
  - **latest** ckpt (3339 samples): MSE 0.007014 • MS-SSIM 0.8529 • PSNR 28.69 dB • LPIPS 0.5115

#### `exp9_4_scoreaug_v2_23871242`  [✅ completed]
*job 23871242 • idun-07-06 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006644 • MS-SSIM 0.8674 • PSNR 29.09 dB • LPIPS 0.5265
  - **latest** ckpt (3339 samples): MSE 0.007801 • MS-SSIM 0.8484 • PSNR 28.33 dB • LPIPS 0.4943

#### `exp9_5_scoreaug_compose_23871257`  [✅ completed]
*job 23871257 • idun-06-07 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006478 • MS-SSIM 0.8713 • PSNR 29.13 dB • LPIPS 0.5011
  - **latest** ckpt (3339 samples): MSE 0.006871 • MS-SSIM 0.8608 • PSNR 28.78 dB • LPIPS 0.4894

#### `exp9_1_scoreaug_23873623`  [✅ completed]
*job 23873623 • idun-09-16 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00654 • MS-SSIM 0.8668 • PSNR 29.06 dB • LPIPS 0.5224
  - **latest** ckpt (3339 samples): MSE 0.007714 • MS-SSIM 0.8488 • PSNR 28.41 dB • LPIPS 0.4908

#### `exp9_2_scoreaug_full_23873624`  [✅ completed]
*job 23873624 • idun-09-16 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006536 • MS-SSIM 0.8685 • PSNR 29.10 dB • LPIPS 0.5222
  - **latest** ckpt (3339 samples): MSE 0.008436 • MS-SSIM 0.8472 • PSNR 28.27 dB • LPIPS 0.4886

#### `exp9_3_scoreaug_combined_23873625`  [✅ completed]
*job 23873625 • idun-07-09 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006966 • MS-SSIM 0.8344 • PSNR 28.30 dB • LPIPS 0.5521
  - **latest** ckpt (3339 samples): MSE 0.007103 • MS-SSIM 0.8393 • PSNR 28.36 dB • LPIPS 0.5446

#### `exp9_4_scoreaug_v2_23873627`  [✅ completed]
*job 23873627 • idun-07-10 • A100 • epoch 500/500 • 92KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006658 • MS-SSIM 0.8662 • PSNR 29.02 dB • LPIPS 0.5396
  - **latest** ckpt (3339 samples): MSE 0.008101 • MS-SSIM 0.8523 • PSNR 28.39 dB • LPIPS 0.4856

#### `exp9_5_scoreaug_compose_23873628`  [✅ completed]
*job 23873628 • idun-07-08 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006656 • MS-SSIM 0.8684 • PSNR 29.06 dB • LPIPS 0.5043
  - **latest** ckpt (3339 samples): MSE 0.00681 • MS-SSIM 0.8602 • PSNR 28.71 dB • LPIPS 0.4900

#### `exp9_6_scoreaug_compose08_23888431`  [✅ completed]
*job 23888431 • idun-09-16 • A100 • epoch 500/500 • 95KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006392 • MS-SSIM 0.8683 • PSNR 29.44 dB • LPIPS 0.5681
  - **latest** ckpt (3339 samples): MSE 0.006651 • MS-SSIM 0.8674 • PSNR 29.35 dB • LPIPS 0.5466

#### `exp9_7_scoreaug_v2_23888432`  [✅ completed]
*job 23888432 • idun-07-08 • A100 • epoch 500/500 • 95KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00628 • MS-SSIM 0.8660 • PSNR 29.40 dB • LPIPS 0.5125
  - **latest** ckpt (3339 samples): MSE 0.007717 • MS-SSIM 0.8526 • PSNR 28.90 dB • LPIPS 0.4775

#### `exp9_ldm_8x_bravo_23997506`  [⚠️ truncated]
*job 23997506 • idun-06-03 • A100 • epoch 500/500 • 92KB log*


#### `exp9_ldm_4x_bravo_23997507`  [⚠️ truncated]
*job 23997507 • idun-06-03 • A100 • epoch 500/500 • 93KB log*


#### `exp9_0_ldm_8x_bravo_small_23997680`  [⚠️ truncated]
*job 23997680 • idun-01-04 • H100 • epoch 100/100 • 30KB log*


#### `exp9_0_ldm_8x_bravo_small_23997808`  [❌ crashed]
*job 23997808 • idun-06-07 • A100 • epoch 100/100 • 31KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[W201 21:15:53.130025056 AllocatorCo
```

#### `exp9_1_ldm_4x_bravo_24036417`  [⚠️ truncated]
*job 24036417 • idun-07-08 • A100 • epoch 354/500 • 69KB log*


#### `exp9_ldm_8x_bravo_24039867`  [⚠️ truncated]
*job 24039867 • idun-06-01 • A100 • epoch 335/500 • 5284KB log*


#### `exp9_ldm_4x_bravo_24039868`  [⚠️ truncated]
*job 24039868 • idun-01-04 • H100 • epoch 302/500 • 4452KB log*


#### `exp9_0_ldm_8x_bravo_small_24061951`  [⚠️ truncated]
*job 24061951 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 38KB log*


#### `exp9_0_ldm_8x_bravo_small_24062097`  [⚠️ truncated]
*job 24062097 • idun-06-02 • A100 • epoch 170/500 • chain 0/20 • 50KB log*


#### `exp9_0_ldm_8x_bravo_small_24062563`  [✅ completed]
*job 24062563 • idun-06-01 • A100 • 7.58h training • epoch 500/500 • chain 0/20 • 119KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 3.232 • MS-SSIM 0.9997 • PSNR 55.21 dB • LPIPS 0.0014

#### `exp9_ldm_8x_bravo_24063647`  [💥 oom_killed]
*job 24063647 • idun-06-03 • A100 • 38.24h training • epoch 500/500 • 89KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/serialization.py", line 1864, in restore_location
    return default_restore_location(storage, str(map_location))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/serializat
```

### exp10 (19 jobs)

*Status: ❌ crashed=18 💥 oom_killed=1*

2D multi-modality.

#### `exp10_1_multi_23873080`  [❌ crashed]
*job 23873080 • idun-09-18 • A100 • epoch 90/125 • 23KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 343, in generate_samples
    raise ValueError(f"Unknown mode: {self.mode_name}")
ValueError: Unknown mode: multi
```

#### `exp10_2_multi_scoreaug_23873081`  [❌ crashed]
*job 23873081 • idun-06-07 • A100 • epoch 115/125 • 27KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 343, in generate_samples
    raise ValueError(f"Unknown mode: {self.mode_name}")
...
[2025-12-26 10:10:34,117][medgen.pipeline.utils][INFO] - [10:10:34] Epoch  97/125 ( 77.6%) completed | Total: 0.006058 | MSE: 0.005841 | Perceptual: 0.216766 | Time: 303.9s
[2025-12-26 10:16:36,734][medgen.pipeline.utils][INFO] - [10:16:36] Epoch  98/125 ( 78.4%) completed | Total: 0.005999 | MSE: 0.005780 | Perceptual: 0.218638 | Time: 303.5s
[2025-12-26 10:22:40,833][medgen.pipeline.utils][INFO] - [10:22:40] Epoch  99/125 ( 79.2%) completed | Total: 0.006010 | MSE: 0.005793 | Perceptual: 0.216892 | Time: 305.2s
[2025-12-26 10:28:43,901][medgen.pipeline.utils][INFO] - [10:28:43] Epoch 100/125 ( 80.0%) completed | Total: 0.006057 | MSE: 0.005833 | Perceptual: 0.224205 | Time: 303.8s
[2025-12-26 10:34:47,300][medgen
```

#### `exp10_1_multi_23873173`  [❌ crashed]
*job 23873173 • idun-07-09 • A100 • epoch 125/125 • 38KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007378 • MS-SSIM 0.8625 • PSNR 28.41 dB • LPIPS 0.5479
  - **latest** ckpt (13359 samples): MSE 0.009486 • MS-SSIM 0.8430 • PSNR 27.54 dB • LPIPS 0.5009
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 343, in generate_samples
    raise ValueError(f"Unknown mode: {self.mode_name}")
...
[2025-12-27 02:47:16,593][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_1_rflow_128_20251226-120701/test_worst_batch_best.png
[2025-12-27 02:47:16,903][medgen.pipeline.trainer][INFO] - Loaded latest checkpoint for test evaluation
[2025-12-27 02:47:16,903][medgen.pipeline.trainer][INFO] - ============================================================
[2025-12-27 02:47:16,903][medgen.pipeline.trainer][INFO] - EVALUATING ON TEST SET (LATEST MODEL)
[2025-12
```

#### `exp10_2_multi_scoreaug_23873174`  [❌ crashed]
*job 23873174 • idun-07-09 • A100 • epoch 125/125 • 39KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007156 • MS-SSIM 0.8643 • PSNR 28.35 dB • LPIPS 0.5086
  - **latest** ckpt (13359 samples): MSE 0.007523 • MS-SSIM 0.8588 • PSNR 28.10 dB • LPIPS 0.4956
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 343, in generate_samples
    raise ValueError(f"Unknown mode: {self.mode_name}")
...
[2025-12-27 02:48:55,937][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_2_rflow_128_20251226-120701/test_worst_batch_best.png
[2025-12-27 02:48:56,239][medgen.pipeline.trainer][INFO] - Loaded latest checkpoint for test evaluation
[2025-12-27 02:48:56,239][medgen.pipeline.trainer][INFO] - ============================================================
[2025-12-27 02:48:56,239][medgen.pipeline.trainer][INFO] - EVALUATING ON TEST SET (LATEST MODEL)
[2025-12
```

#### `exp10_3_multi_compose_23873213`  [❌ crashed]
*job 23873213 • idun-06-02 • A100 • epoch 125/125 • 39KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007056 • MS-SSIM 0.8688 • PSNR 28.49 dB • LPIPS 0.5523
  - **latest** ckpt (13359 samples): MSE 0.007188 • MS-SSIM 0.8660 • PSNR 28.39 dB • LPIPS 0.5007
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 343, in generate_samples
    raise ValueError(f"Unknown mode: {self.mode_name}")
...
[2025-12-27 03:32:25,485][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_3_rflow_128_20251226-142311/test_worst_batch_best.png
[2025-12-27 03:32:25,729][medgen.pipeline.trainer][INFO] - Loaded latest checkpoint for test evaluation
[2025-12-27 03:32:25,729][medgen.pipeline.trainer][INFO] - ============================================================
[2025-12-27 03:32:25,729][medgen.pipeline.trainer][INFO] - EVALUATING ON TEST SET (LATEST MODEL)
[2025-12
```

#### `exp10_2_multi_scoreaug_23873621`  [❌ crashed]
*job 23873621 • idun-06-07 • A100 • epoch 125/125 • 48KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007026 • MS-SSIM 0.8660 • PSNR 28.43 dB • LPIPS 0.5507
  - **latest** ckpt (13359 samples): MSE 0.007711 • MS-SSIM 0.8579 • PSNR 28.06 dB • LPIPS 0.4954
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 353, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2025-12-28 03:56:06,452][medgen.pipeline.trainer][INFO] -   MS-SSIM: 0.8660
[2025-12-28 03:56:06,452][medgen.pipeline.trainer][INFO] -   PSNR:    28.43 dB
[2025-12-28 03:56:06,452][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5507
[2025-12-28 03:56:06,473][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_2_rflow_128_20251227-144605/test_results_best.json
[2025-12-28 03:56:07,421][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900
```

#### `exp10_3_multi_compose_23873622`  [❌ crashed]
*job 23873622 • idun-07-07 • A100 • epoch 125/125 • 48KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007064 • MS-SSIM 0.8677 • PSNR 28.44 dB • LPIPS 0.5334
  - **latest** ckpt (13359 samples): MSE 0.00714 • MS-SSIM 0.8674 • PSNR 28.41 dB • LPIPS 0.4919
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 353, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2025-12-28 07:02:54,567][medgen.pipeline.trainer][INFO] -   MS-SSIM: 0.8677
[2025-12-28 07:02:54,567][medgen.pipeline.trainer][INFO] -   PSNR:    28.44 dB
[2025-12-28 07:02:54,567][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5334
[2025-12-28 07:02:54,569][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_3_rflow_128_20251227-145449/test_results_best.json
[2025-12-28 07:02:55,919][medgen.pipeline.trainer][INFO] - Test worst batch saved to: /cluster/work/modestas/AIS4900
```

#### `exp10_4_multi_23888484`  [❌ crashed]
*job 23888484 • idun-07-06 • A100 • epoch 125/125 • 84KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007298 • MS-SSIM 0.8678 • PSNR 28.38 dB • LPIPS 0.4807
  - **latest** ckpt (13359 samples): MSE 0.01087 • MS-SSIM 0.8360 • PSNR 27.17 dB • LPIPS 0.4784
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-10 17:53:14,062][medgen.pipeline.trainer][INFO] -   LPIPS:   0.4807
[2026-01-10 17:53:14,063][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_4_rflow_128_20260110-014655/test_results_best.json
[2026-01-10 17:53:14,070][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', '
```

#### `exp10_5_multi_scoreaug_23888485`  [❌ crashed]
*job 23888485 • idun-07-05 • A100 • epoch 125/125 • 85KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.006967 • MS-SSIM 0.8722 • PSNR 28.63 dB • LPIPS 0.5412
  - **latest** ckpt (13359 samples): MSE 0.007102 • MS-SSIM 0.8694 • PSNR 28.39 dB • LPIPS 0.4651
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-10 17:33:48,303][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5412
[2026-01-10 17:33:48,305][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_5_rflow_128_20260110-014655/test_results_best.json
[2026-01-10 17:33:48,312][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', '
```

#### `exp10_6_multi_dropout_23888512`  [❌ crashed]
*job 23888512 • idun-06-06 • A100 • epoch 34/125 • 27KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-11 09:00:26,102][medgen.pipeline.utils][INFO] - [09:00:26] Epoch  33/125 ( 26.4%) completed | Total: 0.006566 | MSE: 0.006147 | Perceptual: 0.419032 | Time: 37411.6s
[2026-01-11 09:01:20,968][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/val/Mets_009/multi.nii.gz
Available modalities in Mets_009: ['seg', 't1_gd', 'flair', 't1_pre', 'bravo']
[2026-01-12 21:52:55,909][medgen.pipeline.utils][INFO] -
```

#### `exp10_7_multi_none_23888513`  [❌ crashed]
*job 23888513 • idun-07-05 • A100 • epoch 125/125 • 84KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007166 • MS-SSIM 0.8647 • PSNR 28.38 dB • LPIPS 0.5235
  - **latest** ckpt (13359 samples): MSE 0.00948 • MS-SSIM 0.8397 • PSNR 27.50 dB • LPIPS 0.5067
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-11 03:26:44,957][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5235
[2026-01-11 03:26:44,959][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_7_rflow_128_20260110-113040/test_results_best.json
[2026-01-11 03:26:44,962][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', '
```

#### `exp10_8_multi_late_23888514`  [❌ crashed]
*job 23888514 • idun-07-05 • A100 • epoch 125/125 • 84KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.007188 • MS-SSIM 0.8623 • PSNR 28.30 dB • LPIPS 0.5167
  - **latest** ckpt (13359 samples): MSE 0.009574 • MS-SSIM 0.8416 • PSNR 27.52 dB • LPIPS 0.5024
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-11 03:30:10,216][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5167
[2026-01-11 03:30:10,238][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_8_rflow_128_20260110-113040/test_results_best.json
[2026-01-11 03:30:10,245][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', '
```

#### `exp10_10_multi_film_scoreaug_23888810`  [❌ crashed]
*job 23888810 • idun-07-05 • A100 • epoch 125/125 • 85KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.006784 • MS-SSIM 0.8733 • PSNR 28.57 dB • LPIPS 0.5098
  - **latest** ckpt (13359 samples): MSE 0.007025 • MS-SSIM 0.8682 • PSNR 28.37 dB • LPIPS 0.4693
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-11 18:19:54,419][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5098
[2026-01-11 18:19:54,420][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_10_rflow_128_20260111-001830/test_results_best.json
[2026-01-11 18:19:54,423][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', 
```

#### `exp10_9_multi_film_23888811`  [❌ crashed]
*job 23888811 • idun-09-16 • A100 • epoch 125/125 • 84KB log*

**Final test metrics:**
  - **best** ckpt (13359 samples): MSE 0.00725 • MS-SSIM 0.8687 • PSNR 28.42 dB • LPIPS 0.5119
  - **latest** ckpt (13359 samples): MSE 0.01096 • MS-SSIM 0.8358 • PSNR 27.14 dB • LPIPS 0.4757
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/visualization.py", line 369, in generate_samples
    seg_masks, gt_images = self._sample_positive_masks(
...
[2026-01-11 19:55:02,668][medgen.pipeline.trainer][INFO] -   LPIPS:   0.5119
[2026-01-11 19:55:02,670][medgen.pipeline.trainer][INFO] - Test results saved to: /cluster/work/modestas/AIS4900_master/runs/diffusion_2d/multi/exp10_9_rflow_128_20260111-001830/test_results_best.json
[2026-01-11 19:55:02,688][medgen.data.loaders.vae][WARNING] - Volume validation directory misconfigured: Modality 'multi' not found in dataset.
Expected: /cluster/work/modestas/MedicalDataSets/brainmetshare-3/test_new/Mets_021/multi.nii.gz
Available modalities in Mets_021: ['flair', 'bravo', '
```

#### `exp10_1_sit_dcae_8x8_23998878`  [💥 oom_killed]
*job 23998878 • idun-06-02 • A100 • epoch 500/500 • 97KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	1900m19.037s
user	1326m19.930s
sys	578m11.133s
[2026-02-03T19:39:34.813] error: Detected 1 oom_kill event in StepId=23998878.batch. Some of the step tasks have been OOM Killed.
```

#### `exp10_2_sit_dcae_4x4_23998879`  [❌ crashed]
*job 23998879 • idun-06-02 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (2560) at non-singleton dimension 1

Set the environment variable HY
```

#### `exp10_3_sit_dcae_2x2_23998880`  [❌ crashed]
*job 23998880 • idun-06-02 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 106, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (640) at non-singleton dimension 1

Set the environment variable HYD
```

#### `exp10_2_sit_dcae_4x4_24039865`  [❌ crashed]
*job 24039865 • idun-06-01 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (2560) at non-singleton dimension 1

Set the environment variable HYDRA_FULL_E
```

#### `exp10_3_sit_dcae_2x2_24039866`  [❌ crashed]
*job 24039866 • idun-06-01 • A100 • 13KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    x = self.x_embedder(x) + self.pos_embed  # [B, N, D]
        ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (10240) must match the size of tensor b (640) at non-singleton dimension 1

Set the environment variable HYDRA_FULL_ER
```

### exp11 (6 jobs)

*Status: ✅ completed=2 ❌ crashed=2 💥 oom_killed=2*

#### `exp11_1_sam_23873928`  [✅ completed]
*job 23873928 • idun-09-16 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006618 • MS-SSIM 0.8620 • PSNR 28.93 dB • LPIPS 0.5806
  - **latest** ckpt (3339 samples): MSE 0.009596 • MS-SSIM 0.8358 • PSNR 27.96 dB • LPIPS 0.5505

#### `exp11_2_asam_23873929`  [✅ completed]
*job 23873929 • idun-09-16 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006975 • MS-SSIM 0.8609 • PSNR 28.82 dB • LPIPS 0.5550
  - **latest** ckpt (3339 samples): MSE 0.01153 • MS-SSIM 0.8268 • PSNR 27.63 dB • LPIPS 0.5007

#### `exp11_s2d_pixel_bravo_24031954`  [❌ crashed]
*job 24031954 • idun-07-10 • A100 • epoch 500/500 • 227KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1186, in train
    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_ms
```

#### `exp11_1_s2d_pixel_bravo_24031955`  [❌ crashed]
*job 24031955 • idun-06-02 • A100 • epoch 64/500 • 64KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 99, in add_event
    self.event_writer.add_event(event)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 117, in add_event
    self._async_writer.write(event.SerializeToString())
  File "/cluster/home/modestas/
```

#### `exp11_1_s2d_pixel_bravo_24039873`  [💥 oom_killed]
*job 24039873 • idun-01-03 • H100 • epoch 3/500 • 8KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	16m41.222s
user	11m6.118s
sys	4m22.979s
[2026-02-13T03:50:42.532] error: Detected 1 oom_kill event in StepId=24039873.batch. Some of the step tasks have been OOM Killed.
```

#### `exp11_s2d_pixel_bravo_24039874`  [💥 oom_killed]
*job 24039874 • idun-09-16 • A100 • epoch 4/500 • 15KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	18m34.137s
user	16m40.842s
sys	1m27.822s
[2026-02-13T00:29:46.878] error: Detected 1 oom_kill event in StepId=24039874.batch. Some of the step tasks have been OOM Killed.
```

### exp12 (9 jobs)

*Status: 💥 oom_killed=4 ✅ completed=3 ⚠️ truncated=1 ❌ crashed=1*

#### `exp12_1_sit_s_23875062`  [✅ completed]
*job 23875062 • idun-07-09 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006907 • MS-SSIM 0.8661 • PSNR 29.06 dB • LPIPS 0.5078
  - **latest** ckpt (3339 samples): MSE 0.01121 • MS-SSIM 0.8321 • PSNR 27.75 dB • LPIPS 0.4945

#### `exp12_2_sit_b_23875068`  [✅ completed]
*job 23875068 • idun-06-05 • A100 • epoch 500/500 • 93KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006996 • MS-SSIM 0.8669 • PSNR 29.05 dB • LPIPS 0.5079
  - **latest** ckpt (3339 samples): MSE 0.01265 • MS-SSIM 0.8268 • PSNR 27.56 dB • LPIPS 0.4999

#### `exp12_1b_sit_s_23962280`  [✅ completed]
*job 23962280 • idun-06-04 • A100 • epoch 500/500 • 101KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006555 • MS-SSIM 0.8641 • PSNR 29.32 dB • LPIPS 1.0032 • FID 45.95 • KID 0.0566 ± 0.0078 • CMMD 0.2510
  - **latest** ckpt (3339 samples): MSE 0.009847 • MS-SSIM 0.8369 • PSNR 28.41 dB • LPIPS 0.8424 • FID 33.43 • KID 0.0375 ± 0.0078 • CMMD 0.2155

#### `exp12_2b_sit_b_23962281`  [⚠️ truncated]
*job 23962281 • idun-07-09 • A100 • epoch 410/500 • 81KB log*


#### `exp12_1c_sit_s_23971736`  [💥 oom_killed]
*job 23971736 • idun-09-18 • A100 • epoch 500/500 • 91KB log*

**Final test metrics:**
  - **best** ckpt (3342 samples): MSE 0.005555 • MS-SSIM 0.9070 • PSNR 30.17 dB • LPIPS 0.7489
**Traceback excerpt:**
```
/var/slurm_spool/job23971736/slurm_script: line 39: 2441551 Killed                  python -m medgen.scripts.train paths=cluster strategy=rflow mode=bravo model=sit model.variant=S model.patch_size=4 model.image_size=256 training.name=exp12_1c_

real	3852m33.440s
user	3805m25.736s
sys	132m55.166s
[2026-01-23T18:47:10.861] error: Detected 1 oom_kill event in StepId=23971736.batch. Some of the step tasks have been OOM Killed.
```

#### `exp12_wavelet_pixel_bravo_24031956`  [💥 oom_killed]
*job 24031956 • idun-07-07 • A100 • epoch 49/500 • 55KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/unified.py", line 975, in log_generated_samples
    log_generated_samples(self, samples, epoch, tag, nrow, num_slices)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/unified_visualization.py", line 164, in log_generated_samples
    _log_generated_samples_3d(metrics, samples, epoch, tag, num_slices)
  File "/cluster/work/m
```

#### `exp12_1_wavelet_pixel_bravo_24031957`  [❌ crashed]
*job 24031957 • idun-06-07 • A100 • epoch 500/500 • 242KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
    _train_3d(cfg)
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 670, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/trainer.py", line 1186, in train
    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_ms
```

#### `exp12_1_wavelet_pixel_bravo_24039875`  [💥 oom_killed]
*job 24039875 • idun-01-03 • H100 • epoch 210/500 • 182KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	1006m44.493s
user	674m26.029s
sys	384m3.159s
[2026-02-13T20:20:46.722] error: Detected 1 oom_kill event in StepId=24039875.batch. Some of the step tasks have been OOM Killed.
```

#### `exp12_wavelet_pixel_bravo_24039876`  [💥 oom_killed]
*job 24039876 • idun-09-16 • A100 • epoch 10/500 • 17KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	45m17.717s
user	40m47.783s
sys	3m50.149s
[2026-02-13T00:57:01.381] error: Detected 1 oom_kill event in StepId=24039876.batch. Some of the step tasks have been OOM Killed.
```

### exp13 (2 jobs)

*Status: ✅ completed=2*

#### `exp13_1_ema_simple_23888433`  [✅ completed]
*job 23888433 • idun-06-06 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006899 • MS-SSIM 0.8373 • PSNR 28.70 dB • LPIPS 0.5354
  - **latest** ckpt (3339 samples): MSE 0.007719 • MS-SSIM 0.8262 • PSNR 28.37 dB • LPIPS 0.5475

#### `exp13_2_ema_slow_23888434`  [✅ completed]
*job 23888434 • idun-07-08 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006953 • MS-SSIM 0.8373 • PSNR 28.67 dB • LPIPS 0.5329
  - **latest** ckpt (3339 samples): MSE 0.007249 • MS-SSIM 0.8288 • PSNR 28.43 dB • LPIPS 0.5424

### exp14 (3 jobs)

*Status: ✅ completed=3*

#### `exp14_1_sit_s_drop01_23888435`  [✅ completed]
*job 23888435 • idun-07-08 • A100 • epoch 500/500 • 95KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006803 • MS-SSIM 0.8630 • PSNR 29.24 dB • LPIPS 0.5025
  - **latest** ckpt (3339 samples): MSE 0.008866 • MS-SSIM 0.8434 • PSNR 28.50 dB • LPIPS 0.4900

#### `exp14_2_sit_s_drop02_23888436`  [✅ completed]
*job 23888436 • idun-07-09 • A100 • epoch 500/500 • 95KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006873 • MS-SSIM 0.8668 • PSNR 29.28 dB • LPIPS 0.5006
  - **latest** ckpt (3339 samples): MSE 0.008374 • MS-SSIM 0.8403 • PSNR 28.45 dB • LPIPS 0.4982

#### `exp14_3_sit_s_drop03_23888437`  [✅ completed]
*job 23888437 • idun-07-10 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006707 • MS-SSIM 0.8688 • PSNR 29.37 dB • LPIPS 0.5056
  - **latest** ckpt (3339 samples): MSE 0.007963 • MS-SSIM 0.8443 • PSNR 28.56 dB • LPIPS 0.5007

### exp15 (3 jobs)

*Status: ✅ completed=3*

#### `exp15_1_rflow_wd001_23888828`  [✅ completed]
*job 23888828 • idun-07-04 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006569 • MS-SSIM 0.8658 • PSNR 29.26 dB • LPIPS 0.5509
  - **latest** ckpt (3339 samples): MSE 0.01148 • MS-SSIM 0.8287 • PSNR 28.20 dB • LPIPS 0.5028

#### `exp15_2_rflow_wd005_23888829`  [✅ completed]
*job 23888829 • idun-07-06 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006687 • MS-SSIM 0.8641 • PSNR 29.30 dB • LPIPS 0.5374
  - **latest** ckpt (3339 samples): MSE 0.01175 • MS-SSIM 0.8298 • PSNR 28.24 dB • LPIPS 0.5064

#### `exp15_3_rflow_wd01_23888830`  [✅ completed]
*job 23888830 • idun-07-07 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006695 • MS-SSIM 0.8632 • PSNR 29.24 dB • LPIPS 0.5227
  - **latest** ckpt (3339 samples): MSE 0.01098 • MS-SSIM 0.8329 • PSNR 28.32 dB • LPIPS 0.5107

### exp16 (4 jobs)

*Status: ✅ completed=3 ⚠️ truncated=1*

#### `exp16_bs32_23889063`  [⚠️ truncated]
*job 23889063 • idun-06-06 • A100 • 6KB log*


#### `exp16_bs24_23889064`  [✅ completed]
*job 23889064 • idun-01-04 • H100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006429 • MS-SSIM 0.8639 • PSNR 29.28 dB • LPIPS 0.6900
  - **latest** ckpt (3339 samples): MSE 0.01156 • MS-SSIM 0.8327 • PSNR 28.32 dB • LPIPS 0.5757

#### `exp16_bs4_23889065`  [✅ completed]
*job 23889065 • idun-09-18 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006662 • MS-SSIM 0.8715 • PSNR 29.73 dB • LPIPS 0.5159
  - **latest** ckpt (3339 samples): MSE 0.01218 • MS-SSIM 0.8302 • PSNR 28.54 dB • LPIPS 0.5033

#### `exp16_bs8_23889066`  [✅ completed]
*job 23889066 • idun-09-18 • A100 • epoch 500/500 • 94KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006576 • MS-SSIM 0.8671 • PSNR 29.54 dB • LPIPS 0.5042
  - **latest** ckpt (3339 samples): MSE 0.0118 • MS-SSIM 0.8294 • PSNR 28.35 dB • LPIPS 0.5049

### exp17 (3 jobs)

*Status: ✅ completed=3*

#### `exp17_1_droppath_23898319`  [✅ completed]
*job 23898319 • idun-07-04 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00684 • MS-SSIM 0.8671 • PSNR 29.35 dB • LPIPS 0.4966
  - **latest** ckpt (3339 samples): MSE 0.00868 • MS-SSIM 0.8448 • PSNR 28.60 dB • LPIPS 0.5026

#### `exp17_2_droppath_23898320`  [✅ completed]
*job 23898320 • idun-07-04 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006765 • MS-SSIM 0.8646 • PSNR 29.28 dB • LPIPS 0.5014
  - **latest** ckpt (3339 samples): MSE 0.008754 • MS-SSIM 0.8416 • PSNR 28.48 dB • LPIPS 0.5263

#### `exp17_3_droppath_23898321`  [✅ completed]
*job 23898321 • idun-07-04 • A100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006787 • MS-SSIM 0.8677 • PSNR 29.37 dB • LPIPS 0.5009
  - **latest** ckpt (3339 samples): MSE 0.008296 • MS-SSIM 0.8485 • PSNR 28.72 dB • LPIPS 0.4904

### exp18 (1 jobs)

*Status: ✅ completed=1*

#### `exp18_1_sda_23914579`  [✅ completed]
*job 23914579 • idun-07-07 • A100 • epoch 500/500 • 101KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006523 • MS-SSIM 0.8662 • PSNR 29.34 dB • LPIPS 0.5099 • FID 38.19 • KID 0.0446 ± 0.0087 • CMMD 0.2420
  - **latest** ckpt (3339 samples): MSE 0.006974 • MS-SSIM 0.8630 • PSNR 29.22 dB • LPIPS 0.4739 • FID 40.37 • KID 0.0502 ± 0.0101 • CMMD 0.2296

### exp19 (1 jobs)

*Status: ✅ completed=1*

#### `exp19_1_constant_lr_23898376`  [✅ completed]
*job 23898376 • idun-01-03 • H100 • epoch 500/500 • 95KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006319 • MS-SSIM 0.8653 • PSNR 29.40 dB • LPIPS 0.6434
  - **latest** ckpt (3339 samples): MSE 0.009889 • MS-SSIM 0.8402 • PSNR 28.55 dB • LPIPS 0.5694

### exp20 (1 jobs)

*Status: ✅ completed=1*

#### `exp20_1_grad_noise_23898377`  [✅ completed]
*job 23898377 • idun-01-03 • H100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.00628 • MS-SSIM 0.8688 • PSNR 29.49 dB • LPIPS 0.6690
  - **latest** ckpt (3339 samples): MSE 0.007585 • MS-SSIM 0.8484 • PSNR 28.82 dB • LPIPS 0.6032

### exp21 (1 jobs)

*Status: ✅ completed=1*

#### `exp21_1_curriculum_23898378`  [✅ completed]
*job 23898378 • idun-01-03 • H100 • epoch 500/500 • 96KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006557 • MS-SSIM 0.8644 • PSNR 29.40 dB • LPIPS 0.6381
  - **latest** ckpt (3339 samples): MSE 0.009095 • MS-SSIM 0.8418 • PSNR 28.57 dB • LPIPS 0.5856

### exp22 (1 jobs)

*Status: ✅ completed=1*

#### `exp22_1_timestep_jitter_23914620`  [✅ completed]
*job 23914620 • idun-09-16 • A100 • epoch 500/500 • 100KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006554 • MS-SSIM 0.8668 • PSNR 29.35 dB • LPIPS 0.4919 • FID 55.40 • KID 0.0735 ± 0.0098 • CMMD 0.2404
  - **latest** ckpt (3339 samples): MSE 0.01146 • MS-SSIM 0.8282 • PSNR 28.19 dB • LPIPS 0.5016 • FID 37.16 • KID 0.0452 ± 0.0107 • CMMD 0.2276

### exp23 (1 jobs)

*Status: ✅ completed=1*

#### `exp23_1_noise_aug_23914630`  [✅ completed]
*job 23914630 • idun-09-18 • A100 • epoch 500/500 • 100KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006886 • MS-SSIM 0.8660 • PSNR 29.36 dB • LPIPS 0.5628 • FID 55.89 • KID 0.0706 ± 0.0080 • CMMD 0.2588
  - **latest** ckpt (3339 samples): MSE 0.01183 • MS-SSIM 0.8307 • PSNR 28.26 dB • LPIPS 0.5079 • FID 36.67 • KID 0.0417 ± 0.0080 • CMMD 0.2270

### exp24 (1 jobs)

*Status: ✅ completed=1*

#### `exp24_1_feature_perturb_23914631`  [✅ completed]
*job 23914631 • idun-01-03 • H100 • epoch 500/500 • 100KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006492 • MS-SSIM 0.8674 • PSNR 29.45 dB • LPIPS 0.6518 • FID 46.13 • KID 0.0586 ± 0.0096 • CMMD 0.2342
  - **latest** ckpt (3339 samples): MSE 0.01153 • MS-SSIM 0.8314 • PSNR 28.34 dB • LPIPS 0.5888 • FID 37.65 • KID 0.0440 ± 0.0088 • CMMD 0.2256

### exp25 (1 jobs)

*Status: ✅ completed=1*

#### `exp25_1_self_cond_23914632`  [✅ completed]
*job 23914632 • idun-01-03 • H100 • epoch 500/500 • 100KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.006714 • MS-SSIM 0.8594 • PSNR 29.25 dB • LPIPS 0.6803 • FID 60.96 • KID 0.0844 ± 0.0120 • CMMD 0.2691
  - **latest** ckpt (3339 samples): MSE 0.01181 • MS-SSIM 0.8327 • PSNR 28.35 dB • LPIPS 0.5887 • FID 37.42 • KID 0.0409 ± 0.0091 • CMMD 0.2264

### exp26 (1 jobs)

*Status: ✅ completed=1*

#### `exp26_regional_weight_23939370`  [✅ completed]
*job 23939370 • idun-06-04 • A100 • epoch 500/500 • 99KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.007059 • MS-SSIM 0.8666 • PSNR 29.43 dB • LPIPS 0.4980 • FID 48.37 • KID 0.0599 ± 0.0100 • CMMD 0.2442
  - **latest** ckpt (3339 samples): MSE 0.01161 • MS-SSIM 0.8288 • PSNR 28.27 dB • LPIPS 0.5017 • FID 36.45 • KID 0.0433 ± 0.0096 • CMMD 0.2295

### exp27 (1 jobs)

*Status: 💥 oom_killed=1*

#### `exp27_plateau_23973550`  [💥 oom_killed]
*job 23973550 • idun-07-06 • A100 • epoch 375/500 • 67KB log*

**Traceback excerpt:**
```
/var/slurm_spool/job23973550/slurm_script: line 53: 188354 Killed                  python -m medgen.scripts.train paths=cluster strategy=rflow mode=bravo model.image_size=128 training.scheduler=plateau training.plateau.factor=0.5 training.plateau.patience=10 training.plateau.min_lr=1e-6 training.name=exp27_rflow_plateau_

real	1340m15.850s
user	1335m32.194s
sys	59m12.667s
[2026-01-23T11:42:29.587] error: Detected 1 oom_kill event in StepId=23973550.batch. Some of the step tasks have been OOM Killed.
```

### OTHER (623 jobs)

*Status: 🔗 chained=416 ✅ completed=115 ❌ crashed=35 💥 oom_killed=32 ⚠️ truncated=25*

#### `24039895_24039895`  [🔗 chained]
*job 24039895 • idun-09-16 • A100 • epoch 7/100 • chain 0/20 • 11KB log*


#### `24039896_24039896`  [🔗 chained]
*job 24039896 • idun-09-16 • A100 • epoch 15/100 • chain 1/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039899_24039899`  [🔗 chained]
*job 24039899 • idun-09-16 • A100 • epoch 21/100 • chain 2/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039900_24039900`  [🔗 chained]
*job 24039900 • idun-09-16 • A100 • epoch 29/100 • chain 3/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039902_24039902`  [🔗 chained]
*job 24039902 • idun-09-16 • A100 • epoch 37/100 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039904_24039904`  [🔗 chained]
*job 24039904 • idun-09-16 • A100 • epoch 44/100 • chain 5/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039907_24039907`  [🔗 chained]
*job 24039907 • idun-07-07 • A100 • epoch 52/100 • chain 6/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039908_24039908`  [🔗 chained]
*job 24039908 • idun-09-16 • A100 • epoch 60/100 • chain 7/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039909_24039909`  [🔗 chained]
*job 24039909 • idun-09-16 • A100 • epoch 68/100 • chain 8/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039911_24039911`  [🔗 chained]
*job 24039911 • idun-09-16 • A100 • epoch 75/100 • chain 9/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039915_24039915`  [🔗 chained]
*job 24039915 • idun-09-16 • A100 • epoch 83/100 • chain 10/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039922_24039922`  [🔗 chained]
*job 24039922 • idun-07-07 • A100 • epoch 91/100 • chain 11/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039936_24039936`  [🔗 chained]
*job 24039936 • idun-09-16 • A100 • epoch 99/100 • chain 12/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24039971_24039971`  [✅ completed]
*job 24039971 • idun-09-16 • A100 • 0.11h training • epoch 100/100 • chain 13/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_chained_20260213-031857/checkpoint_latest.pt`

#### `24042396_24042396`  [🔗 chained]
*job 24042396 • idun-09-16 • A100 • epoch 405/500 • chain 0/20 • 69KB log*


#### `24042475_24042475`  [🔗 chained]
*job 24042475 • idun-07-05 • A100 • epoch 489/500 • chain 0/20 • 110KB log*


#### `24042655_24042655`  [🔗 chained]
*job 24042655 • idun-06-07 • A100 • epoch 374/500 • chain 0/20 • 127KB log*


#### `24042656_24042656`  [🔗 chained]
*job 24042656 • idun-06-07 • A100 • epoch 18/500 • chain 0/20 • 11KB log*


#### `24042660_24042660`  [🔗 chained]
*job 24042660 • idun-07-09 • A100 • epoch 488/500 • chain 0/20 • 83KB log*


#### `24042663_24042663`  [❌ crashed]
*job 24042663 • idun-06-07 • A100 • 2.05h training • epoch 500/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260214-163133/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042671_24042671`  [❌ crashed]
*job 24042671 • idun-06-07 • A100 • 0.26h training • epoch 500/500 • chain 1/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp11_2_20260214-203209/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042824_24042824`  [❌ crashed]
*job 24042824 • idun-06-04 • A100 • 4.17h training • epoch 500/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_20260215-023826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surf
```

#### `24042825_24042825`  [🔗 chained]
*job 24042825 • idun-06-07 • A100 • epoch 35/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24042844_24042844`  [❌ crashed]
*job 24042844 • idun-06-07 • A100 • 0.27h training • epoch 500/500 • chain 1/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp13_dit_8x_bravo_20260215-024826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 139, in marching_cubes
    return _marching_cubes_lewiner(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/skimage/measure/_marching_cubes_lewiner.py", line 180, in _marching_cubes_lewiner
    raise ValueError("Surface level must be w
```

#### `24042958_24042958`  [✅ completed]
*job 24042958 • idun-06-07 • A100 • 10.88h training • epoch 500/500 • chain 0/20 • 88KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001044 • MS-SSIM 0.8980 • PSNR 30.31 dB • LPIPS 0.6941 • FID 76.67 • KID 0.0576 ± 0.0089 • CMMD 0.3962
  - **latest** ckpt (26 samples): MSE 0.000989 • MS-SSIM 0.9043 • PSNR 30.64 dB • LPIPS 0.6711 • FID 78.37 • KID 0.0591 ± 0.0077 • CMMD 0.3767

#### `24042959_24042959`  [🔗 chained]
*job 24042959 • idun-07-09 • A100 • epoch 483/500 • chain 0/20 • 110KB log*


#### `24042968_24042968`  [❌ crashed]
*job 24042968 • idun-07-08 • A100 • 11.86h training • epoch 500/500 • chain 0/20 • 89KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.09429 • MS-SSIM 0.9455 • PSNR 32.14 dB • LPIPS 0.3282
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W216 06:31:15.521623016 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	715m47.475s
user	697m25.791s
sys	121m3.781s
```

#### `24043083_24043083`  [🔗 chained]
*job 24043083 • idun-07-10 • A100 • epoch 234/500 • chain 0/20 • 54KB log*


#### `24043086_24043086`  [🔗 chained]
*job 24043086 • idun-07-10 • A100 • epoch 358/500 • chain 0/20 • 64KB log*


#### `24043087_24043087`  [🔗 chained]
*job 24043087 • idun-06-07 • A100 • epoch 52/500 • chain 2/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24043097_24043097`  [✅ completed]
*job 24043097 • idun-06-04 • A100 • 0.39h training • epoch 500/500 • chain 1/20 • 15KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.06197 • MS-SSIM 0.6490 • PSNR 23.16 dB • LPIPS 1.7949 • FID 277.37 • KID 0.3363 ± 0.0188 • CMMD 0.6074
  - **latest** ckpt (26 samples): MSE 0.06664 • MS-SSIM 0.7233 • PSNR 25.53 dB • LPIPS 1.7187 • FID 282.13 • KID 0.3426 ± 0.0202 • CMMD 0.6064
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_3_20260215-175625/checkpoint_latest.pt`

#### `24043682_24043682`  [🔗 chained]
*job 24043682 • idun-07-10 • A100 • epoch 464/500 • chain 1/20 • 558KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_3_sit_s_128_patch4_20260216-020438/checkpoint_latest.pt`

#### `24043701_24043701`  [💥 oom_killed]
*job 24043701 • idun-07-10 • A100 • 4.95h training • epoch 500/500 • chain 1/20 • 352KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001115 • MS-SSIM 0.8986 • PSNR 30.10 dB • LPIPS 0.7983 • FID 71.11 • KID 0.0443 ± 0.0050 • CMMD 0.3321
  - **latest** ckpt (26 samples): MSE 0.001632 • MS-SSIM 0.8403 • PSNR 28.19 dB • LPIPS 0.7647
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_5_20260216-022108/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	305m34.660s
user	257m48.428s
sys	43m24.249s
[2026-02-16T19:24:47.568] error: Detected 2 oom_kill events in StepId=24043701.batch. Some of the step tasks have been OOM Killed.
```

#### `24043721_24043721`  [🔗 chained]
*job 24043721 • idun-06-07 • A100 • epoch 70/500 • chain 3/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24044737_24044737`  [🔗 chained]
*job 24044737 • idun-09-16 • A100 • epoch 199/500 • chain 0/20 • 483KB log*


#### `24045125_24045125`  [🔗 chained]
*job 24045125 • idun-07-09 • A100 • epoch 115/500 • chain 0/20 • 127KB log*


#### `24045647_24045647`  [✅ completed]
*job 24045647 • idun-01-03 • H100 • 1.1h training • epoch 500/500 • chain 2/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002765 • MS-SSIM 0.9342 • PSNR 32.06 dB • LPIPS 0.6456 • FID 126.78 • KID 0.1514 ± 0.0092 • CMMD 0.4015
  - **latest** ckpt (26 samples): MSE 0.00399 • MS-SSIM 0.9511 • PSNR 32.83 dB • LPIPS 0.5543 • FID 115.82 • KID 0.1340 ± 0.0066 • CMMD 0.3700
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_3_sit_s_128_patch4_20260216-020438/checkpoint_latest.pt`

#### `24045747_24045747`  [🔗 chained]
*job 24045747 • idun-06-07 • A100 • epoch 86/500 • chain 4/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24045963_24045963`  [🔗 chained]
*job 24045963 • idun-09-16 • A100 • epoch 397/500 • chain 1/20 • 197KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_6_20260216-161013/checkpoint_latest.pt`

#### `24047474_24047474`  [🔗 chained]
*job 24047474 • idun-01-04 • H100 • epoch 311/500 • chain 1/20 • 47KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt`

#### `24047482_24047482`  [✅ completed]
*job 24047482 • idun-07-06 • A100 • 6.2h training • epoch 500/500 • chain 2/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00105 • MS-SSIM 0.8980 • PSNR 30.19 dB • LPIPS 0.6533 • FID 59.49 • KID 0.0385 ± 0.0059 • CMMD 0.3065
  - **latest** ckpt (26 samples): MSE 0.001116 • MS-SSIM 0.8908 • PSNR 30.03 dB • LPIPS 0.6292 • FID 60.42 • KID 0.0388 ± 0.0039 • CMMD 0.3124
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_6_20260216-161013/checkpoint_latest.pt`

#### `24051502_24051502`  [🔗 chained]
*job 24051502 • idun-07-10 • A100 • epoch 100/500 • chain 5/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24052178_24052178`  [✅ completed]
*job 24052178 • idun-01-04 • H100 • 11.49h training • epoch 500/500 • chain 2/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt`

#### `24052709_24052709`  [🔗 chained]
*job 24052709 • idun-07-08 • A100 • epoch 114/500 • chain 6/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24053109_24053109`  [🔗 chained]
*job 24053109 • idun-07-10 • A100 • epoch 313/2000 • chain 0/20 • 104KB log*


#### `24053112_24053112`  [⚠️ truncated]
*job 24053112 • idun-07-08 • A100 • epoch 390/2000 • chain 0/20 • 108KB log*


#### `24055687_24055687`  [🔗 chained]
*job 24055687 • idun-01-04 • H100 • epoch 490/500 • chain 0/20 • 107KB log*


#### `24055689_24055689`  [🔗 chained]
*job 24055689 • idun-01-04 • H100 • epoch 176/500 • chain 0/20 • 51KB log*


#### `24055690_24055690`  [🔗 chained]
*job 24055690 • idun-07-04 • A100 • epoch 291/500 • chain 0/20 • 65KB log*


#### `24056427_24056427`  [✅ completed]
*job 24056427 • idun-07-09 • A100 • 3.77h training • epoch 500/500 • chain 0/20 • 146KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01576 • MS-SSIM 0.7086 • PSNR 27.10 dB • LPIPS 1.7616 • FID 273.99 • KID 0.3872 ± 0.0076 • CMMD 0.6156
  - **latest** ckpt (26 samples): MSE 0.01507 • MS-SSIM 0.7001 • PSNR 26.98 dB • LPIPS 1.7743 • FID 264.66 • KID 0.3702 ± 0.0086 • CMMD 0.6171

#### `24056428_24056428`  [✅ completed]
*job 24056428 • idun-07-10 • A100 • 12.66h training • epoch 500/500 • chain 0/20 • 117KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007365 • MS-SSIM 0.9459 • PSNR 32.78 dB • LPIPS 0.6684 • FID 132.46 • KID 0.1628 ± 0.0091 • CMMD 0.4628
  - **latest** ckpt (26 samples): MSE 0.005478 • MS-SSIM 0.9252 • PSNR 31.31 dB • LPIPS 0.4983 • FID 98.14 • KID 0.1178 ± 0.0069 • CMMD 0.2885

#### `24056484_24056484`  [🔗 chained]
*job 24056484 • idun-06-04 • A100 • epoch 131/500 • chain 7/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24056491_24056491`  [🔗 chained]
*job 24056491 • idun-06-04 • A100 • epoch 687/2000 • chain 1/20 • 102KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24056742_24056742`  [✅ completed]
*job 24056742 • idun-07-04 • A100 • 8.55h training • epoch 500/500 • chain 1/20 • 45KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_pixel_seg_improved_20260218-183124/checkpoint_latest.pt`

#### `24056743_24056743`  [🔗 chained]
*job 24056743 • idun-07-08 • A100 • epoch 281/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24056744_24056744`  [✅ completed]
*job 24056744 • idun-07-08 • A100 • 0.39h training • epoch 500/500 • chain 1/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_pixel_seg_20260218-183155/checkpoint_latest.pt`

#### `24060136_24060136`  [🔗 chained]
*job 24060136 • idun-06-05 • A100 • epoch 149/500 • chain 8/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24060137_24060137`  [🔗 chained]
*job 24060137 • idun-07-10 • A100 • epoch 997/2000 • chain 2/20 • 93KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24060138_24060138`  [🔗 chained]
*job 24060138 • idun-06-05 • A100 • epoch 1/500 • chain 0/20 • 4KB log*


#### `24060139_24060139`  [🔗 chained]
*job 24060139 • idun-06-06 • A100 • epoch 25/500 • chain 0/20 • 14KB log*


#### `24060169_24060169`  [❌ crashed]
*job 24060169 • idun-07-08 • A100 • epoch 385/500 • chain 2/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 690, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/base_trainer.py", line 794, in train
    self._handle_checkpoints(epoch, merged_metrics)
  Fil
```

#### `24060846_24060846`  [🔗 chained]
*job 24060846 • idun-07-10 • A100 • epoch 1175/2000 • chain 3/20 • 86KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24060847_24060847`  [🔗 chained]
*job 24060847 • idun-06-06 • A100 • epoch 166/500 • chain 9/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24060855_24060855`  [🔗 chained]
*job 24060855 • idun-07-08 • A100 • epoch 1/500 • chain 1/20 • 4KB log*


#### `24060856_24060856`  [🔗 chained]
*job 24060856 • idun-07-08 • A100 • epoch 25/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24061044_24061044`  [🔗 chained]
*job 24061044 • idun-06-01 • A100 • epoch 426/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24061288_24061288`  [🔗 chained]
*job 24061288 • idun-06-01 • A100 • epoch 163/500 • chain 0/20 • 57KB log*


#### `24061290_24061290`  [🔗 chained]
*job 24061290 • idun-01-03 • H100 • epoch 162/500 • chain 0/20 • 53KB log*


#### `24061291_24061291`  [✅ completed]
*job 24061291 • idun-06-04 • A100 • 10.58h training • epoch 500/500 • chain 0/20 • 144KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003976 • MS-SSIM 0.9285 • PSNR 31.00 dB • LPIPS 0.7093 • FID 161.63 • KID 0.2054 ± 0.0134 • CMMD 0.3920
  - **latest** ckpt (26 samples): MSE 0.005496 • MS-SSIM 0.9285 • PSNR 31.13 dB • LPIPS 0.7534 • FID 167.24 • KID 0.2119 ± 0.0118 • CMMD 0.3988

#### `24061292_24061292`  [🔗 chained]
*job 24061292 • idun-06-03 • A100 • epoch 404/500 • chain 0/20 • 120KB log*


#### `24061494_24061494`  [🔗 chained]
*job 24061494 • idun-07-09 • A100 • epoch 159/500 • chain 0/20 • 52KB log*


#### `24061806_24061806`  [🔗 chained]
*job 24061806 • idun-07-08 • A100 • epoch 1488/2000 • chain 4/20 • 66KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24061807_24061807`  [🔗 chained]
*job 24061807 • idun-06-03 • A100 • epoch 183/500 • chain 10/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`

#### `24061811_24061811`  [🔗 chained]
*job 24061811 • idun-06-03 • A100 • epoch 2/500 • chain 2/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_0_hdit_s_p2_256_20260219-205831/checkpoint_latest.pt`

#### `24061813_24061813`  [🔗 chained]
*job 24061813 • idun-06-01 • A100 • epoch 50/500 • chain 2/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24061859_24061859`  [✅ completed]
*job 24061859 • idun-06-04 • A100 • 7.87h training • epoch 500/500 • chain 4/20 • 27KB log*

**Final test metrics:**
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt`

#### `24061920_24061920`  [🔗 chained]
*job 24061920 • idun-08-01 • H100 • epoch 407/500 • chain 1/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_l_256_patch8_20260220-131107/checkpoint_latest.pt`

#### `24061922_24061922`  [🔗 chained]
*job 24061922 • idun-08-01 • H100 • epoch 322/500 • chain 1/20 • 43KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24061931_24061931`  [✅ completed]
*job 24061931 • idun-08-01 • H100 • 1.94h training • epoch 500/500 • chain 1/20 • 27KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.005235 • MS-SSIM 0.9316 • PSNR 31.53 dB • LPIPS 1.3710 • FID 154.84 • KID 0.1896 ± 0.0116 • CMMD 0.3878
  - **latest** ckpt (26 samples): MSE 0.01005 • MS-SSIM 0.9220 • PSNR 30.64 dB • LPIPS 0.5395 • FID 152.29 • KID 0.1937 ± 0.0107 • CMMD 0.3421
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_3_hdit_xl_p8_256_20260220-141245/checkpoint_latest.pt`

#### `24061961_24061961`  [🔗 chained]
*job 24061961 • idun-06-03 • A100 • epoch 342/500 • chain 1/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp18_uvit_l_p8_256_20260220-170256/checkpoint_latest.pt`

#### `24062040_24062040`  [🔗 chained]
*job 24062040 • idun-07-10 • A100 • epoch 1798/2000 • chain 5/20 • 60KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24062070_24062070`  [❌ crashed]
*job 24062070 • idun-06-01 • A100 • 11.36h training • epoch 199/500 • chain 11/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_s_256_patch4_20260215-023826/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
       ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'PixelSpace' object has no attribute 'needs_decode'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[W222 00:43:50.819646877 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA
```

#### `24062071_24062071`  [❌ crashed]
*job 24062071 • idun-06-01 • A100 • 3.2h training • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_0_hdit_s_p2_256_20260219-205831/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W221 16:39:03.520194663 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	199m51.125s
user	147m16.880s
sys	51m7.182s
```

#### `24062089_24062089`  [🔗 chained]
*job 24062089 • idun-07-08 • A100 • epoch 73/500 • chain 3/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24062511_24062511`  [🔗 chained]
*job 24062511 • idun-08-01 • H100 • epoch 458/500 • chain 0/20 • 87KB log*


#### `24062564_24062564`  [🔗 chained]
*job 24062564 • idun-06-01 • A100 • epoch 351/500 • chain 0/20 • 94KB log*


#### `24062597_24062597`  [❌ crashed]
*job 24062597 • idun-06-03 • A100 • 8.65h training • epoch 500/500 • chain 0/20 • 101KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 18.29 • MS-SSIM 0.9987 • PSNR 50.68 dB • LPIPS 0.0052
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 08:28:53.147816293 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	525m30.097s
user	749m21.804s
sys	91m41.577s
```

#### `24062598_24062598`  [❌ crashed]
*job 24062598 • idun-06-03 • A100 • 9.85h training • epoch 500/500 • chain 0/20 • 94KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.642 • MS-SSIM 0.9997 • PSNR 55.49 dB • LPIPS 0.0014
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 09:39:54.723334518 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	596m32.186s
user	809m43.339s
sys	86m35.424s
```

#### `24062820_24062820`  [✅ completed]
*job 24062820 • idun-06-01 • A100 • 6.91h training • epoch 500/500 • chain 2/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00473 • MS-SSIM 0.9466 • PSNR 32.39 dB • LPIPS 0.4509 • FID 132.38 • KID 0.1440 ± 0.0095 • CMMD 0.3918
  - **latest** ckpt (26 samples): MSE 0.003225 • MS-SSIM 0.9466 • PSNR 32.17 dB • LPIPS 0.4568 • FID 131.99 • KID 0.1460 ± 0.0081 • CMMD 0.3862
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_l_256_patch8_20260220-131107/checkpoint_latest.pt`

#### `24062821_24062821`  [🔗 chained]
*job 24062821 • idun-07-08 • A100 • epoch 411/500 • chain 2/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24062848_24062848`  [🔗 chained]
*job 24062848 • idun-07-06 • A100 • epoch 436/500 • chain 0/20 • 75KB log*


#### `24062849_24062849`  [🔗 chained]
*job 24062849 • idun-09-18 • A100 • epoch 384/500 • chain 0/20 • 76KB log*


#### `24062850_24062850`  [🔗 chained]
*job 24062850 • idun-09-18 • A100 • epoch 375/500 • chain 0/20 • 76KB log*


#### `24062935_24062935`  [✅ completed]
*job 24062935 • idun-06-01 • A100 • 10.2h training • epoch 500/500 • chain 2/20 • 41KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.008652 • MS-SSIM 0.9488 • PSNR 32.94 dB • LPIPS 0.9663 • FID 149.15 • KID 0.1802 ± 0.0114 • CMMD 0.3810
  - **latest** ckpt (26 samples): MSE 0.003476 • MS-SSIM 0.9502 • PSNR 32.67 dB • LPIPS 0.7485 • FID 147.86 • KID 0.1877 ± 0.0125 • CMMD 0.3879
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp18_uvit_l_p8_256_20260220-170256/checkpoint_latest.pt`

#### `24062965_24062965`  [✅ completed]
*job 24062965 • idun-06-05 • A100 • 6.49h training • epoch 2000/2000 • chain 6/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004644 • MS-SSIM 0.9529 • PSNR 33.03 dB • LPIPS 0.5121 • FID 157.82 • KID 0.2049 ± 0.0107 • CMMD 0.3550
  - **latest** ckpt (26 samples): MSE 0.00343 • MS-SSIM 0.9479 • PSNR 32.27 dB • LPIPS 0.4261 • FID 129.36 • KID 0.1599 ± 0.0091 • CMMD 0.2721
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_2000_20260218-165041/checkpoint_latest.pt`

#### `24062966_24062966`  [🔗 chained]
*job 24062966 • idun-07-10 • A100 • epoch 94/500 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24062979_24062979`  [✅ completed]
*job 24062979 • idun-06-03 • A100 • 1.53h training • epoch 500/500 • chain 1/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007356 • MS-SSIM 0.9663 • PSNR 34.59 dB • LPIPS 0.5287 • FID 84.13 • KID 0.0849 ± 0.0169 • CMMD 0.3426
  - **latest** ckpt (26 samples): MSE 0.02154 • MS-SSIM 0.9387 • PSNR 32.16 dB • LPIPS 0.4058 • FID 90.56 • KID 0.0990 ± 0.0173 • CMMD 0.3006
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_20260221-172156/checkpoint_latest.pt`

#### `24063022_24063022`  [✅ completed]
*job 24063022 • idun-07-05 • A100 • 1.77h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.377 • MS-SSIM 0.9278 • PSNR 33.50 dB • LPIPS 0.5723 • FID 112.82 • KID 0.1046 ± 0.0091 • CMMD 0.4834
  - **latest** ckpt (26 samples): MSE 0.5867 • MS-SSIM 0.9093 • PSNR 32.33 dB • LPIPS 0.5931 • FID 98.08 • KID 0.0847 ± 0.0069 • CMMD 0.4365
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_20260221-215953/checkpoint_latest.pt`

#### `24063070_24063070`  [❌ crashed]
*job 24063070 • idun-06-02 • A100 • 5.06h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.936 • MS-SSIM 0.9998 • PSNR 57.96 dB • LPIPS 0.0010
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260221-224417/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W222 18:24:08.692785036 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	312m1.101s
user	277m36.075s
sys	109m44.250s
```

#### `24063071_24063071`  [✅ completed]
*job 24063071 • idun-09-18 • A100 • 3.59h training • epoch 500/500 • chain 1/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.6159 • MS-SSIM 0.8946 • PSNR 34.96 dB • LPIPS 1.4661 • FID 305.22 • KID 0.4231 ± 0.0133 • CMMD 0.6040
  - **latest** ckpt (26 samples): MSE 0.6088 • MS-SSIM 0.9065 • PSNR 35.86 dB • LPIPS 1.4473 • FID 302.24 • KID 0.4122 ± 0.0114 • CMMD 0.6139
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_3_20260221-224417/checkpoint_latest.pt`

#### `24063079_24063079`  [✅ completed]
*job 24063079 • idun-09-18 • A100 • 4.05h training • epoch 500/500 • chain 1/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004118 • MS-SSIM 0.8509 • PSNR 30.23 dB • LPIPS 1.2873 • FID 149.50 • KID 0.1519 ± 0.0122 • CMMD 0.5199
  - **latest** ckpt (26 samples): MSE 0.003497 • MS-SSIM 0.8730 • PSNR 31.02 dB • LPIPS 1.2673 • FID 146.14 • KID 0.1463 ± 0.0126 • CMMD 0.5326
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260221-225423/checkpoint_latest.pt`

#### `24063241_24063241`  [🔗 chained]
*job 24063241 • idun-07-10 • A100 • epoch 499/500 • chain 3/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24063375_24063375`  [🔗 chained]
*job 24063375 • idun-06-01 • A100 • epoch 119/500 • chain 5/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24063576_24063576`  [🔗 chained]
*job 24063576 • idun-06-02 • A100 • epoch 106/500 • chain 0/20 • 28KB log*


#### `24063579_24063579`  [❌ crashed]
*job 24063579 • idun-06-01 • A100 • 8.04h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2911 • MS-SSIM 0.9970 • PSNR 46.86 dB • LPIPS 0.0183
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W223 07:11:28.639423918 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	487m4.098s
user	605m26.713s
sys	151m24.145s
```

#### `24063615_24063615`  [🔗 chained]
*job 24063615 • idun-07-06 • A100 • epoch 434/500 • chain 0/20 • 85KB log*


#### `24063616_24063616`  [🔗 chained]
*job 24063616 • idun-09-16 • A100 • epoch 381/500 • chain 0/20 • 66KB log*


#### `24063622_24063622`  [🔗 chained]
*job 24063622 • idun-06-03 • A100 • epoch 349/500 • chain 0/20 • 79KB log*


#### `24063623_24063623`  [🔗 chained]
*job 24063623 • idun-06-01 • A100 • epoch 121/500 • chain 0/20 • 33KB log*


#### `24063648_24063648`  [❌ crashed]
*job 24063648 • idun-01-04 • H100 • 5.8h training • epoch 500/500 • chain 0/20 • 87KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.377 • MS-SSIM 0.9954 • PSNR 43.53 dB • LPIPS 0.0190
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
[W223 10:19:21.504942614 AllocatorConfig.cpp:28] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())

real	353m44.036s
user	439m55.989s
sys	89m39.309s
```

#### `24063682_24063682`  [✅ completed]
*job 24063682 • idun-01-04 • H100 • 0.27h training • epoch 500/500 • chain 4/20 • 9KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003828 • MS-SSIM 0.9372 • PSNR 31.91 dB • LPIPS 0.5120 • FID 152.97 • KID 0.1881 ± 0.0139 • CMMD 0.3809
  - **latest** ckpt (26 samples): MSE 0.009043 • MS-SSIM 0.9507 • PSNR 33.08 dB • LPIPS 0.4149 • FID 157.10 • KID 0.1974 ± 0.0134 • CMMD 0.3905
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_xl_256_patch8_20260220-131913/checkpoint_latest.pt`

#### `24063686_24063686`  [🔗 chained]
*job 24063686 • idun-01-04 • H100 • epoch 133/500 • chain 6/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24063887_24063887`  [🔗 chained]
*job 24063887 • idun-06-03 • A100 • epoch 433/500 • chain 1/20 • 58KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260222-225625/checkpoint_latest.pt`

#### `24063940_24063940`  [✅ completed]
*job 24063940 • idun-07-07 • A100 • 1.8h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004617 • MS-SSIM 0.8374 • PSNR 30.01 dB • LPIPS 1.2790 • FID 256.54 • KID 0.3115 ± 0.0133 • CMMD 0.5483
  - **latest** ckpt (26 samples): MSE 0.0045 • MS-SSIM 0.8429 • PSNR 29.95 dB • LPIPS 1.2748 • FID 243.49 • KID 0.2878 ± 0.0158 • CMMD 0.5594
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12_2_20260222-233635/checkpoint_latest.pt`

#### `24063965_24063965`  [✅ completed]
*job 24063965 • idun-09-16 • A100 • 3.73h training • epoch 500/500 • chain 1/20 • 28KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.5198 • MS-SSIM 0.8889 • PSNR 31.44 dB • LPIPS 0.7354 • FID 152.36 • KID 0.1398 ± 0.0127 • CMMD 0.5845
  - **latest** ckpt (26 samples): MSE 0.4083 • MS-SSIM 0.9164 • PSNR 32.83 dB • LPIPS 0.5358 • FID 154.74 • KID 0.1432 ± 0.0128 • CMMD 0.5774
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_20260222-235610/checkpoint_latest.pt`

#### `24063991_24063991`  [✅ completed]
*job 24063991 • idun-06-03 • A100 • 5.46h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.07605 • MS-SSIM 0.9849 • PSNR 38.78 dB • LPIPS 0.2314 • FID 113.97 • KID 0.0997 ± 0.0059 • CMMD 0.5071
  - **latest** ckpt (26 samples): MSE 0.1444 • MS-SSIM 0.9730 • PSNR 35.92 dB • LPIPS 0.2615 • FID 143.60 • KID 0.1613 ± 0.0069 • CMMD 0.4406
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_20260223-002347/checkpoint_latest.pt`

#### `24064017_24064017`  [🔗 chained]
*job 24064017 • idun-01-03 • H100 • epoch 297/500 • chain 1/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24065812_24065812`  [🔗 chained]
*job 24065812 • idun-06-03 • A100 • epoch 157/500 • chain 7/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24067712_24067712`  [🔗 chained]
*job 24067712 • idun-06-05 • A100 • epoch 112/500 • chain 0/20 • 85KB log*


#### `24067738_24067738`  [✅ completed]
*job 24067738 • idun-06-02 • A100 • 2.27h training • epoch 500/500 • chain 2/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2609 • MS-SSIM 0.9968 • PSNR 46.45 dB • LPIPS 0.0203 • FID 232.16 • KID 0.2637 ± 0.0267 • CMMD 0.5815
  - **latest** ckpt (26 samples): MSE 0.8212 • MS-SSIM 0.9923 • PSNR 44.12 dB • LPIPS 0.0453 • FID 182.09 • KID 0.2008 ± 0.0301 • CMMD 0.4864
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_20260222-225625/checkpoint_latest.pt`

#### `24067773_24067773`  [🔗 chained]
*job 24067773 • idun-01-03 • H100 • epoch 473/500 • chain 2/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24067802_24067802`  [🔗 chained]
*job 24067802 • idun-06-01 • A100 • epoch 190/500 • chain 0/20 • 36KB log*


#### `24067803_24067803`  [🔗 chained]
*job 24067803 • idun-06-02 • A100 • epoch 190/500 • chain 0/20 • 35KB log*


#### `24067804_24067804`  [🔗 chained]
*job 24067804 • idun-06-02 • A100 • epoch 199/500 • chain 0/20 • 39KB log*


#### `24067805_24067805`  [🔗 chained]
*job 24067805 • idun-06-02 • A100 • epoch 289/500 • chain 0/20 • 51KB log*


#### `24067806_24067806`  [🔗 chained]
*job 24067806 • idun-01-03 • H100 • epoch 257/500 • chain 0/20 • 45KB log*


#### `24067807_24067807`  [🔗 chained]
*job 24067807 • idun-07-08 • A100 • epoch 174/500 • chain 0/20 • 34KB log*


#### `24067808_24067808`  [⚠️ truncated]
*job 24067808 • idun-07-09 • A100 • chain 0/20 • 1KB log*


#### `24070764_24070764`  [🔗 chained]
*job 24070764 • idun-07-08 • A100 • epoch 178/500 • chain 8/20 • 9KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_1_hdit_b_p4_256_20260219-210002/checkpoint_latest.pt`

#### `24070899_24070899`  [🔗 chained]
*job 24070899 • idun-06-01 • A100 • epoch 224/500 • chain 1/20 • 71KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_1_pixel_seg_aux_bin_20260224-000544/checkpoint_latest.pt`

#### `24070923_24070923`  [🔗 chained]
*job 24070923 • idun-06-02 • A100 • epoch 190/500 • chain 0/20 • 36KB log*


#### `24070925_24070925`  [✅ completed]
*job 24070925 • idun-06-02 • A100 • 2.72h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.07344 • MS-SSIM 0.9920 • PSNR 41.04 dB • LPIPS 0.1385 • FID 295.98 • KID 0.4118 ± 0.0165 • CMMD 0.6002
  - **latest** ckpt (26 samples): MSE 0.06805 • MS-SSIM 0.9903 • PSNR 39.89 dB • LPIPS 0.1525 • FID 306.39 • KID 0.4247 ± 0.0167 • CMMD 0.6081
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_1_20260223-004121/checkpoint_latest.pt`

#### `24071122_24071122`  [🔗 chained]
*job 24071122 • idun-07-09 • A100 • epoch 360/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_0_20260224-015924/checkpoint_latest.pt`

#### `24071123_24071123`  [🔗 chained]
*job 24071123 • idun-01-03 • H100 • epoch 465/500 • chain 1/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_paper_20260224-015924/checkpoint_latest.pt`

#### `24071215_24071215`  [❌ crashed]
*job 24071215 • idun-06-02 • A100 • 0.01h training • chain 0/20 • 4KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	2m23.684s
user	1m31.990s
sys	0m31.447s
```

#### `24071216_24071216`  [✅ completed]
*job 24071216 • idun-06-02 • A100 • 10.59h training • epoch 500/500 • chain 0/20 • 122KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.0756 • MS-SSIM 0.9939 • PSNR 41.11 dB • LPIPS 0.1051 • FID 117.07 • KID 0.1017 ± 0.0049 • CMMD 0.5555
  - **latest** ckpt (26 samples): MSE 0.1331 • MS-SSIM 0.9800 • PSNR 38.07 dB • LPIPS 0.1776 • FID 95.12 • KID 0.0724 ± 0.0054 • CMMD 0.4502

#### `24071218_24071218`  [🔗 chained]
*job 24071218 • idun-06-05 • A100 • epoch 395/500 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_1_20260224-022448/checkpoint_latest.pt`

#### `24071219_24071219`  [✅ completed]
*job 24071219 • idun-07-08 • A100 • 10.67h training • epoch 500/500 • chain 1/20 • 42KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3851 • MS-SSIM 0.9582 • PSNR 35.94 dB • LPIPS 0.2392 • FID 68.92 • KID 0.0453 ± 0.0052 • CMMD 0.2383
  - **latest** ckpt (26 samples): MSE 0.2968 • MS-SSIM 0.9651 • PSNR 36.67 dB • LPIPS 0.1924 • FID 67.32 • KID 0.0437 ± 0.0051 • CMMD 0.2351
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_2_20260224-022548/checkpoint_latest.pt`

#### `24071365_24071365`  [🔗 chained]
*job 24071365 • idun-07-08 • A100 • epoch 424/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_3_20260224-034637/checkpoint_latest.pt`

#### `24071590_24071590`  [🔗 chained]
*job 24071590 • idun-06-05 • A100 • epoch 333/500 • chain 0/20 • 248KB log*


#### `24072508_24072508`  [✅ completed]
*job 24072508 • idun-07-08 • A100 • 8.76h training • epoch 500/500 • chain 0/20 • 90KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.518 • MS-SSIM 0.9918 • PSNR 34.89 dB • LPIPS 0.0665 • FID 75.13 • KID 0.0712 ± 0.0129 • CMMD 0.3318
  - **latest** ckpt (26 samples): MSE 1.65 • MS-SSIM 0.9926 • PSNR 35.68 dB • LPIPS 0.0623 • FID 59.94 • KID 0.0506 ± 0.0098 • CMMD 0.3079

#### `24072517_24072517`  [🔗 chained]
*job 24072517 • idun-09-16 • A100 • epoch 245/500 • chain 0/20 • 188KB log*


#### `24075917_24075917`  [🔗 chained]
*job 24075917 • idun-06-02 • A100 • epoch 372/500 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_4_20260224-092339/checkpoint_latest.pt`

#### `24076104_24076104`  [🔗 chained]
*job 24076104 • idun-06-01 • A100 • epoch 49/500 • chain 0/20 • 20KB log*


#### `24076153_24076153`  [🔗 chained]
*job 24076153 • idun-07-08 • A100 • epoch 327/500 • chain 2/20 • 57KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_1_pixel_seg_aux_bin_20260224-000544/checkpoint_latest.pt`

#### `24076158_24076158`  [🔗 chained]
*job 24076158 • idun-06-05 • A100 • epoch 378/500 • chain 1/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_5_20260224-142309/checkpoint_latest.pt`

#### `24076159_24076159`  [✅ completed]
*job 24076159 • idun-07-09 • A100 • 9.7h training • epoch 500/500 • chain 2/20 • 32KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3549 • MS-SSIM 0.9663 • PSNR 36.29 dB • LPIPS 0.2495 • FID 128.82 • KID 0.1176 ± 0.0095 • CMMD 0.4297
  - **latest** ckpt (26 samples): MSE 0.2181 • MS-SSIM 0.9787 • PSNR 38.65 dB • LPIPS 0.1673 • FID 131.75 • KID 0.1200 ± 0.0138 • CMMD 0.4457
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_0_20260224-015924/checkpoint_latest.pt`

#### `24076160_24076160`  [✅ completed]
*job 24076160 • idun-06-02 • A100 • 2.26h training • epoch 500/500 • chain 2/20 • 14KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00351 • MS-SSIM 0.9264 • PSNR 31.45 dB • LPIPS 0.6003 • FID 87.01 • KID 0.0544 ± 0.0052 • CMMD 0.3535
  - **latest** ckpt (26 samples): MSE 0.003405 • MS-SSIM 0.9307 • PSNR 31.99 dB • LPIPS 0.4647 • FID 81.22 • KID 0.0446 ± 0.0047 • CMMD 0.2846
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_paper_20260224-015924/checkpoint_latest.pt`

#### `24076418_24076418`  [✅ completed]
*job 24076418 • idun-06-01 • A100 • 6.35h training • epoch 500/500 • chain 2/20 • 29KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.2198 • MS-SSIM 0.9956 • PSNR 44.83 dB • LPIPS 0.0690 • FID 235.69 • KID 0.2532 ± 0.0148 • CMMD 0.5038
  - **latest** ckpt (26 samples): MSE 0.2166 • MS-SSIM 0.9960 • PSNR 46.04 dB • LPIPS 0.0603 • FID 235.27 • KID 0.2484 ± 0.0164 • CMMD 0.4904
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_1_20260224-022448/checkpoint_latest.pt`

#### `24076724_24076724`  [✅ completed]
*job 24076724 • idun-07-09 • A100 • 5.37h training • epoch 500/500 • chain 2/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3201 • MS-SSIM 0.9623 • PSNR 35.46 dB • LPIPS 0.2349 • FID 107.45 • KID 0.0788 ± 0.0065 • CMMD 0.3993
  - **latest** ckpt (26 samples): MSE 0.2795 • MS-SSIM 0.9703 • PSNR 37.63 dB • LPIPS 0.2013 • FID 123.96 • KID 0.0999 ± 0.0085 • CMMD 0.3737
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_3_20260224-034637/checkpoint_latest.pt`

#### `24076868_24076868`  [⚠️ truncated]
*job 24076868 • idun-06-05 • A100 • epoch 438/500 • chain 1/20 • 79KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_pixel_seg_aux_bin_20260225-005914/checkpoint_latest.pt`

#### `24077301_24077301`  [⚠️ truncated]
*job 24077301 • idun-06-05 • A100 • epoch 306/500 • chain 1/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2d_pixel_seg_aux_bin_20260225-040703/checkpoint_latest.pt`

#### `24077858_24077858`  [⚠️ truncated]
*job 24077858 • idun-06-01 • A100 • chain 0/20 • 1KB log*


#### `24081617_24081617`  [⚠️ truncated]
*job 24081617 • idun-06-01 • A100 • epoch 55/500 • chain 1/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp17_4_hdit_s_p4_256_20260225-111541/checkpoint_latest.pt`

#### `24082354_24082354`  [✅ completed]
*job 24082354 • idun-07-09 • A100 • 10.57h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.341 • MS-SSIM 0.9922 • PSNR 44.11 dB • LPIPS 0.1116 • FID 176.51 • KID 0.1888 ± 0.0271 • CMMD 0.3985
  - **latest** ckpt (26 samples): MSE 1.618 • MS-SSIM 0.9948 • PSNR 46.36 dB • LPIPS 0.0723 • FID 164.34 • KID 0.1784 ± 0.0280 • CMMD 0.3946

#### `24082355_24082355`  [✅ completed]
*job 24082355 • idun-06-05 • A100 • 9.68h training • epoch 500/500 • chain 0/20 • 92KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.1575 • MS-SSIM 0.9951 • PSNR 47.77 dB • LPIPS 0.0237 • FID 99.68 • KID 0.1043 ± 0.0163 • CMMD 0.2696
  - **latest** ckpt (26 samples): MSE 0.1553 • MS-SSIM 0.9952 • PSNR 49.70 dB • LPIPS 0.0239 • FID 94.77 • KID 0.0935 ± 0.0143 • CMMD 0.2922

#### `24082412_24082412`  [🔗 chained]
*job 24082412 • idun-06-01 • A100 • epoch 172/500 • chain 0/20 • 34KB log*


#### `24082413_24082413`  [✅ completed]
*job 24082413 • idun-06-05 • A100 • 10.59h training • epoch 500/500 • chain 0/20 • 85KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00288 • MS-SSIM 0.9420 • PSNR 32.85 dB • LPIPS 0.7587 • FID 76.43 • KID 0.0919 ± 0.0099 • CMMD 0.2239
  - **latest** ckpt (26 samples): MSE 0.004557 • MS-SSIM 0.9171 • PSNR 30.69 dB • LPIPS 0.5204 • FID 36.85 • KID 0.0272 ± 0.0038 • CMMD 0.1809

#### `24082489_24082489`  [✅ completed]
*job 24082489 • idun-01-03 • H100 • 5.36h training • epoch 500/500 • chain 2/20 • 37KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.05143 • MS-SSIM 0.9517 • PSNR 35.14 dB • LPIPS 0.3715 • FID 83.03 • KID 0.0615 ± 0.0057 • CMMD 0.2622
  - **latest** ckpt (26 samples): MSE 0.04286 • MS-SSIM 0.9619 • PSNR 37.02 dB • LPIPS 0.2952 • FID 92.77 • KID 0.0722 ± 0.0048 • CMMD 0.2548
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_5_20260224-142309/checkpoint_latest.pt`

#### `24082612_24082612`  [🔗 chained]
*job 24082612 • idun-06-05 • A100 • epoch 333/500 • chain 0/20 • 228KB log*


#### `24085100_24085100`  [🔗 chained]
*job 24085100 • idun-06-05 • A100 • epoch 169/500 • chain 0/20 • 41KB log*


#### `24085101_24085101`  [✅ completed]
*job 24085101 • idun-06-07 • A100 • 10.9h training • epoch 500/500 • chain 0/20 • 108KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006735 • MS-SSIM 0.9265 • PSNR 31.31 dB • LPIPS 0.4632 • FID 54.48 • KID 0.0541 ± 0.0064 • CMMD 0.1649
  - **latest** ckpt (26 samples): MSE 0.007308 • MS-SSIM 0.9148 • PSNR 30.60 dB • LPIPS 0.4801 • FID 54.94 • KID 0.0557 ± 0.0061 • CMMD 0.1632

#### `24085346_24085346`  [🔗 chained]
*job 24085346 • idun-06-01 • A100 • epoch 343/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1g_1_pixel_bravo_pseudo_huber_20260226-034319/checkpoint_latest.pt`

#### `24089369_24089369`  [❌ crashed]
*job 24089369 • idun-06-01 • A100 • 5.95h training • epoch 500/500 • chain 1/20 • 99KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp2e_pixel_seg_multilevel_aux_20260226-130811/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	361m26.664s
user	384m7.568s
sys	108m7.598s
```

#### `24089584_24089584`  [🔗 chained]
*job 24089584 • idun-06-05 • A100 • epoch 336/500 • chain 1/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1h_1_pixel_bravo_lpips_huber_20260226-202457/checkpoint_latest.pt`

#### `24089829_24089829`  [✅ completed]
*job 24089829 • idun-06-01 • A100 • 10.99h training • epoch 500/500 • chain 2/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002612 • MS-SSIM 0.9526 • PSNR 32.69 dB • LPIPS 0.5583 • FID 84.71 • KID 0.0898 ± 0.0092 • CMMD 0.1824
  - **latest** ckpt (26 samples): MSE 0.004478 • MS-SSIM 0.9384 • PSNR 31.62 dB • LPIPS 0.4834 • FID 68.30 • KID 0.0688 ± 0.0035 • CMMD 0.1143
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1g_1_pixel_bravo_pseudo_huber_20260226-034319/checkpoint_latest.pt`

#### `24093199_24093199`  [✅ completed]
*job 24093199 • idun-06-01 • A100 • 11.6h training • epoch 500/500 • chain 2/20 • 43KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003908 • MS-SSIM 0.9571 • PSNR 33.05 dB • LPIPS 0.4572 • FID 71.87 • KID 0.0755 ± 0.0064 • CMMD 0.1446
  - **latest** ckpt (26 samples): MSE 0.004614 • MS-SSIM 0.9480 • PSNR 32.18 dB • LPIPS 0.4736 • FID 63.96 • KID 0.0636 ± 0.0099 • CMMD 0.1391
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1h_1_pixel_bravo_lpips_huber_20260226-202457/checkpoint_latest.pt`

#### `24096039_24096039`  [🔗 chained]
*job 24096039 • idun-08-01 • H100 • epoch 141/500 • chain 0/20 • 34KB log*


#### `24096042_24096042`  [🔗 chained]
*job 24096042 • idun-08-01 • H100 • epoch 173/500 • chain 0/20 • 43KB log*


#### `24096044_24096044`  [🔗 chained]
*job 24096044 • idun-01-04 • H100 • epoch 263/500 • chain 0/20 • 48KB log*


#### `24096047_24096047`  [🔗 chained]
*job 24096047 • idun-01-04 • H100 • epoch 127/1000 • chain 0/40 • 29KB log*


#### `24096390_24096390`  [🔗 chained]
*job 24096390 • idun-07-08 • A100 • epoch 226/500 • chain 1/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`

#### `24096391_24096391`  [🔗 chained]
*job 24096391 • idun-01-04 • H100 • epoch 345/500 • chain 1/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24096393_24096393`  [✅ completed]
*job 24096393 • idun-01-04 • H100 • 10.85h training • epoch 500/500 • chain 1/20 • 45KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00313 • MS-SSIM 0.9567 • PSNR 33.10 dB • LPIPS 0.6425 • FID 69.23 • KID 0.0621 ± 0.0089 • CMMD 0.2253
  - **latest** ckpt (26 samples): MSE 0.003924 • MS-SSIM 0.9598 • PSNR 33.34 dB • LPIPS 0.5712 • FID 73.83 • KID 0.0689 ± 0.0090 • CMMD 0.2072
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1j_1_pixel_bravo_grad_accum_20260301-015744/checkpoint_latest.pt`

#### `24096397_24096397`  [🔗 chained]
*job 24096397 • idun-01-03 • H100 • epoch 304/1000 • chain 1/40 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24097015_24097015`  [🔗 chained]
*job 24097015 • idun-06-01 • A100 • epoch 121/500 • chain 0/20 • 29KB log*


#### `24097351_24097351`  [🔗 chained]
*job 24097351 • idun-07-10 • A100 • epoch 455/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24097352_24097352`  [🔗 chained]
*job 24097352 • idun-07-10 • A100 • epoch 314/500 • chain 2/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`

#### `24097361_24097361`  [🔗 chained]
*job 24097361 • idun-07-10 • A100 • epoch 417/1000 • chain 2/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24097410_24097410`  [🔗 chained]
*job 24097410 • idun-06-01 • A100 • epoch 241/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`

#### `24099367_24099367`  [✅ completed]
*job 24099367 • idun-07-10 • A100 • 4.86h training • epoch 500/500 • chain 3/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00299 • MS-SSIM 0.9592 • PSNR 33.46 dB • LPIPS 0.8082 • FID 139.99 • KID 0.1811 ± 0.0143 • CMMD 0.2642
  - **latest** ckpt (26 samples): MSE 0.002944 • MS-SSIM 0.9523 • PSNR 33.08 dB • LPIPS 0.9080 • FID 142.26 • KID 0.1851 ± 0.0154 • CMMD 0.2886
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt`

#### `24099368_24099368`  [💥 oom_killed]
*job 24099368 • idun-06-06 • A100 • 0.14h training • epoch 315/500 • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24099393_24099393`  [❌ crashed]
*job 24099393 • idun-08-01 • H100 • 9.02h training • epoch 551/1000 • chain 3/40 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	543m41.949s
user	537m23.007s
sys	174m52.959s
```

#### `24101988_24101988`  [❌ crashed]
*job 24101988 • idun-01-04 • H100 • 8.23h training • epoch 362/500 • chain 2/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	496m47.857s
user	484m37.372s
sys	166m12.672s
```

#### `24102847_24102847`  [🔗 chained]
*job 24102847 • idun-06-06 • A100 • epoch 109/500 • chain 0/20 • 37KB log*


#### `24102848_24102848`  [🔗 chained]
*job 24102848 • idun-01-04 • H100 • epoch 142/500 • chain 0/20 • 45KB log*


#### `24104590_24104590`  [💥 oom_killed]
*job 24104590 • idun-06-05 • A100 • 0.14h training • epoch 316/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24104596_24104596`  [✅ completed]
*job 24104596 • idun-01-04 • H100 • 9.33h training • epoch 500/500 • chain 0/20 • 33KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003122 • MS-SSIM 0.9579 • PSNR 33.24 dB • LPIPS 0.5616 • FID 91.46 • KID 0.0862 ± 0.0060 • CMMD 0.2347
  - **latest** ckpt (26 samples): MSE 0.003422 • MS-SSIM 0.9529 • PSNR 32.68 dB • LPIPS 0.5209 • FID 51.17 • KID 0.0332 ± 0.0055 • CMMD 0.1934
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt`

#### `24104601_24104601`  [❌ crashed]
*job 24104601 • idun-01-04 • H100 • 6.1h training • epoch 642/1000 • chain 0/40 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	368m13.195s
user	363m32.088s
sys	122m31.923s
```

#### `24105243_24105243`  [💥 oom_killed]
*job 24105243 • idun-06-06 • A100 • 0.15h training • epoch 317/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24105272_24105272`  [🔗 chained]
*job 24105272 • idun-07-09 • A100 • epoch 232/500 • chain 1/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24105273_24105273`  [❌ crashed]
*job 24105273 • idun-07-09 • A100 • 3.35h training • epoch 136/500 • chain 1/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	204m25.835s
user	181m0.160s
sys	66m59.130s
```

#### `24107156_24107156`  [🔗 chained]
*job 24107156 • idun-07-09 • A100 • epoch 753/1000 • chain 0/40 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24107157_24107157`  [💥 oom_killed]
*job 24107157 • idun-06-05 • A100 • 0.15h training • epoch 318/500 • chain 0/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_1_pixel_bravo_attn_l3_20260301-015714/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    _engine_run_backward(
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.25 GiB. GPU 0 has a total capacity of 79.25 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 74.47 GiB memory in use. Of the allocated memory 57.44 GiB is allocated by PyTorch, with 8.38 GiB allocated in private pools (e.g., CUDA Graphs), and 16.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for
```

#### `24107165_24107165`  [🔗 chained]
*job 24107165 • idun-06-03 • A100 • epoch 244/500 • chain 0/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24107260_24107260`  [❌ crashed]
*job 24107260 • idun-06-07 • A100 • 6.16h training • epoch 175/500 • chain 0/20 • 44KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 117, in add_event
    self._async_writer.write(event.SerializeToString())
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.py", line 171, in write
    self._check_worker_status()
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/tensorboard/summary/writer/event_file_writer.p
```

#### `24107261_24107261`  [❌ crashed]
*job 24107261 • idun-06-03 • A100 • 10.03h training • epoch 500/500 • chain 0/20 • 99KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.414 • MS-SSIM 0.9899 • PSNR 32.97 dB • LPIPS 0.0874 • FID 58.19 • KID 0.0499 ± 0.0144 • CMMD 0.1982
  - **latest** ckpt (26 samples): MSE 3.565 • MS-SSIM 0.9776 • PSNR 31.07 dB • LPIPS 0.1821 • FID 50.89 • KID 0.0413 ± 0.0102 • CMMD 0.1740
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/logging/__init__.py", line 1164, in emit
    self.flush()
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 851, in _train_3d
    trainer.train(
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/base_trainer.py", line 794, in train
    self._handle_checkpoints(epoch, merged_metrics)
  Fil
```

#### `24107262_24107262`  [✅ completed]
*job 24107262 • idun-06-05 • A100 • 6.56h training • epoch 500/500 • chain 0/20 • 118KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.894 • MS-SSIM 0.9833 • PSNR 31.18 dB • LPIPS 0.1372 • FID 134.56 • KID 0.1355 ± 0.0266 • CMMD 0.3688
  - **latest** ckpt (26 samples): MSE 3.571 • MS-SSIM 0.9749 • PSNR 30.19 dB • LPIPS 0.1784 • FID 83.90 • KID 0.0852 ± 0.0181 • CMMD 0.2561

#### `24107263_24107263`  [🔗 chained]
*job 24107263 • idun-06-03 • A100 • epoch 405/500 • chain 0/20 • 76KB log*


#### `24107264_24107264`  [🔗 chained]
*job 24107264 • idun-07-10 • A100 • epoch 149/500 • chain 0/20 • 41KB log*


#### `24107272_24107272`  [🔗 chained]
*job 24107272 • idun-07-09 • A100 • epoch 674/2000 • chain 0/40 • 120KB log*


#### `24107429_24107429`  [🔗 chained]
*job 24107429 • idun-01-03 • H100 • epoch 375/500 • chain 2/20 • 64KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24110352_24110352`  [🔗 chained]
*job 24110352 • idun-06-02 • A100 • epoch 875/1000 • chain 1/40 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24110741_24110741`  [🔗 chained]
*job 24110741 • idun-07-09 • A100 • epoch 277/500 • chain 1/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24111006_24111006`  [✅ completed]
*job 24111006 • idun-06-05 • A100 • 2.74h training • epoch 500/500 • chain 1/20 • 27KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.339 • MS-SSIM 0.9887 • PSNR 33.10 dB • LPIPS 0.0876 • FID 73.74 • KID 0.0679 ± 0.0094 • CMMD 0.3877
  - **latest** ckpt (26 samples): MSE 2.407 • MS-SSIM 0.9863 • PSNR 33.45 dB • LPIPS 0.1075 • FID 48.99 • KID 0.0376 ± 0.0075 • CMMD 0.2661
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_1_ldm_4x_dit_b_20260304-100512/checkpoint_latest.pt`

#### `24111008_24111008`  [🔗 chained]
*job 24111008 • idun-07-10 • A100 • epoch 297/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24111018_24111018`  [🔗 chained]
*job 24111018 • idun-07-09 • A100 • epoch 1348/2000 • chain 1/40 • 112KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_3_ldm_4x_dit_s_long_20260304-102347/checkpoint_latest.pt`

#### `24111052_24111052`  [🔗 chained]
*job 24111052 • idun-07-09 • A100 • epoch 398/500 • chain 3/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24111102_24111102`  [🔗 chained]
*job 24111102 • idun-07-09 • A100 • epoch 988/1000 • chain 2/40 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24111271_24111271`  [🔗 chained]
*job 24111271 • idun-07-08 • A100 • epoch 376/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24111387_24111387`  [🔗 chained]
*job 24111387 • idun-06-01 • A100 • epoch 457/500 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24111388_24111388`  [✅ completed]
*job 24111388 • idun-07-10 • A100 • 11.21h training • epoch 2000/2000 • chain 2/40 • 114KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.48 • MS-SSIM 0.9923 • PSNR 34.59 dB • LPIPS 0.0613 • FID 68.87 • KID 0.0630 ± 0.0120 • CMMD 0.3208
  - **latest** ckpt (26 samples): MSE 4.891 • MS-SSIM 0.9686 • PSNR 28.41 dB • LPIPS 0.2423 • FID 61.14 • KID 0.0521 ± 0.0087 • CMMD 0.2833
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_3_ldm_4x_dit_s_long_20260304-102347/checkpoint_latest.pt`

#### `24111664_24111664`  [🔗 chained]
*job 24111664 • idun-07-10 • A100 • epoch 489/500 • chain 4/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24113841_24113841`  [✅ completed]
*job 24113841 • idun-06-07 • A100 • 9.93h training • epoch 500/500 • chain 0/20 • 85KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.5202 • MS-SSIM 0.9262 • PSNR 32.45 dB • LPIPS 0.3218 • FID 162.98 • KID 0.1618 ± 0.0090 • CMMD 0.4848
  - **latest** ckpt (26 samples): MSE 0.3158 • MS-SSIM 0.9507 • PSNR 33.98 dB • LPIPS 0.1938 • FID 110.88 • KID 0.0938 ± 0.0080 • CMMD 0.4088

#### `24114000_24114000`  [✅ completed]
*job 24114000 • idun-06-02 • A100 • 1.33h training • epoch 1000/1000 • chain 3/40 • 11KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002886 • MS-SSIM 0.9560 • PSNR 33.12 dB • LPIPS 0.7589 • FID 111.86 • KID 0.0921 ± 0.0067 • CMMD 0.3259
  - **latest** ckpt (26 samples): MSE 0.008323 • MS-SSIM 0.9237 • PSNR 30.16 dB • LPIPS 0.3856 • FID 49.53 • KID 0.0352 ± 0.0069 • CMMD 0.1495
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt`

#### `24114364_24114364`  [💥 oom_killed]
*job 24114364 • idun-07-09 • A100 • 11.43h training • epoch 500/500 • chain 0/20 • 96KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004095 • MS-SSIM 0.9497 • PSNR 33.44 dB • LPIPS 0.7918 • FID 78.30 • KID 0.0887 ± 0.0060 • CMMD 0.2198
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	690m37.038s
user	524m6.108s
sys	150m11.531s
[2026-03-06T23:06:23.424] error: Detected 1 oom_kill event in StepId=24114364.batch. Some of the step tasks have been OOM Killed.
```

#### `24114365_24114365`  [🔗 chained]
*job 24114365 • idun-06-01 • A100 • epoch 120/500 • chain 0/20 • 29KB log*


#### `24114366_24114366`  [✅ completed]
*job 24114366 • idun-06-01 • A100 • 10.89h training • epoch 500/500 • chain 0/20 • 102KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003311 • MS-SSIM 0.9459 • PSNR 32.94 dB • LPIPS 0.8903 • FID 82.85 • KID 0.0975 ± 0.0079 • CMMD 0.2354
  - **latest** ckpt (26 samples): MSE 0.007188 • MS-SSIM 0.9322 • PSNR 31.70 dB • LPIPS 0.5438 • FID 70.01 • KID 0.0820 ± 0.0069 • CMMD 0.1784

#### `24115304_24115304`  [🔗 chained]
*job 24115304 • idun-06-04 • A100 • epoch 484/500 • chain 3/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24115341_24115341`  [✅ completed]
*job 24115341 • idun-07-10 • A100 • 3.53h training • epoch 500/500 • chain 3/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.307 • MS-SSIM 0.9921 • PSNR 34.30 dB • LPIPS 0.0661 • FID 75.79 • KID 0.0675 ± 0.0125 • CMMD 0.3399
  - **latest** ckpt (26 samples): MSE 2.396 • MS-SSIM 0.9899 • PSNR 33.95 dB • LPIPS 0.0837 • FID 47.41 • KID 0.0355 ± 0.0075 • CMMD 0.2515
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt`

#### `24115380_24115380`  [✅ completed]
*job 24115380 • idun-07-10 • A100 • 1.51h training • epoch 500/500 • chain 5/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004045 • MS-SSIM 0.9579 • PSNR 33.47 dB • LPIPS 0.6612 • FID 129.77 • KID 0.1583 ± 0.0140 • CMMD 0.2623
  - **latest** ckpt (26 samples): MSE 0.006105 • MS-SSIM 0.9321 • PSNR 31.11 dB • LPIPS 0.4891 • FID 98.00 • KID 0.1100 ± 0.0114 • CMMD 0.1903
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_3_pixel_bravo_deep_wide_attn_l3_20260303-031712/checkpoint_latest.pt`

#### `24118770_24118770`  [🔗 chained]
*job 24118770 • idun-08-01 • H100 • epoch 260/500 • chain 0/20 • 60KB log*


#### `24118771_24118771`  [💥 oom_killed]
*job 24118771 • idun-06-07 • A100 • 10.62h training • epoch 500/500 • chain 0/20 • 91KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01449 • MS-SSIM 0.9058 • PSNR 32.83 dB • LPIPS 1.3048 • FID 118.72 • KID 0.1116 ± 0.0045 • CMMD 0.4160
  - **latest** ckpt (26 samples): MSE 1.003 • MS-SSIM 0.3960 • PSNR 15.75 dB • LPIPS 1.8775
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	642m20.340s
user	448m34.979s
sys	189m9.467s
[2026-03-07T09:49:58.272] error: Detected 1 oom_kill event in StepId=24118771.batch. Some of the step tasks have been OOM Killed.
```

#### `24121041_24121041`  [🔗 chained]
*job 24121041 • idun-06-07 • A100 • epoch 240/500 • chain 1/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`

#### `24121176_24121176`  [✅ completed]
*job 24121176 • idun-08-01 • H100 • 1.28h training • epoch 500/500 • chain 4/20 • 13KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003615 • MS-SSIM 0.9542 • PSNR 32.83 dB • LPIPS 0.6113 • FID 127.91 • KID 0.1477 ± 0.0090 • CMMD 0.3290
  - **latest** ckpt (26 samples): MSE 0.005463 • MS-SSIM 0.9336 • PSNR 30.99 dB • LPIPS 0.4606 • FID 99.62 • KID 0.1141 ± 0.0091 • CMMD 0.1601
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_2_pixel_bravo_deep_wide_20260303-031712/checkpoint_latest.pt`

#### `24121342_24121342`  [✅ completed]
*job 24121342 • idun-01-03 • H100 • 10.96h training • epoch 500/500 • chain 1/20 • 67KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006567 • MS-SSIM 0.9635 • PSNR 34.07 dB • LPIPS 0.5974 • FID 134.31 • KID 0.1252 ± 0.0093 • CMMD 0.3261
  - **latest** ckpt (26 samples): MSE 0.007464 • MS-SSIM 0.9716 • PSNR 34.76 dB • LPIPS 0.4379 • FID 139.64 • KID 0.1382 ± 0.0106 • CMMD 0.2737
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1m_1_pixel_bravo_global_norm_20260306-215802/checkpoint_latest.pt`

#### `24121607_24121607`  [🔗 chained]
*job 24121607 • idun-06-07 • A100 • epoch 359/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`

#### `24121732_24121732`  [💥 oom_killed]
*job 24121732 • idun-01-05 • H100 • epoch 125/1000 • chain 0/20 • 130KB log*

**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[2026-03-08T16:09:23.152] error: *** JOB 24121732 ON idun-01-05 CANCELLED AT 2026-03-08T16:09:23 DUE to SIGNAL Terminated ***
```

#### `24122196_24122196`  [💥 oom_killed]
*job 24122196 • idun-06-07 • A100 • epoch 392/500 • chain 3/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260306-113941/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[2026-03-08T16:11:27.682] error: *** JOB 24122196 ON idun-06-07 CANCELLED AT 2026-03-08T16:11:27 DUE to SIGNAL Terminated ***
```

#### `24122599_24122599`  [❌ crashed]
*job 24122599 • idun-07-09 • A100 • 0.5h training • epoch 3/3 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.06258 • MS-SSIM 0.6368 • PSNR 22.54 dB • LPIPS 1.7175 • FID 365.45 • KID 0.4820 ± 0.0135 • CMMD 0.6928
  - **latest** ckpt (26 samples): MSE 0.062 • MS-SSIM 0.5090 • PSNR 19.90 dB • LPIPS 1.7322 • FID 364.20 • KID 0.4810 ± 0.0143 • CMMD 0.6919
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<string>", line 13, in <module>
AttributeError: module 'medgen.pipeline.validation' has no attribute 'run_validation'
...
  covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

real	46m25.144s
user	47m20.104s
sys	13m24.493s
```

#### `24122694_24122694`  [❌ crashed]
*job 24122694 • idun-01-05 • H100 • 10.07h training • epoch 146/1000 • chain 0/20 • 140KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 381, in add_scalar
    self._get_file_writer().add_summary(summary, global_step, walltime)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", line 115, in add_summary
    self.add_event(event, global_step, walltime)
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/utils/tensorboard/writer.py", lin
```

#### `24122695_24122695`  [❌ crashed]
*job 24122695 • idun-07-08 • A100 • 3.21h training • epoch 29/500 • chain 0/20 • 31KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24122698_24122698`  [❌ crashed]
*job 24122698 • idun-01-05 • H100 • 2.33h training • epoch 33/500 • chain 0/20 • 36KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24126433_24126433`  [🔗 chained]
*job 24126433 • idun-01-05 • H100 • epoch 169/500 • chain 0/20 • 39KB log*


#### `24126434_24126434`  [🔗 chained]
*job 24126434 • idun-06-02 • A100 • epoch 117/500 • chain 0/20 • 97KB log*


#### `24126436_24126436`  [🔗 chained]
*job 24126436 • idun-07-10 • A100 • epoch 110/1000 • chain 0/20 • 29KB log*


#### `24126648_24126648`  [🔗 chained]
*job 24126648 • idun-06-02 • A100 • epoch 113/500 • chain 0/20 • 30KB log*


#### `24127045_24127045`  [🔗 chained]
*job 24127045 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 31KB log*


#### `24127090_24127090`  [🔗 chained]
*job 24127090 • idun-06-07 • A100 • epoch 231/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24127098_24127098`  [🔗 chained]
*job 24127098 • idun-07-09 • A100 • epoch 277/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`

#### `24127181_24127181`  [🔗 chained]
*job 24127181 • idun-08-01 • H100 • epoch 282/1000 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24128564_24128564`  [🔗 chained]
*job 24128564 • idun-06-01 • A100 • epoch 228/500 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24129682_24129682`  [❌ crashed]
*job 24129682 • idun-01-03 • H100 • epoch 185/500 • chain 0/20 • 914KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/pipeline/validation.py", line 513, in compute_validation_losses
    gen_results = trainer._gen_metrics.compute_epoch_metrics(
...
    pred_cond = self._call_model(
                ^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/diffusion/strategy_rflow.py", line 195, in _call_model
    return super()._call_model(model, model_input, timesteps, omega, mode_id, size_bins)
           ^^^^^^^^^
```

#### `24130511_24130511`  [🔗 chained]
*job 24130511 • idun-06-02 • A100 • epoch 236/500 • chain 1/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24131281_24131281`  [🔗 chained]
*job 24131281 • idun-07-09 • A100 • epoch 341/500 • chain 2/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24131353_24131353`  [💥 oom_killed]
*job 24131353 • idun-08-01 • H100 • epoch 432/500 • chain 2/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	650m29.237s
user	486m9.064s
sys	173m15.159s
[2026-03-11T20:38:27.894] error: Detected 1 oom_kill event in StepId=24131353.batch. Some of the step tasks have been OOM Killed.
```

#### `24131377_24131377`  [🔗 chained]
*job 24131377 • idun-06-01 • A100 • epoch 401/1000 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24131382_24131382`  [🔗 chained]
*job 24131382 • idun-07-10 • A100 • epoch 336/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24133692_24133692`  [❌ crashed]
*job 24133692 • idun-06-06 • A100 • 0.1h training • chain 1/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24137988_24137988`  [🔗 chained]
*job 24137988 • idun-08-01 • H100 • epoch 172/500 • chain 0/20 • 39KB log*


#### `24139725_24139725`  [❌ crashed]
*job 24139725 • idun-08-01 • H100 • 5.98h training • epoch 322/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet, chunk_sz, orig_d).cpu())
                              ^^^^^^^^
```

#### `24139869_24139869`  [❌ crashed]
*job 24139869 • idun-01-05 • H100 • 4.61h training • epoch 407/500 • chain 3/20 • 15KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 551, in generate_and_extract_features_3d_streaming
    all_resnet_3d.append(extract_features_3d_triplanar(sample_gpu, self_.resnet, chunk_sz, orig_d).cpu())
                              ^^^^^^^^
```

#### `24139886_24139886`  [🔗 chained]
*job 24139886 • idun-06-01 • A100 • epoch 521/1000 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24139898_24139898`  [💥 oom_killed]
*job 24139898 • idun-07-08 • A100 • epoch 339/500 • chain 3/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	22m1.456s
user	14m18.872s
sys	6m11.402s
[2026-03-12T01:40:51.118] error: Detected 1 oom_kill event in StepId=24139898.batch. Some of the step tasks have been OOM Killed.
```

#### `24139926_24139926`  [❌ crashed]
*job 24139926 • idun-07-10 • A100 • 1.0h training • epoch 193/500 • chain 2/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 563, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24139930_24139930`  [❌ crashed]
*job 24139930 • idun-01-03 • H100 • 0.96h training • epoch 444/500 • chain 3/20 • 7KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/generation_sampling.py", line 563, in generate_and_extract_features_3d_streaming
    all_resnet_rin.append(extract_features_batched(self_, sample_gpu, self_.resnet_rin).cpu())
                          ^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24139968_24139968`  [🔗 chained]
*job 24139968 • idun-07-08 • A100 • epoch 110/500 • chain 0/20 • 50KB log*


#### `24140632_24140632`  [🔗 chained]
*job 24140632 • idun-06-07 • A100 • epoch 288/500 • chain 1/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`

#### `24142845_24142845`  [🔗 chained]
*job 24142845 • idun-06-01 • A100 • epoch 640/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24144753_24144753`  [🔗 chained]
*job 24144753 • idun-07-08 • A100 • epoch 219/500 • chain 1/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24147112_24147112`  [🔗 chained]
*job 24147112 • idun-06-07 • A100 • epoch 404/500 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`

#### `24147147_24147147`  [🔗 chained]
*job 24147147 • idun-01-03 • H100 • epoch 499/500 • chain 0/20 • 81KB log*


#### `24147155_24147155`  [💥 oom_killed]
*job 24147155 • idun-08-01 • H100 • epoch 343/500 • chain 4/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	21m33.700s
user	15m35.729s
sys	5m54.117s
[2026-03-12T21:34:07.271] error: Detected 1 oom_kill event in StepId=24147155.batch. Some of the step tasks have been OOM Killed.
```

#### `24147160_24147160`  [🔗 chained]
*job 24147160 • idun-08-01 • H100 • epoch 369/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`

#### `24147165_24147165`  [✅ completed]
*job 24147165 • idun-06-05 • A100 • 9.36h training • epoch 500/500 • chain 4/20 • 24KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002761 • MS-SSIM 0.9476 • PSNR 32.69 dB • LPIPS 0.8929 • FID 165.65 • KID 0.2147 ± 0.0149 • CMMD 0.3337
  - **latest** ckpt (26 samples): MSE 0.004248 • MS-SSIM 0.9355 • PSNR 31.18 dB • LPIPS 0.4949 • FID 132.83 • KID 0.1739 ± 0.0152 • CMMD 0.2477
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt`

#### `24147168_24147168`  [✅ completed]
*job 24147168 • idun-06-06 • A100 • 5.81h training • epoch 500/500 • chain 4/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002532 • MS-SSIM 0.9477 • PSNR 32.36 dB • LPIPS 0.8944 • FID 92.49 • KID 0.1024 ± 0.0139 • CMMD 0.2773
  - **latest** ckpt (26 samples): MSE 0.002621 • MS-SSIM 0.9487 • PSNR 32.23 dB • LPIPS 0.6694 • FID 72.52 • KID 0.0704 ± 0.0123 • CMMD 0.2260
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt`

#### `24147171_24147171`  [🔗 chained]
*job 24147171 • idun-07-09 • A100 • epoch 431/500 • chain 3/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24147343_24147343`  [✅ completed]
*job 24147343 • idun-01-05 • H100 • 11.01h training • epoch 500/500 • chain 5/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002897 • MS-SSIM 0.9469 • PSNR 32.68 dB • LPIPS 0.8119 • FID 80.55 • KID 0.0889 ± 0.0123 • CMMD 0.2226
  - **latest** ckpt (26 samples): MSE 0.003812 • MS-SSIM 0.9590 • PSNR 33.36 dB • LPIPS 0.5329 • FID 62.64 • KID 0.0537 ± 0.0092 • CMMD 0.1898
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt`

#### `24147348_24147348`  [🔗 chained]
*job 24147348 • idun-07-09 • A100 • epoch 750/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24147766_24147766`  [🔗 chained]
*job 24147766 • idun-06-05 • A100 • epoch 337/500 • chain 2/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24148199_24148199`  [💥 oom_killed]
*job 24148199 • idun-01-05 • H100 • 6.62h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003326 • MS-SSIM 0.9508 • PSNR 32.67 dB • LPIPS 0.8495
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	398m53.534s
user	300m44.359s
sys	103m48.996s
[2026-03-13T14:49:52.774] error: Detected 2 oom_kill events in StepId=24148199.batch. Some of the step tasks have been OOM Killed.
```

#### `24148244_24148244`  [✅ completed]
*job 24148244 • idun-06-07 • A100 • 0.05h training • epoch 500/500 • chain 1/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006818 • MS-SSIM 0.9335 • PSNR 31.67 dB • LPIPS 0.5036 • FID 52.98 • KID 0.0520 ± 0.0066 • CMMD 0.1799
  - **latest** ckpt (26 samples): MSE 0.005155 • MS-SSIM 0.9279 • PSNR 31.26 dB • LPIPS 0.5373 • FID 49.62 • KID 0.0470 ± 0.0072 • CMMD 0.1624
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_pixel_bravo_controlnet_stage2_20260312-210758/checkpoint_latest.pt`

#### `24148265_24148265`  [💥 oom_killed]
*job 24148265 • idun-01-03 • H100 • 8.77h training • epoch 500/500 • chain 4/20 • 31KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003448 • MS-SSIM 0.9577 • PSNR 33.57 dB • LPIPS 0.5714
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	530m12.740s
user	401m55.684s
sys	135m11.556s
[2026-03-13T18:23:08.744] error: Detected 1 oom_kill event in StepId=24148265.batch. Some of the step tasks have been OOM Killed.
```

#### `24150904_24150904`  [🔗 chained]
*job 24150904 • idun-01-05 • H100 • epoch 437/500 • chain 0/20 • 79KB log*


#### `24150929_24150929`  [✅ completed]
*job 24150929 • idun-01-03 • H100 • 4.78h training • epoch 500/500 • chain 4/20 • 25KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.009399 • MS-SSIM 0.9676 • PSNR 34.99 dB • LPIPS 0.5624 • FID 116.55 • KID 0.1107 ± 0.0096 • CMMD 0.2171
  - **latest** ckpt (26 samples): MSE 0.01513 • MS-SSIM 0.9345 • PSNR 31.27 dB • LPIPS 0.4882 • FID 72.00 • KID 0.0689 ± 0.0078 • CMMD 0.1667
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_1_20260310-123920/checkpoint_latest.pt`

#### `24150939_24150939`  [🔗 chained]
*job 24150939 • idun-08-01 • H100 • epoch 410/500 • chain 0/20 • 74KB log*


#### `24150962_24150962`  [🔗 chained]
*job 24150962 • idun-01-05 • H100 • epoch 925/1000 • chain 6/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24151901_24151901`  [🔗 chained]
*job 24151901 • idun-01-03 • H100 • epoch 189/500 • chain 0/20 • 36KB log*


#### `24152108_24152108`  [🔗 chained]
*job 24152108 • idun-07-09 • A100 • epoch 447/500 • chain 3/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24154205_24154205`  [✅ completed]
*job 24154205 • idun-01-05 • H100 • 1.75h training • epoch 500/500 • chain 1/20 • 19KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.005689 • MS-SSIM 0.9552 • PSNR 33.69 dB • LPIPS 0.5217 • FID 54.60 • KID 0.0509 ± 0.0069 • CMMD 0.1899
  - **latest** ckpt (26 samples): MSE 0.004787 • MS-SSIM 0.9195 • PSNR 30.86 dB • LPIPS 0.5152 • FID 43.45 • KID 0.0323 ± 0.0055 • CMMD 0.1681
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_pixel_bravo_weight_decay_20260313-141301/checkpoint_latest.pt`

#### `24154213_24154213`  [✅ completed]
*job 24154213 • idun-01-05 • H100 • 2.48h training • epoch 500/500 • chain 1/20 • 26KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002354 • MS-SSIM 0.9408 • PSNR 32.63 dB • LPIPS 0.7802 • FID 82.21 • KID 0.0887 ± 0.0077 • CMMD 0.2667
  - **latest** ckpt (26 samples): MSE 0.008438 • MS-SSIM 0.9187 • PSNR 30.55 dB • LPIPS 0.4523 • FID 49.32 • KID 0.0466 ± 0.0045 • CMMD 0.1629
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1r_pixel_bravo_attn_dropout_20260313-142342/checkpoint_latest.pt`

#### `24154217_24154217`  [🔗 chained]
*job 24154217 • idun-07-09 • A100 • epoch 258/500 • chain 0/20 • 54KB log*


#### `24154218_24154218`  [🔗 chained]
*job 24154218 • idun-07-09 • A100 • epoch 187/500 • chain 0/20 • 40KB log*


#### `24154219_24154219`  [🔗 chained]
*job 24154219 • idun-01-05 • H100 • epoch 425/500 • chain 0/20 • 109KB log*


#### `24154238_24154238`  [❌ crashed]
*job 24154238 • idun-07-09 • A100 • 10.82h training • epoch 239/500 • chain 0/20 • 74KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/work/modestas/AIS4900_master/src/medgen/metrics/quality.py", line 320, in _get_lpips_metric
    metric = PerceptualLoss(
             ^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/losses/perceptual.py", 
```

#### `24154257_24154257`  [✅ completed]
*job 24154257 • idun-01-05 • H100 • 5.15h training • epoch 1000/1000 • chain 7/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002195 • MS-SSIM 0.9553 • PSNR 33.04 dB • LPIPS 0.5915 • FID 72.39 • KID 0.0579 ± 0.0061 • CMMD 0.1936
  - **latest** ckpt (26 samples): MSE 0.003152 • MS-SSIM 0.9569 • PSNR 33.34 dB • LPIPS 0.5307 • FID 62.57 • KID 0.0531 ± 0.0091 • CMMD 0.1850
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt`

#### `24154353_24154353`  [🔗 chained]
*job 24154353 • idun-07-09 • A100 • epoch 292/500 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24154591_24154591`  [✅ completed]
*job 24154591 • idun-07-08 • A100 • 5.81h training • epoch 500/500 • chain 4/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003009 • MS-SSIM 0.9484 • PSNR 32.84 dB • LPIPS 0.8018 • FID 62.00 • KID 0.0508 ± 0.0097 • CMMD 0.2228
  - **latest** ckpt (26 samples): MSE 0.00416 • MS-SSIM 0.9495 • PSNR 33.16 dB • LPIPS 0.6704 • FID 58.85 • KID 0.0464 ± 0.0108 • CMMD 0.1977
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt`

#### `24156955_24156955`  [✅ completed]
*job 24156955 • idun-06-02 • A100 • 9.96h training • epoch 500/500 • chain 1/20 • 56KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003971 • MS-SSIM 0.9517 • PSNR 32.92 dB • LPIPS 0.6851 • FID 95.85 • KID 0.0944 ± 0.0045 • CMMD 0.2430
  - **latest** ckpt (26 samples): MSE 0.004889 • MS-SSIM 0.9298 • PSNR 30.77 dB • LPIPS 0.5877 • FID 99.86 • KID 0.1104 ± 0.0092 • CMMD 0.2452
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_4_pixel_bravo_small_67m_20260314-064756/checkpoint_latest.pt`

#### `24156956_24156956`  [🔗 chained]
*job 24156956 • idun-01-05 • H100 • epoch 490/500 • chain 1/20 • 59KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_5_pixel_bravo_mid_152m_20260314-064756/checkpoint_latest.pt`

#### `24156991_24156991`  [✅ completed]
*job 24156991 • idun-06-07 • A100 • 2.98h training • epoch 500/500 • chain 1/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00329 • MS-SSIM 0.9580 • PSNR 33.06 dB • LPIPS 0.5910 • FID 112.87 • KID 0.1394 ± 0.0134 • CMMD 0.2344
  - **latest** ckpt (26 samples): MSE 0.004855 • MS-SSIM 0.9556 • PSNR 32.90 dB • LPIPS 0.5191 • FID 94.75 • KID 0.1017 ± 0.0100 • CMMD 0.2306
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_6_pixel_bravo_tiny_20m_20260314-072201/checkpoint_latest.pt`

#### `24157008_24157008`  [✅ completed]
*job 24157008 • idun-06-06 • A100 • 10.42h training • epoch 500/500 • chain 0/20 • 58KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003105 • MS-SSIM 0.9525 • PSNR 32.63 dB • LPIPS 0.6132 • FID 112.64 • KID 0.1244 ± 0.0135 • CMMD 0.3127
  - **latest** ckpt (26 samples): MSE 0.004998 • MS-SSIM 0.9585 • PSNR 33.26 dB • LPIPS 0.5157 • FID 92.16 • KID 0.0977 ± 0.0087 • CMMD 0.2342
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_7_pixel_bravo_67m_no_attn_20260314-072441/checkpoint_latest.pt`

#### `24158373_24158373`  [🔗 chained]
*job 24158373 • idun-08-01 • H100 • epoch 477/500 • chain 2/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24158696_24158696`  [🔗 chained]
*job 24158696 • idun-01-05 • H100 • epoch 283/2000 • chain 0/40 • 129KB log*


#### `24158706_24158706`  [🔗 chained]
*job 24158706 • idun-01-05 • H100 • epoch 172/1000 • chain 0/20 • 51KB log*


#### `24158726_24158726`  [✅ completed]
*job 24158726 • idun-01-05 • H100 • 0.44h training • epoch 500/500 • chain 2/20 • 11KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003212 • MS-SSIM 0.9569 • PSNR 33.13 dB • LPIPS 0.6350 • FID 95.00 • KID 0.0895 ± 0.0118 • CMMD 0.3304
  - **latest** ckpt (26 samples): MSE 0.004167 • MS-SSIM 0.9329 • PSNR 30.95 dB • LPIPS 0.5574 • FID 72.11 • KID 0.0706 ± 0.0097 • CMMD 0.1885
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp20_5_pixel_bravo_mid_152m_20260314-064756/checkpoint_latest.pt`

#### `24158771_24158771`  [✅ completed]
*job 24158771 • idun-01-05 • H100 • 1.45h training • epoch 500/500 • chain 3/20 • 15KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002549 • MS-SSIM 0.9522 • PSNR 32.99 dB • LPIPS 0.6602 • FID 76.11 • KID 0.0749 ± 0.0102 • CMMD 0.2481
  - **latest** ckpt (26 samples): MSE 0.002899 • MS-SSIM 0.9573 • PSNR 33.40 dB • LPIPS 0.5920 • FID 82.82 • KID 0.0896 ± 0.0093 • CMMD 0.2242
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6b_1_pixel_bravo_controlnet_stage2_20260313-211034/checkpoint_latest.pt`

#### `24160233_24160233`  [🔗 chained]
*job 24160233 • idun-06-02 • A100 • epoch 491/2000 • chain 1/40 • 96KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24160234_24160234`  [🔗 chained]
*job 24160234 • idun-06-02 • A100 • epoch 291/1000 • chain 1/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24167549_24167549`  [🔗 chained]
*job 24167549 • idun-08-01 • H100 • epoch 744/2000 • chain 2/40 • 119KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24167550_24167550`  [🔗 chained]
*job 24167550 • idun-07-09 • A100 • epoch 400/1000 • chain 2/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24188995_24188995`  [🔗 chained]
*job 24188995 • idun-01-03 • H100 • epoch 1031/2000 • chain 3/40 • 138KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24189005_24189005`  [🔗 chained]
*job 24189005 • idun-01-05 • H100 • epoch 572/1000 • chain 3/20 • 55KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24193972_24193972`  [🔗 chained]
*job 24193972 • idun-08-01 • H100 • epoch 1314/2000 • chain 4/40 • 142KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24193976_24193976`  [🔗 chained]
*job 24193976 • idun-01-05 • H100 • epoch 743/1000 • chain 4/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24199806_24199806`  [🔗 chained]
*job 24199806 • idun-01-05 • H100 • epoch 1599/2000 • chain 5/40 • 138KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24200116_24200116`  [🔗 chained]
*job 24200116 • idun-06-05 • A100 • epoch 860/1000 • chain 5/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24204011_24204011`  [🔗 chained]
*job 24204011 • idun-08-01 • H100 • epoch 1887/2000 • chain 6/40 • 135KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24204272_24204272`  [✅ completed]
*job 24204272 • idun-08-01 • H100 • 9.63h training • epoch 1000/1000 • chain 6/20 • 54KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01121 • MS-SSIM 0.9504 • PSNR 34.24 dB • LPIPS 0.7429 • FID 65.46 • KID 0.0542 ± 0.0107 • CMMD 0.2549
  - **latest** ckpt (26 samples): MSE 0.005543 • MS-SSIM 0.9404 • PSNR 32.70 dB • LPIPS 0.8810 • FID 62.87 • KID 0.0526 ± 0.0093 • CMMD 0.2594
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt`

#### `24206608_24206608`  [✅ completed]
*job 24206608 • idun-07-08 • A100 • 7.14h training • epoch 2000/2000 • chain 7/40 • 62KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.008342 • MS-SSIM 0.9470 • PSNR 33.54 dB • LPIPS 0.7312 • FID 72.95 • KID 0.0579 ± 0.0051 • CMMD 0.3016
  - **latest** ckpt (26 samples): MSE 0.02007 • MS-SSIM 0.9580 • PSNR 34.20 dB • LPIPS 0.6224 • FID 73.49 • KID 0.0565 ± 0.0070 • CMMD 0.3082
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt`

#### `24232399_24232399`  [🔗 chained]
*job 24232399 • idun-06-05 • A100 • epoch 281/1000 • chain 0/20 • 52KB log*


#### `24232401_24232401`  [🔗 chained]
*job 24232401 • idun-06-05 • A100 • epoch 580/1000 • chain 0/20 • 103KB log*


#### `24233992_24233992`  [🔗 chained]
*job 24233992 • idun-06-05 • A100 • epoch 558/1000 • chain 1/20 • 52KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`

#### `24234000_24234000`  [✅ completed]
*job 24234000 • idun-06-05 • A100 • 8.81h training • epoch 1000/1000 • chain 1/20 • 82KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.112 • MS-SSIM 0.9830 • PSNR 31.39 dB • LPIPS 0.1553 • FID 131.66 • KID 0.1368 ± 0.0237 • CMMD 0.3866
  - **latest** ckpt (26 samples): MSE 3.005 • MS-SSIM 0.9895 • PSNR 33.44 dB • LPIPS 0.0970 • FID 79.91 • KID 0.0786 ± 0.0173 • CMMD 0.2612
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_ldm_4x_unet_maisi_scoreaug_20260325-013810/checkpoint_latest.pt`

#### `24235208_24235208`  [🔗 chained]
*job 24235208 • idun-07-08 • A100 • epoch 136/1000 • chain 0/20 • 120KB log*


#### `24235533_24235533`  [🔗 chained]
*job 24235533 • idun-06-05 • A100 • epoch 569/1000 • chain 0/20 • 108KB log*


#### `24235992_24235992`  [🔗 chained]
*job 24235992 • idun-06-05 • A100 • epoch 838/1000 • chain 2/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`

#### `24236459_24236459`  [🔗 chained]
*job 24236459 • idun-07-10 • A100 • epoch 270/1000 • chain 1/20 • 151KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24236678_24236678`  [✅ completed]
*job 24236678 • idun-06-05 • A100 • 8.97h training • epoch 1000/1000 • chain 1/20 • 86KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.104 • MS-SSIM 0.9868 • PSNR 32.68 dB • LPIPS 0.1136 • FID 168.88 • KID 0.1857 ± 0.0279 • CMMD 0.3986
  - **latest** ckpt (26 samples): MSE 2.499 • MS-SSIM 0.9812 • PSNR 32.74 dB • LPIPS 0.1553 • FID 98.56 • KID 0.0960 ± 0.0211 • CMMD 0.3725
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_1_ldm_4x_unet_maisi_scoreaug_mixup_20260325-211501/checkpoint_latest.pt`

#### `24238367_24238367`  [💥 oom_killed]
*job 24238367 • idun-06-07 • A100 • 6.95h training • epoch 1000/1000 • chain 3/20 • 36KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.3704 • MS-SSIM 0.9661 • PSNR 36.27 dB • LPIPS 0.2521
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt`
**Traceback excerpt:**
```
nown)"))'), '(Request ID: 08de47ff-3be5-4f0c-aa43-3fd382d13d69)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json
Retrying in 2s [Retry 2/5].
'(MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json (Caused by NameResolutionError("HTTPSConnection(host=\'huggingface.co\', port=443): Failed to resolve \'huggingface.co\' ([Errno -2] Name or service not known)"))'), '(Request ID: ed818f74-6c73-426e-991c-05c7d142f3f4)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_config.json
...

real	423m24.548s
user	308m18.299s
sys	107m46.389s
[2026-03-26T21:36:56.874] error: Detected 1 oom_kill event in StepId=24238367.batch. Some of the step tasks have been OOM Killed.
```

#### `24238431_24238431`  [🔗 chained]
*job 24238431 • idun-06-05 • A100 • epoch 568/1000 • chain 0/20 • 102KB log*


#### `24239579_24239579`  [🔗 chained]
*job 24239579 • idun-07-10 • A100 • epoch 259/500 • chain 0/20 • 85KB log*


#### `24239580_24239580`  [🔗 chained]
*job 24239580 • idun-07-09 • A100 • epoch 268/500 • chain 0/20 • 95KB log*


#### `24240420_24240420`  [🔗 chained]
*job 24240420 • idun-07-10 • A100 • epoch 399/1000 • chain 2/20 • 398KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24241623_24241623`  [🔗 chained]
*job 24241623 • idun-06-07 • A100 • epoch 282/1000 • chain 0/20 • 74KB log*


#### `24241654_24241654`  [🔗 chained]
*job 24241654 • idun-06-03 • A100 • epoch 319/500 • chain 0/20 • 94KB log*


#### `24241655_24241655`  [🔗 chained]
*job 24241655 • idun-06-06 • A100 • epoch 318/500 • chain 0/20 • 90KB log*


#### `24241672_24241672`  [✅ completed]
*job 24241672 • idun-07-10 • A100 • 10.82h training • epoch 1000/1000 • chain 1/20 • 92KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 2.24 • MS-SSIM 0.9905 • PSNR 35.24 dB • LPIPS 0.0970 • FID 93.32 • KID 0.1057 ± 0.0250 • CMMD 0.2819
  - **latest** ckpt (26 samples): MSE 2.342 • MS-SSIM 0.9836 • PSNR 33.51 dB • LPIPS 0.1540 • FID 80.17 • KID 0.0776 ± 0.0167 • CMMD 0.2945
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_2_ldm_4x_unet_maisi_scoreaug_v2_20260326-144538/checkpoint_latest.pt`

#### `24241848_24241848`  [✅ completed]
*job 24241848 • idun-07-10 • A100 • 11.1h training • epoch 500/500 • chain 1/20 • 79KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002548 • MS-SSIM 0.9370 • PSNR 32.47 dB • LPIPS 0.9223 • FID 79.68 • KID 0.0887 ± 0.0104 • CMMD 0.3276
  - **latest** ckpt (26 samples): MSE 0.00376 • MS-SSIM 0.9475 • PSNR 32.93 dB • LPIPS 0.6537 • FID 65.37 • KID 0.0642 ± 0.0063 • CMMD 0.2506
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_pixel_triple_20260326-164117/checkpoint_latest.pt`

#### `24241849_24241849`  [✅ completed]
*job 24241849 • idun-06-04 • A100 • 9.13h training • epoch 500/500 • chain 1/20 • 70KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002441 • MS-SSIM 0.9439 • PSNR 33.44 dB • LPIPS 0.7884 • FID 32.80 • KID 0.0216 ± 0.0022 • CMMD 0.1715
  - **latest** ckpt (26 samples): MSE 0.003058 • MS-SSIM 0.9418 • PSNR 32.39 dB • LPIPS 0.6395 • FID 35.91 • KID 0.0299 ± 0.0044 • CMMD 0.1756
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_pixel_dual_20260326-164147/checkpoint_latest.pt`

#### `24241903_24241903`  [🔗 chained]
*job 24241903 • idun-07-10 • A100 • epoch 531/1000 • chain 3/20 • 283KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24244849_24244849`  [🔗 chained]
*job 24244849 • idun-06-07 • A100 • epoch 563/1000 • chain 1/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24244953_24244953`  [✅ completed]
*job 24244953 • idun-01-03 • H100 • 5.13h training • epoch 500/500 • chain 1/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004596 • MS-SSIM 0.9559 • PSNR 32.84 dB • LPIPS 0.5267 • FID 24.30 • KID 0.0097 ± 0.0027 • CMMD 0.2267
  - **latest** ckpt (26 samples): MSE 0.003629 • MS-SSIM 0.9261 • PSNR 30.64 dB • LPIPS 0.6767 • FID 44.54 • KID 0.0376 ± 0.0048 • CMMD 0.1839
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_1_pixel_dual_joint_norm_20260327-031630/checkpoint_latest.pt`

#### `24245009_24245009`  [✅ completed]
*job 24245009 • idun-06-04 • A100 • 6.84h training • epoch 500/500 • chain 1/20 • 44KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006714 • MS-SSIM 0.9434 • PSNR 32.16 dB • LPIPS 0.7475 • FID 80.54 • KID 0.0890 ± 0.0050 • CMMD 0.3843
  - **latest** ckpt (26 samples): MSE 0.007807 • MS-SSIM 0.9412 • PSNR 32.04 dB • LPIPS 0.6518 • FID 66.57 • KID 0.0735 ± 0.0056 • CMMD 0.3145
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_1_pixel_triple_joint_norm_20260327-033611/checkpoint_latest.pt`

#### `24245657_24245657`  [🔗 chained]
*job 24245657 • idun-06-07 • A100 • epoch 684/1000 • chain 4/20 • 32KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24247235_24247235`  [🔗 chained]
*job 24247235 • idun-07-10 • A100 • epoch 806/1000 • chain 2/20 • 44KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24248097_24248097`  [🔗 chained]
*job 24248097 • idun-07-09 • A100 • epoch 821/1000 • chain 5/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24250164_24250164`  [✅ completed]
*job 24250164 • idun-06-02 • A100 • 8.16h training • epoch 1000/1000 • chain 3/20 • 41KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.395 • MS-SSIM 0.9514 • PSNR 34.81 dB • LPIPS 0.2612 • FID 79.03 • KID 0.0557 ± 0.0060 • CMMD 0.2320
  - **latest** ckpt (26 samples): MSE 0.4457 • MS-SSIM 0.9430 • PSNR 34.30 dB • LPIPS 0.2602 • FID 77.28 • KID 0.0511 ± 0.0049 • CMMD 0.2174
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt`

#### `24250963_24250963`  [🔗 chained]
*job 24250963 • idun-07-10 • A100 • epoch 959/1000 • chain 6/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24253509_24253509`  [✅ completed]
*job 24253509 • idun-06-03 • A100 • 3.2h training • epoch 1000/1000 • chain 7/20 • 20KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 1.288 • MS-SSIM 0.9864 • PSNR 33.30 dB • LPIPS 0.1150 • FID 60.02 • KID 0.0493 ± 0.0080 • CMMD 0.2899
  - **latest** ckpt (26 samples): MSE 3.105 • MS-SSIM 0.9863 • PSNR 33.96 dB • LPIPS 0.1160 • FID 57.24 • KID 0.0477 ± 0.0109 • CMMD 0.2514
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt`

#### `24262210_24262210`  [🔗 chained]
*job 24262210 • idun-06-03 • A100 • epoch 119/1000 • chain 0/40 • 30KB log*


#### `24262211_24262211`  [🔗 chained]
*job 24262211 • idun-06-07 • A100 • epoch 116/1000 • chain 0/20 • 30KB log*


#### `24262212_24262212`  [🔗 chained]
*job 24262212 • idun-06-02 • A100 • epoch 118/500 • chain 0/20 • 29KB log*


#### `24269769_24269769`  [🔗 chained]
*job 24269769 • idun-01-05 • H100 • epoch 292/1000 • chain 1/40 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24269836_24269836`  [🔗 chained]
*job 24269836 • idun-01-04 • H100 • epoch 289/1000 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24269857_24269857`  [🔗 chained]
*job 24269857 • idun-06-03 • A100 • epoch 236/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24269897_24269897`  [🔗 chained]
*job 24269897 • idun-06-02 • A100 • epoch 113/500 • chain 0/20 • 58KB log*


#### `24269898_24269898`  [🔗 chained]
*job 24269898 • idun-06-01 • A100 • epoch 87/1000 • chain 0/20 • 22KB log*


#### `24270064_24270064`  [🔗 chained]
*job 24270064 • idun-07-10 • A100 • epoch 401/1000 • chain 2/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24270897_24270897`  [🔗 chained]
*job 24270897 • idun-07-08 • A100 • epoch 399/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24270899_24270899`  [🔗 chained]
*job 24270899 • idun-06-03 • A100 • epoch 353/500 • chain 2/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24271085_24271085`  [🔗 chained]
*job 24271085 • idun-01-05 • H100 • epoch 279/500 • chain 1/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24271087_24271087`  [🔗 chained]
*job 24271087 • idun-07-10 • A100 • epoch 197/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24271626_24271626`  [🔗 chained]
*job 24271626 • idun-07-10 • A100 • epoch 511/1000 • chain 3/40 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24271654_24271654`  [🔗 chained]
*job 24271654 • idun-07-09 • A100 • epoch 509/1000 • chain 3/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24271661_24271661`  [🔗 chained]
*job 24271661 • idun-07-09 • A100 • epoch 463/500 • chain 3/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24271681_24271681`  [🔗 chained]
*job 24271681 • idun-01-05 • H100 • epoch 172/1000 • chain 0/20 • 42KB log*


#### `24271682_24271682`  [🔗 chained]
*job 24271682 • idun-01-05 • H100 • epoch 170/1000 • chain 0/20 • 39KB log*


#### `24271719_24271719`  [🔗 chained]
*job 24271719 • idun-01-05 • H100 • epoch 444/500 • chain 2/20 • 40KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24271730_24271730`  [🔗 chained]
*job 24271730 • idun-07-08 • A100 • epoch 307/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24272484_24272484`  [🔗 chained]
*job 24272484 • idun-01-04 • H100 • epoch 686/1000 • chain 4/40 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24272485_24272485`  [🔗 chained]
*job 24272485 • idun-07-09 • A100 • epoch 619/1000 • chain 4/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24272487_24272487`  [🔗 chained]
*job 24272487 • idun-01-04 • H100 • epoch 126/500 • chain 0/20 • 33KB log*


#### `24272492_24272492`  [✅ completed]
*job 24272492 • idun-07-09 • A100 • 4.08h training • epoch 500/500 • chain 4/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003614 • MS-SSIM 0.9596 • PSNR 33.64 dB • LPIPS 0.5065 • FID 70.23 • KID 0.0559 ± 0.0078 • CMMD 0.2434
  - **latest** ckpt (26 samples): MSE 0.003257 • MS-SSIM 0.9574 • PSNR 33.31 dB • LPIPS 0.5367 • FID 72.30 • KID 0.0599 ± 0.0094 • CMMD 0.2741
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_1_pixel_bravo_mixup_20260402-131808/checkpoint_latest.pt`

#### `24272501_24272501`  [🔗 chained]
*job 24272501 • idun-01-04 • H100 • epoch 436/500 • chain 0/20 • 76KB log*


#### `24272522_24272522`  [🔗 chained]
*job 24272522 • idun-06-02 • A100 • epoch 292/1000 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24272524_24272524`  [🔗 chained]
*job 24272524 • idun-07-08 • A100 • epoch 280/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24272595_24272595`  [✅ completed]
*job 24272595 • idun-01-03 • H100 • 4.12h training • epoch 500/500 • chain 3/20 • 21KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002772 • MS-SSIM 0.9539 • PSNR 33.49 dB • LPIPS 0.9445 • FID 137.93 • KID 0.1472 ± 0.0065 • CMMD 0.4062
  - **latest** ckpt (26 samples): MSE 0.00264 • MS-SSIM 0.9622 • PSNR 34.13 dB • LPIPS 0.7044 • FID 88.85 • KID 0.0839 ± 0.0070 • CMMD 0.3084
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_2_pixel_triple_256_20260403-035904/checkpoint_latest.pt`

#### `24272609_24272609`  [🔗 chained]
*job 24272609 • idun-01-05 • H100 • epoch 478/1000 • chain 3/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24274934_24274934`  [🔗 chained]
*job 24274934 • idun-06-02 • A100 • epoch 339/500 • chain 0/20 • 63KB log*


#### `24274953_24274953`  [🔗 chained]
*job 24274953 • idun-01-04 • H100 • epoch 861/1000 • chain 5/40 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24274955_24274955`  [🔗 chained]
*job 24274955 • idun-01-04 • H100 • epoch 795/1000 • chain 5/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24274957_24274957`  [🔗 chained]
*job 24274957 • idun-01-03 • H100 • epoch 297/500 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24274974_24274974`  [✅ completed]
*job 24274974 • idun-01-03 • H100 • 1.75h training • epoch 500/500 • chain 1/20 • 20KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.00252 • MS-SSIM 0.9334 • PSNR 32.49 dB • LPIPS 0.8540 • FID 67.67 • KID 0.0723 ± 0.0077 • CMMD 0.2184
  - **latest** ckpt (26 samples): MSE 0.005569 • MS-SSIM 0.9198 • PSNR 30.74 dB • LPIPS 0.5132 • FID 35.56 • KID 0.0267 ± 0.0043 • CMMD 0.1265
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_02_pixel_bravo_weight_decay_128_20260404-144426/checkpoint_latest.pt`

#### `24274999_24274999`  [🔗 chained]
*job 24274999 • idun-01-04 • H100 • epoch 419/1000 • chain 2/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24275061_24275061`  [🔗 chained]
*job 24275061 • idun-01-03 • H100 • epoch 452/1000 • chain 2/20 • 33KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24275088_24275088`  [🔗 chained]
*job 24275088 • idun-01-05 • H100 • epoch 651/1000 • chain 4/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24275414_24275414`  [💥 oom_killed]
*job 24275414 • idun-06-01 • A100 • 5.76h training • epoch 500/500 • chain 1/20 • 49KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002792 • MS-SSIM 0.9370 • PSNR 32.57 dB • LPIPS 0.7480 • FID 57.25 • KID 0.0544 ± 0.0068 • CMMD 0.2225
  - **latest** ckpt (26 samples): MSE 0.003154 • MS-SSIM 0.9360 • PSNR 32.23 dB • LPIPS 0.5959
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_pixel_bravo_mixup_128_20260405-042205/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	354m51.968s
user	268m14.444s
sys	88m7.109s
[2026-04-05T23:56:29.313] error: Detected 1 oom_kill event in StepId=24275414.batch. Some of the step tasks have been OOM Killed.
```

#### `24275450_24275450`  [🔗 chained]
*job 24275450 • idun-07-08 • A100 • epoch 973/1000 • chain 6/40 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24275452_24275452`  [🔗 chained]
*job 24275452 • idun-08-01 • H100 • epoch 968/1000 • chain 6/20 • 45KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`

#### `24275454_24275454`  [🔗 chained]
*job 24275454 • idun-08-01 • H100 • epoch 467/500 • chain 2/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24275460_24275460`  [🔗 chained]
*job 24275460 • idun-08-01 • H100 • epoch 593/1000 • chain 3/20 • 43KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24275551_24275551`  [🔗 chained]
*job 24275551 • idun-06-02 • A100 • epoch 286/1000 • chain 0/20 • 143KB log*


#### `24275552_24275552`  [🔗 chained]
*job 24275552 • idun-06-02 • A100 • epoch 206/2000 • chain 0/40 • 80KB log*


#### `24276244_24276244`  [❌ crashed]
*job 24276244 • idun-01-04 • H100 • 1.65h training • epoch 476/1000 • chain 3/20 • 13KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	102m1.530s
user	99m34.169s
sys	31m57.012s
```

#### `24276638_24276638`  [🔗 chained]
*job 24276638 • idun-01-05 • H100 • epoch 798/1000 • chain 5/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24276704_24276704`  [🔗 chained]
*job 24276704 • idun-01-03 • H100 • epoch 170/1000 • chain 0/20 • 55KB log*


#### `24276705_24276705`  [💥 oom_killed]
*job 24276705 • idun-07-09 • A100 • 3.93h training • epoch 500/500 • chain 2/20 • 26KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003494 • MS-SSIM 0.9385 • PSNR 32.02 dB • LPIPS 0.5698 • FID 60.61 • KID 0.0529 ± 0.0058 • CMMD 0.2814
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1t_pixel_bravo_mixup_128_20260405-042205/checkpoint_latest.pt`
**Traceback excerpt:**
```
gface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: d377e735-8d9c-4089-a2a8-b9f890ab774b)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/resolve/main/config.json
Retrying in 1s [Retry 1/5].
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a71fa5e2-6f1b-4e69-90d4-77e6649a000b)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_model.safetensors
...

real	244m33.104s
user	201m17.104s
sys	44m17.548s
[2026-04-06T09:39:20.032] error: Detected 1 oom_kill event in StepId=24276705.batch. Some of the step tasks have been OOM Killed.
```

#### `24276706_24276706`  [🔗 chained]
*job 24276706 • idun-01-04 • H100 • epoch 624/1000 • chain 4/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24276792_24276792`  [💥 oom_killed]
*job 24276792 • idun-06-02 • A100 • 2.73h training • epoch 1000/1000 • chain 7/40 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	168m32.841s
user	110m34.444s
sys	61m55.793s
[2026-04-06T12:00:16.518] error: Detected 2 oom_kill events in StepId=24276792.batch. Some of the step tasks have been OOM Killed.
```

#### `24276842_24276842`  [🔗 chained]
*job 24276842 • idun-07-09 • A100 • epoch 702/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24276843_24276843`  [💥 oom_killed]
*job 24276843 • idun-06-03 • A100 • 3.26h training • epoch 1000/1000 • chain 7/20 • 16KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004321 • MS-SSIM 0.9474 • PSNR 32.40 dB • LPIPS 0.4816 • FID 62.30 • KID 0.0478 ± 0.0073 • CMMD 0.1836
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_1_pixel_bravo_scoreaug_safe_20260402-122902/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	208m40.817s
user	138m34.530s
sys	72m0.487s
[2026-04-06T12:44:07.615] error: Detected 1 oom_kill event in StepId=24276843.batch. Some of the step tasks have been OOM Killed.
```

#### `24276844_24276844`  [✅ completed]
*job 24276844 • idun-06-03 • A100 • 3.54h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001808 • MS-SSIM 0.9573 • PSNR 33.72 dB • LPIPS 0.7451 • FID 54.31 • KID 0.0364 ± 0.0054 • CMMD 0.2795
  - **latest** ckpt (26 samples): MSE 0.003421 • MS-SSIM 0.9681 • PSNR 35.07 dB • LPIPS 0.5920 • FID 50.32 • KID 0.0377 ± 0.0059 • CMMD 0.2169
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_pixel_dual_256_20260404-140254/checkpoint_latest.pt`

#### `24276863_24276863`  [🔗 chained]
*job 24276863 • idun-01-04 • H100 • epoch 690/1000 • chain 1/20 • 190KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24276866_24276866`  [🔗 chained]
*job 24276866 • idun-06-03 • A100 • epoch 414/2000 • chain 1/40 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24277127_24277127`  [🔗 chained]
*job 24277127 • idun-01-05 • H100 • epoch 971/1000 • chain 6/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`

#### `24277645_24277645`  [🔗 chained]
*job 24277645 • idun-01-05 • H100 • epoch 340/1000 • chain 1/20 • 52KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24277739_24277739`  [🔗 chained]
*job 24277739 • idun-08-01 • H100 • epoch 794/1000 • chain 5/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24277906_24277906`  [🔗 chained]
*job 24277906 • idun-07-10 • A100 • epoch 812/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24277912_24277912`  [🔗 chained]
*job 24277912 • idun-07-10 • A100 • epoch 950/1000 • chain 2/20 • 126KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24277915_24277915`  [🔗 chained]
*job 24277915 • idun-01-04 • H100 • epoch 700/2000 • chain 2/40 • 108KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24278291_24278291`  [💥 oom_killed]
*job 24278291 • idun-01-05 • H100 • 2.08h training • epoch 1000/1000 • chain 7/20 • 14KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005083 • MS-SSIM 0.9184 • PSNR 29.88 dB • LPIPS 0.3982 • FID 50.69 • KID 0.0349 ± 0.0090 • CMMD 0.1653
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1s_01_pixel_bravo_weight_decay_20260403-040937/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	137m4.532s
user	99m36.552s
sys	34m58.079s
[2026-04-07T08:16:15.786] error: Detected 1 oom_kill event in StepId=24278291.batch. Some of the step tasks have been OOM Killed.
```

#### `24278590_24278590`  [🔗 chained]
*job 24278590 • idun-06-03 • A100 • epoch 457/1000 • chain 2/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24278628_24278628`  [🔗 chained]
*job 24278628 • idun-06-03 • A100 • epoch 912/1000 • chain 6/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`

#### `24278984_24278984`  [🔗 chained]
*job 24278984 • idun-01-03 • H100 • epoch 985/1000 • chain 6/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24278999_24278999`  [✅ completed]
*job 24278999 • idun-08-01 • H100 • 2.8h training • epoch 1000/1000 • chain 3/20 • 940KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.01737 • MS-SSIM 0.9371 • PSNR 32.40 dB • LPIPS 0.7213 • FID 65.04 • KID 0.0546 ± 0.0078 • CMMD 0.2479
  - **latest** ckpt (26 samples): MSE 0.01111 • MS-SSIM 0.9415 • PSNR 32.92 dB • LPIPS 0.7381 • FID 71.03 • KID 0.0647 ± 0.0134 • CMMD 0.2529
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125/checkpoint_latest.pt`

#### `24279000_24279000`  [🔗 chained]
*job 24279000 • idun-08-01 • H100 • epoch 867/2000 • chain 3/40 • 2842KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24281999_24281999`  [🔗 chained]
*job 24281999 • idun-06-03 • A100 • epoch 575/1000 • chain 3/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24282106_24282106`  [💥 oom_killed]
*job 24282106 • idun-06-01 • A100 • 8.92h training • epoch 1000/1000 • chain 7/20 • 27KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004809 • MS-SSIM 0.9382 • PSNR 31.44 dB • LPIPS 0.4736 • FID 90.82 • KID 0.0889 ± 0.0092 • CMMD 0.3366
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp30_pixel_bravo_wd_mixup_ema_20260404-033007/checkpoint_latest.pt`
**Traceback excerpt:**
```
ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 52358bf0-35d4-4416-a838-21ad86b32d6a)')' thrown while requesting HEAD https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_pytorch_model.bin
...

real	548m47.841s
user	345m55.073s
sys	207m41.298s
[2026-04-08T10:57:49.988] error: Detected 1 oom_kill event in StepId=24282106.batch. Some of the step tasks have been OOM Killed.
```

#### `24282576_24282576`  [✅ completed]
*job 24282576 • idun-06-07 • A100 • 1.55h training • epoch 1000/1000 • chain 7/20 • 12KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002987 • MS-SSIM 0.9508 • PSNR 32.59 dB • LPIPS 0.5341 • FID 66.73 • KID 0.0529 ± 0.0091 • CMMD 0.2423
  - **latest** ckpt (26 samples): MSE 0.00505 • MS-SSIM 0.9355 • PSNR 31.50 dB • LPIPS 0.4936 • FID 87.29 • KID 0.0819 ± 0.0114 • CMMD 0.3418
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp29_pixel_bravo_mixup_1000_20260404-032937/checkpoint_latest.pt`

#### `24282670_24282670`  [🔗 chained]
*job 24282670 • idun-07-09 • A100 • epoch 1052/2000 • chain 4/40 • 76KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24282692_24282692`  [🔗 chained]
*job 24282692 • idun-06-07 • A100 • epoch 284/1000 • chain 0/20 • 168KB log*


#### `24284984_24284984`  [🔗 chained]
*job 24284984 • idun-06-07 • A100 • epoch 693/1000 • chain 4/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24285861_24285861`  [🔗 chained]
*job 24285861 • idun-06-04 • A100 • epoch 1252/2000 • chain 5/40 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24285928_24285928`  [🔗 chained]
*job 24285928 • idun-01-05 • H100 • epoch 804/1000 • chain 1/20 • 240KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_2_pixel_seg_67m_20260408-035801/checkpoint_latest.pt`

#### `24287503_24287503`  [🔗 chained]
*job 24287503 • idun-06-07 • A100 • epoch 810/1000 • chain 5/20 • 47KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24287525_24287525`  [🔗 chained]
*job 24287525 • idun-01-05 • H100 • epoch 1530/2000 • chain 6/40 • 126KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24287534_24287534`  [✅ completed]
*job 24287534 • idun-01-05 • H100 • 4.57h training • epoch 1000/1000 • chain 2/20 • 101KB log*

**Final test metrics:**
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_2_pixel_seg_67m_20260408-035801/checkpoint_latest.pt`

#### `24294144_24294144`  [💥 oom_killed]
*job 24294144 • idun-01-05 • H100 • 7.49h training • epoch 100/100 • chain 0/5 • 27KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005135 • MS-SSIM 0.9175 • PSNR 29.78 dB • LPIPS 0.4468 • FID 43.07 • KID 0.0256 ± 0.0066 • CMMD 0.1705
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
...

real	459m12.736s
user	355m48.694s
sys	114m53.337s
[2026-04-10T03:02:49.296] error: Detected 1 oom_kill event in StepId=24294144.batch. Some of the step tasks have been OOM Killed.
```

#### `24294145_24294145`  [✅ completed]
*job 24294145 • idun-01-04 • H100 • 9.83h training • epoch 100/100 • chain 0/5 • 29KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005167 • MS-SSIM 0.9229 • PSNR 30.53 dB • LPIPS 0.4719 • FID 90.71 • KID 0.1037 ± 0.0085 • CMMD 0.1817
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24294359_24294359`  [🔗 chained]
*job 24294359 • idun-07-08 • A100 • epoch 917/1000 • chain 6/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24294389_24294389`  [🔗 chained]
*job 24294389 • idun-06-05 • A100 • epoch 1737/2000 • chain 7/40 • 83KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24295517_24295517`  [✅ completed]
*job 24295517 • idun-07-08 • A100 • 9.23h training • epoch 1000/1000 • chain 7/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004286 • MS-SSIM 0.9386 • PSNR 32.64 dB • LPIPS 1.0223 • FID 81.96 • KID 0.0797 ± 0.0124 • CMMD 0.2929
  - **latest** ckpt (26 samples): MSE 0.004319 • MS-SSIM 0.9463 • PSNR 32.87 dB • LPIPS 0.7149 • FID 59.22 • KID 0.0509 ± 0.0116 • CMMD 0.2082
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427/checkpoint_latest.pt`

#### `24295987_24295987`  [🔗 chained]
*job 24295987 • idun-06-02 • A100 • epoch 1945/2000 • chain 8/40 • 84KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24297432_24297432`  [🔗 chained]
*job 24297432 • idun-06-02 • A100 • epoch 86/1000 • chain 0/20 • 24KB log*


#### `24297433_24297433`  [🔗 chained]
*job 24297433 • idun-06-04 • A100 • epoch 110/1000 • chain 0/20 • 33KB log*


#### `24297472_24297472`  [✅ completed]
*job 24297472 • idun-01-04 • H100 • 2.42h training • epoch 2000/2000 • chain 9/40 • 32KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006804 • MS-SSIM 0.9354 • PSNR 32.48 dB • LPIPS 0.9823 • FID 77.24 • KID 0.0716 ± 0.0134 • CMMD 0.2938
  - **latest** ckpt (26 samples): MSE 0.04487 • MS-SSIM 0.9130 • PSNR 31.89 dB • LPIPS 0.6392 • FID 48.08 • KID 0.0264 ± 0.0049 • CMMD 0.1795
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_2_pixel_bravo_combined_no_sa_17m_20260405-211255/checkpoint_latest.pt`

#### `24298186_24298186`  [🔗 chained]
*job 24298186 • idun-07-10 • A100 • epoch 213/1000 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24298187_24298187`  [🔗 chained]
*job 24298187 • idun-06-07 • A100 • epoch 171/1000 • chain 1/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24298538_24298538`  [✅ completed]
*job 24298538 • idun-06-01 • A100 • 11.51h training • epoch 100/100 • chain 0/5 • 94KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.006176 • MS-SSIM 0.9249 • PSNR 30.35 dB • LPIPS 0.4021 • FID 53.12 • KID 0.0379 ± 0.0059 • CMMD 0.1696
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24298633_24298633`  [🔗 chained]
*job 24298633 • idun-06-01 • A100 • epoch 119/1000 • chain 0/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24299033_24299033`  [🔗 chained]
*job 24299033 • idun-07-10 • A100 • epoch 316/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24299035_24299035`  [🔗 chained]
*job 24299035 • idun-06-07 • A100 • epoch 256/1000 • chain 2/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24299399_24299399`  [🔗 chained]
*job 24299399 • idun-01-03 • H100 • epoch 294/1000 • chain 1/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24299653_24299653`  [🔗 chained]
*job 24299653 • idun-01-03 • H100 • epoch 170/1000 • chain 0/20 • 71KB log*


#### `24299684_24299684`  [🔗 chained]
*job 24299684 • idun-06-05 • A100 • epoch 117/1000 • chain 0/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24299685_24299685`  [🔗 chained]
*job 24299685 • idun-07-08 • A100 • epoch 108/1000 • chain 0/20 • 97KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24301332_24301332`  [🔗 chained]
*job 24301332 • idun-01-04 • H100 • epoch 487/1000 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24301334_24301334`  [🔗 chained]
*job 24301334 • idun-01-03 • H100 • epoch 383/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24303660_24303660`  [🔗 chained]
*job 24303660 • idun-07-10 • A100 • epoch 404/1000 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24303687_24303687`  [🔗 chained]
*job 24303687 • idun-01-04 • H100 • epoch 90/500 • chain 0/20 • 21KB log*


#### `24303688_24303688`  [🔗 chained]
*job 24303688 • idun-07-10 • A100 • epoch 59/500 • chain 0/20 • 17KB log*


#### `24303775_24303775`  [🔗 chained]
*job 24303775 • idun-06-05 • A100 • epoch 114/500 • chain 0/20 • 31KB log*


#### `24303782_24303782`  [🔗 chained]
*job 24303782 • idun-06-06 • A100 • epoch 282/1000 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24303825_24303825`  [🔗 chained]
*job 24303825 • idun-06-05 • A100 • epoch 235/1000 • chain 1/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24303833_24303833`  [🔗 chained]
*job 24303833 • idun-07-10 • A100 • epoch 217/1000 • chain 1/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24304029_24304029`  [🔗 chained]
*job 24304029 • idun-01-04 • H100 • epoch 657/1000 • chain 4/20 • 89KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24304062_24304062`  [🔗 chained]
*job 24304062 • idun-01-04 • H100 • epoch 510/1000 • chain 4/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24304474_24304474`  [🔗 chained]
*job 24304474 • idun-06-06 • A100 • epoch 523/1000 • chain 3/20 • 41KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24304514_24304514`  [🔗 chained]
*job 24304514 • idun-06-05 • A100 • epoch 167/500 • chain 1/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24305858_24305858`  [🔗 chained]
*job 24305858 • idun-01-04 • H100 • epoch 149/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24305987_24305987`  [🔗 chained]
*job 24305987 • idun-01-04 • H100 • epoch 236/500 • chain 1/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24305993_24305993`  [🔗 chained]
*job 24305993 • idun-01-04 • H100 • epoch 453/1000 • chain 2/20 • 49KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24306093_24306093`  [🔗 chained]
*job 24306093 • idun-01-04 • H100 • epoch 408/1000 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24306132_24306132`  [🔗 chained]
*job 24306132 • idun-01-04 • H100 • epoch 389/1000 • chain 2/20 • 151KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24308606_24308606`  [🔗 chained]
*job 24308606 • idun-07-09 • A100 • epoch 760/1000 • chain 5/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24308625_24308625`  [🔗 chained]
*job 24308625 • idun-07-09 • A100 • epoch 590/1000 • chain 5/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24308833_24308833`  [🔗 chained]
*job 24308833 • idun-07-09 • A100 • epoch 634/1000 • chain 4/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24308908_24308908`  [🔗 chained]
*job 24308908 • idun-06-06 • A100 • epoch 243/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24308951_24308951`  [🔗 chained]
*job 24308951 • idun-07-10 • A100 • epoch 75/500 • chain 0/20 • 21KB log*


#### `24309326_24309326`  [🔗 chained]
*job 24309326 • idun-01-04 • H100 • epoch 624/1000 • chain 3/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24309327_24309327`  [🔗 chained]
*job 24309327 • idun-07-09 • A100 • epoch 345/500 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24309328_24309328`  [🔗 chained]
*job 24309328 • idun-06-01 • A100 • epoch 224/500 • chain 2/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24309382_24309382`  [🔗 chained]
*job 24309382 • idun-07-08 • A100 • epoch 518/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24309383_24309383`  [🔗 chained]
*job 24309383 • idun-06-04 • A100 • epoch 507/1000 • chain 3/20 • 107KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24310828_24310828`  [🔗 chained]
*job 24310828 • idun-07-08 • A100 • epoch 669/1000 • chain 6/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24310829_24310829`  [🔗 chained]
*job 24310829 • idun-06-05 • A100 • epoch 872/1000 • chain 6/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24310900_24310900`  [🔗 chained]
*job 24310900 • idun-06-06 • A100 • epoch 752/1000 • chain 5/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24311036_24311036`  [🔗 chained]
*job 24311036 • idun-06-05 • A100 • epoch 120/1000 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24311411_24311411`  [🔗 chained]
*job 24311411 • idun-06-06 • A100 • epoch 324/500 • chain 3/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24311744_24311744`  [🔗 chained]
*job 24311744 • idun-01-04 • H100 • epoch 186/500 • chain 1/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630/checkpoint_latest.pt`

#### `24313294_24313294`  [🔗 chained]
*job 24313294 • idun-07-10 • A100 • epoch 75/500 • chain 0/20 • 19KB log*


#### `24313642_24313642`  [🔗 chained]
*job 24313642 • idun-06-07 • A100 • epoch 736/1000 • chain 4/20 • 31KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24313728_24313728`  [🔗 chained]
*job 24313728 • idun-06-05 • A100 • epoch 458/500 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24313866_24313866`  [🔗 chained]
*job 24313866 • idun-01-04 • H100 • epoch 298/500 • chain 3/20 • 19KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24313917_24313917`  [🔗 chained]
*job 24313917 • idun-07-10 • A100 • epoch 628/1000 • chain 4/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24313978_24313978`  [🔗 chained]
*job 24313978 • idun-06-04 • A100 • epoch 624/1000 • chain 4/20 • 107KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24313980_24313980`  [🔗 chained]
*job 24313980 • idun-06-04 • A100 • epoch 753/1000 • chain 7/20 • 20KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24314133_24314133`  [🔗 chained]
*job 24314133 • idun-07-09 • A100 • epoch 976/1000 • chain 7/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24314432_24314432`  [🔗 chained]
*job 24314432 • idun-06-01 • A100 • epoch 416/800 • chain 0/20 • 359KB log*


#### `24314447_24314447`  [🔗 chained]
*job 24314447 • idun-07-09 • A100 • epoch 444/800 • chain 0/20 • 387KB log*


#### `24314507_24314507`  [🔗 chained]
*job 24314507 • idun-07-10 • A100 • epoch 863/1000 • chain 6/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24314543_24314543`  [🔗 chained]
*job 24314543 • idun-07-10 • A100 • epoch 231/1000 • chain 1/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24314556_24314556`  [🔗 chained]
*job 24314556 • idun-07-10 • A100 • epoch 380/500 • chain 4/20 • 18KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2_rflow_bridge_restoration_20260413-004114/checkpoint_latest.pt`

#### `24314691_24314691`  [🔗 chained]
*job 24314691 • idun-07-10 • A100 • epoch 265/500 • chain 2/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3b_rflow_bridge_noise_restoration_17m_20260414-044630/checkpoint_latest.pt`

#### `24314954_24314954`  [🔗 chained]
*job 24314954 • idun-06-07 • A100 • epoch 115/500 • chain 0/20 • 49KB log*


#### `24314955_24314955`  [🔗 chained]
*job 24314955 • idun-06-07 • A100 • epoch 277/500 • chain 0/20 • 107KB log*


#### `24315002_24315002`  [🔗 chained]
*job 24315002 • idun-06-07 • A100 • epoch 195/500 • chain 1/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_2b_rflow_bridge_restoration_17m_20260415-122714/checkpoint_latest.pt`

#### `24315035_24315035`  [💥 oom_killed]
*job 24315035 • idun-07-09 • A100 • 6.81h training • epoch 24/500 • chain 0/20 • 8KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
```

#### `24315037_24315037`  [💥 oom_killed]
*job 24315037 • idun-07-09 • A100 • 6.86h training • epoch 24/500 • chain 0/20 • 8KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "
```

#### `24315038_24315038`  [💥 oom_killed]
*job 24315038 • idun-01-04 • H100 • 5.6h training • epoch 24/500 • chain 0/20 • 9KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/monai/networks/nets/diffusion_model_unet.py", line 1803, in forward
    h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

#### `24315064_24315064`  [⚠️ truncated]
*job 24315064 • idun-07-09 • A100 • epoch 59/500 • chain 0/20 • 15KB log*


#### `24315065_24315065`  [⚠️ truncated]
*job 24315065 • idun-07-09 • A100 • epoch 49/500 • chain 0/20 • 20KB log*


#### `24315083_24315083`  [⚠️ truncated]
*job 24315083 • idun-06-01 • A100 • epoch 74/500 • chain 0/20 • 18KB log*


#### `24315089_24315089`  [🔗 chained]
*job 24315089 • idun-01-04 • H100 • epoch 907/1000 • chain 5/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24315090_24315090`  [✅ completed]
*job 24315090 • idun-01-04 • H100 • 4.26h training • epoch 500/500 • chain 4/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004296 • MS-SSIM 0.9514 • PSNR 33.26 dB • LPIPS 1.1520 • FID 143.44 • KID 0.1907 ± 0.0132 • CMMD 0.2380
  - **latest** ckpt (26 samples): MSE 0.004361 • MS-SSIM 0.9520 • PSNR 33.21 dB • LPIPS 1.2162 • FID 140.32 • KID 0.1903 ± 0.0110 • CMMD 0.2237
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304/checkpoint_latest.pt`

#### `24315096_24315096`  [⚠️ truncated]
*job 24315096 • idun-06-05 • A100 • epoch 322/500 • chain 4/20 • 10KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/exp33_3_rflow_bridge_noise_restoration_20260413-014148/checkpoint_latest.pt`

#### `24315097_24315097`  [🔗 chained]
*job 24315097 • idun-07-10 • A100 • epoch 739/1000 • chain 5/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24315098_24315098`  [🔗 chained]
*job 24315098 • idun-06-07 • A100 • epoch 741/1000 • chain 5/20 • 100KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24315099_24315099`  [🔗 chained]
*job 24315099 • idun-06-01 • A100 • epoch 839/1000 • chain 8/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24315101_24315101`  [✅ completed]
*job 24315101 • idun-06-07 • A100 • 2.58h training • epoch 1000/1000 • chain 8/20 • 14KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002314 • MS-SSIM 0.9713 • PSNR 35.45 dB • LPIPS 0.5361 • FID 67.76 • KID 0.0583 ± 0.0039 • CMMD 0.2307
  - **latest** ckpt (26 samples): MSE 0.005111 • MS-SSIM 0.9439 • PSNR 32.01 dB • LPIPS 0.3772 • FID 41.15 • KID 0.0260 ± 0.0042 • CMMD 0.1980
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt`

#### `24315111_24315111`  [⚠️ truncated]
*job 24315111 • idun-06-07 • A100 • chain 1/20 • 5KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_2d/restoration/exp33_1_irsde_restoration_20260415-194624/checkpoint_latest.pt`

#### `24315115_24315115`  [⚠️ truncated]
*job 24315115 • idun-07-09 • A100 • chain 1/20 • 2KB log*


#### `24315535_24315535`  [🔗 chained]
*job 24315535 • idun-07-09 • A100 • epoch 341/1000 • chain 2/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24315536_24315536`  [🔗 chained]
*job 24315536 • idun-07-09 • A100 • epoch 973/1000 • chain 7/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`

#### `24315690_24315690`  [🔗 chained]
*job 24315690 • idun-07-09 • A100 • epoch 221/500 • chain 1/20 • 42KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24315691_24315691`  [✅ completed]
*job 24315691 • idun-07-09 • A100 • 11.0h training • epoch 500/500 • chain 1/20 • 74KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007023 • MS-SSIM 0.9283 • PSNR 30.95 dB • LPIPS 0.8128 • FID 195.38 • KID 0.2653 ± 0.0182 • CMMD 0.4760
  - **latest** ckpt (26 samples): MSE 0.0241 • MS-SSIM 0.9151 • PSNR 29.89 dB • LPIPS 0.4458 • FID 178.97 • KID 0.2269 ± 0.0159 • CMMD 0.4156
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_s_p4_20260416-005333/checkpoint_latest.pt`

#### `24315963_24315963`  [🔗 chained]
*job 24315963 • idun-06-06 • A100 • epoch 116/1000 • chain 0/20 • 48KB log*


#### `24315964_24315964`  [🔗 chained]
*job 24315964 • idun-06-01 • A100 • epoch 277/1000 • chain 0/20 • 116KB log*


#### `24316690_24316690`  [✅ completed]
*job 24316690 • idun-01-03 • H100 • 6.43h training • epoch 1000/1000 • chain 6/20 • 30KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002472 • MS-SSIM 0.9638 • PSNR 34.62 dB • LPIPS 0.7534 • FID 108.28 • KID 0.1273 ± 0.0080 • CMMD 0.2461
  - **latest** ckpt (26 samples): MSE 0.00512 • MS-SSIM 0.9314 • PSNR 30.77 dB • LPIPS 0.4605 • FID 65.81 • KID 0.0652 ± 0.0071 • CMMD 0.1940
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_156_pixel_dual_20260412-143252/checkpoint_latest.pt`

#### `24316691_24316691`  [✅ completed]
*job 24316691 • idun-01-04 • H100 • 11.56h training • epoch 500/500 • chain 0/20 • 191KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.05237 • MS-SSIM 0.4655 • PSNR 21.96 dB • LPIPS 1.7800 • FID 346.41 • KID 0.4726 ± 0.0159 • CMMD 0.6346
  - **latest** ckpt (26 samples): MSE 0.0537 • MS-SSIM 0.4721 • PSNR 22.17 dB • LPIPS 1.7801 • FID 349.33 • KID 0.4697 ± 0.0127 • CMMD 0.6382

#### `24316692_24316692`  [🔗 chained]
*job 24316692 • idun-07-09 • A100 • epoch 148/500 • chain 0/20 • 41KB log*


#### `24316927_24316927`  [🔗 chained]
*job 24316927 • idun-07-09 • A100 • epoch 849/1000 • chain 6/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24316958_24316958`  [🔗 chained]
*job 24316958 • idun-01-03 • H100 • epoch 914/1000 • chain 6/20 • 143KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24316963_24316963`  [🔗 chained]
*job 24316963 • idun-01-03 • H100 • epoch 965/1000 • chain 9/20 • 30KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24316991_24316991`  [🔗 chained]
*job 24316991 • idun-06-05 • A100 • epoch 459/1000 • chain 3/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24316992_24316992`  [💥 oom_killed]
*job 24316992 • idun-06-05 • A100 • 2.71h training • epoch 1000/1000 • chain 8/20 • 15KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.007728 • MS-SSIM 0.9022 • PSNR 28.98 dB • LPIPS 0.3477 • FID 44.08 • KID 0.0172 ± 0.0039 • CMMD 0.1499
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
/var/slurm_spool/job24316992/slurm_script: line 91: 139790 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=1000 training.learning_rate=1e-4 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	174m47.036s
user	113m35.289s
sys	64m20.278s
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/multiprocessing/resource_sharer.py", line 138, in _serve
[2026-04-17T09:06:11.316] error: Detected 1 oom_kill event in StepId=24316992.batch. Some of the step tasks have been OOM Killed.
```

#### `24316998_24316998`  [🔗 chained]
*job 24316998 • idun-06-05 • A100 • epoch 336/500 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24317029_24317029`  [🔗 chained]
*job 24317029 • idun-06-06 • A100 • epoch 231/1000 • chain 1/20 • 46KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_l_p4_gc_20260416-203245/checkpoint_latest.pt`

#### `24317062_24317062`  [🔗 chained]
*job 24317062 • idun-06-01 • A100 • epoch 558/1000 • chain 1/20 • 88KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_s_p4_20260416-210331/checkpoint_latest.pt`

#### `24318602_24318602`  [🔗 chained]
*job 24318602 • idun-01-04 • H100 • epoch 376/500 • chain 1/20 • 68KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260417-031657/checkpoint_latest.pt`

#### `24318666_24318666`  [🔗 chained]
*job 24318666 • idun-06-04 • A100 • epoch 965/1000 • chain 7/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24318712_24318712`  [💥 oom_killed]
*job 24318712 • idun-07-08 • A100 • 9.49h training • epoch 1000/1000 • chain 7/20 • 79KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005326 • MS-SSIM 0.9108 • PSNR 29.46 dB • LPIPS 0.4418 • FID 44.29 • KID 0.0225 ± 0.0053 • CMMD 0.1562
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24318712/slurm_script: line 93: 1545972 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=1000 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.perceptual_weight=0.1 training.perceptual_max_timestep=250 training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	585m57.251s
user	424m46.393s
sys	165m35.549s
[2026-04-18T09:15:19.614] error: Detected 1 oom_kill event in StepId=24318712.batch. Some of the step tasks have been OOM Killed.
```

#### `24318753_24318753`  [✅ completed]
*job 24318753 • idun-07-08 • A100 • 5.21h training • epoch 1000/1000 • chain 10/20 • 17KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.001339 • MS-SSIM 0.9875 • PSNR 37.40 dB • LPIPS 0.2908 • FID 46.57 • KID 0.0262 ± 0.0049 • CMMD 0.1679
  - **latest** ckpt (26 samples): MSE 0.001936 • MS-SSIM 0.9890 • PSNR 37.72 dB • LPIPS 0.2615 • FID 47.07 • KID 0.0265 ± 0.0064 • CMMD 0.1642
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_156_pixel_bravo_20260411-024806/checkpoint_latest.pt`

#### `24318757_24318757`  [🔗 chained]
*job 24318757 • idun-07-08 • A100 • epoch 570/1000 • chain 4/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24318809_24318809`  [🔗 chained]
*job 24318809 • idun-07-08 • A100 • epoch 441/500 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24318831_24318831`  [🔗 chained]
*job 24318831 • idun-06-06 • A100 • epoch 116/500 • chain 0/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24318832_24318832`  [🔗 chained]
*job 24318832 • idun-06-05 • A100 • epoch 114/500 • chain 0/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24318948_24318948`  [🔗 chained]
*job 24318948 • idun-01-04 • H100 • epoch 417/1000 • chain 2/20 • 82KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_l_p4_gc_20260416-203245/checkpoint_latest.pt`

#### `24318972_24318972`  [🔗 chained]
*job 24318972 • idun-07-08 • A100 • epoch 802/1000 • chain 2/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1000_mamba_s_p4_20260416-210331/checkpoint_latest.pt`

#### `24319138_24319138`  [✅ completed]
*job 24319138 • idun-06-01 • A100 • 9.11h training • epoch 500/500 • chain 2/20 • 80KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.006321 • MS-SSIM 0.9198 • PSNR 30.76 dB • LPIPS 1.4442 • FID 229.45 • KID 0.2909 ± 0.0148 • CMMD 0.5112
  - **latest** ckpt (26 samples): MSE 0.006731 • MS-SSIM 0.9150 • PSNR 30.61 dB • LPIPS 1.4622 • FID 236.27 • KID 0.3020 ± 0.0156 • CMMD 0.4990
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260417-031657/checkpoint_latest.pt`

#### `24319180_24319180`  [✅ completed]
*job 24319180 • idun-06-05 • A100 • 3.6h training • epoch 1000/1000 • chain 8/20 • 19KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.00767 • MS-SSIM 0.9081 • PSNR 29.34 dB • LPIPS 0.4096 • FID 46.48 • KID 0.0275 ± 0.0078 • CMMD 0.1760
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt`

#### `24319336_24319336`  [✅ completed]
*job 24319336 • idun-01-03 • H100 • 3.86h training • epoch 500/500 • chain 4/20 • 66KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.004804 • MS-SSIM 0.9343 • PSNR 31.55 dB • LPIPS 1.4403 • FID 220.82 • KID 0.3037 ± 0.0172 • CMMD 0.5052
  - **latest** ckpt (26 samples): MSE 0.0052 • MS-SSIM 0.9106 • PSNR 29.34 dB • LPIPS 0.3282 • FID 161.57 • KID 0.2066 ± 0.0159 • CMMD 0.3652
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_mamba_l_p4_gc_20260416-005333/checkpoint_latest.pt`

#### `24319337_24319337`  [🔗 chained]
*job 24319337 • idun-07-08 • A100 • epoch 680/1000 • chain 5/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24319346_24319346`  [🔗 chained]
*job 24319346 • idun-07-08 • A100 • epoch 223/500 • chain 1/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24319503_24319503`  [🔗 chained]
*job 24319503 • idun-06-07 • A100 • epoch 229/500 • chain 1/20 • 71KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24323474_24323474`  [🔗 chained]
*job 24323474 • idun-01-03 • H100 • epoch 391/500 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24323475_24323475`  [🔗 chained]
*job 24323475 • idun-06-04 • A100 • epoch 798/1000 • chain 6/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24323478_24323478`  [🔗 chained]
*job 24323478 • idun-06-04 • A100 • epoch 345/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24324706_24324706`  [🔗 chained]
*job 24324706 • idun-06-07 • A100 • epoch 114/1000 • chain 0/20 • 45KB log*


#### `24324707_24324707`  [🔗 chained]
*job 24324707 • idun-06-07 • A100 • epoch 274/1000 • chain 0/20 • 104KB log*


#### `24324708_24324708`  [🔗 chained]
*job 24324708 • idun-06-01 • A100 • epoch 162/500 • chain 0/20 • 45KB log*


#### `24324709_24324709`  [🔗 chained]
*job 24324709 • idun-06-01 • A100 • epoch 413/500 • chain 0/20 • 123KB log*


#### `24325059_24325059`  [✅ completed]
*job 24325059 • idun-06-04 • A100 • 11.19h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005893 • MS-SSIM 0.9270 • PSNR 30.59 dB • LPIPS 0.3724 • FID 69.58 • KID 0.0637 ± 0.0078 • CMMD 0.1682
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt`

#### `24325060_24325060`  [🔗 chained]
*job 24325060 • idun-07-08 • A100 • epoch 908/1000 • chain 7/20 • 23KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24325084_24325084`  [🔗 chained]
*job 24325084 • idun-06-04 • A100 • epoch 460/500 • chain 3/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24325324_24325324`  [💥 oom_killed]
*job 24325324 • idun-08-01 • H100 • 11.04h training • epoch 150/150 • chain 0/20 • 38KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004514 • MS-SSIM 0.9310 • PSNR 30.63 dB • LPIPS 0.4476 • FID 96.02 • KID 0.1156 ± 0.0143 • CMMD 0.2001
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24325324/slurm_script: line 101: 458587 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=150 training.learning_rate=2e-5 training.warmup_epochs=10 training.eta_min=1e-7 training.gradient_clip_norm=0.5 training.perceptual_weight=0.5 'training.perceptual_t_schedule=[0.05,0.20,0.70]' training.focal_frequency_weight=0.7 'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	674m32.944s
user	518m29.499s
sys	169m25.572s
[2026-04-20T12:16:12.543] error: Detected 2 oom_kill events in StepId=24325324.batch. Some of the step tasks have been OOM Killed.
```

#### `24325358_24325358`  [🔗 chained]
*job 24325358 • idun-01-03 • H100 • epoch 299/1000 • chain 1/20 • 66KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24325359_24325359`  [🔗 chained]
*job 24325359 • idun-06-07 • A100 • epoch 548/1000 • chain 1/20 • 83KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24325360_24325360`  [🔗 chained]
*job 24325360 • idun-06-07 • A100 • epoch 324/500 • chain 1/20 • 50KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/checkpoint_latest.pt`

#### `24325363_24325363`  [✅ completed]
*job 24325363 • idun-07-08 • A100 • 3.14h training • epoch 500/500 • chain 1/20 • 33KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.03708 • MS-SSIM 0.4959 • PSNR 22.78 dB • LPIPS 1.7538 • FID 334.70 • KID 0.4441 ± 0.0148 • CMMD 0.6253
  - **latest** ckpt (26 samples): MSE 0.04212 • MS-SSIM 0.5510 • PSNR 23.96 dB • LPIPS 1.7220 • FID 349.43 • KID 0.4688 ± 0.0136 • CMMD 0.6237
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_3_mamba_5s_p4_20260419-162046/checkpoint_latest.pt`

#### `24325380_24325380`  [✅ completed]
*job 24325380 • idun-06-01 • A100 • 9.15h training • epoch 1000/1000 • chain 8/20 • 25KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.01025 • MS-SSIM 0.9068 • PSNR 29.28 dB • LPIPS 0.3977 • FID 48.22 • KID 0.0284 ± 0.0058 • CMMD 0.1632
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt`

#### `24325381_24325381`  [✅ completed]
*job 24325381 • idun-06-01 • A100 • 4.11h training • epoch 500/500 • chain 4/20 • 18KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.005817 • MS-SSIM 0.9255 • PSNR 30.55 dB • LPIPS 0.4031 • FID 68.98 • KID 0.0631 ± 0.0085 • CMMD 0.1734
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt`

#### `24326102_24326102`  [🔗 chained]
*job 24326102 • idun-01-03 • H100 • epoch 484/1000 • chain 2/20 • 58KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24326194_24326194`  [💥 oom_killed]
*job 24326194 • idun-01-05 • H100 • chain 2/20 • 6KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
  File "/cluster/work/modestas/AIS4900_master/src/medgen/models/mamba_diff.py", line 528, in forward
    x = self.encoder_stages[i](x, c)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*a
```

#### `24326398_24326398`  [🔗 chained]
*job 24326398 • idun-08-01 • H100 • epoch 940/1000 • chain 0/20 • 101KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24327639_24327639`  [❌ crashed]
*job 24327639 • idun-07-08 • A100 • epoch 352/500 • chain 0/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/checkpoint_latest.pt`
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
         ^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_2_mamba_5s_p2_20260419-154607/regional_losses.json'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[2026-04-21T02:02:40.652] error: *** JOB 24327639 ON idun-07-08 CANCELLED AT 2026-04-21T02:02:40 DUE to SIGNAL Terminated ***
```

#### `24327729_24327729`  [🔗 chained]
*job 24327729 • idun-01-03 • H100 • epoch 173/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327730_24327730`  [🔗 chained]
*job 24327730 • idun-01-03 • H100 • epoch 171/500 • chain 0/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327732_24327732`  [🔗 chained]
*job 24327732 • idun-08-01 • H100 • epoch 167/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327859_24327859`  [🔗 chained]
*job 24327859 • idun-08-01 • H100 • epoch 142/500 • chain 0/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327860_24327860`  [🔗 chained]
*job 24327860 • idun-01-03 • H100 • epoch 173/500 • chain 0/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327861_24327861`  [🔗 chained]
*job 24327861 • idun-07-09 • A100 • epoch 109/500 • chain 0/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327864_24327864`  [🔗 chained]
*job 24327864 • idun-07-10 • A100 • epoch 109/500 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt`

#### `24327887_24327887`  [🔗 chained]
*job 24327887 • idun-07-10 • A100 • epoch 590/1000 • chain 3/20 • 34KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24327896_24327896`  [✅ completed]
*job 24327896 • idun-07-09 • A100 • 2.97h training • epoch 1000/1000 • chain 1/20 • 24KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.01837 • MS-SSIM 0.9197 • PSNR 30.02 dB • LPIPS 0.3322 • FID 144.59 • KID 0.1791 ± 0.0106 • CMMD 0.3520
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_0_1000_mamba_s_p4_20260419-154005/checkpoint_latest.pt`

#### `24328879_24328879`  [🔗 chained]
*job 24328879 • idun-07-10 • A100 • epoch 281/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24329911_24329911`  [🔗 chained]
*job 24329911 • idun-01-03 • H100 • epoch 343/500 • chain 1/20 • 37KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24330209_24330209`  [🔗 chained]
*job 24330209 • idun-07-09 • A100 • epoch 274/500 • chain 1/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24330471_24330471`  [🔗 chained]
*job 24330471 • idun-07-08 • A100 • epoch 251/500 • chain 1/20 • 25KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24330559_24330559`  [🔗 chained]
*job 24330559 • idun-01-03 • H100 • epoch 345/500 • chain 1/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_2_scoreaug_structure_20260421-105310/checkpoint_latest.pt`

#### `24330580_24330580`  [🔗 chained]
*job 24330580 • idun-01-03 • H100 • epoch 282/500 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`

#### `24330583_24330583`  [🔗 chained]
*job 24330583 • idun-06-05 • A100 • epoch 704/1000 • chain 4/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24330584_24330584`  [🔗 chained]
*job 24330584 • idun-08-01 • H100 • epoch 250/500 • chain 1/20 • 29KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24330608_24330608`  [🔗 chained]
*job 24330608 • idun-06-05 • A100 • epoch 399/500 • chain 2/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24330613_24330613`  [💥 oom_killed]
*job 24330613 • idun-06-01 • A100 • 6.98h training • epoch 60/60 • chain 0/20 • 20KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.006013 • MS-SSIM 0.9343 • PSNR 31.37 dB • LPIPS 0.3986 • FID 90.80 • KID 0.1022 ± 0.0114 • CMMD 0.1978
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24330613/slurm_script: line 109: 3109653 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=60 training.learning_rate=1e-5 training.warmup_epochs=3 training.eta_min=1e-7 training.gradient_clip_norm=0.5 training.perceptual_weight=0.5 'training.perceptual_t_schedule=[0.05,0.20,0.70]' training.focal_frequency_weight=0.7 'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' 'training.mse_t_schedule=[0.05,0.15,1.0]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	428m20.930s
user	277m17.674s
sys	156m56.871s
[2026-04-22T16:29:00.437] error: Detected 1 oom_kill event in StepId=24330613.batch. Some of the step tasks have been OOM Killed.
```

#### `24330636_24330636`  [🔗 chained]
*job 24330636 • idun-06-05 • A100 • epoch 456/500 • chain 2/20 • 27KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24330682_24330682`  [🔗 chained]
*job 24330682 • idun-06-01 • A100 • epoch 390/500 • chain 2/20 • 28KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24332507_24332507`  [🔗 chained]
*job 24332507 • idun-07-09 • A100 • epoch 361/500 • chain 2/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24332650_24332650`  [🔗 chained]
*job 24332650 • idun-06-05 • A100 • epoch 99/100 • chain 0/20 • 24KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt`

#### `24333401_24333401`  [🔗 chained]
*job 24333401 • idun-09-02 • H200 • epoch 462/500 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`

#### `24333402_24333402`  [💥 oom_killed]
*job 24333402 • idun-09-02 • H200 • 10.27h training • epoch 500/500 • chain 2/20 • 40KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.007519 • MS-SSIM 0.9338 • PSNR 31.05 dB • LPIPS 0.3894 • FID 54.39 • KID 0.0362 ± 0.0069 • CMMD 0.1816
  - **latest** ckpt (26 samples): MSE 0.007733 • MS-SSIM 0.9241 • PSNR 30.26 dB • LPIPS 0.4485
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_2_scoreaug_structure_20260421-105310/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24333402/slurm_script: line 96: 19389 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=500 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.score_aug.enabled=true 'training.scoreaug_t_schedule=[0.70,0.85,1.0]' training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	626m19.223s
user	459m20.353s
sys	174m16.112s
[2026-04-23T08:59:57.217] error: Detected 1 oom_kill event in StepId=24333402.batch. Some of the step tasks have been OOM Killed.
```

#### `24333414_24333414`  [🔗 chained]
*job 24333414 • idun-09-02 • H200 • epoch 913/1000 • chain 5/20 • 54KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24333461_24333461`  [🔗 chained]
*job 24333461 • idun-09-02 • H200 • epoch 429/500 • chain 2/20 • 36KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24334406_24334406`  [✅ completed]
*job 24334406 • idun-09-02 • H200 • 6.72h training • epoch 500/500 • chain 3/20 • 28KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003164 • MS-SSIM 0.9553 • PSNR 33.20 dB • LPIPS 0.5584 • FID 68.42 • KID 0.0458 ± 0.0054 • CMMD 0.2512
  - **latest** ckpt (26 samples): MSE 0.002815 • MS-SSIM 0.9637 • PSNR 33.73 dB • LPIPS 0.4990 • FID 70.27 • KID 0.0545 ± 0.0071 • CMMD 0.2545
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_1_augment_medium_uniform_20260421-004136/checkpoint_latest.pt`

#### `24334407_24334407`  [✅ completed]
*job 24334407 • idun-09-02 • H200 • 7.41h training • epoch 500/500 • chain 3/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003855 • MS-SSIM 0.9374 • PSNR 31.45 dB • LPIPS 0.4567 • FID 50.17 • KID 0.0347 ± 0.0062 • CMMD 0.1746
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_4_augment_mri_detail_20260421-042913/checkpoint_latest.pt`

#### `24334408_24334408`  [✅ completed]
*job 24334408 • idun-09-02 • H200 • 2.94h training • epoch 500/500 • chain 3/20 • 19KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.004374 • MS-SSIM 0.9423 • PSNR 31.97 dB • LPIPS 0.4540 • FID 57.25 • KID 0.0397 ± 0.0082 • CMMD 0.1670
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt`

#### `24334547_24334547`  [✅ completed]
*job 24334547 • idun-08-01 • H100 • 8.96h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334549_24334549`  [✅ completed]
*job 24334549 • idun-09-02 • H200 • 8.23h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334588_24334588`  [✅ completed]
*job 24334588 • idun-09-02 • H200 • 7.66h training • epoch 80/80 • chain 0/20 • 17KB log*


#### `24334610_24334610`  [✅ completed]
*job 24334610 • idun-09-02 • H200 • 9.24h training • epoch 500/500 • chain 3/20 • 35KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003243 • MS-SSIM 0.9334 • PSNR 32.08 dB • LPIPS 0.6479 • FID 125.83 • KID 0.1346 ± 0.0108 • CMMD 0.4178
  - **latest** ckpt (26 samples): MSE 0.004973 • MS-SSIM 0.9414 • PSNR 31.65 dB • LPIPS 0.4449 • FID 45.36 • KID 0.0261 ± 0.0048 • CMMD 0.1611
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_1_scoreaug_detail_20260421-093244/checkpoint_latest.pt`

#### `24334832_24334832`  [✅ completed]
*job 24334832 • idun-09-02 • H200 • 0.11h training • epoch 100/100 • chain 1/20 • 10KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.003924 • MS-SSIM 0.9236 • PSNR 30.26 dB • LPIPS 0.3856 • FID 81.49 • KID 0.0880 ± 0.0097 • CMMD 0.1803
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp40_patchgan_20260422-220017/checkpoint_latest.pt`

#### `24334911_24334911`  [💥 oom_killed]
*job 24334911 • idun-09-02 • H200 • 2.54h training • epoch 500/500 • chain 3/20 • 16KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.003958 • MS-SSIM 0.9559 • PSNR 33.28 dB • LPIPS 0.4711 • FID 60.03 • KID 0.0448 ± 0.0114 • CMMD 0.1609
  - **latest** ckpt (26 samples): MSE 0.004998 • MS-SSIM 0.9469 • PSNR 32.46 dB • LPIPS 0.4508
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp35_3_scoreaug_uniform_20260421-115605/checkpoint_latest.pt`
**Traceback excerpt:**
```
/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
/var/slurm_spool/job24334911/slurm_script: line 95: 622642 Killed                  python -m medgen.scripts.train --config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160 training.epochs=500 training.learning_rate=1e-5 training.warmup_epochs=10 training.gradient_clip_norm=0.5 training.score_aug.enabled=true training.name=${EXP_NAME}_ ${CHAIN_ARGS}

real	161m16.088s
user	118m51.179s
sys	43m51.984s
[2026-04-23T14:29:31.120] error: Detected 1 oom_kill event in StepId=24334911.batch. Some of the step tasks have been OOM Killed.
```

#### `24334912_24334912`  [✅ completed]
*job 24334912 • idun-09-02 • H200 • 4.73h training • epoch 500/500 • chain 3/20 • 23KB log*

**Final test metrics:**
  - **best** ckpt (26 samples): MSE 0.002311 • MS-SSIM 0.9599 • PSNR 33.31 dB • LPIPS 0.5531 • FID 70.99 • KID 0.0525 ± 0.0082 • CMMD 0.2511
  - **latest** ckpt (26 samples): MSE 0.002523 • MS-SSIM 0.9560 • PSNR 33.11 dB • LPIPS 0.5534 • FID 73.28 • KID 0.0540 ± 0.0050 • CMMD 0.2858
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp36_3_augment_mri_uniform_20260421-120440/checkpoint_latest.pt`

#### `24334913_24334913`  [✅ completed]
*job 24334913 • idun-09-02 • H200 • 4.97h training • epoch 1000/1000 • chain 6/20 • 31KB log*

**Final test metrics:**
  - **latest** ckpt (26 samples): MSE 0.008194 • MS-SSIM 0.9118 • PSNR 29.21 dB • LPIPS 0.2896 • FID 148.60 • KID 0.1831 ± 0.0131 • CMMD 0.3263
**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24336018_24336018`  [🔗 chained]
*job 24336018 • idun-01-03 • H100 • epoch 171/1000 • chain 0/20 • 41KB log*


#### `24336117_24336117`  [✅ completed]
*job 24336117 • idun-09-02 • H200 • 6.86h training • epoch 80/80 • chain 0/20 • 18KB log*


#### `24336401_24336401`  [⚠️ truncated]
*job 24336401 • idun-01-03 • H100 • chain 0/20 • 2KB log*


#### `24336457_24336457`  [🔗 chained]
*job 24336457 • idun-08-01 • H100 • epoch 344/1000 • chain 1/20 • 35KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24337305_24337305`  [⚠️ truncated]
*job 24337305 • idun-07-08 • A100 • chain 0/20 • 1KB log*


#### `24338909_24338909`  [🔗 chained]
*job 24338909 • idun-09-02 • H200 • epoch 525/1000 • chain 2/20 • 38KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24341731_24341731`  [⚠️ truncated]
*job 24341731 • idun-09-02 • H200 • chain 0/20 • 1KB log*


#### `24341732_24341732`  [✅ completed]
*job 24341732 • idun-09-02 • H200 • 9.38h training • epoch 100/100 • chain 0/20 • 20KB log*


#### `24341733_24341733`  [⚠️ truncated]
*job 24341733 • idun-08-01 • H100 • chain 0/20 • 1KB log*


#### `24341756_24341756`  [🔗 chained]
*job 24341756 • idun-08-01 • H100 • epoch 166/500 • chain 0/20 • 39KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341757_24341757`  [🔗 chained]
*job 24341757 • idun-09-02 • H200 • epoch 178/500 • chain 0/20 • 160KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341758_24341758`  [🔗 chained]
*job 24341758 • idun-06-06 • A100 • epoch 113/1000 • chain 0/20 • 100KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24341759_24341759`  [🔗 chained]
*job 24341759 • idun-08-01 • H100 • epoch 127/1000 • chain 0/20 • 51KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp34_1_1000_mamba_l_p4_gc_20260419-152401/checkpoint_latest.pt`

#### `24341775_24341775`  [🔗 chained]
*job 24341775 • idun-07-08 • A100 • epoch 108/500 • chain 0/20 • 105KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341776_24341776`  [🔗 chained]
*job 24341776 • idun-07-10 • A100 • epoch 107/500 • chain 0/20 • 96KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341777_24341777`  [🔗 chained]
*job 24341777 • idun-09-02 • H200 • epoch 180/500 • chain 0/20 • 139KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341778_24341778`  [🔗 chained]
*job 24341778 • idun-06-04 • A100 • epoch 117/500 • chain 0/20 • 102KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24341791_24341791`  [🔗 chained]
*job 24341791 • idun-06-04 • A100 • epoch 645/1000 • chain 3/20 • 26KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260423-180215/checkpoint_latest.pt`

#### `24342024_24342024`  [⚠️ truncated]
*job 24342024 • idun-07-08 • A100 • epoch 200/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp45_stack_lpips_scoreaug_20260425-031502/checkpoint_latest.pt`

#### `24342052_24342052`  [⚠️ truncated]
*job 24342052 • idun-09-02 • H200 • epoch 230/500 • chain 1/20 • 48KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_extended_pixel_bravo_lpips_lowt_20260425-034324/checkpoint_latest.pt`

#### `24342065_24342065`  [⚠️ truncated]
*job 24342065 • idun-06-04 • A100 • epoch 30/500 • chain 0/20 • 11KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342066_24342066`  [❌ crashed]
*job 24342066 • idun-06-06 • A100 • chain 0/20 • 2KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train.py", line 118, in main
    _train_3d(cfg)
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	0m20.443s
user	0m6.265s
sys	0m4.810s
```

#### `24342067_24342067`  [⚠️ truncated]
*job 24342067 • idun-06-06 • A100 • epoch 27/500 • chain 0/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342068_24342068`  [⚠️ truncated]
*job 24342068 • idun-01-04 • H100 • epoch 24/500 • chain 0/20 • 21KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt`

#### `24342079_24342079`  [⚠️ truncated]
*job 24342079 • idun-07-08 • A100 • epoch 124/1000 • chain 1/20 • 13KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp46_mamba_l_lpips_lowt_20260425-042726/checkpoint_latest.pt`

#### `24342089_24342089`  [⚠️ truncated]
*job 24342089 • idun-07-10 • A100 • epoch 138/1000 • chain 1/20 • 8KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp46b_mamba_l_lpips_scoreaug_20260425-043934/checkpoint_latest.pt`

#### `24342186_24342186`  [⚠️ truncated]
*job 24342186 • idun-09-02 • H200 • epoch 127/500 • chain 1/20 • 22KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp47a_lpips_strong_20260425-055252/checkpoint_latest.pt`

#### `24342187_24342187`  [⚠️ truncated]
*job 24342187 • idun-06-04 • A100 • epoch 115/500 • chain 1/20 • 12KB log*

**Resumed from:** `/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp47b_huber_lpips_lowt_20260425-055554/checkpoint_latest.pt`

---
## train/downstream/

*150 jobs across 1 families.*

### OTHER (150 jobs)

*Status: 🔗 chained=93 ⚠️ truncated=33 ✅ completed=19 ❌ crashed=5*

#### `24044743_24044743`  [✅ completed]
*job 24044743 • idun-07-08 • A100 • 11.78h training • epoch 500/500 • chain 0/10 • 190KB log*


#### `24044745_24044745`  [✅ completed]
*job 24044745 • idun-06-04 • A100 • 10.75h training • epoch 500/500 • chain 0/10 • 191KB log*


#### `24046560_24046560`  [🔗 chained]
*job 24046560 • idun-09-18 • A100 • epoch 108/500 • chain 0/10 • 46KB log*


#### `24046582_24046582`  [🔗 chained]
*job 24046582 • idun-06-07 • A100 • epoch 73/500 • chain 0/10 • 33KB log*


#### `24051336_24051336`  [🔗 chained]
*job 24051336 • idun-09-18 • A100 • epoch 216/500 • chain 1/10 • 46KB log*


#### `24051514_24051514`  [🔗 chained]
*job 24051514 • idun-06-07 • A100 • epoch 148/500 • chain 1/10 • 34KB log*


#### `24052529_24052529`  [🔗 chained]
*job 24052529 • idun-09-18 • A100 • epoch 324/500 • chain 2/10 • 46KB log*


#### `24055670_24055670`  [🔗 chained]
*job 24055670 • idun-01-04 • H100 • epoch 257/500 • chain 2/10 • 46KB log*


#### `24056354_24056354`  [🔗 chained]
*job 24056354 • idun-09-18 • A100 • epoch 432/500 • chain 3/10 • 46KB log*


#### `24056697_24056697`  [🔗 chained]
*job 24056697 • idun-07-10 • A100 • epoch 336/500 • chain 3/10 • 35KB log*


#### `24057171_24057171`  [✅ completed]
*job 24057171 • idun-09-18 • A100 • 7.46h training • epoch 500/500 • chain 4/10 • 33KB log*


#### `24060149_24060149`  [🔗 chained]
*job 24060149 • idun-09-18 • A100 • epoch 408/500 • chain 4/10 • 33KB log*


#### `24060837_24060837`  [🔗 chained]
*job 24060837 • idun-07-05 • A100 • epoch 481/500 • chain 5/10 • 33KB log*


#### `24061764_24061764`  [✅ completed]
*job 24061764 • idun-07-05 • A100 • 3.0h training • epoch 500/500 • chain 6/10 • 15KB log*


#### `24077018_24077018`  [🔗 chained]
*job 24077018 • idun-06-05 • A100 • epoch 112/500 • chain 0/10 • 49KB log*


#### `24077019_24077019`  [❌ crashed]
*job 24077019 • idun-06-01 • A100 • epoch 500/500 • chain 0/10 • 201KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_segmentation.py", line 153, in main
    trainer.train(
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	712m56.059s
user	1277m29.811s
sys	291m16.130s
```

#### `24077020_24077020`  [🔗 chained]
*job 24077020 • idun-06-05 • A100 • epoch 75/500 • chain 0/10 • 35KB log*


#### `24077021_24077021`  [❌ crashed]
*job 24077021 • idun-06-05 • A100 • epoch 500/500 • chain 0/10 • 207KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_segmentation.py", line 153, in main
    trainer.train(
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	702m44.995s
user	1589m15.831s
sys	439m10.293s
```

#### `24082451_24082451`  [🔗 chained]
*job 24082451 • idun-07-06 • A100 • epoch 118/500 • chain 0/10 • 52KB log*


#### `24082452_24082452`  [❌ crashed]
*job 24082452 • idun-06-01 • A100 • epoch 500/500 • chain 0/10 • 210KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_segmentation.py", line 153, in main
    trainer.train(
...
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

real	716m59.144s
user	2005m58.581s
sys	549m41.781s
```

#### `24082453_24082453`  [🔗 chained]
*job 24082453 • idun-06-05 • A100 • epoch 75/500 • chain 0/10 • 35KB log*


#### `24082454_24082454`  [🔗 chained]
*job 24082454 • idun-06-05 • A100 • epoch 488/500 • chain 0/10 • 205KB log*


#### `24083993_24083993`  [🔗 chained]
*job 24083993 • idun-06-01 • A100 • epoch 224/500 • chain 1/10 • 50KB log*


#### `24084999_24084999`  [🔗 chained]
*job 24084999 • idun-07-06 • A100 • epoch 155/500 • chain 1/10 • 37KB log*


#### `24088055_24088055`  [🔗 chained]
*job 24088055 • idun-06-05 • A100 • epoch 229/500 • chain 1/10 • 49KB log*


#### `24089061_24089061`  [🔗 chained]
*job 24089061 • idun-06-02 • A100 • epoch 149/500 • chain 1/10 • 35KB log*


#### `24089080_24089080`  [✅ completed]
*job 24089080 • idun-06-05 • A100 • 0.29h training • epoch 500/500 • chain 1/10 • 13KB log*


#### `24089575_24089575`  [🔗 chained]
*job 24089575 • idun-06-01 • A100 • epoch 336/500 • chain 2/10 • 50KB log*


#### `24089576_24089576`  [🔗 chained]
*job 24089576 • idun-06-02 • A100 • epoch 229/500 • chain 2/10 • 35KB log*


#### `24091049_24091049`  [🔗 chained]
*job 24091049 • idun-06-07 • A100 • epoch 359/500 • chain 2/10 • 57KB log*


#### `24091072_24091072`  [🔗 chained]
*job 24091072 • idun-06-05 • A100 • epoch 224/500 • chain 2/10 • 35KB log*


#### `24092998_24092998`  [🔗 chained]
*job 24092998 • idun-09-17 • A100 • epoch 444/500 • chain 3/10 • 48KB log*


#### `24093077_24093077`  [🔗 chained]
*job 24093077 • idun-06-01 • A100 • epoch 305/500 • chain 3/10 • 36KB log*


#### `24093827_24093827`  [🔗 chained]
*job 24093827 • idun-07-06 • A100 • epoch 454/500 • chain 3/10 • 43KB log*


#### `24093829_24093829`  [🔗 chained]
*job 24093829 • idun-07-06 • A100 • epoch 288/500 • chain 3/10 • 31KB log*


#### `24093864_24093864`  [✅ completed]
*job 24093864 • idun-01-05 • H100 • 4.52h training • epoch 500/500 • chain 4/10 • 31KB log*


#### `24093869_24093869`  [🔗 chained]
*job 24093869 • idun-01-05 • H100 • epoch 409/500 • chain 4/10 • 46KB log*


#### `24093972_24093972`  [✅ completed]
*job 24093972 • idun-08-01 • H100 • 3.42h training • epoch 500/500 • chain 4/10 • 27KB log*


#### `24093977_24093977`  [🔗 chained]
*job 24093977 • idun-01-03 • H100 • epoch 396/500 • chain 4/10 • 48KB log*


#### `24094012_24094012`  [✅ completed]
*job 24094012 • idun-01-04 • H100 • 9.45h training • epoch 500/500 • chain 0/10 • 201KB log*


#### `24094013_24094013`  [✅ completed]
*job 24094013 • idun-01-04 • H100 • 5.75h training • epoch 100/100 • chain 0/10 • 51KB log*


#### `24094014_24094014`  [🔗 chained]
*job 24094014 • idun-01-04 • H100 • epoch 214/500 • chain 0/10 • 96KB log*


#### `24094015_24094015`  [🔗 chained]
*job 24094015 • idun-08-01 • H100 • epoch 417/500 • chain 0/10 • 186KB log*


#### `24094016_24094016`  [🔗 chained]
*job 24094016 • idun-07-09 • A100 • epoch 253/500 • chain 0/10 • 116KB log*


#### `24094018_24094018`  [🔗 chained]
*job 24094018 • idun-01-04 • H100 • epoch 317/500 • chain 0/10 • 152KB log*


#### `24094019_24094019`  [🔗 chained]
*job 24094019 • idun-08-01 • H100 • epoch 255/500 • chain 0/10 • 152KB log*


#### `24095753_24095753`  [🔗 chained]
*job 24095753 • idun-07-06 • A100 • epoch 489/500 • chain 5/10 • 37KB log*


#### `24096037_24096037`  [✅ completed]
*job 24096037 • idun-01-03 • H100 • 11.33h training • epoch 500/500 • chain 5/10 • 50KB log*


#### `24096072_24096072`  [🔗 chained]
*job 24096072 • idun-07-10 • A100 • epoch 350/500 • chain 1/10 • 61KB log*


#### `24096130_24096130`  [✅ completed]
*job 24096130 • idun-08-01 • H100 • 2.22h training • epoch 500/500 • chain 1/10 • 43KB log*


#### `24096321_24096321`  [✅ completed]
*job 24096321 • idun-06-05 • A100 • 1.75h training • epoch 500/500 • chain 6/10 • 14KB log*


#### `24096322_24096322`  [✅ completed]
*job 24096322 • idun-06-01 • A100 • 9.13h training • epoch 500/500 • chain 1/10 • 109KB log*


#### `24096352_24096352`  [✅ completed]
*job 24096352 • idun-06-01 • A100 • 9.44h training • epoch 500/500 • chain 1/10 • 85KB log*


#### `24096378_24096378`  [🔗 chained]
*job 24096378 • idun-06-05 • A100 • epoch 433/500 • chain 1/10 • 84KB log*


#### `24096998_24096998`  [🔗 chained]
*job 24096998 • idun-06-07 • A100 • epoch 497/500 • chain 2/10 • 66KB log*


#### `24097330_24097330`  [✅ completed]
*job 24097330 • idun-06-01 • A100 • 4.47h training • epoch 500/500 • chain 2/10 • 37KB log*


#### `24097346_24097346`  [🔗 chained]
*job 24097346 • idun-07-10 • A100 • epoch 350/500 • chain 0/10 • 145KB log*


#### `24097347_24097347`  [🔗 chained]
*job 24097347 • idun-07-10 • A100 • epoch 309/500 • chain 0/10 • 128KB log*


#### `24097349_24097349`  [🔗 chained]
*job 24097349 • idun-07-10 • A100 • epoch 297/500 • chain 0/10 • 131KB log*


#### `24097403_24097403`  [✅ completed]
*job 24097403 • idun-06-01 • A100 • 0.25h training • epoch 500/500 • chain 3/10 • 10KB log*


#### `24098121_24098121`  [✅ completed]
*job 24098121 • idun-07-10 • A100 • 6.24h training • epoch 500/500 • chain 1/10 • 83KB log*


#### `24098122_24098122`  [✅ completed]
*job 24098122 • idun-07-09 • A100 • 4.46h training • epoch 500/500 • chain 1/10 • 68KB log*


#### `24098123_24098123`  [✅ completed]
*job 24098123 • idun-08-01 • H100 • 4.85h training • epoch 500/500 • chain 1/10 • 89KB log*


#### `prep_24122475`  [⚠️ truncated]
*job 24122475 • idun-03-01 • 20KB log*


#### `24126472_24126472`  [🔗 chained]
*job 24126472 • idun-07-06 • A100 • chain 0/20 • 3KB log*


#### `24126473_24126473`  [🔗 chained]
*job 24126473 • idun-07-07 • A100 • chain 0/20 • 3KB log*


#### `24126474_24126474`  [🔗 chained]
*job 24126474 • idun-07-07 • A100 • chain 0/20 • 3KB log*


#### `24126475_24126475`  [🔗 chained]
*job 24126475 • idun-07-04 • A100 • chain 0/20 • 3KB log*


#### `24126476_24126476`  [🔗 chained]
*job 24126476 • idun-01-03 • H100 • chain 0/20 • 3KB log*


#### `24126477_24126477`  [🔗 chained]
*job 24126477 • idun-08-01 • H100 • chain 0/20 • 3KB log*


#### `24126478_24126478`  [🔗 chained]
*job 24126478 • idun-07-08 • A100 • chain 0/20 • 3KB log*


#### `24126479_24126479`  [🔗 chained]
*job 24126479 • idun-06-07 • A100 • chain 0/20 • 3KB log*


#### `24126480_24126480`  [🔗 chained]
*job 24126480 • idun-01-05 • H100 • chain 0/20 • 3KB log*


#### `24126481_24126481`  [🔗 chained]
*job 24126481 • idun-07-06 • A100 • chain 0/20 • 3KB log*


#### `24126482_24126482`  [🔗 chained]
*job 24126482 • idun-07-07 • A100 • chain 0/20 • 3KB log*


#### `24126483_24126483`  [🔗 chained]
*job 24126483 • idun-07-07 • A100 • chain 0/20 • 3KB log*


#### `24126484_24126484`  [🔗 chained]
*job 24126484 • idun-07-10 • A100 • chain 0/20 • 3KB log*


#### `24126485_24126485`  [🔗 chained]
*job 24126485 • idun-07-04 • A100 • chain 0/20 • 3KB log*


#### `24126486_24126486`  [🔗 chained]
*job 24126486 • idun-07-04 • A100 • chain 0/20 • 3KB log*


#### `24127089_24127089`  [🔗 chained]
*job 24127089 • idun-06-01 • A100 • chain 1/20 • 3KB log*


#### `24127091_24127091`  [⚠️ truncated]
*job 24127091 • idun-08-01 • H100 • chain 1/20 • 220KB log*


#### `24127092_24127092`  [🔗 chained]
*job 24127092 • idun-09-18 • A100 • chain 1/20 • 3KB log*


#### `24127839_24127839`  [🔗 chained]
*job 24127839 • idun-07-05 • A100 • chain 1/20 • 3KB log*


#### `24128401_24128401`  [⚠️ truncated]
*job 24128401 • idun-07-05 • A100 • chain 1/20 • 124KB log*


#### `24128562_24128562`  [🔗 chained]
*job 24128562 • idun-07-05 • A100 • chain 1/20 • 3KB log*


#### `24129363_24129363`  [🔗 chained]
*job 24129363 • idun-06-05 • A100 • chain 1/20 • 3KB log*


#### `24129804_24129804`  [⚠️ truncated]
*job 24129804 • idun-01-03 • H100 • chain 1/20 • 207KB log*


#### `24129994_24129994`  [⚠️ truncated]
*job 24129994 • idun-01-03 • H100 • chain 1/20 • 124KB log*


#### `24130044_24130044`  [🔗 chained]
*job 24130044 • idun-09-16 • A100 • chain 1/20 • 3KB log*


#### `24130072_24130072`  [🔗 chained]
*job 24130072 • idun-06-05 • A100 • chain 1/20 • 3KB log*


#### `24130073_24130073`  [🔗 chained]
*job 24130073 • idun-07-04 • A100 • chain 1/20 • 3KB log*


#### `24130256_24130256`  [🔗 chained]
*job 24130256 • idun-06-05 • A100 • chain 1/20 • 3KB log*


#### `24130481_24130481`  [⚠️ truncated]
*job 24130481 • idun-08-01 • H100 • chain 1/20 • 221KB log*


#### `24130482_24130482`  [⚠️ truncated]
*job 24130482 • idun-01-05 • H100 • chain 1/20 • 221KB log*


#### `24131277_24131277`  [⚠️ truncated]
*job 24131277 • idun-06-07 • A100 • chain 2/20 • 92KB log*


#### `24131329_24131329`  [⚠️ truncated]
*job 24131329 • idun-07-05 • A100 • chain 2/20 • 124KB log*


#### `24131369_24131369`  [⚠️ truncated]
*job 24131369 • idun-07-05 • A100 • chain 2/20 • 108KB log*


#### `24131370_24131370`  [⚠️ truncated]
*job 24131370 • idun-07-05 • A100 • chain 2/20 • 76KB log*


#### `24133562_24133562`  [⚠️ truncated]
*job 24133562 • idun-01-05 • H100 • chain 2/20 • 92KB log*


#### `24135546_24135546`  [🔗 chained]
*job 24135546 • idun-09-16 • A100 • chain 2/20 • 3KB log*


#### `24137767_24137767`  [⚠️ truncated]
*job 24137767 • idun-06-01 • A100 • chain 2/20 • 93KB log*


#### `24137896_24137896`  [⚠️ truncated]
*job 24137896 • idun-01-03 • H100 • chain 2/20 • 108KB log*


#### `24137897_24137897`  [⚠️ truncated]
*job 24137897 • idun-09-18 • A100 • chain 2/20 • 92KB log*


#### `24140055_24140055`  [⚠️ truncated]
*job 24140055 • idun-07-05 • A100 • chain 3/20 • 28KB log*


#### `24147719_24147719`  [🔗 chained]
*job 24147719 • idun-09-18 • A100 • chain 0/20 • 3KB log*


#### `24147720_24147720`  [🔗 chained]
*job 24147720 • idun-07-10 • A100 • chain 0/20 • 3KB log*


#### `24147721_24147721`  [🔗 chained]
*job 24147721 • idun-08-01 • H100 • chain 0/20 • 3KB log*


#### `24147722_24147722`  [🔗 chained]
*job 24147722 • idun-06-03 • A100 • chain 0/20 • 3KB log*


#### `24152068_24152068`  [🔗 chained]
*job 24152068 • idun-01-05 • H100 • chain 1/20 • 3KB log*


#### `24152077_24152077`  [🔗 chained]
*job 24152077 • idun-08-01 • H100 • chain 1/20 • 3KB log*


#### `24152082_24152082`  [🔗 chained]
*job 24152082 • idun-08-01 • H100 • chain 1/20 • 3KB log*


#### `24152084_24152084`  [🔗 chained]
*job 24152084 • idun-08-01 • H100 • chain 1/20 • 3KB log*


#### `24154362_24154362`  [⚠️ truncated]
*job 24154362 • idun-07-09 • A100 • chain 2/20 • 76KB log*


#### `24154368_24154368`  [⚠️ truncated]
*job 24154368 • idun-07-09 • A100 • chain 2/20 • 92KB log*


#### `24154371_24154371`  [⚠️ truncated]
*job 24154371 • idun-07-09 • A100 • chain 2/20 • 76KB log*


#### `24154373_24154373`  [⚠️ truncated]
*job 24154373 • idun-09-16 • A100 • chain 2/20 • 92KB log*


#### `prep_24294100`  [⚠️ truncated]
*job 24294100 • idun-04-01 • 18KB log*


#### `24294387_24294387`  [🔗 chained]
*job 24294387 • idun-01-04 • H100 • chain 0/20 • 3KB log*


#### `24294388_24294388`  [🔗 chained]
*job 24294388 • idun-01-05 • H100 • chain 0/20 • 3KB log*


#### `24295384_24295384`  [🔗 chained]
*job 24295384 • idun-11-18 • A100 • chain 0/20 • 3KB log*


#### `24295385_24295385`  [🔗 chained]
*job 24295385 • idun-06-06 • A100 • chain 0/20 • 3KB log*


#### `24295386_24295386`  [🔗 chained]
*job 24295386 • idun-07-07 • A100 • chain 0/20 • 3KB log*


#### `24295387_24295387`  [🔗 chained]
*job 24295387 • idun-01-04 • H100 • chain 0/20 • 3KB log*


#### `24295985_24295985`  [⚠️ truncated]
*job 24295985 • idun-01-04 • H100 • chain 1/20 • 172KB log*


#### `24295986_24295986`  [🔗 chained]
*job 24295986 • idun-07-04 • A100 • chain 1/20 • 3KB log*


#### `24296207_24296207`  [🔗 chained]
*job 24296207 • idun-11-18 • A100 • chain 0/20 • 3KB log*


#### `24296208_24296208`  [🔗 chained]
*job 24296208 • idun-06-07 • A100 • chain 0/20 • 3KB log*


#### `24296209_24296209`  [🔗 chained]
*job 24296209 • idun-07-06 • A100 • chain 0/20 • 3KB log*


#### `24296210_24296210`  [🔗 chained]
*job 24296210 • idun-01-04 • H100 • chain 0/20 • 3KB log*


#### `24296592_24296592`  [🔗 chained]
*job 24296592 • idun-01-04 • H100 • chain 1/20 • 3KB log*


#### `24296604_24296604`  [🔗 chained]
*job 24296604 • idun-01-04 • H100 • chain 1/20 • 3KB log*


#### `24296612_24296612`  [🔗 chained]
*job 24296612 • idun-01-04 • H100 • chain 1/20 • 3KB log*


#### `24296707_24296707`  [🔗 chained]
*job 24296707 • idun-01-04 • H100 • chain 1/20 • 3KB log*


#### `baseline_24297020`  [⚠️ truncated]
*job 24297020 • idun-07-07 • A100 • 4KB log*


#### `24297465_24297465`  [⚠️ truncated]
*job 24297465 • idun-11-19 • A100 • chain 2/20 • 60KB log*


#### `24297496_24297496`  [🔗 chained]
*job 24297496 • idun-11-18 • A100 • chain 1/20 • 3KB log*


#### `24297501_24297501`  [🔗 chained]
*job 24297501 • idun-07-06 • A100 • chain 1/20 • 3KB log*


#### `24297502_24297502`  [🔗 chained]
*job 24297502 • idun-01-04 • H100 • chain 1/20 • 3KB log*


#### `24297504_24297504`  [⚠️ truncated]
*job 24297504 • idun-01-04 • H100 • chain 1/20 • 173KB log*


#### `24297639_24297639`  [⚠️ truncated]
*job 24297639 • idun-01-04 • H100 • chain 2/20 • 77KB log*


#### `24297640_24297640`  [⚠️ truncated]
*job 24297640 • idun-01-04 • H100 • chain 2/20 • 44KB log*


#### `24297654_24297654`  [🔗 chained]
*job 24297654 • idun-11-19 • A100 • chain 2/20 • 3KB log*


#### `24297657_24297657`  [⚠️ truncated]
*job 24297657 • idun-01-04 • H100 • chain 2/20 • 60KB log*


#### `24298216_24298216`  [⚠️ truncated]
*job 24298216 • idun-01-03 • H100 • chain 2/20 • 140KB log*


#### `24298219_24298219`  [⚠️ truncated]
*job 24298219 • idun-06-07 • A100 • chain 2/20 • 93KB log*


#### `24298222_24298222`  [⚠️ truncated]
*job 24298222 • idun-06-04 • A100 • chain 2/20 • 60KB log*


#### `24298554_24298554`  [⚠️ truncated]
*job 24298554 • idun-11-19 • A100 • chain 3/20 • 28KB log*


#### `24298933_24298933`  [❌ crashed]
*job 24298933 • idun-07-04 • A100 • 12KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.
```

#### `prep_24299658`  [⚠️ truncated]
*job 24299658 • idun-03-03 • 27KB log*


#### `24299661_24299661`  [❌ crashed]
*job 24299661 • idun-07-06 • A100 • 12KB log*

**Traceback excerpt:**
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
...
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
<frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.
```

---
## tensorboard/

*70 jobs.*

TensorBoard server jobs — these are SLURM jobs that just run `tensorboard --logdir runs/` so the user can browse curves remotely. They auto-resubmit (chain mode) on wall-time approach. No training data is in these logs.

Bulk-summarized — only `[status]` and metadata, no per-job content.

- 🔗 chained `24042892_24042892` — job 24042892 • idun-04-32 • chain 0/50
- 🔗 chained `24044086_24044086` — job 24044086 • idun-03-32 • chain 1/50
- 🔗 chained `24047392_24047392` — job 24047392 • idun-03-32 • chain 2/50
- 🔗 chained `24052800_24052800` — job 24052800 • idun-03-02 • chain 3/50
- 🔗 chained `24059013_24059013` — job 24059013 • idun-03-10 • chain 4/50
- 🔗 chained `24061562_24061562` — job 24061562 • idun-04-18 • chain 5/50
- 🔗 chained `24062499_24062499` — job 24062499 • idun-04-32 • chain 6/50
- 🔗 chained `24063277_24063277` — job 24063277 • idun-03-01 • chain 7/50
- 🔗 chained `24064882_24064882` — job 24064882 • idun-03-19 • chain 8/50
- 🔗 chained `24071285_24071285` — job 24071285 • idun-03-19 • chain 9/50
- 🔗 chained `24077157_24077157` — job 24077157 • idun-04-26 • chain 10/50
- 🔗 chained `24085272_24085272` — job 24085272 • idun-04-25 • chain 11/50
- 🔗 chained `24092201_24092201` — job 24092201 • idun-04-28 • chain 12/50
- 🔗 chained `24094026_24094026` — job 24094026 • idun-03-19 • chain 13/50
- 🔗 chained `24096477_24096477` — job 24096477 • idun-04-18 • chain 14/50
- 🔗 chained `24100081_24100081` — job 24100081 • idun-03-19 • chain 15/50
- 🔗 chained `24105216_24105216` — job 24105216 • idun-03-30 • chain 16/50
- 🔗 chained `24108769_24108769` — job 24108769 • idun-03-32 • chain 17/50
- 🔗 chained `24113505_24113505` — job 24113505 • idun-03-28 • chain 18/50
- 🔗 chained `24118683_24118683` — job 24118683 • idun-03-15 • chain 19/50
- 🔗 chained `24121474_24121474` — job 24121474 • idun-04-13 • chain 20/50
- 🔗 chained `24122300_24122300` — job 24122300 • idun-04-11 • chain 21/50
- 🔗 chained `24125638_24125638` — job 24125638 • idun-04-11 • chain 22/50
- 🔗 chained `24129012_24129012` — job 24129012 • idun-03-23 • chain 23/50
- 🔗 chained `24135777_24135777` — job 24135777 • idun-04-06 • chain 24/50
- 🔗 chained `24145289_24145289` — job 24145289 • idun-03-20 • chain 25/50
- 🔗 chained `24151037_24151037` — job 24151037 • idun-04-18 • chain 26/50
- 🔗 chained `24155070_24155070` — job 24155070 • idun-05-33 • chain 27/50
- 🔗 chained `24158713_24158713` — job 24158713 • idun-05-33 • chain 28/50
- 🔗 chained `24166462_24166462` — job 24166462 • idun-05-34 • chain 29/50
- 🔗 chained `24189085_24189085` — job 24189085 • idun-05-33 • chain 30/50
- 🔗 chained `24192938_24192938` — job 24192938 • idun-03-14 • chain 31/50
- 🔗 chained `24200721_24200721` — job 24200721 • idun-05-11 • chain 32/50
- 🔗 chained `24208282_24208282` — job 24208282 • idun-03-09 • chain 33/50
- 🔗 chained `24213706_24213706` — job 24213706 • idun-03-07 • chain 34/50
- 🔗 chained `24216610_24216610` — job 24216610 • idun-03-07 • chain 35/50
- 🔗 chained `24225491_24225491` — job 24225491 • idun-04-01 • chain 36/50
- 🔗 chained `24231962_24231962` — job 24231962 • idun-04-22 • chain 37/50
- 🔗 chained `24234698_24234698` — job 24234698 • idun-05-34 • chain 38/50
- 🔗 chained `24239576_24239576` — job 24239576 • idun-05-34 • chain 39/50
- 🔗 chained `24245389_24245389` — job 24245389 • idun-03-04 • chain 40/50
- 🔗 chained `24248017_24248017` — job 24248017 • idun-05-33 • chain 41/50
- 🔗 chained `24252534_24252534` — job 24252534 • idun-05-33 • chain 42/50
- 🔗 chained `24255255_24255255` — job 24255255 • idun-05-34 • chain 43/50
- 🔗 chained `24260344_24260344` — job 24260344 • idun-05-33 • chain 44/50
- 🔗 chained `24261761_24261761` — job 24261761 • idun-05-33 • chain 45/50
- 🔗 chained `24269233_24269233` — job 24269233 • idun-05-34 • chain 46/50
- 🔗 chained `24271143_24271143` — job 24271143 • idun-05-33 • chain 47/50
- 🔗 chained `24272603_24272603` — job 24272603 • idun-05-33 • chain 48/50
- 🔗 chained `24275487_24275487` — job 24275487 • idun-05-33 • chain 49/50
- ⚠️ truncated `24278366_24278366` — job 24278366 • idun-05-33 • chain 0/50
- 🔗 chained `24278427_24278427` — job 24278427 • idun-05-33 • chain 0/50
- 🔗 chained `24282584_24282584` — job 24282584 • idun-05-34 • chain 1/50
- 🔗 chained `24287335_24287335` — job 24287335 • idun-05-34 • chain 2/50
- 🔗 chained `24295418_24295418` — job 24295418 • idun-04-18 • chain 3/50
- 🔗 chained `24297437_24297437` — job 24297437 • idun-05-34 • chain 4/50
- 🔗 chained `24298806_24298806` — job 24298806 • idun-05-34 • chain 5/50
- 🔗 chained `24303723_24303723` — job 24303723 • idun-05-33 • chain 6/50
- 🔗 chained `24308850_24308850` — job 24308850 • idun-05-33 • chain 7/50
- 🔗 chained `24313334_24313334` — job 24313334 • idun-05-33 • chain 8/50
- 🔗 chained `24315074_24315074` — job 24315074 • idun-05-33 • chain 9/50
- 🔗 chained `24316971_24316971` — job 24316971 • idun-05-33 • chain 10/50
- 🔗 chained `24319093_24319093` — job 24319093 • idun-05-34 • chain 11/50
- 🔗 chained `24323417_24323417` — job 24323417 • idun-03-02 • chain 12/50
- 🔗 chained `24325343_24325343` — job 24325343 • idun-05-33 • chain 13/50
- 🔗 chained `24327851_24327851` — job 24327851 • idun-05-33 • chain 14/50
- 🔗 chained `24330617_24330617` — job 24330617 • idun-05-34 • chain 15/50
- 🔗 chained `24334582_24334582` — job 24334582 • idun-05-34 • chain 16/50
- 🔗 chained `24336390_24336390` — job 24336390 • idun-05-33 • chain 17/50
- ⚠️ truncated `24341716_24341716` — job 24341716 • idun-05-33 • chain 18/50
---
## test/

*3 jobs.*

Smoke-test jobs for verifying pipeline integrity (TB tag schemas, config resolution, etc.) — not real training.

#### `verify_tb_tags_23898096`  [✅ completed]
*job 23898096 • idun-07-06 • A100 • 0.07h training • epoch 2/2 • 18KB log*


#### `verify_tb_tags_23898100`  [💥 oom_killed]
*job 23898100 • idun-07-06 • A100 • 0.07h training • epoch 2/2 • 97KB log*

**Final test metrics:**
  - **best** ckpt (3339 samples): MSE 0.9103 • MS-SSIM 0.2281 • PSNR 9.83 dB • LPIPS 1.1893
**Traceback excerpt:**
```
Traceback (most recent call last):
  File "/cluster/work/modestas/AIS4900_master/src/medgen/scripts/train_dcae_3d.py", line 32, in main
    train_compression_3d(cfg, trainer_type='dcae_3d')
...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/modestas/.conda/envs/AIS4900/lib/python3.12/site-packages/torch/_compile.py", line 53, in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/cluster/home/mode
```

#### `vram_test_23962232`  [⚠️ truncated]
*job 23962232 • idun-07-02 • 8KB log*


---

---

## Global summary

### Status distribution (1,291 jobs)

- **chained:** 631 (49%) — most are TensorBoard-server resubmits
- **completed:** 308 (24%) — successful training/eval/generation
- **truncated:** 172 (13%) — ran but didn't reach a recognized end marker (often eval scripts that wrote a JSON file separately)
- **crashed:** 124 (10%) — Python traceback in stdout/stderr
- **oom_killed:** 56 (4%) — `Killed` / `oom_kill` / `CUDA out of memory`

### Headline data extracted (not in `runs_tb/`)

1. **Post-hoc FID + optimal step count** for the canonical models — `find_optimal_steps_exp1_1_1000_24128889` reports FID 19.12 @ 27 steps using ImageNet features, matching `memory/project_3d_experiment_findings.md`.
2. **End-of-training test metrics** (best + latest) for ~308 completed training jobs: MSE / MS-SSIM / PSNR / LPIPS / FID / KID / CMMD on the 26-volume held-out test set.
3. **Generation-pipeline outputs** with sample counts and retry counts due to atlas/PCA seg-validation failures (`generate/compare_exp1_1_vs_exp37_24326455` produced 7×10 volumes with 6 retries).
4. **Restoration calibration** — IR-SDE t₀ search results from `eval/calibrate_degradation_*` (best t₀=0.50 confirmed in memory).
5. **Mean-blur diagnostics** — `eval/diagnose_mean_blur_*` results across exp1_1_1000 and exp37_{1,2,3} (output dirs cited in memory).
6. **OOM/crash fingerprints** — 56 OOMs (mostly DC-AE branch and very-large-model attempts) + 124 crashes with traceback excerpts captured per job.

### How to use this file

- Browse by subdir → family for a chronological view of one experiment's SLURM lifecycle (initial submit → resumes → final eval).
- For per-epoch loss/FID curves of the same experiment, see `EXPERIMENT_SUMMARIES.md`.
- Cross-reference between the two via `## IDUN log cross-reference` at the bottom of `EXPERIMENT_SUMMARIES.md`.
- For the underlying JSON, `idun_logs_extracted.json` (14 MB) has every record's full structured fields.

### Regenerating this document

```
python scripts/extract_idun_logs.py --out idun_logs_extracted.json
python /tmp/emit_idun_eval.py > /tmp/idun_eval.md
python /tmp/emit_idun_generate.py > /tmp/idun_gen.md
python /tmp/emit_idun_misc.py profiling > /tmp/idun_prof.md
python /tmp/emit_idun_misc.py debug > /tmp/idun_debug.md
python /tmp/emit_idun_train_d3.py > /tmp/idun_d3.md
python /tmp/emit_idun_train.py train/compression > /tmp/idun_tc.md
python /tmp/emit_idun_train.py train/diffusion > /tmp/idun_t2d.md
python /tmp/emit_idun_train.py train/downstream > /tmp/idun_td.md
python /tmp/emit_idun_misc.py tensorboard brief > /tmp/idun_tb.md
python /tmp/emit_idun_misc.py test > /tmp/idun_test.md
# concatenate header + sections
```

The `FAMILY_INTRO` and `INTROS` dicts in `/tmp/emit_idun_*.py` should be
extended when new experiments are added.

### Caveats

- Wall-time `real Xm Ys` from the bash `time` command is captured when present (most train jobs); some jobs don't run under `time` so `real_time` is None.
- For chained jobs, only the per-segment time is captured — the cumulative training time across segments is in `Training completed in N hours` for the final segment only.
- "truncated" status is a catch-all — many such jobs DID complete normally but didn't write a recognizable end marker (some eval scripts write to JSON files, not to stdout). Read the per-job entry to confirm.
- 56 OOMs include the 3 DC-AE-related ones noted in the original review; the remainder are mostly large-model + small-VRAM combinations or very-long-running chains that hit the wall time without explicit graceful exit.
