# Tier 1 — find_optimal_steps batch

Motivation: experiments whose TB extended-FID looked mediocre may actually be
competitive once their per-experiment optimal Euler step count is found, as
demonstrated by Mamba (`exp34_0_1000`) where step=11 gave a much better picture
than the implicit 25-step extended-eval. This batch runs `find_optimal_steps`
for 13 experiments that previously had no optima search at all (or hit the
search-range boundary).

## What's in here

| Script | Tier | Experiment | Range | Wall-time | Mem | Why |
|---|---|---|---|---|---|---|
| `find_optimal_steps_exp34_0_1000.slurm` | 1d | Mamba-S /p4 1000ep | 5-50 | 8h | 64 G | prior search hit floor at best=11 in [10, 100] |
| `find_optimal_steps_exp34_1_1000.slurm` | 1d | Mamba-L /p4 +gc 1000ep | 5-50 | 10h | 80 G | prior search hit floor at best=14 in [10, 100] |
| `find_optimal_steps_exp15.slurm` | 1c | UViT pixel | 10-100 | 8h | 64 G | never searched |
| `find_optimal_steps_exp18.slurm` | 1c | UViT-L /p8 256 | 10-100 | 8h | 80 G | never searched |
| `find_optimal_steps_exp29.slurm` | 1c | mixup 1000ep | 10-100 | 12h | 64 G | never searched |
| `find_optimal_steps_exp20_4.slurm` | 1b | UNet 67M small | 5-80 | 6h | 48 G | small; expect lower-step optimum |
| `find_optimal_steps_exp20_6.slurm` | 1b | UNet 20M tiny | 5-80 | 6h | 32 G | smallest model |
| `find_optimal_steps_exp20_7.slurm` | 1b | UNet 67M no-attn | 5-80 | 6h | 48 G | small no-attn |
| `find_optimal_steps_exp20_2.slurm` | 1b | UNet deep+wide 270M | 10-100 | 8h | 64 G | model-scaling sweep |
| `find_optimal_steps_exp20_3.slurm` | 1b | UNet deep+wide+attn_l3 | 10-100 | 8h | 64 G | model-scaling sweep |
| `find_optimal_steps_exp20_5.slurm` | 1b | UNet 152M mid | 10-100 | 6h | 64 G | model-scaling sweep |
| `find_optimal_steps_exp17_2.slurm` | 1a | HDiT-L /p8 256 | 10-100 | 12h | 80 G | never searched |
| `find_optimal_steps_exp17_3.slurm` | 1a | HDiT-XL /p8 256 | 10-100 | 14h | 80 G | never searched (largest) |

**Skipped from Tier 1** (training didn't finish enough to be worth searching):
- exp17_0 (HDiT-S/p2): only 2 epochs trained
- exp17_1 (HDiT-B/p4): 178/500 epochs
- exp17_4 (HDiT-S/p4): 54 epochs
- exp20_1 (UNet attn_l3): 317/500 epochs

## How to run

```bash
# Dry-run to see what would be submitted
./IDUN/eval/tier1/submit_all_tier1.sh --dry-run

# Submit everything
./IDUN/eval/tier1/submit_all_tier1.sh

# Submit only specific experiments
./IDUN/eval/tier1/submit_all_tier1.sh exp34_0_1000 exp34_1_1000

# Watch progress
squeue -u $USER --format='%.10i %.30j %.8T %.10M %R'
tail -f IDUN/output/eval/find_optimal_steps_exp34_0_1000_*.out
```

## Output locations

Each job writes to two places on the cluster:
- **Per-job stdout** → `IDUN/output/eval/find_optimal_steps_<exp>_<jobid>.{out,err}`
- **Search results JSON** → `${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_<exp>/search_results.json`

## Reading results back

After all jobs complete, re-run the IDUN log extractor to pull `best_steps` +
best-FID + best-FID_radimagenet + best-PCA values into the structured JSON:

```bash
python scripts/extract_idun_logs.py --out idun_logs_extracted.json
# Then review IDUN_LOG_SUMMARIES.md (regenerate the eval/ section):
python /tmp/emit_idun_eval.py > /tmp/idun_eval.md
```

## What success looks like

For each run, the search emits `best_steps` for each metric:
- `fid` — ImageNet feature-extractor FID
- `fid_radimagenet` — RadImageNet feature-extractor FID
- `pca` — PCA brain-shape error (anatomical fidelity proxy)

Compare best-FID values to the existing leaderboard:

| Reference | Best step | FID (ImageNet) |
|---|---|---|
| exp1_1_1000 (current best) | 27 | 19.12 |
| exp23 ScoreAug | 23/27 | 20.38 |
| **exp34_*_1000 (Mamba) — to find** | **5-50** | **?** |
| **exp17_2/3 (HDiT L/XL) — to find** | **10-100** | **?** |
| **exp20_* (UNet sizes) — to find** | **5-100** | **?** |

If any Tier 1 result lands FID < 25 at its own optimum, it changes the
"baseline 270M wins" conclusion in `memory/project_3d_experiment_findings.md`.

## Total compute

If all 13 jobs run sequentially: ~108 GPU-hours.
If submitted in parallel (different nodes): wall-clock ~14 hours bound by
the slowest job (exp17_3 HDiT-XL).
