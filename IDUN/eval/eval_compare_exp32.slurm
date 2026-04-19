#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_exp32"
#SBATCH --output=IDUN/output/eval/eval_compare_exp32_%j.out
#SBATCH --error=IDUN/output/eval/eval_compare_exp32_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Compare exp1_1 (baseline MSE) vs exp32 (LPIPS fine-tune) vs exp32_1 (FFL fine-tune)
# All three evaluated at 32 steps (ImageNet FID optimum) with same noise seeds.
# Metrics: FID, RadImageNet FID, KID, CMMD, PCA, FWD (per-frequency-band)

cd ${SLURM_SUBMIT_DIR}

echo "=== Compare: exp1_1 vs exp32 (LPIPS) vs exp32_1 (FFL) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

python -c "
import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

from medgen.diffusion import load_diffusion_model, RFlowStrategy
from medgen.scripts.eval_ode_solvers import (
    SolverConfig, compute_all_metrics, discover_splits,
    generate_noise_tensors, generate_volumes,
    get_or_cache_reference_features, load_conditioning,
)
from medgen.metrics.fwd import compute_fwd_3d

device = torch.device('cuda')

DATA_ROOT = Path('${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3')
OUT_DIR = Path('${CLUSTER_BASE}/MedicalDataSets/eval_compare_exp32')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    'exp1_1_baseline': '${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt',
    'exp32_lpips': '${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp32_pixel_bravo_perceptual_20260409-192621/checkpoint_latest.pt',
    'exp32_1_ffl': '${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_pixel_bravo_ffl_20260409-192332/checkpoint_latest.pt',
}

NUM_VOLUMES = 25
NUM_STEPS = 32
IMAGE_SIZE = 256
DEPTH = 160
TRIM_SLICES = 10
SEED = 42

# ── Setup ────────────────────────────────────────────────────────
logger.info('Setting up strategy...')
strategy = RFlowStrategy()
strategy.setup_scheduler(num_timesteps=1000, image_size=IMAGE_SIZE, depth_size=DEPTH, spatial_dims=3)

# ── Load conditioning masks ──────────────────────────────────────
splits = discover_splits(DATA_ROOT, modality='bravo')
logger.info(f'Loading {NUM_VOLUMES} conditioning masks...')
cond_list = load_conditioning(splits['val'], NUM_VOLUMES, DEPTH)

# ── Pre-generate noise (shared across all models) ────────────────
logger.info(f'Generating {NUM_VOLUMES} noise tensors (seed={SEED})...')
noise_list = generate_noise_tensors(NUM_VOLUMES, DEPTH, IMAGE_SIZE, device, SEED, out_channels=1)

# ── Load reference features for FID/KID/CMMD ─────────────────────
cache_dir = OUT_DIR / 'reference_features'
logger.info('Loading reference features (test split)...')
needed_splits = {'test': splits['test']}
ref_features = get_or_cache_reference_features(
    needed_splits, cache_dir, device, DEPTH, TRIM_SLICES, IMAGE_SIZE,
    modality='bravo', build_all=False,
)
eval_ref = {'test': ref_features['test']}
del ref_features
gc.collect()

# ── Load real volumes for FWD reference ───────────────────────────
logger.info('Loading real volumes for FWD...')
import nibabel as nib
real_volumes = []
for split_name in ['test']:
    seg_files = sorted(splits[split_name].glob('*/bravo.nii.gz'))
    for fp in seg_files:
        vol = nib.load(str(fp)).get_fdata().astype(np.float32)
        # Normalize to [0,1]
        if vol.max() > 0:
            vol = vol / vol.max()
        # Transpose [H,W,D] -> [D,H,W]
        vol = np.transpose(vol, (2, 0, 1))
        if vol.shape[0] < DEPTH:
            vol = np.pad(vol, ((0, DEPTH - vol.shape[0]), (0, 0), (0, 0)), mode='constant')
        elif vol.shape[0] > DEPTH:
            vol = vol[:DEPTH]
        real_volumes.append(vol)
logger.info(f'Loaded {len(real_volumes)} real volumes for FWD')

# ── Load PCA model ────────────────────────────────────────────────
brain_pca = None
repo_root = Path('${CLUSTER_BASE}/AIS4900_master')
pca_path = repo_root / 'data' / f'brain_pca_{IMAGE_SIZE}x{IMAGE_SIZE}x{DEPTH}.npz'
if pca_path.exists():
    from medgen.metrics.brain_mask import BrainPCAModel, create_brain_mask
    brain_pca = BrainPCAModel(pca_path)
    logger.info(f'Brain PCA loaded (threshold={brain_pca.error_threshold:.6f})')

# ── Evaluate each model ──────────────────────────────────────────
solver_cfg = SolverConfig(solver='euler', steps=NUM_STEPS)
all_results = {}

for name, ckpt_path in CHECKPOINTS.items():
    if not Path(ckpt_path).exists():
        logger.warning(f'Checkpoint not found: {ckpt_path}, skipping {name}')
        continue

    logger.info(f'')
    logger.info(f'={\"=\" * 60}')
    logger.info(f'  Evaluating: {name}')
    logger.info(f'={\"=\" * 60}')

    # Load model
    logger.info(f'Loading model: {ckpt_path}')
    model = load_diffusion_model(
        ckpt_path, device=device,
        in_channels=2, out_channels=1, compile_model=False, spatial_dims=3,
    )

    # Generate volumes
    t0 = time.time()
    volumes, nfe, wall_time = generate_volumes(
        model, strategy, noise_list, cond_list, solver_cfg, device,
        is_seg=False,
    )
    gen_time = time.time() - t0
    logger.info(f'Generated {len(volumes)} volumes in {gen_time:.0f}s')

    # Free model GPU memory
    del model
    torch.cuda.empty_cache()

    # ── FID / KID / CMMD ──────────────────────────────────────────
    split_metrics = compute_all_metrics(volumes, eval_ref, device, TRIM_SLICES)
    m = asdict(split_metrics['test'])

    # ── PCA ───────────────────────────────────────────────────────
    pca_errors = []
    if brain_pca is not None:
        for vol in volumes:
            mask = create_brain_mask(vol, threshold=0.05, fill_holes=True, dilate_pixels=0)
            _, err = brain_pca.is_valid(mask)
            pca_errors.append(err)

    # ── FWD ───────────────────────────────────────────────────────
    logger.info(f'Computing FWD...')
    fwd_score, fwd_bands = compute_fwd_3d(
        real_volumes, volumes,
        trim_slices=TRIM_SLICES, max_level=4,
    )

    # Group FWD bands into low/mid/high frequency
    n_bands = len(fwd_bands)
    quarter = n_bands // 4
    band_values = list(fwd_bands.values())
    fwd_low = float(np.mean(band_values[:quarter]))
    fwd_mid = float(np.mean(band_values[quarter:3*quarter]))
    fwd_high = float(np.mean(band_values[3*quarter:]))

    # ── Collect results ───────────────────────────────────────────
    result = {
        'checkpoint': ckpt_path,
        'num_steps': NUM_STEPS,
        'num_volumes': NUM_VOLUMES,
        'fid': m['fid'],
        'kid_mean': m['kid_mean'],
        'kid_std': m['kid_std'],
        'cmmd': m['cmmd'],
        'fid_radimagenet': m['fid_radimagenet'],
        'kid_radimagenet_mean': m['kid_radimagenet_mean'],
        'kid_radimagenet_std': m['kid_radimagenet_std'],
        'fwd': fwd_score,
        'fwd_low_freq': fwd_low,
        'fwd_mid_freq': fwd_mid,
        'fwd_high_freq': fwd_high,
        'fwd_per_band': {str(k): v for k, v in fwd_bands.items()},
        'generation_time_s': gen_time,
    }
    if pca_errors:
        result['pca_mean_error'] = float(np.mean(pca_errors))
        result['pca_pass_rate'] = float(np.mean([e <= brain_pca.error_threshold for e in pca_errors]))

    all_results[name] = result

    logger.info(f'  FID={m[\"fid\"]:.2f}  FID_RIN={m[\"fid_radimagenet\"]:.2f}  '
                f'KID={m[\"kid_mean\"]:.6f}  CMMD={m[\"cmmd\"]:.4f}')
    logger.info(f'  FWD={fwd_score:.4f}  (low={fwd_low:.4f}  mid={fwd_mid:.4f}  high={fwd_high:.4f})')
    if pca_errors:
        logger.info(f'  PCA={result[\"pca_mean_error\"]:.6f}  pass={result[\"pca_pass_rate\"]:.0%}')

    del volumes
    gc.collect()
    torch.cuda.empty_cache()

# ── Save results ─────────────────────────────────────────────────
with open(OUT_DIR / 'compare_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ── Summary table ────────────────────────────────────────────────
logger.info('')
logger.info('=' * 90)
logger.info('COMPARISON SUMMARY (32 Euler steps, 25 volumes, test split)')
logger.info('=' * 90)
header = f'{\"Model\":>20}  {\"FID\":>8}  {\"FID_RIN\":>8}  {\"KID\":>10}  {\"CMMD\":>8}  {\"FWD\":>8}  {\"FWD_hi\":>8}  {\"PCA\":>8}'
logger.info(header)
logger.info('-' * 90)

for name, r in all_results.items():
    pca_str = f'{r.get(\"pca_mean_error\", 0):.6f}' if 'pca_mean_error' in r else 'N/A'
    logger.info(
        f'{name:>20}  {r[\"fid\"]:>8.2f}  {r[\"fid_radimagenet\"]:>8.2f}  '
        f'{r[\"kid_mean\"]:>10.6f}  {r[\"cmmd\"]:>8.4f}  '
        f'{r[\"fwd\"]:>8.4f}  {r[\"fwd_high_freq\"]:>8.4f}  {pca_str:>8}'
    )
logger.info('=' * 90)
logger.info(f'Results saved to: {OUT_DIR / \"compare_results.json\"}'  )
"

echo ""
echo "=== Results ==="
cat ${CLUSTER_BASE}/MedicalDataSets/eval_compare_exp32/compare_results.json 2>/dev/null

conda deactivate
