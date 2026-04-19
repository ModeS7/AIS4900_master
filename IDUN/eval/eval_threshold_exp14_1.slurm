#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="thresh_14_1"
#SBATCH --output=IDUN/output/eval/eval_threshold_exp14_1_%j.out
#SBATCH --error=IDUN/output/eval/eval_threshold_exp14_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp14_1: Sweep binarization thresholds on seg model output
# Generate raw volumes once (37 steps, no binarization), then sweep thresholds
# Evaluate each threshold with morphological score

cd ${SLURM_SUBMIT_DIR}

echo "=== Threshold Sweep: exp14_1 (3D Pixel Seg) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_threshold_exp14_1"

python -c "
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

from medgen.diffusion import load_diffusion_model, RFlowStrategy
from medgen.scripts.eval_ode_solvers import (
    SolverConfig, discover_splits, generate_noise_tensors, generate_volumes,
)
from medgen.metrics.morphological import compute_morphological_score

device = torch.device('cuda')
checkpoint = '${CKPT}'
data_root = Path('${DATA_ROOT}')
output_dir = Path('${OUT_DIR}')
output_dir.mkdir(parents=True, exist_ok=True)

NUM_VOLUMES = 25
NUM_STEPS = 37
IMAGE_SIZE = 256
DEPTH = 160
SEED = 42
THRESHOLDS = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]

# ── Load model ──────────────────────────────────────────────────
logger.info('Loading model...')
model = load_diffusion_model(
    checkpoint, device=device,
    in_channels=1, out_channels=1, compile_model=False, spatial_dims=3,
)

strategy = RFlowStrategy()
strategy.setup_scheduler(num_timesteps=1000, image_size=IMAGE_SIZE, depth_size=DEPTH, spatial_dims=3)

# ── Generate noise ──────────────────────────────────────────────
logger.info(f'Generating {NUM_VOLUMES} noise tensors (seed={SEED})...')
noise_list = generate_noise_tensors(NUM_VOLUMES, DEPTH, IMAGE_SIZE, device, SEED, out_channels=1)

# ── Generate raw volumes (NO binarization) ──────────────────────
logger.info(f'Generating {NUM_VOLUMES} raw volumes at {NUM_STEPS} steps...')
solver_cfg = SolverConfig(solver='euler', steps=NUM_STEPS)

t0 = time.time()
raw_volumes, nfe, wall_time = generate_volumes(
    model, strategy, noise_list, None, solver_cfg, device,
    is_seg=False,  # Raw [0,1] output — no thresholding
)
gen_time = time.time() - t0
logger.info(f'Generated {len(raw_volumes)} volumes in {gen_time:.0f}s (NFE={nfe})')

# Free GPU memory — threshold sweep is CPU-only
del model
torch.cuda.empty_cache()

# ── Load reference masks ────────────────────────────────────────
logger.info('Loading reference seg masks...')
splits = discover_splits(data_root, modality='seg')
ref_masks = []
import nibabel as nib
for split_name, split_dir in splits.items():
    seg_files = sorted(split_dir.glob('*/seg.nii.gz'))
    for fp in seg_files:
        vol = nib.load(str(fp)).get_fdata()
        if vol.shape[0] < DEPTH:
            vol = np.pad(vol, ((0, DEPTH - vol.shape[0]), (0, 0), (0, 0)), mode='constant')
        elif vol.shape[0] > DEPTH:
            vol = vol[:DEPTH]
        ref_masks.append((vol > 0.5).astype(np.float32))
logger.info(f'Loaded {len(ref_masks)} reference masks from {len(splits)} splits')

# ── Sweep thresholds ────────────────────────────────────────────
logger.info(f'Sweeping {len(THRESHOLDS)} thresholds...')
results = []

for threshold in THRESHOLDS:
    gen_masks = [(vol > threshold).astype(np.float32) for vol in raw_volumes]

    # Count tumors
    from scipy.ndimage import label as scipy_label
    tumor_counts = []
    for mask in gen_masks:
        _, n = scipy_label(mask > 0.5)
        tumor_counts.append(n)
    avg_tumors = np.mean(tumor_counts)
    total_tumors = sum(tumor_counts)

    # Morphological score
    morph = compute_morphological_score(ref_masks, gen_masks)

    entry = {
        'threshold': threshold,
        'morphological_total': morph.total,
        'morph_volume': morph.volume_dist,
        'morph_feret': morph.feret_dist,
        'morph_spatial_d': morph.spatial_d_dist,
        'morph_spatial_h': morph.spatial_h_dist,
        'morph_spatial_w': morph.spatial_w_dist,
        'morph_count': morph.count_dist,
        'n_real_tumors': morph.n_real_tumors,
        'n_gen_tumors': morph.n_gen_tumors,
        'avg_tumors_per_vol': avg_tumors,
        'total_gen_tumors': total_tumors,
    }
    results.append(entry)

    logger.info(
        f'  threshold={threshold:.2f}: morph={morph.total:.4f} '
        f'(vol={morph.volume_dist:.4f}, feret={morph.feret_dist:.4f}, '
        f'count={morph.count_dist:.4f}) '
        f'tumors={total_tumors} ({avg_tumors:.1f}/vol)'
    )

# ── Save results ────────────────────────────────────────────────
output = {
    'checkpoint': checkpoint,
    'num_volumes': NUM_VOLUMES,
    'num_steps': NUM_STEPS,
    'seed': SEED,
    'generation_time_s': gen_time,
    'n_ref_masks': len(ref_masks),
    'results': results,
}
with open(output_dir / 'threshold_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# ── Summary table ───────────────────────────────────────────────
logger.info('')
logger.info('=' * 90)
logger.info('THRESHOLD SWEEP RESULTS')
logger.info('=' * 90)
logger.info(f\"{'Thresh':>7}  {'Morph':>8}  {'Volume':>8}  {'Feret':>8}  {'Count':>8}  {'Tumors':>7}  {'Per Vol':>8}\")
logger.info('-' * 90)
best = min(results, key=lambda x: x['morphological_total'])
for r in results:
    marker = ' <--' if r['threshold'] == best['threshold'] else ''
    logger.info(
        f\"{r['threshold']:>7.2f}  {r['morphological_total']:>8.4f}  \"
        f\"{r['morph_volume']:>8.4f}  {r['morph_feret']:>8.4f}  \"
        f\"{r['morph_count']:>8.4f}  {r['total_gen_tumors']:>7d}  \"
        f\"{r['avg_tumors_per_vol']:>8.1f}{marker}\"
    )
logger.info('=' * 90)
logger.info(f'Best threshold: {best[\"threshold\"]:.2f} (morph={best[\"morphological_total\"]:.4f})')
"

echo "=== Results ==="
cat "${OUT_DIR}/threshold_results.json" 2>/dev/null

conda deactivate
