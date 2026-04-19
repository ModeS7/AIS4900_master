#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cfg_exp1n"
#SBATCH --output=IDUN/output/eval/find_optimal_cfg_exp1n_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_cfg_exp1n_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp1n: Pixel-space RFlow bravo (256x256x160) with CFG-Zero*
# Architecture: 270M UNet (default_3d), channels [16,32,64,256,512,512]
# Mode: bravo (in=2, out=1), trained with conditioning_dropout.prob=0.15
# CFG mode: zero_star (Fan et al., 2025)
# Sweeps: CFG-Zero* scales, then standard CFG scales for comparison

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal CFG: exp1n (Pixel RFlow bravo 256x256x160 CFG-Zero*) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
BASE_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --num-steps 25 --ref-split test --seed 42"

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp1n, aborting"
    exit 1
fi

echo "Checkpoint: $CKPT"

# ============================================================
# 1) CFG-Zero* sweep (what exp1n was validated with)
# ============================================================
echo ""
echo "=========================================="
echo "  exp1n: CFG-Zero* scale sweep (FID)"
echo "=========================================="

OUT_ZERO_STAR="${CLUSTER_BASE}/MedicalDataSets/eval_cfg_exp1n_zero_star"

time python -m medgen.scripts.find_optimal_cfg \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_ZERO_STAR}" \
    --cfg-mode zero_star \
    --cfg-values 1.0 1.25 1.5 1.75 2.0 2.5 3.0 4.0 5.0 \
    --metric fid \
    ${BASE_ARGS}

echo "--- CFG-Zero* results ---"
cat "${OUT_ZERO_STAR}/cfg_results.json"

# ============================================================
# 2) Standard CFG sweep for comparison
# ============================================================
echo ""
echo "=========================================="
echo "  exp1n: Standard CFG scale sweep (FID)"
echo "=========================================="

OUT_STANDARD="${CLUSTER_BASE}/MedicalDataSets/eval_cfg_exp1n_standard"

time python -m medgen.scripts.find_optimal_cfg \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_STANDARD}" \
    --cfg-mode standard \
    --cfg-values 1.0 1.25 1.5 1.75 2.0 2.5 3.0 4.0 5.0 \
    --metric fid \
    ${BASE_ARGS}

echo "--- Standard CFG results ---"
cat "${OUT_STANDARD}/cfg_results.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp1n CFG search"
echo "=========================================="
echo "--- CFG-Zero* ---"
cat "${OUT_ZERO_STAR}/cfg_results.json" 2>/dev/null
echo ""
echo "--- Standard CFG ---"
cat "${OUT_STANDARD}/cfg_results.json" 2>/dev/null
echo ""

conda deactivate
