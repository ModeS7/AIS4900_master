#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp1_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp1_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp1_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp1_1: Pixel-space RFlow bravo (256x256x160)
# Architecture: UNet 270M (default_3d), channels [16,32,64,256,512,512]
# Mode: bravo (in=2 [noisy_bravo, seg], out=1), raw [0,1] normalization
# Strategy: RFlow (continuous timesteps)
# Runs: ImageNet FID sweep, then RadImageNet FID sweep

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp1_1 (Pixel RFlow bravo 256x256x160) ==="
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
BASE_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --lo 10 --hi 50 --ref-split test --seed 42"

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp1_1, aborting"
    exit 1
fi

echo "Checkpoint: $CKPT"

# ============================================================
# 1) Optimize by ImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp1_1: Optimal steps by ImageNet FID"
echo "=========================================="

OUT_FID="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp1_1_fid"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID}" \
    --metric fid \
    ${BASE_ARGS}

echo "--- ImageNet FID results ---"
cat "${OUT_FID}/search_results.json"

# ============================================================
# 2) Optimize by RadImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp1_1: Optimal steps by RadImageNet FID"
echo "=========================================="

OUT_FID_RIN="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp1_1_fid_rin"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID_RIN}" \
    --metric fid_radimagenet \
    ${BASE_ARGS}

echo "--- RadImageNet FID results ---"
cat "${OUT_FID_RIN}/search_results.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp1_1"
echo "=========================================="
echo "--- ImageNet FID ---"
cat "${OUT_FID}/search_results.json" 2>/dev/null
echo ""
echo "--- RadImageNet FID ---"
cat "${OUT_FID_RIN}/search_results.json" 2>/dev/null
echo ""

conda deactivate
