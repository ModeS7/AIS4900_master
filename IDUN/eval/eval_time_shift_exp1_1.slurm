#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="tshift_exp1_1"
#SBATCH --output=IDUN/output/eval/eval_time_shift_exp1_1_%j.out
#SBATCH --error=IDUN/output/eval/eval_time_shift_exp1_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp1_1: Pixel-space RFlow bravo (256x256x160)
# Evaluate time-shifted sampling schedules at inference
# Ratios: 1.0 (no shift), 1.5, 2.0, 3.0, 4.0, 6.84 (MONAI default)

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Time-Shift Eval: exp1_1 (Pixel RFlow bravo 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
COMMON_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --num-steps 25 --metric fid --ref-split test --seed 42"

# ============================================================
# exp1_1 (run 0129): later training run
# ============================================================
echo ""
echo "=========================================="
echo "  exp1_1 (run 0129): Time-Shift Eval"
echo "=========================================="

CKPT_0129="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_best.pt"
if [ ! -f "$CKPT_0129" ]; then
    CKPT_0129="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt"
fi
OUT_0129="${CLUSTER_BASE}/MedicalDataSets/eval_time_shift_exp1_1_0129"

if [ -f "$CKPT_0129" ]; then
    echo "Checkpoint: $CKPT_0129"
    time python -m medgen.scripts.eval_time_shift \
        --checkpoint "${CKPT_0129}" \
        --output-dir "${OUT_0129}" \
        ${COMMON_ARGS}
    echo "--- exp1_1 (0129) time-shift results ---"
    cat "${OUT_0129}/time_shift_results.json"
else
    echo "WARNING: No checkpoint found for exp1_1 run 0129, skipping"
fi

conda deactivate
