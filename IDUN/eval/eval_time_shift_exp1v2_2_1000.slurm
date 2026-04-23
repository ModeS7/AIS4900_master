#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="tshift_1v2_2"
#SBATCH --output=IDUN/output/eval/eval_time_shift_exp1v2_2_1000_%j.out
#SBATCH --error=IDUN/output/eval/eval_time_shift_exp1v2_2_1000_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp1v2_2_1000: dual-mode (T1pre + T1gd) pixel-space RFlow, 256x256x160, 1000ep.
# Golden section search for optimal time-shift ratio at both step optima from
# find_optimal_steps_exp1v2_2_1000:
#   1) 69 steps (ImageNet FID optimum, FID=65.67) -> optimize shift by FID
#   2) 87 steps (RadImageNet FID optimum, FID=5.16) -> optimize shift by FID_radimagenet

cd ${SLURM_SUBMIT_DIR}

echo "=== Time-Shift Search: exp1v2_2_1000 (dual pixel) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_BASE="${CLUSTER_BASE}/MedicalDataSets/eval_time_shift_exp1v2_2_1000"
COMMON_ARGS="--checkpoint ${CKPT} --mode dual --data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --seed 42 --search --search-lo 1.0 --search-hi 5.0 --search-tol 0.1 --ref-split test"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"

# ============================================================
# 1) Optimal shift at 69 steps (ImageNet FID optimum)
# ============================================================
echo ""
echo "=========================================="
echo "  Optimal shift by ImageNet FID (69 steps)"
echo "=========================================="

time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_BASE}_fid_69" \
    --metric fid \
    --num-steps 69 \
    ${COMMON_ARGS}

echo "--- ImageNet FID @ 69 steps ---"
cat "${OUT_BASE}_fid_69/time_shift_results.json" 2>/dev/null

# ============================================================
# 2) Optimal shift at 87 steps (RadImageNet FID optimum)
# ============================================================
echo ""
echo "=========================================="
echo "  Optimal shift by RadImageNet FID (87 steps)"
echo "=========================================="

time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_BASE}_fid_rin_87" \
    --metric fid_radimagenet \
    --num-steps 87 \
    ${COMMON_ARGS}

echo "--- RadImageNet FID @ 87 steps ---"
cat "${OUT_BASE}_fid_rin_87/time_shift_results.json" 2>/dev/null

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp1v2_2_1000"
echo "=========================================="
echo "--- ImageNet FID @ 69 steps ---"
cat "${OUT_BASE}_fid_69/time_shift_results.json" 2>/dev/null
echo ""
echo "--- RadImageNet FID @ 87 steps ---"
cat "${OUT_BASE}_fid_rin_87/time_shift_results.json" 2>/dev/null
echo ""

conda deactivate
