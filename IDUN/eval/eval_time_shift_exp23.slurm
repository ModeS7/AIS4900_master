#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="tshift_exp23"
#SBATCH --output=IDUN/output/eval/eval_time_shift_exp23_%j.out
#SBATCH --error=IDUN/output/eval/eval_time_shift_exp23_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp23: Pixel-space RFlow bravo + ScoreAug (256x256x160), 1000 epochs
# Golden section search for optimal time-shift ratio
# 27 Euler steps (optimal from step search)
# Search range [1.0, 5.0] — exp1_1 optimal was 2.0

cd ${SLURM_SUBMIT_DIR}

echo "=== Time-Shift Search: exp23 (Pixel RFlow bravo + ScoreAug 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_time_shift_exp23"

COMMON_ARGS="--checkpoint ${CKPT} --data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --ref-split test --seed 42 --search --search-lo 1.0 --search-hi 5.0 --search-tol 0.1"

echo ""
echo "=========================================="
echo "  exp23: Optimal shift by ImageNet FID (27 steps)"
echo "=========================================="
time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_DIR}_fid" \
    --metric fid \
    --num-steps 27 \
    ${COMMON_ARGS}
echo "--- ImageNet FID results ---"
cat "${OUT_DIR}_fid/time_shift_results.json"

echo ""
echo "=========================================="
echo "  exp23: Optimal shift by RadImageNet FID (48 steps)"
echo "=========================================="
time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_DIR}_fid_rin" \
    --metric fid_radimagenet \
    --num-steps 48 \
    ${COMMON_ARGS}
echo "--- RadImageNet FID results ---"
cat "${OUT_DIR}_fid_rin/time_shift_results.json"

echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp23"
echo "=========================================="
echo "--- ImageNet FID ---"
cat "${OUT_DIR}_fid/time_shift_results.json"
echo ""
echo "--- RadImageNet FID ---"
cat "${OUT_DIR}_fid_rin/time_shift_results.json"

conda deactivate
