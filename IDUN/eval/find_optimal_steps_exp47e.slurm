#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_47e"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp47e_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp47e_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp47e (RadImageNet FID) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp47e_perceptual_only_lowt_* 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp47e"

[ ! -f "$CKPT" ] && { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
echo "Checkpoint: $CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid_radimagenet \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --ref-split test \
    --seed 42

echo ""
echo "=== Results: exp47e ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
