#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_exp47b"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp47b_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp47b_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp47b — dense step-grid sweep × 3 seeds, RadImageNet FID.
# Grid: 20,25,...,100 (17 points). Seeds: 42, 43, 44.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Dense optima sweep: exp47b ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp47b_huber_lpips_lowt_* 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp47b"

[ ! -f "$CKPT" ] && { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
echo "Checkpoint: $CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --step-grid 20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
    --num-seeds 3 \
    --ref-split test \
    --seed 42

echo ""
echo "=== Results: exp47b ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
