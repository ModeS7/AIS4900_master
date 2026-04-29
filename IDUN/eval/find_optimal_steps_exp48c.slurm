#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_exp48c_st"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp48c_standalone_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp48c_standalone_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp48c — standalone optima sweep (full t=1→0 trajectory uses only exp48c weights).
# Grid 20-100 step 5, 3 seeds, RadImageNet FID.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Optima sweep: exp48c standalone ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp48c_lowt_only_lpips_huber_* 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp48c/standalone"

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
echo "=== Results: exp48c standalone ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
