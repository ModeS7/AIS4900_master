#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_exp19_2_dense"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp19_2_dense_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp19_2_dense_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp19_2 — UNet WDM (Haar wavelet, DDPM x0, linear_beta).
# Dense grid 20-100 step 5, 3 seeds, ImageNet+RadImageNet+PCA.
# Replaces the old GSS sweep (find_optimal_steps_exp19_2.slurm).
# Points to the rerun: bravo/exp19_2_20260501-020717.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Optima dense sweep: exp19_2 (UNet WDM) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp19_2_20260501-020717"
CKPT="${RUN_DIR}/checkpoint_latest.pt"
[ ! -f "${CKPT}" ] && CKPT="${RUN_DIR}/checkpoint_best.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp19_2_dense"

[ ! -f "$CKPT" ] && { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
echo "Checkpoint: $CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --space wavelet \
    --strategy ddpm \
    --prediction-type sample \
    --schedule linear_beta \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --step-grid 20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 \
    --num-seeds 3 \
    --ref-split test \
    --seed 42

echo ""
echo "=== Results: exp19_2 dense ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
