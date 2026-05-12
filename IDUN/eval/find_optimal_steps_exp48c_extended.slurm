#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_exp48c_st_ext"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp48c_standalone_extended_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp48c_standalone_extended_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# R8 — Step-count boundary check for exp48c standalone.
# Original run found FID 0.5197 @ 100 steps (the search ceiling). This SLURM
# extends the grid to 200 to confirm 100 is an interior optimum.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== R8 Optima boundary sweep: exp48c standalone (80-200) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp48c_lowt_only_lpips_huber_* 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp48c/standalone_extended"

[ ! -f "$CKPT" ] && { echo "ERROR: ckpt not found: $CKPT"; exit 1; }
echo "Checkpoint: $CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --step-grid 80,90,100,110,120,130,140,150,160,170,180,190,200 \
    --num-seeds 3 \
    --ref-split test \
    --seed 42

echo ""
echo "=== Results: exp48c standalone (extended 80-200) ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
