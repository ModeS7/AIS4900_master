#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_exp48c_ext"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp48c_handoff_exp32_2_1000_extended_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp48c_handoff_exp32_2_1000_extended_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# R8 — Step-count boundary check for exp48c_handoff_exp32_2_1000.
# Original run found FID 0.4876 @ 100 steps (the search ceiling). This SLURM
# extends the grid to 200 to confirm 100 is an interior optimum.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== R8 Optima boundary sweep: exp32_2_1000 → exp48c @ t=0.25 (80-200) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
LOW_RUN=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp48c_lowt_only_lpips_huber_* 2>/dev/null | head -1)
CKPT_LOW="${LOW_RUN}/checkpoint_latest.pt"
CKPT_HIGH="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp48c/handoff_exp32_2_1000_extended"

[ ! -f "$CKPT_LOW" ] && { echo "ERROR: low-t ckpt not found: $CKPT_LOW"; exit 1; }
[ ! -f "$CKPT_HIGH" ] && { echo "ERROR: high-t ckpt not found: $CKPT_HIGH"; exit 1; }
echo "  low-t (exp48c):  $CKPT_LOW"
echo "  high-t (exp32_2_1000): $CKPT_HIGH"

time python -m medgen.scripts.find_optimal_steps \
    --high-t-checkpoint "${CKPT_HIGH}" \
    --low-t-checkpoint "${CKPT_LOW}" \
    --handoff-t 0.25 \
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
echo "=== Results: exp48c handoff exp32_2_1000 (extended 80-200) ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
