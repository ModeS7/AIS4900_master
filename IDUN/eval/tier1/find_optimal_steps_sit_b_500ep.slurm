#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_sit_b_500"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_sit_b_500ep_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_sit_b_500ep_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Tier-1 optima search: 3D pixel SiT-B 256×256×160 trained for 500 epochs.
# Earlier checkpoint of the SiT-B model (KID 0.0714 vs 2000ep's 0.0284).
# Useful for showing the effect of extended training on transformer.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: SiT-B 256x256x160 500ep ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp7_2_sit_b_256_patch8_20260215-023826"

CKPT="${RUN_DIR}/checkpoint_latest.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${RUN_DIR}/checkpoint_best.pt"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found in $RUN_DIR"
    exit 1
fi
echo "Checkpoint: $CKPT"

OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_sit_b_500ep"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --lo 5 --hi 80 \
    --ref-split test \
    --seed 42

echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: SiT-B 500ep"
echo "=========================================="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
echo ""

conda deactivate
