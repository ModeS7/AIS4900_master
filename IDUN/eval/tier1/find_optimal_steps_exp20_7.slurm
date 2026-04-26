#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_o20_7"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp20_7_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp20_7_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Tier-1 optima search: UNet 67M no-attention (vs exp20_4 which has attention).
# Range: --lo 5 --hi 80 (motivated by extend low-end to 5; smallest no-attn model)

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp20_7 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Expected >=60GB. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp20_7_pixel_bravo_67m_no_attn_20260314-072441"

CKPT="${RUN_DIR}/checkpoint_latest.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${RUN_DIR}/checkpoint_best.pt"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found in $RUN_DIR"
    exit 1
fi
echo "Checkpoint: $CKPT"

OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp20_7"

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
echo "  RESULTS SUMMARY: exp20_7"
echo "=========================================="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
echo ""

conda deactivate
