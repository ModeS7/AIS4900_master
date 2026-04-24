#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_36_2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp36_2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp36_2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp36_2: augment medium + detail schedule, pixel-space RFlow bravo 256x256x160
# Find optimal Euler steps for: ImageNet FID, RadImageNet FID, PCA shape.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp36_2 (augment medium detail) ==="
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
CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp36_2_augment_medium_detail_20260421-031742/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp36_2"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --ref-split test \
    --seed 42

echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp36_2"
echo "=========================================="
cat "${OUT_DIR}/search_results.json" 2>/dev/null
echo ""
