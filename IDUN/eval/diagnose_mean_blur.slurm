#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="diag_blur"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/diagnose_mean_blur_%j.out
#SBATCH --error=IDUN/output/eval/diagnose_mean_blur_%j.err

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Mean-blur diagnostic for exp1_1_1000
# Diagnostic 1: 7 t-values × 5 volumes × 8 noise seeds ≈ 280 forward passes
# Diagnostic 2: 7 t-values × 5 volumes × (1 single + 32 chain) ≈ 1155 forward passes
# ~30-50 minutes total on H100

cd ${SLURM_SUBMIT_DIR}

echo "=== Mean-Blur Diagnostic ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/diagnose_mean_blur_${TIMESTAMP}"

time python -m medgen.scripts.diagnose_mean_blur \
    --bravo-model "${BRAVO_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-noise 8 \
    --n-volumes 5 \
    --n-chain-steps 32 \
    --depth 160 \
    --seed 42

echo "Results saved to: ${OUTPUT_DIR}"
