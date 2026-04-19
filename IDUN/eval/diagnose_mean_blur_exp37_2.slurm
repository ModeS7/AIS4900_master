#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="diag_37_2"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/diagnose_mean_blur_exp37_2_%j.out
#SBATCH --error=IDUN/output/eval/diagnose_mean_blur_exp37_2_%j.err

# Mean-blur diagnostic for exp37_2 intermediate checkpoint (LPIPS + FFL)
# Uses same settings as baseline (exp1_1_1000) so results are directly comparable.

cd ${SLURM_SUBMIT_DIR}

echo "=== Mean-Blur Diagnostic (exp37_2) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_2_${TIMESTAMP}"

echo "Checkpoint: ${BRAVO_CKPT}"
ls -la "${BRAVO_CKPT}"

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
