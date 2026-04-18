#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="diag_37_1"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/diagnose_mean_blur_exp37_1_%j.out
#SBATCH --error=IDUN/output/eval/diagnose_mean_blur_exp37_1_%j.err

# Mean-blur diagnostic for exp37_1 intermediate checkpoint
# Uses same settings as baseline (exp1_1_1000) so results are directly comparable.

cd ${SLURM_SUBMIT_DIR}

echo "=== Mean-Blur Diagnostic (exp37_1) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/diagnose_mean_blur_exp37_1_${TIMESTAMP}"

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
