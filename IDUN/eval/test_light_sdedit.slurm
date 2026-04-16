#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="sde_sweep"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/light_sdedit/%j.out
#SBATCH --error=IDUN/output/eval/light_sdedit/%j.err

# Light SDEdit sweep: t₀ from 0.02 to 0.50
# Shows the transition from texture-only smoothing to structure destruction
# 5 volumes × 9 t₀ values × 32 steps each ≈ 1-2 hours on A100

cd ${SLURM_SUBMIT_DIR}

echo "=== Light SDEdit Sweep ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
GEN_DIR="/cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/light_sdedit_${TIMESTAMP}"

time python -m medgen.scripts.test_light_sdedit \
    --bravo-model "${BRAVO_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --generated-dir "${GEN_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-volumes 5 \
    --depth 160 \
    --num-steps 32

echo "Results saved to: ${OUTPUT_DIR}"
cat "${OUTPUT_DIR}/summary.txt" 2>/dev/null
