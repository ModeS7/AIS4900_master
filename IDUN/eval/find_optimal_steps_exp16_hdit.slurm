#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp16"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp16_hdit_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp16_hdit_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# exp16: HDiT-S pixel-space bravo, 128x128x160, patch=4, rflow

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp16_hdit (HDiT-S pixel bravo 128x128x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp16_hdit_pixel_bravo_*/checkpoint_latest.pt 2>/dev/null | head -1)
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp16_hdit"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found for exp16_hdit_pixel_bravo_*"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""
echo "Space: pixel"
echo "Architecture: HDiT-S (auto-detected from checkpoint)"
echo "Mode: bravo (auto-detected)"
echo "Strategy: rflow (auto-detected)"
echo "Resolution: 128x128x160"
echo ""

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-size 128 \
    --num-volumes 25 \
    --lo 10 --hi 50 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
