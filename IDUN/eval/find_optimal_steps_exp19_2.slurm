#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp19_2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp19_2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp19_2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp19_2: WDM Wavelet DDPM N(0,1) normalized bravo (256x256x160)
# Generates in wavelet space, evaluates in pixel space via inverse Haar DWT
# Strategy: DDPM with x0 prediction (NOT RFlow)
# Channels: [32,64,256,512,512], attn L3+L4, grad checkpointing

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp19_2 (WDM Wavelet DDPM, bravo 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_2_*/checkpoint_best.pt 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp19_2_*/checkpoint_latest.pt 2>/dev/null | head -1)
fi

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp19_2"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found for exp19_2_*"
    exit 1
fi

echo "Diffusion checkpoint: $CHECKPOINT"
echo ""
echo "Space: wavelet (Haar 3D, per-subband N(0,1) normalized)"
echo "Strategy: DDPM (x0 prediction, linear_beta schedule)"
echo "Pixel resolution: 256x256x160 (decoded from 128x128x80 wavelet)"
echo ""

# WDM uses DDPM strategy with x0 prediction
# --image-size 256: exp19_2 trains at 256x256
# 80GB GPU required for 256x256x160 wavelet generation
time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --space wavelet \
    --strategy ddpm \
    --prediction-type sample \
    --schedule linear_beta \
    --image-size 256 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
