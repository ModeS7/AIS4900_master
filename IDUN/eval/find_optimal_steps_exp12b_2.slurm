#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="optim_exp12b_2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp12b_2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp12b_2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# exp12b_2: WDM Wavelet DDPM N(0,1) normalized bravo (128x128x160)
# Generates in wavelet space, evaluates in pixel space via inverse Haar DWT
# Strategy: DDPM with x0 prediction (NOT RFlow)

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp12b_2 (WDM Wavelet DDPM, bravo 128x128x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
# Diffusion model checkpoint (WDM wavelet DDPM)
# NOTE: Update this path to the actual run directory
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_*/checkpoint_latest.pt 2>/dev/null | head -1)

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp12b_2"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found for exp12b_2_*"
    exit 1
fi

echo "Diffusion checkpoint: $CHECKPOINT"
echo ""
echo "Space: wavelet (Haar 3D, per-subband N(0,1) normalized)"
echo "Strategy: DDPM (x0 prediction, linear_beta schedule)"
echo "Pixel resolution: 128x128x160 (decoded from 64x64x80 wavelet)"
echo ""

# WDM uses DDPM strategy, not RFlow
# --wavelet-normalize: recomputes per-subband stats from training data
# --strategy ddpm: overrides auto-detection (checkpoint stores 'ddpm')
# --prediction-type sample: x0 prediction (stored as 'sample' in code)
# --image-size 128: WDM trains at 128x128 (not 256)
time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --space wavelet \
    --wavelet-normalize \
    --strategy ddpm \
    --prediction-type sample \
    --schedule linear_beta \
    --image-size 128 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
