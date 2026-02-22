#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp9_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp9_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp9_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# exp9_1: LDM 4x UNet bravo_seg_cond (VQ-VAE 4x, RFlow)
# Generates in latent space, evaluates in pixel space via VQ-VAE decode

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp9_1 (LDM 4x UNet, bravo_seg_cond) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
# Diffusion model checkpoint (LDM 4x UNet, bravo_seg_cond)
# NOTE: Update this path to the actual run directory
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_*/checkpoint_latest.pt 2>/dev/null | head -1)

# VQ-VAE 4x compression model (exp8_1) — needed for latent -> pixel decode
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp9_1"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found for exp9_1_ldm_4x_bravo_*"
    exit 1
fi

echo "Diffusion checkpoint: $CHECKPOINT"
echo "Compression checkpoint: $COMPRESSION_CHECKPOINT"
echo ""
echo "Space: latent (VQ-VAE 4x decode)"
echo "Mode: bravo_seg_cond (auto-detected)"
echo "Strategy: rflow (auto-detected)"
echo "Latent stats: read from diffusion checkpoint"
echo ""

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --space latent \
    --compression-checkpoint "${COMPRESSION_CHECKPOINT}" \
    --compression-type vqvae \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
