#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp22_1_timestep_jitter"
#SBATCH --output=IDUN/output/train/diffusion/exp22_1_timestep_jitter_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp22_1_timestep_jitter_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp22_1: UNet with Timestep Jitter
# =============================================================================
# Adds small Gaussian noise to sampled timesteps during training.
# This increases noise-level diversity without changing output distribution.
#
# Key insight: Like ScoreAug, provides regularization without leaking
# augmentation patterns into generated samples.
#
# Configuration:
# - std=0.05: Gaussian noise std added to normalized timesteps [0,1]
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Timestep Jitter (std=0.05)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Timestep jitter configuration:
# - enabled=true: activate timestep jitter
# - std=0.05: noise std in normalized [0,1] space
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.timestep_jitter.enabled=true \
    training.timestep_jitter.std=0.05 \
    training.name=exp22_1_unet_timestep_jitter_

conda deactivate
