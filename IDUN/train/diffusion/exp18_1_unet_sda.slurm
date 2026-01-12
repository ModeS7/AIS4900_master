#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp18_1_sda"
#SBATCH --output=IDUN/output/train/diffusion/exp18_1_sda_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp18_1_sda_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp18_1: UNet with SDA (Shifted Data Augmentation)
# =============================================================================
# SDA is a novel regularization technique that transforms CLEAN data (before
# noise addition) and uses SHIFTED timesteps to prevent leakage.
#
# Key difference from ScoreAug:
# - ScoreAug: transforms noisy data, requires omega conditioning
# - SDA: transforms clean data, uses noise shift (no conditioning needed)
#
# SDA dual-path training:
# 1. Standard path: x -> add_noise(t) -> denoise -> x
# 2. Augmented path: T(x) -> add_noise(t+0.1) -> denoise -> T(x)
#
# Reference: IEEE Access 2025 - "Regularization for Unconditional Image
# Diffusion Models via Shifted Data Augmentation"
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with SDA (noise_shift=0.1, prob=0.5)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# SDA configuration:
# - rotation + flip transforms (geometric augmentation)
# - noise_shift=0.1 (shift timesteps by 0.1 for augmented path)
# - prob=0.5 (50% chance of using augmented path)
# - weight=1.0 (equal weight for standard and augmented loss)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.sda.enabled=true \
    training.sda.rotation=true \
    training.sda.flip=true \
    training.sda.noise_shift=0.1 \
    training.sda.prob=0.5 \
    training.sda.weight=1.0 \
    training.name=exp18_1_unet_sda_

conda deactivate
