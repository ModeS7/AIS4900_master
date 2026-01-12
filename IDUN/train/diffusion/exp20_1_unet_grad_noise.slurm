#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp20_1_grad_noise"
#SBATCH --output=IDUN/output/train/diffusion/exp20_1_grad_noise_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp20_1_grad_noise_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp20_1: UNet with Gradient Noise Injection
# =============================================================================
# Gradient noise adds Gaussian noise to gradients during training as
# implicit regularization. This helps:
# - Escape sharp minima for better generalization
# - Act like dropout but in weight space
#
# Noise decays over training: sigma / (1 + step)^decay
# Reference: "Adding Gradient Noise Improves Learning" (Neelakantan et al., 2015)
#
# Paper recommends:
# - sigma=0.01 initial noise std
# - decay=0.55 noise decay exponent
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Gradient Noise (sigma=0.01, decay=0.55)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Gradient noise configuration:
# - enabled=true: activate gradient noise injection
# - sigma=0.01: initial noise std (decays over training)
# - decay=0.55: noise decay exponent (per paper recommendation)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.gradient_noise.enabled=true \
    training.gradient_noise.sigma=0.01 \
    training.gradient_noise.decay=0.55 \
    training.name=exp20_1_unet_grad_noise_

conda deactivate
