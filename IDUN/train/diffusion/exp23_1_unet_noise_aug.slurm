#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp23_1_noise_aug"
#SBATCH --output=IDUN/output/train/diffusion/exp23_1_noise_aug_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp23_1_noise_aug_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp23_1: UNet with Input Noise Augmentation
# =============================================================================
# Perturbs the noise vector before adding to clean image.
# Increases noise diversity without changing what model learns to output.
#
# Key insight: Like ScoreAug, provides regularization without leaking
# augmentation patterns into generated samples.
#
# The perturbed noise is renormalized to maintain variance.
#
# Configuration:
# - std=0.1: Std of perturbation added to noise vector
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Noise Augmentation (std=0.1)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Noise augmentation configuration:
# - enabled=true: activate noise augmentation
# - std=0.1: perturbation std (noise is renormalized after)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.noise_augmentation.enabled=true \
    training.noise_augmentation.std=0.1 \
    training.name=exp23_1_unet_noise_aug_

conda deactivate
