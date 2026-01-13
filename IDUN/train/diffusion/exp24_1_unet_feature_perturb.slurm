#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp24_1_feature_perturb"
#SBATCH --output=IDUN/output/train/diffusion/exp24_1_feature_perturb_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp24_1_feature_perturb_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp24_1: UNet with Feature Perturbation
# =============================================================================
# Adds Gaussian noise to intermediate features during training.
# Like continuous dropout, but applied to feature activations.
#
# Key insight: Like ScoreAug, provides regularization without leaking
# augmentation patterns into generated samples.
#
# Uses forward hooks to inject noise at specified layers.
#
# Configuration:
# - std=0.1: Std of noise added to features
# - layers=["mid"]: Apply to mid block only (bottleneck)
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Feature Perturbation (std=0.1, mid block)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Feature perturbation configuration:
# - enabled=true: activate feature perturbation
# - std=0.1: noise std added to features
# - layers=["mid"]: perturb mid block (bottleneck)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.feature_perturbation.enabled=true \
    training.feature_perturbation.std=0.1 \
    'training.feature_perturbation.layers=["mid"]' \
    training.name=exp24_1_unet_feature_perturb_

conda deactivate
