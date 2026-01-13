#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp25_1_self_cond"
#SBATCH --output=IDUN/output/train/diffusion/exp25_1_self_cond_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp25_1_self_cond_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp25_1: UNet with Self-Conditioning (Consistency)
# =============================================================================
# Self-conditioning via consistency: model runs twice, consistency loss
# encourages stable predictions.
#
# Key insight: Like ScoreAug, provides regularization without leaking
# augmentation patterns into generated samples.
#
# With probability p:
# 1. First forward pass (no grad) -> prediction P1
# 2. Add consistency loss: MSE(prediction, P1)
#
# Configuration:
# - prob=0.5: probability of self-conditioning per batch
# - consistency_weight=0.1: weight for consistency loss term
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Self-Conditioning (prob=0.5, weight=0.1)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Self-conditioning configuration:
# - enabled=true: activate self-conditioning
# - prob=0.5: probability of using self-conditioning per batch
# - consistency_weight=0.1: weight for consistency loss
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.self_conditioning.enabled=true \
    training.self_conditioning.prob=0.5 \
    training.self_conditioning.consistency_weight=0.1 \
    training.name=exp25_1_unet_self_cond_

conda deactivate
