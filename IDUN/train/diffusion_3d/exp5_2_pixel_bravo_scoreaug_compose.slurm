#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp5_2_pixel_bravo_scoreaug_compose"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp5_2_pixel_bravo_scoreaug_compose_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp5_2_pixel_bravo_scoreaug_compose_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp5_2 3D Pixel Diffusion: Bravo + ScoreAug Compose ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D Pixel-Space Diffusion: Bravo + ScoreAug Compose Mode
# ============================================================================
# Based on: 2D exp9_5 (ScoreAug Compose) adapted for 3D
# Reference: https://arxiv.org/abs/2508.07926
#
# Compose mode: each transform has 50% independent probability
# Can get: none, one, multiple, or all transforms per batch
#
# ScoreAug Config (3D transforms, each 50% chance):
#   - rotation: 90/180/270 around D/H/W axes
#   - flip: along D/H/W axes
#   - translation: +/-40% W, +/-20% H, 0% D
#   - cutout: 3D rectangular regions (10-30%)
#   - omega_conditioning: enabled
#
# Possible combinations: 2^4 = 16 (rotation, flip, translate, cutout)
#
# Architecture: 270M UNet (default_3d)
# Mode: bravo (in=2, out=1)
# Resolution: 128x128x160
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.compose=true \
    training.score_aug.compose_prob=0.5 \
    training.score_aug.use_omega_conditioning=true \
    training.name=exp5_2_pixel_bravo_scoreaug_compose_

conda deactivate
