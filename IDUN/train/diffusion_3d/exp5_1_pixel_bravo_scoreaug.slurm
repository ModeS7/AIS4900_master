#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp5_1_pixel_bravo_scoreaug"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp5_1_pixel_bravo_scoreaug_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp5_1_pixel_bravo_scoreaug_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp5_1 3D Pixel Diffusion: Bravo + ScoreAug ==="
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
# 3D Pixel-Space Diffusion: Bravo + ScoreAug
# ============================================================================
# Based on: 2D exp9_1 (ScoreAug) adapted for 3D
# Reference: https://arxiv.org/abs/2508.07926
#
# ScoreAug Config (3D transforms):
#   - rotation: 90/180/270 around D/H/W axes
#   - translation: +/-40% W, +/-20% H, 0% D (brain centered in Z)
#   - cutout: 3D rectangular regions (10-30%)
#   - omega_conditioning: enabled (required for non-invertible transforms)
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
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.use_omega_conditioning=true \
    training.name=exp5_1_pixel_bravo_scoreaug_

conda deactivate
