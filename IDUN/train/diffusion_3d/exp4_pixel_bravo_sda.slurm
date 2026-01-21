#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp4_pixel_bravo_sda"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp4_pixel_bravo_sda_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp4_pixel_bravo_sda_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp4 3D Pixel Diffusion: Bravo + SDA ==="
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
# 3D Pixel-Space Diffusion: Bravo conditioned on Seg Mask + SDA
# ============================================================================
# Based on: exp1_pixel_bravo (baseline)
# Added: Shifted Data Augmentation (SDA) for regularization
#
# SDA Config:
#   - rotation: 90/180/270 around D/H/W axes
#   - flip: along D/H/W axes
#   - noise_shift: 0.1 (timestep shift for augmented path)
#   - prob: 0.5 (50% of batches use augmented path)
#   - weight: 1.0 (equal weight for augmented loss)
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
    training.sda.enabled=true \
    training.sda.rotation=true \
    training.sda.flip=true \
    training.sda.noise_shift=0.1 \
    training.sda.prob=0.5 \
    training.sda.weight=1.0 \
    training.name=exp4_pixel_bravo_sda_

conda deactivate
