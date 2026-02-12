#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp11_s2d_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp11_s2d_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp11_s2d_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp11 S2D 3D Pixel Diffusion: Bravo 128x128 (Space-to-Depth 2x2x2) ==="
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
# 3D Pixel-Space Diffusion with Space-to-Depth: Bravo 128x128x160
# ============================================================================
# Space-to-depth 2x2x2:
#   - Input:  [B, 1, 160, 128, 128] (downsampled resolution)
#   - S2D:    [B, 8, 80, 64, 64]    (lossless rearrangement)
#   - UNet 5-level [64,128,512,1024,1024] r=2,2,3,3,3
#     1424M params, ~40 GB training
#     in=16 out=8, L0=64ch@64x64x80 ... L4=1024ch@4x4x5
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d_5lvl \
    space_to_depth.enabled=true \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp11_s2d_bravo_

conda deactivate
