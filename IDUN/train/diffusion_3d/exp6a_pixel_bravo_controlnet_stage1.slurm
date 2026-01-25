#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp6a_bravo_cn_stage1"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp6a_pixel_bravo_controlnet_stage1_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp6a_pixel_bravo_controlnet_stage1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp6a 3D Pixel Diffusion: Bravo ControlNet Stage 1 (Unconditional UNet) ==="
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
# 3D Pixel-Space Diffusion: Bravo ControlNet Stage 1
# ============================================================================
# Stage 1: Train UNCONDITIONAL UNet (no seg mask input)
#
# This prepares the UNet for ControlNet Stage 2:
#   - UNet in_channels=1 (just noisy_bravo, no seg concatenation)
#   - UNet learns to denoise bravo images without conditioning
#   - The checkpoint can then be used with ControlNet in Stage 2
#
# Architecture: 270M UNet (default_3d)
#   - in_channels: 1 (unconditional)
#   - out_channels: 1
#
# Resolution: 256x256x160 (full resolution - no ControlNet overhead)
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
    controlnet.stage1=true \
    training.name=exp6a_pixel_bravo_controlnet_stage1_

conda deactivate
