#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp6b_bravo_cn_stage2"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp6b_pixel_bravo_controlnet_stage2_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp6b_pixel_bravo_controlnet_stage2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp6b 3D Pixel Diffusion: Bravo ControlNet Stage 2 (Train ControlNet) ==="
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
# 3D Pixel-Space Diffusion: Bravo ControlNet Stage 2
# ============================================================================
# Stage 2: Freeze UNet from Stage 1, train only ControlNet
#
# Prerequisites:
#   - Stage 1 checkpoint (unconditional UNet with in_channels=1)
#   - UPDATE THE CHECKPOINT PATH BELOW!
#
# Architecture:
#   - UNet: 270M (frozen from Stage 1)
#   - ControlNet: Mirrors UNet encoder (~135M params)
#
# Mode: bravo
#   - UNet: in_channels=1 (loaded from Stage 1)
#   - ControlNet: takes seg_mask, injects via residuals
#
# Memory: ~40-50GB (only ControlNet gradients, UNet frozen)
# Resolution: 256x256x160
# ============================================================================

# !!! UPDATE THIS PATH to your Stage 1 checkpoint !!!
STAGE1_CHECKPOINT="runs/diffusion_3d/bravo/exp6a_pixel_bravo_controlnet_stage1_YYYYMMDD-HHMMSS/checkpoint_best.pt"

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CHECKPOINT"
    echo "Please update STAGE1_CHECKPOINT in this script to point to your Stage 1 checkpoint"
    exit 1
fi

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
    controlnet.enabled=true \
    controlnet.freeze_unet=true \
    controlnet.gradient_checkpointing=true \
    pretrained_checkpoint=$STAGE1_CHECKPOINT \
    training.name=exp6b_pixel_bravo_controlnet_stage2_

conda deactivate
