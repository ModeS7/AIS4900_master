#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1_pixel_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp1 3D Pixel Diffusion: Bravo (conditioned on Seg Mask) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D Pixel-Space Diffusion: Bravo conditioned on Seg Mask
# ============================================================================
# Architecture: 270M UNet (default_3d)
#   - channels: [16, 32, 64, 256, 512, 512]
#   - attention_levels: [false, false, false, false, true, true]
#   - num_res_blocks: [1, 1, 1, 2, 2, 2]
#
# Mode: bravo
#   - in_channels: 2 = [noisy_bravo, seg_mask]
#   - out_channels: 1 = predicted noise/velocity for bravo
#
# Resolution: 128x128x160 (train), 256x256x160 (val)
# Memory: ~20GB (fits 24GB GPU with headroom)
# ============================================================================

time python -m medgen.scripts.train_diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.train_height=128 \
    volume.train_width=128 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.batch_size=1 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.gradient_checkpointing=true \
    training.name=exp1_pixel_bravo_

conda deactivate
