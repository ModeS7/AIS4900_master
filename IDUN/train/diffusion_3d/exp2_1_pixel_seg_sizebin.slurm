#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp2_1_pixel_seg_sizebin"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp2_1_pixel_seg_sizebin_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp2_1_pixel_seg_sizebin_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp2 3D Pixel Diffusion: Seg (conditioned on Tumor Size Bins) ==="
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
# 3D Pixel-Space Diffusion: Seg Mask conditioned on Tumor Size Bins
# ============================================================================
# Architecture: 270M UNet (default_3d)
#   - channels: [16, 32, 64, 256, 512, 512]
#   - attention_levels: [false, false, false, false, true, true]
#   - num_res_blocks: [1, 1, 1, 2, 2, 2]
#
# Mode: seg (uses seg_diffusion_3d config)
#   - in_channels: 1 = noisy_seg (conditioning via embedding, not concat)
#   - out_channels: 1 = predicted noise/velocity for seg
#   - conditioning: 7-dim size bin vector (FiLM modulation on timestep embed)
#
# Size bins (mm, RANO-BM aligned): [0-3, 3-6, 6-10, 10-15, 15-20, 20-30, 30+]
# Example: [0, 0, 2, 0, 0, 0, 1] = 2 tumors 6-10mm + 1 tumor 30+mm
#
# Uses 3D connected components: tumors touching in ANY direction count as ONE
#
# Resolution: 256x256x160
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=seg_diffusion_3d \
    model=default_3d \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp2_1_pixel_seg_sizebin_

conda deactivate
