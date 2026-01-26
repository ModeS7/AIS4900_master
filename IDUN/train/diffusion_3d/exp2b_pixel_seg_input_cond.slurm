#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp2b_pixel_seg_input_cond"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp2b_pixel_seg_input_cond_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp2b_pixel_seg_input_cond_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp2b 3D Pixel Diffusion: Seg with Input Channel Conditioning ==="
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
# 3D Pixel-Space Diffusion: Seg Mask with Input Channel Conditioning
# ============================================================================
# Architecture: 270M UNet (default_3d)
#   - channels: [16, 32, 64, 256, 512, 512]
#   - attention_levels: [false, false, false, false, true, true]
#   - num_res_blocks: [1, 1, 1, 2, 2, 2]
#
# Mode: seg_conditioned_input (uses seg_conditioned_input_3d config)
#   - in_channels: 8 = 1 noisy_seg + 7 bin_maps (spatial conditioning)
#   - out_channels: 1 = predicted noise/velocity for seg
#   - conditioning: 7 spatial bin maps concatenated with noise input
#   - STRONGER than FiLM: conditioning present at every network layer
#
# Size bins (mm, RANO-BM aligned): [0-3, 3-6, 6-10, 10-15, 15-20, 20-30, 30+]
# Bin maps: constant-valued spatial volumes (normalized count / max_count)
# CFG dropout: 15% (zero bin_maps for unconditional)
#
# Resolution: 128x128x160
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=seg_conditioned_input_3d \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp2b_pixel_seg_input_cond_

conda deactivate
