#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp2c_seg_improved_128"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp2c_pixel_seg_improved_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp2c_pixel_seg_improved_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp2c 3D Pixel Diffusion: Seg (Improved Size Bin Conditioning) 128x128x160 ==="
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
# 3D Pixel-Space Diffusion: Seg Mask with Improved Size Bin Conditioning
# ============================================================================
# Changes from exp2_1:
#   A. Deeper projection MLP: embedding_dim=64, hidden=512, 4 hidden layers
#      7×Embedding(11,64) → [B,448] → 512→512→512→512→256(zero-init) → FiLM
#      Adds ~741K params (~0.3% of 270M UNet)
#   B. Regional weighting: loss weighted by tumor region size
#   C. CFG dropout: 15% → 25% for stronger conditioned/unconditioned contrast
#
# Resolution: 128x128x160 (same as exp2)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    mode=seg_conditioned_3d \
    volume.height=128 \
    volume.width=128 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    mode.size_bins.embedding_dim=64 \
    mode.size_bins.projection_hidden_dim=512 \
    mode.size_bins.projection_num_layers=4 \
    mode.cfg_dropout_prob=0.25 \
    training.regional_weighting.enabled=true \
    training.name=exp2c_pixel_seg_improved_

conda deactivate
