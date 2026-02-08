#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp7_1_sit3d_256"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp7_1_sit_s_256_patch16_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp7_1_sit_s_256_patch16_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp7_1: 3D-DiT-S @ 160x256x256, patch=16 (2,560 tokens) ==="
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
# 3D-DiT-S: Bravo mode @ 160x256x256, patch_size=16
# ============================================================================
# Architecture: DiT-S (33M params)
#   - hidden_size: 384, depth: 12, heads: 6
#   - patch_size: 16 -> 2,560 tokens (smaller than 2D baseline)
#
# Mode: bravo
#   - in_channels: 2 = [noisy_bravo, seg_mask]
#   - out_channels: 1 = predicted velocity for bravo
#
# Resolution: 256x256x160 (pixel space, full resolution)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=dit_3d \
    model.variant=S \
    model.patch_size=16 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp7_1_sit_s_256_patch16_

conda deactivate
