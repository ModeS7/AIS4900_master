#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp7_sit3d_b_128_2000"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp7_sit_b_128_patch8_2000_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp7_sit_b_128_patch8_2000_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp7: 3D-DiT-B @ 160x128x128, patch=8 (5,120 tokens) ==="
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
# 3D-DiT-B: Bravo mode @ 160x128x128, patch_size=8
# ============================================================================
# Architecture: DiT-B (130M params)
#   - hidden_size: 768, depth: 12, heads: 12
#   - patch_size: 8 -> 5,120 tokens (comparable to 2D 256x256 @ patch=4)
#
# Mode: bravo
#   - in_channels: 2 = [noisy_bravo, seg_mask]
#   - out_channels: 1 = predicted velocity for bravo
#
# Resolution: 128x128x160 (pixel space)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=dit_3d \
    model.variant=B \
    model.patch_size=8 \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=2000 \
    training.learning_rate=1e-4 \
    training.name=exp7_sit_b_128_patch8_2000_

conda deactivate
